"""
Agentic Cloud - Phase 3, Step 3.1
Credential vault: per-user secrets for tool providers, encrypted at rest.

Encryption: Fernet (cryptography library)
  token = base64( version | timestamp | IV(16) | AES-128-CBC(K_enc, IV, plaintext) | HMAC-SHA256(K_sign, ...) )
  - a fresh random IV every time -> the same secret encrypts to a different token each time
  - the HMAC tag detects ANY change to the token (1 flipped bit -> InvalidToken)

Binding: the plaintext is  {"v":1, "u":user_id, "p":provider, "s":secret}
  After decrypting we check u/p against the row we read. A ciphertext copied from
  amrith's row into rahul's row decrypts fine but fails this check.

Master keys: env var VAULT_MASTER_KEY = "newest_key,older_key,..."
  MultiFernet encrypts with the FIRST key and decrypts with any of them, so a key can
  be rotated without downtime: put the new key first, call rotate(), then drop the old one.
  The master key is never stored in the database and never given to agents.

Nothing in this module ever returns, logs or audits the plaintext secret, except
get(), which the gateway (Step 3.3) calls at the moment of a tool call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

log = logging.getLogger("control_plane.vault")

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


# ---- errors (mapped to HTTP status codes in main.py) ---------------------------

class VaultError(Exception):
    status_code = 400


class VaultDisabled(VaultError):
    status_code = 503


class CredentialNotFound(VaultError):
    status_code = 404


class CredentialTampered(VaultError):
    status_code = 409


# ---- helpers ---------------------------------------------------------------------

def key_id(key: str) -> str:
    """Short, non-secret identifier of a master key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def fingerprint(secret: str) -> str:
    """Short, non-reversible identifier of a stored secret (for API keys, not passwords)."""
    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def generate_master_key() -> str:
    return Fernet.generate_key().decode("ascii")


class Vault:
    def __init__(self, pool: ConnectionPool, master_keys: list[str]) -> None:
        self.pool = pool
        self.keys = [k.strip() for k in master_keys if k and k.strip()]
        self.enabled = bool(self.keys)
        self.primary_key_id: Optional[str] = None
        if self.enabled:
            try:
                self._f = MultiFernet([Fernet(k.encode("ascii")) for k in self.keys])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "VAULT_MASTER_KEY is not a valid Fernet key (must be 32 url-safe base64 bytes). "
                    "Generate one with: python -c \"from cryptography.fernet import Fernet; "
                    "print(Fernet.generate_key().decode())\""
                ) from exc
            self.primary_key_id = key_id(self.keys[0])

    @classmethod
    def from_env(cls, pool: ConnectionPool, env_name: str = "VAULT_MASTER_KEY") -> "Vault":
        raw = os.environ.get(env_name, "")
        return cls(pool, raw.split(",") if raw else [])

    def status(self) -> dict:
        return {"enabled": self.enabled, "master_keys": len(self.keys),
                "primary_key_id": self.primary_key_id}

    # ================================================================ crypto

    def _require(self) -> None:
        if not self.enabled:
            raise VaultDisabled("vault disabled: set VAULT_MASTER_KEY (see .env) and restart")

    def _encrypt(self, user_id: str, provider: str, secret: str) -> str:
        payload = json.dumps({"v": 1, "u": user_id, "p": provider, "s": secret},
                             separators=(",", ":")).encode("utf-8")
        return self._f.encrypt(payload).decode("ascii")

    def _decrypt(self, user_id: str, provider: str, token: str) -> str:
        try:
            data = json.loads(self._f.decrypt(token.encode("ascii")))
        except (InvalidToken, ValueError) as exc:
            raise CredentialTampered(
                f"credential {user_id}/{provider} failed the integrity check "
                f"(ciphertext was modified, or encrypted with a different master key)"
            ) from exc
        if data.get("u") != user_id or data.get("p") != provider:
            raise CredentialTampered(
                f"credential {user_id}/{provider} contains a ciphertext that belongs to "
                f"{data.get('u')}/{data.get('p')} (row swap detected)"
            )
        return data["s"]

    @staticmethod
    def _check_names(user_id: str, provider: str) -> None:
        if not user_id or len(user_id) > 128:
            raise VaultError("invalid user_id")
        if not _NAME.match(provider):
            raise VaultError("provider must be lowercase letters/digits/._- (max 64 chars)")

    @staticmethod
    def _audit(conn, user_id: str, provider: str, action: str, actor: str,
               session_id: Optional[str] = None, detail: Optional[dict] = None) -> None:
        conn.execute(
            "INSERT INTO vault_audit (user_id, provider, action, actor, session_id, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, provider, action, actor, session_id, Jsonb(detail or {})),
        )

    # ================================================================ write

    def put(self, user_id: str, provider: str, secret: str,
            metadata: Optional[dict] = None, actor: str = "admin") -> dict:
        """Store or replace a secret. Returns only non-secret info."""
        self._require()
        self._check_names(user_id, provider)
        if not secret:
            raise VaultError("secret is empty")

        ct = self._encrypt(user_id, provider, secret)
        fp = fingerprint(secret)
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                """
                INSERT INTO credentials (user_id, provider, ciphertext, key_id, fingerprint, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, provider) DO UPDATE
                    SET ciphertext = EXCLUDED.ciphertext, key_id = EXCLUDED.key_id,
                        fingerprint = EXCLUDED.fingerprint, metadata = EXCLUDED.metadata
                RETURNING (xmax = 0) AS inserted
                """,
                (user_id, provider, ct, self.primary_key_id, fp, Jsonb(metadata or {})),
            ).fetchone()
            action = "stored" if row["inserted"] else "replaced"
            self._audit(conn, user_id, provider, action, actor,
                        detail={"fingerprint": fp, "key_id": self.primary_key_id})
        log.info("vault: %s %s/%s (%s)", action, user_id, provider, fp)
        return {"user_id": user_id, "provider": provider, "action": action,
                "fingerprint": fp, "key_id": self.primary_key_id,
                "ciphertext_length": len(ct)}

    def delete(self, user_id: str, provider: str, actor: str = "admin") -> bool:
        with self.pool.connection() as conn, conn.transaction():
            cur = conn.execute("DELETE FROM credentials WHERE user_id = %s AND provider = %s",
                               (user_id, provider))
            if cur.rowcount:
                self._audit(conn, user_id, provider, "deleted", actor)
        return bool(cur.rowcount)

    # ================================================================ read

    def get(self, user_id: str, provider: str, session_id: Optional[str] = None,
            actor: str = "gateway") -> str:
        """Decrypt and return the secret. ONLY the gateway should call this, per tool call."""
        self._require()
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "SELECT id, ciphertext FROM credentials WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            ).fetchone()
            if row is None:
                raise CredentialNotFound(f"no {provider} credential stored for {user_id}")
            try:
                secret = self._decrypt(user_id, provider, row["ciphertext"])
            except CredentialTampered as exc:
                self._audit(conn, user_id, provider, "tamper_detected", actor, session_id,
                            {"error": str(exc)})
                log.error("vault: %s", exc)
                raise
            conn.execute(
                "UPDATE credentials SET use_count = use_count + 1, last_used_at = now() WHERE id = %s",
                (row["id"],),
            )
            self._audit(conn, user_id, provider, "decrypted", actor, session_id)
        return secret

    def verify(self, user_id: str, provider: str) -> dict:
        """Prove the stored credential decrypts correctly - without revealing it."""
        secret = self.get(user_id, provider, actor="admin-verify")
        stored = self.metadata(user_id, provider, include_fingerprint=True)["fingerprint"]
        fp = fingerprint(secret)
        return {"user_id": user_id, "provider": provider, "decrypts": True,
                "fingerprint": fp, "matches_stored_fingerprint": fp == stored}

    def metadata(self, user_id: str, provider: str, include_fingerprint: bool = False) -> dict:
        """Non-secret settings (e.g. the owner's own chat id for the owner_only rule)."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT metadata, fingerprint FROM credentials WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            ).fetchone()
        if row is None:
            raise CredentialNotFound(f"no {provider} credential stored for {user_id}")
        out = dict(row["metadata"] or {})
        if include_fingerprint:
            out = {"metadata": out, "fingerprint": row["fingerprint"]}
        return out

    def list(self, user_id: str) -> list[dict]:
        """Everything about a user's credentials EXCEPT the secrets and ciphertexts."""
        with self.pool.connection() as conn:
            return conn.execute(
                """
                SELECT provider, fingerprint, key_id, metadata, use_count,
                       created_at, updated_at, last_used_at
                FROM credentials WHERE user_id = %s ORDER BY provider
                """,
                (user_id,),
            ).fetchall()

    def audit_log(self, user_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self.pool.connection() as conn:
            if user_id:
                return conn.execute(
                    "SELECT * FROM vault_audit WHERE user_id = %s ORDER BY id DESC LIMIT %s",
                    (user_id, limit),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM vault_audit ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()

    # ================================================================ key rotation

    def rotate(self) -> dict:
        """Re-encrypt every credential with the CURRENT primary master key.

        Use: VAULT_MASTER_KEY="NEW,OLD" -> restart -> rotate() -> VAULT_MASTER_KEY="NEW" -> restart.
        """
        self._require()
        rotated, already, failed = 0, 0, []
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, user_id, provider, ciphertext, key_id FROM credentials"
            ).fetchall()
        for r in rows:
            if r["key_id"] == self.primary_key_id:
                already += 1
                continue
            try:
                self._decrypt(r["user_id"], r["provider"], r["ciphertext"])  # also checks binding
                new_ct = self._f.rotate(r["ciphertext"].encode("ascii")).decode("ascii")
            except (CredentialTampered, InvalidToken) as exc:
                failed.append({"user_id": r["user_id"], "provider": r["provider"], "error": str(exc)})
                continue
            with self.pool.connection() as conn, conn.transaction():
                conn.execute("UPDATE credentials SET ciphertext = %s, key_id = %s WHERE id = %s",
                             (new_ct, self.primary_key_id, r["id"]))
                self._audit(conn, r["user_id"], r["provider"], "rotated", "rotate",
                            detail={"from_key_id": r["key_id"], "to_key_id": self.primary_key_id})
            rotated += 1
        return {"primary_key_id": self.primary_key_id, "rotated": rotated,
                "already_current": already, "failed": failed}