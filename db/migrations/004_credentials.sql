-- Agentic Cloud - Phase 3, Step 3.1
-- Credential vault: per-user secrets for tool providers, stored ENCRYPTED.
--
--   ciphertext  : Fernet token (AES-128-CBC + HMAC-SHA256) of
--                 {"v":1, "u":<user_id>, "p":<provider>, "s":<secret>}
--                 user_id + provider are encrypted together with the secret, so a
--                 ciphertext copied into another user's row is detected on decrypt.
--   key_id      : first 8 hex chars of sha256(master key) used to encrypt this row
--                 (shows which rows still need re-encryption after a key rotation)
--   fingerprint : "sha256:" + first 12 hex chars of sha256(secret) - lets you check
--                 WHICH key is stored without ever showing it
--   metadata    : NON-secret settings for the provider (e.g. the user's own chat id,
--                 used later by the gateway's owner_only rule)
--
-- The master key is NEVER stored in the database (env var VAULT_MASTER_KEY).
-- Applied automatically by the control plane on startup.

CREATE TABLE IF NOT EXISTS credentials (
    id            BIGSERIAL PRIMARY KEY,
    user_id       TEXT        NOT NULL,
    provider      TEXT        NOT NULL,
    ciphertext    TEXT        NOT NULL,
    key_id        TEXT        NOT NULL,
    fingerprint   TEXT        NOT NULL,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    use_count     INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    UNIQUE (user_id, provider)
);

DROP TRIGGER IF EXISTS trg_credentials_updated_at ON credentials;
CREATE TRIGGER trg_credentials_updated_at
    BEFORE UPDATE ON credentials
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- Every vault operation is recorded (never the secret itself).
CREATE TABLE IF NOT EXISTS vault_audit (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    provider    TEXT        NOT NULL,
    action      TEXT        NOT NULL
                CHECK (action IN ('stored', 'replaced', 'decrypted', 'deleted',
                                  'rotated', 'tamper_detected')),
    actor       TEXT        NOT NULL,          -- admin | admin-verify | gateway | rotate
    session_id  TEXT,                          -- set when the gateway decrypts for a session
    detail      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vault_audit_user
    ON vault_audit (user_id, created_at);