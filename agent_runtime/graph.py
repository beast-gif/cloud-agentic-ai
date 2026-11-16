"""
Agentic Cloud - Phase 2, Step 2.4
Research-assistant agent as a LangGraph graph - now using all three memory layers.

Flow:
  START -> recall_memory -> search_papers -> rank_papers -> await_selection --(valid)-->
           summarize_papers -> write_report -> remember -> END
                                 ^      |
                                 +------+ (invalid selection: ask again)

  L1 (checkpoint)  : the whole state below, saved by LangGraph after every node
  L2 (workspace)   : write_report saves report.md + summaries.json to /workspace
  L3 (long-term)   : recall_memory reads past papers of THIS user (Memory API);
                     rank_papers skips papers already read; remember saves new ones

await_selection calls interrupt(): the graph PAUSES there and its state is saved as a
checkpoint in Postgres (keyed by thread_id). The container can then be stopped (FROZEN).
A brand-new container can later resume the same thread_id with Command(resume=...).

Important LangGraph rule: when a run resumes, the interrupted node runs again FROM ITS
START, and interrupt() returns the resume value. So everything in await_selection
before interrupt() must be side-effect free (it is: it only builds a payload).
Nodes that already finished (recall_memory, search, rank) are NOT run again.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from llm import LLM
from memory_client import MemoryClient
from tools import search_papers

log = logging.getLogger("agent.graph")

SEARCH_K = 8          # papers fetched by the search tool
SHORTLIST_SIZE = 5    # papers shown to the user
RECALL_K = 10         # past memories fetched before searching


class ResearchState(TypedDict, total=False):
    topic: str
    recalled: list[dict]            # L3: related past memories of this user
    candidates: list[dict]          # raw search results
    skipped_already_read: list[dict]  # candidates removed because the user read them before
    shortlist: list[dict]           # top papers with "rank" (1-based) and "score"
    selection_error: Optional[str]  # set when the user's selection was invalid
    selected: list[int]             # ranks chosen by the user
    summaries: list[dict]
    report: str
    workspace_files: list[str]      # L2: files written to /workspace
    memories_written: list[dict]    # L3: result of each memory write (inserted / deduplicated)


def _parse_selection(answer: Any, max_rank: int) -> tuple[list[int], Optional[str]]:
    """Accepts {"selected": [1, 4]} or just [1, 4]. Returns (valid ranks, error)."""
    raw = answer.get("selected") if isinstance(answer, dict) else answer
    if not isinstance(raw, (list, tuple)):
        return [], f"Expected a list of paper numbers, got {raw!r}"

    ranks: set[int] = set()
    for item in raw:
        try:
            r = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= r <= max_rank:
            ranks.add(r)

    if not ranks:
        return [], f"No valid paper numbers in {list(raw)!r}; choose between 1 and {max_rank}"
    return sorted(ranks), None


def build_graph(llm: LLM, checkpointer, memory: MemoryClient, workspace_dir: str = "/workspace"):
    """Build and compile the research-assistant graph."""

    # -------------------------------------------------------------- L3: recall

    def recall_node(state: ResearchState) -> dict:
        hits = memory.search(state["topic"], k=RECALL_K, kinds=["paper_read"])
        recalled = [
            {
                "paper_id": (h.get("metadata") or {}).get("paper_id"),
                "title": (h.get("metadata") or {}).get("title") or h.get("content", "")[:120],
                "score": h.get("score"),
                "read_at": str(h.get("created_at", ""))[:19],
            }
            for h in hits
        ]
        log.info("recalled %d related memories for %r", len(recalled), state["topic"])
        return {"recalled": recalled}

    # -------------------------------------------------------------- search + rank

    def search_node(state: ResearchState) -> dict:
        return {"candidates": search_papers(state["topic"], k=SEARCH_K)}

    def rank_node(state: ResearchState) -> dict:
        topic = state["topic"]

        # skip papers this user has already read (exact id match from L3 metadata)
        read_ids = {r["paper_id"] for r in state.get("recalled", []) if r.get("paper_id")}
        fresh = [p for p in state["candidates"] if p["id"] not in read_ids]
        skipped = [{"id": p["id"], "title": p["title"]}
                   for p in state["candidates"] if p["id"] in read_ids]

        scored = [
            {**p, "score": llm.score_relevance(topic, p["title"], p["abstract"])}
            for p in fresh
        ]
        scored.sort(key=lambda p: p["score"], reverse=True)
        shortlist = [{**p, "rank": i + 1} for i, p in enumerate(scored[:SHORTLIST_SIZE])]
        return {"shortlist": shortlist, "skipped_already_read": skipped}

    # -------------------------------------------------------------- human in the loop

    def await_selection_node(state: ResearchState) -> dict:
        shortlist = state["shortlist"]
        payload = {
            "waiting_for": "paper_selection",
            "message": "Which papers should I summarize? Reply with their numbers.",
            "shortlist": [
                {
                    "rank": p["rank"],
                    "title": p["title"],
                    "year": p["year"],
                    "venue": p["venue"],
                    "score": p["score"],
                }
                for p in shortlist
            ],
            # L3 context shown to the user
            "already_read_skipped": state.get("skipped_already_read", []),
            "related_past_reading": state.get("recalled", [])[:3],
            "error": state.get("selection_error"),
        }

        # ---- the agent PAUSES here until a resume value arrives ----
        answer = interrupt(payload)

        selected, error = _parse_selection(answer, max_rank=len(shortlist))
        if error:
            return {"selected": [], "selection_error": error}
        return {"selected": selected, "selection_error": None}

    def route_after_selection(state: ResearchState) -> str:
        return "summarize_papers" if state.get("selected") else "await_selection"

    # -------------------------------------------------------------- work

    def summarize_node(state: ResearchState) -> dict:
        by_rank = {p["rank"]: p for p in state["shortlist"]}
        summaries = []
        for r in state["selected"]:
            p = by_rank[r]
            summaries.append(
                {
                    "rank": r,
                    "id": p["id"],
                    "title": p["title"],
                    "venue": p["venue"],
                    "year": p["year"],
                    "summary": llm.summarize(p["title"], p["abstract"]),
                }
            )
        return {"summaries": summaries}

    # -------------------------------------------------------------- L2: workspace

    def report_node(state: ResearchState) -> dict:
        lines = [f"# Research brief: {state['topic']}", ""]
        for s in state["summaries"]:
            lines.append(f"## {s['rank']}. {s['title']} ({s['venue']} {s['year']})")
            lines.append(s["summary"])
            lines.append("")
        skipped = state.get("skipped_already_read", [])
        if skipped:
            lines.append("_Skipped (already read before):_ " + "; ".join(x["title"] for x in skipped))
        report = "\n".join(lines).strip()

        # Overwriting is idempotent, so a retried node does no harm.
        written: list[str] = []
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            with open(os.path.join(workspace_dir, "report.md"), "w", encoding="utf-8") as f:
                f.write(report + "\n")
            written.append("report.md")
            with open(os.path.join(workspace_dir, "summaries.json"), "w", encoding="utf-8") as f:
                json.dump({"topic": state["topic"], "summaries": state["summaries"]}, f, indent=2)
            written.append("summaries.json")
        except OSError as exc:  # workspace problems must not lose the report itself
            log.warning("could not write workspace files: %r", exc)

        return {"report": report, "workspace_files": written}

    # -------------------------------------------------------------- L3: remember

    def remember_node(state: ResearchState) -> dict:
        """Save what was read. Safe to repeat: the Memory API deduplicates (cos >= 0.95)."""
        results = []
        for s in state["summaries"]:
            res = memory.write(
                kind="paper_read",
                content=f"Read '{s['title']}' ({s['venue']} {s['year']}): {s['summary']}",
                metadata={"paper_id": s["id"], "title": s["title"], "venue": s["venue"],
                          "year": s["year"], "topic": state["topic"]},
            )
            results.append({"title": s["title"],
                            "action": (res or {}).get("action", "failed"),
                            "memory_id": (res or {}).get("id")})

        titles = "; ".join(s["title"] for s in state["summaries"])
        res = memory.write(
            kind="topic_researched",
            content=f"Researched '{state['topic']}' and read {len(state['summaries'])} paper(s): {titles}",
            metadata={"topic": state["topic"], "papers": [s["id"] for s in state["summaries"]]},
        )
        results.append({"title": f"topic: {state['topic']}",
                        "action": (res or {}).get("action", "failed"),
                        "memory_id": (res or {}).get("id")})
        return {"memories_written": results}

    # -------------------------------------------------------------- wiring
    # Node names must differ from state keys (LangGraph rule).

    g = StateGraph(ResearchState)
    g.add_node("recall_memory", recall_node)
    g.add_node("search_papers", search_node)
    g.add_node("rank_papers", rank_node)
    g.add_node("await_selection", await_selection_node)
    g.add_node("summarize_papers", summarize_node)
    g.add_node("write_report", report_node)
    g.add_node("remember", remember_node)

    g.add_edge(START, "recall_memory")
    g.add_edge("recall_memory", "search_papers")
    g.add_edge("search_papers", "rank_papers")
    g.add_edge("rank_papers", "await_selection")
    g.add_conditional_edges(
        "await_selection",
        route_after_selection,
        {"summarize_papers": "summarize_papers", "await_selection": "await_selection"},
    )
    g.add_edge("summarize_papers", "write_report")
    g.add_edge("write_report", "remember")
    g.add_edge("remember", END)

    return g.compile(checkpointer=checkpointer)