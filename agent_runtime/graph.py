"""
Agentic Cloud - Phase 1, Step 1.2
Research-assistant agent as a LangGraph graph.

Flow:
  START -> search_papers -> rank_papers -> await_selection --(valid)--> summarize_papers -> write_report -> END
                                               ^      |
                                               +------+ (invalid selection: ask again)

await_selection calls interrupt(): the graph PAUSES there and its state is saved as a
checkpoint in Postgres (keyed by thread_id). The container can then be stopped (FROZEN).
A brand-new container can later resume the same thread_id with Command(resume=...).

Important LangGraph rule: when a run resumes, the interrupted node runs again FROM ITS
START, and interrupt() returns the resume value. So everything in await_selection
before interrupt() must be side-effect free (it is: it only builds a payload).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_runtime.llm import LLM
from agent_runtime.tools import search_papers

SEARCH_K = 8          # papers fetched by the search tool
SHORTLIST_SIZE = 5    # papers shown to the user


class ResearchState(TypedDict, total=False):
    topic: str
    candidates: list[dict]          # raw search results
    shortlist: list[dict]           # top papers with "rank" (1-based) and "score"
    selection_error: Optional[str]  # set when the user's selection was invalid
    selected: list[int]             # ranks chosen by the user
    summaries: list[dict]
    report: str


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


def build_graph(llm: LLM, checkpointer):
    """Build and compile the research-assistant graph with the given checkpointer."""

    # -------------------------------------------------------------- nodes

    def search_node(state: ResearchState) -> dict:
        return {"candidates": search_papers(state["topic"], k=SEARCH_K)}

    def rank_node(state: ResearchState) -> dict:
        topic = state["topic"]
        scored = [
            {**p, "score": llm.score_relevance(topic, p["title"], p["abstract"])}
            for p in state["candidates"]
        ]
        scored.sort(key=lambda p: p["score"], reverse=True)
        shortlist = [{**p, "rank": i + 1} for i, p in enumerate(scored[:SHORTLIST_SIZE])]
        return {"shortlist": shortlist}

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

    def summarize_node(state: ResearchState) -> dict:
        by_rank = {p["rank"]: p for p in state["shortlist"]}
        summaries = []
        for r in state["selected"]:
            p = by_rank[r]
            summaries.append(
                {
                    "rank": r,
                    "title": p["title"],
                    "venue": p["venue"],
                    "year": p["year"],
                    "summary": llm.summarize(p["title"], p["abstract"]),
                }
            )
        return {"summaries": summaries}

    def report_node(state: ResearchState) -> dict:
        lines = [f"# Research brief: {state['topic']}", ""]
        for s in state["summaries"]:
            lines.append(f"## {s['rank']}. {s['title']} ({s['venue']} {s['year']})")
            lines.append(s["summary"])
            lines.append("")
        return {"report": "\n".join(lines).strip()}

    # -------------------------------------------------------------- wiring
    # Node names must differ from state keys (LangGraph rule), hence *_papers / write_report.

    g = StateGraph(ResearchState)
    g.add_node("search_papers", search_node)
    g.add_node("rank_papers", rank_node)
    g.add_node("await_selection", await_selection_node)
    g.add_node("summarize_papers", summarize_node)
    g.add_node("write_report", report_node)

    g.add_edge(START, "search_papers")
    g.add_edge("search_papers", "rank_papers")
    g.add_edge("rank_papers", "await_selection")
    g.add_conditional_edges(
        "await_selection",
        route_after_selection,
        {"summarize_papers": "summarize_papers", "await_selection": "await_selection"},
    )
    g.add_edge("summarize_papers", "write_report")
    g.add_edge("write_report", END)

    return g.compile(checkpointer=checkpointer)