"""
LangGraph wiring for Phase 3 Agentic RAG.

Flow:
  retrieve → grade_context → (rewrite_query → retrieve)* → generate

Conditional edge after grade_context:
  - grade == "no" AND retry_count < MAX_AGENTIC_RETRIES (2) → rewrite_query
  - otherwise → generate (good context *or* hard stop)

Why a hard stop at 2?
Unbounded retry loops waste tokens and can thrash on impossible questions.
Two retries is enough to surface a second policy doc for the demo query.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.config import MAX_AGENTIC_RETRIES
from src.stage3_agentic.nodes import generate, grade_context, retrieve, rewrite_query
from src.stage3_agentic.state import AgenticState


def route_after_grade(state: AgenticState) -> Literal["rewrite_query", "generate"]:
    """Decide whether to retry or produce a final answer."""
    grade = (state.get("grade") or "").lower().strip()
    retry_count = int(state.get("retry_count") or 0)
    if grade == "no" and retry_count < MAX_AGENTIC_RETRIES:
        return "rewrite_query"
    return "generate"


def build_agentic_graph():
    """Compile the agentic state machine (reusable by Phase 5 routing)."""
    graph = StateGraph(AgenticState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# Lazy singleton for demos / later router integration.
_compiled = None


def get_agentic_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_agentic_graph()
    return _compiled


def run_agentic_rag(question: str) -> AgenticState:
    """End-to-end Phase 3 pipeline for a single question."""
    app = get_agentic_graph()
    initial: AgenticState = {
        "original_query": question,
        "query": question,
        "documents": [],
        "context": "",
        "grade": "",
        "grade_reason": "",
        "retry_count": 0,
        "final_answer": "",
        "attempt_log": [],
    }
    return app.invoke(initial)
