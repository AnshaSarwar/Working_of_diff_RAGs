"""
LangGraph state for Phase 3 Agentic RAG.

Why a typed state machine?
Vanilla RAG is a straight line: retrieve → generate. Agentic RAG needs
*memory across steps* (what we retrieved, how many times we retried, the
rewritten query). LangGraph's state is that shared scratchpad.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AttemptRecord(TypedDict, total=False):
    """One retrieve→grade cycle, kept for demos / later tracing (Phase 6)."""

    attempt: int
    query_used: str
    sources: list[str]
    grade: str
    grade_reason: str


class AgenticState(TypedDict, total=False):
    """
    Shared state flowing through the agentic graph.

    Fields:
      original_query  — user question (never mutated)
      query           — query used for the *next* retrieve (may be rewritten)
      documents       — latest retrieved LangChain Document dicts / payloads
      context         — stuffed text from those documents
      grade           — "yes" | "no" from grade_context
      grade_reason    — short rationale (educational / debugging)
      retry_count     — how many rewrite→retrieve loops have completed
      final_answer    — set by the generate node
      attempt_log     — append-only history of each retrieve/grade cycle
    """

    original_query: str
    query: str
    documents: list[dict[str, Any]]
    context: str
    grade: str
    grade_reason: str
    retry_count: int
    final_answer: str
    # Append across loops so the demo can show every retry fired.
    attempt_log: Annotated[list[AttemptRecord], operator.add]
