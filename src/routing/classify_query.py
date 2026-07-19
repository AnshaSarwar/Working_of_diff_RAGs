"""
Cheap query classifier for tiered routing.

Why classify first?
Not every question needs GraphRAG or an agentic retry loop. A fast label lets
simple facts stay on the cheap vanilla path while multi-hop / ambiguous asks
get the heavier machinery — the efficiency layer of this project.
"""

from __future__ import annotations

import re
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_CLASSIFIER_MODEL


class QueryRoute(str, Enum):
    SIMPLE_FACTUAL = "SIMPLE_FACTUAL"
    MULTI_HOP_RELATIONAL = "MULTI_HOP_RELATIONAL"
    COMPLEX_AMBIGUOUS = "COMPLEX_AMBIGUOUS"


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions for a company knowledge-base RAG system.\n"
            "Reply with EXACTLY one label and nothing else:\n"
            "SIMPLE_FACTUAL — single fact from one doc (policies, definitions, codes like PTO250)\n"
            "MULTI_HOP_RELATIONAL — needs relationship hops (who manages X, owner of Y's manager)\n"
            "COMPLEX_AMBIGUOUS — spans multiple docs / conditions / eligibility edge cases\n",
        ),
        ("human", "{query}"),
    ]
)


def _heuristic_classify(query: str) -> QueryRoute | None:
    """
    Deterministic backup when the LLM is unavailable or returns garbage.

    Keeps demos working offline and makes routing explainable.
    """
    q = query.lower()

    relational_cues = (
        "who manages",
        "manager of",
        "reports to",
        "managed by",
        "owner of",
        "leading project",
        "person leading",
        "direct report",
    )
    if any(c in q for c in relational_cues):
        return QueryRoute.MULTI_HOP_RELATIONAL

    complex_cues = (
        "if i'm remote",
        "if i am remote",
        "and also",
        "eligible",
        "can i expense",
        "depending on",
        "both",
        "under the manager",
    )
    if any(c in q for c in complex_cues):
        return QueryRoute.COMPLEX_AMBIGUOUS

    simple_cues = (
        "what is",
        "what's",
        "define",
        "pto250",
        "policy",
    )
    if any(c in q for c in simple_cues):
        return QueryRoute.SIMPLE_FACTUAL

    return None


def _parse_label(text: str) -> QueryRoute | None:
    upper = text.upper().strip()
    for route in QueryRoute:
        if route.value in upper:
            return route
    # Single-token / partial
    if re.search(r"\bSIMPLE\b", upper):
        return QueryRoute.SIMPLE_FACTUAL
    if re.search(r"\bMULTI[_\s-]?HOP|RELATIONAL\b", upper):
        return QueryRoute.MULTI_HOP_RELATIONAL
    if re.search(r"\bCOMPLEX|AMBIGUOUS\b", upper):
        return QueryRoute.COMPLEX_AMBIGUOUS
    return None


def classify_query(query: str, *, use_llm: bool = True) -> QueryRoute:
    """
    Label a query for the top-level router.

    Prefer deterministic heuristics for clear patterns (stable demos), then a
    constrained LLM call for everything else. Never fail open — default SIMPLE.
    """
    heuristic = _heuristic_classify(query)
    # Strong cues → trust heuristics (avoids flaky single-token LLM variance).
    if heuristic is not None and heuristic != QueryRoute.SIMPLE_FACTUAL:
        return heuristic
    if heuristic is QueryRoute.SIMPLE_FACTUAL and not use_llm:
        return heuristic

    if use_llm and OPENAI_API_KEY:
        try:
            llm = ChatOpenAI(
                model=OPENAI_CLASSIFIER_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=16,
            )
            messages = CLASSIFY_PROMPT.format_messages(query=query)
            raw = llm.invoke(messages).content
            text = raw if isinstance(raw, str) else str(raw)
            parsed = _parse_label(text)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    return heuristic or QueryRoute.SIMPLE_FACTUAL
