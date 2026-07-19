"""
Agentic RAG nodes.

Why these four nodes?
1. retrieve      — reuse Phase 1's vector retriever (don't reinvent embeddings)
2. grade_context — cheap LLM yes/no: "is this enough to answer?"
3. rewrite_query — only runs on failure; reformulates to cover missing pieces
4. generate      — final answer once context is good *or* retries are exhausted

Query rewriting lives *only* in the retry loop (Phase 4 note): we do not
rewrite every query up front — that would add latency to simple factual asks.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import AGENTIC_TOP_K, MAX_AGENTIC_RETRIES
from src.stage1_vanilla.embed_store import build_vector_store
from src.stage1_vanilla.retrieve_generate import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    format_context,
    get_llm,
)
from src.stage3_agentic.state import AgenticState, AttemptRecord
from src.stage4_advanced.hybrid_search import get_bm25_index, hybrid_search
from src.stage4_advanced.pipeline import advanced_retrieve

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether retrieved context is sufficient to answer a "
            "question about Acme Corp policies/people/projects.\n\n"
            "Be strict on multi-part questions:\n"
            "- If the question combines topics (e.g. expense/conference + remote "
            "work), YES only when the context covers ALL parts — typically "
            "chunks from BOTH the expense policy and the remote-work policy.\n"
            "- Prefer NO when a related policy is merely named/linked but its "
            "rules are not present in the context.\n"
            "- YES only if a careful reader could answer fully from the context "
            "alone without guessing.\n\n"
            "Source paths in this retrieval (use these when judging coverage):\n"
            "{sources}\n\n"
            "Respond in exactly this format:\n"
            "GRADE: YES|NO\n"
            "REASON: <one short sentence>",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\nGRADE and REASON:",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a company knowledge base.\n"
            "The previous retrieval was insufficient. Produce ONE improved "
            "query that is more specific and likely to retrieve the missing "
            "documents (e.g. name both policies/topics involved).\n"
            "Return only the rewritten query text — no quotes or preamble.",
        ),
        (
            "human",
            "Original question: {original}\n"
            "Query just used: {current}\n"
            "Grader reason: {reason}\n"
            "Sources already retrieved: {sources}\n"
            "Previous context (may be incomplete):\n{context}\n\n"
            "Rewritten query:",
        ),
    ]
)

_vector_store = None


def _get_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(rebuild=False)
    return _vector_store


def _get_retriever_docs(query: str) -> list[Document]:
    """
    Attempt-1 retrieve: Phase 4 hybrid + rerank, then keep AGENTIC_TOP_K.

    Still narrow on purpose so multi-doc questions can trigger rewrite;
    rewriting itself remains Phase-3-only (not applied on every query).
    """
    return advanced_retrieve(
        query,
        store=_get_store(),
        top_k=AGENTIC_TOP_K,
        use_hybrid=True,
        use_rerank=True,
    )


def _diversity_retrieve(
    query: str,
    existing: list[dict[str, Any]],
    *,
    pool_k: int = 8,
    take: int = 2,
) -> list[Document]:
    """
    On retries, hybrid-search a wider pool and prefer chunks from *new* sources.

    Why?
    Rewriting the query alone often still returns the same top hit on a tiny
    corpus. Preferring unseen sources is what lets attempt 2 add expense_policy
    after attempt 1 already locked onto remote_work_policy.
    """
    seen_sources = {
        (p.get("metadata") or {}).get("source_path")
        for p in existing
        if (p.get("metadata") or {}).get("source_path")
    }
    store = _get_store()
    pool = hybrid_search(
        store,
        query,
        bm25_index=get_bm25_index(store),
        candidate_k=pool_k,
    )
    novel = [
        d
        for d in pool
        if (d.metadata or {}).get("source_path") not in seen_sources
    ]
    chosen = novel[:take] if novel else pool[:take]
    return chosen


def _docs_to_payloads(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {"page_content": d.page_content, "metadata": dict(d.metadata or {})}
        for d in docs
    ]


def _payloads_to_docs(payloads: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(page_content=p["page_content"], metadata=p.get("metadata") or {})
        for p in payloads
    ]


def _merge_documents(
    existing: list[dict[str, Any]],
    new_docs: list[Document],
) -> list[dict[str, Any]]:
    """
    Union chunks across retries (by chunk_id / content).

    Why accumulate?
    Multi-doc questions often need attempt 1's remote-policy chunk *and*
    attempt 2's expense-policy chunk. Replacing context each loop would
    throw away progress.
    """
    merged = list(existing or [])
    seen: set[str] = set()
    for p in merged:
        meta = p.get("metadata") or {}
        key = meta.get("chunk_id") or p.get("page_content", "")[:120]
        seen.add(key)

    for d in new_docs:
        meta = dict(d.metadata or {})
        key = meta.get("chunk_id") or d.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append({"page_content": d.page_content, "metadata": meta})
    return merged


def _parse_grade(text: str) -> tuple[str, str]:
    """Extract YES/NO + reason; default to NO if the model is ambiguous."""
    m = re.search(r"GRADE:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if m:
        grade = m.group(1).lower()
    elif re.search(r"\bYES\b", text, flags=re.IGNORECASE) and not re.search(
        r"\bNO\b", text, flags=re.IGNORECASE
    ):
        grade = "yes"
    else:
        grade = "no"

    reason = ""
    rm = re.search(r"REASON:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if rm:
        reason = rm.group(1).strip().splitlines()[0].strip()
    else:
        reason = text.strip()[:200]
    return grade, reason


def _source_paths(payloads: list[dict[str, Any]]) -> list[str]:
    return [p.get("metadata", {}).get("source_path", "?") for p in payloads]


def _distinct_sources(payloads: list[dict[str, Any]]) -> set[str]:
    return {
        src
        for src in _source_paths(payloads)
        if src and src != "?"
    }


def _looks_multipart(question: str) -> bool:
    """
    Cheap structural cue that an answer likely needs multiple source docs.

    Corpus-agnostic: conjunctions / multi-clause shape, not specific file paths.
    """
    q = question.lower()
    if " and " in q or ";" in q:
        return True
    # "both X and Y" / "as well as" often need coverage from more than one doc.
    if "both " in q or "as well as" in q:
        return True
    return False


def _coverage_insufficient(
    question: str,
    payloads: list[dict[str, Any]],
    *,
    min_sources: int = 2,
) -> str | None:
    """
    Soft check: multi-part questions should span enough distinct source files.

    Returns a reason string when coverage looks thin, else None.
    Keeps the agentic loop educational without hard-coding OKF paths.
    """
    if not _looks_multipart(question):
        return None
    sources = _distinct_sources(payloads)
    if len(sources) >= min_sources:
        return None
    return (
        f"Multi-part question needs at least {min_sources} distinct sources; "
        f"have {len(sources)} ({', '.join(sorted(sources)) or 'none'})."
    )


def retrieve(state: AgenticState) -> dict:
    """Call Phase 1's Chroma retriever; merge new hits into accumulated context."""
    query = state.get("query") or state["original_query"]
    existing = state.get("documents") or []
    retry_count = int(state.get("retry_count") or 0)

    if retry_count == 0:
        new_docs = _get_retriever_docs(query)
    else:
        # Diversity retrieve so retries add missing policies, not duplicates.
        new_docs = _diversity_retrieve(query, existing, pool_k=8, take=2)

    merged = _merge_documents(existing, new_docs)
    context = format_context(_payloads_to_docs(merged))
    sources = _source_paths(merged)
    return {
        "documents": merged,
        "context": context,
        "attempt_log": [
            AttemptRecord(
                attempt=retry_count + 1,
                query_used=query,
                sources=sources,
                grade="",
                grade_reason="",
            )
        ],
    }


def grade_context(state: AgenticState) -> dict:
    """
    LLM judges sufficiency.

    Why grade before generating?
    Generating on incomplete context produces confident-but-wrong answers.
    Grading lets us rewrite and retrieve again — the core agentic idea.
    """
    docs = state.get("documents") or []
    sources = ", ".join(sorted(set(_source_paths(docs)))) or "(none)"
    llm = get_llm()
    messages = GRADE_PROMPT.format_messages(
        question=state["original_query"],
        context=state.get("context") or "(empty)",
        sources=sources,
    )
    raw = llm.invoke(messages).content
    text = raw if isinstance(raw, str) else str(raw)
    grade, reason = _parse_grade(text)

    # Soft, corpus-agnostic coverage: multi-part questions need multiple
    # source files before we trust a YES (lets diversity-retrieve fill gaps).
    coverage_reason = _coverage_insufficient(state["original_query"], docs)
    if coverage_reason:
        grade = "no"
        reason = f"{coverage_reason} ({reason})"

    return {
        "grade": grade,
        "grade_reason": reason,
        "attempt_log": [
            AttemptRecord(
                attempt=int(state.get("retry_count", 0)) + 1,
                query_used=state.get("query") or state["original_query"],
                sources=_source_paths(docs),
                grade=grade,
                grade_reason=reason,
            )
        ],
    }


def rewrite_query(state: AgenticState) -> dict:
    """
    Reformulate the query after a failed grade.

    Increments retry_count here (entering a rewrite counts as using one retry).
    Hard-capped by the graph's conditional edge at MAX_AGENTIC_RETRIES.
    """
    docs = state.get("documents") or []
    llm = get_llm()
    messages = REWRITE_PROMPT.format_messages(
        original=state["original_query"],
        current=state.get("query") or state["original_query"],
        reason=state.get("grade_reason") or "context incomplete",
        sources=", ".join(_source_paths(docs)) or "(none)",
        context=state.get("context") or "(empty)",
    )
    raw = llm.invoke(messages).content
    rewritten = (raw if isinstance(raw, str) else str(raw)).strip().strip('"').strip("'")
    if not rewritten:
        rewritten = state["original_query"]

    new_count = min(int(state.get("retry_count", 0)) + 1, MAX_AGENTIC_RETRIES)
    return {
        "query": rewritten,
        "retry_count": new_count,
    }


def generate(state: AgenticState) -> dict:
    """Final answer — runs after a YES grade *or* when retries are exhausted."""
    llm = get_llm()
    docs = _payloads_to_docs(state.get("documents") or [])
    context = state.get("context") or format_context(docs)
    messages = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    ).format_messages(
        context=context or "(no context retrieved)",
        question=state["original_query"],
    )
    raw = llm.invoke(messages).content
    answer = (raw if isinstance(raw, str) else str(raw)).strip()
    return {"final_answer": answer}
