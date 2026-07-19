"""
Phase 4 retrieval pipeline: hybrid (BM25 + vector RRF) → cross-encoder rerank.

Composed by the Phase 5 router (and Phase 3 agentic retrieve) — Stage 1 stays
vector-only so demos can A/B baseline vs advanced. Query rewriting stays
*only* in Phase 3's retry loop — not applied here.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import HYBRID_CANDIDATE_K, RERANK_TOP_K
from src.routing.metadata_filter import (
    MetadataFilters,
    apply_metadata_filter,
    infer_metadata_filters,
)
from src.stage1_vanilla.embed_store import build_vector_store
from src.stage1_vanilla.retrieve_generate import VanillaRAGResult, generate_answer
from src.stage4_advanced.hybrid_search import get_bm25_index, hybrid_search, vector_search
from src.stage4_advanced.reranker import rerank_documents


def advanced_retrieve(
    query: str,
    *,
    store: Chroma | None = None,
    candidate_k: int = HYBRID_CANDIDATE_K,
    top_k: int = RERANK_TOP_K,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    metadata_filters: MetadataFilters | None = None,
    apply_inferred_filters: bool = True,
) -> list[Document]:
    """
    Hybrid/vector retrieve → optional metadata filter → optional rerank.

    Phase 5: metadata filters (OKF type/tags) run on candidates before rerank
    so every route that uses this helper gets the same filtering behaviour.
    """
    store = store or build_vector_store(rebuild=False)

    if use_hybrid:
        candidates = hybrid_search(
            store,
            query,
            bm25_index=get_bm25_index(store),
            candidate_k=candidate_k,
        )
    else:
        candidates = vector_search(store, query, top_k=candidate_k)
        for d in candidates:
            d.metadata = dict(d.metadata or {})
            d.metadata["retrieval"] = "vector_only"

    filters = metadata_filters
    if filters is None and apply_inferred_filters:
        filters = infer_metadata_filters(query)
    candidates = apply_metadata_filter(candidates, filters)

    if use_rerank:
        return rerank_documents(query, candidates, top_k=top_k)
    return candidates[:top_k]


def run_advanced_rag(
    question: str,
    *,
    top_k: int = RERANK_TOP_K,
    rebuild_store: bool = False,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    metadata_filters: MetadataFilters | None = None,
    apply_inferred_filters: bool = True,
) -> VanillaRAGResult:
    """
    Phase 4 end-to-end: advanced_retrieve → same stuff/generate as Stage 1.

    Used by the router SIMPLE path so Stage 1 demos remain pure vector RAG.
    """
    store = build_vector_store(rebuild=rebuild_store)
    filters = metadata_filters
    if filters is None and apply_inferred_filters:
        filters = infer_metadata_filters(question)

    retrieved = advanced_retrieve(
        question,
        store=store,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        metadata_filters=filters,
        apply_inferred_filters=False,
    )

    techniques: list[str] = []
    if use_hybrid:
        techniques.append("hybrid_rrf")
    else:
        techniques.append("vector_only")
    if use_rerank:
        techniques.append("rerank")
    if filters is not None and not filters.is_empty():
        techniques.append("metadata_filter")

    answer = generate_answer(question, retrieved)
    return VanillaRAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved,
        techniques=techniques,
    )
