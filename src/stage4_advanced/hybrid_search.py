"""
Hybrid search: BM25 keyword + vector similarity, fused with RRF.

Why hybrid?
Dense embeddings capture *meaning* ("paid time off policy") but often fail on
*exact tokens* like internal codes ("PTO250"). BM25 is the opposite: great at
literal matches, weak at paraphrase. Reciprocal Rank Fusion (RRF) merges both
ranked lists without needing calibrated scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import HYBRID_CANDIDATE_K, RRF_K


def tokenize(text: str) -> list[str]:
    """
    Keep alphanumeric tokens so codes like PTO250 stay intact.

    Splitting on non-letters alone would shred "PTO250" into noise; keeping
    digits attached is the whole point of BM25 for this demo.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Cormack et al. RRF: score(d) = Σ 1 / (k + rank_i(d)).

    `ranked_lists` are ordered id lists (best first). Returns (id, score)
    sorted by descending score.
    """
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class HybridSearchIndex:
    """
    In-memory BM25 over the same chunks stored in Chroma.

    Built once per process and reused — BM25 is cheap to keep around for a
    small OKF bundle.
    """

    documents: list[Document] = field(default_factory=list)
    _id_to_doc: dict[str, Document] = field(default_factory=dict)
    _bm25: BM25Okapi | None = None
    _corpus_tokens: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_documents(cls, documents: list[Document]) -> HybridSearchIndex:
        index = cls()
        index.documents = list(documents)
        for doc in documents:
            doc_id = _doc_id(doc)
            index._id_to_doc[doc_id] = doc
        index._corpus_tokens = [tokenize(d.page_content) for d in documents]
        index._bm25 = BM25Okapi(index._corpus_tokens)
        return index

    @classmethod
    def from_chroma(cls, store: Chroma) -> HybridSearchIndex:
        """Pull every chunk out of Chroma to build the BM25 side."""
        raw = store.get(include=["documents", "metadatas"])
        docs: list[Document] = []
        ids = raw.get("ids") or []
        texts = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        for i, text in enumerate(texts):
            meta = dict(metas[i] or {})
            # Prefer stored chunk_id; fall back to Chroma's id.
            if not meta.get("chunk_id") and i < len(ids):
                meta["chunk_id"] = ids[i]
            docs.append(Document(page_content=text or "", metadata=meta))
        if not docs:
            raise RuntimeError("Chroma store is empty — rebuild with --rebuild first.")
        return cls.from_documents(docs)

    def bm25_search(self, query: str, *, top_k: int = HYBRID_CANDIDATE_K) -> list[Document]:
        assert self._bm25 is not None
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Argsort descending
        ranked_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Document] = []
        for i in ranked_idxs[:top_k]:
            if scores[i] <= 0:
                break
            out.append(self.documents[i])
        return out


def _doc_id(doc: Document) -> str:
    return (
        doc.metadata.get("chunk_id")
        or f"{doc.metadata.get('source_path', '')}::{hash(doc.page_content) & 0xFFFF}"
    )


def vector_search(
    store: Chroma,
    query: str,
    *,
    top_k: int = HYBRID_CANDIDATE_K,
) -> list[Document]:
    return store.similarity_search(query, k=top_k)


def hybrid_search(
    store: Chroma,
    query: str,
    *,
    bm25_index: HybridSearchIndex | None = None,
    candidate_k: int = HYBRID_CANDIDATE_K,
    rrf_k: int = RRF_K,
) -> list[Document]:
    """
    Run BM25 + vector, fuse with RRF, return up to `candidate_k` docs.

    Order is RRF rank (best first). Downstream reranker expects ~20 candidates.
    """
    index = bm25_index or HybridSearchIndex.from_chroma(store)
    bm25_hits = index.bm25_search(query, top_k=candidate_k)
    vector_hits = vector_search(store, query, top_k=candidate_k)

    bm25_ids = [_doc_id(d) for d in bm25_hits]
    vector_ids = [_doc_id(d) for d in vector_hits]
    fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)

    # Resolve ids back to Document objects (prefer richer metadata from either list).
    by_id: dict[str, Document] = {}
    for d in bm25_hits + vector_hits:
        by_id[_doc_id(d)] = d

    results: list[Document] = []
    for doc_id, score in fused[:candidate_k]:
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        # Stamp RRF score for demos / tracing (Phase 6).
        meta = dict(doc.metadata or {})
        meta["rrf_score"] = round(score, 6)
        meta["retrieval"] = "hybrid_rrf"
        results.append(Document(page_content=doc.page_content, metadata=meta))
    return results


# Process-wide cache so demos don't rebuild BM25 every call.
_cached_index: HybridSearchIndex | None = None


def get_bm25_index(store: Chroma, *, rebuild: bool = False) -> HybridSearchIndex:
    global _cached_index
    if _cached_index is None or rebuild:
        _cached_index = HybridSearchIndex.from_chroma(store)
    return _cached_index
