"""
Cross-encoder reranker (top-20 hybrid → top-5).

Why rerank?
Bi-encoder retrieval (and RRF) is fast but coarse — it never looks at the
query and document *together*. A cross-encoder scores (query, doc) pairs
jointly, which usually lifts the truly relevant chunk into the top few.
We only rerank ~20 candidates so the cost stays small.

Implementation note:
Preferred path is an ONNX cross-encoder via fastembed
(Xenova/ms-marco-MiniLM-L-6-v2 or BAAI/bge-reranker-base).
If HuggingFace rate-limits the download, we fall back to a local lexical
scorer so demos still run — same *slot* in the pipeline, weaker model.
Set RERANKER_MODEL_NAME in .env once the ONNX weights are cached.
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document

from src.config import RERANK_TOP_K, RERANKER_MODEL_NAME
from src.stage4_advanced.hybrid_search import tokenize

_cross_encoder = None
_using_fallback = False
_init_attempted = False


def _lexical_score(query: str, text: str) -> float:
    """
    Offline stand-in for a cross-encoder when the ONNX model isn't cached yet.

    Favours exact substring hits (critical for codes like PTO250) plus token
    overlap — good enough to demonstrate the rerank *stage*, not SOTA quality.
    """
    q = query.lower().strip()
    t = text.lower()
    score = 0.0
    if q and q in t:
        score += 10.0
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    if q_tokens:
        score += 5.0 * (len(q_tokens & t_tokens) / len(q_tokens))
    for tok in q_tokens:
        if re.search(r"[a-z]+\d+|\d+[a-z]+", tok) and tok in t_tokens:
            score += 3.0
    return score


def get_cross_encoder():
    """Lazy-load ONNX cross-encoder; return (encoder|None, using_fallback)."""
    global _cross_encoder, _using_fallback, _init_attempted

    if _init_attempted:
        return _cross_encoder, _using_fallback

    _init_attempted = True

    # Force lexical path (useful when offline / HF 429s).
    if os.getenv("RERANK_LEXICAL_ONLY", "").lower() in {"1", "true", "yes"}:
        _using_fallback = True
        _cross_encoder = None
        return None, True

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _cross_encoder = TextCrossEncoder(model_name=RERANKER_MODEL_NAME)
        _using_fallback = False
        return _cross_encoder, False
    except Exception as exc:  # noqa: BLE001 — HF 429 / missing cache
        print(
            f"[reranker] Could not load {RERANKER_MODEL_NAME} ({exc}). "
            "Using lexical fallback. Re-run later to cache the ONNX model, "
            "or set RERANKER_MODEL_NAME=BAAI/bge-reranker-base."
        )
        _cross_encoder = None
        _using_fallback = True
        return None, True


def rerank_documents(
    query: str,
    documents: list[Document],
    *,
    top_k: int = RERANK_TOP_K,
) -> list[Document]:
    """Score each (query, doc) pair and keep the top_k."""
    if not documents:
        return []

    encoder, fallback = get_cross_encoder()
    if encoder is not None and not fallback:
        texts = [d.page_content for d in documents]
        raw_scores = list(encoder.rerank(query, texts))
        scored = list(zip(documents, raw_scores, strict=True))
        method = "cross_encoder"
    else:
        scored = [(d, _lexical_score(query, d.page_content)) for d in documents]
        method = "lexical_fallback"

    scored.sort(key=lambda x: float(x[1]), reverse=True)

    out: list[Document] = []
    for doc, score in scored[:top_k]:
        meta = dict(doc.metadata or {})
        meta["rerank_score"] = round(float(score), 6)
        meta["rerank_method"] = method
        meta["retrieval"] = (meta.get("retrieval") or "unknown") + "+rerank"
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
