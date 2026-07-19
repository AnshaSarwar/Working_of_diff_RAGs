"""
Semantic cache wrapping the top-level router.

Why cache by embedding similarity (not exact string match)?
Users rephrase ("What's PTO250?" vs "What is PTO250?"). Exact-key caches miss;
embedding near-neighbors catch paraphrases and skip the full pipeline.

Wraps the entire router_graph — checked *before* classify/retrieve/generate.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from src.config import SEMANTIC_CACHE_PATH, SEMANTIC_CACHE_THRESHOLD
from src.routing.router_graph import RouterResult, run_router
from src.stage1_vanilla.embed_store import get_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """JSON-backed store of {query, embedding, answer, route, techniques}."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> None:
        self.path = path or SEMANTIC_CACHE_PATH
        self.threshold = threshold
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._entries = []
        else:
            self._entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _embed(self, text: str) -> list[float]:
        emb = get_embeddings()
        # FastEmbedEmbeddings.embed_query returns List[float]
        vec = emb.embed_query(text)
        return list(vec)

    def lookup(self, query: str) -> tuple[RouterResult | None, float]:
        """Return (cached result, best_similarity) — result is None on miss."""
        if not self._entries:
            return None, 0.0
        q_vec = self._embed(query)
        best_sim = -1.0
        best: dict[str, Any] | None = None
        for entry in self._entries:
            sim = _cosine(q_vec, entry.get("embedding") or [])
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self.threshold:
            return (
                RouterResult(
                    query=query,
                    route=best.get("route", ""),
                    answer=best.get("answer", ""),
                    techniques=list(best.get("techniques") or []) + ["cache_hit"],
                    sources=list(best.get("sources") or []),
                    contexts=list(best.get("contexts") or []),
                    metadata_filter_summary=best.get("metadata_filter_summary", ""),
                    cache_hit=True,
                ),
                best_sim,
            )
        return None, best_sim

    def store(self, result: RouterResult) -> None:
        if not result.answer or result.cache_hit:
            return
        entry = {
            "query": result.query,
            "embedding": self._embed(result.query),
            "answer": result.answer,
            "route": result.route,
            "techniques": result.techniques,
            "sources": result.sources,
            "contexts": result.contexts,
            "metadata_filter_summary": result.metadata_filter_summary,
            "stored_at": time.time(),
        }
        self._entries.append(entry)
        self._save()

    def clear(self) -> None:
        self._entries = []
        if self.path.is_file():
            self.path.unlink()


_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


def run_routed_rag(
    query: str,
    *,
    use_cache: bool = True,
    store_in_cache: bool = True,
) -> RouterResult:
    """
    Public entrypoint: semantic cache → router_graph.

    This is what Phase 6 eval / demos should call.
    """
    if use_cache:
        cached, _sim = get_semantic_cache().lookup(query)
        if cached is not None:
            return cached

    result = run_router(query)
    if store_in_cache and result.answer and not result.error:
        get_semantic_cache().store(result)
    return result
