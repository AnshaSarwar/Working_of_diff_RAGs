"""
Phase 6 — Trace logger.

Wraps every routed query and records:
  - route taken
  - total latency (+ optional per-node timings)
  - which advanced techniques fired (hybrid, rerank, cache, rewrite, …)

Why instrument?
Without traces you can't tell *why* an eval score moved — was it routing,
caching, or retrieval? This log is the glue between latency and RAGAS quality.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import PROJECT_ROOT
from src.cache.semantic_cache import run_routed_rag
from src.routing.router_graph import RouterResult

KNOWN_TECHNIQUES = (
    "hybrid_rrf",
    "rerank",
    "cache_hit",
    "query_rewrite",
    "graph_rag",
    "agentic",
    "metadata_filter",
    "vector_only",
    "classify",
)


@dataclass
class TraceRecord:
    query: str
    route_taken: str
    latency_ms: float
    techniques_fired: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    answer: str = ""
    cache_hit: bool = False
    node_latencies_ms: dict[str, float] = field(default_factory=dict)
    error: str = ""
    timestamp: str = ""

    def techniques_csv(self) -> str:
        return "|".join(self.techniques_fired)


def normalize_techniques(techniques: list[str]) -> list[str]:
    """Deduplicate while preserving order; drop empty tags."""
    seen: set[str] = set()
    out: list[str] = []
    for t in techniques:
        key = (t or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def traced_run(
    query: str,
    *,
    use_cache: bool = False,
    runner: Callable[..., RouterResult] = run_routed_rag,
) -> TraceRecord:
    """
    Time one full pipeline call and snapshot techniques/route.

    Eval defaults use_cache=False so latency/quality aren't skewed by hits.
    """
    t0 = time.perf_counter()
    result = runner(query, use_cache=use_cache, store_in_cache=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # If the router already timed nodes, keep them; else record total only.
    node_latencies = dict(result.node_latencies_ms or {})
    if "total" not in node_latencies:
        node_latencies["total"] = round(latency_ms, 2)

    return TraceRecord(
        query=query,
        route_taken=result.route,
        latency_ms=round(latency_ms, 2),
        techniques_fired=normalize_techniques(list(result.techniques or [])),
        sources=list(result.sources or []),
        contexts=list(result.contexts or []),
        answer=result.answer or "",
        cache_hit=bool(result.cache_hit),
        node_latencies_ms=node_latencies,
        error=result.error or "",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


class TraceLogger:
    """Collect TraceRecords and flush to JSONL / CSV."""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def run(self, query: str, **kwargs: Any) -> TraceRecord:
        rec = traced_run(query, **kwargs)
        self.records.append(rec)
        return rec

    def to_dicts(self) -> list[dict[str, Any]]:
        rows = []
        for r in self.records:
            d = asdict(r)
            d["techniques_fired"] = r.techniques_csv()
            d["sources"] = "|".join(r.sources)
            # Keep contexts out of flat CSV by default (too large); RAGAS uses object.
            d.pop("contexts", None)
            d["node_latencies_ms"] = json.dumps(r.node_latencies_ms)
            rows.append(d)
        return rows

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self.records:
                payload = asdict(r)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_csv(self, path: Path, extra_fields: dict[str, list[Any]] | None = None) -> None:
        """
        Write the Phase 6 summary CSV.

        Columns: query | route_taken | latency_ms | faithfulness | relevance |
                 techniques_fired  (+ optional extras if provided).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        extras = extra_fields or {}
        fieldnames = [
            "query",
            "route_taken",
            "latency_ms",
            "faithfulness",
            "relevance",
            "techniques_fired",
        ]
        rows = []
        for i, r in enumerate(self.records):
            row = {
                "query": r.query,
                "route_taken": r.route_taken,
                "latency_ms": r.latency_ms,
                "faithfulness": extras.get("faithfulness", [""] * len(self.records))[i]
                if "faithfulness" in extras
                else "",
                "relevance": extras.get("relevance", [""] * len(self.records))[i]
                if "relevance" in extras
                else "",
                "techniques_fired": r.techniques_csv(),
            }
            rows.append(row)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def default_results_dir() -> Path:
    return PROJECT_ROOT / "eval" / "results"
