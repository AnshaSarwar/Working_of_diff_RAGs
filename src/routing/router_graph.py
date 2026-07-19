"""
Top-level LangGraph router: classify → Phase 1 / 2 / 3.

Why a router?
Phase 1 is fastest, Phase 2 handles relationship hops, Phase 3 handles
multi-doc ambiguity. Classification picks the cheapest path that can still
answer — that's the efficiency story of this learning project.

SIMPLE uses Phase 4 `run_advanced_rag` (hybrid+rerank+metadata). Stage 1's
`run_vanilla_rag` stays vector-only for demos. Graph filters neighborhood
types *before* generate; agentic uses advanced_retrieve on each attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.routing.classify_query import QueryRoute, classify_query
from src.routing.metadata_filter import infer_metadata_filters
from src.stage2_graphrag.graph_retrieve import run_graph_rag
from src.stage3_agentic.graph import run_agentic_rag
from src.stage4_advanced.pipeline import run_advanced_rag


class RouterState(TypedDict, total=False):
    query: str
    route: str
    answer: str
    techniques: list[str]
    sources: list[str]
    contexts: list[str]
    metadata_filter_summary: str
    error: str
    # Per-node latencies in ms (filled by Phase 6 trace wrapper when used).
    node_latencies_ms: dict[str, float]


@dataclass
class RouterResult:
    query: str
    route: str
    answer: str
    techniques: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    metadata_filter_summary: str = ""
    cache_hit: bool = False
    error: str = ""
    node_latencies_ms: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


def _filters_summary(query: str) -> str:
    f = infer_metadata_filters(query)
    if f.is_empty():
        return "(none)"
    parts = []
    if f.types:
        parts.append("types=" + "|".join(f.types))
    if f.tags:
        parts.append("tags=" + "|".join(f.tags))
    return ", ".join(parts)


def classify_node(state: RouterState) -> dict:
    route = classify_query(state["query"])
    return {
        "route": route.value,
        "metadata_filter_summary": _filters_summary(state["query"]),
        "techniques": ["classify"],
    }


def route_simple(state: RouterState) -> dict:
    """Phase 4 advanced path (hybrid+rerank+metadata), composed for SIMPLE queries."""
    result = run_advanced_rag(state["query"], use_hybrid=True, use_rerank=True)
    sources = [
        d.metadata.get("source_path", "?") for d in result.retrieved_chunks
    ]
    contexts = [d.page_content for d in result.retrieved_chunks]
    techniques = list(state.get("techniques") or []) + list(result.techniques)
    return {
        "answer": result.answer,
        "sources": sources,
        "contexts": contexts,
        "techniques": techniques,
    }


def route_relational(state: RouterState) -> dict:
    """Phase 2 GraphRAG — type filters applied before generate inside run_graph_rag."""
    filters = infer_metadata_filters(state["query"])
    try:
        result = run_graph_rag(state["query"], metadata_filters=filters)
    except Exception as exc:  # noqa: BLE001 — Neo4j may be down; degrade gracefully
        return {
            "answer": "",
            "error": f"GraphRAG unavailable ({exc}); try starting Neo4j.",
            "sources": [],
            "contexts": [],
            "techniques": list(state.get("techniques") or []) + ["graph_rag_failed"],
        }

    sources = [n.path for n in result.neighborhood]
    contexts = [n.body for n in result.neighborhood if getattr(n, "body", None)]
    techniques = list(state.get("techniques") or []) + ["graph_rag"]
    if not filters.is_empty():
        techniques.append("metadata_filter")
    return {
        "answer": result.answer,
        "sources": sources,
        "contexts": contexts,
        "techniques": techniques,
    }


def route_complex(state: RouterState) -> dict:
    """Phase 3 agentic — retrieve path already uses Phase 4 + metadata filters."""
    result = run_agentic_rag(state["query"])
    sources: list[str] = []
    contexts: list[str] = []
    for payload in result.get("documents") or []:
        src = (payload.get("metadata") or {}).get("source_path")
        if src:
            sources.append(src)
        text = payload.get("page_content")
        if text:
            contexts.append(text)
    techniques = list(state.get("techniques") or []) + ["agentic"]
    filters = infer_metadata_filters(state["query"])
    if not filters.is_empty():
        techniques.append("metadata_filter")
    if int(result.get("retry_count") or 0) > 0:
        techniques.append("query_rewrite")
    return {
        "answer": result.get("final_answer") or "",
        "sources": sources,
        "contexts": contexts,
        "techniques": techniques,
    }


def dispatch_route(
    state: RouterState,
) -> Literal["route_simple", "route_relational", "route_complex"]:
    route = state.get("route") or QueryRoute.SIMPLE_FACTUAL.value
    if route == QueryRoute.MULTI_HOP_RELATIONAL.value:
        return "route_relational"
    if route == QueryRoute.COMPLEX_AMBIGUOUS.value:
        return "route_complex"
    return "route_simple"


def build_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classify", classify_node)
    graph.add_node("route_simple", route_simple)
    graph.add_node("route_relational", route_relational)
    graph.add_node("route_complex", route_complex)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        dispatch_route,
        {
            "route_simple": "route_simple",
            "route_relational": "route_relational",
            "route_complex": "route_complex",
        },
    )
    graph.add_edge("route_simple", END)
    graph.add_edge("route_relational", END)
    graph.add_edge("route_complex", END)
    return graph.compile()


_compiled = None


def get_router_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_router_graph()
    return _compiled


def run_router(query: str) -> RouterResult:
    """Run classify → branch (no cache). Prefer run_routed_rag for production."""
    app = get_router_graph()
    final: RouterState = app.invoke(
        {
            "query": query,
            "route": "",
            "answer": "",
            "techniques": [],
            "sources": [],
            "contexts": [],
            "metadata_filter_summary": "",
            "error": "",
        }
    )
    return RouterResult(
        query=query,
        route=final.get("route") or "",
        answer=final.get("answer") or final.get("error") or "",
        techniques=list(final.get("techniques") or []),
        sources=list(final.get("sources") or []),
        contexts=list(final.get("contexts") or []),
        metadata_filter_summary=final.get("metadata_filter_summary") or "",
        error=final.get("error") or "",
    )
