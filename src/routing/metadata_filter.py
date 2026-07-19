"""
OKF metadata filtering (type / tags).

Why filter on every route?
Vector/hybrid search is similarity-only. OKF frontmatter (`type`, `tags`) is
structured signal — e.g. prefer HR Policy chunks for "What is PTO250?".
Phase 5 applies the same filter helper on vanilla, graph, and agentic paths
so routing changes *how* we retrieve, not whether metadata is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class MetadataFilters:
    """Soft filters inferred from the query (empty = no filtering)."""

    types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.types and not self.tags


def infer_metadata_filters(query: str) -> MetadataFilters:
    """
    Cheap heuristic hints — no LLM call.

    Conservative on purpose: wrong hard filters drop the right docs. We treat
    these as *preferences*; apply_metadata_filter falls back if nothing matches.
    """
    q = query.lower()
    types: list[str] = []
    tags: list[str] = []

    if any(w in q for w in ("policy", "pto", "leave", "expense", "remote", "stipend")):
        types.append("HR Policy")
        tags.extend(["policy", "hr"])
    if "pto250" in q.replace(" ", "") or "pto 250" in q:
        tags.append("PTO250")
    if any(w in q for w in ("project", "atlas", "nova", "orion")):
        types.append("Project")
        tags.append("project")
    if any(w in q for w in ("who", "manager", "manages", "reports", "team", "owner")):
        types.append("Person")
        tags.append("team")
    if any(w in q for w in ("jira", "slack", "bigquery", "tool")):
        types.append("Tool")
        tags.append("tool")

    # Deduplicate preserving order.
    def uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            key = x.lower()
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    return MetadataFilters(types=uniq(types), tags=uniq(tags))


def _doc_matches(doc: Document, filters: MetadataFilters) -> bool:
    meta = doc.metadata or {}
    doc_type = str(meta.get("type") or "")
    doc_tags = {
        t.strip().lower()
        for t in str(meta.get("tags") or "").split(",")
        if t.strip()
    }

    type_ok = True
    if filters.types:
        type_ok = any(
            ft.lower() in doc_type.lower() or doc_type.lower() in ft.lower()
            for ft in filters.types
        )

    tags_ok = True
    if filters.tags:
        wanted = {t.lower() for t in filters.tags}
        tags_ok = bool(doc_tags & wanted)

    # Match if *either* type or tags hits when both are set — OR keeps recall up.
    if filters.types and filters.tags:
        return type_ok or tags_ok
    if filters.types:
        return type_ok
    if filters.tags:
        return tags_ok
    return True


def apply_metadata_filter(
    documents: list[Document],
    filters: MetadataFilters | None,
    *,
    fallback_to_all: bool = True,
) -> list[Document]:
    """
    Filter retrieved docs by OKF type/tags.

    If filtering would remove *everything*, return the original list when
    fallback_to_all=True (safer for a tiny learning corpus).
    """
    if not documents or filters is None or filters.is_empty():
        return documents

    kept = [d for d in documents if _doc_matches(d, filters)]
    if not kept and fallback_to_all:
        return documents
    return kept


def filter_graph_nodes(
    nodes: list,
    filters: MetadataFilters | None,
    *,
    type_attr: str = "doc_type",
) -> list:
    """
    Apply the same type preference to GraphRAG neighborhood nodes.

    Graph nodes don't carry comma-tags the same way; we match on `doc_type`.
    """
    if not nodes or filters is None or not filters.types:
        return nodes

    kept = []
    for n in nodes:
        doc_type = str(getattr(n, type_attr, "") or "")
        if any(ft.lower() in doc_type.lower() for ft in filters.types):
            kept.append(n)
    return kept or nodes
