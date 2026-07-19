"""
OKF markdown loader.

Why extract links in Phase 1 even though vanilla RAG doesn't use them?
Phase 2 (GraphRAG) needs one node per concept file and one edge per markdown
link. Extracting and normalizing links here means we parse each file once and
reuse the same structured records for both the vector store and the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

# Markdown links: [label](target) — ignore images (![alt](url)) and bare URLs.
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


@dataclass
class OKFDocument:
    """One OKF-conformant markdown file, parsed into structured fields."""

    path: Path
    """Absolute path to the source file."""

    bundle_path: str
    """Bundle-relative path with forward slashes, e.g. 'team/alice.md'."""

    doc_type: str
    """Required OKF frontmatter field `type` (e.g. 'Person', 'HR Policy')."""

    title: str | None = None
    description: str | None = None
    resource: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None
    body: str = ""
    """Markdown body without YAML frontmatter."""

    outgoing_links: list[str] = field(default_factory=list)
    """Normalized bundle-relative link targets extracted from the body."""

    frontmatter: dict[str, Any] = field(default_factory=dict)
    """Full frontmatter dict for anything stages need later."""


def _to_bundle_relative(path: Path, bundle_root: Path) -> str:
    return path.resolve().relative_to(bundle_root.resolve()).as_posix()


def normalize_link_target(
    target: str,
    source_file: Path,
    bundle_root: Path,
) -> str | None:
    """
    Turn a markdown link target into a bundle-relative path.

    Why normalize now?
    - OKF files mix relative links (./bob.md, ../team/dave.md).
    - Neo4j needs stable node IDs; bundle-relative paths are the natural key.
    - Skip external URLs / fragments / directory-only links — they aren't
      concept files we can retrieve or traverse as graph nodes.
    """
    target = target.strip()
    if not target:
        return None

    # Drop fragment anchors: leave_policy.md#carryover → leave_policy.md
    target = target.split("#", 1)[0].strip()
    if not target:
        return None

    # External / scheme-based links are not OKF concept edges.
    if "://" in target or target.startswith("mailto:"):
        return None

    # Absolute-from-bundle style (/team/alice.md) → strip leading slash
    if target.startswith("/"):
        candidate = (bundle_root / target.lstrip("/")).resolve()
    else:
        candidate = (source_file.parent / target).resolve()

    try:
        rel = candidate.relative_to(bundle_root.resolve())
    except ValueError:
        # Link points outside the bundle — ignore for graph/RAG purposes.
        return None

    # Directory links (./team/) aren't concept files.
    if candidate.is_dir() or str(rel).endswith(("/", "\\")):
        return None

    # Only keep markdown concept files.
    if candidate.suffix.lower() not in {".md", ".markdown"}:
        # If the file doesn't exist yet, still accept .md-looking targets.
        if not target.lower().endswith((".md", ".markdown")):
            return None

    return rel.as_posix()


def extract_markdown_links(
    body: str,
    source_file: Path,
    bundle_root: Path,
) -> list[str]:
    """Extract unique outgoing markdown links, normalized to bundle paths."""
    seen: set[str] = set()
    links: list[str] = []

    for _label, raw_target in _MD_LINK_RE.findall(body):
        normalized = normalize_link_target(raw_target, source_file, bundle_root)
        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return links


def load_okf_file(path: Path, bundle_root: Path) -> OKFDocument | None:
    """
    Parse a single OKF markdown file.

    Returns None if `type` is missing (not OKF-conformant) so callers can
    skip quietly rather than crash the whole ingest.
    """
    post = frontmatter.load(path)
    meta = dict(post.metadata)
    doc_type = meta.get("type")
    if not doc_type:
        return None

    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    body = post.content or ""
    return OKFDocument(
        path=path.resolve(),
        bundle_path=_to_bundle_relative(path, bundle_root),
        doc_type=str(doc_type),
        title=meta.get("title"),
        description=meta.get("description"),
        resource=meta.get("resource"),
        tags=[str(t) for t in tags],
        timestamp=str(meta["timestamp"]) if meta.get("timestamp") is not None else None,
        body=body,
        outgoing_links=extract_markdown_links(body, path, bundle_root),
        frontmatter=meta,
    )


def load_okf_bundle(
    bundle_root: Path,
    *,
    skip_index_type: bool = True,
) -> list[OKFDocument]:
    """
    Walk an OKF bundle directory and return parsed documents.

    Why skip Index-type by default?
    The index is a table of contents, not answerable knowledge. Including it
    in the vector store adds noise to retrieval for factual queries.
    """
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"OKF bundle not found: {bundle_root}")

    docs: list[OKFDocument] = []
    for path in sorted(bundle_root.rglob("*.md")):
        doc = load_okf_file(path, bundle_root)
        if doc is None:
            continue
        if skip_index_type and doc.doc_type.lower() == "index":
            continue
        docs.append(doc)

    return docs
