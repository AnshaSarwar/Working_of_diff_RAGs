"""
OKF markdown loader (Google Cloud Open Knowledge Format v0.1).

Parses concept documents (YAML frontmatter + body) and normalizes markdown
links for GraphRAG. Reserved files (`index.md`, `log.md`) are never treated
as concepts — see https://github.com/GoogleCloudPlatform/knowledge-catalog
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

# Markdown links: [label](target) — ignore images (![alt](url)) and bare URLs.
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")

# OKF §3.1 — reserved at every directory level; not concept documents.
RESERVED_FILENAMES = frozenset({"index.md", "log.md"})


def is_reserved_okf_filename(path: Path) -> bool:
    """True for OKF reserved names (index.md / log.md), case-insensitive."""
    return path.name.lower() in RESERVED_FILENAMES


@dataclass
class OKFDocument:
    """One OKF-conformant concept markdown file, parsed into structured fields."""

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

    @property
    def concept_id(self) -> str:
        """
        OKF concept ID: bundle-relative path without the `.md` suffix.

        Example: 'team/alice.md' → 'team/alice'
        """
        p = self.bundle_path
        lower = p.lower()
        if lower.endswith(".markdown"):
            return p[: -len(".markdown")]
        if lower.endswith(".md"):
            return p[:-3]
        return p


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
    - OKF supports absolute (`/team/alice.md`) and relative (`./bob.md`) links.
    - Neo4j needs stable node IDs; bundle-relative paths are the natural key.
    - Skip external URLs / fragments / directory-only / reserved files —
      they aren't concept documents we should traverse as graph nodes.
    """
    target = target.strip()
    if not target:
        return None

    # Drop fragment anchors: leave_policy.md#carryover → leave_policy.md
    target = target.split("#", 1)[0].strip()
    if not target:
        return None

    # External / scheme-based links are not OKF concept edges (citations OK in body).
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

    # Links into reserved filenames are not concept edges (OKF §3.1).
    if is_reserved_okf_filename(Path(rel.as_posix())):
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
    Parse a single OKF *concept* markdown file.

    Returns None if:
    - the path is a reserved OKF filename (`index.md` / `log.md`), or
    - `type` is missing (not concept-conformant).
    """
    if is_reserved_okf_filename(path):
        return None

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
    skip_reserved: bool = True,
    skip_index_type: bool = True,
) -> list[OKFDocument]:
    """
    Walk an OKF bundle and return concept documents only.

    By default:
    - Skips reserved filenames `index.md` / `log.md` (OKF §3.1) — also enforced
      inside `load_okf_file`.
    - Skips any non-reserved file mistakenly tagged `type: Index` (demo safety).
    """
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"OKF bundle not found: {bundle_root}")

    docs: list[OKFDocument] = []
    for path in sorted(bundle_root.rglob("*.md")):
        if skip_reserved and is_reserved_okf_filename(path):
            continue
        doc = load_okf_file(path, bundle_root)
        if doc is None:
            continue
        if skip_index_type and doc.doc_type.lower() == "index":
            continue
        docs.append(doc)

    return docs
