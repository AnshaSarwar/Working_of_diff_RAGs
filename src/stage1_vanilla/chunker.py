"""
Structure-aware markdown chunker for Phase 1 vanilla RAG.

Why split on headers instead of fixed character counts?
Fixed-size chunking often cuts mid-section (e.g. half of "Equipment" lands in
the next chunk) and loses the heading that tells the LLM *what* the text is
about. Splitting on # / ## / ### keeps each policy subsection intact and
attaches a breadcrumb path so retrieval stays section-aware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.loaders.okf_loader import OKFDocument

# Match ATX headers at levels 1–3 only (as agreed for Phase 1).
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class TextChunk:
    """One retrieval unit derived from an OKF document section."""

    chunk_id: str
    text: str
    """Text sent to the embedder / stuffed into the LLM prompt."""

    header_path: str
    """Breadcrumb of headers, e.g. 'Remote Work Policy > Equipment'."""

    source_path: str
    """Bundle-relative path of the parent file."""

    doc_type: str
    tags: list[str] = field(default_factory=list)
    title: str | None = None
    outgoing_links: list[str] = field(default_factory=list)
    """Copied from the parent doc — ready for Phase 2 graph building."""


def _build_header_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def chunk_okf_document(doc: OKFDocument) -> list[TextChunk]:
    """
    Split one OKF document on markdown headers (#, ##, ###).

    Algorithm:
    1. Find all header positions.
    2. Maintain a stack of (level, title) so nested ## under # become a path.
    3. Emit one chunk per section that has non-empty body text.

    Sections that are only a header with no body are skipped — empty chunks
    waste embedding slots and pollute nearest-neighbor search.
    """
    body = doc.body or ""
    matches = list(_HEADER_RE.finditer(body))

    # No headers → treat the whole body as a single chunk.
    if not matches:
        text = body.strip()
        if not text:
            return []
        header_path = doc.title or doc.bundle_path
        return [
            TextChunk(
                chunk_id=f"{doc.bundle_path}::0",
                text=_format_chunk_text(header_path, text, doc),
                header_path=header_path,
                source_path=doc.bundle_path,
                doc_type=doc.doc_type,
                tags=list(doc.tags),
                title=doc.title,
                outgoing_links=list(doc.outgoing_links),
            )
        ]

    chunks: list[TextChunk] = []
    stack: list[tuple[int, str]] = []
    chunk_index = 0

    # Preamble before the first header (rare in OKF files, but handle it).
    preamble = body[: matches[0].start()].strip()
    if preamble:
        header_path = doc.title or doc.bundle_path
        chunks.append(
            TextChunk(
                chunk_id=f"{doc.bundle_path}::{chunk_index}",
                text=_format_chunk_text(header_path, preamble, doc),
                header_path=header_path,
                source_path=doc.bundle_path,
                doc_type=doc.doc_type,
                tags=list(doc.tags),
                title=doc.title,
                outgoing_links=list(doc.outgoing_links),
            )
        )
        chunk_index += 1

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Pop deeper-or-equal headers so the stack reflects the current path.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_body = body[section_start:section_end].strip()
        if not section_body:
            continue

        header_path = _build_header_path(stack)
        chunks.append(
            TextChunk(
                chunk_id=f"{doc.bundle_path}::{chunk_index}",
                text=_format_chunk_text(header_path, section_body, doc),
                header_path=header_path,
                source_path=doc.bundle_path,
                doc_type=doc.doc_type,
                tags=list(doc.tags),
                title=doc.title,
                outgoing_links=list(doc.outgoing_links),
            )
        )
        chunk_index += 1

    return chunks


def _format_chunk_text(header_path: str, section_body: str, doc: OKFDocument) -> str:
    """
    Prefix section text with source metadata.

    Why?
    The LLM never sees the vector store metadata unless we put it in the
    prompt string. A short header path + source path makes citations and
    multi-doc answers much clearer.
    """
    parts = [
        f"Source: {doc.bundle_path}",
        f"Type: {doc.doc_type}",
        f"Section: {header_path}",
        "",
        section_body,
    ]
    return "\n".join(parts)


def chunk_okf_documents(docs: list[OKFDocument]) -> list[TextChunk]:
    """Chunk an entire loaded OKF bundle."""
    chunks: list[TextChunk] = []
    for doc in docs:
        chunks.extend(chunk_okf_document(doc))
    return chunks
