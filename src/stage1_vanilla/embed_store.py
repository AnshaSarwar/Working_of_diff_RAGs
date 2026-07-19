"""
Embed OKF chunks and persist them in a local Chroma vector store.

Why Chroma (local)?
- No cloud account — fits a learning project
- Persists to disk so you don't re-embed on every demo run
- Metadata filters (type / tags) are available for Phase 5 routing later

Why attach type + tags as metadata?
Vanilla retrieval is similarity-only. Metadata lets later stages filter
(e.g. "only HR Policy docs") without changing the embedding space.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    OKF_BUNDLE_DIR,
)
from src.loaders.okf_loader import load_okf_bundle
from src.stage1_vanilla.chunker import TextChunk, chunk_okf_documents


def get_embeddings() -> FastEmbedEmbeddings:
    """
    Local MiniLM embedder via fastembed (ONNX Runtime).

    First call downloads the model weights; later calls reuse the cache.
    We use fastembed instead of sentence-transformers so we don't need torch
    (which often fails to install on Windows due to MAX_PATH limits).
    """
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def chunks_to_documents(chunks: list[TextChunk]) -> list[Document]:
    """Convert our TextChunk dataclass into LangChain Documents for Chroma."""
    docs: list[Document] = []
    for chunk in chunks:
        # Chroma metadata values must be scalar (str/int/float/bool).
        # Store tags as a comma-separated string; Phase 5 can split them back.
        metadata = {
            "chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "header_path": chunk.header_path,
            "type": chunk.doc_type,
            "tags": ",".join(chunk.tags),
            "title": chunk.title or "",
            # Keep links as a string for Phase 2 reuse / debugging.
            "outgoing_links": ",".join(chunk.outgoing_links),
        }
        docs.append(Document(page_content=chunk.text, metadata=metadata))
    return docs


def build_vector_store(
    *,
    bundle_dir: Path | None = None,
    persist_dir: Path | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    rebuild: bool = False,
) -> Chroma:
    """
    Load OKF → chunk → embed → Chroma.

    If a persisted store already exists and rebuild=False, open it instead of
    re-embedding (much faster for repeated demos).
    """
    bundle_dir = bundle_dir or OKF_BUNDLE_DIR
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    # Reuse existing collection when possible.
    if not rebuild and any(persist_dir.iterdir()):
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    okf_docs = load_okf_bundle(bundle_dir, skip_index_type=True)
    chunks = chunk_okf_documents(okf_docs)
    documents = chunks_to_documents(chunks)

    if not documents:
        raise RuntimeError(f"No chunks produced from OKF bundle at {bundle_dir}")

    # Drop prior collection contents when rebuilding so IDs stay consistent.
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    if rebuild:
        try:
            store.delete_collection()
        except Exception:
            # Collection may not exist yet on a fresh rebuild.
            pass
        store = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    store.add_documents(
        documents,
        ids=[c.chunk_id for c in chunks],
    )
    return store


def get_retriever(store: Chroma, top_k: int = 4):
    """Thin wrapper — keeps top_k configurable for demos and Phase 4 hybrid."""
    return store.as_retriever(search_kwargs={"k": top_k})
