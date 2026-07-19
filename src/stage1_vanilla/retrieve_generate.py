"""
Vanilla retrieve → stuff → generate chain.

Phase 1 is deliberately vector-only:
1. Embed the question → top-k similarity search in Chroma
2. Stuff those chunks into a prompt
3. Ask the LLM for an answer grounded in that context only

Hybrid BM25+RRF and reranking live in Phase 4 (`stage4_advanced`) and are
composed at the router (or via `run_advanced_rag`) — not imported here —
so Stage 1 stays a clean baseline for demos and A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import DEFAULT_TOP_K, OPENAI_API_KEY, OPENAI_MODEL
from src.stage1_vanilla.embed_store import build_vector_store

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Acme Corp \
using ONLY the provided context from the company knowledge base.

Rules:
- Base your answer strictly on the context below.
- If the context is insufficient, say what is missing rather than inventing facts.
- Be concise and cite the source file path when useful (e.g. policies/remote_work_policy.md).
"""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


@dataclass
class VanillaRAGResult:
    """Structured result so demos can print answer + retrieved chunks clearly."""

    question: str
    answer: str
    retrieved_chunks: list[Document]
    techniques: list[str] = field(default_factory=list)


def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)


def format_context(docs: list[Document]) -> str:
    """Join retrieved docs with clear separators for the LLM."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_path", "unknown")
        section = doc.metadata.get("header_path", "")
        parts.append(f"[Chunk {i} | {source} | {section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, docs: list[Document]) -> str:
    """Shared stuff→generate step used by vanilla and advanced pipelines."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )
    llm = get_llm()
    messages = prompt.format_messages(
        context=format_context(docs) or "(no context retrieved)",
        question=question,
    )
    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return answer.strip()


def retrieve_vanilla(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    rebuild_store: bool = False,
) -> list[Document]:
    """Pure Phase 1 retrieve: Chroma vector similarity only."""
    store = build_vector_store(rebuild=rebuild_store)
    return store.similarity_search(question, k=top_k)


def run_vanilla_rag(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    rebuild_store: bool = False,
) -> VanillaRAGResult:
    """End-to-end Phase 1 pipeline (vector retrieve → generate)."""
    retrieved = retrieve_vanilla(
        question,
        top_k=top_k,
        rebuild_store=rebuild_store,
    )
    answer = generate_answer(question, retrieved)
    return VanillaRAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved,
        techniques=["vector_only"],
    )
