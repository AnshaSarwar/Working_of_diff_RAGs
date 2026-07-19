"""
Phase 4 demo — Hybrid BM25+vector (RRF) + cross-encoder rerank.

Shows why codes like PTO250 need keyword search: vector-only often misses
them; hybrid surfaces the leave policy via BM25 literal match.

    python scripts/demo_stage4.py
    python scripts/demo_stage4.py "What is PTO250?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HYBRID_CANDIDATE_K, RERANK_TOP_K
from src.stage1_vanilla.embed_store import build_vector_store
from src.stage1_vanilla.retrieve_generate import run_vanilla_rag
from src.stage4_advanced.hybrid_search import get_bm25_index, hybrid_search, vector_search
from src.stage4_advanced.pipeline import advanced_retrieve, run_advanced_rag

DEFAULT_QUESTION = "What is PTO250?"


def _print_docs(title: str, docs: list, *, limit: int = 5) -> None:
    print("-" * 72)
    print(title)
    print("-" * 72)
    if not docs:
        print("  (none)")
        return
    for i, doc in enumerate(docs[:limit], start=1):
        meta = doc.metadata or {}
        src = meta.get("source_path", "?")
        section = meta.get("header_path", "")
        rrf = meta.get("rrf_score")
        rr = meta.get("rerank_score")
        extras = []
        if rrf is not None:
            extras.append(f"rrf={rrf}")
        if rr is not None:
            extras.append(f"rerank={rr}")
        extra = f"  ({', '.join(extras)})" if extras else ""
        print(f"  [{i}] {src} | {section}{extra}")
        preview = doc.page_content.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "…"
        print(f"      {preview}")


def _leave_policy_rank(docs: list) -> int | None:
    for i, doc in enumerate(docs, start=1):
        src = (doc.metadata or {}).get("source_path", "")
        text = doc.page_content.upper()
        if "leave_policy" in src or "PTO250" in text:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 hybrid + rerank demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f'Question to ask (default: "{DEFAULT_QUESTION}")',
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only show retrieval ranks (no LLM call)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 4 — Hybrid search + rerank demo")
    print("=" * 72)
    print(f"Question: {args.question}")
    print(f"Hybrid candidates: {HYBRID_CANDIDATE_K}  →  rerank top: {RERANK_TOP_K}\n")

    store = build_vector_store(rebuild=False)
    bm25_index = get_bm25_index(store)

    vector_docs = vector_search(store, args.question, top_k=HYBRID_CANDIDATE_K)
    bm25_docs = bm25_index.bm25_search(args.question, top_k=HYBRID_CANDIDATE_K)
    hybrid_docs = hybrid_search(
        store,
        args.question,
        bm25_index=bm25_index,
        candidate_k=HYBRID_CANDIDATE_K,
    )
    final_docs = advanced_retrieve(
        args.question,
        store=store,
        use_hybrid=True,
        use_rerank=True,
    )

    _print_docs("Vector-only (top 5 of pool)", vector_docs, limit=5)
    print(f"\n  leave/PTO250 rank in vector pool: {_leave_policy_rank(vector_docs)}")

    _print_docs("BM25-only (top 5)", bm25_docs, limit=5)
    print(f"\n  leave/PTO250 rank in BM25: {_leave_policy_rank(bm25_docs)}")

    _print_docs("Hybrid RRF (top 5)", hybrid_docs, limit=5)
    print(f"\n  leave/PTO250 rank after RRF: {_leave_policy_rank(hybrid_docs)}")

    _print_docs("Hybrid + rerank (final top-k)", final_docs, limit=RERANK_TOP_K)
    print(f"\n  leave/PTO250 rank after rerank: {_leave_policy_rank(final_docs)}")

    if not args.skip_generate:
        print("\n" + "=" * 72)
        print("Generate with Phase 4 advanced path (hybrid+rerank)")
        print("=" * 72)
        result = run_advanced_rag(args.question, use_hybrid=True, use_rerank=True)
        print(f"Techniques: {', '.join(result.techniques)}")
        print("\nAnswer:")
        print(result.answer)

        print("\n" + "=" * 72)
        print("Generate with Phase 1 vector-only (for contrast)")
        print("=" * 72)
        weak = run_vanilla_rag(args.question)
        print(f"Techniques: {', '.join(weak.techniques)}")
        print("Retrieved:")
        for i, doc in enumerate(weak.retrieved_chunks, start=1):
            print(f"  [{i}] {doc.metadata.get('source_path')} | {doc.metadata.get('header_path')}")
        print("\nAnswer:")
        print(weak.answer)

    print()


if __name__ == "__main__":
    main()
