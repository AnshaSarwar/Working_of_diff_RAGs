"""
Phase 1 demo — Vanilla RAG in isolation.

Run from the project root:
    python scripts/demo_stage1.py

Optional:
    python scripts/demo_stage1.py --rebuild   # force re-embed into Chroma
    python scripts/demo_stage1.py "Your question here"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles often default to cp1252; force UTF-8 so LLM answers
# with en-dashes / non-breaking hyphens don't crash print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow `python scripts/demo_stage1.py` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_vanilla.retrieve_generate import run_vanilla_rag


DEFAULT_QUESTION = "What is the remote work policy?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Vanilla RAG demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f'Question to ask (default: "{DEFAULT_QUESTION}")',
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Chroma index from the OKF bundle before querying",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 1 — Vanilla RAG demo")
    print("=" * 72)
    print(f"Question: {args.question}\n")

    result = run_vanilla_rag(
        args.question,
        top_k=args.top_k,
        rebuild_store=args.rebuild,
    )

    print("-" * 72)
    print("Retrieved chunks")
    print("-" * 72)
    if not result.retrieved_chunks:
        print("(none)")
    for i, doc in enumerate(result.retrieved_chunks, start=1):
        meta = doc.metadata
        print(f"\n[{i}] {meta.get('source_path')} | {meta.get('header_path')}")
        print(f"    type={meta.get('type')}  tags={meta.get('tags')}")
        preview = doc.page_content.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        print(f"    {preview}")

    print("\n" + "-" * 72)
    print("Answer")
    print("-" * 72)
    print(result.answer)
    print()


if __name__ == "__main__":
    main()
