"""
Phase 3 demo — Agentic RAG on a multi-document question.

The target question spans Expense Policy + Remote Work Policy, so the first
retrieval is often graded insufficient and should trigger rewrite → retrieve
(up to 2 retries).

    python scripts/demo_stage3.py
    python scripts/demo_stage3.py "Your question"
    python scripts/demo_stage3.py --compare-vanilla
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

from src.config import MAX_AGENTIC_RETRIES
from src.stage3_agentic.graph import run_agentic_rag

DEFAULT_QUESTION = "Can I expense a conference ticket if I'm remote?"


def _dedupe_attempts(log: list[dict]) -> list[dict]:
    """
    retrieve + grade_context each append a row; keep the graded row per attempt
    when present, otherwise the retrieve-only row.
    """
    by_attempt: dict[int, dict] = {}
    for row in log:
        n = int(row.get("attempt") or 0)
        prev = by_attempt.get(n)
        if prev is None or row.get("grade"):
            by_attempt[n] = row
    return [by_attempt[k] for k in sorted(by_attempt)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Agentic RAG demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f'Question to ask (default: "{DEFAULT_QUESTION}")',
    )
    parser.add_argument(
        "--compare-vanilla",
        action="store_true",
        help="Also run Phase 1 vanilla RAG for comparison",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 3 — Agentic RAG demo")
    print("=" * 72)
    print(f"Question: {args.question}")
    print(f"Max retries: {MAX_AGENTIC_RETRIES}\n")

    result = run_agentic_rag(args.question)
    attempts = _dedupe_attempts(list(result.get("attempt_log") or []))

    print("-" * 72)
    print("Attempt log (retrieve → grade; rewrite on NO)")
    print("-" * 72)
    rewrites_fired = 0
    for row in attempts:
        grade = (row.get("grade") or "?").upper()
        print(f"\nAttempt {row.get('attempt')}")
        print(f"  query:  {row.get('query_used')}")
        print(f"  sources: {', '.join(row.get('sources') or [])}")
        print(f"  grade:   {grade}")
        if row.get("grade_reason"):
            print(f"  reason:  {row.get('grade_reason')}")

    # Count how many times we actually rewrote (retry_count after run).
    rewrites_fired = int(result.get("retry_count") or 0)
    print(f"\nRewrites fired: {rewrites_fired} (cap={MAX_AGENTIC_RETRIES})")
    print(f"Final grade:    {(result.get('grade') or '?').upper()}")

    print("\n" + "-" * 72)
    print("Final answer")
    print("-" * 72)
    print(result.get("final_answer") or "(empty)")

    if args.compare_vanilla:
        from src.stage1_vanilla.retrieve_generate import run_vanilla_rag

        print("\n" + "=" * 72)
        print("Phase 1 — Vanilla RAG (same question)")
        print("=" * 72)
        vanilla = run_vanilla_rag(args.question)
        print("Retrieved:")
        for i, doc in enumerate(vanilla.retrieved_chunks, start=1):
            print(
                f"  [{i}] {doc.metadata.get('source_path')} | "
                f"{doc.metadata.get('header_path')}"
            )
        print("\nVanilla answer:")
        print(vanilla.answer)

    print()


if __name__ == "__main__":
    main()
