"""
Phase 5 demo — Tiered routing + semantic cache.

Shows classify → Phase 1 / 2 / 3, then a paraphrase cache hit.

    python scripts/demo_stage5.py
    python scripts/demo_stage5.py --no-cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.semantic_cache import get_semantic_cache, run_routed_rag
from src.routing.classify_query import classify_query

DEMO_QUESTIONS = [
    ("What is the remote work policy?", "expect ~ SIMPLE_FACTUAL → Phase 1"),
    ("Who manages the person leading Project Atlas?", "expect ~ MULTI_HOP → Phase 2"),
    (
        "Can I expense a conference ticket if I'm remote?",
        "expect ~ COMPLEX_AMBIGUOUS → Phase 3",
    ),
]


def _print_result(label: str, result, *, elapsed_ms: float) -> None:
    print("-" * 72)
    print(label)
    print("-" * 72)
    print(f"  route:      {result.route}")
    print(f"  cache_hit:  {result.cache_hit}")
    print(f"  filters:    {result.metadata_filter_summary}")
    print(f"  techniques: {', '.join(result.techniques)}")
    print(f"  sources:    {', '.join(result.sources[:6])}")
    print(f"  latency:    {elapsed_ms:.0f} ms")
    if result.error:
        print(f"  error:      {result.error}")
    print("\n  Answer:")
    answer = result.answer or "(empty)"
    for line in answer.splitlines():
        print(f"    {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 routing + cache demo")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable semantic cache for this run",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Wipe the semantic cache file before running",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional single question (otherwise runs the 3 demo questions)",
    )
    args = parser.parse_args()

    if args.clear_cache:
        get_semantic_cache().clear()
        print("Semantic cache cleared.\n")

    use_cache = not args.no_cache

    print("=" * 72)
    print("Phase 5 — Tiered routing + semantic cache")
    print("=" * 72)
    print(f"Cache enabled: {use_cache}\n")

    if args.question:
        questions = [(args.question, "custom")]
    else:
        questions = DEMO_QUESTIONS

    for q, note in questions:
        print(f"Question: {q}")
        print(f"  ({note})")
        guessed = classify_query(q)
        print(f"  classifier preview: {guessed.value}")

        t0 = time.perf_counter()
        result = run_routed_rag(q, use_cache=use_cache, store_in_cache=use_cache)
        elapsed = (time.perf_counter() - t0) * 1000
        _print_result("Router result", result, elapsed_ms=elapsed)

    if use_cache and not args.question:
        # Paraphrase of the first question — should hit cache if embeddings agree.
        paraphrase = "What's our remote work policy?"
        print(f"Cache test paraphrase: {paraphrase}")
        t0 = time.perf_counter()
        cached = run_routed_rag(paraphrase, use_cache=True, store_in_cache=False)
        elapsed = (time.perf_counter() - t0) * 1000
        _print_result("Paraphrase lookup", cached, elapsed_ms=elapsed)

    print("Done.")


if __name__ == "__main__":
    main()
