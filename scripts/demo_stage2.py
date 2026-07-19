"""
Phase 2 demo — GraphRAG vs Vanilla RAG on a multi-hop question.

Run from the project root (Neo4j must be up, graph built):

    docker compose up -d
    python scripts/build_graph.py
    python scripts/demo_stage2.py

Optional:
    python scripts/demo_stage2.py --rebuild-graph
    python scripts/demo_stage2.py "Your multi-hop question"
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

from src.config import GRAPH_HOP_COUNT
from src.stage1_vanilla.retrieve_generate import run_vanilla_rag
from src.stage2_graphrag.graph_builder import build_okf_graph
from src.stage2_graphrag.graph_retrieve import run_graph_rag

DEFAULT_QUESTION = "Who manages the person leading Project Atlas?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 GraphRAG demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f'Question to ask (default: "{DEFAULT_QUESTION}")',
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Clear and rebuild the Neo4j graph from the OKF bundle first",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=GRAPH_HOP_COUNT,
        help="Neighborhood hop count for local search (default: 2)",
    )
    parser.add_argument(
        "--skip-vanilla",
        action="store_true",
        help="Only run GraphRAG (skip Phase 1 comparison)",
    )
    args = parser.parse_args()

    if args.rebuild_graph:
        print("Rebuilding Neo4j graph from OKF bundle…")
        stats = build_okf_graph(clear=True)
        print(
            f"  nodes={stats.nodes} edges={stats.edges} "
            f"skipped_dangling={stats.skipped_dangling_links}\n"
        )

    print("=" * 72)
    print("Phase 2 — GraphRAG demo")
    print("=" * 72)
    print(f"Question: {args.question}")
    print(f"Hops: {args.hops}\n")

    graph_result = run_graph_rag(args.question, hops=args.hops)

    print("-" * 72)
    print("GraphRAG — seed entities")
    print("-" * 72)
    if graph_result.seed_paths:
        for p in graph_result.seed_paths:
            print(f"  • {p}")
    else:
        print("  (none — check that the graph is built)")

    print("\n" + "-" * 72)
    print("GraphRAG — neighborhood nodes")
    print("-" * 72)
    for node in graph_result.neighborhood:
        print(
            f"  [hops={node.hops_from_seed}] {node.path} "
            f"| {node.title} | type={node.doc_type}"
        )

    print("\n" + "-" * 72)
    print("GraphRAG — answer")
    print("-" * 72)
    print(graph_result.answer)

    if not args.skip_vanilla:
        print("\n" + "=" * 72)
        print("Phase 1 — Vanilla RAG (same question, for comparison)")
        print("=" * 72)
        vanilla = run_vanilla_rag(args.question)
        print("\nRetrieved chunks:")
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
