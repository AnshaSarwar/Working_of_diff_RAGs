"""
One-off: build (or rebuild) the Neo4j graph from the OKF bundle.

    python scripts/build_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_graphrag.graph_builder import build_okf_graph


def main() -> None:
    print("Building OKF → Neo4j graph…")
    stats = build_okf_graph(clear=True)
    print(
        f"Done. nodes={stats.nodes} edges={stats.edges} "
        f"skipped_dangling_links={stats.skipped_dangling_links}"
    )


if __name__ == "__main__":
    main()
