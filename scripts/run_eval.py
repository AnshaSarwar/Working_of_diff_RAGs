"""
Run Phase 6 evaluation from the project root:

    python scripts/run_eval.py
    python scripts/run_eval.py --limit 3 --skip-ragas
    python scripts/run_eval.py --limit 3
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

from eval.ragas_runner import main


if __name__ == "__main__":
    main()
