"""Reassess a completed theta benchmark at an explicit CZ3 acceptance limit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cz3-tolerance", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        from qudits_on_qubits.benchmarks.theta_continuation.reassessment import run_reassessment
        result = run_reassessment(**vars(args), progress=lambda message: print(message, flush=True))
        print(json.dumps({"stage": "completed", "counts": result["counts"], "report": result["report"]},
                         ensure_ascii=True, allow_nan=False), flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; saved raw recoveries and complete points remain resumable.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"Configuration/artifact error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Execution error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
