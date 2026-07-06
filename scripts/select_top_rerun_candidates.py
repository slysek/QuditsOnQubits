from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qudits_on_qubits.benchmarks.direct_basis.rerun_selection import (
    RerunSelectionConfig,
    write_rerun_selection_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select per-state direct-basis candidates for reruns.",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="Input benchmark CSV. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/iqm_runs/processed/rerun_selection",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to current YYYYMMDD_HHMMSS timestamp.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-label", default=None)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        output = write_rerun_selection_files(
            RerunSelectionConfig(
                input_csvs=tuple(Path(value) for value in args.input_csv),
                output_root=Path(args.output_root),
                run_id=run_id,
                top_k=args.top_k,
                include_label=args.include_label,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Output dir: {output.output_dir}")
    for state_output in output.state_outputs:
        print(
            f"{state_output.state_name}: {state_output.csv_path} "
            f"candidates={state_output.selected_count} "
            "baseline_equivalent_excluded="
            f"{state_output.baseline_equivalent_excluded_count} "
            f"unresolved={state_output.unresolved_count}"
        )
    for warning in output.warnings:
        print(f"WARNING: {warning}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
