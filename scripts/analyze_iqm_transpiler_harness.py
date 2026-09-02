"""Re-rank completed IQM transpiler trials without contacting a backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.pareto_selection import (
    analyze_iqm_trials,
    write_pareto_analysis_outputs,
)
from qudits_on_qubits.benchmarks.direct_basis.phase_equivalence import (
    PHASE_DUPLICATE_COLUMNS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-transpilation Pareto analysis of completed IQM trials; "
            "uses no backend, hardware, or transpilation."
        )
    )
    parser.add_argument("--all-trials", required=True, metavar="PATH")
    parser.add_argument("--output-dir", metavar="PATH", default=None)
    parser.add_argument("--two-qubit-weight", type=float, default=0.50)
    parser.add_argument("--depth-weight", type=float, default=0.30)
    parser.add_argument("--std-depth-weight", type=float, default=0.20)
    parser.add_argument("--max-state-qubits", type=int, default=12)
    return parser


def _load_existing_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"summary JSON is invalid: {error.msg}") from error
    if not isinstance(summary, dict):
        raise ValueError("summary JSON must contain a top-level object")
    return summary


def _write_phase_audit_if_absent(path: Path) -> None:
    if not path.exists():
        pd.DataFrame(columns=PHASE_DUPLICATE_COLUMNS).to_csv(path, index=False)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    all_trials_path = Path(args.all_trials).resolve()
    if not all_trials_path.is_file():
        parser.error(f"all-trials CSV does not exist: {all_trials_path}")
    if args.max_state_qubits < 1:
        parser.error("max-state-qubits must be at least 1")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else all_trials_path.parent
    all_trials = pd.read_csv(all_trials_path)
    objective_weights = {
        "mean_two_qubit_gate_count": args.two_qubit_weight,
        "mean_depth": args.depth_weight,
        "std_depth": args.std_depth_weight,
    }
    analysis = analyze_iqm_trials(
        all_trials,
        objective_weights=objective_weights,
        max_state_qubits=args.max_state_qubits,
    )
    analysis_paths = write_pareto_analysis_outputs(output_dir, analysis)
    phase_csv = output_dir / "candidate_global_phase_duplicates.csv"
    _write_phase_audit_if_absent(phase_csv)
    summary_json = output_dir / "summary.json"
    existing_summary = _load_existing_summary(summary_json)
    summary_json.write_text(
        json.dumps({**existing_summary, **analysis.summary_counts}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Candidate global phase duplicates CSV: {phase_csv}")
    print(f"Strategy statistics CSV: {analysis_paths['strategy_statistics_csv']}")
    print(f"Pareto ranked CSV: {analysis_paths['pareto_ranked_csv']}")
    print(f"State equivalence groups CSV: {analysis_paths['state_equivalence_groups_csv']}")
    print(f"Recommended circuits CSV: {analysis_paths['recommended_circuits_csv']}")
    print(f"Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
