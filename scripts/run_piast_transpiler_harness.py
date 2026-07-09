from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    candidates_from_old_csv,
    generate_all_qutrit_u3_candidates,
    generate_legacy_qutrit_u3_candidates,
    generate_sanity_basis_candidates,
    generate_v2_stage1_direct_candidates,
    limit_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.circuits import resolve_direct_state
from qudits_on_qubits.benchmarks.direct_basis.piast_backend import load_piast_backend
from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_harness import (
    PiastTranspilerHarnessConfig,
    default_piast_transpiler_harness_output_dir,
    default_piast_transpiler_harness_run_id,
    run_piast_transpiler_harness,
    write_piast_transpiler_harness_outputs,
)
from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies import (
    piast_transpiler_strategy_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Piast/AQT transpilation strategies for direct-basis candidates.",
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--n-qutrits", type=int, default=None)
    parser.add_argument(
        "--candidate-set",
        choices=("sanity", "all-qutrit-u3", "old_qutrit", "v2-stage1", "from-old-csv"),
        default="from-old-csv",
    )
    parser.add_argument("--old-csv", default=None)
    parser.add_argument("--random-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument(
        "--piast",
        action="store_true",
        help="Enable Piast direct-access backend loading through PCSS QAPI.",
    )
    parser.add_argument("--n-transpile-runs", type=_positive_int, default=1)
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=piast_transpiler_strategy_names(),
        help="Piast/AQT transpiler strategy to run. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--quantum-circuits-dir",
        default=None,
        help="Directory for exported QPY and E/W matrix artifacts.",
    )
    parser.add_argument(
        "--no-export-quantum-circuits",
        action="store_true",
        help="Disable per-candidate QPY and E/W matrix artifact exports.",
    )
    parser.add_argument("--max-depth-warning", type=int, default=100)
    parser.add_argument("--max-rxx-warning", type=int, default=50)
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _safe_filename_part(value) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "value"


def _default_results_prefix(args) -> str:
    return "_".join(
        [
            "piast_transpiler_harness",
            "piast",
            _safe_filename_part(args.state),
            _safe_filename_part(args.candidate_set),
            f"runs{int(args.n_transpile_runs)}",
        ]
    )


def _load_candidates(args):
    if args.candidate_set == "sanity":
        candidates = generate_sanity_basis_candidates(
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.candidate_set == "all-qutrit-u3":
        candidates = generate_all_qutrit_u3_candidates()
    elif args.candidate_set == "old_qutrit":
        candidates = generate_legacy_qutrit_u3_candidates("old_qutrit")
    elif args.candidate_set == "v2-stage1":
        candidates = generate_v2_stage1_direct_candidates(include_unsupported=True)
    elif args.candidate_set == "from-old-csv":
        if not args.old_csv:
            raise ValueError("--old-csv is required for --candidate-set from-old-csv.")
        candidates = candidates_from_old_csv(args.old_csv, include_unsupported=True)
    else:
        raise ValueError(f"Unsupported candidate set: {args.candidate_set}")
    return limit_candidates(candidates, args.limit_candidates)


def _validate_state_args(args) -> None:
    resolve_direct_state(args.state, n_qutrits=args.n_qutrits)


def _validate_piast_args(parser: argparse.ArgumentParser, args) -> None:
    if not args.piast:
        parser.error("--piast is required for Piast transpilation.")


def _output_dir_from_args(args) -> str:
    if args.output_dir:
        return args.output_dir
    if args.run_id:
        return default_piast_transpiler_harness_output_dir(args.run_id)
    return default_piast_transpiler_harness_output_dir(
        default_piast_transpiler_harness_run_id()
    )


def _quantum_circuits_dir_from_args(args, output_dir: str) -> str | None:
    if args.no_export_quantum_circuits:
        return None
    if args.quantum_circuits_dir:
        return args.quantum_circuits_dir
    return str(Path(output_dir) / "quantum_circuits")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_piast_args(parser, args)

    try:
        _validate_state_args(args)
        candidates = _load_candidates(args)
    except ValueError as exc:
        parser.error(str(exc))

    backend = load_piast_backend()
    output_dir = _output_dir_from_args(args)
    config = PiastTranspilerHarnessConfig(
        state_name=args.state,
        n_qutrits=args.n_qutrits,
        backend=backend,
        candidates=candidates,
        quantum_circuits_dir=_quantum_circuits_dir_from_args(args, output_dir),
        strategy_names=tuple(args.strategy),
        n_transpile_runs=args.n_transpile_runs,
        optimization_level=3,
        max_depth_warning=args.max_depth_warning,
        max_rxx_warning=args.max_rxx_warning,
    )

    all_trials, best_by_candidate, summary = run_piast_transpiler_harness(config)
    paths = write_piast_transpiler_harness_outputs(
        output_dir,
        all_trials=all_trials,
        best_by_candidate=best_by_candidate,
        summary=summary,
    )

    print(f"All trials CSV: {paths['all_trials_csv']}")
    print(f"Best by candidate CSV: {paths['best_by_candidate_csv']}")
    print(f"Summary JSON: {paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
