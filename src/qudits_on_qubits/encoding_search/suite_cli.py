"""CLI dla jednoetapowego pipeline'u benchmarkowego (suite).

Przyklad uruchomienia nocnego:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m encoding_search_v2.suite_cli \
        --suite graph_states_extended \
        --jobs 32

Zmienne srodowiskowe ``OMP_NUM_THREADS=1`` itd. nakazuja BLAS/OMP nie
forkowac dodatkowych watkow w kazdym workerze, co znacznie redukuje
oversubscription na maszynach 32-rdzeniowych (kazdy proces transpilera
qiskit i tak swietnie wykorzystuje pojedynczy rdzen).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from qudits_on_qubits.core.graph_states import (
    BENCHMARK_SUITES,
    list_suites,
)
from qudits_on_qubits.encoding_search.suite import SuiteConfig, run_benchmark_suite
from qudits_on_qubits.encoding_search.runner import DEFAULT_FIDELITY_THRESHOLDS


def _parse_float_list(value):
    if value is None:
        return None
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_list(value):
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items) if items else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-stage benchmark suite over many qutrit graph states "
            "and ALL implemented encoding base classes."
        ),
    )
    parser.add_argument(
        "--suite",
        default="graph_states_extended",
        help=(
            "Predefined benchmark suite to run. "
            f"Available: {', '.join(list_suites())}."
        ),
    )
    parser.add_argument(
        "--states",
        default=None,
        help=(
            "Optional comma-separated override of state names "
            "(e.g. 'path5,cycle6,wheel5'). When given, --suite is ignored."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (use 32 for a 32-core night run).",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--n-transpile-runs", type=int, default=20)
    parser.add_argument(
        "--candidate-mode",
        choices=["full", "original", "extended"],
        default="full",
        help=(
            "Which candidate generators to use. 'full' (default) covers ALL "
            "implemented base classes; the user typically wants this."
        ),
    )
    parser.add_argument(
        "--class-filter",
        default=None,
        help="Optional comma-separated list of class names to keep.",
    )
    parser.add_argument(
        "--encoding-strategy",
        default="append_w",
        choices=("append_w", "prepared_w_then_conjugated_entanglers"),
        help="Which circuit-build strategy to benchmark (default: append_w).",
    )
    parser.add_argument(
        "--fidelity-thresholds",
        default=None,
        help="Comma-separated fidelity thresholds (default: 0.85,0.90,0.95).",
    )
    parser.add_argument(
        "--approximation-values",
        default=None,
        help="Comma-separated approximation_degree values for the sweep.",
    )
    parser.add_argument(
        "--approximation-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--export-circuits",
        action="store_true",
        help="Also dump per-candidate transpiled QPY files (heavy, optional).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a state if its main CSV already exists. Useful for "
            "resuming an interrupted nightly run."
        ),
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve states and print the run plan, then exit.",
    )
    return parser


def _resolve_states(args) -> tuple[str, ...]:
    if args.states:
        return tuple(item.strip() for item in args.states.split(",") if item.strip())
    if args.suite not in BENCHMARK_SUITES:
        raise SystemExit(
            f"Unknown --suite={args.suite!r}. "
            f"Available suites: {', '.join(list_suites())}."
        )
    return BENCHMARK_SUITES[args.suite]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    states = _resolve_states(args)
    suite_name = args.suite if not args.states else "custom_states"

    fidelity_thresholds = (
        _parse_float_list(args.fidelity_thresholds) or DEFAULT_FIDELITY_THRESHOLDS
    )
    approximation_values = _parse_float_list(args.approximation_values)
    class_filter = _parse_str_list(args.class_filter)

    config = SuiteConfig(
        suite_name=suite_name,
        states=tuple(states),
        jobs=int(args.jobs),
        output_root=args.output_root,
        n_transpile_runs=int(args.n_transpile_runs),
        candidate_mode=args.candidate_mode,
        class_filter=class_filter,
        encoding_strategy=args.encoding_strategy,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=int(args.approximation_seed),
        export_circuits=bool(args.export_circuits),
        skip_existing=bool(args.skip_existing),
        top_k=int(args.top_k),
    )

    if args.dry_run:
        print(f"[suite {config.suite_name}] dry-run")
        print(f"  states ({len(states)}): {', '.join(states)}")
        print(f"  jobs={config.jobs}, mode={config.candidate_mode}, "
              f"strategy={config.encoding_strategy}")
        print(f"  output_root={config.output_root}")
        print(f"  export_circuits={config.export_circuits}, "
              f"skip_existing={config.skip_existing}")
        return 0

    summary = run_benchmark_suite(config)

    print("\nSuite summary:")
    print(f"  suite={summary['suite_name']}")
    print(f"  states={len(summary['states'])}")
    for state_id in summary["states"]:
        elapsed = summary["state_durations"].get(state_id, 0.0)
        print(f"    {state_id}: {elapsed:.1f}s")
    print(f"  combined CSV: {summary['combined_csv']}")
    print(f"  total elapsed: {summary['total_elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
