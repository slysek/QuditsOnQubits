from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    benchmark_direct_basis_candidates,
    default_iqm_quantum_circuits_dir,
    default_iqm_results_dir,
    default_quantum_circuits_dir,
    timestamped_results_path,
)
from qudits_on_qubits.benchmarks.direct_basis.selection import (
    DEFAULT_APPROXIMATION_THRESHOLDS,
    SelectionConfig,
    materialize_selected_artifacts,
    parse_approximation_thresholds,
    require_supported_bell_state,
    selection_label,
)
from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    candidates_from_old_csv,
    generate_all_qutrit_u3_candidates,
    generate_legacy_qutrit_u3_candidates,
    generate_sanity_basis_candidates,
    generate_v2_stage1_direct_candidates,
    limit_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    backend_metadata,
    load_iqm_backend,
    safe_backend_slug,
)
from qudits_on_qubits.encoding_search.candidates import CandidateSearchConfig
from qudits_on_qubits.core.project_paths import repo_path, repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run direct qutrit-basis encoding benchmarks.",
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--n-qutrits", type=int, default=None)
    parser.add_argument(
        "--candidate-set",
        choices=("sanity", "all-qutrit-u3", "old_qutrit", "v2-stage1", "from-old-csv"),
        default="all-qutrit-u3",
        help=(
            "sanity: I/F3/F3dg plus a small phase/permutation/random set; "
            "all-qutrit-u3: full encoding_search_v2 stage-1 pool plus non-duplicated legacy qutrit classes; "
            "old_qutrit: legacy qutrit U(3) classes only; "
            "v2-stage1: raw encoding_search_v2 stage-1 pool converted for direct benchmarking; "
            "from-old-csv: regenerate candidates requested by an old CSV"
        ),
    )
    parser.add_argument("--old-csv", default=None, help="Old CSV for --candidate-set from-old-csv.")
    parser.add_argument("--random-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-class",
        dest="candidate_classes",
        action="append",
        default=None,
        help=(
            "Benchmark only candidates whose class_name matches this value, "
            "for example monomial_full or product. May be passed multiple times "
            "or as a comma-separated list."
        ),
    )
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--max-monomial-full", type=int, default=None)
    parser.add_argument("--max-product", type=int, default=None)
    parser.add_argument("--include-product-grid", action="store_true")
    parser.add_argument("--max-product-grid", type=int, default=None)
    parser.add_argument("--product-grid-phase-steps", type=int, default=4)
    parser.add_argument("--product-grid-polar-steps", type=int, default=3)
    parser.add_argument("--include-near-identity", action="store_true")
    parser.add_argument("--near-identity-samples-per-eps", type=int, default=2)
    parser.add_argument("--near-identity-seed", type=int, default=500)
    parser.add_argument("--n-transpile-runs", type=int, default=1)
    parser.add_argument(
        "--iqm-backend",
        default=None,
        help="IQM backend name to use for transpilation, for example garnet.",
    )
    parser.add_argument(
        "--iqm-use-metrics",
        action="store_true",
        help="Load IQM calibration metrics when constructing the backend.",
    )
    parser.add_argument(
        "--layout-method",
        default=None,
        help="Qiskit layout method passed to the backend transpiler.",
    )
    parser.add_argument(
        "--routing-method",
        default=None,
        help="Qiskit routing method passed to the backend transpiler.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--quantum-circuits-dir",
        default=None,
        help=(
            "Directory for per-candidate QPY exports. Defaults to "
            "artifacts/direct_basis_runs/raw/quantum_circuits."
        ),
    )
    parser.add_argument(
        "--no-export-quantum-circuits",
        action="store_true",
        help="Disable per-candidate QPY exports of F3_W, CZ3_W, and the full circuit.",
    )
    parser.add_argument("--no-fidelity", action="store_true")
    parser.add_argument("--max-fidelity-qubits", type=int, default=10)
    parser.add_argument(
        "--approximation-thresholds",
        default=None,
        help=(
            "Comma-separated approximation_degree thresholds. "
            "When set, exact plus these threshold runs are benchmarked. "
            "Default pipeline thresholds are 0.99,0.95,0.90."
        ),
    )
    parser.add_argument(
        "--select-top-k",
        type=int,
        default=None,
        help="Copy Top-K selected circuits into artifacts/direct_basis_runs/selected_best.",
    )
    parser.add_argument(
        "--selection-run-id",
        default=None,
        help="Timestamp/id for selected_best output. Defaults to current YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--local-line-coupling",
        action="store_true",
        help="Use a nearest-neighbor line over the physical qubits for quick smoke runs.",
    )
    return parser


def _candidate_config_from_args(args, *, include_limit: bool) -> CandidateSearchConfig:
    return CandidateSearchConfig(
        max_monomial_full=args.max_monomial_full,
        max_product=args.max_product,
        include_product_grid=args.include_product_grid,
        max_product_grid=args.max_product_grid,
        product_grid_phase_steps=args.product_grid_phase_steps,
        product_grid_polar_steps=args.product_grid_polar_steps,
        include_near_identity=args.include_near_identity,
        near_identity_samples_per_eps=args.near_identity_samples_per_eps,
        near_identity_seed=args.near_identity_seed,
        limit_candidates=args.limit_candidates if include_limit else None,
    )


def _safe_filename_part(value) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "value"


def _candidate_classes_from_args(args) -> tuple[str, ...]:
    values = getattr(args, "candidate_classes", None) or ()
    classes: list[str] = []
    for value in values:
        for class_name in str(value).split(","):
            class_name = class_name.strip()
            if class_name:
                classes.append(class_name)
    return tuple(classes)


def _default_results_prefix(args) -> str:
    from qudits_on_qubits.encoding_search.states import resolve_benchmark_state

    state = resolve_benchmark_state(args.state, n_qutrits=args.n_qutrits)
    if getattr(args, "iqm_backend", None):
        parts = [
            "direct_basis",
            "iqm",
            safe_backend_slug(args.iqm_backend),
            _safe_filename_part(state.state_id),
            _safe_filename_part(args.candidate_set),
        ]
    else:
        parts = [
            "direct_basis",
            _safe_filename_part(state.state_id),
            _safe_filename_part(args.candidate_set),
        ]
    candidate_classes = _candidate_classes_from_args(args)
    if candidate_classes:
        classes = "_".join(_safe_filename_part(class_name) for class_name in candidate_classes)
        parts.append(f"classes_{classes}")
    if args.limit_candidates is not None:
        parts.append(f"limit{int(args.limit_candidates)}")
    parts.append(f"runs{int(args.n_transpile_runs)}")
    return "_".join(parts)


def _deduplicate_candidates(candidates):
    seen: set[tuple[str, str]] = set()
    unique = []
    for candidate in candidates:
        key = (candidate.class_name, candidate.candidate_name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _filter_candidates_by_class(candidates, candidate_classes: tuple[str, ...], limit=None):
    values = list(candidates)
    if not candidate_classes:
        return limit_candidates(values, limit)

    requested = set(candidate_classes)
    baseline = [candidate for candidate in values if candidate.class_name == "baseline"]
    selected = [
        candidate
        for candidate in values
        if candidate.class_name in requested and candidate.class_name != "baseline"
    ]
    if not selected and not ("baseline" in requested and baseline):
        available = ", ".join(sorted({candidate.class_name for candidate in values}))
        requested_text = ", ".join(candidate_classes)
        raise ValueError(
            f"No candidates matched --candidate-class {requested_text}. "
            f"Available classes: {available}"
        )
    return _deduplicate_candidates(baseline + limit_candidates(selected, limit))


def _load_candidates(args):
    candidate_classes = _candidate_classes_from_args(args)
    if args.candidate_set == "sanity":
        candidates = generate_sanity_basis_candidates(
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.candidate_set == "all-qutrit-u3":
        candidates = generate_all_qutrit_u3_candidates(
            candidate_config=_candidate_config_from_args(args, include_limit=False),
        )
    elif args.candidate_set == "old_qutrit":
        candidates = generate_legacy_qutrit_u3_candidates("old_qutrit")
    elif args.candidate_set == "v2-stage1":
        candidates = generate_v2_stage1_direct_candidates(
            include_unsupported=True,
            candidate_config=_candidate_config_from_args(
                args,
                include_limit=not candidate_classes,
            ),
        )
    else:
        if not args.old_csv:
            raise ValueError("--old-csv is required for --candidate-set from-old-csv.")
        candidates = candidates_from_old_csv(args.old_csv, include_unsupported=True)
    return _filter_candidates_by_class(
        candidates,
        candidate_classes,
        limit=args.limit_candidates,
    )


def _validate_cli_selection_args(args) -> None:
    if args.select_top_k is not None and int(args.select_top_k) < 1:
        raise ValueError("--select-top-k must be positive.")
    if args.select_top_k is not None:
        require_supported_bell_state(args.state)
    if args.no_fidelity and args.approximation_thresholds:
        raise ValueError("--no-fidelity cannot be combined with --approximation-thresholds.")
    if args.select_top_k is not None and args.no_export_quantum_circuits:
        raise ValueError("--select-top-k requires quantum circuit exports.")


def _resolved_approximation_thresholds(args) -> tuple[float, ...]:
    if args.approximation_thresholds is None:
        return ()
    parsed = parse_approximation_thresholds(args.approximation_thresholds)
    return parsed or DEFAULT_APPROXIMATION_THRESHOLDS


def _selection_labels_for_run(thresholds: tuple[float, ...]) -> tuple[str, ...]:
    return ("exact",) + tuple(selection_label(value) for value in thresholds)


def _iqm_quantum_circuits_dir_from_args(args) -> str:
    return default_iqm_quantum_circuits_dir(safe_backend_slug(args.iqm_backend))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_selection_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    approximation_thresholds = _resolved_approximation_thresholds(args)
    candidates = _load_candidates(args)
    transpiler_backend = None
    transpiler_metadata = None
    output_dir = args.output_dir
    quantum_circuits_dir = args.quantum_circuits_dir

    if args.iqm_backend:
        transpiler_backend = load_iqm_backend(
            args.iqm_backend,
            use_metrics=args.iqm_use_metrics,
        )
        transpiler_metadata = backend_metadata(
            transpiler_backend,
            iqm_backend_name=args.iqm_backend,
            iqm_use_metrics=args.iqm_use_metrics,
            optimization_level=3,
            layout_method=args.layout_method,
            routing_method=args.routing_method,
        )
        if output_dir is None:
            output_dir = default_iqm_results_dir()
        if quantum_circuits_dir is None:
            quantum_circuits_dir = _iqm_quantum_circuits_dir_from_args(args)

    output_csv = args.output_csv or timestamped_results_path(
        output_dir=output_dir,
        prefix=_default_results_prefix(args),
    )

    print(
        f"Running direct_basis_encoding: state={args.state}, "
        f"candidates={len(candidates)}, output={output_csv}"
    )
    coupling_map = None
    if args.local_line_coupling:
        from qudits_on_qubits.encoding_search.states import resolve_benchmark_state

        state = resolve_benchmark_state(args.state, n_qutrits=args.n_qutrits)
        n_physical = 2 * state.num_qutrits
        coupling_map = [[idx, idx + 1] for idx in range(n_physical - 1)]

    df, path = benchmark_direct_basis_candidates(
        state_name=args.state,
        n_qutrits=args.n_qutrits,
        candidates=candidates,
        coupling_map=coupling_map,
        n_transpile_runs=args.n_transpile_runs,
        compute_fidelity=not args.no_fidelity,
        max_fidelity_qubits=args.max_fidelity_qubits,
        output_csv=output_csv,
        quantum_circuits_dir=(
            None
            if args.no_export_quantum_circuits
            else (quantum_circuits_dir or default_quantum_circuits_dir())
        ),
        approximation_degrees=approximation_thresholds or None,
        transpiler_backend=transpiler_backend,
        transpiler_metadata=transpiler_metadata,
        optimization_level=3,
        layout_method=args.layout_method,
        routing_method=args.routing_method,
    )
    print(f"Done. Results saved to: {path}")

    if args.select_top_k is not None:
        run_id = args.selection_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        selection_output = materialize_selected_artifacts(
            df,
            SelectionConfig(
                repo_root=Path(repo_root()),
                state_name=args.state,
                run_id=run_id,
                top_k=int(args.select_top_k),
                labels=_selection_labels_for_run(approximation_thresholds),
                processed_dir=Path(repo_path("artifacts", "direct_basis_runs", "processed")),
                selected_root=Path(repo_path("artifacts", "direct_basis_runs", "selected_best")),
            ),
        )
        print(f"Selected manifest: {selection_output.manifest_csv}")
        print(f"Processed selected manifest: {selection_output.processed_manifest_csv}")
        for warning in selection_output.warnings:
            print(f"WARNING: {warning}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
