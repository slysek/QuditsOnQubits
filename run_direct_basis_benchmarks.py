from __future__ import annotations

import argparse

from basis_direct_encoding_benchmarks.benchmark import (
    benchmark_direct_basis_candidates,
    default_quantum_circuits_dir,
    timestamped_results_path,
)
from basis_direct_encoding_benchmarks.candidates import (
    candidates_from_old_csv,
    generate_legacy_qutrit_u3_candidates,
    generate_sanity_basis_candidates,
    limit_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run direct qutrit-basis encoding benchmarks.",
    )
    parser.add_argument("--state", default="two_qutrit")
    parser.add_argument("--n-qutrits", type=int, default=None)
    parser.add_argument(
        "--candidate-set",
        choices=("sanity", "all-qutrit-u3", "old_qutrit", "from-old-csv"),
        default="sanity",
        help=(
            "sanity: I/F3/F3dg plus a small phase/permutation/random set; "
            "all-qutrit-u3: all regenerated legacy candidates that are qutrit U(3) bases; "
            "old_qutrit: compatibility alias for all-qutrit-u3; "
            "from-old-csv: regenerate candidates requested by an old CSV"
        ),
    )
    parser.add_argument("--old-csv", default=None, help="Old CSV for --candidate-set from-old-csv.")
    parser.add_argument("--random-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--n-transpile-runs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--quantum-circuits-dir",
        default=None,
        help=(
            "Directory for per-candidate QPY exports. Defaults to "
            "basis_direct_encoding_benchmarks/quantum_circuits."
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
        "--local-line-coupling",
        action="store_true",
        help="Use a nearest-neighbor line over the physical qubits for quick smoke runs.",
    )
    return parser


def _load_candidates(args):
    if args.candidate_set == "sanity":
        candidates = generate_sanity_basis_candidates(
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.candidate_set in ("all-qutrit-u3", "old_qutrit"):
        candidates = generate_legacy_qutrit_u3_candidates("old_qutrit")
    else:
        if not args.old_csv:
            raise ValueError("--old-csv is required for --candidate-set from-old-csv.")
        candidates = candidates_from_old_csv(args.old_csv, include_unsupported=True)
    return limit_candidates(candidates, args.limit_candidates)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    candidates = _load_candidates(args)
    output_csv = args.output_csv or timestamped_results_path(output_dir=args.output_dir)

    print(
        f"Running direct_basis_encoding: state={args.state}, "
        f"candidates={len(candidates)}, output={output_csv}"
    )
    coupling_map = None
    if args.local_line_coupling:
        from encoding_search_v2.states import resolve_benchmark_state

        state = resolve_benchmark_state(args.state, n_qutrits=args.n_qutrits)
        n_physical = 2 * state.num_qutrits
        coupling_map = [[idx, idx + 1] for idx in range(n_physical - 1)]

    _, path = benchmark_direct_basis_candidates(
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
            else (args.quantum_circuits_dir or default_quantum_circuits_dir())
        ),
    )
    print(f"Done. Results saved to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
