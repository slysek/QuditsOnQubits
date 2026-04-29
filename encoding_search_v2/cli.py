import argparse

from encoding_search_v2.candidates import generate_stage1_candidates
from encoding_search_v2.reports import write_state_report
from encoding_search_v2.runner import PipelineConfig, run_stage1, run_stage2
from encoding_search_v2.triviality import _filter_trivial_candidates


STATES = ("two_qutrit", "ghz3", "ame43")
RANK_BY = ("exact_depth", "exact_2q", "fid085", "fid090", "fid095")


def _parse_float_list(value):
    if value is None:
        return None
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Stage-based qutrit encoding search v2 pipeline",
    )
    parser.add_argument("--state", required=True, choices=STATES)
    parser.add_argument("--stage", required=True, choices=("1", "2", "report"))
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ranking-csv", default=None, help="stage-1 CSV or ranking for stage 2")
    parser.add_argument("--rank-by", choices=RANK_BY, default="exact_depth")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--n-transpile-runs", type=int, default=20)
    parser.add_argument("--export-circuits", action="store_true")
    parser.add_argument("--approximation-values", default=None)
    parser.add_argument("--fidelity-thresholds", default="0.85,0.90,0.95")
    parser.add_argument("--include-product-grid", action="store_true")
    parser.add_argument("--product-grid-phase-steps", type=int, default=4)
    parser.add_argument("--product-grid-polar-steps", type=int, default=3)
    parser.add_argument("--include-near-identity", action="store_true")
    parser.add_argument("--near-identity-samples-per-eps", type=int, default=2)
    parser.add_argument("--max-monomial-full", type=int, default=None)
    parser.add_argument("--max-product", type=int, default=None)
    parser.add_argument("--max-product-grid", type=int, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="absolute tolerance for baseline-equivalence checks",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="relative tolerance for baseline-equivalence checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print candidate count and exit without transpilation",
    )
    return parser


def _config_from_args(args) -> PipelineConfig:
    return PipelineConfig(
        state_name=args.state,
        stage=1 if args.stage == "1" else 2,
        output_root=args.output_root,
        jobs=args.jobs,
        n_transpile_runs=args.n_transpile_runs,
        top_k=args.top_k,
        ranking_csv=args.ranking_csv,
        rank_by=args.rank_by,
        export_circuits=args.export_circuits,
        approximation_values=_parse_float_list(args.approximation_values),
        fidelity_thresholds=_parse_float_list(args.fidelity_thresholds) or (0.85, 0.90, 0.95),
        max_monomial_full=args.max_monomial_full,
        max_product=args.max_product,
        include_product_grid=args.include_product_grid,
        max_product_grid=args.max_product_grid,
        product_grid_phase_steps=args.product_grid_phase_steps,
        product_grid_polar_steps=args.product_grid_polar_steps,
        include_near_identity=args.include_near_identity,
        near_identity_samples_per_eps=args.near_identity_samples_per_eps,
        limit_candidates=args.limit_candidates,
        atol=args.atol,
        rtol=args.rtol,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stage == "report":
        path = write_state_report(state_name=args.state, output_root=args.output_root)
        print(f"Report written to: {path}")
        return 0

    config = _config_from_args(args)
    if args.dry_run:
        candidates = generate_stage1_candidates(config.candidate_config())
        benchmarked, skipped = _filter_trivial_candidates(
            candidates,
            state_name=args.state,
            stage=int(args.stage),
            atol=config.atol,
            rtol=config.rtol,
        )
        print(
            f"Dry run [{args.state}, stage {args.stage}]: "
            f"{len(candidates)} generated, {len(benchmarked)} benchmarked, "
            f"{len(skipped)} baseline-equivalent skipped"
        )
        return 0

    if args.stage == "1":
        _, paths = run_stage1(config)
    else:
        _, paths = run_stage2(config)

    print("Written files:")
    for key, path in sorted(paths.items()):
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
