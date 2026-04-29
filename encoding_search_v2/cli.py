import argparse

from encoding_search_v2.candidates import generate_stage1_candidates
from encoding_search_v2.reports import write_state_report
from encoding_search_v2.runner import PipelineConfig, run_stage1, run_stage2
from encoding_search_v2.states import parse_n_values, resolve_benchmark_state
from encoding_search_v2.triviality import _filter_trivial_candidates


RANK_BY = ("exact_depth", "exact_2q", "fid085", "fid090", "fid095")


def _parse_float_list(value):
    if value is None:
        return None
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Stage-based qutrit encoding search v2 pipeline",
    )
    parser.add_argument(
        "--state",
        required=True,
        help="two_qutrit, ghz3, ame43, ghz_star, ghz_n, or ghz_star_<n>",
    )
    parser.add_argument("--stage", required=True, choices=("1", "2", "report"))
    parser.add_argument(
        "--n-qutrits",
        "--num-qutrits",
        dest="n_qutrits",
        type=int,
        default=None,
        help="number of qutrits for --state ghz_star / ghz_n",
    )
    parser.add_argument(
        "--n-values",
        default=None,
        help="comma-separated n values for a GHZ star range, e.g. 3,4,5,6",
    )
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


def _config_from_args(args, *, n_qutrits=None, ranking_csv=None) -> PipelineConfig:
    return PipelineConfig(
        state_name=args.state,
        stage=1 if args.stage == "1" else 2,
        n_qutrits=n_qutrits if n_qutrits is not None else args.n_qutrits,
        output_root=args.output_root,
        jobs=args.jobs,
        n_transpile_runs=args.n_transpile_runs,
        top_k=args.top_k,
        ranking_csv=ranking_csv if ranking_csv is not None else args.ranking_csv,
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


def _requested_n_values(args, parser) -> tuple:
    try:
        values = parse_n_values(args.n_values)
    except ValueError as exc:
        parser.error(str(exc))

    if values and args.n_qutrits is not None:
        parser.error("Use either --n-qutrits or --n-values, not both.")
    if values:
        return values
    if args.n_qutrits is not None:
        return (args.n_qutrits,)
    return (None,)


def _ranking_csv_for_state(template, state_id, n_qutrits):
    if template is None:
        return None
    if "{n}" in template or "{state_id}" in template:
        return template.format(n=n_qutrits, state_id=state_id)
    return template


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    n_values = _requested_n_values(args, parser)

    if args.stage == "2" and len(n_values) > 1 and args.ranking_csv:
        if "{n}" not in args.ranking_csv and "{state_id}" not in args.ranking_csv:
            parser.error(
                "Stage 2 with --n-values requires --ranking-csv containing "
                "{n} or {state_id} so each n gets its own stage-1 ranking."
            )

    if args.stage == "report":
        for n_qutrits in n_values:
            try:
                state_spec = resolve_benchmark_state(args.state, n_qutrits=n_qutrits)
            except ValueError as exc:
                parser.error(str(exc))
            path = write_state_report(state_name=state_spec.state_id, output_root=args.output_root)
            print(f"Report written to: {path}")
        return 0

    all_paths = []
    for n_qutrits in n_values:
        try:
            state_spec = resolve_benchmark_state(args.state, n_qutrits=n_qutrits)
        except ValueError as exc:
            parser.error(str(exc))
        ranking_csv = _ranking_csv_for_state(args.ranking_csv, state_spec.state_id, state_spec.num_qutrits)
        config = _config_from_args(args, n_qutrits=n_qutrits, ranking_csv=ranking_csv)

        if args.dry_run:
            candidates = generate_stage1_candidates(config.candidate_config())
            benchmarked, skipped = _filter_trivial_candidates(
                candidates,
                state_name=state_spec.state_id,
                stage=int(args.stage),
                atol=config.atol,
                rtol=config.rtol,
            )
            print(
                f"Dry run [{state_spec.state_id}, stage {args.stage}]: "
                f"{len(candidates)} generated, {len(benchmarked)} benchmarked, "
                f"{len(skipped)} baseline-equivalent skipped"
            )
            continue

        if args.stage == "1":
            _, paths = run_stage1(config)
        else:
            _, paths = run_stage2(config)
        all_paths.append((state_spec.state_id, paths))

    if args.dry_run:
        return 0

    print("Written files:")
    for state_id, paths in all_paths:
        print(f"[{state_id}]")
        for key, path in sorted(paths.items()):
            print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
