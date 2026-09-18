"""Single command-line entry point for the unified two-qutrit example."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from .config import BenchmarkConfig
from .families import LocalSU2, SchmidtTheta
from .runner import run_benchmark
from .synthesis import (ExactSynthesis, OptimizedSynthesis, SavedGateSynthesis,
                        ThetaContinuationSynthesis, saved_candidate)
from .targets import BackendTarget
from .theta_continuation.artifacts import RunStore
from .theta_continuation.models import ThetaBenchmarkConfig
from .workloads import two_qutrit_graph_circuit


def parser():
    result = argparse.ArgumentParser(description="Encoding search with backend-native routing and Pareto analysis (no QPU jobs).")
    result.add_argument("--backend", choices=("local", "iqm", "ibm"), required=True)
    result.add_argument("--device", help="garnet/emerald for IQM, backend name for IBM")
    result.add_argument("--family", action="append", choices=("local-su2", "schmidt-theta"))
    result.add_argument("--local-samples", type=int, default=20)
    result.add_argument("--search-seed", type=int, default=42)
    result.add_argument("--theta-points", type=int)
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument("--synthesis", choices=("optimized", "exact", "theta"), default="optimized")
    result.add_argument("--theta-source-run", type=Path)
    result.add_argument("--theta-baseline-qpy", type=Path)
    result.add_argument("--saved-basis", type=Path)
    result.add_argument("--reuse-saved-preparation", action="store_true")
    result.add_argument("--cz3-tolerance", type=float)
    result.add_argument("--minimum-fidelity", type=float, default=1 - 1e-6)
    result.add_argument("--maximum-leakage", type=float, default=1e-6)
    result.add_argument("--max-validation-qubits", type=int, default=12)
    result.add_argument("--initial-layout", help="ordered comma-separated physical qubits")
    result.add_argument("--routing-method", default="sabre")
    result.add_argument("--layout-method", default="sabre")
    result.add_argument("--optimization-level", type=int, default=3)
    result.add_argument("--iqm-strategy", default="transpile_to_iqm_exact")
    result.add_argument("--output-dir", type=Path)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    selected = args.family or (["schmidt-theta"] if args.synthesis == "theta" else ["local-su2", "schmidt-theta"])
    if len(set(selected)) != len(selected):
        arguments.error("Each family may be selected only once")
    if args.backend != "local" and not args.device:
        arguments.error("--device is required for IQM and IBM")
    if args.backend == "local" and args.device:
        arguments.error("--device is not used with the local target")
    if args.synthesis == "theta" and selected != ["schmidt-theta"]:
        arguments.error("theta continuation requires only the schmidt-theta family; use Python strategy mapping for mixed synthesis")
    if args.synthesis != "theta" and (args.theta_source_run or args.theta_baseline_qpy):
        arguments.error("theta source/baseline options require --synthesis theta")
    if args.reuse_saved_preparation and not args.saved_basis:
        arguments.error("--reuse-saved-preparation requires --saved-basis")
    try:
        theta_points = args.theta_points if args.theta_points is not None else 41
        cz3_tolerance = args.cz3_tolerance if args.cz3_tolerance is not None else 1e-5
        if args.synthesis == "theta":
            if args.theta_source_run:
                theta_config = ThetaBenchmarkConfig.from_dict(RunStore.open(args.theta_source_run).manifest["config"])
                if args.theta_points is not None and args.theta_points != theta_config.theta_points:
                    raise ValueError("--theta-points does not match source run")
                if theta_config.limit_points is not None and theta_config.limit_points != theta_config.theta_points:
                    raise ValueError("CLI full-grid replay requires an untruncated source; use Python with explicit candidates for a prefix")
                theta_points = theta_config.theta_points
                if args.cz3_tolerance is not None and args.cz3_tolerance != theta_config.cz3_tolerance:
                    raise ValueError("--cz3-tolerance does not match source run")
                cz3_tolerance = theta_config.cz3_tolerance
            else:
                if args.theta_baseline_qpy is None:
                    raise ValueError("New theta continuation requires --theta-baseline-qpy pointing to the pinned six-CZ baseline")
                theta_config = ThetaBenchmarkConfig(theta_points=theta_points, cz3_tolerance=cz3_tolerance,
                    baseline_qpy=str(args.theta_baseline_qpy.resolve()),
                    baseline_sha256=hashlib.sha256(args.theta_baseline_qpy.read_bytes()).hexdigest())
            strategy = ThetaContinuationSynthesis(theta_config, source_run=args.theta_source_run)
        else:
            strategy = ExactSynthesis() if args.synthesis == "exact" else OptimizedSynthesis()
        families = [
            LocalSU2(samples=args.local_samples, seed=args.search_seed) if name == "local-su2"
            else SchmidtTheta(points=theta_points)
            for name in selected
        ]
        config = BenchmarkConfig(transpiler_seeds=tuple(args.seeds), cz3_tolerance=cz3_tolerance,
                                 minimum_fidelity=args.minimum_fidelity, maximum_leakage=args.maximum_leakage,
                                 max_validation_qubits=args.max_validation_qubits)
        options = dict(initial_layout=None if args.initial_layout is None else tuple(int(x) for x in args.initial_layout.split(",")),
                       routing_method=args.routing_method, layout_method=args.layout_method,
                       optimization_level=args.optimization_level)
        if args.backend == "local":
            backend = BackendTarget.local(**options)
        elif args.backend == "iqm":
            backend = BackendTarget.iqm(args.device, strategy=args.iqm_strategy, **options)
        else:
            backend = BackendTarget.ibm(args.device, **options)
        circuit = two_qutrit_graph_circuit()
        candidates = []
        policies = strategy
        if args.saved_basis:
            candidate = saved_candidate(args.saved_basis)
            candidates.append(candidate)
            policies = {next(iter(family.generate())).family: strategy for family in families}
            policies["saved"] = SavedGateSynthesis(args.saved_basis,
                workload_hash=circuit.stable_hash() if args.reuse_saved_preparation else None)
        result = run_benchmark(circuit, families=families, backend=backend, config=config,
                               synthesis=policies, baseline_synthesis=strategy,
                               saved_candidates=candidates, output_dir=args.output_dir, progress=print)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"benchmark error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"trials={len(result.trials)} accepted={int(result.trials.success.sum())} pareto={len(result.pareto_front)}")
    print(f"artifacts={result.output_dir}")
    return 0


def entrypoint():
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
