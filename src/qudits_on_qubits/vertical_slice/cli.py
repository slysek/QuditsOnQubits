"""Installed command for the clean-room two-qutrit Bell reference run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..experiments.models import AerIdeal, TranspilationConfig
from .bell import BellReferenceCircuitSpec, canonical_qutrit_encoding
from .models import ExecutionSpec, QuditExperimentSpec
from .runner import run_vertical_slice


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audited two-qutrit Bell vertical slice on ideal Aer."
    )
    parser.add_argument("--shots", type=_positive_integer, default=2048)
    parser.add_argument("--seed", type=_nonnegative_integer, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/vertical_slice_runs"),
    )
    return parser


def two_qutrit_bell_spec(
    *, shots: int = 2048, seed: int = 42, output_root: Path | str
) -> QuditExperimentSpec:
    """Build the documented clean-room reference specification."""

    return QuditExperimentSpec(
        circuit=BellReferenceCircuitSpec("two_qutrit"),
        encoding=canonical_qutrit_encoding(),
        backend=AerIdeal(seed_simulator=seed),
        execution=ExecutionSpec(
            shots=shots,
            seed=seed,
            transpilation=TranspilationConfig(
                optimization_level=1, seed_transpiler=seed
            ),
        ),
        output_root=Path(output_root),
        tags={"example": "two-qutrit-bell-vertical-slice"},
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completed = run_vertical_slice(
        two_qutrit_bell_spec(
            shots=args.shots,
            seed=args.seed,
            output_root=args.output_root,
        )
    )
    result = completed.result
    print("status=completed")
    print(f"benchmark={result['benchmark']}")
    print(f"encoding={completed.manifest.encoding['encoding_id']}")
    print(f"circuit_count={result['circuit_count']}")
    print(f"shots_per_circuit={args.shots}")
    print(f"bell_unconditional={result['bell_unconditional']['real']:.10f}")
    print(f"bell_conditional={result['bell_conditional']['real']:.10f}")
    print(f"leakage_rate={result['leakage_rate']:.10f}")
    print(f"manifest={completed.artifact_dir / 'run-manifest.json'}")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
