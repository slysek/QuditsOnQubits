from __future__ import annotations

import argparse
import json

from .bell_builders import build_bell_operator, candidate_statevector
from .classical_bounds import bound_for_candidate
from .encoding import default_qutrit_encoding
from .estimator_backend import bell_value_estimator
from .sampler_backend import bell_value_sampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute qutrit graph-state Bell functionals for encoded qutrits.",
    )
    parser.add_argument(
        "--candidate",
        choices=("two_qutrit", "ghz3", "ame43"),
        required=True,
    )
    parser.add_argument(
        "--backend",
        choices=("estimator", "sampler"),
        required=True,
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    E = default_qutrit_encoding()
    state = candidate_statevector(args.candidate, E)
    bounds = bound_for_candidate(args.candidate)

    if args.backend == "estimator":
        bell_operator = build_bell_operator(args.candidate, E)
        result = bell_value_estimator(state, bell_operator, E=E, shots=args.shots)
    else:
        result = bell_value_sampler(
            state,
            args.candidate,
            E=E,
            shots=args.shots,
            seed=args.seed,
        )

    payload = {
        "candidate": args.candidate,
        "backend": args.backend,
        "value_real": result.value.real,
        "value_imag": result.value.imag,
        "leakage_probability": result.leakage_probability,
        "shots": args.shots,
        "beta_Q": bounds.quantum,
        "beta_L": bounds.classical,
        "beta_L_source": bounds.classical_source,
    }

    if args.as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"candidate: {payload['candidate']}")
        print(f"backend: {payload['backend']}")
        print(f"I: {payload['value_real']:.12g} + {payload['value_imag']:.3g}j")
        print(f"leakage_probability: {payload['leakage_probability']:.3g}")
        print(
            "bounds: "
            f"beta_Q={payload['beta_Q']:.12g}, "
            f"beta_L={payload['beta_L']:.12g} ({payload['beta_L_source']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
