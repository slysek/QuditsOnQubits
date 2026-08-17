from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from qudits_on_qubits.reference_experiments import get_reference_experiment


@dataclass(frozen=True)
class BoundResult:
    candidate: str
    quantum: float
    classical: float
    classical_source: str


def bound_for_candidate(candidate: str) -> BoundResult:
    spec = get_reference_experiment(candidate)
    return BoundResult(
        candidate=spec.experiment_id,
        quantum=spec.expected.ideal_bell_value,
        classical=spec.bell_functional.classical_bound,
        classical_source=spec.bell_functional.classical_bound_source,
    )


@lru_cache(maxsize=None)
def brute_force_classical_bound(candidate: str) -> BoundResult:
    spec = get_reference_experiment(candidate)
    omega = np.exp(2j * np.pi / 3)
    terms = spec.bell_functional.terms
    ordered_keys = sorted(
        {
            (factor.party, factor.setting_label)
            for term in terms
            for factor in term.factors
        }
    )

    best = -float("inf")
    for assignment_values in product(range(3), repeat=len(ordered_keys)):
        assignment = dict(zip(ordered_keys, assignment_values))
        value = 0.0 + 0.0j
        for term in terms:
            phase = 1.0 + 0.0j
            for factor in term.factors:
                phase *= omega ** (
                    factor.outcome_power
                    * assignment[(factor.party, factor.setting_label)]
                )
            value += term.sampling_coefficient() * phase
        best = max(best, float(value.real))

    return BoundResult(
        candidate=spec.experiment_id,
        quantum=spec.expected.ideal_bell_value,
        classical=best,
        classical_source="numeric_bruteforce",
    )
