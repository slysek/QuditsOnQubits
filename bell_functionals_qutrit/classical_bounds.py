from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from .bell_builders import bell_terms
from .operators import make_XZ_qutrit


@dataclass(frozen=True)
class BoundResult:
    candidate: str
    quantum: float
    classical: float
    classical_source: str


def bound_for_candidate(candidate: str) -> BoundResult:
    if candidate == "two_qutrit":
        return BoundResult(
            candidate=candidate,
            quantum=6.0,
            classical=6 * math.cos(math.pi / 9),
            classical_source="pdf",
        )
    if candidate == "ame43":
        return BoundResult(
            candidate=candidate,
            quantum=8.0,
            classical=7.63816,
            classical_source="pdf",
        )
    if candidate == "ghz3":
        return brute_force_classical_bound(candidate)
    raise ValueError(f"unknown candidate: {candidate!r}")


@lru_cache(maxsize=None)
def brute_force_classical_bound(candidate: str) -> BoundResult:
    _, _, omega = make_XZ_qutrit()
    terms = bell_terms(candidate)
    settings = _settings_by_party(terms)
    ordered_keys = [(party, setting) for party in sorted(settings) for setting in settings[party]]

    best = -float("inf")
    for assignment_values in product(range(3), repeat=len(ordered_keys)):
        assignment = dict(zip(ordered_keys, assignment_values))
        value = 0.0 + 0.0j
        for term in terms:
            phase = 1.0 + 0.0j
            for factor in term.factors:
                phase *= omega ** (factor.power * assignment[(factor.party, factor.setting)])
            value += term.coefficient * phase
        best = max(best, float(value.real))

    return BoundResult(
        candidate=candidate,
        quantum=_quantum_bound(candidate),
        classical=best,
        classical_source="numeric_bruteforce",
    )


def _settings_by_party(terms):
    settings: dict[int, set[int]] = {}
    for term in terms:
        for factor in term.factors:
            settings.setdefault(factor.party, set()).add(factor.setting)
    return {party: tuple(sorted(values)) for party, values in settings.items()}


def _quantum_bound(candidate: str) -> float:
    if candidate == "two_qutrit":
        return 6.0
    if candidate == "ame43":
        return 8.0
    if candidate == "ghz3":
        d = 3
        n = 3
        n1 = 2
        return float((d - 1) * (n - n1 + d - 1))
    raise ValueError(f"unknown candidate: {candidate!r}")
