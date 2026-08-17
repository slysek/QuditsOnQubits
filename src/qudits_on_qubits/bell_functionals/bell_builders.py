from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from qiskit.quantum_info import Statevector

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .encoding import (
    default_qutrit_encoding,
    embed_operator_E,
    encode_qutrit_state,
    kron_all,
)


@dataclass(frozen=True)
class LocalFactor:
    party: int
    setting: int
    power: int
    base_observable: np.ndarray
    matrix: np.ndarray
    label: str


@dataclass(frozen=True)
class BellTerm:
    coefficient: complex
    factors: tuple[LocalFactor, ...]


def num_qutrits_for_candidate(candidate: str) -> int:
    return get_reference_experiment(candidate).state.num_parties


def build_bell_operator_two_qutrit(E: np.ndarray | None = None) -> np.ndarray:
    return build_bell_operator("two_qutrit", E)


def build_bell_operator_ghz_graph(E: np.ndarray | None = None) -> np.ndarray:
    return build_bell_operator("ghz3", E)


def build_bell_operator_ame43(E: np.ndarray | None = None) -> np.ndarray:
    return build_bell_operator("ame43", E)


def build_bell_operator(candidate: str, E: np.ndarray | None = None) -> np.ndarray:
    encoding = default_qutrit_encoding() if E is None else E
    n = num_qutrits_for_candidate(candidate)
    size = 4**n
    bell = np.zeros((size, size), dtype=complex)
    for term in bell_terms(candidate):
        bell += _embed_bell_term(term, encoding, n)
    return np.real_if_close(bell, tol=1000).astype(complex)


def bell_terms(candidate: str) -> tuple[BellTerm, ...]:
    spec = get_reference_experiment(candidate)
    terms: list[BellTerm] = []
    for term in spec.bell_functional.terms:
        factors: list[LocalFactor] = []
        for factor in term.factors:
            observable = spec.observable(factor.setting_label)
            match = re.fullmatch(r"[A-Z]+(\d+)", factor.setting_label)
            if match is None:
                raise ValueError(
                    f"invalid observable setting label: {factor.setting_label!r}"
                )
            factors.append(
                LocalFactor(
                    party=factor.party,
                    setting=int(match.group(1)),
                    power=factor.outcome_power,
                    base_observable=observable.as_array(),
                    matrix=factor.logical_operator(observable),
                    label=factor.setting_label,
                )
            )
        terms.append(BellTerm(coefficient=term.coefficient, factors=tuple(factors)))
    return tuple(terms)


def candidate_statevector(
    candidate: str,
    E: np.ndarray | None = None,
) -> Statevector:
    """Return the encoded ideal state for a supported graph-state candidate."""
    spec = get_reference_experiment(candidate)
    encoding = (
        default_qutrit_encoding()
        if E is None
        else np.asarray(E, dtype=complex)
    )
    return Statevector(
        encode_qutrit_state(
            spec.state.statevector(),
            encoding,
            spec.state.num_parties,
        )
    )


def _embed_bell_term(term: BellTerm, E: np.ndarray, num_qutrits: int) -> np.ndarray:
    local = [np.eye(4, dtype=complex) for _ in range(num_qutrits)]
    for factor in term.factors:
        local[factor.party] = embed_operator_E(E, factor.matrix)
    return term.coefficient * kron_all(local)
