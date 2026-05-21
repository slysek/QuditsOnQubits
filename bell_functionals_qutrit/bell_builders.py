from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from qiskit.quantum_info import Statevector

from .encoding import (
    default_qutrit_encoding,
    embed_operator_E,
    encode_qutrit_state,
    kron_all,
)
from .operators import (
    make_XZ_qutrit,
    make_measurement_observables_qutrit_d3,
    qutrit_lambda,
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
    if candidate == "two_qutrit":
        return 2
    if candidate == "ghz3":
        return 3
    if candidate == "ame43":
        return 4
    raise ValueError(f"unknown candidate: {candidate!r}")


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
    if candidate == "two_qutrit":
        return _two_qutrit_terms()
    if candidate == "ghz3":
        return _ghz3_terms()
    if candidate == "ame43":
        return _ame43_terms()
    raise ValueError(f"unknown candidate: {candidate!r}")


def candidate_statevector(
    candidate: str,
    E: np.ndarray | None = None,
) -> Statevector:
    """Return the encoded ideal state for a supported graph-state candidate."""
    encoding = default_qutrit_encoding() if E is None else E
    if candidate == "two_qutrit":
        qutrit_state = _qutrit_graph_state(2, [(0, 1, 1)])
    elif candidate == "ghz3":
        qutrit_state = _qutrit_graph_state(3, [(0, 1, 1), (0, 2, 1)])
    elif candidate == "ame43":
        qutrit_state = _qutrit_graph_state(
            4,
            [(0, 1, 1), (0, 3, 1), (1, 2, 1), (2, 3, 2)],
        )
    else:
        raise ValueError(f"unknown candidate: {candidate!r}")
    return Statevector(encode_qutrit_state(qutrit_state, encoding, len(_dims(qutrit_state))))


def _dims(qutrit_state: np.ndarray) -> tuple[int, ...]:
    n = round(math.log(np.asarray(qutrit_state).size, 3))
    if 3**n != np.asarray(qutrit_state).size:
        raise ValueError("state length is not 3^N")
    return (3,) * n


def _qutrit_graph_state(
    num_qutrits: int,
    edges: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    _, _, omega = make_XZ_qutrit()
    edge_list = list(edges)
    state = np.zeros(3**num_qutrits, dtype=complex)
    for values in np.ndindex(*((3,) * num_qutrits)):
        phase_exponent = sum(weight * values[u] * values[v] for u, v, weight in edge_list)
        index = 0
        for value in values:
            index = 3 * index + value
        state[index] = omega**phase_exponent / math.sqrt(3**num_qutrits)
    return state


def _embed_bell_term(term: BellTerm, E: np.ndarray, num_qutrits: int) -> np.ndarray:
    local = [np.eye(4, dtype=complex) for _ in range(num_qutrits)]
    for factor in term.factors:
        local[factor.party] = embed_operator_E(E, factor.matrix)
    return term.coefficient * kron_all(local)


def _powered(base: np.ndarray, power: int) -> np.ndarray:
    return np.linalg.matrix_power(base, power % 3)


def _factor(
    party: int,
    setting: int,
    power: int,
    base: np.ndarray,
    label: str,
    matrix: np.ndarray | None = None,
) -> LocalFactor:
    observable_matrix = _powered(base, power) if matrix is None else np.asarray(matrix, dtype=complex)
    return LocalFactor(
        party=party,
        setting=setting,
        power=power,
        base_observable=observable_matrix,
        matrix=observable_matrix,
        label=label,
    )


def _base_observables() -> dict[str, list[np.ndarray]]:
    x, z, _ = make_XZ_qutrit()
    return {
        "A": make_measurement_observables_qutrit_d3(1),
        "B": [z @ np.linalg.matrix_power(x, k) for k in range(3)],
        "C_ghz": [z, z @ x],
        "C_ame": [z, x],
        "D": [z, z @ x],
    }


@lru_cache(maxsize=1)
def _two_qutrit_terms() -> tuple[BellTerm, ...]:
    _, _, omega = make_XZ_qutrit()
    bases = _base_observables()
    terms: list[BellTerm] = []
    for power in (1, 2):
        a_power = make_measurement_observables_qutrit_d3(power)
        lam = qutrit_lambda(power)
        coeff_y0 = 1 / (lam * math.sqrt(3))
        coeff_y1 = 1 / (lam * (omega ** (2 * power)) * math.sqrt(3))
        coeff_y2 = coeff_y0
        y_specs = [
            (0, coeff_y0, lambda x, p: 1),
            (1, coeff_y1, lambda x, p: omega ** (-p * x)),
            (2, coeff_y2, lambda x, p: omega ** (-2 * p * x)),
        ]
        for y, base_coeff, phase_fn in y_specs:
            for x in range(3):
                coeff = base_coeff * phase_fn(x, power)
                terms.append(
                    BellTerm(
                        coeff,
                        (
                            _factor(0, x, power, bases["A"][x], f"A{x}", matrix=a_power[x]),
                            _factor(1, y, power, bases["B"][y], f"B{y}"),
                        ),
                    )
                )
    return tuple(terms)


@lru_cache(maxsize=1)
def _ghz3_terms() -> tuple[BellTerm, ...]:
    _, _, omega = make_XZ_qutrit()
    bases = _base_observables()
    terms: list[BellTerm] = []
    for power in (1, 2):
        a_power = make_measurement_observables_qutrit_d3(power)
        lam = qutrit_lambda(power)
        a1 = 1 / (lam * math.sqrt(3))
        a2 = (omega**power) / (2 * lam * math.sqrt(3))
        specs = [
            (0, 0, a1, lambda x, p: 1),
            (1, 0, a2, lambda x, p: omega ** (-p * x)),
            (2, 0, a1, lambda x, p: omega ** (-2 * p * x)),
            (0, 1, a2, lambda x, p: omega ** (-p * x)),
        ]
        for b_setting, c_setting, base_coeff, phase_fn in specs:
            for a_setting in range(3):
                coeff = base_coeff * phase_fn(a_setting, power)
                terms.append(
                    BellTerm(
                        coeff,
                        (
                            _factor(
                                0,
                                a_setting,
                                power,
                                bases["A"][a_setting],
                                f"A{a_setting}",
                                matrix=a_power[a_setting],
                            ),
                            _factor(1, b_setting, power, bases["B"][b_setting], f"B{b_setting}"),
                            _factor(2, c_setting, power, bases["C_ghz"][c_setting], f"C{c_setting}"),
                        ),
                    )
                )
    return tuple(terms)


@lru_cache(maxsize=1)
def _ame43_terms() -> tuple[BellTerm, ...]:
    _, _, omega = make_XZ_qutrit()
    bases = _base_observables()
    terms: list[BellTerm] = []
    for power in (1, 2):
        a_power = make_measurement_observables_qutrit_d3(power)
        lam = qutrit_lambda(power)
        a1 = 1 / (math.sqrt(3) * lam)
        a2 = 1 / (2 * math.sqrt(3) * lam * (omega ** (2 * power)))

        for a_setting in range(3):
            terms.append(
                BellTerm(
                    a1,
                    (
                        _factor(
                            0,
                            a_setting,
                            power,
                            bases["A"][a_setting],
                            f"A{a_setting}",
                            matrix=a_power[a_setting],
                        ),
                        _factor(1, 0, power, bases["B"][0], "B0"),
                        _factor(3, 0, power, bases["D"][0], "D0"),
                    ),
                )
            )

        for a_setting in range(3):
            phase = [1, omega ** (-2 * power), omega ** (-power)][a_setting]
            terms.append(
                BellTerm(
                    a1 * phase,
                    (
                        _factor(
                            0,
                            a_setting,
                            power,
                            bases["A"][a_setting],
                            f"A{a_setting}",
                            matrix=a_power[a_setting],
                        ),
                        _factor(1, 2, power, bases["B"][2], "B2"),
                        _factor(2, 0, 2 * power, bases["C_ame"][0], "C0"),
                        _factor(3, 0, power, bases["D"][0], "D0"),
                    ),
                )
            )

        for a_setting in range(3):
            phase = [1, omega ** (-power), omega ** (-2 * power)][a_setting]
            terms.append(
                BellTerm(
                    a2 * phase,
                    (
                        _factor(
                            0,
                            a_setting,
                            power,
                            bases["A"][a_setting],
                            f"A{a_setting}",
                            matrix=a_power[a_setting],
                        ),
                        _factor(1, 1, power, bases["B"][1], "B1"),
                        _factor(2, 0, power, bases["C_ame"][0], "C0"),
                        _factor(3, 0, power, bases["D"][0], "D0"),
                    ),
                )
            )

        for a_setting in range(3):
            phase = [1, omega ** (-power), omega ** (-2 * power)][a_setting]
            terms.append(
                BellTerm(
                    a2 * phase,
                    (
                        _factor(
                            0,
                            a_setting,
                            power,
                            bases["A"][a_setting],
                            f"A{a_setting}",
                            matrix=a_power[a_setting],
                        ),
                        _factor(1, 0, power, bases["B"][0], "B0"),
                        _factor(2, 0, 2 * power, bases["C_ame"][0], "C0"),
                        _factor(3, 1, power, bases["D"][1], "D1"),
                    ),
                )
            )

        terms.append(
            BellTerm(
                1.0,
                (
                    _factor(1, 0, power, bases["B"][0], "B0"),
                    _factor(2, 1, power, bases["C_ame"][1], "C1"),
                    _factor(3, 0, 2 * power, bases["D"][0], "D0"),
                ),
            )
        )
    return tuple(terms)
