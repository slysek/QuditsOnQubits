from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .basis import (
    canonical_Ez,
    measurement_physical_index_from_bits,
    omega,
    physical_index_from_bits,
    physical_to_logical_outcome_map,
)


@dataclass(frozen=True)
class InvalidCodewordStatistics:
    total_weight: float
    accepted_weight: float
    invalid_weight: float
    invalid_fraction: float


@dataclass(frozen=True)
class ReferenceBellEvaluation:
    unconditional: complex
    conditional: complex
    leakage_rate: float
    total_shots: int
    accepted_shots: int


def bit_pair_to_qutrit_outcome(
    bit0: int,
    bit1: int,
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
    d: int = 3,
) -> int | None:
    """Map a two-qubit computational outcome to a qutrit outcome or leakage."""
    if bit0 not in (0, 1) or bit1 not in (0, 1):
        raise ValueError("bits must be 0 or 1")
    mapping = _resolve_outcome_map(E=E, outcome_map=outcome_map, d=d)
    index_fn = (
        measurement_physical_index_from_bits
        if outcome_map is not None
        else physical_index_from_bits
    )
    return mapping[index_fn(bit0, bit1)]


def bitstring_to_qutrit_outcomes(
    bitstring: str,
    qutrit_bit_indices: Sequence[tuple[int, int]],
    bit_order: str = "qiskit",
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
    d: int = 3,
) -> tuple[int | None, ...]:
    """Decode a count bitstring into qutrit outcomes.

    Warning: Qiskit count keys are displayed with classical bit 0 at the
    rightmost character. With ``bit_order="qiskit"``, index 0 therefore reads
    from the right. Use ``bit_order="left-to-right"`` when indices should refer
    directly to string positions from the left.
    """
    decode_kwargs = {"E": E, "outcome_map": outcome_map, "d": d}
    outcomes: list[int | None] = []
    for bit0_index, bit1_index in qutrit_bit_indices:
        bit0 = _bit_at(bitstring, bit0_index, bit_order)
        bit1 = _bit_at(bitstring, bit1_index, bit_order)
        outcomes.append(bit_pair_to_qutrit_outcome(bit0, bit1, **decode_kwargs))
    return tuple(outcomes)


def invalid_codeword_statistics(
    counts_by_setting: Mapping[tuple, Mapping[str, float]],
    qutrit_bit_indices_by_setting: Mapping[tuple, Sequence[tuple[int, int]]],
    bit_order: str = "qiskit",
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
    d: int = 3,
    require_non_negative: bool,
) -> InvalidCodewordStatistics:
    """Summarize weights whose bitstrings contain an invalid qutrit codeword."""
    if not isinstance(require_non_negative, bool):
        raise ValueError("require_non_negative must be a bool")

    decode_kwargs = {"E": E, "outcome_map": outcome_map, "d": d}
    total_weight = 0.0
    invalid_weight = 0.0
    for setting, counts in counts_by_setting.items():
        qutrit_bit_indices = qutrit_bit_indices_by_setting[setting]
        for bitstring, weight in counts.items():
            _validate_weight(weight)
            if require_non_negative and weight < 0:
                raise ValueError("weights must be non-negative")

            total_weight += weight
            outcomes = bitstring_to_qutrit_outcomes(
                bitstring,
                qutrit_bit_indices,
                bit_order,
                **decode_kwargs,
            )
            if any(outcome is None for outcome in outcomes):
                invalid_weight += weight

    accepted_weight = total_weight - invalid_weight
    invalid_fraction = (
        0.0 if total_weight == 0 else invalid_weight / total_weight
    )
    return InvalidCodewordStatistics(
        total_weight=total_weight,
        accepted_weight=accepted_weight,
        invalid_weight=invalid_weight,
        invalid_fraction=invalid_fraction,
    )


def leakage_rate(
    counts: Mapping[str, int],
    qutrit_bit_indices: Sequence[tuple[int, int]],
    bit_order: str = "qiskit",
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
    d: int = 3,
) -> float:
    """Return the fraction of shots with at least one decoded leakage outcome."""
    decode_kwargs = {"E": E, "outcome_map": outcome_map, "d": d}
    total = 0
    leakage = 0
    for bitstring, count in counts.items():
        _validate_count(count)
        total += count
        outcomes = bitstring_to_qutrit_outcomes(
            bitstring,
            qutrit_bit_indices,
            bit_order,
            **decode_kwargs,
        )
        if any(outcome is None for outcome in outcomes):
            leakage += count
    if total == 0:
        return 0.0
    return leakage / total


def compute_complex_expectation(
    counts: Mapping[str, float],
    powers: Sequence[int],
    qutrit_bit_indices: Sequence[tuple[int, int]],
    d: int = 3,
    bit_order: str = "qiskit",
    discard_leakage: bool = True,
    renormalize_after_discard: bool = True,
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
) -> complex:
    """Compute a complex qutrit correlator from backend/Sampler counts.

    Leakage outcomes contribute no phase. If ``discard_leakage`` and
    ``renormalize_after_discard`` are both true, the result is postselected on
    non-leakage shots. If renormalization is disabled, leakage shots remain in
    the denominator and reduce the correlator norm.
    """
    if len(powers) != len(qutrit_bit_indices):
        raise ValueError("powers and qutrit_bit_indices must have the same length")

    decode_kwargs = {"E": E, "outcome_map": outcome_map, "d": d}
    root = omega(d)
    total_shots = 0
    accepted_shots = 0
    accumulator = 0.0 + 0.0j

    for bitstring, count in counts.items():
        _validate_weight(count)
        total_shots += count
        outcomes = bitstring_to_qutrit_outcomes(
            bitstring,
            qutrit_bit_indices,
            bit_order,
            **decode_kwargs,
        )
        if any(outcome is None for outcome in outcomes):
            continue

        accepted_shots += count
        exponent = sum(
            int(power) * int(outcome)
            for power, outcome in zip(powers, outcomes)
        )
        accumulator += count * (root ** (exponent % d))

    if discard_leakage and renormalize_after_discard:
        denominator = accepted_shots
    else:
        denominator = total_shots

    if denominator == 0:
        return 0.0 + 0.0j
    return complex(accumulator / denominator)


def compute_bell_value_from_counts(
    counts_by_setting: Mapping[tuple, Mapping[str, float]],
    terms: Sequence[Mapping[str, object]],
    qutrit_bit_indices_by_setting: Mapping[tuple, Sequence[tuple[int, int]]],
    d: int = 3,
    bit_order: str = "qiskit",
    discard_leakage: bool = True,
    renormalize_after_discard: bool = True,
    *,
    E: np.ndarray | None = None,
    outcome_map: dict[int, int | None] | None = None,
) -> complex:
    """Evaluate a Bell expression from per-setting backend/Sampler counts."""
    decode_kwargs = {"E": E, "outcome_map": outcome_map, "d": d}
    value = 0.0 + 0.0j
    for term in terms:
        settings = tuple(term["settings"])  # type: ignore[index]
        counts = counts_by_setting[settings]
        qutrit_bit_indices = qutrit_bit_indices_by_setting[settings]
        expectation = compute_complex_expectation(
            counts,
            powers=tuple(term["powers"]),  # type: ignore[index,arg-type]
            qutrit_bit_indices=qutrit_bit_indices,
            bit_order=bit_order,
            discard_leakage=discard_leakage,
            renormalize_after_discard=renormalize_after_discard,
            **decode_kwargs,
        )
        value += complex(term["coeff"]) * expectation  # type: ignore[index]
    return complex(value)


def evaluate_reference_bell_values_from_counts(
    candidate: str,
    counts_by_setting: Mapping[tuple, Mapping[str, int]],
    qutrit_bit_indices_by_setting: Mapping[tuple, Sequence[tuple[int, int]]],
    bit_order: str = "qiskit",
) -> ReferenceBellEvaluation:
    """Evaluate registry Bell values with and without leakage postselection."""
    spec = get_reference_experiment(candidate)
    outcome_map = dict(spec.outcome_convention.measurement_basis_index_map)
    terms = [
        {
            "coeff": term.sampling_coefficient(),
            "settings": spec.setting_for_term(term),
            "powers": spec.powers_for_term(term),
        }
        for term in spec.bell_functional.terms
    ]

    unconditional = compute_bell_value_from_counts(
        counts_by_setting,
        terms,
        qutrit_bit_indices_by_setting,
        d=3,
        bit_order=bit_order,
        discard_leakage=False,
        renormalize_after_discard=False,
        outcome_map=outcome_map,
    )
    conditional = compute_bell_value_from_counts(
        counts_by_setting,
        terms,
        qutrit_bit_indices_by_setting,
        d=3,
        bit_order=bit_order,
        discard_leakage=True,
        renormalize_after_discard=True,
        outcome_map=outcome_map,
    )

    total_shots = 0
    leakage_shots = 0
    for setting in spec.measurement_settings():
        counts = counts_by_setting[setting]
        qutrit_bit_indices = qutrit_bit_indices_by_setting[setting]
        for bitstring, count in counts.items():
            _validate_count(count)
            total_shots += count
            outcomes = bitstring_to_qutrit_outcomes(
                bitstring,
                qutrit_bit_indices,
                bit_order,
                outcome_map=outcome_map,
                d=3,
            )
            if any(outcome is None for outcome in outcomes):
                leakage_shots += count

    accepted_shots = total_shots - leakage_shots
    if total_shots == 0:
        leakage_rate_value = 0.0
    else:
        leakage_rate_value = leakage_shots / total_shots
    return ReferenceBellEvaluation(
        unconditional=unconditional,
        conditional=conditional,
        leakage_rate=leakage_rate_value,
        total_shots=total_shots,
        accepted_shots=accepted_shots,
    )


def _resolve_outcome_map(
    *,
    E: np.ndarray | None,
    outcome_map: dict[int, int | None] | None,
    d: int,
) -> dict[int, int | None]:
    if outcome_map is not None:
        return outcome_map
    encoding = canonical_Ez(d) if E is None else np.asarray(E, dtype=complex)
    return physical_to_logical_outcome_map(encoding, d=d)


def _bit_at(bitstring: str, index: int, bit_order: str) -> int:
    normalized = bitstring.replace(" ", "")
    if index < 0 or index >= len(normalized):
        raise ValueError(f"bit index {index} is out of range for {bitstring!r}")

    if bit_order in {"qiskit", "little-endian", "right-to-left"}:
        position = len(normalized) - 1 - index
    elif bit_order in {"left-to-right", "big-endian", "msb"}:
        position = index
    else:
        raise ValueError(
            "bit_order must be 'qiskit', 'little-endian', 'right-to-left', "
            "'left-to-right', 'big-endian', or 'msb'"
        )

    bit = normalized[position]
    if bit not in {"0", "1"}:
        raise ValueError(f"bitstring contains non-binary character {bit!r}")
    return int(bit)


def _validate_weight(weight: int | float) -> None:
    if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(weight):
        raise ValueError("weights must be finite real values")


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("counts must be non-negative integers")
