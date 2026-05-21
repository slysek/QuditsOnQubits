from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .basis import (
    canonical_Ez,
    measurement_physical_index_from_bits,
    omega,
    physical_index_from_bits,
    physical_to_logical_outcome_map,
)


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
    counts: Mapping[str, int],
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
        _validate_count(count)
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
    counts_by_setting: Mapping[tuple, Mapping[str, int]],
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


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or count < 0:
        raise ValueError("counts must be non-negative integers")
