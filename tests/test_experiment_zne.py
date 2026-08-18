from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Parameter

from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.mitigation import (
    fold_cz_batch,
    linear_zne_extrapolate,
    validate_zne_factors,
)


def _operation_names(circuit: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in circuit.data]


def test_validate_zne_factors_normalizes_to_deterministic_order() -> None:
    assert validate_zne_factors([5, 1, 3]) == (1, 3, 5)


@pytest.mark.parametrize(
    "factors",
    [[], [3, 5], [1, 1, 3], [1, 0, 3], [1, -3], [1, 2, 3], [True, 1, 3]],
)
def test_validate_zne_factors_rejects_invalid_values(factors: list[object]) -> None:
    with pytest.raises(ExperimentValidationError):
        validate_zne_factors(factors)


@pytest.mark.parametrize("factor", [1, 3, 5])
def test_fold_cz_batch_repeats_every_cz_and_preserves_instruction_order(factor: int) -> None:
    theta = Parameter("theta")
    original = QuantumCircuit(3, 2, name="zne-source", metadata={"nested": {"safe": True}})
    original.global_phase = theta
    original.h(0)
    original.cz(0, 1)
    original.rx(theta, 2)
    original.cz(1, 2)
    original.measure(0, 0)
    original.measure(2, 1)
    before_names = _operation_names(original)

    [folded] = fold_cz_batch([original], factor)

    assert folded is not original
    assert _operation_names(folded) == (
        ["h"] + ["cz"] * factor + ["rx"] + ["cz"] * factor + ["measure", "measure"]
    )
    assert folded.name == original.name
    assert folded.global_phase == original.global_phase
    assert folded.metadata == original.metadata
    assert folded.metadata is not original.metadata
    assert folded.num_clbits == original.num_clbits
    assert _operation_names(original) == before_names


def test_fold_cz_batch_factor_one_is_exact_structural_copy_without_aliasing() -> None:
    original = QuantumCircuit(2, 1, name="factor-one", metadata={"items": [1]})
    original.cz(0, 1)
    original.measure(1, 0)

    [folded] = fold_cz_batch([original], 1)

    assert folded == original
    folded.metadata["items"].append(2)
    assert original.metadata == {"items": [1]}


def test_fold_cz_batch_preserves_transpiler_layout_when_present() -> None:
    source = QuantumCircuit(2)
    source.cz(0, 1)
    laid_out = transpile(source, basis_gates=["cz"], initial_layout=[1, 0])
    assert laid_out.layout is not None

    [folded] = fold_cz_batch([laid_out], 3)

    assert folded.layout == laid_out.layout


@pytest.mark.parametrize("factor", [0, -1, 2, True, 1.0])
def test_fold_cz_batch_rejects_invalid_factor(factor: object) -> None:
    with pytest.raises(ExperimentValidationError):
        fold_cz_batch([QuantumCircuit(2)], factor)  # type: ignore[arg-type]


def test_fold_cz_batch_rejects_custom_instruction_named_cz() -> None:
    circuit = QuantumCircuit(2)
    circuit.append(Gate("cz", 2, []), [0, 1])

    with pytest.raises(ExperimentValidationError, match="custom|calibrated"):
        fold_cz_batch([circuit], 3)


def test_fold_cz_batch_rejects_cz_inside_control_flow() -> None:
    circuit = QuantumCircuit(2, 1)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.cz(0, 1)

    with pytest.raises(ExperimentValidationError, match="control-flow"):
        fold_cz_batch([circuit], 3)


def test_linear_zne_extrapolate_recovers_known_real_line_and_sorts_pairs() -> None:
    estimate, fit = linear_zne_extrapolate([5, 1, 3], [9.0, 1.0, 5.0])

    assert estimate == pytest.approx(-1.0 + 0.0j)
    assert fit.factors == (1, 3, 5)
    assert fit.values == (1.0 + 0.0j, 5.0 + 0.0j, 9.0 + 0.0j)
    assert fit.real_coefficients == pytest.approx((2.0, -1.0))
    assert fit.imag_coefficients == pytest.approx((0.0, 0.0))
    assert fit.residuals == pytest.approx((0j, 0j, 0j))


def test_linear_zne_extrapolate_fits_complex_values_and_returns_immutable_metadata() -> None:
    factors = (1, 3, 5)
    values = tuple((2.0 + 3.0j) * factor + (-1.0 + 4.0j) for factor in factors)

    estimate, fit = linear_zne_extrapolate(factors, values)

    assert estimate == pytest.approx(-1.0 + 4.0j)
    assert fit.real_coefficients == pytest.approx((2.0, -1.0))
    assert fit.imag_coefficients == pytest.approx((3.0, 4.0))
    with pytest.raises(FrozenInstanceError):
        fit.factors = (1, 3)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factors", "values"),
    [
        ((1,), (1.0,)),
        ((1, 3), (1.0,)),
        ((1, 3), (1.0, np.nan)),
        ((1, 3), (1.0, complex(0.0, np.inf))),
    ],
)
def test_linear_zne_extrapolate_rejects_invalid_data(
    factors: tuple[int, ...], values: tuple[complex, ...]
) -> None:
    with pytest.raises(ExperimentValidationError):
        linear_zne_extrapolate(factors, values)
