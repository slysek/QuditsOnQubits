"""Parameterize the existing CZ3 local gates without changing its topology."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import CZGate, U3Gate, UGate

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import validate_code_space_gate


PINNED_CZ3_SHA256 = "a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791"
_ANGLE_NAMES = ("theta", "phi", "lambda")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, ParameterExpression):
        raise ValueError(f"{name} must not be symbolic.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number.")
    return converted


def _operations(circuit: QuantumCircuit) -> list[dict]:
    return [
        {
            "instruction_index": index,
            "name": instruction.operation.name,
            "qubits": [circuit.find_bit(qubit).index for qubit in instruction.qubits],
            "parameters": [float(value) for value in instruction.operation.params],
        }
        for index, instruction in enumerate(circuit.data)
    ]


@dataclass(frozen=True)
class Cz3Template:
    """A copied CZ3 topology with an explicit instruction-ordered angle vector.

    Each U/U3 contributes theta, phi, lambda in that order. The original
    instruction index and physical qubit are recorded for every angle. Binding
    uses the stored Parameter objects, independent of Qiskit's name sorting.
    """

    circuit: QuantumCircuit
    initial_parameters: np.ndarray
    template_id: str
    parameter_metadata: tuple[dict, ...]
    original_circuit: QuantumCircuit
    baseline_sha256: str
    _parameters: tuple[Parameter, ...] = field(repr=False)

    def bind(self, parameters: object) -> QuantumCircuit:
        """Return an independent circuit bound to one finite real angle vector."""
        try:
            values = np.asarray(parameters, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be a one-dimensional real vector.") from exc
        expected_shape = (len(self._parameters),)
        if values.shape != expected_shape:
            raise ValueError(f"parameters must have shape {expected_shape}, got {values.shape}.")
        angles = [_finite_real(value, "parameters") for value in values]
        return self.circuit.assign_parameters(dict(zip(self._parameters, angles)), inplace=False)

    def to_dict(self) -> dict:
        """Describe the source operations and parameter layout as JSON-safe data."""
        return {
            "template_id": self.template_id,
            "baseline_sha256": self.baseline_sha256,
            "num_qubits": self.circuit.num_qubits,
            "global_phase": float(self.original_circuit.global_phase),
            "initial_parameters": self.initial_parameters.tolist(),
            "parameter_metadata": copy.deepcopy(list(self.parameter_metadata)),
            "operations": _operations(self.original_circuit),
        }


def parameterize_cz3_circuit(
    circuit: QuantumCircuit,
    *,
    baseline_sha256: str = "",
) -> Cz3Template:
    """Replace all existing U/U3 angles, retaining gate classes, wires and phase.

    This constructor supports synthetic circuits and does not assert logical
    CZ3 correctness. Use :func:`load_cz3_template` for a validated baseline.
    Only standard U, U3 and CZ gates are accepted; nothing is transpiled.
    """
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != 4:
        raise ValueError("CZ3 source must be a QuantumCircuit with exactly four qubits.")
    if circuit.num_clbits:
        raise ValueError("CZ3 source must not contain classical bits.")
    phase = _finite_real(circuit.global_phase, "global_phase")
    original = circuit.copy()
    parameterized = circuit.copy()
    angles: list[float] = []
    parameters: list[Parameter] = []
    metadata: list[dict] = []

    for instruction_index, instruction in enumerate(original.data):
        operation = instruction.operation
        if (
            operation.base_class not in {UGate, U3Gate, CZGate}
            or operation.name not in {"u", "u3", "cz"}
            or instruction.clbits
        ):
            raise ValueError(f"Unsupported CZ3 instruction {operation.name!r} at index {instruction_index}.")
        if operation.base_class is CZGate:
            continue
        replacement = operation.copy()
        local_parameters = []
        for angle_name, value in zip(_ANGLE_NAMES, operation.params, strict=True):
            angle = _finite_real(value, f"instruction {instruction_index} {angle_name}")
            index = len(parameters)
            parameter = Parameter(f"cz3_angle_{index}")
            parameters.append(parameter)
            local_parameters.append(parameter)
            angles.append(angle)
            metadata.append({
                "index": index,
                "instruction_index": instruction_index,
                "qubit": original.find_bit(instruction.qubits[0]).index,
                "angle": angle_name,
            })
        replacement.params = local_parameters
        parameterized.data[instruction_index] = instruction.replace(operation=replacement)

    specification = {
        "format_version": 1,
        "num_qubits": original.num_qubits,
        "global_phase": phase,
        "operations": _operations(original),
        "parameter_metadata": metadata,
        "baseline_sha256": baseline_sha256,
    }
    template_id = hashlib.sha256(json.dumps(
        specification, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    initial = np.asarray(angles, dtype=float)
    initial.setflags(write=False)
    return Cz3Template(
        circuit=parameterized,
        initial_parameters=initial,
        template_id=template_id,
        parameter_metadata=tuple(metadata),
        original_circuit=original,
        baseline_sha256=baseline_sha256,
        _parameters=tuple(parameters),
    )


def load_cz3_template(
    baseline_qpy: str | Path,
    *,
    expected_sha256: str | None = PINNED_CZ3_SHA256,
    tolerance: float = 1e-5,
) -> Cz3Template:
    """Verify the QPY, sibling canonical E.npy and code-space action before use.

    The default SHA256 pins the canonical optimized baseline. Passing ``None``
    permits another source but retains all structural and numerical validation.
    The hash is computed from the same bytes subsequently passed to QPY.
    """
    tolerance = _finite_real(tolerance, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance must be a finite positive real number.")
    path = Path(baseline_qpy)
    source_bytes = path.read_bytes()
    baseline_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if expected_sha256 is not None and baseline_sha256 != expected_sha256:
        raise ValueError(f"CZ3 baseline SHA256 mismatch: expected {expected_sha256}, got {baseline_sha256}.")
    encoding = np.load(path.with_name("E.npy"), allow_pickle=False)
    if not np.array_equal(encoding, np.eye(4, 3)):
        raise ValueError("Baseline E.npy must contain the canonical encoding np.eye(4, 3).")
    circuits = qpy.load(io.BytesIO(source_bytes))
    if len(circuits) != 1:
        raise ValueError("Expected exactly one CZ3 circuit in the baseline QPY file.")
    template = parameterize_cz3_circuit(circuits[0], baseline_sha256=baseline_sha256)
    validation = validate_code_space_gate(
        template.original_circuit, np.kron(encoding, encoding), qutrit_cz(),
    )
    errors = np.array([validation.E_norm, validation.L_norm])
    if not np.isfinite(errors).all() or np.any(errors > tolerance):
        raise ValueError(
            f"CZ3 baseline failed code-space validation: E_norm={validation.E_norm}, "
            f"L_norm={validation.L_norm}, tolerance={tolerance}."
        )
    return template
