"""Explicit logical operations and synthesis-independent qutrit references.

Qutrit 0 is the least significant base-3 digit in logical statevectors. Its
physical block is Qiskit qubits [0, 1]; qutrit i uses [2*i, 2*i+1]. Each block
has matrix row order |00>, |01>, |10>, |11>, with its lower qubit as the least
significant bit. Thus tensor products read E_(n-1) tensor ... tensor E_0.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz, qutrit_fourier
from qudits_on_qubits.benchmarks.models import validate_encoding


_ARITIES = {"f3": 1, "cz3": 2}


@dataclass(frozen=True)
class LogicalOperation:
    """An F3 or CZ3 operation; the first target is locally least significant."""

    name: str
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _ARITIES:
            raise ValueError("operation name must be 'f3' or 'cz3'")
        if not isinstance(self.targets, (tuple, list)):
            raise TypeError("targets must be a tuple or list of integer qutrit indices")
        if len(self.targets) != _ARITIES[self.name]:
            raise ValueError(f"{self.name} requires {_ARITIES[self.name]} target(s)")
        for target in self.targets:
            if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
                raise ValueError("targets must be nonnegative integer qutrit indices")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("cz3 requires distinct targets")
        object.__setattr__(self, "targets", tuple(int(target) for target in self.targets))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "targets": list(self.targets)}


@dataclass(frozen=True)
class LogicalCircuit:
    """An ordered circuit whose identity does not select an execution algorithm.

    There is no artificial limit on register size. Exact reference generation
    requires O(3**num_qutrits) memory; its encoded state requires O(4**num_qutrits).
    """

    num_qutrits: int
    operations: tuple[LogicalOperation, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        if isinstance(self.num_qutrits, bool) or not isinstance(self.num_qutrits, Integral):
            raise TypeError("num_qutrits must be a positive integer")
        if self.num_qutrits < 1:
            raise ValueError("num_qutrits must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.operations, (tuple, list)):
            raise TypeError("operations must be a tuple or list of LogicalOperation objects")
        for operation in self.operations:
            if not isinstance(operation, LogicalOperation):
                raise TypeError("operations must contain LogicalOperation objects")
            if any(target >= self.num_qutrits for target in operation.targets):
                raise ValueError("operation target is outside the qutrit register")
        object.__setattr__(self, "num_qutrits", int(self.num_qutrits))
        object.__setattr__(self, "operations", tuple(self.operations))

    def reference_state(self) -> np.ndarray:
        """Evolve logical |0...0> with NumPy, without any synthesized qubit gate."""
        state = np.zeros(3**self.num_qutrits, dtype=complex)
        state[0] = 1
        gates = {"f3": qutrit_fourier(), "cz3": qutrit_cz()}
        for operation in self.operations:
            # A matrix's leading tensor axis is the most significant target.
            axes = [self.num_qutrits - 1 - target for target in reversed(operation.targets)]
            axes += [axis for axis in range(self.num_qutrits) if axis not in axes]
            tensor = state.reshape((3,) * self.num_qutrits).transpose(axes)
            evolved = gates[operation.name] @ tensor.reshape(3 ** len(operation.targets), -1)
            state = evolved.reshape((3,) * self.num_qutrits).transpose(np.argsort(axes)).reshape(-1)
        return state

    def encoded_reference(self, encoding: np.ndarray) -> np.ndarray:
        """Apply E to every tensor axis without allocating the full embedding matrix."""
        embedding = validate_encoding(encoding)
        tensor = self.reference_state().reshape((3,) * self.num_qutrits)
        for axis in range(self.num_qutrits):
            tensor = np.tensordot(embedding, tensor, axes=(1, axis))
            tensor = np.moveaxis(tensor, 0, axis)
        return tensor.reshape(-1)

    def build(self, encoding: np.ndarray, gate_library: Any) -> QuantumCircuit:
        """Prepare encoded zero on every block, then compose supplied gate circuits."""
        embedding = validate_encoding(encoding)
        circuit = QuantumCircuit(2 * self.num_qutrits, name=self.name)
        zero = StatePreparation(embedding[:, 0], label="encoded_zero")
        for index in range(self.num_qutrits):
            circuit.append(zero, [2 * index, 2 * index + 1])
        for operation in self.operations:
            block = getattr(gate_library, operation.name, None)
            if not isinstance(block, QuantumCircuit) or block.num_qubits != 2 * len(operation.targets):
                raise ValueError(f"gate_library.{operation.name} must be a {2 * len(operation.targets)}-qubit circuit")
            qubits = [qubit for target in operation.targets for qubit in (2 * target, 2 * target + 1)]
            circuit.compose(block, qubits=qubits, inplace=True)
        return circuit

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qutrits": self.num_qutrits,
            "operations": [operation.to_dict() for operation in self.operations],
            "ordering": "qutrit_0_least_significant",
        }

    def stable_hash(self) -> str:
        """Hash the complete ordered configuration in a canonical JSON format."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def two_qutrit_graph_circuit() -> LogicalCircuit:
    """Prepare the maximally entangled two-qutrit graph state CZ3 (F3 tensor F3)|00>."""
    return LogicalCircuit(2, (
        LogicalOperation("f3", (0,)),
        LogicalOperation("f3", (1,)),
        LogicalOperation("cz3", (0, 1)),
    ), name="two_qutrit_graph")
