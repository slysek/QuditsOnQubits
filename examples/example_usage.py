from __future__ import annotations

import os
import sys
from pprint import pprint

import numpy as np
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from qutrit_bell_measurements import (
    append_measurement_for_global_setting,
    canonical_Ez,
    omega,
)


def qutrit_X_Z() -> tuple[np.ndarray, np.ndarray]:
    w = omega(3)
    X = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        X[(j + 1) % 3, j] = 1.0
    Z = np.diag([w**j for j in range(3)]).astype(complex)
    return X, Z


X, Z = qutrit_X_Z()


def observable_from_label(label: str) -> np.ndarray:
    family = label[0]
    idx = int(label[1:])
    if family == "A":
        return Z if idx == 0 else Z @ np.linalg.matrix_power(X, idx)
    if family == "B":
        return X if idx == 1 else Z @ np.linalg.matrix_power(X, idx)
    raise ValueError(f"unknown observable label {label!r}")


state_circuit = QuantumCircuit(4)
state_circuit.h(0)
state_circuit.cx(0, 2)

qc, metadata = append_measurement_for_global_setting(
    state_circuit=state_circuit,
    global_setting=("A0", "B1"),
    qutrit_qubits=[(0, 1), (2, 3)],
    E=canonical_Ez(),
    observable_from_label=observable_from_label,
    d=3,
)

print(qc)
pprint(metadata)
