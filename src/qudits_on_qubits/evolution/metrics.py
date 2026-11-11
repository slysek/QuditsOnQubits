"""Metrics of an encoded gate and its physical circuit."""

import numpy as np
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.validation import circuit_metrics


def code_space_metrics(
    encoding: np.ndarray, target: np.ndarray, actual: np.ndarray,
) -> dict[str, float]:
    """Compare the final columns C B with B G, aligning one phase for all inputs."""
    dimension = encoding.shape[1]
    ideal = encoding @ target
    overlap = np.vdot(ideal, actual)
    phase = np.exp(1j * np.angle(overlap))
    # Apply I - B B† to the final output without allocating a physical projector.
    leaked = actual - encoding @ (encoding.conj().T @ actual)
    error_norm = float(np.linalg.norm(actual - phase * ideal, "fro"))
    return {
        "F": float(min(1.0, (abs(overlap) / dimension) ** 2)),
        "L": float(min(1.0, np.linalg.norm(leaked, "fro") ** 2 / dimension)),
        "eps": float(error_norm / np.sqrt(dimension)),
        "error_norm": error_norm,
    }


def gate_costs(circuit: QuantumCircuit) -> dict[str, int]:
    """Count elementary gates and their parallel layers on a validated circuit."""
    metrics = circuit_metrics(circuit)
    counts = circuit.count_ops()
    return {
        "cz_count": counts.get("cz", 0),
        "cx_count": counts.get("cx", 0),
        "two_qubit_gate_count": metrics["two_qubit_gate_count"],
        "depth": metrics["depth"],
        "two_qubit_depth": metrics["two_qubit_depth"],
        "one_qubit_gate_count": metrics["one_qubit_gate_count"],
    }
