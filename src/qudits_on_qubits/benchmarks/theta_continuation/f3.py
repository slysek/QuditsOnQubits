"""Exact F3 synthesis with a free leakage phase and an alpha-zero control."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    encoding_embedding,
    qutrit_fourier,
)
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import GateSynthesisError


@dataclass(frozen=True)
class F3Result:
    """Validated circuits and costs under the same exact transpilation protocol."""

    optimal: QuantumCircuit
    alpha_zero: QuantumCircuit
    alpha: float
    optimal_metrics: dict
    alpha_zero_metrics: dict
    synthesis_seconds: float


def _validate_options(tolerance, transpiler_seeds, optimization_level):
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be a finite positive real number.")
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, Integral)
        or not 0 <= optimization_level <= 3
    ):
        raise ValueError("optimization_level must be an integer from 0 through 3.")
    try:
        seeds = tuple(transpiler_seeds)
    except TypeError as exc:
        raise ValueError("transpiler_seeds must be a nonempty sequence of uint32 integers.") from exc
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, Integral)
        or not 0 <= seed <= 2**32 - 1
        for seed in seeds
    ):
        raise ValueError("transpiler_seeds must be a nonempty sequence of uint32 integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("transpiler_seeds must not contain duplicates.")
    return tuple(int(seed) for seed in seeds)


def _validated_metrics(circuit, embedding, target, tolerance, *, allowed_gates, max_2q=None):
    if (
        not isinstance(circuit, QuantumCircuit)
        or circuit.num_qubits != 2
        or circuit.num_clbits
        or circuit.num_parameters
        or not set(circuit.count_ops()) <= allowed_gates
    ):
        raise GateSynthesisError("F3 requires a bound two-qubit circuit in the requested basis.")
    try:
        actual = Operator(circuit).data
        validation = optimized_gates.validate_code_space_gate(circuit, embedding, qutrit_fourier())
    except (QiskitError, ValueError, TypeError) as exc:
        raise GateSynthesisError("F3 operator validation failed.") from exc
    phase = np.exp(1j * np.angle(np.vdot(actual, target)))
    metrics = {
        **asdict(validation),
        "N_2q": sum(item.operation.num_qubits == 2 for item in circuit.data),
        "full_error_norm": float(np.linalg.norm(phase * actual - target, ord="fro")),
    }
    if (
        not np.isfinite(list(metrics.values())).all()
        or metrics["E_norm"] > tolerance
        or metrics["L_norm"] > tolerance
        or metrics["full_error_norm"] > tolerance
        or (max_2q is not None and metrics["N_2q"] > max_2q)
    ):
        raise GateSynthesisError(f"F3 failed phase or code-space validation: {metrics}", metrics)
    return metrics


def _best_exact_circuit(source, embedding, target, tolerance, seeds, optimization_level,
                        *, alpha, synthesis_method, max_cz):
    candidates = []
    failures = []
    for seed in seeds:
        try:
            circuit = transpile(
                source, basis_gates=["u", "cz"], coupling_map=None,
                routing_method="none", approximation_degree=1.0,
                optimization_level=int(optimization_level), seed_transpiler=seed,
            )
            metrics = _validated_metrics(
                circuit, embedding, target, tolerance,
                allowed_gates={"u", "cz"}, max_2q=max_cz,
            )
            metrics.update({
                "N_CZ": circuit.count_ops().get("cz", 0),
                "N_1q": sum(item.operation.num_qubits == 1 for item in circuit.data),
                "depth": circuit.depth(), "seed": seed,
                "leakage_phase": alpha, "synthesis_method": synthesis_method,
            })
            circuit.metadata = {**(source.metadata or {}), **metrics}
            rank = tuple(metrics[key] for key in ("N_CZ", "depth", "N_1q", "seed"))
            candidates.append((rank, circuit, metrics))
        except (GateSynthesisError, QiskitError, ValueError, TypeError) as exc:
            failures.append({"seed": seed, "error": str(exc), "metrics": getattr(exc, "metrics", {})})
    if not candidates:
        raise GateSynthesisError("No exact U/CZ F3 candidate passed validation.", {"attempts": failures})
    _, circuit, metrics = min(candidates, key=lambda candidate: candidate[0])
    return circuit, metrics


def synthesize_theta_f3(
    encoding: np.ndarray,
    *,
    tolerance=1e-10,
    transpiler_seeds=(0, 1, 2),
    optimization_level=3,
) -> F3Result:
    """Synthesize F3 with at most two CZ gates and an alpha-zero comparison.

    The free phase comes from the existing two-CNOT synthesis. Each returned
    circuit is checked against its entire physical extension, up to one global
    phase, as well as its logical action and leakage. The ``optimal`` name
    identifies that phase choice; it is not a proof of global cost optimality.
    """
    seeds = _validate_options(tolerance, transpiler_seeds, optimization_level)
    encoding = np.asarray(encoding, dtype=complex)
    if not np.isfinite(encoding).all():
        raise ValueError("encoding must contain only finite values.")
    embedding = encoding_embedding(encoding)
    coded = embedding @ qutrit_fourier() @ embedding.conj().T
    complement = np.eye(4, dtype=complex) - embedding @ embedding.conj().T
    started = time.perf_counter()
    source = optimized_gates.synthesize_f3(embedding)
    if not isinstance(source, QuantumCircuit):
        raise GateSynthesisError("F3 synthesis did not return a quantum circuit.")
    metadata = source.metadata or {}
    alpha = metadata.get("leakage_phase")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha):
        raise GateSynthesisError("F3 synthesis returned an invalid leakage phase.")
    alpha = float(np.mod(alpha, 2 * np.pi))
    target = coded + np.exp(1j * alpha) * complement
    _validated_metrics(
        source, embedding, target, tolerance, allowed_gates={"u", "u3", "cx"}, max_2q=2,
    )
    optimal, optimal_metrics = _best_exact_circuit(
        source, embedding, target, tolerance, seeds, optimization_level,
        alpha=alpha, synthesis_method=metadata.get("synthesis_method", "two_cnot_invariant"), max_cz=2,
    )
    optimal.name = "F3_theta_optimal"
    zero_target = coded + complement
    zero_source = QuantumCircuit(2, name="F3_theta_alpha_zero")
    zero_source.unitary(zero_target, [0, 1])
    alpha_zero, alpha_zero_metrics = _best_exact_circuit(
        zero_source, embedding, zero_target, tolerance, seeds, optimization_level,
        alpha=0.0, synthesis_method="exact_alpha_zero", max_cz=None,
    )
    elapsed = time.perf_counter() - started
    if not np.isfinite(elapsed) or elapsed < 0:
        raise GateSynthesisError("F3 synthesis returned a nonfinite or negative duration.")
    return F3Result(optimal, alpha_zero, alpha, optimal_metrics, alpha_zero_metrics, elapsed)
