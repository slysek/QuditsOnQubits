"""Exact, reproducible CZ3 compilation with independent physical validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate, U3Gate, UGate
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding, qutrit_cz
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import GateSynthesisError, validate_code_space_gate
from .f3 import _validate_options


@dataclass(frozen=True)
class CompilationResult:
    circuit: QuantumCircuit
    metrics: dict
    trials: list[dict]


def validate_cz3_candidate(circuit, encoding, *, tolerance=1e-5, source=None, compiled=False):
    """Check elementary operations, code action and exact source preservation."""
    _validate_options(tolerance, (0,), 0)
    embedding = encoding_embedding(np.asarray(encoding, dtype=complex))
    if not np.isfinite(embedding).all():
        raise GateSynthesisError("CZ3 encoding must be finite.")
    allowed = {UGate, CZGate} if compiled else {UGate, U3Gate, CZGate}
    if (not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != 4
            or circuit.num_clbits or circuit.num_parameters):
        raise GateSynthesisError("CZ3 requires a bound four-qubit circuit without classical bits.")
    if any(item.operation.base_class not in allowed or item.clbits for item in circuit.data):
        raise GateSynthesisError("CZ3 requires elementary U/CZ operations.")
    try:
        if not np.isfinite(float(circuit.global_phase)) or any(
            not np.isfinite(float(value)) for item in circuit.data for value in item.operation.params
        ):
            raise GateSynthesisError("CZ3 parameters must be finite.")
        actual = Operator(circuit).data
        validation = validate_code_space_gate(circuit, np.kron(embedding, embedding), qutrit_cz())
    except (TypeError, OverflowError) as exc:
        raise GateSynthesisError("CZ3 circuit has invalid numerical parameters.") from exc
    metrics = {**asdict(validation), "N_CZ": int(circuit.count_ops().get("cz", 0)),
               "N_1q": sum(item.operation.num_qubits == 1 for item in circuit.data),
               "depth": int(circuit.depth())}
    if not np.isfinite(actual).all() or not np.isfinite(list(metrics.values())).all():
        raise GateSynthesisError("CZ3 produced nonfinite validation metrics.")
    if max(validation.E_norm, validation.L_norm) > tolerance:
        raise GateSynthesisError("CZ3 failed code-space validation.", metrics)
    if source is not None:
        target = Operator(source).data
        overlap = np.vdot(actual, target)
        phase = np.exp(1j * np.angle(overlap)) if abs(overlap) > 1e-15 else 1.0
        metrics["full_error_norm"] = float(np.linalg.norm(phase * actual - target, ord="fro"))
        if not np.isfinite(metrics["full_error_norm"]) or metrics["full_error_norm"] > 1e-10:
            raise GateSynthesisError("CZ3 compilation changed the full operator.", metrics)
    return metrics


def _exact_u_cz_copy(circuit):
    """Relabel U3 as U without numerical resynthesis or dropping small rotations."""
    compiled = circuit.copy_empty_like()
    compiled.global_phase = circuit.global_phase
    for item in circuit.data:
        operation = item.operation
        if operation.base_class is U3Gate:
            operation = UGate(*operation.params, label=operation.label)
        qubits = [compiled.qubits[circuit.find_bit(bit).index] for bit in item.qubits]
        compiled.append(operation, qubits)
    return compiled


def compile_cz3_candidate(circuit, encoding, *, tolerance=1e-5, seeds=(0, 1, 2), optimization_level=3):
    """Rank validated transpiler results, then use exact relabeling if all fail."""
    seeds = _validate_options(tolerance, seeds, optimization_level)
    validate_cz3_candidate(circuit, encoding, tolerance=tolerance)
    trials, candidates = [], []
    for seed in seeds:
        try:
            compiled = transpile(
                circuit.copy(), basis_gates=["u", "cz"], coupling_map=None,
                routing_method="none", approximation_degree=1.0,
                optimization_level=int(optimization_level), seed_transpiler=seed,
            )
            metrics = validate_cz3_candidate(compiled, encoding, tolerance=tolerance, source=circuit, compiled=True)
            metrics.update(seed=seed, compilation_method="transpile")
            compiled.metadata = {**(circuit.metadata or {}), **metrics}
            trials.append({"seed": seed, "compilation_method": "transpile", "valid": True, "metrics": metrics})
            rank = tuple(metrics[key] for key in ("N_CZ", "depth", "N_1q", "seed"))
            candidates.append((rank, compiled, metrics))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            trial = {"seed": seed, "compilation_method": "transpile", "valid": False, "error": str(exc)}
            if isinstance(exc, GateSynthesisError):
                trial["metrics"] = exc.metrics
            trials.append(trial)
    if not candidates:
        # Qiskit may discard rotations below its internal synthesis tolerance,
        # even with approximation_degree=1.0. This exact conversion has no such
        # approximation and must pass the same independent physical checks.
        seed = min(seeds)
        method = "exact_u3_to_u"
        try:
            compiled = _exact_u_cz_copy(circuit)
            metrics = validate_cz3_candidate(compiled, encoding, tolerance=tolerance, source=circuit, compiled=True)
            metrics.update(seed=seed, compilation_method=method)
            compiled.metadata = {**(circuit.metadata or {}), **metrics}
            trials.append({"seed": seed, "compilation_method": method, "valid": True, "metrics": metrics})
            return CompilationResult(compiled, metrics, trials)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            trial = {"seed": seed, "compilation_method": method, "valid": False, "error": str(exc)}
            if isinstance(exc, GateSynthesisError):
                trial["metrics"] = exc.metrics
            trials.append(trial)
            raise GateSynthesisError("No exact U/CZ CZ3 candidate passed validation.", {"trials": trials}) from exc
    _, best, metrics = min(candidates, key=lambda item: item[0])
    return CompilationResult(best, metrics, trials)
