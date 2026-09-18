"""Validated persisted theta gates for the existing direct-basis benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding, qutrit_fourier
from .compilation import validate_cz3_candidate
from .f3 import _validated_metrics


def validate_saved_f3(circuit, encoding, alpha, *, tolerance=1e-10):
    """Recheck the entire leakage-phase extension instead of trusting metadata."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha):
        raise ValueError("F3 leakage phase must be finite.")
    embedding = encoding_embedding(np.asarray(encoding, dtype=complex))
    target = embedding @ qutrit_fourier() @ embedding.conj().T
    target += np.exp(1j * alpha) * (np.eye(4) - embedding @ embedding.conj().T)
    metrics = _validated_metrics(circuit, embedding, target, tolerance, allowed_gates={"u", "cz"})
    metrics.update({"N_CZ": int(circuit.count_ops().get("cz", 0)), "depth": int(circuit.depth()),
                    "N_1q": sum(item.operation.num_qubits == 1 for item in circuit.data)})
    return metrics


@dataclass(frozen=True)
class ThetaGateLibrary:
    """Gate adapter with no synthesis or disk-cache lookup path."""

    f3: QuantumCircuit
    cz3: QuantumCircuit
    synthesis_seconds: float
    selected_method: str

    @classmethod
    def from_bundle(cls, bundle, encoding, *, f3_tolerance=1e-10, cz3_tolerance=1e-5):
        metadata = bundle["metadata"]
        if metadata.get("correct") is not True:
            raise ValueError("Only a correct gate point can supply full circuits.")
        f3 = bundle["circuits"]["f3_optimal.qpy"].copy()
        cz3 = bundle["circuits"]["cz3_selected.qpy"].copy()
        f3_metrics = validate_saved_f3(f3, encoding, metadata["f3_alpha"], tolerance=f3_tolerance)
        if f3_metrics["N_CZ"] > 2:
            raise ValueError("F3 optimal circuit exceeds the two-CZ limit.")
        cz3_metrics = validate_cz3_candidate(cz3, encoding, tolerance=cz3_tolerance, compiled=True)
        f3.metadata = {**metadata["f3_metrics"], **f3_metrics, "leakage_phase": metadata["f3_alpha"]}
        cz3.metadata = cz3_metrics
        elapsed = sum(metadata.get(key, 0.0) for key in (
            "f3_synthesis_seconds", "continuation_seconds", "fallback_seconds",
        ))
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Gate synthesis duration must be finite and nonnegative.")
        method = metadata["selected_method"]
        if method not in {"baseline", "continuation", "bqskit"}:
            raise ValueError("Unknown CZ3 synthesis method.")
        return cls(f3, cz3, float(elapsed), method)

    def benchmark_metrics(self):
        return {
            "gate_library": "theta_continuation_v1",
            **{key: self.cz3.metadata[key] for key in ("E_norm", "L_norm", "N_2q")},
            **{f"f3_{key}": self.f3.metadata[key] for key in ("E_norm", "L_norm", "N_2q")},
            "f3_leakage_phase": self.f3.metadata["leakage_phase"],
            "f3_synthesis_method": self.f3.metadata["synthesis_method"],
            "cz3_synthesis_method": self.selected_method,
            "cz3_synthesis_seed": optimized_gates.SYNTHESIS_SEED if self.selected_method == "bqskit" else None,
            "cz3_synthesis_epsilon": optimized_gates.SYNTHESIS_EPSILON if self.selected_method == "bqskit" else None,
            "gate_synthesis_seconds": self.synthesis_seconds,
            "gate_cache_hit": True,
        }
