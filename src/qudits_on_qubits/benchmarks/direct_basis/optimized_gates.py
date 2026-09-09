"""Validated code-space synthesis of the default benchmark gate library.

F3 leaves one phase free; CZ3 leaves the entire seven-dimensional complement
free, exactly as in notebooks/CZ3_bqckit_optimalization.ipynb. Stored circuits
contain elementary gates, never opaque four-qubit unitaries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock
from qiskit import QuantumCircuit, qpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from scipy.optimize import brentq

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    encoding_embedding,
    optimal_f3_leakage_phase,
    qutrit_cz,
    qutrit_fourier,
)
from qudits_on_qubits.core.project_paths import repo_path

F3_TOLERANCE = 1e-10
CZ3_TOLERANCE = 1e-5
SYNTHESIS_EPSILON = 1e-8
SYNTHESIS_SEED = 0
GATE_LIBRARY = "code_space_optimized_v1"
_SYNTHESIS_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateValidation:
    E_norm: float
    L_norm: float
    N_2q: int
    raw_error_norm: float


class GateSynthesisError(ValueError):
    """Synthesis failed an acceptance test; never use a legacy fallback."""

    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


def validate_code_space_gate(
    circuit: QuantumCircuit, encoding: np.ndarray, logical_gate: np.ndarray,
) -> GateValidation:
    """Unnormalised Frobenius norms, using ONE global phase for all columns."""
    actual = Operator(circuit).data @ encoding
    target = encoding @ logical_gate
    overlap = np.vdot(actual, target)
    phase = np.exp(1j * np.angle(overlap)) if abs(overlap) > 1e-15 else 1.0
    projector = encoding @ encoding.conj().T
    return GateValidation(
        E_norm=float(np.linalg.norm(phase * actual - target, ord="fro")),
        L_norm=float(np.linalg.norm((np.eye(len(projector)) - projector) @ actual, ord="fro")),
        N_2q=sum(item.operation.num_qubits == 2 for item in circuit.data),
        raw_error_norm=float(np.linalg.norm(actual - target, ord="fro")),
    )


def _accept(
    circuit: QuantumCircuit,
    validation: GateValidation,
    *,
    name: str,
    tolerance: float,
    max_2q: int | None = None,
) -> None:
    allowed = {"u", "u3", "cx"} if name == "F3" else {"u", "u3", "cz"}
    valid = (
        set(circuit.count_ops()) <= allowed
        and np.isfinite([validation.E_norm, validation.L_norm]).all()
        and validation.E_norm <= tolerance
        and validation.L_norm <= tolerance
        and (max_2q is None or validation.N_2q <= max_2q)
    )
    if not valid:
        metrics = asdict(validation)
        row_metrics = (
            {f"f3_{key}": value for key, value in metrics.items()}
            if name == "F3" else metrics
        )
        raise GateSynthesisError(f"{name} failed gate validation: {metrics}", row_metrics)


def gate_metric_defaults() -> dict:
    return {
        "gate_library": GATE_LIBRARY,
        **{key: None for key in ("E_norm", "L_norm", "N_2q", "f3_E_norm", "f3_L_norm", "f3_N_2q")},
    }


def synthesize_f3(encoding: np.ndarray) -> QuantumCircuit:
    """Choose the free phase and synthesize F3 with at most two CNOTs.

    Outside monomial bases, solve Im tr(U YY U^T YY / sqrt(det U)) = 0.
    This is the two-CNOT criterion of Shende, Bullock and Markov:
    https://arxiv.org/abs/quant-ph/0308045 . Track the determinant square root
    continuously: its sign flips over a full phase turn, bracketing a root.
    """
    embedding = encoding_embedding(encoding)
    coded = embedding @ qutrit_fourier() @ embedding.conj().T
    free = np.eye(4) - embedding @ embedding.conj().T
    try:
        phase = optimal_f3_leakage_phase(embedding).phase
        method = "analytic_monomial"
    except ValueError:
        yy = np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
        determinant_root = np.sqrt(np.linalg.det(coded + free))

        def invariant(phi):
            unitary = coded + np.exp(1j * phi) * free
            return float(np.imag(
                np.trace(unitary @ yy @ unitary.T @ yy)
                / (determinant_root * np.exp(0.5j * phi))
            ))

        phase = (
            0.0 if abs(invariant(0.0)) < 1e-14 else
            brentq(invariant, 0.0, 2 * np.pi, xtol=1e-14)
        )
        method = "numerical_two_cnot_invariant"
    unitary = coded + np.exp(1j * phase) * free
    circuit = TwoQubitBasisDecomposer(CXGate(), euler_basis="U")(unitary)
    circuit.name = "F3_W"
    validation = validate_code_space_gate(circuit, embedding, qutrit_fourier())
    _accept(circuit, validation, name="F3", tolerance=F3_TOLERANCE, max_2q=2)
    circuit.metadata = {
        "gate_library": GATE_LIBRARY, "leakage_phase": float(phase),
        "synthesis_method": method, **asdict(validation),
    }
    return circuit


def _compile_cz3(embedding: np.ndarray) -> QuantumCircuit:
    # Lazy import keeps the F3 optimizer usable without BQSKit installed.
    from bqskit import MachineModel, compile
    from bqskit.ext import bqskit_to_qiskit
    from bqskit.ir.gates import CZGate, U3Gate
    from bqskit.qis import StateSystem, StateVector

    b2 = np.kron(embedding, embedding)
    outputs = b2 @ qutrit_cz()
    target = StateSystem({
        StateVector(b2[:, j], radixes=[2] * 4):
        StateVector(outputs[:, j], radixes=[2] * 4)
        for j in range(9)
    })
    compiled = compile(
        target, model=MachineModel(4, gate_set={U3Gate(), CZGate()}),
        optimization_level=2, max_synthesis_size=4,
        synthesis_epsilon=SYNTHESIS_EPSILON, seed=SYNTHESIS_SEED,
        num_workers=1,
    )
    # BQSKit matrix axes are big endian; QASM conversion preserves wire labels.
    # Reverse wires to make Operator(result) act on the original B2 columns.
    return bqskit_to_qiskit(compiled).reverse_bits()


def synthesize_cz3(encoding: np.ndarray) -> QuantumCircuit:
    embedding = encoding_embedding(encoding)
    # BQSKit launches a runtime; serialize launches also for threaded benchmarks.
    with _SYNTHESIS_LOCK:
        circuit = _compile_cz3(embedding)
    circuit.name = "CZ3_W"
    validation = validate_code_space_gate(circuit, np.kron(embedding, embedding), qutrit_cz())
    _accept(circuit, validation, name="CZ3", tolerance=CZ3_TOLERANCE)
    circuit.metadata = {"gate_library": GATE_LIBRARY, **asdict(validation)}
    return circuit


@dataclass(frozen=True)
class OptimizedGateLibrary:
    f3: QuantumCircuit
    cz3: QuantumCircuit
    synthesis_seconds: float
    cache_hit: bool = False

    def benchmark_metrics(self) -> dict:
        return {
            "gate_library": GATE_LIBRARY,
            # Requested bare names refer to the optimized four-qubit CZ3 block.
            **{key: self.cz3.metadata[key] for key in ("E_norm", "L_norm", "N_2q")},
            **{f"f3_{key}": self.f3.metadata[key] for key in ("E_norm", "L_norm", "N_2q")},
            "f3_leakage_phase": self.f3.metadata["leakage_phase"],
            "f3_synthesis_method": self.f3.metadata["synthesis_method"],
            "cz3_synthesis_seed": SYNTHESIS_SEED,
            "cz3_synthesis_epsilon": SYNTHESIS_EPSILON,
            "gate_synthesis_seconds": self.synthesis_seconds,
            "gate_cache_hit": self.cache_hit,
        }


def _load_cached_library(directory, embedding, specification) -> OptimizedGateLibrary:
    manifest = json.loads((directory / "synthesis.json").read_text(encoding="utf-8"))
    if any(manifest[name] != value for name, value in specification.items()):
        raise ValueError("Cache specification does not match the requested library.")
    if not np.array_equal(np.load(directory / "E.npy", allow_pickle=False), embedding):
        raise ValueError("Cached encoding does not match the requested encoding.")
    elapsed = manifest["synthesis_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid cached synthesis time.")
    circuits = []
    for filename, code, logical, name, tolerance, limit in (
        ("F3_W.qpy", embedding, qutrit_fourier(), "F3", F3_TOLERANCE, 2),
        ("CZ3_W.qpy", np.kron(embedding, embedding), qutrit_cz(), "CZ3", CZ3_TOLERANCE, None),
    ):
        with (directory / filename).open("rb") as handle:
            loaded = qpy.load(handle)
        if len(loaded) != 1:
            raise ValueError("Expected exactly one cached gate per QPY file.")
        circuit = loaded[0]
        validation = validate_code_space_gate(circuit, code, logical)
        _accept(circuit, validation, name=name, tolerance=tolerance, max_2q=limit)
        circuit.metadata = {**manifest[name], **asdict(validation)}
        circuits.append(circuit)
    library = OptimizedGateLibrary(*circuits, elapsed, True)
    library.benchmark_metrics()  # Check metadata needed by callers before accepting the hit.
    return library


def _publish_cached_library(directory, embedding, specification, library) -> None:
    # Stage on the same filesystem. The per-key process lock remains held until
    # the complete directory is renamed into place; no reader sees partial QPYs.
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for circuit, filename in ((library.f3, "F3_W.qpy"), (library.cz3, "CZ3_W.qpy")):
            with (staging / filename).open("wb") as handle:
                qpy.dump(circuit, handle)
        np.save(staging / "E.npy", embedding)
        (staging / "synthesis.json").write_text(json.dumps({
            **specification, "synthesis_seconds": library.synthesis_seconds,
            "F3": library.f3.metadata, "CZ3": library.cz3.metadata,
        }, indent=2), encoding="utf-8")
        if directory.exists():
            # Only an invalid cache reaches here. Move it aside under the lock;
            # interruption before publication leaves a cache miss, not partial data.
            directory.replace(Path(temporary) / "previous")
        staging.replace(directory)


def optimized_gate_library(
    encoding: np.ndarray,
    *,
    cache_dir: str | Path | None = None,
) -> OptimizedGateLibrary:
    """Synthesize once per exact encoding and reuse across states and trials.

    Cache identity includes the algorithm and dependency versions. On every
    load, recompute both code-space errors and inspect elementary gate counts.
    A per-key OS file lock covers reads, synthesis and atomic publication across
    processes. Incomplete or invalid cached libraries are synthesized again.
    """
    embedding = np.ascontiguousarray(encoding_embedding(encoding), dtype=np.complex128)
    specification = {
        # Isolate this layout from older processes that did not use a file lock.
        "cache_format": 2,
        "library": GATE_LIBRARY, "bqskit": version("bqskit"), "qiskit": version("qiskit"),
        "epsilon": SYNTHESIS_EPSILON, "seed": SYNTHESIS_SEED,
        "f3_tolerance": F3_TOLERANCE, "cz3_tolerance": CZ3_TOLERANCE,
    }
    key = hashlib.sha256(embedding.tobytes() + json.dumps(specification, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) if cache_dir is not None else Path(repo_path(
        "artifacts", "direct_basis_runs", "optimized_gates",
    ))
    directory = root / key
    root.mkdir(parents=True, exist_ok=True)
    # Keep lock files outside the directories that publication replaces. OS
    # locks are released even if the holder process terminates unexpectedly.
    with _SYNTHESIS_LOCK, FileLock(str(root / f"{key}.lock")):
        if directory.exists():
            try:
                return _load_cached_library(directory, embedding, specification)
            except (FileNotFoundError, ValueError, TypeError, KeyError, IndexError,
                    EOFError, struct.error, QiskitError) as exc:
                _LOGGER.warning("Rebuilding invalid gate cache %s: %s", directory, exc)
        started = time.perf_counter()
        f3 = synthesize_f3(embedding)
        cz3 = synthesize_cz3(embedding)
        elapsed = time.perf_counter() - started
        library = OptimizedGateLibrary(f3, cz3, elapsed)
        _publish_cached_library(directory, embedding, specification, library)
        return library
