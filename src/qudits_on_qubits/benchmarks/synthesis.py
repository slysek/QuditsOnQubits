"""Explicit gate-synthesis policies and immutable replay of saved circuits.

Theta continuation uses the existing sequential fitting and fallback runner.
Saved policies never optimize or substitute imported gates. A saved full circuit
is exposed only when its caller explicitly binds it to a logical workload hash.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
from numbers import Integral, Real
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError

from .direct_basis import optimized_gates
from .direct_basis.math_utils import (
    encoding_embedding,
    physical_single_qutrit_gate_in_encoding,
    physical_two_qutrit_gate_in_encoding,
    qutrit_cz,
    qutrit_fourier,
)
from .theta_continuation.artifacts import RunStore, fingerprint, runtime_provenance
from .theta_continuation.encoding import theta_embedding
from .theta_continuation.library import ThetaGateLibrary
from .theta_continuation.models import ThetaBenchmarkConfig
from .theta_continuation.runner import run_theta_benchmark


class SynthesisArtifactError(ValueError):
    """Imported synthesis artifacts are invalid; abort rather than rank a failure."""


def _artifact_read(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (OSError, ValueError, TypeError, KeyError, EOFError, struct.error, QiskitError) as exc:
        raise SynthesisArtifactError(str(exc)) from exc


def _provenance() -> dict:
    result = runtime_provenance()
    result["source_hashes"]["qudits_on_qubits/benchmarks/synthesis.py"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return result


def _read_file(directory: Path, name: str) -> bytes:
    path = directory / name
    if not path.resolve().is_relative_to(directory):
        raise SynthesisArtifactError(f"Saved artifact escapes its directory: {name}")
    if not path.is_file():
        raise SynthesisArtifactError(f"Missing saved artifact: {name}")
    return _artifact_read(path.read_bytes)


def _validate_circuit(circuit: QuantumCircuit, *, width: int | None, name: str) -> None:
    if not isinstance(circuit, QuantumCircuit):
        raise SynthesisArtifactError(f"{name} must contain a QuantumCircuit")
    if width is not None and circuit.num_qubits != width:
        raise SynthesisArtifactError(f"{name} must have {width} qubits")
    if circuit.num_clbits or circuit.parameters:
        raise SynthesisArtifactError(f"{name} must have no classical bits or unbound parameters")
    if not np.isfinite(float(circuit.global_phase)):
        raise SynthesisArtifactError(f"{name} has a nonfinite global phase")
    for item in circuit.data:
        operation = item.operation
        if operation.name == "barrier":
            continue
        if not isinstance(operation, Gate) or operation.num_clbits:
            raise SynthesisArtifactError(f"{name} contains a nonunitary operation: {operation.name}")
        for parameter in operation.params:
            try:
                finite = np.isfinite(np.asarray(parameter, dtype=complex)).all()
            except (TypeError, ValueError) as exc:
                raise SynthesisArtifactError(f"{name} contains invalid gate parameters") from exc
            if not finite:
                raise SynthesisArtifactError(f"{name} contains nonfinite gate parameters")
        if operation.definition is not None:
            _validate_circuit(operation.definition, width=operation.num_qubits, name=name)


def _read_qpy(content: bytes, *, width: int | None, name: str) -> QuantumCircuit:
    circuits = _artifact_read(qpy.load, io.BytesIO(content))
    if len(circuits) != 1:
        raise SynthesisArtifactError(f"{name} must contain exactly one circuit")
    circuit = circuits[0]
    _validate_circuit(circuit, width=width, name=name)
    return circuit


def _read_encoding(content: bytes) -> np.ndarray:
    encoding = _artifact_read(np.load, io.BytesIO(content), allow_pickle=False)
    if encoding.shape != (4, 3) or encoding.dtype.kind not in "iufc":
        raise SynthesisArtifactError("Saved encoding must be a numeric 4x3 isometry")
    return _artifact_read(encoding_embedding, encoding).copy()


@dataclass(frozen=True)
class SynthesisGateLibrary:
    """Gate pair plus immutable-source provenance for a benchmark trial."""

    f3: QuantumCircuit
    cz3: QuantumCircuit
    synthesis_seconds: float = 0.0
    selected_method: str = "saved"
    provenance: dict[str, Any] = field(default_factory=dict)
    source_circuit: QuantumCircuit | None = None
    workload_hash: str | None = None
    _metrics: dict[str, Any] = field(default_factory=dict, repr=False)

    def benchmark_metrics(self) -> dict:
        return deepcopy(self._metrics)


class OptimizedSynthesis:
    """Existing code-space F3/CZ3 optimization with its validated disk cache."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = (Path.home() / ".cache" / "qudits_on_qubits" / "optimized_gates"
                          if cache_dir is None else Path(cache_dir))
        self._identity = {
            "strategy": "optimized", "schema_version": 1,
            "library": optimized_gates.GATE_LIBRARY,
            "seed": optimized_gates.SYNTHESIS_SEED,
            "epsilon": optimized_gates.SYNTHESIS_EPSILON,
            "f3_tolerance": optimized_gates.F3_TOLERANCE,
            "cz3_tolerance": optimized_gates.CZ3_TOLERANCE,
            "provenance": _provenance(),
        }

    def to_dict(self) -> dict:
        return deepcopy(self._identity)

    def prepare(self, candidates, output_dir, progress=None) -> None:
        """This policy builds each independent gate pair on demand."""

    def build(self, candidate):
        return optimized_gates.optimized_gate_library(candidate.encoding, cache_dir=self.cache_dir)


class ExactSynthesis:
    """Explicit dense-unitary control policy; never used as an optimized fallback."""

    def __init__(self):
        self._identity = {"strategy": "exact", "schema_version": 1,
                          "leakage_extension": "identity", "provenance": _provenance()}

    def to_dict(self) -> dict:
        return deepcopy(self._identity)

    def prepare(self, candidates, output_dir, progress=None) -> None:
        """Exact embedding has no cross-candidate preparation."""

    def build(self, candidate) -> SynthesisGateLibrary:
        f3, cz3 = QuantumCircuit(2, name="F3_W"), QuantumCircuit(4, name="CZ3_W")
        f3.unitary(physical_single_qutrit_gate_in_encoding(qutrit_fourier(), candidate.encoding),
                   [0, 1], label="F3_W")
        cz3.unitary(physical_two_qutrit_gate_in_encoding(qutrit_cz(), candidate.encoding),
                    [0, 1, 2, 3], label="CZ3_W")
        return SynthesisGateLibrary(f3, cz3, selected_method="exact",
                                    provenance={"parameter_source": "exact_embedding"})


class SavedGateSynthesis:
    """Replay an exported E.npy/F3_W.qpy/CZ3_W.qpy bundle without synthesis.

    ``workload_hash`` must be the associated ``LogicalCircuit.stable_hash()``.
    Without that binding, a present graph-state QPY is recorded but not reused.
    """

    def __init__(self, directory: str | Path, workload_hash: str | None = None):
        if workload_hash is not None and (
            not isinstance(workload_hash, str) or re.fullmatch(r"[0-9a-f]{64}", workload_hash) is None
        ):
            raise ValueError("workload_hash must be a lowercase SHA256 string")
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise ValueError("Saved gate directory must exist")
        self.workload_hash = workload_hash
        self._filenames = ["E.npy", "F3_W.qpy", "CZ3_W.qpy"]
        source = self.directory / "graph_state_direct_basis.qpy"
        if source.exists() or source.is_symlink():
            self._filenames.append(source.name)
        if workload_hash is not None and source.name not in self._filenames:
            raise ValueError("workload_hash requires graph_state_direct_basis.qpy")
        contents = {name: _read_file(self.directory, name) for name in self._filenames}
        self._hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
        self._encoding = _read_encoding(contents["E.npy"])
        self._decode(contents)  # Reject malformed imports before protocol identity is recorded.
        self._identity = {"strategy": "saved_gates", "schema_version": 1,
                          "parameter_source": "saved_qpy", "directory": str(self.directory),
                          "files": self._hashes, "workload_hash": workload_hash,
                          "provenance": _provenance()}

    def _decode(self, contents: dict[str, bytes]) -> tuple:
        f3 = _read_qpy(contents["F3_W.qpy"], width=2, name="F3_W.qpy")
        cz3 = _read_qpy(contents["CZ3_W.qpy"], width=4, name="CZ3_W.qpy")
        graph = None
        if "graph_state_direct_basis.qpy" in contents:
            graph = _read_qpy(contents["graph_state_direct_basis.qpy"], width=None,
                              name="graph_state_direct_basis.qpy")
            if graph.num_qubits < 2 or graph.num_qubits % 2:
                raise SynthesisArtifactError("Saved graph circuit width must be a positive number of qutrit pairs")
        return f3, cz3, graph

    def to_dict(self) -> dict:
        return deepcopy(self._identity)

    def prepare(self, candidates, output_dir, progress=None) -> None:
        for candidate in candidates:
            if not np.array_equal(candidate.encoding, self._encoding):
                raise ValueError("Saved encoding does not match the requested candidate encoding")

    def build(self, candidate) -> SynthesisGateLibrary:
        self.prepare([candidate], None)
        contents = {name: _read_file(self.directory, name) for name in self._filenames}
        if any(hashlib.sha256(content).hexdigest() != self._hashes[name]
               for name, content in contents.items()):
            raise SynthesisArtifactError("Saved artifact hash changed after synthesis identity was recorded")
        f3, cz3, graph = self._decode(contents)
        return SynthesisGateLibrary(
            f3, cz3, selected_method="saved", provenance={"parameter_source": "saved_qpy",
                                                         "files": dict(self._hashes)},
            source_circuit=graph if self.workload_hash is not None else None,
            workload_hash=self.workload_hash,
        )


def saved_candidate(directory: str | Path, candidate_id: str = "saved"):
    """Create a replay candidate from an exported encoding, without gate fitting."""
    from .models import EncodingCandidate

    directory = Path(directory).resolve()
    content = _read_file(directory, "E.npy")
    return EncodingCandidate(candidate_id=candidate_id, family="saved", encoding=_read_encoding(content),
                             parameters={"encoding_sha256": hashlib.sha256(content).hexdigest()})


def _matches_theta_encoding(encoding, theta: float) -> bool:
    # libm sin/cos can differ by a few ULPs across platforms. Bundle hashes
    # remain exact; this only checks numerical agreement with the theta family.
    expected = theta_embedding(theta)
    return (isinstance(encoding, np.ndarray) and encoding.shape == expected.shape
            and np.issubdtype(encoding.dtype, np.number)
            and np.allclose(encoding, expected, atol=1e-14, rtol=0.0))


class ThetaContinuationSynthesis:
    """Sequential angle fitting/fallback, or exact reuse of a completed theta run."""

    def __init__(self, config: ThetaBenchmarkConfig, source_run: str | Path | None = None):
        if not isinstance(config, ThetaBenchmarkConfig):
            raise TypeError("config must be a ThetaBenchmarkConfig")
        self.config = config
        self.source_run = None if source_run is None else Path(source_run).resolve()
        self._store: RunStore | None = None
        self._source_snapshot: dict | None = None
        if self.source_run is not None:
            self._store = self._open(self.source_run)
            self._source_snapshot = self._snapshot(self._store)
        parameter_source = "continuation_fit"
        if self._store is not None:
            parameter_source = ("saved_theta_reassessment"
                                if self._store.manifest["benchmark"] == "theta_threshold_reassessment_v1"
                                else "saved_theta_run")
        self._identity = {"strategy": "theta_continuation", "schema_version": 1,
                          "config": config.to_dict(), "provenance": _provenance(),
                          "parameter_source": parameter_source,
                          "source": deepcopy(self._source_snapshot)}

    def to_dict(self) -> dict:
        return deepcopy(self._identity)

    def _open(self, root: Path) -> RunStore:
        store = _artifact_read(RunStore.open, root)
        manifest = store.manifest
        payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if fingerprint(payload) != manifest["fingerprint"]:
            raise SynthesisArtifactError("Theta manifest fingerprint does not match its content")
        if manifest.get("format_version") != 1 or manifest.get("benchmark") not in {
            "theta_continuation_v1", "theta_threshold_reassessment_v1",
        }:
            raise SynthesisArtifactError("Unsupported theta run manifest schema")
        if manifest.get("config") != self.config.to_dict():
            raise SynthesisArtifactError("Theta source config does not match the explicit synthesis config")
        if manifest["benchmark"] == "theta_threshold_reassessment_v1":
            self._validate_reassessment_manifest(manifest)
        if manifest.get("baseline_sha256") != self.config.baseline_sha256:
            raise SynthesisArtifactError("Theta source baseline hash does not match config")
        if not isinstance(manifest.get("template_id"), str) or not manifest["template_id"]:
            raise SynthesisArtifactError("Theta source template identity is missing")
        if store.completed_point_indices() != list(range(len(self.config.grid()))):
            raise SynthesisArtifactError("Theta source must contain the complete configured grid")
        return store

    def _validate_reassessment_manifest(self, manifest: dict) -> None:
        for name in ("source_fingerprint", "source_inventory_fingerprint"):
            value = manifest.get(name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SynthesisArtifactError(f"Invalid reassessment source hash: {name}")
        tolerance = manifest.get("source_cz3_tolerance")
        if (type(manifest.get("procedure_version")) is not int or manifest["procedure_version"] != 1
                or manifest.get("continuation_policy") != "reassess_saved_attempts_preserve_historical_branch"
                or not isinstance(manifest.get("recovery_protocol"), dict)
                or isinstance(tolerance, bool) or not isinstance(tolerance, Real)
                or not np.isfinite(tolerance) or not 0 < tolerance <= self.config.cz3_tolerance):
            raise SynthesisArtifactError("Unsupported or invalid theta reassessment provenance")

    def _point(self, store: RunStore, index: int) -> dict:
        bundle = _artifact_read(store.read_bundle, f"points/{index:05d}")
        metadata = bundle["metadata"]
        theta = self.config.grid()[index]
        if (metadata.get("index") != index or metadata.get("theta") != theta
                or metadata.get("template_id") != store.manifest["template_id"]):
            raise SynthesisArtifactError("Saved theta point identity does not match the configured grid")
        if not _matches_theta_encoding(bundle["arrays"].get("E.npy"), theta):
            raise SynthesisArtifactError("Saved theta encoding does not match its grid point")
        if store.manifest["benchmark"] == "theta_threshold_reassessment_v1":
            details = metadata.get("reassessment")
            if (not isinstance(details, dict) or details != bundle["documents"].get("reassessment.json")
                    or details.get("cz3_tolerance") != self.config.cz3_tolerance
                    or details.get("source_cz3_tolerance") != store.manifest["source_cz3_tolerance"]
                    or details.get("continuation_policy") != store.manifest["continuation_policy"]
                    or not isinstance(details.get("source_point_fingerprint"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", details["source_point_fingerprint"]) is None):
                raise SynthesisArtifactError("Saved theta reassessment point provenance mismatch")
        return bundle

    def _snapshot(self, store: RunStore) -> dict:
        return {"directory": str(store.root), "manifest": store.manifest["fingerprint"],
                "manifest_data": store.manifest,
                "points": {f"{index:05d}": self._point(store, index)["complete"]["files"]
                           for index in range(len(self.config.grid()))}}

    def _index(self, candidate) -> int:
        if candidate.family == "canonical":
            index, theta = 0, 0.0
        elif candidate.family == "schmidt_theta":
            theta = candidate.parameters.get("theta")
            if isinstance(theta, bool) or not isinstance(theta, Real) or not np.isfinite(theta):
                raise ValueError("Theta candidate must declare finite theta")
            try:
                index = self.config.grid().index(float(theta))
            except ValueError as exc:
                raise ValueError("Theta candidate does not belong to the configured grid") from exc
            supplied_index = candidate.parameters.get("grid_index", index)
            if isinstance(supplied_index, bool) or not isinstance(supplied_index, Integral) or supplied_index != index:
                raise ValueError("Theta candidate grid_index does not match the configured grid")
        else:
            raise ValueError(f"Theta continuation cannot synthesize candidate family: {candidate.family}")
        if not _matches_theta_encoding(candidate.encoding, theta):
            raise ValueError("Candidate encoding does not match its theta grid point")
        return index

    def prepare(self, candidates, output_dir, progress=None) -> None:
        for candidate in candidates:
            self._index(candidate)
        if self._store is None:
            destination = Path(output_dir) / "theta_synthesis"
            run_theta_benchmark(self.config, destination, mode="gates", progress=progress)
            self._store = self._open(destination)
            self._source_snapshot = self._snapshot(self._store)

    def build(self, candidate) -> SynthesisGateLibrary:
        index = self._index(candidate)
        if self._store is None:
            raise RuntimeError("Call prepare before building a new theta continuation run")
        current = self._open(self._store.root)
        if current.manifest != self._store.manifest:
            raise SynthesisArtifactError("Theta source manifest changed after its synthesis identity was recorded")
        bundle = self._point(current, index)
        if bundle["complete"]["files"] != self._source_snapshot["points"][f"{index:05d}"]:
            raise SynthesisArtifactError("Theta source artifact hashes changed after its synthesis identity was recorded")
        if bundle["metadata"].get("correct") is not True:
            raise ValueError(f"Theta gate point is not correct: {bundle['metadata'].get('errors', [])}")
        for name, width in (("f3_optimal.qpy", 2), ("cz3_selected.qpy", 4)):
            if name in bundle["circuits"]:
                _validate_circuit(bundle["circuits"][name], width=width, name=name)
        library = _artifact_read(ThetaGateLibrary.from_bundle, bundle, candidate.encoding,
                                 f3_tolerance=self.config.f3_tolerance,
                                 cz3_tolerance=self.config.cz3_tolerance)
        return SynthesisGateLibrary(
            library.f3, library.cz3, library.synthesis_seconds, library.selected_method,
            provenance={"parameter_source": self._identity["parameter_source"],
                        "point_index": index, "selected_method": library.selected_method,
                        "source_manifest": current.manifest["fingerprint"],
                        "source_benchmark": current.manifest["benchmark"],
                        "reassessment": deepcopy(bundle["metadata"].get("reassessment")),
                        "files": deepcopy(bundle["complete"]["files"])},
            _metrics=_artifact_read(library.benchmark_metrics),
        )
