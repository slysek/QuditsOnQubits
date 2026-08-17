"""Validated loading of direct-basis artifacts used by experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from qiskit import qpy

from .errors import ExperimentValidationError
from .models import Basis, BenchmarkBasis, PathBasis


_EXPECTED_QUBITS = {"two_qutrit": 4, "ghz3": 6, "ame43": 8}
_STATE_FILENAME = "graph_state_direct_basis.qpy"
_ENCODING_FILENAME = "E.npy"


@dataclass(frozen=True)
class BasisArtifacts:
    """In-memory, validated source artifacts and reproducibility evidence."""

    directory: Path
    state: str
    state_circuit: Any
    encoding: np.ndarray
    source_paths: Mapping[str, Path]
    source_hashes: Mapping[str, str]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory).resolve())
        object.__setattr__(self, "source_paths", MappingProxyType(dict(self.source_paths)))
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        encoding = np.asarray(self.encoding, dtype=complex).copy()
        encoding.setflags(write=False)
        object.__setattr__(self, "encoding", encoding)


def resolve_basis_directory(
    basis: Basis,
    state: str,
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve a safe basis selector to its artifact directory."""
    _validate_state(state)
    if isinstance(basis, PathBasis):
        directory = basis.directory.resolve()
        if not directory.is_dir():
            raise ExperimentValidationError(f"basis directory does not exist: {directory}")
        return directory
    if not isinstance(basis, BenchmarkBasis):
        raise ExperimentValidationError("basis must be a supported basis specification")

    for field_name in ("run_kind", "run_id", "selection"):
        _validate_selector_component(getattr(basis, field_name), field_name)
    _validate_selector_component(state, "state")
    root = Path.cwd() if repo_root is None else Path(repo_root)
    base = (root.resolve() / "artifacts" / basis.run_kind / "selected_best" / state / basis.run_id / basis.selection)
    if basis.candidate is not None:
        _validate_selector_component(basis.candidate, "candidate")
        directory = base / basis.candidate
        if not directory.is_dir():
            raise ExperimentValidationError(f"candidate basis directory does not exist: {directory}")
        return _resolve_benchmark_directory(directory, base)

    assert basis.rank is not None  # Guaranteed by BenchmarkBasis validation.
    candidates = sorted(path for path in base.glob(f"rank{basis.rank:02d}_*") if path.is_dir())
    if len(candidates) != 1:
        raise ExperimentValidationError(
            f"rank basis directory resolution failed at {base} for rank{basis.rank:02d}_*: count={len(candidates)}"
        )
    return _resolve_benchmark_directory(candidates[0], base)



def _resolve_benchmark_directory(directory: Path, base: Path) -> Path:
    resolved_base = base.resolve()
    try:
        resolved_directory = directory.resolve(strict=True)
    except OSError as error:
        raise ExperimentValidationError("could not resolve benchmark basis directory") from error
    if not resolved_directory.is_relative_to(resolved_base):
        raise ExperimentValidationError(f"benchmark basis directory escapes selected-best base: {resolved_directory} (base: {resolved_base})")
    return resolved_directory

def load_basis_artifacts(
    basis: Basis,
    state: str,
    repo_root: Path | str | None = None,
) -> BasisArtifacts:
    """Load the raw graph-state QPY and encoding array with strict validation."""
    _validate_state(state)
    directory = resolve_basis_directory(basis, state, repo_root)
    state_path = directory / _STATE_FILENAME
    encoding_path = directory / _ENCODING_FILENAME
    _require_file(state_path)
    _require_file(encoding_path)

    state_bytes = _read_artifact_bytes(state_path, "state circuit")
    encoding_bytes = _read_artifact_bytes(encoding_path, "encoding")
    state_circuit = _load_single_circuit(state_path)
    _validate_state_circuit(state_circuit, state)
    encoding = _load_encoding(encoding_path)

    return BasisArtifacts(
        directory=directory,
        state=state,
        state_circuit=state_circuit,
        encoding=encoding,
        source_paths={"state": state_path, "encoding": encoding_path},
        source_hashes={
            "state": hashlib.sha256(state_bytes).hexdigest(),
            "encoding": hashlib.sha256(encoding_bytes).hexdigest(),
        },
        provenance={"state": state, "basis": _safe_basis_provenance(basis)},
    )


def _validate_state(state: str) -> None:
    if state not in _EXPECTED_QUBITS:
        raise ExperimentValidationError("state must be two_qutrit, ghz3, or ame43")


def _validate_selector_component(value: str, field_name: str) -> None:
    path = Path(value)
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or path.is_absolute()
        or "/" in value
        or "\\" in value
    ):
        raise ExperimentValidationError(f"{field_name} must be a single safe path component")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ExperimentValidationError(f"required artifact is missing: {path}")


def _read_artifact_bytes(path: Path, artifact_name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ExperimentValidationError(f"could not read {artifact_name} artifact") from error


def _load_single_circuit(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            circuits = qpy.load(handle)
    except Exception as error:
        raise ExperimentValidationError("could not decode source state circuit") from error
    if len(circuits) != 1:
        raise ExperimentValidationError("source state circuit QPY must contain exactly one QuantumCircuit")
    return circuits[0]


def _validate_state_circuit(circuit: Any, state: str) -> None:
    expected_qubits = _EXPECTED_QUBITS[state]
    if circuit.num_qubits != expected_qubits:
        raise ExperimentValidationError(f"{state} source state circuit must contain {expected_qubits} qubits")
    for instruction in circuit.data:
        operation = instruction.operation
        if hasattr(operation, "blocks"):
            raise ExperimentValidationError("source state circuit must not contain control flow")
        if getattr(operation, "condition", None) is not None:
            raise ExperimentValidationError("source state circuit must not contain conditioned instructions")
    if circuit.num_clbits:
        raise ExperimentValidationError("source state circuit must not contain classical bits or measurements")
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "measure":
            raise ExperimentValidationError("source state circuit must not contain measurements")
        if operation.name == "reset":
            raise ExperimentValidationError("source state circuit must not contain resets")

def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(value) for key, value in values.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _load_encoding(path: Path) -> np.ndarray:
    try:
        encoding = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ExperimentValidationError("could not load encoding artifact") from error
    if not np.issubdtype(encoding.dtype, np.number):
        raise ExperimentValidationError("encoding artifact must contain numeric values")
    if encoding.shape != (4, 3):
        raise ExperimentValidationError("encoding artifact must have shape (4, 3)")
    if not np.all(np.isfinite(encoding)):
        raise ExperimentValidationError("encoding artifact must contain only finite values")
    encoding = np.asarray(encoding, dtype=complex)
    if not np.allclose(encoding.conj().T @ encoding, np.eye(3), rtol=1e-9, atol=1e-10):
        raise ExperimentValidationError("encoding artifact must be an isometry")
    return encoding


def _safe_basis_provenance(basis: Basis) -> Mapping[str, Any]:
    if isinstance(basis, PathBasis):
        return {"kind": "path", "directory": str(basis.directory.resolve())}
    return {
        "kind": "benchmark",
        "run_kind": basis.run_kind,
        "run_id": basis.run_id,
        "selection": basis.selection,
        "rank": basis.rank,
        "candidate": basis.candidate,
    }
