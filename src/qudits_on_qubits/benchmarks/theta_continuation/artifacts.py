"""Atomic, validated artifacts for resumable theta-continuation runs.

A bundle becomes reusable only when ``complete.json`` has been published. Files
left by an interrupted write remain available for inspection and can be rewritten
by a later attempt; a completed bundle is never overwritten.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POINT_INDEX = re.compile(r"[0-9]{5,}\Z")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
}


def _validate_json_keys(value: Any) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        for item in value.values():
            _validate_json_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_keys(item)


def _json_bytes(payload: dict) -> bytes:
    if not isinstance(payload, dict):
        raise TypeError("JSON payload must be an object")
    _validate_json_keys(payload)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def fingerprint(payload: dict) -> str:
    """Hash a canonical, sorted JSON object; reject nonfinite JSON numbers."""
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Publish strict JSON using a temporary file in the destination directory."""
    _atomic_write_bytes(Path(path), _json_bytes(payload))


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_json(content: bytes) -> dict:
    payload = json.loads(
        content, parse_constant=_reject_constant, object_pairs_hook=_unique_object,
    )
    # Re-encoding also rejects finite-looking literals that overflow to infinity.
    _json_bytes(payload)
    return payload


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Artifact path must be a nonempty POSIX relative path")
    parts = tuple(relative.split("/"))
    for part in parts:
        if (
            part in ("", ".", "..")
            or part.endswith((".", " "))
            or any(character in '<>:"|?*' or ord(character) < 32 for character in part)
            or part.split(".")[0].upper() in _WINDOWS_RESERVED
        ):
            raise ValueError(f"Unsafe artifact path: {relative!r}")
    return parts


def _safe_path(root: Path, relative: str) -> Path:
    path = root.joinpath(*_relative_parts(relative))
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError) as exc:
        raise ValueError(f"Artifact path escapes run directory: {relative!r}") from exc
    return path


def _validate_filename(name: str, suffix: str | None = None) -> None:
    parts = _relative_parts(name)
    if len(parts) != 1 or (suffix is not None and not name.endswith(suffix)):
        raise ValueError(f"Artifact filename must be a leaf ending in {suffix}: {name!r}")


def _validate_array(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "iufc":
        raise ValueError("NPY artifacts must be numeric NumPy arrays without object values")
    if not np.isfinite(array).all():
        raise ValueError("NPY artifacts must contain only finite values")


def _validate_manifest(manifest: dict) -> None:
    _json_bytes(manifest)
    if not isinstance(manifest.get("fingerprint"), str) or not manifest["fingerprint"]:
        raise ValueError("Run manifest must contain a nonempty fingerprint string")


def runtime_provenance() -> dict:
    """Return package versions and source hashes without machine-specific paths."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "qiskit", "bqskit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    directory = Path(__file__).resolve().parent
    source_root = directory.parents[2]
    package_root = directory.parents[1]
    # Both direct-basis builders resolve their targets through states.py and
    # graph_states.py; fixed graph edges come from reference_experiments.py.
    # Include the whole dependency chain so changed targets invalidate resume.
    paths = sorted([
        *directory.glob("*.py"),
        *directory.parent.joinpath("direct_basis").glob("*.py"),
        package_root / "encoding_search" / "states.py",
        package_root / "core" / "graph_states.py",
        package_root / "reference_experiments.py",
    ])
    source_hashes = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return {"versions": versions, "source_hashes": source_hashes}


class RunStore:
    """One run directory containing immutable completed bundles and a checkpoint."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self._manifest_bytes = _json_bytes(manifest)

    @property
    def manifest(self) -> dict:
        """Return an independent copy of the manifest as it was opened."""
        return _decode_json(self._manifest_bytes)

    @classmethod
    def create(cls, output_dir: str | Path, manifest: dict) -> RunStore:
        _validate_manifest(manifest)
        root = Path(output_dir).resolve()
        if root.exists():
            if not root.is_dir() or any(root.iterdir()):
                raise FileExistsError(f"Run directory must be new or empty: {root}")
        else:
            root.mkdir(parents=True)
        atomic_write_json(_safe_path(root, "manifest.json"), manifest)
        return cls(root, manifest)

    @classmethod
    def open(
        cls, output_dir: str | Path, *, expected_fingerprint: str | None = None,
    ) -> RunStore:
        root = Path(output_dir).resolve()
        manifest = _decode_json(_safe_path(root, "manifest.json").read_bytes())
        _validate_manifest(manifest)
        if expected_fingerprint is not None and manifest["fingerprint"] != expected_fingerprint:
            raise ValueError("Run manifest fingerprint does not match expected fingerprint")
        return cls(root, manifest)

    def write_bundle(
        self,
        relative_dir: str,
        *,
        metadata: dict,
        circuits: dict[str, QuantumCircuit] | None = None,
        arrays: dict[str, np.ndarray] | None = None,
        documents: dict[str, dict] | None = None,
    ) -> dict:
        """Write a bundle and publish its complete hash manifest last."""
        directory = _safe_path(self.root, relative_dir)
        complete_path = _safe_path(self.root, f"{relative_dir}/complete.json")
        if complete_path.exists() or complete_path.is_symlink():
            raise FileExistsError(f"Completed bundle cannot be overwritten: {relative_dir}")
        files = {"metadata.json": _json_bytes(metadata)}
        names = {"metadata.json", "complete.json"}
        for collection, suffix in ((circuits, ".qpy"), (arrays, ".npy"), (documents, ".json")):
            if collection is None:
                continue
            if not isinstance(collection, dict):
                raise TypeError("Artifact collections must map filenames to payloads")
            for name, payload in collection.items():
                _validate_filename(name, suffix)
                if name.casefold() in names:
                    raise ValueError(f"Duplicate or reserved artifact filename: {name}")
                names.add(name.casefold())
                _safe_path(self.root, f"{relative_dir}/{name}")
                if suffix == ".json":
                    files[name] = _json_bytes(payload)
                else:
                    buffer = io.BytesIO()
                    if suffix == ".npy":
                        _validate_array(payload)
                        np.save(buffer, payload, allow_pickle=False)
                    else:
                        if not isinstance(payload, QuantumCircuit):
                            raise TypeError("Each QPY artifact must contain one QuantumCircuit")
                        qpy.dump(payload, buffer)
                    files[name] = buffer.getvalue()
        # Validate reserved metadata paths too, including preexisting symlinks.
        _safe_path(self.root, f"{relative_dir}/metadata.json")
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            _atomic_write_bytes(_safe_path(self.root, f"{relative_dir}/{name}"), content)
        complete = {
            "schema_version": 1,
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        }
        atomic_write_json(complete_path, complete)
        return complete

    def read_bundle(self, relative_dir: str) -> dict:
        """Verify every tracked file before decoding any circuit or array."""
        _safe_path(self.root, relative_dir)
        complete = _decode_json(_safe_path(self.root, f"{relative_dir}/complete.json").read_bytes())
        entries = complete.get("files")
        if complete.get("schema_version") != 1 or not isinstance(entries, dict) or "metadata.json" not in entries:
            raise ValueError("Invalid bundle completion manifest")
        contents = {}
        names = {"complete.json"}
        for name, digest in entries.items():
            _validate_filename(name)
            if name.casefold() in names or Path(name).suffix not in (".json", ".qpy", ".npy"):
                raise ValueError(f"Invalid or duplicate tracked artifact filename: {name}")
            names.add(name.casefold())
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"Invalid artifact hash: {name}")
            path = _safe_path(self.root, f"{relative_dir}/{name}")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Artifact hash mismatch: {name}")
            contents[name] = content
        result: dict = {
            "metadata": {}, "circuits": {}, "arrays": {}, "documents": {}, "complete": complete,
        }
        for name, content in contents.items():
            suffix = Path(name).suffix
            if name == "metadata.json":
                result["metadata"] = _decode_json(content)
            elif suffix == ".json":
                result["documents"][name] = _decode_json(content)
            elif suffix == ".npy":
                array = np.load(io.BytesIO(content), allow_pickle=False)
                _validate_array(array)
                result["arrays"][name] = array
            else:
                circuits = qpy.load(io.BytesIO(content))
                if len(circuits) != 1 or not isinstance(circuits[0], QuantumCircuit):
                    raise ValueError(f"QPY artifact must contain exactly one circuit: {name}")
                result["circuits"][name] = circuits[0]
        return result

    def completed_point_indices(self) -> list[int]:
        """List canonical point directories with completion markers, without reuse."""
        directory = _safe_path(self.root, "points")
        if not directory.exists():
            return []
        indices = []
        for candidate in directory.iterdir():
            if _POINT_INDEX.fullmatch(candidate.name) is None:
                continue
            index = int(candidate.name)
            if candidate.name != f"{index:05d}":
                continue
            path = _safe_path(self.root, f"points/{candidate.name}")
            marker = _safe_path(self.root, f"points/{candidate.name}/complete.json")
            if path.is_dir() and (marker.exists() or marker.is_symlink()):
                indices.append(index)
        return sorted(indices)

    def save_checkpoint(self, payload: dict) -> None:
        """Atomically replace the latest checkpoint, including its payload hash."""
        envelope = {"payload": payload, "sha256": fingerprint(payload)}
        atomic_write_json(_safe_path(self.root, "checkpoints/latest.json"), envelope)

    def load_checkpoint(self) -> dict | None:
        """Load a validated checkpoint, or None when no checkpoint exists."""
        path = _safe_path(self.root, "checkpoints/latest.json")
        if not path.exists() and not path.is_symlink():
            return None
        envelope = _decode_json(path.read_bytes())
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("sha256") != fingerprint(payload):
            raise ValueError("Checkpoint payload hash mismatch")
        return payload
