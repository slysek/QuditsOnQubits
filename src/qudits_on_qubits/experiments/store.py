"""Safe, deterministic persistence for experiment run artifacts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib
import io
import json
import math
import numbers
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Any
import uuid

import numpy as np
from qiskit import qpy
from qiskit.circuit import QuantumCircuit

from .errors import ExperimentPersistenceError


_TAG = "__qoq_type__"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_NDARRAY_KINDS = frozenset("biufcU")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _safe_relative_path(value: str | Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise ExperimentPersistenceError("artifact filename must be a safe relative path") from error
    if not isinstance(raw, str) or not raw:
        raise ExperimentPersistenceError("artifact filename must be a safe relative path")
    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise ExperimentPersistenceError("artifact filename must be a safe relative path")
    parts = raw.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ExperimentPersistenceError("artifact filename must be a safe relative path without traversal")
    for part in parts:
        stem = part.split(".", 1)[0].upper()
        if (
            any(character in part for character in '<>:"|?*\0')
            or part[-1] in {" ", "."}
            or stem in _WINDOWS_RESERVED
        ):
            raise ExperimentPersistenceError("artifact filename must be a safe relative path")
    return Path(*parts)


def _raise_nonfinite(_: str) -> None:
    raise ExperimentPersistenceError("JSON numbers must be finite")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentPersistenceError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _encode(value: Any, active: set[int] | None = None) -> Any:
    if active is None:
        active = set()
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ExperimentPersistenceError("JSON numbers must be finite")
        return value
    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            _TAG: "enum",
            "module": enum_type.__module__,
            "name": value.name,
            "qualname": enum_type.__qualname__,
        }
    if isinstance(value, Path):
        return {_TAG: "path", "value": str(value)}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ExperimentPersistenceError("complex components must be finite")
        return {_TAG: "complex", "imag": value.imag, "real": value.real}
    if isinstance(value, np.generic):
        dtype = value.dtype
        if dtype.fields is not None or dtype.hasobject or dtype.kind not in _NDARRAY_KINDS:
            raise ExperimentPersistenceError(f"unsupported numpy scalar dtype {dtype}")
        return {_TAG: "numpy_scalar", "dtype": dtype.str, "value": _encode(value.item(), active)}
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if dtype.fields is not None or dtype.hasobject or dtype.kind not in _NDARRAY_KINDS:
            raise ExperimentPersistenceError(f"unsupported numpy array dtype {dtype}")
        identity = id(value)
        if identity in active:
            raise ExperimentPersistenceError("cyclic values cannot be persisted")
        active.add(identity)
        try:
            data = _encode(value.tolist(), active)
        finally:
            active.remove(identity)
        return {_TAG: "numpy_array", "data": data, "dtype": dtype.str, "shape": list(value.shape)}
    if isinstance(value, tuple):
        identity = id(value)
        if identity in active:
            raise ExperimentPersistenceError("cyclic values cannot be persisted")
        active.add(identity)
        try:
            return {_TAG: "tuple", "items": [_encode(item, active) for item in value]}
        finally:
            active.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise ExperimentPersistenceError("cyclic values cannot be persisted")
        active.add(identity)
        try:
            return [_encode(item, active) for item in value]
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ExperimentPersistenceError("cyclic values cannot be persisted")
        active.add(identity)
        try:
            if all(isinstance(key, str) for key in value) and _TAG not in value:
                return {key: _encode(item, active) for key, item in value.items()}
            return {
                _TAG: "mapping",
                "entries": [[_encode(key, active), _encode(item, active)] for key, item in value.items()],
            }
        finally:
            active.remove(identity)

    for method_name in ("to_dict", "to_safe_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            identity = id(value)
            if identity in active:
                raise ExperimentPersistenceError("cyclic values cannot be persisted")
            active.add(identity)
            try:
                converted = method()
                if converted is value:
                    raise ExperimentPersistenceError(f"{method_name} must return a new JSON-safe value")
                return _encode(converted, active)
            except ExperimentPersistenceError:
                raise
            except Exception as error:
                raise ExperimentPersistenceError(f"could not encode {type(value).__name__}.{method_name}()") from error
            finally:
                active.remove(identity)
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in active:
            raise ExperimentPersistenceError("cyclic values cannot be persisted")
        active.add(identity)
        try:
            return {field.name: _encode(getattr(value, field.name), active) for field in fields(value)}
        finally:
            active.remove(identity)
    raise ExperimentPersistenceError(f"unsupported value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ExperimentPersistenceError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        raise ExperimentPersistenceError(f"unsupported decoded JSON value: {type(value).__name__}")
    if _TAG not in value:
        return {key: _decode(item) for key, item in value.items()}

    tag = value.get(_TAG)
    if tag == "path" and set(value) == {_TAG, "value"} and isinstance(value["value"], str):
        return Path(value["value"])
    if tag == "complex" and set(value) == {_TAG, "imag", "real"}:
        real = _decode(value["real"])
        imag = _decode(value["imag"])
        if type(real) not in {int, float} or type(imag) not in {int, float}:
            raise ExperimentPersistenceError("complex components must be finite numbers")
        result = complex(real, imag)
        if not math.isfinite(result.real) or not math.isfinite(result.imag):
            raise ExperimentPersistenceError("complex components must be finite")
        return result
    if tag == "tuple" and set(value) == {_TAG, "items"} and isinstance(value["items"], list):
        return tuple(_decode(item) for item in value["items"])
    if tag == "mapping" and set(value) == {_TAG, "entries"} and isinstance(value["entries"], list):
        result: dict[Any, Any] = {}
        for entry in value["entries"]:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ExperimentPersistenceError("mapping entries must be key/value pairs")
            key, item = _decode(entry[0]), _decode(entry[1])
            try:
                duplicate = key in result
            except TypeError as error:
                raise ExperimentPersistenceError("decoded mapping key is not hashable") from error
            if duplicate:
                raise ExperimentPersistenceError("decoded mapping contains a duplicate key")
            result[key] = item
        return result
    if tag == "enum" and set(value) == {_TAG, "module", "name", "qualname"}:
        module_name, qualname, member_name = value["module"], value["qualname"], value["name"]
        if not all(isinstance(item, str) and item for item in (module_name, qualname, member_name)):
            raise ExperimentPersistenceError("invalid enum representation")
        try:
            enum_type: Any = importlib.import_module(module_name)
            for component in qualname.split("."):
                if component == "<locals>":
                    raise AttributeError("local enum classes cannot be reconstructed")
                enum_type = getattr(enum_type, component)
            if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
                raise TypeError("encoded type is not an Enum")
            return enum_type[member_name]
        except Exception as error:
            raise ExperimentPersistenceError("could not reconstruct encoded Enum") from error
    if tag == "numpy_scalar" and set(value) == {_TAG, "dtype", "value"}:
        dtype = _decode_dtype(value["dtype"])
        decoded = _decode(value["value"])
        try:
            return np.asarray(decoded, dtype=dtype).reshape(())[()]
        except (TypeError, ValueError, OverflowError) as error:
            raise ExperimentPersistenceError("invalid numpy scalar representation") from error
    if tag == "numpy_array" and set(value) == {_TAG, "data", "dtype", "shape"}:
        dtype = _decode_dtype(value["dtype"])
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ExperimentPersistenceError("invalid numpy array shape")
        try:
            return np.asarray(_decode(value["data"]), dtype=dtype).reshape(tuple(shape))
        except (TypeError, ValueError, OverflowError) as error:
            raise ExperimentPersistenceError("invalid numpy array representation") from error
    raise ExperimentPersistenceError(f"invalid or unsupported tagged JSON value {tag!r}")


def _decode_dtype(value: Any) -> np.dtype[Any]:
    if not isinstance(value, str):
        raise ExperimentPersistenceError("numpy dtype must be a string")
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise ExperimentPersistenceError("invalid numpy dtype") from error
    if dtype.fields is not None or dtype.hasobject or dtype.kind not in _NDARRAY_KINDS:
        raise ExperimentPersistenceError(f"unsupported numpy dtype {dtype}")
    return dtype


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = _encode(value)
        text = json.dumps(
            encoded,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ExperimentPersistenceError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ExperimentPersistenceError("could not encode canonical JSON") from error
    return (text + "\n").encode("utf-8")


def _decode_json_bytes(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
        encoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_nonfinite,
        )
        return _decode(encoded)
    except ExperimentPersistenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as error:
        raise ExperimentPersistenceError("could not decode JSON artifact") from error


def _validate_factor(factor: Any) -> int:
    if isinstance(factor, bool) or not isinstance(factor, numbers.Integral) or factor <= 0:
        raise ExperimentPersistenceError("counts factor must be a positive integer")
    return int(factor)


def _validate_counts_mapping(counts_by_setting: Any) -> list[dict[str, Any]]:
    if not isinstance(counts_by_setting, Mapping) or not counts_by_setting:
        raise ExperimentPersistenceError("counts must be a non-empty mapping by setting")
    settings: list[dict[str, Any]] = []
    for setting, counts in counts_by_setting.items():
        _encode(setting)
        try:
            hash(setting)
        except (TypeError, ValueError) as error:
            raise ExperimentPersistenceError("count setting keys must be hashable") from error
        if not isinstance(counts, Mapping) or not counts:
            raise ExperimentPersistenceError("counts for each setting must be a non-empty mapping")
        validated: dict[str, Any] = {}
        for outcome, count in counts.items():
            if not isinstance(outcome, str) or not outcome:
                raise ExperimentPersistenceError("count outcome keys must be non-empty strings")
            if isinstance(count, bool):
                raise ExperimentPersistenceError("count values must be integers or finite floats")
            if isinstance(count, (int, np.integer)):
                if count < 0:
                    raise ExperimentPersistenceError("integer count values must be non-negative")
            elif isinstance(count, (float, np.floating)):
                if not math.isfinite(float(count)):
                    raise ExperimentPersistenceError("floating count values must be finite")
            else:
                raise ExperimentPersistenceError("count values must be integers or finite floats")
            validated[outcome] = count
        settings.append({"setting": setting, "counts": validated})
    return settings


class ExperimentStore:
    """Persist a run beneath one resolved root without following escapes."""

    def __init__(self, root: str | Path):
        try:
            candidate = Path(root).expanduser()
            candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as error:
            raise ExperimentPersistenceError(f"could not initialize experiment root {root!r}") from error
        if not resolved.is_dir():
            raise ExperimentPersistenceError(f"experiment root is not a directory: {resolved}")
        self.root = resolved

    def create_run(self, run_id: str | None = None) -> Path:
        """Create and return a collision-safe UTC-dated run directory."""

        if run_id is not None and (not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id)):
            raise ExperimentPersistenceError("run_id must be one safe relative path component")
        now = _utc_now().astimezone(timezone.utc)
        date_directory = self.root / now.strftime("%Y-%m-%d")
        self._ensure_contained(date_directory.resolve(strict=False), self.root, "run directory")
        try:
            date_directory.mkdir(exist_ok=True)
            date_directory = date_directory.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ExperimentPersistenceError("could not create UTC run date directory") from error
        self._ensure_contained(date_directory, self.root, "run directory")

        timestamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        label = f"-{run_id}" if run_id is not None else ""
        name = f"{timestamp}{label}-{uuid.uuid4().hex[:12]}"
        run = date_directory / name
        self._ensure_contained(run.resolve(strict=False), self.root, "run directory")
        try:
            run.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise ExperimentPersistenceError(f"experiment run already exists and will not be overwritten: {run}") from error
        except OSError as error:
            raise ExperimentPersistenceError(f"could not create experiment run: {run}") from error
        resolved = run.resolve(strict=True)
        self._ensure_contained(resolved, self.root, "run directory")
        return resolved

    def write_json(self, run: str | Path, filename: str | Path, value: Any) -> Path:
        """Atomically write one canonical strict-JSON artifact."""

        path = self._artifact_path(run, filename, create_parent=True)
        self._atomic_write(path, _canonical_json_bytes(value))
        return path

    def read_json(self, run: str | Path, filename: str | Path) -> Any:
        """Read and validate one canonical JSON-compatible artifact."""

        path = self._artifact_path(run, filename, create_parent=False)
        return _decode_json_bytes(self._read_bytes(path))

    def write_experiment(
        self,
        run: str | Path,
        document: Mapping[str, Any] | None = None,
        **fields_by_name: Any,
    ) -> Path:
        """Write caller-supplied experiment state to ``experiment.json``."""

        if document is not None and fields_by_name:
            raise ExperimentPersistenceError("provide either an experiment document or keyword fields")
        if document is None:
            document = fields_by_name
        if not isinstance(document, Mapping):
            raise ExperimentPersistenceError("experiment document must be a mapping")
        return self.write_json(run, "experiment.json", document)

    def read_experiment(self, run: str | Path) -> dict[Any, Any]:
        """Read ``experiment.json`` and require a mapping document."""

        document = self.read_json(run, "experiment.json")
        if not isinstance(document, dict):
            raise ExperimentPersistenceError("experiment.json must contain a mapping")
        return document

    def write_circuits(
        self,
        run: str | Path,
        circuits: Any,
        filename: str | Path = "circuits.qpy",
    ) -> str:
        """Atomically write a non-empty QPY batch and return its exact SHA-256."""

        try:
            batch = tuple(circuits)
        except (TypeError, RuntimeError) as error:
            raise ExperimentPersistenceError("circuits must be an iterable QPY batch") from error
        if not batch:
            raise ExperimentPersistenceError("circuit QPY batch must not be empty")
        if any(not isinstance(circuit, QuantumCircuit) for circuit in batch):
            raise ExperimentPersistenceError("QPY batch values must be QuantumCircuit instances")
        stream = io.BytesIO()
        try:
            qpy.dump(batch, stream)
        except Exception as error:
            raise ExperimentPersistenceError("could not encode circuit QPY batch") from error
        data = stream.getvalue()
        path = self._artifact_path(run, filename, create_parent=True)
        self._atomic_write(path, data)
        return hashlib.sha256(data).hexdigest()

    def read_circuits(
        self,
        run: str | Path,
        filename: str | Path = "circuits.qpy",
    ) -> tuple[QuantumCircuit, ...]:
        """Read one QPY batch as an exact tuple of circuits."""

        path = self._artifact_path(run, filename, create_parent=False)
        stream = io.BytesIO(self._read_bytes(path))
        try:
            batch = tuple(qpy.load(stream))
            trailing = stream.read(1)
        except Exception as error:
            raise ExperimentPersistenceError(f"could not decode QPY artifact: {path}") from error
        if trailing or not batch or any(not isinstance(circuit, QuantumCircuit) for circuit in batch):
            raise ExperimentPersistenceError(f"invalid circuit QPY artifact: {path}")
        return batch

    def write_counts(
        self,
        run: str | Path,
        factor: int,
        counts_by_setting: Mapping[Any, Mapping[str, int | float]],
    ) -> Path:
        """Write validated raw or quasi counts using ordered setting entries."""

        normalized_factor = _validate_factor(factor)
        document = {
            "factor": normalized_factor,
            "settings": _validate_counts_mapping(counts_by_setting),
        }
        return self.write_json(run, f"counts-factor-{normalized_factor}.json", document)

    def read_counts(
        self,
        run: str | Path,
        factor: int,
    ) -> OrderedDict[Any, dict[str, int | float]]:
        """Read validated counts while preserving persisted setting order."""

        normalized_factor = _validate_factor(factor)
        document = self.read_json(run, f"counts-factor-{normalized_factor}.json")
        if not isinstance(document, dict) or set(document) != {"factor", "settings"}:
            raise ExperimentPersistenceError("counts artifact has an invalid document schema")
        if document["factor"] != normalized_factor or type(document["factor"]) is not int:
            raise ExperimentPersistenceError("counts artifact factor does not match its filename")
        settings = document["settings"]
        if not isinstance(settings, list) or not settings:
            raise ExperimentPersistenceError("counts artifact settings must be a non-empty list")
        result: OrderedDict[Any, dict[str, int | float]] = OrderedDict()
        for entry in settings:
            if not isinstance(entry, dict) or set(entry) != {"counts", "setting"}:
                raise ExperimentPersistenceError("counts artifact contains an invalid setting entry")
            setting = entry["setting"]
            try:
                if setting in result:
                    raise ExperimentPersistenceError("counts artifact contains a duplicate setting")
            except TypeError as error:
                raise ExperimentPersistenceError("counts artifact setting is not hashable") from error
            validated = _validate_counts_mapping({setting: entry["counts"]})
            result[setting] = validated[0]["counts"]
        return result

    def _resolve_run(self, run: str | Path) -> Path:
        try:
            candidate = Path(run)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as error:
            raise ExperimentPersistenceError(f"could not resolve experiment run: {run!r}") from error
        self._ensure_contained(resolved, self.root, "experiment root")
        if resolved == self.root or not resolved.is_dir():
            raise ExperimentPersistenceError(f"experiment run is not a directory below root: {resolved}")
        return resolved

    def _artifact_path(self, run: str | Path, filename: str | Path, *, create_parent: bool) -> Path:
        run_directory = self._resolve_run(run)
        relative = _safe_relative_path(filename)
        candidate = run_directory / relative
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ExperimentPersistenceError(f"could not resolve artifact path: {candidate}") from error
        self._ensure_contained(resolved, run_directory, "run directory")
        self._ensure_contained(resolved, self.root, "experiment root")
        if create_parent:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved = candidate.resolve(strict=False)
            except (OSError, RuntimeError) as error:
                raise ExperimentPersistenceError(f"could not create artifact directory: {candidate.parent}") from error
            self._ensure_contained(resolved, run_directory, "run directory")
            self._ensure_contained(resolved, self.root, "experiment root")
        return resolved

    @staticmethod
    def _ensure_contained(path: Path, directory: Path, description: str) -> None:
        if not _is_within(path, directory):
            raise ExperimentPersistenceError(f"artifact path escapes {description}: {path}")

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as error:
            raise ExperimentPersistenceError(f"could not read experiment artifact: {path}") from error

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        except Exception as error:
            raise ExperimentPersistenceError(f"could not atomically write experiment artifact: {path}") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = ["ExperimentStore"]
