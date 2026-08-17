"""Durable orchestration for Bell experiments.

The runner intentionally persists only reconstructible, provider-safe values.
Backend objects, submitted handles, callbacks, run options, and exception
tracebacks never cross the artifact boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from qiskit import QuantumCircuit

from qudits_on_qubits.bell_measurements import decoding_kwargs_from_metadata

from .artifacts import load_basis_artifacts
from .backends import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
    create_backend_adapter,
)
from .errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentPersistenceError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from .mitigation import (
    ReadoutCalibration,
    assignment_matrices_from_counts,
    build_readout_calibration_circuits,
    calibration_cache_is_valid,
    fold_cz_batch,
    validate_zne_factors,
)
from .models import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    RetryConfig,
)
from .preparation import prepare_measurements
from .store import ExperimentStore
from .uncertainty import BootstrapInputs, bootstrap_bell_results


SCHEMA_VERSION = 1
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_CREDENTIAL_MARKER = re.compile(
    r"(?i)(?:authorization|bearer|token|api[-_ ]?key|password|secret)\b"
    r"(?:\s*[:=]\s*|\s+)\S+"
)
_URL = re.compile(r"(?i)https?://[^\s<>\"']+")
_CREDENTIAL_QUERY_KEYS = {
    "authorization",
    "bearer",
    "token",
    "api_key",
    "apikey",
    "password",
    "secret",
}


def _credential_field_name(value: str) -> bool:
    normalized = re.sub(r"[-_\s]+", "_", value.strip().lower())
    return normalized in _CREDENTIAL_QUERY_KEYS or normalized.endswith(
        ("_authorization", "_token", "_api_key", "_apikey", "_password", "_secret")
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ExperimentValidationError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _unsafe_persisted_text(value: str) -> bool:
    if _CONTROL_CHARACTERS.search(value) or _CREDENTIAL_MARKER.search(value):
        return True
    for candidate in _URL.findall(value):
        try:
            parsed = urlsplit(candidate)
            if parsed.username is not None or parsed.password is not None:
                return True
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                if _credential_field_name(key):
                    return True
        except ValueError:
            return True
    return False


def _safe_message(error: BaseException) -> str:
    message = str(error)
    if _unsafe_persisted_text(message):
        return "details redacted"
    message = " ".join(message.split())
    return message[:500] or "operation failed"


def _validate_persisted_strings(
    value: Any,
    *,
    description: str,
    active: set[int] | None = None,
) -> None:
    if isinstance(value, str):
        if _unsafe_persisted_text(value):
            raise BackendCompatibilityError(f"{description} contains unsafe text")
        return
    if isinstance(value, (bytes, bytearray)):
        return
    if not isinstance(value, (Mapping, Sequence, Set)):
        return
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise BackendCompatibilityError(f"{description} contains a recursive value")
    active.add(identity)
    try:
        items = value.items() if isinstance(value, Mapping) else enumerate(value)
        for key, item in items:
            if isinstance(value, Mapping) and isinstance(key, str) and _credential_field_name(key):
                raise BackendCompatibilityError(f"{description} contains unsafe text")
            _validate_persisted_strings(key, description=description, active=active)
            _validate_persisted_strings(item, description=description, active=active)
    finally:
        active.remove(identity)


def _failure(error: BaseException, stage: str, clock: Callable[[], datetime], attempt: int | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": _safe_message(error),
        "attempt": attempt,
        "timestamp": _timestamp(clock),
    }


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ExperimentPersistenceError(f"could not hash experiment artifact: {path.name}") from error


def _output_root(spec: ExperimentSpec, repo_root: Path | str | None) -> Path:
    root = spec.output_root
    if root.is_absolute():
        return root
    base = Path.cwd() if repo_root is None else Path(repo_root)
    return base / root


def _validate_execution_options(
    timeout: float | None, run_options: Mapping[str, Any] | None
) -> None:
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ExperimentValidationError("timeout must be a positive finite number or None")
    if run_options is not None and not isinstance(run_options, Mapping):
        raise ExperimentValidationError("run_options must be a mapping or None")


def _initial_document(
    spec: ExperimentSpec,
    experiment_id: str,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    created = _timestamp(clock)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "spec": spec.to_safe_dict(),
        "status": ExperimentStatus.CREATED.value,
        "timestamps": {"created": created, "updated": created},
        "status_history": [{"status": ExperimentStatus.CREATED.value, "timestamp": created}],
        "attempts": [],
        "backend": None,
        "jobs": {},
        "job_ids": [],
        "counts": {},
        "circuits": {"source": None, "logical": None, "factors": {}},
        "postprocessing": None,
        "calibration": None,
        "result": None,
        "failure": None,
    }


def _write_state(store: ExperimentStore, run: Path, document: Mapping[str, Any]) -> None:
    store.write_experiment(run, document)


def _transition(
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    status: ExperimentStatus,
    clock: Callable[[], datetime],
) -> None:
    timestamp = _timestamp(clock)
    document["status"] = status.value
    document["timestamps"]["updated"] = timestamp
    document["timestamps"][status.value] = timestamp
    document["status_history"].append({"status": status.value, "timestamp": timestamp})
    _write_state(store, run, document)


def _record_attempt(
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    operation: str,
    attempt: int,
    outcome: str,
    clock: Callable[[], datetime],
    error: BaseException | None = None,
) -> None:
    record: dict[str, Any] = {
        "operation": operation,
        "attempt": attempt,
        "outcome": outcome,
        "timestamp": _timestamp(clock),
    }
    if error is not None:
        record["exception_type"] = type(error).__name__
        record["message"] = _safe_message(error)
    document["attempts"].append(record)
    document["timestamps"]["updated"] = record["timestamp"]
    _write_state(store, run, document)


def _retry(
    operation: str,
    action: Callable[[], Any],
    retry: RetryConfig,
    retry_errors: tuple[type[BaseException], ...],
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    sleep: Callable[[float], None],
    clock: Callable[[], datetime],
) -> Any:
    delay = float(retry.initial_delay)
    for attempt in range(1, retry.max_attempts + 1):
        try:
            value = action()
        except retry_errors as error:
            _record_attempt(store, run, document, operation, attempt, "failed", clock, error)
            if attempt == retry.max_attempts:
                raise
            sleep(delay)
            delay = min(float(retry.max_delay), delay * float(retry.multiplier))
        else:
            _record_attempt(store, run, document, operation, attempt, "succeeded", clock)
            return value
    raise AssertionError("unreachable retry state")


def _check_availability(adapter: Any) -> Availability:
    try:
        availability = adapter.availability()
    except (KeyboardInterrupt, SystemExit):
        raise
    except (BackendUnavailableError, BackendCompatibilityError):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter availability check failed") from None
    if not isinstance(availability, Availability):
        raise BackendCompatibilityError("adapter availability must return Availability")
    _validate_persisted_strings(
        availability.to_safe_dict(), description="adapter availability metadata"
    )
    if not availability.available:
        raise BackendUnavailableError(availability.reason or "backend is unavailable")
    return availability


def _preflight(adapter: Any, circuits: Sequence[Any], shots: int) -> None:
    try:
        adapter.preflight(circuits, shots)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (BackendUnavailableError, BackendCompatibilityError):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter preflight failed") from None


def _validate_adapter(adapter: Any) -> tuple[BackendIdentity, BackendCapabilities, Mapping[str, Any]]:
    try:
        identity = adapter.resolve()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter resolution failed") from None
    if not isinstance(identity, BackendIdentity):
        raise BackendCompatibilityError("adapter resolve must return BackendIdentity")
    try:
        capabilities = adapter.capabilities()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter capability discovery failed") from None
    if not isinstance(capabilities, BackendCapabilities):
        raise BackendCompatibilityError("adapter capabilities must return BackendCapabilities")
    try:
        metadata = adapter.metadata()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter metadata discovery failed") from None
    if not isinstance(metadata, Mapping):
        raise BackendCompatibilityError("adapter metadata must return a mapping")
    try:
        _validate_persisted_strings(identity.to_safe_dict(), description="backend identity")
        _validate_persisted_strings(
            capabilities.to_safe_dict(), description="backend capabilities"
        )
        _validate_persisted_strings(metadata, description="adapter metadata")
    except (KeyboardInterrupt, SystemExit, BackendCompatibilityError):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter metadata validation failed") from None
    return identity, capabilities, metadata


def _require_durable_remote_jobs(capabilities: BackendCapabilities) -> None:
    if not capabilities.local and not capabilities.supports_resume:
        raise BackendCompatibilityError("remote backend must support job resume")


def _persist_prepared(
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    artifacts: Any,
    prepared: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    source_hash = store.write_circuits(run, (artifacts.state_circuit,), "source-state.qpy")
    logical_hash = store.write_circuits(run, prepared.circuits, "logical-measurements.qpy")
    store.write_json(run, "source-encoding.json", artifacts.encoding)
    metadata = prepared.metadata
    required = ("setting_by_circuit_index", "terms", "qutrit_bit_indices_by_setting")
    if any(name not in metadata for name in required):
        raise ExperimentValidationError("measurement metadata is incomplete for durable postprocessing")
    settings = tuple(tuple(setting) for setting in metadata["setting_by_circuit_index"])
    if len(settings) != len(tuple(prepared.circuits)):
        raise ExperimentValidationError("measurement settings do not match logical circuits")
    postprocessing = {
        "terms": metadata["terms"],
        "setting_by_circuit_index": settings,
        "qutrit_bit_indices_by_setting": metadata["qutrit_bit_indices_by_setting"],
        "decoding_kwargs": decoding_kwargs_from_metadata(metadata),
        "qutrit_qubits": metadata.get("qutrit_qubits", ()),
        "candidate": metadata.get("candidate"),
    }
    post_path = store.write_json(run, "postprocessing.json", postprocessing)
    document["circuits"]["source"] = {
        "artifact": "source-state.qpy",
        "sha256": source_hash,
        "origin_sha256": artifacts.source_hashes.get("state"),
    }
    document["circuits"]["logical"] = {
        "artifact": "logical-measurements.qpy",
        "sha256": logical_hash,
    }
    document["source"] = {
        "hashes": dict(artifacts.source_hashes),
        "provenance": artifacts.provenance,
        "encoding_artifact": "source-encoding.json",
        "encoding_sha256": _sha256(run / "source-encoding.json"),
    }
    document["postprocessing"] = {
        "artifact": "postprocessing.json",
        "sha256": _sha256(post_path),
    }
    _write_state(store, run, document)
    return settings, postprocessing


def _factor_batches(
    compiled: CompiledBatch,
    factors: Sequence[int],
) -> dict[int, tuple[QuantumCircuit, ...]]:
    batches: dict[int, tuple[QuantumCircuit, ...]] = {}
    for factor in factors:
        if factor == 1:
            batches[factor] = tuple(compiled.circuits)
        else:
            batches[factor] = fold_cz_batch(compiled.circuits, factor)
    return batches


def _persist_factor_batches(
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    batches: Mapping[int, Sequence[QuantumCircuit]],
) -> None:
    for factor, circuits in batches.items():
        filename = f"compiled-factor-{factor}.qpy"
        digest = store.write_circuits(run, circuits, filename)
        document["circuits"]["factors"][str(factor)] = {
            "artifact": filename,
            "sha256": digest,
            "circuit_count": len(circuits),
        }
        document["jobs"][str(factor)] = {
            "factor": factor,
            "job_id": None,
            "status": "pending",
            "circuit_count": len(circuits),
            "shots": document["spec"]["shots"],
        }
    _write_state(store, run, document)


def _load_circuit_checkpoint(
    store: ExperimentStore,
    run: Path,
    record: Mapping[str, Any],
    description: str,
) -> tuple[Any, ...]:
    artifact = record.get("artifact")
    expected_hash = record.get("sha256")
    expected_count = record.get("circuit_count")
    if (
        not isinstance(artifact, str)
        or not isinstance(expected_hash, str)
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ExperimentPersistenceError(f"{description} circuit manifest is invalid")
    path = _manifest_path(run, artifact, description)
    if _sha256(path) != expected_hash:
        raise ExperimentPersistenceError(f"{description} circuit artifact hash mismatch")
    circuits = tuple(store.read_circuits(run, artifact))
    if len(circuits) != expected_count:
        raise ExperimentPersistenceError(
            f"{description} circuit count does not match persisted QPY"
        )
    if _sha256(path) != expected_hash:
        raise ExperimentPersistenceError(f"{description} circuit artifact hash mismatch")
    return circuits


def _factor_checkpoint(
    store: ExperimentStore,
    run: Path,
    document: Mapping[str, Any],
    factor: int,
) -> tuple[Any, ...]:
    try:
        record = document["circuits"]["factors"][str(factor)]
    except (KeyError, TypeError):
        raise ExperimentPersistenceError("factor circuit manifest is invalid") from None
    if not isinstance(record, Mapping):
        raise ExperimentPersistenceError("factor circuit manifest is invalid")
    return _load_circuit_checkpoint(store, run, record, f"factor {factor}")


def _validate_submitted(
    submitted: Any,
    identity: BackendIdentity,
    circuit_count: int,
    shots: int,
) -> SubmittedJob:
    if not isinstance(submitted, SubmittedJob):
        raise BackendCompatibilityError("adapter submit must return SubmittedJob")
    if submitted.target_identity != identity:
        raise BackendCompatibilityError("submitted job target does not match resolved backend")
    if submitted.circuit_count != circuit_count:
        raise BackendCompatibilityError("submitted job circuit count does not match submitted batch")
    if submitted.shots != shots:
        raise BackendCompatibilityError("submitted job shots do not match requested shots")
    _validate_persisted_strings(submitted.to_safe_dict(), description="submitted job metadata")
    return submitted


def _validate_result(
    result: Any,
    submitted: SubmittedJob,
    identity: BackendIdentity,
    circuit_count: int,
    shots: int,
) -> ExecutionResult:
    if not isinstance(result, ExecutionResult):
        raise BackendCompatibilityError("adapter result must return ExecutionResult")
    if result.target_identity != identity or result.job_id != submitted.job_id:
        raise BackendCompatibilityError("execution result does not match submitted backend job")
    if len(result.counts) != circuit_count:
        raise BackendCompatibilityError("execution result order/count does not match circuit batch")
    if any(sum(counts.values()) != shots for counts in result.counts):
        raise BackendCompatibilityError("execution result counts do not sum to requested shots")
    _validate_persisted_strings(result.to_safe_dict(), description="execution result metadata")
    return result


def _submit_once(
    adapter: Any,
    circuits: tuple[Any, ...],
    shots: int,
    run_options: Mapping[str, Any] | None,
    identity: BackendIdentity,
    capabilities: BackendCapabilities,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    job_key: str,
    clock: Callable[[], datetime],
) -> SubmittedJob:
    _require_durable_remote_jobs(capabilities)
    pessimistic_checkpoint = not capabilities.local
    if pessimistic_checkpoint:
        document["jobs"][job_key].update(
            {"job_id": None, "status": "submission_unknown"}
        )
        document["failure"] = None
        _transition(store, run, document, ExperimentStatus.SUBMISSION_UNKNOWN, clock)
    try:
        submitted = adapter.submit(circuits, shots, run_options)
    except BaseException as error:
        document["failure"] = {
            "stage": "submission",
            "exception_type": type(error).__name__,
            "message": "job submission failed; provider details omitted",
            "attempt": 1,
            "timestamp": _timestamp(clock),
        }
        if pessimistic_checkpoint:
            document["timestamps"]["updated"] = document["failure"]["timestamp"]
            _write_state(store, run, document)
        else:
            _transition(store, run, document, ExperimentStatus.SUBMISSION_UNKNOWN, clock)
        try:
            setattr(error, "__qoq_artifact_dir__", run)
        except Exception:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        sanitized = JobSubmissionError(f"job submission failed ({type(error).__name__})")
        setattr(sanitized, "__qoq_artifact_dir__", run)
        raise sanitized from None
    if not isinstance(submitted, SubmittedJob):
        error = BackendCompatibilityError("adapter submit must return SubmittedJob")
        document["failure"] = _failure(error, "submission", clock, attempt=1)
        if pessimistic_checkpoint:
            document["timestamps"]["updated"] = document["failure"]["timestamp"]
            _write_state(store, run, document)
        else:
            _transition(store, run, document, ExperimentStatus.SUBMISSION_UNKNOWN, clock)
        raise error
    _validate_persisted_strings(submitted.job_id, description="submitted job ID")
    job = document["jobs"][job_key]
    job.update({"job_id": submitted.job_id, "status": "submitted"})
    document["job_ids"].append(submitted.job_id)
    _transition(store, run, document, ExperimentStatus.SUBMITTED, clock)
    try:
        submitted = _validate_submitted(submitted, identity, len(circuits), shots)
    except BackendCompatibilityError:
        job["status"] = "incompatible"
        _write_state(store, run, document)
        raise
    return submitted


def _retrieve(
    adapter: Any,
    submitted: SubmittedJob,
    identity: BackendIdentity,
    circuit_count: int,
    shots: int,
    timeout: float | None,
    retry: RetryConfig,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    sleep: Callable[[float], None],
    clock: Callable[[], datetime],
) -> ExecutionResult:
    result = _retry(
        "result",
        lambda: adapter.result(submitted, timeout=timeout),
        retry,
        (JobResultError,),
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    return _validate_result(result, submitted, identity, circuit_count, shots)


def _execute_measurement_factor(
    factor: int,
    settings: tuple[Any, ...],
    adapter: Any,
    identity: BackendIdentity,
    capabilities: BackendCapabilities,
    spec: ExperimentSpec,
    timeout: float | None,
    run_options: Mapping[str, Any] | None,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    sleep: Callable[[float], None],
    clock: Callable[[], datetime],
    submitted: SubmittedJob | None = None,
) -> None:
    preflight_circuits = _factor_checkpoint(store, run, document, factor)
    if len(settings) != len(preflight_circuits):
        raise BackendCompatibilityError("persisted setting order does not match factor circuit batch")
    _retry(
        f"availability-factor-{factor}",
        lambda: _check_availability(adapter),
        spec.retry,
        (BackendUnavailableError,),
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    _retry(
        f"preflight-factor-{factor}",
        lambda: _preflight(adapter, preflight_circuits, spec.shots),
        spec.retry,
        (BackendUnavailableError,),
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    if submitted is None:
        submission_circuits = _factor_checkpoint(store, run, document, factor)
        submitted = _submit_once(
            adapter,
            submission_circuits,
            spec.shots,
            run_options,
            identity,
            capabilities,
            store=store,
            run=run,
            document=document,
            job_key=str(factor),
            clock=clock,
        )
    result = _retrieve(
        adapter,
        submitted,
        identity,
        len(preflight_circuits),
        spec.shots,
        timeout,
        spec.retry,
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    _transition(store, run, document, ExperimentStatus.RUNNING, clock)
    counts = OrderedDict((setting, dict(value)) for setting, value in zip(settings, result.counts, strict=True))
    counts_path = store.write_counts(run, factor, counts)
    document["counts"][str(factor)] = {
        "artifact": counts_path.name,
        "sha256": _sha256(counts_path),
        "settings": list(settings),
    }
    document["jobs"][str(factor)].update(
        {"status": "completed", "result_status": result.status, "timing": result.timing}
    )
    _write_state(store, run, document)


def _identity_key(identity: BackendIdentity) -> str:
    return f"{identity.kind}:{identity.name}"


def _calibration_id(identity: BackendIdentity) -> str:
    value = identity.metadata.get("calibration_set_id")
    if isinstance(value, str) and value:
        return value
    return identity.version or identity.name


def _physical_qubit_mapping(circuits: Sequence[QuantumCircuit]) -> tuple[int, ...]:
    expected: tuple[int, ...] | None = None
    for circuit in circuits:
        measured: dict[int, int] = {}
        for instruction in circuit.data:
            if instruction.operation.name != "measure" or len(instruction.qubits) != 1 or len(instruction.clbits) != 1:
                continue
            classical = circuit.find_bit(instruction.clbits[0]).index
            physical = circuit.find_bit(instruction.qubits[0]).index
            measured[classical] = physical
        if not measured or tuple(sorted(measured)) != tuple(range(len(measured))):
            raise BackendCompatibilityError("compiled measurements require contiguous classical-bit mapping")
        mapping = tuple(measured[index] for index in range(len(measured)))
        if len(set(mapping)) != len(mapping):
            raise BackendCompatibilityError("compiled measurement maps multiple bits to one physical qubit")
        if expected is None:
            expected = mapping
        elif mapping != expected:
            raise BackendCompatibilityError("compiled measurement circuits have inconsistent physical mapping")
    if expected is None:
        raise BackendCompatibilityError("compiled measurement circuits contain no measurements")
    return expected


def _calibration_from_safe_dict(data: Mapping[str, Any]) -> ReadoutCalibration:
    timestamp = data.get("timestamp")
    if not isinstance(timestamp, str):
        raise ExperimentPersistenceError("readout calibration timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ExperimentPersistenceError("readout calibration timestamp is invalid") from error
    return ReadoutCalibration(
        backend_identity=data["backend_identity"],
        calibration_id=data["calibration_id"],
        qubit_mapping=tuple(data["qubit_mapping"]),
        timestamp=parsed,
        shots=data["shots"],
        raw_counts=tuple(data["raw_counts"]),
        assignment_matrices=tuple(data["assignment_matrices"]),
    )


def _require_readout_dependency(readout_strategy: Any | None) -> None:
    if readout_strategy is None and importlib.util.find_spec("mthree") is None:
        raise OptionalDependencyError(
            "readout mitigation requires mthree; install with pip install -e .[mitigation]"
        )


def _obtain_calibration(
    adapter: Any,
    identity: BackendIdentity,
    capabilities: BackendCapabilities,
    compiled_circuits: tuple[QuantumCircuit, ...],
    spec: ExperimentSpec,
    reusable: ReadoutCalibration | None,
    timeout: float | None,
    run_options: Mapping[str, Any] | None,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    sleep: Callable[[float], None],
    clock: Callable[[], datetime],
) -> ReadoutCalibration:
    mapping = _physical_qubit_mapping(compiled_circuits)
    now = clock()
    identity_key = _identity_key(identity)
    calibration_id = _calibration_id(identity)
    if reusable is not None and not spec.mitigation.force_recalibration:
        if not calibration_cache_is_valid(
            reusable,
            backend_identity=identity_key,
            calibration_id=calibration_id,
            qubit_mapping=mapping,
            now=now,
            max_age_hours=spec.mitigation.readout_max_age_hours,
        ):
            raise BackendCompatibilityError("injected readout calibration is not valid for this target")
        path = store.write_json(run, "readout-calibration.json", reusable.to_safe_dict())
        document["calibration"] = {
            "evidence_artifact": path.name,
            "evidence_sha256": _sha256(path),
            "status": "reused",
            "qubit_mapping": list(mapping),
        }
        _write_state(store, run, document)
        return reusable

    source = build_readout_calibration_circuits(mapping)
    compiled = adapter.compile(source, spec.transpilation)
    if not isinstance(compiled, CompiledBatch) or compiled.target_identity != identity:
        raise BackendCompatibilityError("readout calibration compile target does not match backend")
    filename = "readout-calibration-circuits.qpy"
    digest = store.write_circuits(run, compiled.circuits, filename)
    document["calibration"] = {
        "circuits_artifact": filename,
        "circuits_sha256": digest,
        "evidence_artifact": None,
        "evidence_sha256": None,
        "status": "compiled",
        "qubit_mapping": list(mapping),
        "circuit_count": len(compiled.circuits),
        "shots": spec.shots,
    }
    document["jobs"]["calibration"] = {
        "job_id": None,
        "status": "pending",
        "circuit_count": len(compiled.circuits),
        "shots": spec.shots,
    }
    _write_state(store, run, document)
    calibration_record = {
        "artifact": filename,
        "sha256": digest,
        "circuit_count": len(compiled.circuits),
    }
    preflight_circuits = _load_circuit_checkpoint(
        store, run, calibration_record, "readout calibration"
    )
    _retry(
        "preflight-calibration",
        lambda: _preflight(adapter, preflight_circuits, spec.shots),
        spec.retry,
        (BackendUnavailableError,),
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    submission_circuits = _load_circuit_checkpoint(
        store, run, calibration_record, "readout calibration"
    )
    submitted = _submit_once(
        adapter,
        submission_circuits,
        spec.shots,
        run_options,
        identity,
        capabilities,
        store=store,
        run=run,
        document=document,
        job_key="calibration",
        clock=clock,
    )
    document["calibration"]["status"] = "submitted"
    _write_state(store, run, document)
    result = _retrieve(
        adapter,
        submitted,
        identity,
        len(submission_circuits),
        spec.shots,
        timeout,
        spec.retry,
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    raw = tuple(dict(counts) for counts in result.counts)
    matrices = assignment_matrices_from_counts(mapping, raw, shots=spec.shots)
    calibration = ReadoutCalibration(
        backend_identity=identity_key,
        calibration_id=calibration_id,
        qubit_mapping=mapping,
        timestamp=clock(),
        shots=spec.shots,
        raw_counts=raw,
        assignment_matrices=matrices,
    )
    evidence_path = store.write_json(run, "readout-calibration.json", calibration.to_safe_dict())
    document["calibration"].update(
        {
            "evidence_artifact": evidence_path.name,
            "evidence_sha256": _sha256(evidence_path),
            "status": "completed",
        }
    )
    document["jobs"]["calibration"]["status"] = "completed"
    _write_state(store, run, document)
    return calibration


def _postprocess(
    spec: ExperimentSpec,
    factors: Sequence[int],
    calibration: ReadoutCalibration | None,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    clock: Callable[[], datetime],
    readout_strategy: Any | None,
    zne_strategy: Any | None,
    evaluator: Callable[[Any], complex] | None,
) -> ExperimentResult:
    _transition(store, run, document, ExperimentStatus.POSTPROCESSING, clock)
    metadata = store.read_json(run, document["postprocessing"]["artifact"])
    counts = OrderedDict((factor, store.read_counts(run, factor)) for factor in factors)
    inputs = BootstrapInputs(
        counts_by_factor=counts,
        terms=metadata["terms"],
        qutrit_bit_indices_by_setting=metadata["qutrit_bit_indices_by_setting"],
        decoding_kwargs=metadata["decoding_kwargs"],
        readout_calibration=calibration,
    )
    result = bootstrap_bell_results(
        inputs,
        spec.bootstrap,
        readout_strategy=readout_strategy,
        zne_strategy=zne_strategy,
        _evaluator=evaluator,
    )
    safe_result = result.to_safe_dict()
    result_path = store.write_json(run, "result.json", safe_result)
    document["result"] = safe_result
    document["result_artifact"] = {
        "artifact": result_path.name,
        "sha256": _sha256(result_path),
    }
    document["failure"] = None
    _transition(store, run, document, ExperimentStatus.COMPLETED, clock)
    return _result_from_document(run, document)


def _result_from_document(run: Path, document: Mapping[str, Any], *, force_failed: bool = False) -> ExperimentResult:
    status = ExperimentStatus.FAILED if force_failed else ExperimentStatus(document["status"])
    backend = document.get("backend") or {}
    if isinstance(backend, Mapping) and isinstance(backend.get("identity"), Mapping):
        backend = backend["identity"]
    values = document.get("result") or {}
    if force_failed:
        values = {"failure": document.get("failure")}
    return ExperimentResult(
        experiment_id=document["experiment_id"],
        status=status,
        artifact_dir=run,
        values=values,
        backend=backend,
        job_ids=tuple(document.get("job_ids", ())),
    )


def _persist_terminal_failure(
    error: BaseException,
    stage: str,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    clock: Callable[[], datetime],
) -> None:
    document["failure"] = _failure(error, stage, clock)
    _transition(store, run, document, ExperimentStatus.FAILED, clock)


def run_experiment(
    spec: ExperimentSpec,
    *,
    adapter: Any | None = None,
    repo_root: Path | str | None = None,
    timeout: float | None = None,
    run_options: Mapping[str, Any] | None = None,
    readout_calibration: ReadoutCalibration | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    _clock: Callable[[], datetime] = _utc_now,
    _readout_strategy: Any | None = None,
    _zne_strategy: Any | None = None,
    _evaluator: Callable[[Any], complex] | None = None,
) -> ExperimentResult:
    """Run one experiment with durable checkpoints before every remote wait."""

    if not isinstance(spec, ExperimentSpec):
        raise ExperimentValidationError("spec must be ExperimentSpec")
    _validate_execution_options(timeout, run_options)

    store = ExperimentStore(_output_root(spec, repo_root))
    run = store.create_run()
    document = _initial_document(spec, run.name, _clock)
    _write_state(store, run, document)
    stage = "validation"
    known_job_result_failure = False
    try:
        artifacts = load_basis_artifacts(spec.basis, spec.state, repo_root)
        prepared = prepare_measurements(artifacts)
        settings, _ = _persist_prepared(store, run, document, artifacts, prepared)
        _transition(store, run, document, ExperimentStatus.VALIDATED, _clock)

        stage = "backend-resolution"
        resolved_adapter = create_backend_adapter(spec.backend) if adapter is None else adapter
        identity, capabilities, adapter_metadata = _validate_adapter(resolved_adapter)
        _require_durable_remote_jobs(capabilities)
        availability = _retry(
            "availability",
            lambda: _check_availability(resolved_adapter),
            spec.retry,
            (BackendUnavailableError,),
            store=store,
            run=run,
            document=document,
            sleep=_sleep,
            clock=_clock,
        )
        backend_record = {
            "identity": identity.to_safe_dict(),
            "capabilities": capabilities.to_safe_dict(),
            "metadata": adapter_metadata,
            "availability": availability.to_safe_dict(),
        }
        _validate_persisted_strings(backend_record, description="backend metadata")
        document["backend"] = backend_record
        _write_state(store, run, document)
        if spec.mitigation.readout:
            _require_readout_dependency(_readout_strategy)

        stage = "compilation"
        compiled = resolved_adapter.compile(tuple(prepared.circuits), spec.transpilation)
        if not isinstance(compiled, CompiledBatch):
            raise BackendCompatibilityError("adapter compile must return CompiledBatch")
        if compiled.target_identity != identity:
            raise BackendCompatibilityError("compiled target does not match resolved backend")
        factors = (
            validate_zne_factors(spec.mitigation.zne_factors)
            if spec.mitigation.zne
            else (1,)
        )
        batches = _factor_batches(compiled, factors)
        _persist_factor_batches(store, run, document, batches)
        _transition(store, run, document, ExperimentStatus.COMPILED, _clock)

        calibration = None
        if spec.mitigation.readout:
            stage = "readout-calibration"
            calibration = _obtain_calibration(
                resolved_adapter,
                identity,
                capabilities,
                tuple(compiled.circuits),
                spec,
                readout_calibration,
                timeout,
                run_options,
                store=store,
                run=run,
                document=document,
                sleep=_sleep,
                clock=_clock,
            )

        stage = "execution"
        for factor in factors:
            try:
                _execute_measurement_factor(
                    factor,
                    settings,
                    resolved_adapter,
                    identity,
                    capabilities,
                    spec,
                    timeout,
                    run_options,
                    store=store,
                    run=run,
                    document=document,
                    sleep=_sleep,
                    clock=_clock,
                )
            except JobResultError:
                known_job_result_failure = document["jobs"][str(factor)]["job_id"] is not None
                if known_job_result_failure and document["status"] == ExperimentStatus.SUBMITTED.value:
                    _transition(store, run, document, ExperimentStatus.RUNNING, _clock)
                raise

        stage = "postprocessing"
        return _postprocess(
            spec,
            factors,
            calibration,
            store=store,
            run=run,
            document=document,
            clock=_clock,
            readout_strategy=_readout_strategy,
            zne_strategy=_zne_strategy,
            evaluator=_evaluator,
        )
    except (KeyboardInterrupt, SystemExit) as error:
        try:
            setattr(error, "__qoq_artifact_dir__", run)
        except Exception:
            pass
        raise
    except Exception as error:
        if isinstance(error, JobResultError):
            known_job_result_failure = known_job_result_failure or any(
                isinstance(job, Mapping)
                and job.get("job_id") is not None
                and job.get("status") != "completed"
                for job in document["jobs"].values()
            )
            if known_job_result_failure and document["status"] == ExperimentStatus.SUBMITTED.value:
                _transition(store, run, document, ExperimentStatus.RUNNING, _clock)
        if document["status"] != ExperimentStatus.SUBMISSION_UNKNOWN.value and not known_job_result_failure:
            _persist_terminal_failure(
                error,
                stage,
                store=store,
                run=run,
                document=document,
                clock=_clock,
            )
        try:
            setattr(error, "__qoq_artifact_dir__", run)
        except Exception:
            pass
        raise


def _open_run(experiment_dir: Path | str) -> tuple[ExperimentStore, Path, dict[str, Any]]:
    try:
        run = Path(experiment_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ExperimentPersistenceError("experiment_dir must identify an existing run directory") from error
    if not run.is_dir() or len(run.parents) < 2:
        raise ExperimentPersistenceError("experiment_dir must identify a run directory")
    store = ExperimentStore(run.parents[1])
    document = store.read_experiment(run)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentPersistenceError("unsupported experiment schema version")
    if document.get("experiment_id") != run.name:
        raise ExperimentPersistenceError("experiment ID does not match run directory")
    try:
        ExperimentStatus(document.get("status"))
    except (TypeError, ValueError) as error:
        raise ExperimentPersistenceError("experiment status is invalid") from error
    return store, run, document


def _verify_file(run: Path, record: Mapping[str, Any], description: str) -> None:
    artifact = record.get("artifact")
    expected = record.get("sha256")
    if not isinstance(artifact, str) or not isinstance(expected, str):
        raise ExperimentPersistenceError(f"{description} artifact record is invalid")
    path = _manifest_path(run, artifact, description)
    if _sha256(path) != expected:
        raise ExperimentPersistenceError(f"{description} artifact hash mismatch")


def _manifest_path(run: Path, artifact: Any, description: str) -> Path:
    if not isinstance(artifact, str):
        raise ExperimentPersistenceError(f"{description} artifact name is invalid")
    relative = Path(artifact)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ExperimentPersistenceError(f"{description} artifact name is unsafe")
    return run / relative


def _calibration_job_reference(
    document: Mapping[str, Any], *, migrate_legacy: bool = False
) -> tuple[str | None, bool]:
    calibration = document.get("calibration")
    if calibration is None:
        return None, False
    if not isinstance(calibration, Mapping):
        raise ExperimentPersistenceError("readout calibration manifest is invalid")
    jobs = document.get("jobs")
    if not isinstance(jobs, Mapping):
        raise ExperimentPersistenceError("experiment jobs manifest is invalid")
    job = jobs.get("calibration")
    if job is None:
        legacy = calibration.get("job_id")
        if legacy is None:
            return None, False
        raise ExperimentPersistenceError("calibration job ID has no canonical job record")
    if not isinstance(job, Mapping):
        raise ExperimentPersistenceError("canonical calibration job record is invalid")

    canonical = job.get("job_id")
    legacy = calibration.get("job_id")
    for value in (canonical, legacy):
        if value is not None and (not isinstance(value, str) or not value):
            raise ExperimentPersistenceError("calibration job ID is invalid")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ExperimentPersistenceError("calibration job ID records disagree")
    if canonical is not None:
        if legacy is None or not migrate_legacy:
            return canonical, False
        if not isinstance(document, dict) or not isinstance(calibration, dict):
            raise ExperimentPersistenceError("legacy calibration job ID cannot be migrated")
        calibration.pop("job_id")
        return canonical, True
    if legacy is None:
        return None, False
    if not migrate_legacy:
        return legacy, False
    if (
        not isinstance(document, dict)
        or not isinstance(calibration, dict)
        or not isinstance(jobs, dict)
        or not isinstance(job, dict)
    ):
        raise ExperimentPersistenceError("legacy calibration job ID cannot be migrated")
    job["job_id"] = legacy
    job["status"] = "submitted"
    calibration.pop("job_id")
    return legacy, True


def _verify_resume_artifacts(store: ExperimentStore, run: Path, document: Mapping[str, Any]) -> tuple[int, ...]:
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ExperimentPersistenceError("source artifact manifest is invalid")
    encoding_artifact = source.get("encoding_artifact")
    encoding_digest = source.get("encoding_sha256")
    if not isinstance(encoding_artifact, str) or not isinstance(encoding_digest, str):
        raise ExperimentPersistenceError("source encoding artifact manifest is invalid")
    encoding_path = _manifest_path(run, encoding_artifact, "source encoding")
    store.read_json(run, encoding_artifact)
    if _sha256(encoding_path) != encoding_digest:
        raise ExperimentPersistenceError("source encoding artifact hash mismatch")
    circuits = document.get("circuits")
    if not isinstance(circuits, Mapping):
        raise ExperimentPersistenceError("experiment circuit manifest is invalid")
    for name in ("source", "logical"):
        record = circuits.get(name)
        if not isinstance(record, Mapping):
            raise ExperimentPersistenceError(f"{name} circuit manifest is invalid")
        _verify_file(run, record, name)
        store.read_circuits(run, record["artifact"])
    factor_records = circuits.get("factors")
    if not isinstance(factor_records, Mapping) or not factor_records:
        raise ExperimentPersistenceError("factor circuit manifest is invalid")
    try:
        factors = tuple(sorted(int(value) for value in factor_records))
    except (TypeError, ValueError) as error:
        raise ExperimentPersistenceError("factor circuit manifest contains invalid factors") from error
    for factor in factors:
        record = factor_records.get(str(factor))
        if not isinstance(record, Mapping):
            raise ExperimentPersistenceError("factor circuit manifest is invalid")
        _verify_file(run, record, f"factor {factor}")
        batch = store.read_circuits(run, record["artifact"])
        if record.get("circuit_count") != len(batch):
            raise ExperimentPersistenceError("factor circuit count does not match persisted QPY")
    postprocessing = document.get("postprocessing")
    if not isinstance(postprocessing, Mapping):
        raise ExperimentPersistenceError("postprocessing manifest is invalid")
    _verify_file(run, postprocessing, "postprocessing")
    store.read_json(run, postprocessing["artifact"])
    counts_records = document.get("counts")
    if not isinstance(counts_records, Mapping):
        raise ExperimentPersistenceError("counts manifest is invalid")
    for factor_text, record in counts_records.items():
        if not isinstance(record, Mapping):
            raise ExperimentPersistenceError("counts manifest is invalid")
        _verify_file(run, record, f"counts factor {factor_text}")
        store.read_counts(run, int(factor_text))
    calibration = document.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, Mapping):
            raise ExperimentPersistenceError("readout calibration manifest is invalid")
        calibration_circuits = calibration.get("circuits_artifact")
        if calibration_circuits is not None:
            store.read_circuits(run, calibration_circuits)
            calibration_path = _manifest_path(run, calibration_circuits, "readout calibration circuit")
            if _sha256(calibration_path) != calibration.get("circuits_sha256"):
                raise ExperimentPersistenceError("readout calibration circuit hash mismatch")
        calibration_evidence = calibration.get("evidence_artifact")
        if calibration_evidence is not None:
            store.read_json(run, calibration_evidence)
            evidence_path = _manifest_path(run, calibration_evidence, "readout calibration evidence")
            if _sha256(evidence_path) != calibration.get("evidence_sha256"):
                raise ExperimentPersistenceError("readout calibration evidence hash mismatch")
        _calibration_job_reference(document)
    result_record = document.get("result_artifact")
    if result_record is not None:
        if not isinstance(result_record, Mapping):
            raise ExperimentPersistenceError("result manifest is invalid")
        _verify_file(run, result_record, "result")
        if store.read_json(run, result_record["artifact"]) != document.get("result"):
            raise ExperimentPersistenceError("result artifact does not match experiment state")
    return factors


def _resume_spec(document: Mapping[str, Any], supplied: ExperimentSpec | None) -> ExperimentSpec:
    safe = document.get("spec")
    if not isinstance(safe, Mapping):
        raise ExperimentPersistenceError("persisted experiment spec is invalid")
    if supplied is not None:
        if not isinstance(supplied, ExperimentSpec) or supplied.to_safe_dict() != safe:
            raise ExperimentValidationError("injected spec does not match persisted experiment spec")
        return supplied
    try:
        return ExperimentSpec.from_safe_dict(safe)
    except ExperimentValidationError as error:
        raise ExperimentValidationError(
            "custom or noisy experiment resume requires matching spec context injection"
        ) from error


def _resume_calibration(
    adapter: Any,
    identity: BackendIdentity,
    capabilities: BackendCapabilities,
    spec: ExperimentSpec,
    timeout: float | None,
    run_options: Mapping[str, Any] | None,
    *,
    store: ExperimentStore,
    run: Path,
    document: dict[str, Any],
    sleep: Callable[[float], None],
    clock: Callable[[], datetime],
) -> ReadoutCalibration | None:
    record = document.get("calibration")
    if not spec.mitigation.readout:
        return None
    if not isinstance(record, Mapping):
        raise ExperimentPersistenceError("readout calibration manifest is missing")
    evidence = record.get("evidence_artifact")
    if evidence:
        expected = record.get("evidence_sha256")
        if _sha256(run / evidence) != expected:
            raise ExperimentPersistenceError("readout calibration evidence hash mismatch")
        return _calibration_from_safe_dict(store.read_json(run, evidence))
    circuit_artifact = record.get("circuits_artifact")
    if not isinstance(circuit_artifact, str):
        raise ExperimentPersistenceError("readout calibration circuits are missing")
    calibration_manifest = {
        "artifact": circuit_artifact,
        "sha256": record.get("circuits_sha256"),
        "circuit_count": record.get("circuit_count"),
    }
    circuits = _load_circuit_checkpoint(
        store, run, calibration_manifest, "readout calibration"
    )
    job_id, _ = _calibration_job_reference(document)
    if job_id:
        submitted = adapter.restore_job(job_id, circuit_count=len(circuits), shots=spec.shots)
        submitted = _validate_submitted(submitted, identity, len(circuits), spec.shots)
    else:
        _retry(
            "preflight-calibration-resume",
            lambda: _preflight(adapter, circuits, spec.shots),
            spec.retry,
            (BackendUnavailableError,),
            store=store,
            run=run,
            document=document,
            sleep=sleep,
            clock=clock,
        )
        submission_circuits = _load_circuit_checkpoint(
            store, run, calibration_manifest, "readout calibration"
        )
        submitted = _submit_once(
            adapter,
            submission_circuits,
            spec.shots,
            run_options,
            identity,
            capabilities,
            store=store,
            run=run,
            document=document,
            job_key="calibration",
            clock=clock,
        )
    result = _retrieve(
        adapter,
        submitted,
        identity,
        len(circuits),
        spec.shots,
        timeout,
        spec.retry,
        store=store,
        run=run,
        document=document,
        sleep=sleep,
        clock=clock,
    )
    mapping = tuple(record["qubit_mapping"])
    raw = tuple(dict(item) for item in result.counts)
    calibration = ReadoutCalibration(
        backend_identity=_identity_key(identity),
        calibration_id=_calibration_id(identity),
        qubit_mapping=mapping,
        timestamp=clock(),
        shots=spec.shots,
        raw_counts=raw,
        assignment_matrices=assignment_matrices_from_counts(mapping, raw, shots=spec.shots),
    )
    path = store.write_json(run, "readout-calibration.json", calibration.to_safe_dict())
    document["calibration"].update(
        {"evidence_artifact": path.name, "evidence_sha256": _sha256(path), "status": "completed"}
    )
    document["jobs"]["calibration"]["status"] = "completed"
    _write_state(store, run, document)
    return calibration


def resume_experiment(
    experiment_dir: Path | str,
    *,
    adapter: Any | None = None,
    timeout: float | None = None,
    run_options: Mapping[str, Any] | None = None,
    spec: ExperimentSpec | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    _clock: Callable[[], datetime] = _utc_now,
    _readout_strategy: Any | None = None,
    _zne_strategy: Any | None = None,
    _evaluator: Callable[[Any], complex] | None = None,
) -> ExperimentResult:
    """Continue an existing run without resubmitting any known backend job."""

    _validate_execution_options(timeout, run_options)
    store, run, document = _open_run(experiment_dir)
    factors = _verify_resume_artifacts(store, run, document)
    status = ExperimentStatus(document["status"])
    if status is ExperimentStatus.COMPLETED:
        return _result_from_document(run, document)
    if status is ExperimentStatus.SUBMISSION_UNKNOWN:
        raise JobSubmissionError("cannot safely resume an experiment with unknown submission outcome")
    resumed_spec = _resume_spec(document, spec)
    if resumed_spec.mitigation.readout:
        _, migrated = _calibration_job_reference(document, migrate_legacy=True)
        if migrated:
            _write_state(store, run, document)
    resolved_adapter = create_backend_adapter(resumed_spec.backend) if adapter is None else adapter
    identity, capabilities, _ = _validate_adapter(resolved_adapter)
    persisted_backend = document.get("backend")
    if not isinstance(persisted_backend, Mapping) or persisted_backend.get("identity") != identity.to_safe_dict():
        error = BackendCompatibilityError("resume backend identity does not match persisted target")
        _persist_terminal_failure(
            error,
            "resume",
            store=store,
            run=run,
            document=document,
            clock=_clock,
        )
        raise error
    try:
        _require_durable_remote_jobs(capabilities)
    except BackendCompatibilityError as error:
        _persist_terminal_failure(
            error,
            "resume",
            store=store,
            run=run,
            document=document,
            clock=_clock,
        )
        raise
    if resumed_spec.mitigation.readout:
        _require_readout_dependency(_readout_strategy)

    try:
        calibration = _resume_calibration(
            resolved_adapter,
            identity,
            capabilities,
            resumed_spec,
            timeout,
            run_options,
            store=store,
            run=run,
            document=document,
            sleep=_sleep,
            clock=_clock,
        )
        metadata = store.read_json(run, document["postprocessing"]["artifact"])
        settings = tuple(metadata["setting_by_circuit_index"])
        for factor in factors:
            if str(factor) in document["counts"]:
                continue
            circuits = _factor_checkpoint(store, run, document, factor)
            job = document["jobs"][str(factor)]
            submitted = None
            if job.get("job_id"):
                submitted = resolved_adapter.restore_job(
                    job["job_id"], circuit_count=len(circuits), shots=resumed_spec.shots
                )
                submitted = _validate_submitted(
                    submitted, identity, len(circuits), resumed_spec.shots
                )
            _execute_measurement_factor(
                factor,
                settings,
                resolved_adapter,
                identity,
                capabilities,
                resumed_spec,
                timeout,
                run_options,
                store=store,
                run=run,
                document=document,
                sleep=_sleep,
                clock=_clock,
                submitted=submitted,
            )
        return _postprocess(
            resumed_spec,
            factors,
            calibration,
            store=store,
            run=run,
            document=document,
            clock=_clock,
            readout_strategy=_readout_strategy,
            zne_strategy=_zne_strategy,
            evaluator=_evaluator,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except JobResultError:
        if document["status"] == ExperimentStatus.SUBMITTED.value:
            _transition(store, run, document, ExperimentStatus.RUNNING, _clock)
        raise
    except Exception as error:
        if document["status"] == ExperimentStatus.SUBMISSION_UNKNOWN.value:
            raise
        _persist_terminal_failure(
            error,
            "resume",
            store=store,
            run=run,
            document=document,
            clock=_clock,
        )
        raise


def run_experiments(
    specs: Iterable[ExperimentSpec],
    **kwargs: Any,
) -> tuple[ExperimentResult, ...]:
    """Run independent specs in order and retain failures as result entries."""

    if isinstance(specs, (str, bytes, Mapping)):
        raise ExperimentValidationError("specs must be an iterable of ExperimentSpec values")
    try:
        iterator = iter(specs)
    except TypeError as error:
        raise ExperimentValidationError("specs must be iterable") from error
    results: list[ExperimentResult] = []
    for spec in iterator:
        if not isinstance(spec, ExperimentSpec):
            raise ExperimentValidationError("every batch item must be ExperimentSpec")
        try:
            results.append(run_experiment(spec, **kwargs))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            run = getattr(error, "__qoq_artifact_dir__", None)
            if run is None:
                raise
            store = ExperimentStore(Path(run).parents[1])
            document = store.read_experiment(run)
            results.append(_result_from_document(Path(run), document, force_failed=True))
    return tuple(results)


__all__ = ["resume_experiment", "run_experiment", "run_experiments"]
