"""Durable orchestration for Bell experiments.

The runner intentionally persists only reconstructible, provider-safe values.
Backend objects, submitted handles, callbacks, run options, and exception
tracebacks never cross the artifact boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
from typing import Any

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
from .backends.base import _safe_identifier
from .errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentPersistenceError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from .execution import expected_backend_identity_kind
from .manifest import MANIFEST_SCHEMA_VERSION, RunManifest
from .mitigation import (
    ReadoutCalibration,
    TwirledBatch,
    assignment_matrices_from_counts,
    build_readout_calibration_circuits,
    calibration_cache_is_valid,
    fold_cz_batch,
    twirl_iqm_circuits,
    validate_zne_factors,
)
from .models import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    IQMHardware,
    RetryConfig,
    TranspilationConfig,
    _normalize_experiment_spec_dict,
)
from .preparation import prepare_measurements
from .safety import (
    unsafe_persisted_text as _unsafe_persisted_text,
    validate_persisted_strings,
)
from .store import ExperimentStore
from .uncertainty import BootstrapInputs, bootstrap_bell_results
from .workload_metrics import (
    WorkloadMetrics,
    choose_workload_ranking_basis,
    summarize_compiled_workload,
    workload_rank_key,
)


_PROVIDER_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")
_DIRECT_POSTPROCESSING_CHECKPOINT_VERSION = 1
_CHECKPOINT_CODEC_KIND = "kind"


@dataclass(frozen=True)
class _CompiledWorkloadCandidate:
    layout: tuple[int, ...]
    seed: int
    batch: CompiledBatch
    metrics: WorkloadMetrics


@dataclass(frozen=True)
class _CompiledWorkloadSelection:
    batch: CompiledBatch
    physical_mappings: tuple[tuple[int, ...], ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _DirectPostprocessingCheckpoint:
    spec: ExperimentSpec
    identity: BackendIdentity
    job_ids: tuple[str, ...]
    factors: tuple[int, ...]
    calibration: ReadoutCalibration | None
    inputs: BootstrapInputs
    schema_fragments: Mapping[str, Any]


def _safe_provider_exception_type(error: BaseException) -> str | None:
    try:
        value = getattr(error, "provider_exception_type", None)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return None
    if isinstance(value, str) and _PROVIDER_EXCEPTION_TYPE.fullmatch(value):
        return value
    return None


def _provider_failure_detail(error: BaseException) -> tuple[str, str | None]:
    error_type = type(error).__name__
    detail = (
        error_type
        if _PROVIDER_EXCEPTION_TYPE.fullmatch(error_type)
        else "provider error"
    )
    provider_exception_type = _safe_provider_exception_type(error)
    if provider_exception_type is not None:
        detail = f"{detail}; provider exception: {provider_exception_type}"
    return detail, provider_exception_type


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ExperimentValidationError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


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
    validate_persisted_strings(
        value,
        description=description,
        error_type=BackendCompatibilityError,
        active=active,
    )


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
    document = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "spec": spec.to_safe_dict(),
        "status": ExperimentStatus.CREATED.value,
        "timestamps": {"created": created, "updated": created},
        "status_history": [{"status": ExperimentStatus.CREATED.value, "timestamp": created}],
        "attempts": [],
        "backend": None,
        "jobs": {},
        "job_ids": [],
        "source": None,
        "counts": {},
        "circuits": {"source": None, "logical": None, "factors": {}},
        "postprocessing": None,
        "calibration": None,
        "result": None,
        "result_artifact": None,
        "failure": None,
    }
    return document


def _write_state(store: ExperimentStore, run: Path, document: Mapping[str, Any]) -> None:
    try:
        manifest = RunManifest.from_safe_dict(document)
    except ExperimentPersistenceError:
        raise
    except ExperimentValidationError:
        raise ExperimentPersistenceError(
            "experiment manifest is invalid"
        ) from None
    store.write_experiment(run, manifest.to_safe_dict())


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


def _validate_adapter_target(
    spec: ExperimentSpec,
    identity: BackendIdentity,
) -> None:
    backend_config = spec.backend.to_safe_dict()
    backend_kind = backend_config.get("kind")
    try:
        expected_kind = expected_backend_identity_kind(backend_kind)
    except ExperimentValidationError:
        raise BackendCompatibilityError(
            "configured backend kind is unsupported"
        ) from None
    name_field = {
        "iqm_hardware": "device",
        "custom": "identity",
        "noisy_simulator": "identity",
    }.get(backend_kind)
    expected_name = backend_config.get(name_field) if name_field is not None else None
    if identity.kind != expected_kind or (
        expected_name is not None and identity.name != expected_name
    ):
        raise BackendCompatibilityError(
            "resolved adapter identity does not match configured backend"
        )


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


def _execute_batch(
    adapter: Any,
    identity: BackendIdentity,
    circuits: Sequence[QuantumCircuit],
    shots: int,
    *,
    timeout: float | None,
    run_options: Mapping[str, Any] | None,
    retry: RetryConfig,
    sleep: Callable[[float], None],
) -> tuple[SubmittedJob, ExecutionResult]:
    batch = tuple(circuits)
    try:
        submitted = adapter.submit(batch, shots, run_options)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        detail, provider_exception_type = _provider_failure_detail(error)
        sanitized = JobSubmissionError(f"job submission failed ({detail})")
        if provider_exception_type is not None:
            setattr(sanitized, "provider_exception_type", provider_exception_type)
        raise sanitized from None
    if not isinstance(submitted, SubmittedJob):
        raise BackendCompatibilityError("adapter submit must return SubmittedJob")
    if submitted.target_identity != identity:
        raise BackendCompatibilityError(
            "submitted job target does not match resolved backend"
        )
    if submitted.circuit_count != len(batch):
        raise BackendCompatibilityError(
            "submitted job circuit count does not match submitted batch"
        )
    if submitted.shots != shots:
        raise BackendCompatibilityError(
            "submitted job shots do not match requested shots"
        )
    result = None
    for attempt in range(retry.max_attempts):
        try:
            result = adapter.result(submitted, timeout=timeout)
            break
        except JobResultError as error:
            if attempt + 1 < retry.max_attempts:
                delay = min(
                    retry.initial_delay * (retry.multiplier ** attempt),
                    retry.max_delay,
                )
                sleep(delay)
                continue
            detail, provider_exception_type = _provider_failure_detail(error)
            sanitized = JobResultError(f"job result retrieval failed ({detail})")
            if provider_exception_type is not None:
                setattr(sanitized, "provider_exception_type", provider_exception_type)
            raise sanitized from None
        except (BackendCompatibilityError, MemoryError):
            raise
        except Exception as error:
            detail, provider_exception_type = _provider_failure_detail(error)
            sanitized = JobResultError(f"job result retrieval failed ({detail})")
            if provider_exception_type is not None:
                setattr(sanitized, "provider_exception_type", provider_exception_type)
            raise sanitized from None
    if not isinstance(result, ExecutionResult):
        raise BackendCompatibilityError("adapter result must return ExecutionResult")
    if result.target_identity != identity:
        raise BackendCompatibilityError(
            "execution result target does not match resolved backend"
        )
    if result.job_id != submitted.job_id:
        raise BackendCompatibilityError("execution result does not match submitted job")
    if len(result.counts) != len(batch):
        raise BackendCompatibilityError(
            "execution result count does not match circuit batch"
        )
    if any(sum(counts.values()) != shots for counts in result.counts):
        raise BackendCompatibilityError(
            "execution result counts do not sum to requested shots"
        )
    return submitted, result


def _compile_with_adapter(
    adapter: Any,
    circuits: Sequence[QuantumCircuit],
    config: Any,
    *,
    physical: bool = False,
) -> Any:
    operation = "physical compilation" if physical else "compilation"
    try:
        compiler = (
            getattr(adapter, "compile_physical", None)
            if physical
            else getattr(adapter, "compile", None)
        )
        uses_generic_physical_compile = not callable(compiler) and physical
        if uses_generic_physical_compile:
            compiler = getattr(adapter, "compile", None)
        if not callable(compiler):
            raise TypeError("adapter compile operation is not callable")
        if uses_generic_physical_compile:
            widths = {circuit.num_qubits for circuit in circuits}
            if len(widths) != 1:
                raise TypeError("physical calibration circuits must share one width")
            width = next(iter(widths))
            if not isinstance(config, TranspilationConfig):
                raise TypeError("physical compilation requires TranspilationConfig")
            config = replace(config, initial_layout=tuple(range(width)))
        return compiler(circuits, config)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except BaseException as error:
        detail, provider_exception_type = _provider_failure_detail(error)
        sanitized = BackendCompatibilityError(
            f"adapter {operation} failed ({detail})"
        )
        if provider_exception_type is not None:
            setattr(sanitized, "provider_exception_type", provider_exception_type)
        raise sanitized from None


def _counts_by_setting(
    settings: Sequence[Any], result: ExecutionResult
) -> OrderedDict[Any, dict[str, int]]:
    if len(settings) != len(result.counts):
        raise BackendCompatibilityError("setting order does not match result batch")
    return OrderedDict(
        (setting, dict(counts))
        for setting, counts in zip(settings, result.counts, strict=True)
    )


def _validate_twirling_spec(spec: ExperimentSpec) -> None:
    mitigation = spec.mitigation
    if not mitigation.circuit_twirling:
        return
    if not isinstance(spec.backend, IQMHardware):
        raise ExperimentValidationError(
            "circuit twirling requires an IQMHardware backend"
        )
    if spec.shots % mitigation.twirling_instances != 0:
        raise ExperimentValidationError(
            "shots must be divisible by twirling_instances"
        )


def _validated_twirled_batch(
    value: Any,
    *,
    original_count: int,
    instances: int,
    seed: int | None,
) -> TwirledBatch:
    if not isinstance(value, TwirledBatch):
        raise BackendCompatibilityError(
            "IQM twirling transform must return TwirledBatch"
        )
    expected_originals = tuple(
        original_index
        for original_index in range(original_count)
        for _ in range(instances)
    )
    expected_instances = tuple(range(instances)) * original_count
    if value.original_indices != expected_originals:
        raise BackendCompatibilityError(
            "twirled circuit original-setting order is invalid"
        )
    if value.instance_indices != expected_instances:
        raise BackendCompatibilityError(
            "twirled circuit instance order is invalid"
        )
    expected_metadata = {
        "provider": "iqm-error-reduction-tools",
        "method": "circuit_twirling",
        "readout_strategy": "NONE",
        "instances_per_circuit": instances,
        "seed": seed,
    }
    if dict(value.metadata) != expected_metadata:
        raise BackendCompatibilityError("twirling metadata is invalid")
    return value


def _aggregate_twirled_counts(
    settings: Sequence[Any],
    result: ExecutionResult,
    twirled: TwirledBatch,
    *,
    total_shots: int,
) -> OrderedDict[Any, dict[str, int]]:
    if len(result.counts) != len(twirled.circuits):
        raise BackendCompatibilityError(
            "twirled result count does not match randomized circuit batch"
        )
    aggregated: list[dict[str, int]] = [{} for _ in settings]
    for original_index, counts in zip(
        twirled.original_indices, result.counts, strict=True
    ):
        if original_index >= len(settings):
            raise BackendCompatibilityError(
                "twirled result references an unknown measurement setting"
            )
        target = aggregated[original_index]
        for outcome, count in counts.items():
            target[outcome] = target.get(outcome, 0) + count
    if any(sum(counts.values()) != total_shots for counts in aggregated):
        raise BackendCompatibilityError(
            "aggregated twirling counts do not sum to requested shots"
        )
    return OrderedDict(
        (setting, counts)
        for setting, counts in zip(settings, aggregated, strict=True)
    )


def _run_readout_calibration(
    adapter: Any,
    identity: BackendIdentity,
    physical_qubit_mappings: Sequence[Sequence[int]],
    spec: ExperimentSpec,
    *,
    timeout: float | None,
    run_options: Mapping[str, Any] | None,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> tuple[ReadoutCalibration, SubmittedJob]:
    physical_qubits = tuple(
        sorted(
            {
                physical
                for mapping in physical_qubit_mappings
                for physical in mapping
            }
        )
    )
    source = build_readout_calibration_circuits(physical_qubits)
    compiled = _compile_with_adapter(
        adapter,
        source,
        spec.transpilation,
        physical=True,
    )
    if not isinstance(compiled, CompiledBatch):
        raise BackendCompatibilityError(
            "adapter compile_physical must return CompiledBatch"
        )
    if compiled.target_identity != identity:
        raise BackendCompatibilityError(
            "compiled readout calibration target does not match resolved backend"
        )
    _validate_physical_calibration_compile(source, compiled.circuits)
    submitted, execution = _execute_batch(
        adapter,
        identity,
        compiled.circuits,
        spec.shots,
        timeout=timeout,
        run_options=run_options,
        retry=spec.retry,
        sleep=sleep,
    )
    raw = tuple(dict(counts) for counts in execution.counts)
    calibration = ReadoutCalibration(
        backend_identity=_identity_key(identity),
        calibration_id=_calibration_id(identity),
        qubit_mapping=physical_qubits,
        timestamp=clock(),
        shots=spec.shots,
        raw_counts=raw,
        assignment_matrices=assignment_matrices_from_counts(
            physical_qubits,
            raw,
            shots=spec.shots,
        ),
    )
    return calibration, submitted


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
        provider_exception_type = _safe_provider_exception_type(error)
        document["failure"] = {
            "stage": "submission",
            "exception_type": type(error).__name__,
            "message": "job submission failed; provider details omitted",
            "attempt": 1,
            "timestamp": _timestamp(clock),
        }
        if provider_exception_type is not None:
            document["failure"]["provider_exception_type"] = provider_exception_type
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
        detail = type(error).__name__
        if provider_exception_type is not None:
            detail = f"{detail}; provider exception: {provider_exception_type}"
        sanitized = JobSubmissionError(f"job submission failed ({detail})")
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


def _physical_qubit_mappings(
    circuits: Sequence[QuantumCircuit],
) -> tuple[tuple[int, ...], ...]:
    mappings: list[tuple[int, ...]] = []
    for circuit in circuits:
        measured: dict[int, int] = {}
        physical_by_qubit = None
        layout = getattr(circuit, "layout", None)
        initial_layout = getattr(layout, "initial_layout", None)
        get_registers = getattr(initial_layout, "get_registers", None)
        if callable(get_registers) and get_registers() == set(circuit.qregs):
            get_virtual_bits = getattr(initial_layout, "get_virtual_bits", None)
            if not callable(get_virtual_bits):
                raise BackendCompatibilityError(
                    "compiled measurement physical layout is invalid"
                )
            physical_by_qubit = get_virtual_bits()
        for instruction in circuit.data:
            if (
                instruction.operation.name != "measure"
                or len(instruction.qubits) != 1
                or len(instruction.clbits) != 1
            ):
                continue
            classical = circuit.find_bit(instruction.clbits[0]).index
            qubit = instruction.qubits[0]
            if physical_by_qubit is not None:
                physical = physical_by_qubit.get(qubit)
                if (
                    type(physical) is not int
                    or physical < 0
                    or physical >= circuit.num_qubits
                ):
                    raise BackendCompatibilityError(
                        "compiled measurement physical layout is invalid"
                    )
            else:
                physical = circuit.find_bit(qubit).index
            measured[classical] = physical
        if not measured or tuple(sorted(measured)) != tuple(range(len(measured))):
            raise BackendCompatibilityError(
                "compiled measurements require contiguous classical-bit mapping"
            )
        mapping = tuple(measured[index] for index in range(len(measured)))
        if len(set(mapping)) != len(mapping):
            raise BackendCompatibilityError(
                "compiled measurement maps multiple bits to one physical qubit"
            )
        mappings.append(mapping)
    if not mappings:
        raise BackendCompatibilityError(
            "compiled measurement circuits contain no measurements"
        )
    return tuple(mappings)


def _adapter_workload_target(adapter: Any) -> Any | None:
    for backend_attribute in ("backend", "target_backend"):
        try:
            backend = getattr(adapter, backend_attribute, None)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            continue
        if backend is None:
            continue
        try:
            target = getattr(backend, "target", None)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            continue
        if target is not None:
            return target
    return None


def _workload_metrics_metadata(
    metrics: WorkloadMetrics,
    *,
    compact: bool,
) -> dict[str, object]:
    if compact:
        aggregate = _plain_json_value(metrics.aggregate)
        if not isinstance(aggregate, dict):
            raise BackendCompatibilityError("compiled workload metrics are invalid")
        return aggregate

    payload = metrics.to_safe_dict()
    aggregate = payload.get("aggregate")
    circuits = payload.get("circuits")
    if not isinstance(aggregate, dict) or not isinstance(circuits, list):
        raise BackendCompatibilityError("compiled workload metrics are invalid")
    return {
        "circuits": list(circuits),
        "aggregate": dict(aggregate),
    }


def _workload_rejection_category(error: Exception) -> str:
    category = type(error).__name__
    return (
        category
        if _PROVIDER_EXCEPTION_TYPE.fullmatch(category)
        else "Exception"
    )


def _validate_compiled_workload_batch(
    batch: Any,
    *,
    expected_identity: BackendIdentity | None,
    expected_count: int,
    spec: ExperimentSpec,
) -> CompiledBatch:
    if not isinstance(batch, CompiledBatch):
        raise BackendCompatibilityError("adapter compile must return CompiledBatch")
    if expected_identity is not None:
        if batch.target_identity != expected_identity:
            raise BackendCompatibilityError(
                "compiled target does not match resolved backend"
            )
    else:
        _validate_adapter_target(spec, batch.target_identity)
    if len(batch.circuits) != expected_count:
        raise BackendCompatibilityError(
            "compiled circuit count does not match measurement settings"
        )
    return batch


def _compile_measurement_workload(
    adapter: Any,
    circuits: Sequence[QuantumCircuit],
    settings: Sequence[Sequence[str]],
    spec: ExperimentSpec,
    *,
    expected_identity: BackendIdentity | None = None,
) -> _CompiledWorkloadSelection:
    workload_circuits = tuple(circuits)
    workload_settings = tuple(settings)
    if spec.workload_optimization is None:
        batch = _validate_compiled_workload_batch(
            _compile_with_adapter(adapter, workload_circuits, spec.transpilation),
            expected_identity=expected_identity,
            expected_count=len(workload_settings),
            spec=spec,
        )
        return _CompiledWorkloadSelection(
            batch=batch,
            physical_mappings=(),
            metadata={},
        )

    if not workload_circuits:
        raise ExperimentValidationError(
            "workload optimization requires logical measurement circuits"
        )
    if any(
        not isinstance(circuit, QuantumCircuit)
        for circuit in workload_circuits
    ):
        raise ExperimentValidationError(
            "workload optimization requires QuantumCircuit inputs"
        )
    logical_widths = {circuit.num_qubits for circuit in workload_circuits}
    if len(logical_widths) != 1:
        raise ExperimentValidationError(
            "workload optimization requires one logical circuit width"
        )
    logical_width = next(iter(logical_widths))
    search = spec.workload_optimization
    if any(len(layout) != logical_width for layout in search.initial_layouts):
        raise ExperimentValidationError(
            "workload optimization layout width must match logical circuit width"
        )

    target = (
        _adapter_workload_target(adapter)
        if search.prefer_calibration_metrics
        else None
    )
    accepted: list[
        tuple[
            int,
            _CompiledWorkloadCandidate,
            tuple[tuple[int, ...], ...],
        ]
    ] = []
    candidate_rows: list[dict[str, object]] = []
    candidate_index = 0
    for layout in search.initial_layouts:
        for seed in search.seed_transpilers:
            row: dict[str, object] = {
                "status": "rejected",
                "candidate_index": candidate_index,
                "layout": list(layout),
                "seed_transpiler": seed,
            }
            try:
                config = replace(
                    spec.transpilation,
                    initial_layout=layout,
                    seed_transpiler=seed,
                )
                batch = _validate_compiled_workload_batch(
                    _compile_with_adapter(adapter, workload_circuits, config),
                    expected_identity=expected_identity,
                    expected_count=len(workload_settings),
                    spec=spec,
                )
                if any(
                    not isinstance(circuit, QuantumCircuit)
                    for circuit in batch.circuits
                ):
                    raise BackendCompatibilityError(
                        "compiled workload circuits must be QuantumCircuit values"
                    )
                physical_mappings = _physical_qubit_mappings(batch.circuits)
                if any(
                    len(mapping) != logical_width
                    for mapping in physical_mappings
                ):
                    raise BackendCompatibilityError(
                        "compiled workload measurement mapping width is invalid"
                    )
                metrics = summarize_compiled_workload(
                    batch.circuits,
                    settings=workload_settings,
                    physical_mappings=physical_mappings,
                    requested_physical_qubits=layout,
                    target=target,
                )
                if search.require_exact_physical_qubit_set and not metrics.aggregate[
                    "uses_exact_physical_qubit_set"
                ]:
                    raise BackendCompatibilityError(
                        "compiled workload escaped requested physical layout"
                    )
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception as error:
                row["category"] = _workload_rejection_category(error)
                candidate_rows.append(row)
            else:
                candidate = _CompiledWorkloadCandidate(
                    layout=layout,
                    seed=seed,
                    batch=batch,
                    metrics=metrics,
                )
                row["status"] = "accepted"
                row["metrics"] = _workload_metrics_metadata(
                    metrics,
                    compact=True,
                )
                candidate_rows.append(row)
                accepted.append((candidate_index, candidate, physical_mappings))
            candidate_index += 1

    if not accepted:
        raise BackendCompatibilityError(
            f"no workload candidate was accepted ({len(candidate_rows)} rejected)"
        ) from None

    use_error, use_duration = choose_workload_ranking_basis(
        tuple(candidate.metrics for _, candidate, _ in accepted),
        prefer_calibration=search.prefer_calibration_metrics,
    )
    selected_index, selected, selected_mappings = min(
        accepted,
        key=lambda item: workload_rank_key(
            item[1].metrics,
            use_error=use_error,
            use_duration=use_duration,
            seed=item[1].seed,
            layout=item[1].layout,
        ),
    )
    metadata: dict[str, object] = {
        "ranking_basis": (
            "calibration_error_duration" if use_error else "structural"
        ),
        "selected_candidate_index": selected_index,
        "selected_layout": list(selected.layout),
        "selected_seed_transpiler": selected.seed,
        "candidates": candidate_rows,
        "selected_workload": _workload_metrics_metadata(
            selected.metrics,
            compact=False,
        ),
    }
    return _CompiledWorkloadSelection(
        batch=selected.batch,
        physical_mappings=selected_mappings,
        metadata=metadata,
    )


def _validate_selected_physical_set(
    physical_mappings: Sequence[Sequence[int]],
    selected_layout: Sequence[int],
    *,
    require_exact: bool,
    context: str,
) -> None:
    if not require_exact:
        return
    expected = set(selected_layout)
    if not physical_mappings or any(
        set(mapping) != expected
        for mapping in physical_mappings
    ):
        raise BackendCompatibilityError(
            f"{context} measured physical qubit set does not match "
            "selected physical layout"
        )


def _collapse_twirled_physical_qubit_mappings(
    twirled: TwirledBatch,
) -> tuple[tuple[int, ...], ...]:
    variant_mappings = _physical_qubit_mappings(twirled.circuits)
    original_count = max(twirled.original_indices) + 1
    collapsed: list[tuple[int, ...] | None] = [None] * original_count
    for original_index, mapping in zip(
        twirled.original_indices, variant_mappings, strict=True
    ):
        expected = collapsed[original_index]
        if expected is None:
            collapsed[original_index] = mapping
        elif mapping != expected:
            raise BackendCompatibilityError(
                "twirled variants have inconsistent physical measurement mapping"
            )
    if any(mapping is None for mapping in collapsed):
        raise BackendCompatibilityError(
            "twirled physical measurement mapping is incomplete"
        )
    return tuple(mapping for mapping in collapsed if mapping is not None)


def _physical_qubit_union(
    mappings: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            physical
            for mapping in mappings
            for physical in mapping
        )
    )


def _validate_physical_calibration_compile(
    source: Sequence[QuantumCircuit],
    compiled: Sequence[QuantumCircuit],
) -> None:
    if len(compiled) != len(source):
        raise BackendCompatibilityError(
            "compiled readout calibration circuit count does not match source"
        )
    for source_circuit, compiled_circuit in zip(source, compiled, strict=True):
        metadata = source_circuit.metadata or {}
        expected = metadata.get("physical_qubit")
        if type(expected) is not int or expected < 0:
            raise BackendCompatibilityError(
                "readout calibration source physical qubit is invalid"
            )
        actual = _physical_qubit_mappings((compiled_circuit,))[0]
        if actual != (expected,):
            raise BackendCompatibilityError(
                "compiled readout calibration circuit changed its physical qubit"
            )


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
    mappings = _physical_qubit_mappings(compiled_circuits)
    mapping = _physical_qubit_union(mappings)
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
            "mapping_by_circuit_index": [list(item) for item in mappings],
        }
        _write_state(store, run, document)
        return reusable

    source = build_readout_calibration_circuits(
        mapping,
        circuit_width=max(circuit.num_qubits for circuit in compiled_circuits),
    )
    compile_physical = getattr(adapter, "compile_physical", None)
    compiled = (
        compile_physical(source, spec.transpilation)
        if callable(compile_physical)
        else adapter.compile(source, spec.transpilation)
    )
    if not isinstance(compiled, CompiledBatch) or compiled.target_identity != identity:
        raise BackendCompatibilityError(
            "readout calibration compile target does not match backend"
        )
    _validate_physical_calibration_compile(source, compiled.circuits)
    filename = "readout-calibration-circuits.qpy"
    digest = store.write_circuits(run, compiled.circuits, filename)
    document["calibration"] = {
        "circuits_artifact": filename,
        "circuits_sha256": digest,
        "evidence_artifact": None,
        "evidence_sha256": None,
        "status": "compiled",
        "qubit_mapping": list(mapping),
        "mapping_by_circuit_index": [list(item) for item in mappings],
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


def _persisted_physical_qubit_mappings(
    document: Mapping[str, Any],
    calibration: ReadoutCalibration | None,
    setting_count: int,
) -> tuple[tuple[int, ...], ...] | None:
    if calibration is None:
        return None
    record = document.get("calibration")
    if not isinstance(record, Mapping):
        raise ExperimentPersistenceError("readout calibration manifest is missing")
    persisted = record.get("mapping_by_circuit_index")
    if persisted is None:
        return (calibration.qubit_mapping,) * setting_count
    if not isinstance(persisted, Sequence) or isinstance(persisted, (str, bytes)):
        raise ExperimentPersistenceError(
            "readout physical mapping manifest is invalid"
        )
    try:
        mappings = tuple(tuple(mapping) for mapping in persisted)
    except TypeError:
        raise ExperimentPersistenceError(
            "readout physical mapping manifest is invalid"
        ) from None
    calibrated = set(calibration.qubit_mapping)
    if (
        len(mappings) != setting_count
        or any(
            not mapping
            or any(type(qubit) is not int or qubit < 0 for qubit in mapping)
            or len(set(mapping)) != len(mapping)
            or not set(mapping).issubset(calibrated)
            for mapping in mappings
        )
    ):
        raise ExperimentPersistenceError(
            "readout physical mapping manifest is invalid"
        )
    return mappings


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
    checkpoint: _DirectPostprocessingCheckpoint | None = None,
) -> ExperimentResult:
    if checkpoint is not None:
        checkpoint_context = document["postprocessing_checkpoint"]
        evaluator_mode = checkpoint_context["evaluator_mode"]
        readout_strategy_mode = (
            "unused"
            if not checkpoint.spec.mitigation.readout
            else ("default" if readout_strategy is None else "injected")
        )
        zne_strategy_mode = (
            "unused"
            if not checkpoint.spec.mitigation.zne
            else ("default" if zne_strategy is None else "injected")
        )
        if (
            spec.to_safe_dict() != checkpoint.spec.to_safe_dict()
            or tuple(factors) != checkpoint.factors
            or (
                calibration.to_safe_dict() if calibration is not None else None
            )
            != (
                checkpoint.calibration.to_safe_dict()
                if checkpoint.calibration is not None
                else None
            )
            or (evaluator is None) != (evaluator_mode == "default")
            or readout_strategy_mode
            != checkpoint_context["readout_strategy_mode"]
            or zne_strategy_mode != checkpoint_context["zne_strategy_mode"]
        ):
            raise ExperimentValidationError(
                "postprocessing context does not match the durable checkpoint"
            ) from None
        if checkpoint.spec.mitigation.readout:
            _require_readout_dependency(readout_strategy)
        result = bootstrap_bell_results(
            checkpoint.inputs,
            checkpoint.spec.uncertainty,
            readout_strategy=readout_strategy,
            zne_strategy=zne_strategy,
            _evaluator=evaluator,
        )
        return _write_completed_checkpoint(
            checkpoint,
            result.to_safe_dict(),
            clock,
            store=store,
            run=run,
        )

    _transition(store, run, document, ExperimentStatus.POSTPROCESSING, clock)
    metadata = store.read_json(run, document["postprocessing"]["artifact"])
    counts = OrderedDict((factor, store.read_counts(run, factor)) for factor in factors)
    inputs = BootstrapInputs(
        counts_by_factor=counts,
        terms=metadata["terms"],
        qutrit_bit_indices_by_setting=metadata["qutrit_bit_indices_by_setting"],
        decoding_kwargs=metadata["decoding_kwargs"],
        readout_calibration=calibration,
        physical_qubit_mappings=_persisted_physical_qubit_mappings(
            document, calibration, len(metadata["setting_by_circuit_index"])
        ),
    )
    result = bootstrap_bell_results(
        inputs,
        spec.uncertainty,
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


def _serialise_counts(
    counts_by_factor: Mapping[int, Mapping[Any, Mapping[str, int]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(factor): [
            {"setting": list(setting), "counts": dict(counts)}
            for setting, counts in by_setting.items()
        ]
        for factor, by_setting in counts_by_factor.items()
    }


def _encode_checkpoint_value(value: Any, active: set[int] | None = None) -> Any:
    """Encode only the metadata types needed for deterministic postprocessing."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint contains a non-finite number"
            ) from None
        return value
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint contains a non-finite complex number"
            ) from None
        return {
            _CHECKPOINT_CODEC_KIND: "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if active is None:
        active = set()
    if not isinstance(value, (Mapping, list, tuple)):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint contains an unsupported metadata value"
        ) from None
    identity = id(value)
    if identity in active:
        raise ExperimentPersistenceError(
            "postprocessing checkpoint contains recursive metadata"
        ) from None
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                _CHECKPOINT_CODEC_KIND: "mapping",
                "entries": [
                    [
                        _encode_checkpoint_value(key, active),
                        _encode_checkpoint_value(item, active),
                    ]
                    for key, item in value.items()
                ],
            }
        return {
            _CHECKPOINT_CODEC_KIND: "sequence",
            "items": [_encode_checkpoint_value(item, active) for item in value],
        }
    finally:
        active.remove(identity)


def _decode_checkpoint_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ExperimentPersistenceError(
            "postprocessing checkpoint contains a non-finite number"
        ) from None
    if not isinstance(value, dict):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint metadata encoding is invalid"
        ) from None
    kind = value.get(_CHECKPOINT_CODEC_KIND)
    if kind == "complex" and set(value) == {
        _CHECKPOINT_CODEC_KIND,
        "real",
        "imag",
    }:
        real = value["real"]
        imag = value["imag"]
        if (
            type(real) not in {int, float}
            or type(imag) not in {int, float}
            or not math.isfinite(real)
            or not math.isfinite(imag)
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint complex encoding is invalid"
            ) from None
        return complex(real, imag)
    if kind == "sequence" and set(value) == {_CHECKPOINT_CODEC_KIND, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint sequence encoding is invalid"
            ) from None
        return tuple(_decode_checkpoint_value(item) for item in items)
    if kind == "mapping" and set(value) == {_CHECKPOINT_CODEC_KIND, "entries"}:
        entries = value["entries"]
        if not isinstance(entries, list):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint mapping encoding is invalid"
            ) from None
        decoded: dict[Any, Any] = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint mapping entry is invalid"
                ) from None
            key = _decode_checkpoint_value(entry[0])
            try:
                if key in decoded:
                    raise ExperimentPersistenceError(
                        "postprocessing checkpoint mapping contains a duplicate key"
                    ) from None
                decoded[key] = _decode_checkpoint_value(entry[1])
            except TypeError:
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint mapping key is not hashable"
                ) from None
        return decoded
    raise ExperimentPersistenceError(
        "postprocessing checkpoint metadata encoding is invalid"
    ) from None


def _postprocessing_checkpoint_payload(
    metadata: Mapping[str, Any],
    factors: Sequence[int],
    physical_qubit_mappings: Sequence[Sequence[int]] | None,
    *,
    evaluator: Callable[[Any], complex] | None,
    readout_enabled: bool,
    readout_strategy: Any | None,
    zne_enabled: bool,
    zne_strategy: Any | None,
) -> dict[str, Any]:
    return {
        "version": _DIRECT_POSTPROCESSING_CHECKPOINT_VERSION,
        "evaluator_mode": "default" if evaluator is None else "injected",
        "readout_strategy_mode": (
            "unused"
            if not readout_enabled
            else ("default" if readout_strategy is None else "injected")
        ),
        "zne_strategy_mode": (
            "unused"
            if not zne_enabled
            else ("default" if zne_strategy is None else "injected")
        ),
        "factors": list(factors),
        "setting_by_circuit_index": _encode_checkpoint_value(
            metadata["setting_by_circuit_index"]
        ),
        "terms": _encode_checkpoint_value(metadata["terms"]),
        "qutrit_bit_indices_by_setting": _encode_checkpoint_value(
            metadata["qutrit_bit_indices_by_setting"]
        ),
        "decoding_kwargs": _encode_checkpoint_value(
            decoding_kwargs_from_metadata(metadata)
        ),
        "physical_qubit_mappings": _encode_checkpoint_value(
            physical_qubit_mappings
        ),
        "qutrit_qubits": _encode_checkpoint_value(metadata.get("qutrit_qubits", ())),
        "candidate": _encode_checkpoint_value(metadata.get("candidate")),
    }


def _plain_json_value(value: Any, active: set[int] | None = None) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if active is None:
        active = set()
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
        raise ExperimentPersistenceError(
            "final experiment document contains a non-JSON value"
        ) from None
    identity = id(value)
    if identity in active:
        raise ExperimentPersistenceError(
            "final experiment document contains a recursive value"
        ) from None
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ExperimentPersistenceError(
                    "final experiment document keys must be strings"
                ) from None
            if "__qoq_type__" in value:
                raise ExperimentPersistenceError(
                    "final experiment document contains a reserved JSON key"
                ) from None
            return {
                key: _plain_json_value(item, active)
                for key, item in value.items()
            }
        values = (
            sorted(value, key=repr)
            if isinstance(value, (set, frozenset))
            else value
        )
        return [_plain_json_value(item, active) for item in values]
    finally:
        active.remove(identity)


def _validate_final_document(document: Mapping[str, Any]) -> None:
    validate_persisted_strings(
        document,
        description="final experiment document",
        error_type=ExperimentPersistenceError,
    )
    ExperimentStore.validate_plain_json(document)


def _validated_schema_fragments(
    spec: ExperimentSpec,
    artifacts: Any,
    identity: BackendIdentity,
    compiled: CompiledBatch,
) -> dict[str, Any]:
    safe_spec = spec.to_safe_dict()
    if spec.workload_optimization is None:
        safe_spec.pop("workload_optimization", None)
    fragments = _plain_json_value(
        {
            "spec": safe_spec,
            "source": {
                "provenance": dict(artifacts.provenance),
                "paths": {
                    name: str(path)
                    for name, path in artifacts.source_paths.items()
                },
            },
            "backend": identity.to_safe_dict(),
            "transpilation": dict(
                compiled.metadata.get(
                    "transpilation", spec.transpilation.to_safe_dict()
                )
            ),
        }
    )
    if not isinstance(fragments, dict):
        raise ExperimentPersistenceError(
            "final experiment schema fragments must be a mapping"
        ) from None
    _validate_final_document(fragments)
    return fragments


def _completed_direct_document(
    schema_fragments: Mapping[str, Any],
    job_ids: Sequence[str],
    counts: Mapping[int, Mapping[Any, Mapping[str, int]]],
    calibration: ReadoutCalibration | None,
    safe_result: Mapping[str, Any],
    completed_at: str,
    *,
    experiment_id: str,
) -> dict[str, Any]:
    document = _plain_json_value(
        {
            "schema_version": 3,
            "experiment_id": experiment_id,
            "status": ExperimentStatus.COMPLETED.value,
            "completed_at": completed_at,
            **dict(schema_fragments),
            "job_ids": list(job_ids),
            "counts_by_factor": _serialise_counts(counts),
            "calibration": calibration.to_safe_dict() if calibration else None,
            "result": dict(safe_result),
        }
    )
    if not isinstance(document, dict):
        raise ExperimentPersistenceError(
            "final experiment document must be a mapping"
        ) from None
    _validate_final_document(document)
    return document


def _expected_execution_job_shape(
    spec: ExperimentSpec,
    setting_count: int,
) -> tuple[int, int]:
    instances = (
        spec.mitigation.twirling_instances
        if spec.mitigation.circuit_twirling
        else 1
    )
    return setting_count * instances, spec.shots // instances


def _checkpoint_job_records(
    spec: ExperimentSpec,
    jobs: Sequence[SubmittedJob],
    factors: Sequence[int],
    calibration: ReadoutCalibration | None,
    *,
    setting_count: int,
) -> list[dict[str, Any]]:
    calibration_job_count = len(jobs) - len(factors)
    if calibration_job_count not in {0, 1} or (
        calibration_job_count == 1 and calibration is None
    ):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint jobs are inconsistent with executed workload"
        ) from None

    execution_circuit_count, execution_shots = _expected_execution_job_shape(
        spec, setting_count
    )
    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        is_calibration = calibration_job_count == 1 and index == 0
        factor = None if is_calibration else factors[index - calibration_job_count]
        expected_circuit_count = (
            len(calibration.raw_counts)
            if is_calibration and calibration is not None
            else execution_circuit_count
        )
        expected_shots = spec.shots if is_calibration else execution_shots
        if (
            job.circuit_count != expected_circuit_count
            or job.shots != expected_shots
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint job record is inconsistent with executed workload"
            ) from None
        records.append(
            {
                **job.to_safe_dict(),
                "role": "calibration" if is_calibration else "execution",
                "factor": factor,
            }
        )
    return records


def _write_postprocessing_checkpoint(
    spec: ExperimentSpec,
    schema_fragments: Mapping[str, Any],
    jobs: Sequence[SubmittedJob],
    counts: Mapping[int, Mapping[Any, Mapping[str, int]]],
    calibration: ReadoutCalibration | None,
    metadata: Mapping[str, Any],
    factors: Sequence[int],
    physical_qubit_mappings: Sequence[Sequence[int]] | None,
    clock: Callable[[], datetime],
    *,
    repo_root: Path | str | None,
    evaluator: Callable[[Any], complex] | None,
    readout_strategy: Any | None,
    zne_strategy: Any | None,
) -> tuple[ExperimentStore, Path, dict[str, Any]]:
    job_records = _checkpoint_job_records(
        spec,
        jobs,
        factors,
        calibration,
        setting_count=len(metadata["setting_by_circuit_index"]),
    )
    document = _plain_json_value(
        {
            "schema_version": 3,
            "experiment_id": ExperimentStore.generated_run_name_placeholder(),
            "status": ExperimentStatus.POSTPROCESSING.value,
            "checkpointed_at": _timestamp(clock),
            **dict(schema_fragments),
            "job_ids": [job.job_id for job in jobs],
            "jobs": job_records,
            "counts_by_factor": _serialise_counts(counts),
            "calibration": calibration.to_safe_dict() if calibration else None,
            "postprocessing_checkpoint": _postprocessing_checkpoint_payload(
                metadata,
                factors,
                physical_qubit_mappings,
                evaluator=evaluator,
                readout_enabled=spec.mitigation.readout,
                readout_strategy=readout_strategy,
                zne_enabled=spec.mitigation.zne,
                zne_strategy=zne_strategy,
            ),
        }
    )
    if not isinstance(document, dict):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint must be a mapping"
        ) from None
    _validate_final_document(document)

    store = ExperimentStore(_output_root(spec, repo_root))
    staging, run = store.stage_run()
    document["experiment_id"] = run.name
    _validate_final_document(document)
    try:
        store.write_plain_json(staging, "experiment.json", document)
        published = store.publish_staged_run(staging, run)
    except BaseException:
        if staging.exists():
            store.discard_staged_run(staging)
        raise
    return store, published, document


def _write_completed_checkpoint(
    checkpoint: _DirectPostprocessingCheckpoint,
    safe_result: Mapping[str, Any],
    clock: Callable[[], datetime],
    *,
    store: ExperimentStore,
    run: Path,
) -> ExperimentResult:
    document = _completed_direct_document(
        checkpoint.schema_fragments,
        checkpoint.job_ids,
        checkpoint.inputs.counts_by_factor,
        checkpoint.calibration,
        safe_result,
        _timestamp(clock),
        experiment_id=run.name,
    )
    store.write_plain_json(run, "experiment.json", document)
    return ExperimentResult(
        experiment_id=run.name,
        status=ExperimentStatus.COMPLETED,
        artifact_dir=run,
        values=safe_result,
        backend=checkpoint.identity.to_safe_dict(),
        job_ids=checkpoint.job_ids,
    )


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
    _twirling_transform: Callable[..., TwirledBatch] = twirl_iqm_circuits,
    _evaluator: Callable[[Any], complex] | None = None,
) -> ExperimentResult:
    """Run one experiment directly from compiler output held in memory."""

    if not isinstance(spec, ExperimentSpec):
        raise ExperimentValidationError("spec must be ExperimentSpec")
    _validate_execution_options(timeout, run_options)
    _validate_twirling_spec(spec)

    artifacts = load_basis_artifacts(spec.basis, spec.state, repo_root)
    prepared = prepare_measurements(artifacts)
    metadata = prepared.metadata
    required_metadata = (
        "setting_by_circuit_index",
        "terms",
        "qutrit_bit_indices_by_setting",
    )
    if not isinstance(metadata, Mapping) or any(
        name not in metadata for name in required_metadata
    ):
        raise ExperimentValidationError("measurement metadata is incomplete")
    settings = tuple(
        tuple(setting) for setting in metadata["setting_by_circuit_index"]
    )
    circuits = tuple(prepared.circuits)
    if len(settings) != len(circuits):
        raise ExperimentValidationError(
            "measurement settings do not match logical circuits"
        )

    resolved_adapter = create_backend_adapter(spec.backend) if adapter is None else adapter
    try:
        identity = resolved_adapter.resolve()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BackendCompatibilityError("adapter resolution failed") from None
    if not isinstance(identity, BackendIdentity):
        raise BackendCompatibilityError("adapter resolve must return BackendIdentity")
    safe_identity = identity.to_safe_dict()
    _validate_persisted_strings(safe_identity, description="backend identity")
    _validate_adapter_target(spec, identity)

    workload_selection = _compile_measurement_workload(
        resolved_adapter,
        circuits,
        settings,
        spec,
        expected_identity=identity,
    )
    compiled = workload_selection.batch
    selected_layout: tuple[int, ...] = ()
    require_exact_selected_layout = False
    if spec.workload_optimization is not None:
        selected_layout = tuple(
            workload_selection.metadata["selected_layout"]
        )
        require_exact_selected_layout = (
            spec.workload_optimization.require_exact_physical_qubit_set
        )
        _validate_selected_physical_set(
            workload_selection.physical_mappings,
            selected_layout,
            require_exact=require_exact_selected_layout,
            context="selected workload",
        )

    twirled = None
    twirled_physical_qubit_mappings = None
    twirling_metadata = None
    execution_compiled = compiled
    execution_shots = spec.shots
    if spec.mitigation.circuit_twirling:
        try:
            transformed = _twirling_transform(
                compiled.circuits,
                instances=spec.mitigation.twirling_instances,
                seed=spec.mitigation.twirling_seed,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except (BackendCompatibilityError, OptionalDependencyError):
            raise
        except Exception as error:
            raise BackendCompatibilityError(
                f"IQM circuit twirling failed ({type(error).__name__})"
            ) from None
        twirled = _validated_twirled_batch(
            transformed,
            original_count=len(settings),
            instances=spec.mitigation.twirling_instances,
            seed=spec.mitigation.twirling_seed,
        )
        twirled_physical_qubit_mappings = (
            _collapse_twirled_physical_qubit_mappings(twirled)
        )
        _validate_selected_physical_set(
            twirled_physical_qubit_mappings,
            selected_layout,
            require_exact=require_exact_selected_layout,
            context="twirled workload",
        )
        execution_compiled = CompiledBatch(
            twirled.circuits,
            identity,
            compiled.metadata,
        )
        execution_shots = spec.shots // spec.mitigation.twirling_instances
        twirling_metadata = {
            **dict(twirled.metadata),
            "shots_per_instance": execution_shots,
            "total_shots_per_circuit": spec.shots,
        }
    schema_fragments = _validated_schema_fragments(
        spec,
        artifacts,
        identity,
        compiled,
    )
    if spec.workload_optimization is not None:
        schema_fragments["workload_optimization"] = _plain_json_value(
            workload_selection.metadata
        )
    schema_fragments["twirling"] = _plain_json_value(twirling_metadata)
    _validate_final_document(schema_fragments)

    factors = (
        validate_zne_factors(spec.mitigation.zne_factors)
        if spec.mitigation.zne
        else (1,)
    )
    batches = _factor_batches(execution_compiled, factors)
    if require_exact_selected_layout:
        for factor, batch in batches.items():
            _validate_selected_physical_set(
                _physical_qubit_mappings(batch),
                selected_layout,
                require_exact=True,
                context=f"ZNE factor {factor} workload",
            )
    jobs: list[SubmittedJob] = []
    physical_qubit_mappings = None
    calibration = None
    if spec.mitigation.readout:
        _require_readout_dependency(_readout_strategy)
        physical_qubit_mappings = (
            twirled_physical_qubit_mappings
            if twirled_physical_qubit_mappings is not None
            else (
                workload_selection.physical_mappings
                if spec.workload_optimization is not None
                else _physical_qubit_mappings(batches[1])
            )
        )
        calibration_physical_mappings = (
            workload_selection.physical_mappings
            if spec.workload_optimization is not None
            else physical_qubit_mappings
        )
        physical_qubits = tuple(
            sorted(
                {
                    physical
                    for mapping in calibration_physical_mappings
                    for physical in mapping
                }
            )
        )
        if readout_calibration is None or spec.mitigation.force_recalibration:
            calibration, calibration_job = _run_readout_calibration(
                resolved_adapter,
                identity,
                calibration_physical_mappings,
                spec,
                timeout=timeout,
                run_options=run_options,
                clock=_clock,
                sleep=_sleep,
            )
            _validate_persisted_strings(
                calibration_job.job_id,
                description="readout calibration job ID",
            )
            jobs.append(calibration_job)
        else:
            if not calibration_cache_is_valid(
                readout_calibration,
                backend_identity=_identity_key(identity),
                calibration_id=_calibration_id(identity),
                qubit_mapping=physical_qubits,
                now=_clock(),
                max_age_hours=spec.mitigation.readout_max_age_hours,
            ):
                raise BackendCompatibilityError(
                    "injected readout calibration is not valid for this target"
                )
            calibration = readout_calibration
    counts_by_factor: OrderedDict[int, OrderedDict[Any, dict[str, int]]] = (
        OrderedDict()
    )
    for factor, batch in batches.items():
        submitted, execution = _execute_batch(
            resolved_adapter,
            identity,
            batch,
            execution_shots,
            timeout=timeout,
            run_options=run_options,
            retry=spec.retry,
            sleep=_sleep,
        )
        _validate_persisted_strings(
            submitted.job_id,
            description="submitted job ID",
        )
        jobs.append(submitted)
        counts_by_factor[factor] = (
            _aggregate_twirled_counts(
                settings,
                execution,
                twirled,
                total_shots=spec.shots,
            )
            if twirled is not None
            else _counts_by_setting(settings, execution)
        )

    BootstrapInputs(
        counts_by_factor=counts_by_factor,
        terms=metadata["terms"],
        qutrit_bit_indices_by_setting=metadata["qutrit_bit_indices_by_setting"],
        decoding_kwargs=decoding_kwargs_from_metadata(metadata),
        readout_calibration=calibration,
        physical_qubit_mappings=physical_qubit_mappings,
    )
    store, run, _ = _write_postprocessing_checkpoint(
        spec,
        schema_fragments,
        jobs,
        counts_by_factor,
        calibration,
        metadata,
        factors,
        physical_qubit_mappings,
        _clock,
        repo_root=repo_root,
        evaluator=_evaluator,
        readout_strategy=_readout_strategy,
        zne_strategy=_zne_strategy,
    )
    try:
        document = store.read_plain_json(run, "experiment.json")
        checkpoint = _load_direct_postprocessing_checkpoint(document, run, spec)
        return _postprocess(
            checkpoint.spec,
            checkpoint.factors,
            checkpoint.calibration,
            store=store,
            run=run,
            document=document,
            clock=_clock,
            readout_strategy=_readout_strategy,
            zne_strategy=_zne_strategy,
            evaluator=_evaluator,
            checkpoint=checkpoint,
        )
    except BaseException as error:
        try:
            setattr(error, "__qoq_artifact_dir__", run)
        except Exception:
            pass
        raise


def _open_run(
    artifact_dir: Path | str,
) -> tuple[ExperimentStore, Path, dict[str, Any]]:
    manifest = RunManifest.load(artifact_dir)
    run = Path(artifact_dir).expanduser().resolve(strict=True)
    store = ExperimentStore(run.parents[1])
    return store, run, manifest.to_safe_dict()


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
    normalized_safe = _normalize_experiment_spec_dict(safe)
    if supplied is not None:
        supplied_safe = supplied.to_safe_dict() if isinstance(supplied, ExperimentSpec) else None
        if (
            isinstance(supplied_safe, dict)
            and supplied_safe.get("workload_optimization") is None
            and "workload_optimization" not in normalized_safe
        ):
            supplied_safe.pop("workload_optimization")
        if (
            not isinstance(supplied, ExperimentSpec)
            or supplied_safe != normalized_safe
        ):
            raise ExperimentValidationError("injected spec does not match persisted experiment spec")
        return supplied
    try:
        return ExperimentSpec.from_safe_dict(normalized_safe)
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


def _validate_plain_resume_value(value: Any) -> None:
    if value is None or type(value) in {bool, int, str}:
        pass
    elif type(value) is float:
        if not math.isfinite(value):
            raise ExperimentPersistenceError(
                "completed experiment JSON is invalid"
            ) from None
    elif isinstance(value, list):
        for item in value:
            _validate_plain_resume_value(item)
    elif isinstance(value, dict):
        if "__qoq_type__" in value:
            raise ExperimentPersistenceError(
                "completed experiment JSON is invalid"
            ) from None
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ExperimentPersistenceError(
                    "completed experiment JSON is invalid"
                ) from None
            _validate_plain_resume_value(item)
    else:
        raise ExperimentPersistenceError(
            "completed experiment JSON is invalid"
        ) from None


def _deserialise_checkpoint_counts(
    value: Any,
) -> OrderedDict[int, OrderedDict[tuple[Any, ...], dict[str, int]]]:
    if not isinstance(value, dict) or not value:
        raise ExperimentPersistenceError(
            "postprocessing checkpoint counts are invalid"
        ) from None
    result: OrderedDict[int, OrderedDict[tuple[Any, ...], dict[str, int]]] = (
        OrderedDict()
    )
    for factor_text, entries in value.items():
        if (
            not isinstance(factor_text, str)
            or not factor_text.isdigit()
            or factor_text != str(int(factor_text))
            or not isinstance(entries, list)
            or not entries
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint counts are invalid"
            ) from None
        factor = int(factor_text)
        by_setting: OrderedDict[tuple[Any, ...], dict[str, int]] = OrderedDict()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"setting", "counts"}
                or not isinstance(entry["setting"], list)
                or not isinstance(entry["counts"], dict)
            ):
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint count entry is invalid"
                ) from None
            setting = tuple(entry["setting"])
            try:
                if setting in by_setting:
                    raise ExperimentPersistenceError(
                        "postprocessing checkpoint contains a duplicate setting"
                    ) from None
            except TypeError:
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint setting is not hashable"
                ) from None
            counts = entry["counts"]
            if any(
                not isinstance(outcome, str)
                or not outcome
                or type(count) is not int
                or count < 0
                for outcome, count in counts.items()
            ):
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint count entry is invalid"
                ) from None
            by_setting[setting] = dict(counts)
        result[factor] = by_setting
    return result


def _load_direct_postprocessing_checkpoint(
    document: Mapping[str, Any],
    run: Path,
    supplied_spec: ExperimentSpec | None,
) -> _DirectPostprocessingCheckpoint:
    required_fields = {
        "schema_version",
        "experiment_id",
        "status",
        "checkpointed_at",
        "spec",
        "source",
        "backend",
        "transpilation",
        "twirling",
        "job_ids",
        "jobs",
        "counts_by_factor",
        "calibration",
        "postprocessing_checkpoint",
    }
    if "workload_optimization" in document:
        required_fields.add("workload_optimization")
    try:
        if (
            set(document) != required_fields
            or document.get("schema_version") != 3
            or document.get("status") != ExperimentStatus.POSTPROCESSING.value
            or document.get("experiment_id") != run.name
            or not isinstance(document.get("checkpointed_at"), str)
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint is invalid"
            ) from None
        _validate_plain_resume_value(dict(document))
        validate_persisted_strings(
            document,
            description="postprocessing checkpoint",
            error_type=ExperimentPersistenceError,
        )
    except ExperimentPersistenceError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint is invalid"
        ) from None

    resumed_spec = _resume_spec(document, supplied_spec)
    checkpoint = document.get("postprocessing_checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "version",
        "evaluator_mode",
        "readout_strategy_mode",
        "zne_strategy_mode",
        "factors",
        "setting_by_circuit_index",
        "terms",
        "qutrit_bit_indices_by_setting",
        "decoding_kwargs",
        "physical_qubit_mappings",
        "qutrit_qubits",
        "candidate",
    }:
        raise ExperimentPersistenceError(
            "postprocessing checkpoint is invalid"
        ) from None
    try:
        if (
            checkpoint["version"] != _DIRECT_POSTPROCESSING_CHECKPOINT_VERSION
            or checkpoint["evaluator_mode"] not in {"default", "injected"}
            or checkpoint["readout_strategy_mode"]
            not in {"default", "injected", "unused"}
            or checkpoint["zne_strategy_mode"]
            not in {"default", "injected", "unused"}
            or (checkpoint["readout_strategy_mode"] == "unused")
            != (not resumed_spec.mitigation.readout)
            or (checkpoint["zne_strategy_mode"] == "unused")
            != (not resumed_spec.mitigation.zne)
            or not isinstance(checkpoint["factors"], list)
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint is invalid"
            ) from None
        factors = validate_zne_factors(tuple(checkpoint["factors"]))
        expected_factors = (
            validate_zne_factors(resumed_spec.mitigation.zne_factors)
            if resumed_spec.mitigation.zne
            else (1,)
        )
        if factors != expected_factors:
            raise ExperimentPersistenceError(
                "postprocessing checkpoint factors are inconsistent with the spec"
            ) from None
        counts = _deserialise_checkpoint_counts(document["counts_by_factor"])
        if set(counts) != set(factors):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint factors do not match saved counts"
            ) from None
        counts = OrderedDict((factor, counts[factor]) for factor in factors)
        if any(
            sum(setting_counts.values()) != resumed_spec.shots
            for by_setting in counts.values()
            for setting_counts in by_setting.values()
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint shot totals are inconsistent with the spec"
            ) from None

        settings = _decode_checkpoint_value(
            checkpoint["setting_by_circuit_index"]
        )
        if (
            not isinstance(settings, tuple)
            or not settings
            or any(not isinstance(setting, tuple) for setting in settings)
            or any(tuple(by_setting) != settings for by_setting in counts.values())
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint settings do not match saved counts"
            ) from None
        terms = _decode_checkpoint_value(checkpoint["terms"])
        qutrit_indices = _decode_checkpoint_value(
            checkpoint["qutrit_bit_indices_by_setting"]
        )
        decoding_kwargs = _decode_checkpoint_value(
            checkpoint["decoding_kwargs"]
        )
        physical_qubit_mappings = _decode_checkpoint_value(
            checkpoint["physical_qubit_mappings"]
        )
        _decode_checkpoint_value(checkpoint["qutrit_qubits"])
        _decode_checkpoint_value(checkpoint["candidate"])
        if (
            not isinstance(terms, tuple)
            or not isinstance(qutrit_indices, Mapping)
            or not isinstance(decoding_kwargs, Mapping)
            or (
                physical_qubit_mappings is not None
                and not isinstance(physical_qubit_mappings, tuple)
            )
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint metadata is invalid"
            ) from None

        calibration_payload = document["calibration"]
        if calibration_payload is not None and (
            not isinstance(calibration_payload, dict)
            or set(calibration_payload)
            != {
                "backend_identity",
                "calibration_id",
                "qubit_mapping",
                "timestamp",
                "shots",
                "raw_counts",
                "assignment_matrices",
            }
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint calibration is invalid"
            ) from None
        calibration = (
            None
            if calibration_payload is None
            else _calibration_from_safe_dict(calibration_payload)
        )
        if (calibration is None) != (not resumed_spec.mitigation.readout):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint calibration is inconsistent with the spec"
            ) from None
        inputs = BootstrapInputs(
            counts_by_factor=counts,
            terms=terms,
            qutrit_bit_indices_by_setting=qutrit_indices,
            decoding_kwargs=decoding_kwargs,
            readout_calibration=calibration,
            physical_qubit_mappings=physical_qubit_mappings,
        )

        backend = document["backend"]
        if not isinstance(backend, dict):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint backend is invalid"
            ) from None
        identity = BackendIdentity(**backend)
        _validate_adapter_target(resumed_spec, identity)
        if calibration is not None and (
            calibration.backend_identity != _identity_key(identity)
            or calibration.calibration_id != _calibration_id(identity)
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint calibration does not match the saved backend"
            ) from None

        job_ids_payload = document["job_ids"]
        jobs = document["jobs"]
        if not isinstance(job_ids_payload, list) or not isinstance(jobs, list):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint jobs are invalid"
            ) from None
        calibration_job_count = len(jobs) - len(factors)
        if calibration_job_count not in {0, 1} or (
            calibration_job_count == 1 and calibration is None
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint jobs are inconsistent with saved factors"
            ) from None
        execution_circuit_count, execution_shots = _expected_execution_job_shape(
            resumed_spec, len(settings)
        )
        job_ids = tuple(_safe_identifier(item, "job_id") for item in job_ids_payload)
        persisted_job_ids: list[str] = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict) or set(job) != {
                "job_id",
                "target_identity",
                "circuit_count",
                "shots",
                "metadata",
                "role",
                "factor",
            }:
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint job record is invalid"
                ) from None
            persisted_job_ids.append(_safe_identifier(job["job_id"], "job_id"))
            if BackendIdentity(**job["target_identity"]) != identity:
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint job target is invalid"
                ) from None
            is_calibration = calibration_job_count == 1 and index == 0
            expected_factor = (
                None if is_calibration else factors[index - calibration_job_count]
            )
            expected_circuit_count = (
                len(calibration.raw_counts)
                if is_calibration and calibration is not None
                else execution_circuit_count
            )
            expected_shots = resumed_spec.shots if is_calibration else execution_shots
            if (
                job["role"]
                != ("calibration" if is_calibration else "execution")
                or type(job["factor"])
                is not (type(None) if is_calibration else int)
                or job["factor"] != expected_factor
                or type(job["circuit_count"]) is not int
                or job["circuit_count"] != expected_circuit_count
                or type(job["shots"]) is not int
                or job["shots"] != expected_shots
                or not isinstance(job["metadata"], dict)
            ):
                raise ExperimentPersistenceError(
                    "postprocessing checkpoint job record is invalid"
                ) from None
        if tuple(persisted_job_ids) != job_ids:
            raise ExperimentPersistenceError(
                "postprocessing checkpoint job IDs do not match job records"
            ) from None

        schema_fragment_names = (
            "spec",
            "source",
            "backend",
            "transpilation",
            "twirling",
        )
        schema_fragments = {
            name: document[name] for name in schema_fragment_names
        }
        if "workload_optimization" in document:
            schema_fragments["workload_optimization"] = document[
                "workload_optimization"
            ]
        if (resumed_spec.workload_optimization is None) != (
            "workload_optimization" not in schema_fragments
        ):
            raise ExperimentPersistenceError(
                "postprocessing checkpoint workload metadata is inconsistent with the spec"
            ) from None
    except ExperimentPersistenceError:
        raise
    except (
        BackendCompatibilityError,
        ExperimentValidationError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        raise ExperimentPersistenceError(
            "postprocessing checkpoint is invalid"
        ) from None
    return _DirectPostprocessingCheckpoint(
        spec=resumed_spec,
        identity=identity,
        job_ids=job_ids,
        factors=factors,
        calibration=calibration,
        inputs=inputs,
        schema_fragments=schema_fragments,
    )


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
    """Load a completed result or finish a complete direct postprocessing checkpoint."""

    store, run = ExperimentStore.open_existing_run(experiment_dir)
    document = store.read_plain_json(run, "experiment.json")
    if not isinstance(document, dict):
        raise ExperimentPersistenceError(
            "experiment.json must contain a mapping"
        ) from None
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3}:
        raise ExperimentPersistenceError(
            "unsupported experiment schema version"
        ) from None
    status = document.get("status")
    known_statuses = {member.value for member in ExperimentStatus}
    if type(status) is not str or status not in known_statuses:
        raise ExperimentPersistenceError(
            "completed experiment JSON is invalid"
        ) from None
    if status == ExperimentStatus.POSTPROCESSING.value:
        if schema_version != 3 or "postprocessing_checkpoint" not in document:
            raise ExperimentValidationError(
                "resuming unfinished experiments is not supported by the direct pipeline"
            )
        _validate_execution_options(timeout, run_options)
        checkpoint = _load_direct_postprocessing_checkpoint(document, run, spec)
        checkpoint_context = document["postprocessing_checkpoint"]
        if (
            any(
                seam is not None
                for seam in (_readout_strategy, _zne_strategy, _evaluator)
            )
            or checkpoint_context["evaluator_mode"] == "injected"
            or checkpoint_context["readout_strategy_mode"] == "injected"
            or checkpoint_context["zne_strategy_mode"] == "injected"
        ):
            raise ExperimentValidationError(
                "injected postprocessing seams cannot be resumed safely"
            ) from None
        _ = adapter, _sleep
        return _postprocess(
            checkpoint.spec,
            checkpoint.factors,
            checkpoint.calibration,
            store=store,
            run=run,
            document=document,
            clock=_clock,
            readout_strategy=_readout_strategy,
            zne_strategy=_zne_strategy,
            evaluator=_evaluator,
            checkpoint=checkpoint,
        )
    if status != ExperimentStatus.COMPLETED.value:
        raise ExperimentValidationError(
            "resuming unfinished experiments is not supported by the direct pipeline"
        )
    backend = document.get("backend", {})
    if isinstance(backend, Mapping) and isinstance(
        backend.get("identity"), Mapping
    ):
        backend = backend["identity"]
    values = document.get("result")
    experiment_id = document.get("experiment_id")
    job_ids = document.get("job_ids", [])
    if not isinstance(values, dict) or not isinstance(backend, dict):
        raise ExperimentPersistenceError("completed experiment JSON is invalid")
    try:
        _validate_plain_resume_value(backend)
        _validate_plain_resume_value(values)
        validate_persisted_strings(
            backend,
            description="completed backend",
            error_type=ExperimentPersistenceError,
        )
        validate_persisted_strings(
            values,
            description="completed result",
            error_type=ExperimentPersistenceError,
        )
        BackendIdentity(**backend)
        if not isinstance(job_ids, list):
            raise ExperimentPersistenceError(
                "completed experiment JSON is invalid"
            )
        safe_job_ids = tuple(
            _safe_identifier(job_id, "job_id") for job_id in job_ids
        )
        validate_persisted_strings(
            safe_job_ids,
            description="completed job IDs",
            error_type=ExperimentPersistenceError,
        )
        if (
            not isinstance(experiment_id, str)
            or experiment_id != run.name
            or not backend
        ):
            raise ExperimentPersistenceError(
                "completed experiment JSON is invalid"
            )
        validate_persisted_strings(
            experiment_id,
            description="experiment ID",
            error_type=ExperimentPersistenceError,
        )
    except (
        ExperimentPersistenceError,
        ExperimentValidationError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        raise ExperimentPersistenceError(
            "completed experiment JSON is invalid"
        ) from None
    return ExperimentResult(
        experiment_id=experiment_id,
        status=ExperimentStatus.COMPLETED,
        artifact_dir=run,
        values=values,
        backend=backend,
        job_ids=safe_job_ids,
    )


def run_experiments(
    specs: Iterable[ExperimentSpec],
    **kwargs: Any,
) -> tuple[ExperimentResult, ...]:
    """Run validated specs sequentially and fail on the first error."""

    if isinstance(specs, (str, bytes, Mapping)):
        raise ExperimentValidationError("specs must be an iterable of ExperimentSpec values")
    try:
        iterator = iter(specs)
    except TypeError as error:
        raise ExperimentValidationError("specs must be iterable") from error
    values = tuple(iterator)
    if any(not isinstance(spec, ExperimentSpec) for spec in values):
        raise ExperimentValidationError("every batch item must be ExperimentSpec")
    return tuple(run_experiment(spec, **kwargs) for spec in values)


__all__ = ["resume_experiment", "run_experiment", "run_experiments"]
