"""Execution and integrity verification for generic qudit vertical slices."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

from ..experiments.backends import create_backend_adapter
from ..experiments.errors import BackendCompatibilityError, BackendUnavailableError
from ..experiments.safety import unsafe_persisted_text
from ..experiments.store import ExperimentStore
from .models import (
    ArtifactIntegrityError,
    ArtifactRef,
    BackendSnapshot,
    ExecutionMode,
    JsonValue,
    ManifestValidationError,
    QuditExperimentResult,
    QuditExperimentSpec,
    RunManifest,
    SoftwareProvenance,
    SpecValidationError,
)

_MANIFEST_FILENAME = "run-manifest.json"
_DEPENDENCIES = ("numpy", "qiskit", "qiskit-aer")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SpecValidationError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _file_hash(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactIntegrityError(f"could not hash {path.name} artifact") from error


def _artifact(
    run: Path,
    *,
    role: str,
    filename: str,
    media_type: str,
    digest: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        path=filename,
        sha256=_file_hash(run / filename) if digest is None else digest,
        media_type=media_type,
    )


def _write_manifest(store: ExperimentStore, run: Path, manifest: RunManifest) -> None:
    store.write_json(run, _MANIFEST_FILENAME, manifest.to_safe_dict())


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value or unsafe_persisted_text(value):
        return None
    return value


def _software_provenance() -> SoftwareProvenance:
    repository = Path(__file__).resolve().parents[3]
    git_commit = _git_value(repository, "rev-parse", "HEAD")
    status = _git_value(repository, "status", "--porcelain")
    dirty_worktree = None if git_commit is None or status is None else bool(status)
    return SoftwareProvenance(
        git_commit=git_commit,
        package_version=_package_version("qudits-on-qubits"),
        python_version=sys.version.split()[0],
        dependencies={name: _package_version(name) for name in _DEPENDENCIES},
        dirty_worktree=dirty_worktree,
    )


def _backend_snapshot(identity: Any, capabilities: Any) -> BackendSnapshot:
    return BackendSnapshot(
        provider=identity.provider or "qiskit-aer",
        backend_name=identity.name,
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        identity=identity.to_safe_dict(),
        capabilities=capabilities.to_safe_dict(),
    )


def _safe_failure(error: Exception, stage: str, timestamp: str) -> Mapping[str, JsonValue]:
    message = " ".join(str(error).split())[:500] or "operation failed"
    if unsafe_persisted_text(message):
        message = "details redacted"
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": message,
        "timestamp": timestamp,
    }


def _persist_preparation(
    store: ExperimentStore,
    run: Path,
    spec: QuditExperimentSpec,
    prepared: Any,
) -> list[ArtifactRef]:
    artifacts: list[ArtifactRef] = []
    source_digest = store.write_circuits(
        run, prepared.source_circuits, "source-circuits.qpy"
    )
    artifacts.append(
        _artifact(
            run,
            role="source-circuits",
            filename="source-circuits.qpy",
            media_type="application/x-qiskit-qpy",
            digest=source_digest,
        )
    )
    store.write_json(run, "encoding.json", spec.encoding.to_manifest_dict())
    artifacts.append(
        _artifact(
            run,
            role="encoding",
            filename="encoding.json",
            media_type="application/json",
        )
    )
    logical_digest = store.write_circuits(
        run, prepared.executable_circuits, "logical-measurements.qpy"
    )
    artifacts.append(
        _artifact(
            run,
            role="logical-measurements",
            filename="logical-measurements.qpy",
            media_type="application/x-qiskit-qpy",
            digest=logical_digest,
        )
    )
    store.write_json(
        run,
        "postprocessing.json",
        {
            "postprocessor": prepared.postprocessor.to_manifest_dict(),
            "preparation_provenance": prepared.provenance,
        },
    )
    artifacts.append(
        _artifact(
            run,
            role="postprocessing",
            filename="postprocessing.json",
            media_type="application/json",
        )
    )
    return artifacts


def run_vertical_slice(
    spec: QuditExperimentSpec,
    *,
    adapter: Any | None = None,
    _clock: Callable[[], datetime] = _utc_now,
) -> QuditExperimentResult:
    """Execute one validated qudit experiment and persist versioned artifacts."""

    if not isinstance(spec, QuditExperimentSpec):
        raise SpecValidationError("spec must be QuditExperimentSpec")
    store = ExperimentStore(spec.output_root)
    run = store.create_run(spec.circuit.circuit_id)
    manifest = RunManifest.initial(
        run_id=run.name,
        experiment_spec=spec.to_manifest_dict(),
        experiment_hash=spec.stable_hash(),
        encoding=spec.encoding.to_manifest_dict(),
        encoding_hash=spec.encoding.stable_hash(),
        software=_software_provenance(),
        timestamp=_timestamp(_clock),
    )
    _write_manifest(store, run, manifest)
    stage = "preparation"
    try:
        prepared = spec.circuit.prepare(spec.encoding)
        artifacts = _persist_preparation(store, run, spec, prepared)

        stage = "backend-validation"
        resolved_adapter = create_backend_adapter(spec.backend) if adapter is None else adapter
        identity = resolved_adapter.resolve()
        capabilities = resolved_adapter.capabilities()
        availability = resolved_adapter.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason or "backend is unavailable")
        if not capabilities.local:
            raise BackendCompatibilityError(
                "vertical slice currently requires a local ideal simulator"
            )
        resolved_adapter.preflight(prepared.executable_circuits, spec.execution.shots)
        manifest = manifest.transition(
            "validated",
            timestamp=_timestamp(_clock),
            backend=_backend_snapshot(identity, capabilities),
            artifacts=tuple(artifacts),
        )
        _write_manifest(store, run, manifest)

        stage = "compilation"
        compiled = resolved_adapter.compile(
            prepared.executable_circuits, spec.execution.transpilation
        )
        if compiled.target_identity != identity:
            raise BackendCompatibilityError(
                "compiled target does not match resolved backend identity"
            )
        compiled_digest = store.write_circuits(
            run, compiled.circuits, "compiled-circuits.qpy"
        )
        artifacts.append(
            _artifact(
                run,
                role="compiled-circuits",
                filename="compiled-circuits.qpy",
                media_type="application/x-qiskit-qpy",
                digest=compiled_digest,
            )
        )
        manifest = manifest.transition(
            "compiled", timestamp=_timestamp(_clock), artifacts=tuple(artifacts)
        )
        _write_manifest(store, run, manifest)

        stage = "execution"
        submitted = resolved_adapter.submit(
            compiled.circuits, spec.execution.shots, options=None
        )
        if submitted.target_identity != identity:
            raise BackendCompatibilityError(
                "submitted target does not match resolved backend identity"
            )
        manifest = manifest.transition(
            "running",
            timestamp=_timestamp(_clock),
            jobs={"main": submitted.to_safe_dict()},
        )
        _write_manifest(store, run, manifest)
        executed = resolved_adapter.result(submitted)
        if executed.target_identity != identity:
            raise BackendCompatibilityError(
                "result target does not match resolved backend identity"
            )
        if len(executed.counts) != len(prepared.executable_circuits):
            raise BackendCompatibilityError(
                "result count batch does not match executable circuit count"
            )
        store.write_json(
            run,
            "counts.json",
            {
                "job_id": executed.job_id,
                "shots_per_circuit": spec.execution.shots,
                "counts_by_circuit": [dict(counts) for counts in executed.counts],
            },
        )
        artifacts.append(
            _artifact(
                run,
                role="counts",
                filename="counts.json",
                media_type="application/json",
            )
        )
        manifest = manifest.transition(
            "postprocessing",
            timestamp=_timestamp(_clock),
            artifacts=tuple(artifacts),
        )
        _write_manifest(store, run, manifest)

        stage = "postprocessing"
        result = prepared.postprocessor.evaluate(executed.counts)
        store.write_json(run, "result.json", result)
        artifacts.append(
            _artifact(
                run,
                role="result",
                filename="result.json",
                media_type="application/json",
            )
        )
        manifest = manifest.transition(
            "completed",
            timestamp=_timestamp(_clock),
            artifacts=tuple(artifacts),
            result=result,
        )
        _write_manifest(store, run, manifest)
        return QuditExperimentResult(run, manifest, result)
    except Exception as error:
        timestamp = _timestamp(_clock)
        try:
            manifest = manifest.transition(
                "failed",
                timestamp=timestamp,
                failure=_safe_failure(error, stage, timestamp),
            )
            _write_manifest(store, run, manifest)
        except Exception:
            pass
        try:
            setattr(error, "__qoq_artifact_dir__", run)
        except Exception:
            pass
        raise


def load_run_manifest(run_dir: Path | str) -> RunManifest:
    """Load a manifest and verify every referenced artifact stays intact."""

    try:
        run = Path(run_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ArtifactIntegrityError("run directory does not exist") from error
    if not run.is_dir() or len(run.parents) < 2:
        raise ArtifactIntegrityError("run directory is invalid")
    store = ExperimentStore(run.parents[1])
    try:
        manifest = RunManifest.from_safe_dict(
            store.read_json(run, _MANIFEST_FILENAME)
        )
    except ManifestValidationError:
        raise
    except Exception as error:
        raise ManifestValidationError("could not load run manifest") from error
    if manifest.run_id != run.name:
        raise ManifestValidationError("manifest run_id does not match run directory")
    resolved_run = run.resolve(strict=True)
    for artifact in manifest.artifacts:
        try:
            path = (run / artifact.path).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ArtifactIntegrityError(
                f"{artifact.role} artifact is missing"
            ) from error
        if not path.is_relative_to(resolved_run) or not path.is_file():
            raise ArtifactIntegrityError(
                f"{artifact.role} artifact escapes run directory"
            )
        if _file_hash(path) != artifact.sha256:
            raise ArtifactIntegrityError(f"{artifact.role} artifact hash mismatch")
    return manifest


__all__ = ["load_run_manifest", "run_vertical_slice"]
