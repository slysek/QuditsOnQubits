# SZY-43 Unified Provider Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit execution mode and immutable schema-v2 `RunManifest`, then prove that one scientific experiment completes through real Aer and mocked IQM and PiastQ provider contracts without network access.

**Architecture:** Backend specifications own the single serialized execution-mode value. A frozen `RunManifest` validates and deeply freezes every schema-v2 checkpoint, while deterministic built-in schema-v1 documents normalize in memory. The runner keeps its existing mutable orchestration document but must cross the manifest validation boundary before every atomic write.

**Tech Stack:** Python 3.10-3.13, stdlib `dataclasses`/`Enum`/`MappingProxyType`, Qiskit, Qiskit Aer, pytest, pytest-cov, existing `ExperimentStore`.

---

## File Structure

- Create `src/qudits_on_qubits/experiments/execution.py`: execution-mode enum plus internal backend-kind/identity contract helpers.
- Create `src/qudits_on_qubits/experiments/manifest.py`: immutable manifest, schema validation, v1 normalization, safe loader.
- Modify `src/qudits_on_qubits/experiments/models.py`: serialize execution mode for every backend spec and require it for `CustomBackend`.
- Modify `src/qudits_on_qubits/experiments/runner.py`: write schema v2, validate each checkpoint, load normalized manifests, reject adapter/spec mismatch.
- Modify `src/qudits_on_qubits/experiments/__init__.py`: export `ExecutionMode` and `RunManifest`.
- Modify `src/qudits_on_qubits/__init__.py`: add lazy top-level exports.
- Create `tests/test_experiment_manifest.py`: manifest round-trip, immutability, validation, migration, and loader tests.
- Create `tests/test_experiment_provider_contracts.py`: one full Aer/IQM/PiastQ runner contract.
- Modify `tests/test_experiment_models.py`: execution-mode model tests.
- Modify `tests/test_experiment_runner.py`: checkpoint validation and adapter-target mismatch tests.
- Modify `tests/test_experiment_aer_integration.py`: public API and README assertions.
- Modify `tests/test_experiment_backends.py`, `tests/test_experiment_batch.py`, `tests/test_experiment_noisy_adapter.py`, and `tests/test_experiment_resume.py`: pass explicit modes to existing custom backend fixtures.
- Modify `README.md`: public schema-v2, loader, migration, and custom backend usage.
- Read but do not modify `docs/superpowers/specs/2026-08-20-szy-43-unified-provider-contracts-design.md`: accepted design source.

### Task 1: Add explicit execution modes to backend specifications

**Files:**
- Create: `src/qudits_on_qubits/experiments/execution.py`
- Modify: `src/qudits_on_qubits/experiments/models.py:10-12,126-234`
- Modify: `tests/test_experiment_models.py:16-35,94-124,249-306`
- Modify: `tests/test_experiment_backends.py:1-45`
- Modify: `tests/test_experiment_batch.py:1-90`
- Modify: `tests/test_experiment_noisy_adapter.py:1-220`
- Modify: `tests/test_experiment_runner.py:27-35,131-142`
- Modify: `tests/test_experiment_resume.py:1-105`

- [ ] **Step 1: Write failing execution-mode model tests**

Add this import to `tests/test_experiment_models.py`:

```python
from qudits_on_qubits.experiments.execution import ExecutionMode
```

Add these tests after `test_backend_selections_are_validated`:

```python
@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (AerIdeal(), ExecutionMode.IDEAL_SIMULATOR),
        (NoisySimulator(source=object()), ExecutionMode.NOISY_SIMULATOR),
        (IQMHardware("garnet"), ExecutionMode.HARDWARE),
        (PiastQHardware(mode="managed"), ExecutionMode.HARDWARE),
        (
            CustomBackend(
                instance=object(),
                execution_mode=ExecutionMode.HARDWARE,
            ),
            ExecutionMode.HARDWARE,
        ),
    ],
)
def test_backend_specs_serialize_one_explicit_execution_mode(backend, expected):
    assert backend.execution_mode is expected
    assert backend.to_safe_dict()["execution_mode"] == expected.value


def test_custom_backend_requires_typed_execution_mode():
    with pytest.raises(TypeError, match="execution_mode"):
        CustomBackend(instance=object())

    with pytest.raises(ExperimentValidationError, match="execution_mode"):
        CustomBackend(instance=object(), execution_mode="hardware")


def test_backend_deserialization_rejects_conflicting_execution_mode():
    with pytest.raises(ExperimentValidationError, match="execution_mode"):
        AerIdeal.from_safe_dict(
            {
                "kind": "aer_ideal",
                "seed_simulator": 123,
                "execution_mode": "hardware",
            }
        )
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

Run:

```powershell
python -m pytest -q `
    tests/test_experiment_models.py::test_backend_specs_serialize_one_explicit_execution_mode `
    tests/test_experiment_models.py::test_custom_backend_requires_typed_execution_mode `
    tests/test_experiment_models.py::test_backend_deserialization_rejects_conflicting_execution_mode
```

Expected: collection fails with `ModuleNotFoundError: No module named 'qudits_on_qubits.experiments.execution'`.

- [ ] **Step 3: Implement the execution-mode contract**

Create `src/qudits_on_qubits/experiments/execution.py`:

```python
"""Execution provenance shared by experiment specifications and manifests."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .errors import ExperimentValidationError


class ExecutionMode(str, Enum):
    IDEAL_SIMULATOR = "ideal_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    HARDWARE = "hardware"


_FIXED_MODES = {
    "aer_ideal": ExecutionMode.IDEAL_SIMULATOR,
    "noisy_simulator": ExecutionMode.NOISY_SIMULATOR,
    "iqm_hardware": ExecutionMode.HARDWARE,
    "piastq_hardware": ExecutionMode.HARDWARE,
}
_IDENTITY_KINDS = {
    "aer_ideal": "aer_ideal",
    "noisy_simulator": "noisy",
    "iqm_hardware": "iqm",
    "piastq_hardware": "piastq",
    "custom": "custom",
}


def fixed_execution_mode(backend_kind: str) -> ExecutionMode | None:
    if backend_kind == "custom":
        return None
    try:
        return _FIXED_MODES[backend_kind]
    except KeyError:
        raise ExperimentValidationError("backend kind is unsupported") from None


def validate_backend_execution_mode(
    backend_kind: str,
    value: Any,
) -> ExecutionMode:
    try:
        mode = value if isinstance(value, ExecutionMode) else ExecutionMode(value)
    except (TypeError, ValueError):
        raise ExperimentValidationError("execution_mode is invalid") from None
    fixed = fixed_execution_mode(backend_kind)
    if fixed is not None and mode is not fixed:
        raise ExperimentValidationError(
            "execution_mode does not match backend kind"
        ) from None
    return mode


def expected_backend_identity_kind(backend_kind: str) -> str:
    try:
        return _IDENTITY_KINDS[backend_kind]
    except KeyError:
        raise ExperimentValidationError("backend kind is unsupported") from None


__all__ = ["ExecutionMode"]
```

In `src/qudits_on_qubits/experiments/models.py`, import `ClassVar`, `ExecutionMode`, and `validate_backend_execution_mode`:

```python
from typing import Any, ClassVar, Mapping

from .execution import ExecutionMode, validate_backend_execution_mode
```

Replace the five backend specification definitions with these exact mode-bearing sections while preserving their existing validation logic:

```python
@dataclass(frozen=True)
class AerIdeal:
    seed_simulator: int = 123
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.IDEAL_SIMULATOR

    def __post_init__(self) -> None:
        if isinstance(self.seed_simulator, bool) or not isinstance(self.seed_simulator, int):
            raise ExperimentValidationError("seed_simulator must be an integer")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "aer_ideal",
            "seed_simulator": self.seed_simulator,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "AerIdeal":
        validate_backend_execution_mode(
            "aer_ideal",
            data.get("execution_mode", ExecutionMode.IDEAL_SIMULATOR.value),
        )
        return cls(seed_simulator=data.get("seed_simulator", 123))
```

```python
@dataclass(frozen=True)
class IQMHardware:
    device: str
    use_metrics: bool = False
    env_path: Path | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.HARDWARE

    def __post_init__(self) -> None:
        _require_bool(self.use_metrics, "use_metrics")
        _safe_text(self.device, "device")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "iqm_hardware",
            "device": self.device,
            "use_metrics": self.use_metrics,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IQMHardware":
        validate_backend_execution_mode(
            "iqm_hardware",
            data.get("execution_mode", ExecutionMode.HARDWARE.value),
        )
        return cls(data["device"], data.get("use_metrics", False))
```

```python
@dataclass(frozen=True)
class PiastQHardware:
    mode: str = "auto"
    owner: str | None = None
    env_path: Path | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.HARDWARE

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "managed", "direct"}:
            raise ExperimentValidationError("mode must be auto, managed, or direct")
        if self.owner is not None:
            _safe_text(self.owner, "owner")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "piastq_hardware",
            "mode": self.mode,
            "owner": self.owner,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "PiastQHardware":
        validate_backend_execution_mode(
            "piastq_hardware",
            data.get("execution_mode", ExecutionMode.HARDWARE.value),
        )
        return cls(data.get("mode", "auto"), data.get("owner"))
```

```python
@dataclass(frozen=True)
class CustomBackend:
    instance: Any = field(repr=False, compare=False)
    identity: str = "custom"
    supports_resume: bool = False
    execution_mode: ExecutionMode = field(kw_only=True)

    def __post_init__(self) -> None:
        _safe_text(self.identity, "identity")
        _require_bool(self.supports_resume, "supports_resume")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ExperimentValidationError(
                "execution_mode must be ExecutionMode"
            ) from None

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "custom",
            "identity": self.identity,
            "supports_resume": self.supports_resume,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(
        cls,
        data: Mapping[str, Any],
        *,
        instance: Any = None,
    ) -> "CustomBackend":
        if instance is None:
            raise ExperimentValidationError(
                "custom backend reconstruction requires instance injection"
            )
        execution_mode = validate_backend_execution_mode(
            "custom",
            data.get("execution_mode"),
        )
        return cls(
            instance=instance,
            identity=data["identity"],
            supports_resume=data.get("supports_resume", False),
            execution_mode=execution_mode,
        )
```

```python
@dataclass(frozen=True)
class NoisySimulator:
    source: Any = field(default=None, repr=False, compare=False)
    noise_model: Any = field(default=None, repr=False, compare=False)
    target_backend: Any = field(default=None, repr=False, compare=False)
    identity: str | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.NOISY_SIMULATOR

    def __post_init__(self) -> None:
        source_mode = self.source is not None and self.noise_model is None and self.target_backend is None
        model_mode = self.source is None and self.noise_model is not None and self.target_backend is not None
        if not (source_mode or model_mode):
            raise ExperimentValidationError("provide exactly either source or noise_model with target_backend")
        if self.identity is not None:
            _safe_text(self.identity, "identity")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "noisy_simulator",
            "identity": self.identity,
            "source_mode": self.source is not None,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any], **injected: Any) -> "NoisySimulator":
        if not injected:
            raise ExperimentValidationError("noisy simulator reconstruction requires object injection")
        validate_backend_execution_mode(
            "noisy_simulator",
            data.get("execution_mode", ExecutionMode.NOISY_SIMULATOR.value),
        )
        source_mode = data.get("source_mode")
        if type(source_mode) is not bool:
            raise ExperimentValidationError("noisy simulator source_mode must be a boolean")
        reconstructed = cls(identity=data.get("identity"), **injected)
        if (reconstructed.source is not None) != source_mode:
            raise ExperimentValidationError("noisy simulator source_mode does not match injected objects")
        return reconstructed
```

- [ ] **Step 4: Update existing custom backend fixtures explicitly**

In `tests/test_experiment_backends.py`, extend the models import and add one exact helper below `_Backend`:

```python
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import AerIdeal, CustomBackend, TranspilationConfig


def _custom_backend(instance, **kwargs):
    return CustomBackend(
        instance,
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        **kwargs,
    )
```

Mechanically replace every `CustomBackend(` call in `tests/test_experiment_backends.py` with `_custom_backend(`. Keep the `CustomBackend` symbol inside `_custom_backend`; verify the result with:

```powershell
rg -n "CustomBackend\(" tests/test_experiment_backends.py
```

Expected: exactly one match, inside `_custom_backend`.

Apply these exact explicit arguments in other files:

```python
# tests/test_experiment_batch.py
from qudits_on_qubits.experiments.execution import ExecutionMode

backend=CustomBackend(
    object(),
    identity="batch-target",
    supports_resume=True,
    execution_mode=ExecutionMode.HARDWARE,
),
```

```python
# tests/test_experiment_noisy_adapter.py
from qudits_on_qubits.experiments.execution import ExecutionMode

create_backend_adapter(
    CustomBackend(
        source_backend,
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
    )
)
```

```python
# tests/test_experiment_runner.py, inside make_spec
from qudits_on_qubits.experiments.execution import ExecutionMode

"backend": CustomBackend(
    object(),
    identity="target",
    supports_resume=True,
    execution_mode=ExecutionMode.HARDWARE,
),
```

```python
# tests/test_experiment_resume.py, inside make_spec
from qudits_on_qubits.experiments.execution import ExecutionMode

backend=CustomBackend(
    object(),
    identity="resume-target",
    supports_resume=True,
    execution_mode=ExecutionMode.HARDWARE,
),
```

In `tests/test_experiment_models.py`, add `execution_mode=ExecutionMode.IDEAL_SIMULATOR` to every valid direct `CustomBackend` construction. Keep the missing-mode call in `test_custom_backend_requires_typed_execution_mode` unchanged. In `test_deserialization_does_not_coerce_boolean_strings`, use this complete payload so the test reaches boolean validation:

```python
CustomBackend.from_safe_dict(
    {
        "identity": "custom",
        "supports_resume": "false",
        "execution_mode": "ideal_simulator",
    },
    instance=object(),
)
```

- [ ] **Step 5: Run focused model and adapter tests**

Run:

```powershell
python -m pytest -q `
    tests/test_experiment_models.py `
    tests/test_experiment_backends.py `
    tests/test_experiment_batch.py `
    tests/test_experiment_noisy_adapter.py `
    tests/test_experiment_runner.py `
    tests/test_experiment_resume.py
```

Expected: all selected tests pass; no missing `execution_mode` constructor errors.

- [ ] **Step 6: Commit execution modes**

Run:

```powershell
git add `
    src/qudits_on_qubits/experiments/execution.py `
    src/qudits_on_qubits/experiments/models.py `
    tests/test_experiment_models.py `
    tests/test_experiment_backends.py `
    tests/test_experiment_batch.py `
    tests/test_experiment_noisy_adapter.py `
    tests/test_experiment_runner.py `
    tests/test_experiment_resume.py
git diff --cached --check
git commit -m "feat: add explicit backend execution modes"
```

Expected: one production contract commit plus required test-fixture migrations; no README or manifest files staged yet.

### Task 2: Add immutable schema-v2 RunManifest and v1 normalization

**Files:**
- Create: `src/qudits_on_qubits/experiments/manifest.py`
- Create: `tests/test_experiment_manifest.py`

- [ ] **Step 1: Write failing manifest model tests**

Create `tests/test_experiment_manifest.py`. Use this complete schema-v2 fixture:

```python
from __future__ import annotations

from copy import deepcopy

import pytest

from qudits_on_qubits.experiments.backends import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
)
from qudits_on_qubits.experiments.errors import (
    ExperimentPersistenceError,
    ExperimentValidationError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import AerIdeal, ExperimentSpec, PathBasis


TIMESTAMP = "2026-08-20T12:00:00.000000Z"


def resolved_backend(kind: str = "aer_ideal") -> dict[str, object]:
    identity = BackendIdentity(kind, "target", provider="test-provider")
    capabilities = BackendCapabilities(local=True, supports_resume=False)
    return {
        "identity": identity.to_safe_dict(),
        "capabilities": capabilities.to_safe_dict(),
        "metadata": {
            "identity": identity.to_safe_dict(),
            "capabilities": capabilities.to_safe_dict(),
        },
        "availability": Availability(True).to_safe_dict(),
    }


def manifest_document(
    *,
    status: str = "created",
    backend: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis("basis"),
        backend=AerIdeal(seed_simulator=11),
        shots=64,
        output_root="runs",
        tags={"purpose": "contract"},
    )
    history = [{"status": "created", "timestamp": TIMESTAMP}]
    timestamps = {"created": TIMESTAMP, "updated": TIMESTAMP}
    if status != "created":
        history.append({"status": status, "timestamp": TIMESTAMP})
        timestamps[status] = TIMESTAMP
    return {
        "schema_version": 2,
        "experiment_id": "20260820T120000.000000Z-contract-abcdef123456",
        "spec": spec.to_safe_dict(),
        "status": status,
        "timestamps": timestamps,
        "status_history": history,
        "attempts": [],
        "backend": backend,
        "jobs": {},
        "job_ids": [],
        "source": None,
        "circuits": {"source": None, "logical": None, "factors": {}},
        "counts": {},
        "postprocessing": None,
        "calibration": None,
        "result": None,
        "result_artifact": None,
        "failure": None,
    }
```

Append these exact tests:

```python
def test_manifest_round_trip_is_deeply_immutable_and_returns_fresh_copies():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["circuits"]["factors"]["1"] = {
        "artifact": "compiled-factor-1.qpy",
        "sha256": "a" * 64,
        "circuit_count": 1,
    }
    manifest = RunManifest.from_safe_dict(document)
    document["circuits"]["factors"]["1"]["artifact"] = "changed.qpy"
    copy = manifest.to_safe_dict()
    copy["circuits"]["factors"]["1"]["artifact"] = "copy.qpy"

    assert manifest.schema_version == 2
    assert manifest.execution_mode is ExecutionMode.IDEAL_SIMULATOR
    assert manifest.circuits["factors"]["1"]["artifact"] == "compiled-factor-1.qpy"
    assert manifest.to_safe_dict()["circuits"]["factors"]["1"]["artifact"] == "compiled-factor-1.qpy"
    with pytest.raises(TypeError):
        manifest.circuits["factors"]["1"]["artifact"] = "changed.qpy"


@pytest.mark.parametrize(
    ("backend_kind", "expected_mode"),
    [
        ("aer_ideal", "ideal_simulator"),
        ("noisy_simulator", "noisy_simulator"),
        ("iqm_hardware", "hardware"),
        ("piastq_hardware", "hardware"),
    ],
)
def test_schema_v1_builtin_backend_normalizes_in_memory(backend_kind, expected_mode):
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["schema_version"] = 1
    document["spec"]["backend"] = {"kind": backend_kind}
    manifest = RunManifest.from_safe_dict(document)

    assert manifest.schema_version == 2
    assert manifest.to_safe_dict()["spec"]["backend"]["execution_mode"] == expected_mode
    assert "execution_mode" not in document["spec"]["backend"]


def test_schema_v1_custom_backend_is_rejected_without_guessing():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["schema_version"] = 1
    document["spec"]["backend"] = {
        "kind": "custom",
        "identity": "legacy",
        "supports_resume": True,
    }
    with pytest.raises(ExperimentPersistenceError, match="custom"):
        RunManifest.from_safe_dict(document)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda item: item.update(schema_version=99), ExperimentPersistenceError, "schema"),
        (lambda item: item.update(status="unknown"), ExperimentValidationError, "status"),
        (
            lambda item: item["spec"]["backend"].update(execution_mode="hardware"),
            ExperimentValidationError,
            "execution_mode",
        ),
        (
            lambda item: item["spec"]["tags"].update(purpose="token=manifest-secret"),
            ExperimentValidationError,
            "unsafe",
        ),
        (
            lambda item: item["circuits"]["factors"].update(
                {"1": {"artifact": "../escape.qpy", "sha256": "a" * 64}}
            ),
            ExperimentValidationError,
            "artifact",
        ),
    ],
)
def test_manifest_rejects_invalid_schema_state_mode_secret_and_artifact(
    mutation,
    error_type,
    message,
):
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    mutation(document)
    with pytest.raises(error_type, match=message) as caught:
        RunManifest.from_safe_dict(document)
    assert "manifest-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_manifest_rejects_resolved_identity_that_disagrees_with_spec():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document(status="validated", backend=resolved_backend("custom"))
    with pytest.raises(ExperimentValidationError, match="identity"):
        RunManifest.from_safe_dict(document)


def test_manifest_does_not_mutate_caller_during_validation():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    before = deepcopy(document)
    RunManifest.from_safe_dict(document)
    assert document == before
```

- [ ] **Step 2: Run manifest tests and verify failure before implementation**

Run:

```powershell
python -m pytest -q tests/test_experiment_manifest.py
```

Expected: collection fails with `No module named 'qudits_on_qubits.experiments.manifest'`.

- [ ] **Step 3: Implement immutable model, validation, and normalization**

Create `src/qudits_on_qubits/experiments/manifest.py` with these complete foundations:

```python
"""Immutable validation boundary for durable experiment manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .errors import ExperimentPersistenceError, ExperimentValidationError
from .execution import (
    ExecutionMode,
    expected_backend_identity_kind,
    fixed_execution_mode,
    validate_backend_execution_mode,
)
from .models import ExperimentStatus
from .safety import validate_persisted_strings


MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema_version", "experiment_id", "spec", "status", "timestamps",
    "status_history", "attempts", "backend", "jobs", "job_ids", "source",
    "circuits", "counts", "postprocessing", "calibration", "result",
    "result_artifact", "failure",
}


def _freeze_json(value: Any, description: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentValidationError(f"{description} contains a non-finite float") from None
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentValidationError(f"{description} mapping keys must be strings") from None
            frozen[key] = _freeze_json(item, description)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, description) for item in value)
    raise ExperimentValidationError(f"{description} contains an unsupported value") from None


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{description} must be a mapping") from None
    return value


def _sequence(value: Any, description: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentValidationError(f"{description} must be a sequence") from None
    return value


def _timestamp(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExperimentValidationError(f"{description} is invalid") from None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ExperimentValidationError(f"{description} is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExperimentValidationError(f"{description} is invalid") from None
    return value


def _artifact_ref(value: Any, description: str) -> None:
    record = _mapping(value, f"{description} artifact")
    artifact = record.get("artifact")
    digest = record.get("sha256")
    if not isinstance(artifact, str):
        raise ExperimentValidationError(f"{description} artifact name is invalid") from None
    relative = Path(artifact)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ExperimentValidationError(f"{description} artifact name is unsafe") from None
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ExperimentValidationError(f"{description} artifact hash is invalid") from None


def _normalize_document(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_persisted_strings(value, description="manifest", error_type=ExperimentValidationError)
    document = _thaw_json(_freeze_json(value, "manifest"))
    version = document.get("schema_version")
    if version == MANIFEST_SCHEMA_VERSION:
        return document
    if version != LEGACY_MANIFEST_SCHEMA_VERSION:
        raise ExperimentPersistenceError("unsupported experiment schema version") from None
    spec = _mapping(document.get("spec"), "manifest spec")
    backend = dict(_mapping(spec.get("backend"), "manifest backend spec"))
    kind = backend.get("kind")
    if kind == "custom":
        raise ExperimentPersistenceError("schema-v1 custom backend execution mode is ambiguous") from None
    try:
        mode = fixed_execution_mode(kind)
    except ExperimentValidationError:
        raise ExperimentPersistenceError("schema-v1 backend kind is unsupported") from None
    if mode is None:
        raise ExperimentPersistenceError("schema-v1 backend execution mode is ambiguous") from None
    backend["execution_mode"] = mode.value
    mutable_spec = dict(spec)
    mutable_spec["backend"] = backend
    document["spec"] = mutable_spec
    document["schema_version"] = MANIFEST_SCHEMA_VERSION
    document.setdefault("source", None)
    document.setdefault("result_artifact", None)
    return document
```

Continue `manifest.py` with exact validation of mode, lifecycle, jobs, and artifacts:

```python
def _validate_document(document: Mapping[str, Any]) -> None:
    if set(document) != _FIELDS:
        raise ExperimentValidationError("manifest fields do not match schema v2") from None
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExperimentValidationError("manifest schema version is invalid") from None
    experiment_id = document.get("experiment_id")
    if not isinstance(experiment_id, str) or not _RUN_ID.fullmatch(experiment_id):
        raise ExperimentValidationError("experiment_id is invalid") from None
    spec = _mapping(document.get("spec"), "manifest spec")
    backend_spec = _mapping(spec.get("backend"), "manifest backend spec")
    backend_kind = backend_spec.get("kind")
    mode = validate_backend_execution_mode(backend_kind, backend_spec.get("execution_mode"))
    if mode.value != backend_spec.get("execution_mode"):
        raise ExperimentValidationError("execution_mode is not canonical") from None
    try:
        status = ExperimentStatus(document.get("status"))
    except (TypeError, ValueError):
        raise ExperimentValidationError("experiment status is invalid") from None
    timestamps = _mapping(document.get("timestamps"), "manifest timestamps")
    _timestamp(timestamps.get("created"), "created timestamp")
    _timestamp(timestamps.get("updated"), "updated timestamp")
    history = _sequence(document.get("status_history"), "status_history")
    if not history:
        raise ExperimentValidationError("status_history must not be empty") from None
    states: list[ExperimentStatus] = []
    for entry in history:
        record = _mapping(entry, "status_history entry")
        try:
            states.append(ExperimentStatus(record.get("status")))
        except (TypeError, ValueError):
            raise ExperimentValidationError("status_history status is invalid") from None
        _timestamp(record.get("timestamp"), "status_history timestamp")
    if states[0] is not ExperimentStatus.CREATED or states[-1] is not status:
        raise ExperimentValidationError("status_history does not match status") from None
    for entry in _sequence(document.get("attempts"), "attempts"):
        record = _mapping(entry, "attempt record")
        attempt = record.get("attempt")
        if not isinstance(record.get("operation"), str):
            raise ExperimentValidationError("attempt operation is invalid") from None
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise ExperimentValidationError("attempt number is invalid") from None
        if record.get("outcome") not in {"failed", "succeeded"}:
            raise ExperimentValidationError("attempt outcome is invalid") from None
        _timestamp(record.get("timestamp"), "attempt timestamp")
    backend = document.get("backend")
    if backend is not None:
        backend_record = _mapping(backend, "resolved backend")
        for name in ("identity", "capabilities", "metadata", "availability"):
            _mapping(backend_record.get(name), f"backend {name}")
        identity = _mapping(backend_record["identity"], "backend identity")
        if identity.get("kind") != expected_backend_identity_kind(backend_kind):
            raise ExperimentValidationError("resolved backend identity does not match spec") from None
    jobs = _mapping(document.get("jobs"), "jobs")
    recorded_ids: list[str] = []
    for job_key, value in jobs.items():
        if not isinstance(job_key, str):
            raise ExperimentValidationError("job key is invalid") from None
        record = _mapping(value, "job record")
        job_id = record.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or not job_id:
                raise ExperimentValidationError("job_id is invalid") from None
            recorded_ids.append(job_id)
    job_ids = _sequence(document.get("job_ids"), "job_ids")
    if any(not isinstance(item, str) or not item for item in job_ids):
        raise ExperimentValidationError("job_ids are invalid") from None
    if list(job_ids) != recorded_ids:
        raise ExperimentValidationError("job_ids do not match job records") from None
    circuits = _mapping(document.get("circuits"), "circuits")
    if set(circuits) != {"source", "logical", "factors"}:
        raise ExperimentValidationError("circuit manifest fields are invalid") from None
    for name in ("source", "logical"):
        if circuits[name] is not None:
            _artifact_ref(circuits[name], f"{name} circuit")
    for factor, record in _mapping(circuits["factors"], "factor circuits").items():
        if not isinstance(factor, str) or not factor.isdigit():
            raise ExperimentValidationError("factor circuit key is invalid") from None
        _artifact_ref(record, f"factor {factor} circuit")
    for factor, record in _mapping(document.get("counts"), "counts").items():
        if not isinstance(factor, str) or not factor.isdigit():
            raise ExperimentValidationError("counts factor key is invalid") from None
        _artifact_ref(record, f"factor {factor} counts")
    for name in ("postprocessing", "result_artifact"):
        if document.get(name) is not None:
            _artifact_ref(document[name], name)
    for name in ("source", "calibration", "result", "failure"):
        if document.get(name) is not None:
            _mapping(document[name], name)
    if status is ExperimentStatus.COMPLETED:
        if document.get("result") is None or document.get("result_artifact") is None:
            raise ExperimentValidationError("completed manifest requires result artifacts") from None
        if document.get("failure") is not None:
            raise ExperimentValidationError("completed manifest must not contain failure") from None
    if status is ExperimentStatus.FAILED and document.get("failure") is None:
        raise ExperimentValidationError("failed manifest requires failure") from None
```

Finish `manifest.py` with the complete frozen model:

```python
@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    experiment_id: str
    spec: Mapping[str, Any]
    status: ExperimentStatus
    timestamps: Mapping[str, Any]
    status_history: tuple[Mapping[str, Any], ...]
    attempts: tuple[Mapping[str, Any], ...]
    backend: Mapping[str, Any] | None
    jobs: Mapping[str, Any]
    job_ids: tuple[str, ...]
    source: Mapping[str, Any] | None
    circuits: Mapping[str, Any]
    counts: Mapping[str, Any]
    postprocessing: Mapping[str, Any] | None
    calibration: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    result_artifact: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        for name in (
            "spec", "timestamps", "status_history", "attempts", "backend", "jobs",
            "job_ids", "source", "circuits", "counts", "postprocessing",
            "calibration", "result", "result_artifact", "failure",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name), f"manifest {name}"))
        _validate_document(self.to_safe_dict())

    @property
    def execution_mode(self) -> ExecutionMode:
        backend = _mapping(self.spec.get("backend"), "manifest backend spec")
        return validate_backend_execution_mode(backend.get("kind"), backend.get("execution_mode"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "spec": _thaw_json(self.spec),
            "status": self.status.value,
            "timestamps": _thaw_json(self.timestamps),
            "status_history": _thaw_json(self.status_history),
            "attempts": _thaw_json(self.attempts),
            "backend": _thaw_json(self.backend),
            "jobs": _thaw_json(self.jobs),
            "job_ids": _thaw_json(self.job_ids),
            "source": _thaw_json(self.source),
            "circuits": _thaw_json(self.circuits),
            "counts": _thaw_json(self.counts),
            "postprocessing": _thaw_json(self.postprocessing),
            "calibration": _thaw_json(self.calibration),
            "result": _thaw_json(self.result),
            "result_artifact": _thaw_json(self.result_artifact),
            "failure": _thaw_json(self.failure),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(data, Mapping):
            raise ExperimentValidationError("manifest must be a mapping") from None
        document = _normalize_document(data)
        _validate_document(document)
        return cls(
            schema_version=document["schema_version"],
            experiment_id=document["experiment_id"],
            spec=document["spec"],
            status=ExperimentStatus(document["status"]),
            timestamps=document["timestamps"],
            status_history=tuple(document["status_history"]),
            attempts=tuple(document["attempts"]),
            backend=document["backend"],
            jobs=document["jobs"],
            job_ids=tuple(document["job_ids"]),
            source=document["source"],
            circuits=document["circuits"],
            counts=document["counts"],
            postprocessing=document["postprocessing"],
            calibration=document["calibration"],
            result=document["result"],
            result_artifact=document["result_artifact"],
            failure=document["failure"],
        )


__all__ = ["MANIFEST_SCHEMA_VERSION", "RunManifest"]
```

- [ ] **Step 4: Run focused manifest tests**

Run:

```powershell
python -m pytest -q tests/test_experiment_manifest.py
```

Expected: all Task 2 tests pass.

- [ ] **Step 5: Commit immutable manifest model**

Run:

```powershell
git add `
    src/qudits_on_qubits/experiments/manifest.py `
    tests/test_experiment_manifest.py
git diff --cached --check
git commit -m "feat: add immutable run manifest"
```

Expected: only manifest model and focused tests committed.

### Task 3: Add safe RunManifest loading without implicit rewrites

**Files:**
- Modify: `src/qudits_on_qubits/experiments/manifest.py`
- Modify: `tests/test_experiment_manifest.py`

- [ ] **Step 1: Write failing loader tests**

Add this import to `tests/test_experiment_manifest.py`:

```python
from qudits_on_qubits.experiments.store import ExperimentStore
```

Append these tests:

```python
def test_manifest_load_normalizes_v1_without_rewriting_file(tmp_path):
    from qudits_on_qubits.experiments.manifest import RunManifest

    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run("legacy-load")
    document = manifest_document()
    document["experiment_id"] = run.name
    document["schema_version"] = 1
    document["spec"]["backend"].pop("execution_mode")
    document.pop("source")
    document.pop("result_artifact")
    path = store.write_experiment(run, document)
    before = path.read_bytes()

    manifest = RunManifest.load(run)

    assert manifest.schema_version == 2
    assert manifest.execution_mode is ExecutionMode.IDEAL_SIMULATOR
    assert path.read_bytes() == before


def test_manifest_load_wraps_invalid_field_without_leaking_payload(tmp_path):
    from qudits_on_qubits.experiments.manifest import RunManifest

    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run("invalid-load")
    document = manifest_document()
    document["experiment_id"] = run.name
    document["spec"]["tags"] = {"purpose": "token=loader-secret"}
    store.write_experiment(run, document)

    with pytest.raises(ExperimentPersistenceError, match="manifest") as caught:
        RunManifest.load(run)
    assert "loader-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_manifest_load_rejects_directory_id_mismatch(tmp_path):
    from qudits_on_qubits.experiments.manifest import RunManifest

    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run("id-mismatch")
    store.write_experiment(run, manifest_document())

    with pytest.raises(ExperimentPersistenceError, match="ID"):
        RunManifest.load(run)


def test_manifest_load_rejects_missing_run_directory(tmp_path):
    from qudits_on_qubits.experiments.manifest import RunManifest

    with pytest.raises(ExperimentPersistenceError, match="run directory"):
        RunManifest.load(tmp_path / "missing")
```

- [ ] **Step 2: Run loader tests and verify failure**

Run:

```powershell
python -m pytest -q tests/test_experiment_manifest.py -k "load"
```

Expected: failures report `AttributeError: type object 'RunManifest' has no attribute 'load'`.

- [ ] **Step 3: Implement safe loader**

Add this import to `manifest.py`:

```python
from .store import ExperimentStore
```

Add this complete classmethod after `from_safe_dict`:

```python
    @classmethod
    def load(cls, artifact_dir: Path | str) -> "RunManifest":
        try:
            run = Path(artifact_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError):
            raise ExperimentPersistenceError(
                "artifact_dir must identify an existing run directory"
            ) from None
        if not run.is_dir() or len(run.parents) < 2:
            raise ExperimentPersistenceError(
                "artifact_dir must identify a run directory"
            ) from None
        store = ExperimentStore(run.parents[1])
        document = store.read_experiment(run)
        try:
            manifest = cls.from_safe_dict(document)
        except ExperimentPersistenceError:
            raise
        except ExperimentValidationError:
            raise ExperimentPersistenceError("experiment manifest is invalid") from None
        if manifest.experiment_id != run.name:
            raise ExperimentPersistenceError(
                "experiment ID does not match run directory"
            ) from None
        return manifest
```

- [ ] **Step 4: Run all manifest tests**

Run:

```powershell
python -m pytest -q tests/test_experiment_manifest.py
```

Expected: all manifest model and loader tests pass; v1 file bytes remain unchanged.

- [ ] **Step 5: Commit loader behavior**

Run:

```powershell
git add `
    src/qudits_on_qubits/experiments/manifest.py `
    tests/test_experiment_manifest.py
git diff --cached --check
git commit -m "feat: load versioned run manifests"
```

Expected: one focused loader/migration commit.

### Task 4: Route runner checkpoints and resume through `RunManifest`

**Files:**

- Modify: `src/qudits_on_qubits/experiments/runner.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_experiment_resume.py`

- [ ] **Step 1: Add failing runner-boundary tests**

In `tests/test_experiment_runner.py`, change
`test_initial_document_is_validated_before_first_persistent_write` so it expects
`ExperimentPersistenceError`, then add these tests beside the existing persistence
tests:

```python
def test_every_runner_checkpoint_crosses_manifest_validation_boundary(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.manifest import RunManifest
    from qudits_on_qubits.experiments.runner import run_experiment
    from qudits_on_qubits.experiments.store import ExperimentStore

    validation_statuses: list[str] = []
    write_statuses: list[str] = []
    original_from_safe_dict = RunManifest.from_safe_dict.__func__
    original_write_experiment = ExperimentStore.write_experiment

    def recording_from_safe_dict(cls, document):
        validation_statuses.append(document["status"])
        return original_from_safe_dict(cls, document)

    def recording_write_experiment(
        self, run, document=None, **fields_by_name
    ):
        assert document is not None
        write_statuses.append(document["status"])
        assert validation_statuses == write_statuses
        return original_write_experiment(
            self, run, document, **fields_by_name
        )

    monkeypatch.setattr(
        RunManifest,
        "from_safe_dict",
        classmethod(recording_from_safe_dict),
    )
    monkeypatch.setattr(
        ExperimentStore,
        "write_experiment",
        recording_write_experiment,
    )

    result = run_experiment(
        make_spec(tmp_path),
        adapter=RecordingAdapter(),
        _sleep=lambda _: None,
        _evaluator=lambda _: 1.0 + 0.0j,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert validation_statuses == write_statuses
    assert validation_statuses[0] == "created"
    assert validation_statuses[-1] == "completed"


def test_runner_rejects_adapter_identity_that_disagrees_with_backend(
    tmp_path,
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    spec = make_spec(tmp_path, backend=IQMHardware("garnet"))

    with pytest.raises(
        BackendCompatibilityError,
        match="resolved adapter identity does not match configured backend",
    ):
        run_experiment(
            spec,
            adapter=adapter,
            _sleep=lambda _: None,
            _evaluator=lambda _: 1.0 + 0.0j,
        )

    assert adapter.submit_calls == 0
```

Import `ExperimentPersistenceError`, `IQMHardware`, and
`BackendCompatibilityError` from their existing modules. Adapt only the local
fixture argument names if the test file uses a different established name.

In `tests/test_experiment_resume.py`, extend the existing v1 completed-run test:

```python
    before = experiment_path.read_bytes()
    result = resume_experiment(run)
    assert experiment_path.read_bytes() == before
    assert result.status is ExperimentStatus.COMPLETED
```

This locks the no-rewrite migration behavior through the public resume path.

- [ ] **Step 2: Run the runner tests and verify failure**

Run:

```powershell
python -m pytest -q tests/test_experiment_runner.py tests/test_experiment_resume.py
```

Expected: the new manifest-boundary test sees no validations, the identity
mismatch reaches the adapter, and the v1 resume assertion fails until the runner
loads normalized manifests.

- [ ] **Step 3: Integrate the manifest at every persistence boundary**

In `runner.py`, import:

```python
from .execution import expected_backend_identity_kind
from .manifest import MANIFEST_SCHEMA_VERSION, RunManifest
```

Remove the local `SCHEMA_VERSION = 1`. In `_initial_document`, use
`MANIFEST_SCHEMA_VERSION`, add the stable optional fields, and remove the direct
persisted-string validation call:

```python
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": None,
        "result_artifact": None,
```

Replace `_write_state` with:

```python
def _write_state(
    store: ExperimentStore,
    run: Path,
    document: Mapping[str, object],
) -> None:
    try:
        manifest = RunManifest.from_safe_dict(document)
    except ExperimentPersistenceError:
        raise
    except ExperimentValidationError:
        raise ExperimentPersistenceError("experiment manifest is invalid") from None
    store.write_experiment(run, manifest.to_safe_dict())
```

Add the adapter/backend target check next to `_validate_adapter`:

```python
def _validate_adapter_target(
    spec: ExperimentSpec,
    identity: BackendIdentity,
) -> None:
    backend_kind = spec.backend.to_safe_dict().get("kind")
    try:
        expected_kind = expected_backend_identity_kind(backend_kind)
    except ExperimentValidationError:
        raise BackendCompatibilityError(
            "configured backend kind is unsupported"
        ) from None
    if identity.kind != expected_kind:
        raise BackendCompatibilityError(
            "resolved adapter identity does not match configured backend"
        )
```

Call it immediately after every successful `_validate_adapter(...)` call in both
`run_experiment` and `resume_experiment`, before durable-job recovery, compilation,
or submission:

```python
    _validate_adapter_target(spec, identity)
```

Replace `_open_run` with a manifest-backed read that returns the normalized v2
document in memory:

```python
def _open_run(
    artifact_dir: Path | str,
) -> tuple[ExperimentStore, Path, dict[str, object]]:
    manifest = RunManifest.load(artifact_dir)
    run = Path(artifact_dir).expanduser().resolve(strict=True)
    store = ExperimentStore(run.parents[1])
    return store, run, manifest.to_safe_dict()
```

Delete runner-local schema checks now owned by `RunManifest`. Keep runtime state
transition checks and adapter-result validation in the runner.

- [ ] **Step 4: Run runner and resume tests**

Run:

```powershell
python -m pytest -q tests/test_experiment_runner.py tests/test_experiment_resume.py
```

Expected: all tests pass. Every write has one preceding manifest validation; an
IQM spec cannot run against a custom adapter; completed v1 resume does not rewrite
`experiment.json`.

- [ ] **Step 5: Commit runner integration**

Run:

```powershell
git add `
    src/qudits_on_qubits/experiments/runner.py `
    tests/test_experiment_runner.py `
    tests/test_experiment_resume.py
git diff --cached --check
git commit -m "feat: validate runner manifest checkpoints"
```

Expected: one focused runner-boundary commit.

### Task 5: Prove one runner contract with real Aer, IQM, and PiastQ adapters

**Files:**

- Create: `tests/test_experiment_provider_contracts.py`

- [ ] **Step 1: Add fake provider primitives with a hard network guard**

Create `tests/test_experiment_provider_contracts.py`. Reuse the basis/QPY helper
shape from `tests/test_experiment_aer_integration.py`, then add:

```python
def reject_network(*args, **kwargs):
    raise AssertionError("provider contract tests must not use the network")


def install_network_guard(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


class FakeProviderResult:
    def __init__(self, circuits, shots):
        self._counts = [
            {"0" * max(1, circuit.num_clbits): shots}
            for circuit in circuits
        ]

    def get_counts(self, index):
        return self._counts[index]
```

Add an IQM backend/job pair implementing only the members used by `IQMAdapter`:
`name`, `num_qubits`, calibration/version metadata, `run`, `retrieve_job`,
`job_id`, `status`, and `result`. Inject it with a pass-manager factory whose
`run` returns the input circuits.

Add PiastQ client/sampler/job fakes implementing only the members used by
`PiastQAdapter`: backend discovery, `run`, `retrieve_job`, `job_id`, `status`, and
`result`. Inject them with `client_type`, `sampler_type`, `env_loader=lambda: {}`,
and a patched `piastq.transpile` that returns the input circuits. In the fake
sampler's `run`, assert that a new `compiled-factor-1.qpy` exists under the current
run directory before accepting submission. Return deterministic all-zero counts.

- [ ] **Step 2: Add the failing three-provider contract test**

Build one `ExperimentSpec` with a four-level basis, one factor, deterministic
bootstrap (`samples=2`, fixed seed), and `AerIdeal()`. Derive the two hardware
specs only with `dataclasses.replace`:

```python
iqm_spec = replace(base_spec, backend=IQMHardware("garnet"))
piastq_spec = replace(base_spec, backend=PiastQHardware("piastq-main"))
```

Run these cases through the same public `run_experiment` entry point:

```python
cases = (
    (base_spec, None, "ideal_simulator", "aer_ideal"),
    (iqm_spec, iqm_adapter, "hardware", "iqm"),
    (piastq_spec, piastq_adapter, "hardware", "piastq"),
)

manifests = []
for spec, adapter, expected_mode, expected_identity in cases:
    result = run_experiment(spec, adapter=adapter, _sleep=lambda _: None)
    manifest = RunManifest.load(result.artifact_dir)
    assert result.status is ExperimentStatus.COMPLETED
    assert manifest.schema_version == 2
    assert manifest.execution_mode is ExecutionMode(expected_mode)
    assert manifest.backend["kind"] == expected_identity
    assert manifest.result_artifact is not None
    assert (result.artifact_dir / manifest.result_artifact["path"]).is_file()
    compiled_name = manifest.circuits["factors"]["1"]["artifact"]
    assert compiled_name == "compiled-factor-1.qpy"
    assert (result.artifact_dir / compiled_name).is_file()
    assert "token" not in repr(manifest.to_safe_dict()).lower()
    manifests.append(manifest.to_safe_dict())
```

Then assert:

```python
assert {frozenset(document) for document in manifests} == {
    frozenset(manifests[0])
}

scientific_specs = []
for document in manifests:
    serialized_spec = copy.deepcopy(document["spec"])
    serialized_spec.pop("backend")
    scientific_specs.append(serialized_spec)
assert scientific_specs[0] == scientific_specs[1] == scientific_specs[2]
```

Also verify each referenced artifact's `sha256` against the file bytes. Keep all
credentials fake and all provider calls local.

- [ ] **Step 3: Run the contract test and verify failure**

Run:

```powershell
python -m pytest -q tests/test_experiment_provider_contracts.py
```

Expected: collection or assertions fail until `ExecutionMode`, `RunManifest`, and
the runner manifest integration from Tasks 1–4 exist.

- [ ] **Step 4: Complete only the fake protocol surface demanded by failures**

Run the same command after each fake-provider adjustment. Implement no production
fallback and make no network request. Stop when all three real adapters complete
through the same runner and produce the same manifest structure.

- [ ] **Step 5: Commit the provider contract**

Run:

```powershell
git add tests/test_experiment_provider_contracts.py
git diff --cached --check
git commit -m "test: add unified provider contracts"
```

Expected: one test-only commit proving Aer, IQM, and PiastQ parity.

### Task 6: Expose and document the public contract

**Files:**

- Modify: `src/qudits_on_qubits/experiments/__init__.py`
- Modify: `src/qudits_on_qubits/__init__.py`
- Modify: `tests/test_experiment_aer_integration.py`
- Modify: `tests/test_experiment_readme.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing public-export and README tests**

In the expected export set in `tests/test_experiment_aer_integration.py`, add:

```python
"ExecutionMode",
"RunManifest",
```

Extend the lazy-import subprocess assertion so both names are imported from the
top-level `qudits_on_qubits` package before optional provider modules are checked.
Add both names to the subprocess's expected top-level export tuple.

In `tests/test_experiment_readme.py`, add assertions that the README contains:

```python
"ExecutionMode",
"RunManifest.load(result.artifact_dir)",
"execution_mode=ExecutionMode.HARDWARE",
```

- [ ] **Step 2: Run export/documentation tests and verify failure**

Run:

```powershell
python -m pytest -q tests/test_experiment_aer_integration.py tests/test_experiment_readme.py
```

Expected: missing public exports and README snippets fail.

- [ ] **Step 3: Export the new public types**

In `src/qudits_on_qubits/experiments/__init__.py`, import and add to `__all__`:

```python
from .execution import ExecutionMode
from .manifest import RunManifest
```

In `src/qudits_on_qubits/__init__.py`, add both names to `_EXPERIMENT_EXPORTS` and
`__all__`:

```python
"ExecutionMode",
"RunManifest",
```

Preserve lazy optional-provider imports at both package levels.

- [ ] **Step 4: Document explicit custom modes and manifest loading**

Update the README experiment import block to include `ExecutionMode` and
`RunManifest`. Change every `CustomBackend(...)` example to pass an explicit
keyword-only mode, for example:

```python
backend=CustomBackend(
    "lab-adapter",
    execution_mode=ExecutionMode.HARDWARE,
),
```

After the run example, add:

```python
manifest = RunManifest.load(result.artifact_dir)
print(manifest.execution_mode.value)
print(manifest.to_safe_dict()["artifacts"])
```

State that schema v2 stores the mode once at
`spec.backend.execution_mode`; `manifest.execution_mode` is a typed convenience
property. State that built-in v1 manifests migrate only in memory and ambiguous
v1 custom manifests are rejected.

- [ ] **Step 5: Run export/documentation tests**

Run:

```powershell
python -m pytest -q tests/test_experiment_aer_integration.py tests/test_experiment_readme.py
```

Expected: all tests pass without importing IQM or PiastQ SDKs on package import.

- [ ] **Step 6: Commit the public API and documentation**

Run:

```powershell
git add `
    src/qudits_on_qubits/__init__.py `
    src/qudits_on_qubits/experiments/__init__.py `
    tests/test_experiment_aer_integration.py `
    tests/test_experiment_readme.py `
    README.md
git diff --cached --check
git commit -m "feat: expose the run manifest API"
```

Expected: one focused public-contract commit.

### Task 7: Verify coverage, regression safety, and final diff

**Files:**

- Review: all files changed from `origin/main`

- [ ] **Step 1: Run focused coverage gate**

Run:

```powershell
python -m pytest -q `
    tests/test_experiment_models.py `
    tests/test_experiment_manifest.py `
    tests/test_experiment_runner.py `
    tests/test_experiment_resume.py `
    tests/test_experiment_provider_contracts.py `
    --cov=qudits_on_qubits.experiments.execution `
    --cov=qudits_on_qubits.experiments.manifest `
    --cov=qudits_on_qubits.experiments.runner `
    --cov-report=term-missing `
    --cov-fail-under=80
```

Expected: exit 0 and total coverage at least 80% for the new execution/manifest
code plus changed runner paths. If a branch is missed, add a behavior test before
changing production code.

- [ ] **Step 2: Run the complete regression suite**

Run:

```powershell
python -m pytest -q
```

Expected: at least the baseline `703 passed, 3 skipped, 315 subtests passed`, plus
the new tests; zero failures.

- [ ] **Step 3: Compile changed production modules**

Run:

```powershell
python -m py_compile `
    src/qudits_on_qubits/experiments/execution.py `
    src/qudits_on_qubits/experiments/manifest.py `
    src/qudits_on_qubits/experiments/model.py `
    src/qudits_on_qubits/experiments/runner.py
```

Expected: exit 0 with no output.

- [ ] **Step 4: Simplify only without behavior changes**

Invoke the `code-simplifier` skill on the changed production and test files. Keep
the public API, schema, error categories, and test behavior unchanged. Re-run the
focused coverage command and the complete suite after any simplification.

- [ ] **Step 5: Review the final diff**

Run:

```powershell
git diff origin/main...HEAD --check
git diff --stat origin/main...HEAD
git status --short
git log --oneline --decorate origin/main..HEAD
```

Expected: no whitespace errors; only SZY-43 files changed; worktree clean; focused
commits in task order. If review finds a defect, return to its owning TDD task,
add a regression test, apply the smallest fix, rerun both verification commands,
and commit that correction before reporting completion.

- [ ] **Step 6: Verify before claiming completion**

Invoke `superpowers:verification-before-completion`. Report the exact coverage,
full-suite counts, branch name, and commits. Do not push, open a PR, or mutate
Linear without separate user authorization.
