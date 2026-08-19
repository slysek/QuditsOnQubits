"""Contract tests for the generic qudit vertical-slice models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pytest

from qudits_on_qubits.experiments.models import AerIdeal, TranspilationConfig
from qudits_on_qubits.vertical_slice.models import (
    ArtifactRef,
    BackendSnapshot,
    CircuitSpec,
    EncodingValidationError,
    ExecutionMode,
    ExecutionSpec,
    IsometricQuditEncoding,
    JsonValue,
    LogicalOutcome,
    ManifestValidationError,
    PostprocessorSpec,
    PreparedExperiment,
    QuditEncoding,
    QuditExperimentResult,
    QuditExperimentSpec,
    RunManifest,
    SoftwareProvenance,
    SpecValidationError,
)


class _Postprocessor:
    kind = "test.postprocessor"

    def evaluate(
        self, counts_by_circuit: Sequence[Mapping[str, int]]
    ) -> Mapping[str, JsonValue]:
        return {"circuits": len(counts_by_circuit)}

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        return {"kind": self.kind, "schema_version": "test-postprocessor-v1"}


class _Circuit:
    kind = "test.circuit"
    circuit_id = "identity"
    logical_dimensions = (3, 3)

    def prepare(self, encoding: QuditEncoding) -> PreparedExperiment:
        return PreparedExperiment((object(),), (object(),), _Postprocessor(), {})

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        return {
            "kind": self.kind,
            "schema_version": "test-circuit-v1",
            "circuit_id": self.circuit_id,
            "logical_dimensions": list(self.logical_dimensions),
        }

    def stable_hash(self) -> str:
        return "a" * 64


def _encoding(dimension: int = 3) -> IsometricQuditEncoding:
    physical_qubits = 2
    matrix = np.eye(2**physical_qubits, dimension, dtype=complex)
    table = tuple(
        LogicalOutcome(value=index, leaked=False)
        if index < dimension
        else LogicalOutcome(value=None, leaked=True)
        for index in range(2**physical_qubits)
    )
    return IsometricQuditEncoding(
        "canonical-ez" if dimension == 3 else "identity-d4",
        dimension,
        physical_qubits,
        matrix,
        table,
    )


def _software() -> SoftwareProvenance:
    return SoftwareProvenance(
        git_commit="abc123",
        package_version="1.2.3",
        python_version="3.13.5",
        dependencies={"numpy": "2.3.1"},
        dirty_worktree=False,
    )


def _manifest() -> RunManifest:
    encoding = _encoding()
    spec = QuditExperimentSpec(
        circuit=_Circuit(),
        encoding=encoding,
        backend=AerIdeal(),
        execution=ExecutionSpec(shots=100),
    )
    return RunManifest.initial(
        run_id="run-123",
        experiment_spec=spec.to_manifest_dict(),
        experiment_hash=spec.stable_hash(),
        encoding=encoding.to_manifest_dict(),
        encoding_hash=encoding.stable_hash(),
        software=_software(),
        timestamp="2026-08-19T10:00:00Z",
    )


def test_isometric_encoding_decodes_code_space_and_leakage() -> None:
    encoding = _encoding()

    assert isinstance(encoding, QuditEncoding)
    assert encoding.kind == "isometric"
    assert encoding.decode((0, 0)) == LogicalOutcome(0, False)
    assert encoding.decode((0, 1)) == LogicalOutcome(1, False)
    assert encoding.decode((1, 0)) == LogicalOutcome(2, False)
    assert encoding.decode((1, 1)) == LogicalOutcome(None, True)


def test_isometric_encoding_supports_dimension_four() -> None:
    encoding = _encoding(4)

    assert encoding.logical_dimension == 4
    assert encoding.decode((1, 1)) == LogicalOutcome(3, False)


def test_encoding_rejects_bad_isometry_and_wrong_bit_width() -> None:
    bad = np.ones((4, 3), dtype=complex)
    table = tuple(LogicalOutcome(index, False) for index in range(3)) + (
        LogicalOutcome(None, True),
    )

    with pytest.raises(EncodingValidationError, match="isometry"):
        IsometricQuditEncoding("bad", 3, 2, bad, table)

    with pytest.raises(EncodingValidationError, match="exactly 2"):
        _encoding().decode((0,))

    with pytest.raises(EncodingValidationError, match="bits"):
        _encoding().decode((0, 2))


def test_encoding_owns_read_only_matrix_and_returns_defensive_copy() -> None:
    source = np.eye(4, 3)
    encoding = IsometricQuditEncoding(
        "immutable",
        3,
        2,
        source,
        (
            LogicalOutcome(0, False),
            LogicalOutcome(1, False),
            LogicalOutcome(2, False),
            LogicalOutcome(None, True),
        ),
    )
    source[0, 0] = 0

    assert encoding.matrix[0, 0] == 1
    assert not encoding.matrix.flags.writeable
    with pytest.raises(ValueError):
        encoding.matrix[0, 0] = 0
    copy = encoding.isometry()
    copy[0, 0] = 0
    assert encoding.matrix[0, 0] == 1


def test_encoding_manifest_round_trip_has_stable_hash() -> None:
    encoding = _encoding()
    payload = encoding.to_manifest_dict()
    rebuilt = IsometricQuditEncoding.from_manifest_dict(payload)

    assert payload["schema_version"] == "isometric-encoding-v1"
    assert rebuilt == encoding
    assert rebuilt.stable_hash() == encoding.stable_hash()
    assert len(encoding.stable_hash()) == 64


def test_logical_outcome_requires_value_exactly_when_not_leaked() -> None:
    with pytest.raises(EncodingValidationError):
        LogicalOutcome(None, False)
    with pytest.raises(EncodingValidationError):
        LogicalOutcome(0, True)


def test_protocols_and_prepared_experiment_contract() -> None:
    circuit = _Circuit()
    prepared = circuit.prepare(_encoding())

    assert isinstance(circuit, CircuitSpec)
    assert isinstance(prepared.postprocessor, PostprocessorSpec)
    assert prepared.executable_circuits
    assert isinstance(prepared.provenance, MappingProxyType)
    with pytest.raises(SpecValidationError, match="executable"):
        PreparedExperiment((), (), _Postprocessor(), {})


def test_experiment_spec_rejects_dimension_and_seed_mismatch_before_serialization() -> None:
    class WrongDimension(_Circuit):
        logical_dimensions = (4,)

    with pytest.raises(SpecValidationError, match="logical dimensions"):
        QuditExperimentSpec(
            circuit=WrongDimension(),
            encoding=_encoding(),
            backend=AerIdeal(),
            execution=ExecutionSpec(shots=100),
        )

    with pytest.raises(SpecValidationError, match="seed"):
        QuditExperimentSpec(
            circuit=_Circuit(),
            encoding=_encoding(),
            backend=AerIdeal(seed_simulator=11),
            execution=ExecutionSpec(shots=100, seed=12),
        )


def test_experiment_spec_uses_vertical_slice_output_default() -> None:
    spec = QuditExperimentSpec(
        circuit=_Circuit(),
        encoding=_encoding(),
        backend=AerIdeal(),
        execution=ExecutionSpec(shots=100),
    )

    assert spec.output_root == Path('artifacts/vertical_slice_runs')


def test_experiment_hash_excludes_local_output_and_tags() -> None:
    common = dict(
        circuit=_Circuit(),
        encoding=_encoding(),
        backend=AerIdeal(),
        execution=ExecutionSpec(
            shots=100,
            transpilation=TranspilationConfig(optimization_level=2),
        ),
    )
    first = QuditExperimentSpec(
        **common, output_root=Path("first"), tags={"owner": "one"}
    )
    second = QuditExperimentSpec(
        **common, output_root=Path("second"), tags={"owner": "two"}
    )

    assert first.stable_hash() == second.stable_hash()
    assert first.to_manifest_dict()["tags"] == {"owner": "one"}


def test_execution_spec_rejects_non_json_and_nonfinite_values() -> None:
    with pytest.raises(SpecValidationError, match="JSON"):
        ExecutionSpec(shots=10, mitigation={"callback": object()})
    with pytest.raises(SpecValidationError, match="finite"):
        ExecutionSpec(shots=10, uncertainty={"score": float("nan")})


def test_execution_spec_validates_persisted_transpilation_fields() -> None:
    unsafe = TranspilationConfig(layout_method='api_token=secret')
    with pytest.raises(SpecValidationError, match='unsafe'):
        ExecutionSpec(shots=10, transpilation=unsafe)

    nonfinite = TranspilationConfig(seed_transpiler=float('nan'))
    with pytest.raises(SpecValidationError, match='finite'):
        ExecutionSpec(shots=10, transpilation=nonfinite)


@pytest.mark.parametrize(
    'path',
    (
        '',
        '.',
        '..',
        '/tmp/counts.json',
        r'\tmp\counts.json',
        r'C:\tmp\counts.json',
        r'C:tmp\counts.json',
        'nested/../counts.json',
        'nested//counts.json',
        'nested/./counts.json',
    ),
)
def test_artifact_ref_rejects_unsafe_relative_paths(path: str) -> None:
    with pytest.raises(ManifestValidationError, match='path'):
        ArtifactRef('counts', path, 'b' * 64, 'application/json')


def test_artifact_ref_normalizes_windows_separator_to_portable_path() -> None:
    artifact = ArtifactRef(
        'counts', r'nested\counts.json', 'b' * 64, 'application/json'
    )

    assert artifact.path == 'nested/counts.json'
    assert artifact.to_safe_dict()['path'] == 'nested/counts.json'


def test_manifest_safe_dict_round_trip_preserves_nested_contracts() -> None:
    manifest = _manifest().transition(
        "validated", timestamp="2026-08-19T10:01:00Z"
    )
    payload = manifest.to_safe_dict()
    rebuilt = RunManifest.from_safe_dict(payload)

    assert rebuilt == manifest
    assert payload["schema_version"] == "run-manifest-v1"
    assert rebuilt.software == _software()
    assert isinstance(rebuilt.timestamps, MappingProxyType)


def test_manifest_rejects_unknown_schema_and_bad_digest() -> None:
    payload = _manifest().to_safe_dict()
    payload["schema_version"] = "run-manifest-v2"
    with pytest.raises(ManifestValidationError, match="schema"):
        RunManifest.from_safe_dict(payload)

    payload = _manifest().to_safe_dict()
    payload["experiment_hash"] = "A" * 64
    with pytest.raises(ManifestValidationError, match="SHA-256"):
        RunManifest.from_safe_dict(payload)


def test_manifest_rejects_tampered_experiment_and_encoding_snapshots() -> None:
    experiment_payload = _manifest().to_safe_dict()
    experiment_payload['experiment_spec']['execution']['shots'] = 101
    with pytest.raises(ManifestValidationError, match='experiment_hash'):
        RunManifest.from_safe_dict(experiment_payload)

    encoding_payload = _manifest().to_safe_dict()
    encoding_payload['encoding']['encoding_id'] = 'tampered'
    with pytest.raises(ManifestValidationError, match='encoding_hash'):
        RunManifest.from_safe_dict(encoding_payload)


@pytest.mark.parametrize('warnings', ('warning', b'warning'))
def test_manifest_rejects_scalar_warning_containers(warnings: object) -> None:
    with pytest.raises(ManifestValidationError, match='warnings'):
        replace(_manifest(), warnings=warnings)

    payload = _manifest().to_safe_dict()
    payload['warnings'] = warnings
    with pytest.raises(ManifestValidationError, match='warnings'):
        RunManifest.from_safe_dict(payload)


def test_manifest_enforces_terminal_result_and_failure_payloads() -> None:
    manifest = _manifest()
    for index, stage in enumerate(
        ('validated', 'compiled', 'running', 'postprocessing'), start=1
    ):
        manifest = manifest.transition(
            stage, timestamp=f'2026-08-19T10:0{index}:00Z'
        )

    with pytest.raises(ManifestValidationError, match='completed'):
        manifest.transition('completed', timestamp='2026-08-19T10:05:00Z')
    completed = manifest.transition(
        'completed',
        timestamp='2026-08-19T10:05:00Z',
        result={'value': 2},
    )
    assert completed.result == {'value': 2}

    with pytest.raises(ManifestValidationError, match='nonterminal'):
        _manifest().transition(
            'validated',
            timestamp='2026-08-19T10:01:00Z',
            result={'value': 2},
        )
    with pytest.raises(ManifestValidationError, match='failed'):
        _manifest().transition('failed', timestamp='2026-08-19T10:01:00Z')
    failed = _manifest().transition(
        'failed',
        timestamp='2026-08-19T10:01:00Z',
        failure={'type': 'PreparationError'},
    )
    assert failed.failure == {'type': 'PreparationError'}


def test_manifest_allows_only_declared_status_transitions() -> None:
    manifest = _manifest()
    stages = ("validated", "compiled", "running", "postprocessing", "completed")
    for index, stage in enumerate(stages, start=1):
        result = {"value": 2} if stage == "completed" else None
        manifest = manifest.transition(
            stage,
            timestamp=f"2026-08-19T10:0{index}:00Z",
            result=result,
        )

    assert manifest.status == "completed"
    assert tuple(item["status"] for item in manifest.status_history) == (
        "created",
        *stages,
    )
    with pytest.raises(ManifestValidationError, match="transition"):
        manifest.transition("failed", timestamp="2026-08-19T11:00:00Z")

    with pytest.raises(ManifestValidationError, match="transition"):
        _manifest().transition("running", timestamp="2026-08-19T10:01:00Z")


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('created', '2026-08-19T09:59:00Z'),
        ('updated', '2026-08-19T10:02:00Z'),
        ('validated', '2026-08-19T10:02:00Z'),
    ),
)
def test_manifest_rejects_timestamps_inconsistent_with_history(
    field: str, value: str
) -> None:
    payload = _manifest().transition(
        'validated', timestamp='2026-08-19T10:01:00Z'
    ).to_safe_dict()
    payload['timestamps'][field] = value

    with pytest.raises(ManifestValidationError, match='timestamp'):
        RunManifest.from_safe_dict(payload)


def test_transition_copies_validated_payloads_and_result_is_immutable() -> None:
    manifest = _manifest().transition(
        "validated", timestamp="2026-08-19T10:01:00Z"
    ).transition("compiled", timestamp="2026-08-19T10:02:00Z")
    jobs = {"main": {"job_id": "job-1"}}
    backend = BackendSnapshot(
        provider="qiskit-aer",
        backend_name="aer_simulator",
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        identity={"version": "1"},
        capabilities={"shots": True},
    )
    manifest = manifest.transition(
        "running",
        timestamp="2026-08-19T10:03:00Z",
        backend=backend,
        jobs=jobs,
        artifacts=(ArtifactRef("counts", "counts.json", "b" * 64, "application/json"),),
        warnings=("provider delay",),
    )
    jobs["main"]["job_id"] = "tampered"

    result = QuditExperimentResult(Path("run"), manifest, {"value": 2})
    assert manifest.jobs["main"]["job_id"] == "job-1"
    assert result.result["value"] == 2
    with pytest.raises(FrozenInstanceError):
        result.artifact_dir = Path("other")


def test_vertical_slice_package_reexports_public_models() -> None:
    from qudits_on_qubits import vertical_slice

    assert vertical_slice.ArtifactRef is ArtifactRef
    assert vertical_slice.RunManifest is RunManifest
    assert vertical_slice.QuditExperimentSpec is QuditExperimentSpec


def test_safe_contracts_reject_secret_shaped_fields() -> None:
    with pytest.raises(SpecValidationError, match="unsafe"):
        SoftwareProvenance(
            git_commit=None,
            package_version="1",
            python_version="3",
            dependencies={"api_token": "secret"},
            dirty_worktree=None,
        )
