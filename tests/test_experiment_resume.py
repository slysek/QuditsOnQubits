from __future__ import annotations

import json
import os
from unittest.mock import Mock

import pytest

from qudits_on_qubits.experiments import (
    ExperimentPersistenceError,
    ExperimentResult,
    ExperimentStatus,
    ExperimentValidationError,
    resume_experiment,
)
from qudits_on_qubits.experiments.models import BootstrapConfig
from qudits_on_qubits.experiments.store import ExperimentStore
from qudits_on_qubits.experiments.uncertainty import (
    BootstrapInputs,
    bootstrap_bell_results,
)
import qudits_on_qubits.experiments.store as store_module


def _write_experiment(tmp_path, document):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    document = {"experiment_id": run.name, **document}
    store.write_experiment(run, document)
    return run


def _write_plain_experiment(tmp_path, document):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    document = {"experiment_id": run.name, **document}
    (run / "experiment.json").write_text(
        json.dumps(document, allow_nan=False), encoding="utf-8"
    )
    return run


def test_resume_new_completed_result_loads_identically(tmp_path):
    backend = {"kind": "custom", "name": "target", "version": "1"}
    values = {
        "raw": {"estimate": {"real": 1.0, "imag": 0.0}},
        "diagnostics": {"samples": 100},
    }
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": backend,
            "job_ids": ["job-1", "job-2"],
            "result": values,
        },
    )
    expected = ExperimentResult(
        experiment_id=run.name,
        status=ExperimentStatus.COMPLETED,
        artifact_dir=run,
        values=values,
        backend=backend,
        job_ids=("job-1", "job-2"),
    )

    assert resume_experiment(run) == expected


def test_resume_completed_schema_v3_preserves_explicit_bell_semantics(
    tmp_path, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    setting = ("A0",)
    structured_payload = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {setting: {"00": 8, "11": 2}}},
            terms=({"coeff": 1.0, "settings": setting, "powers": (0,)},),
            qutrit_bit_indices_by_setting={setting: ((0, 1),)},
            decoding_kwargs={
                "bit_order": "left-to-right",
                "outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
                "d": 3,
            },
        ),
        BootstrapConfig(samples=2),
    ).to_safe_dict()
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "job_ids": ["job-1"],
            "result": structured_payload,
        },
    )
    artifact = run / "experiment.json"
    original_artifact = artifact.read_bytes()
    recompute = Mock(side_effect=AssertionError("completed result was recomputed"))
    monkeypatch.setattr(runner, "bootstrap_bell_results", recompute)

    resumed = resume_experiment(run)

    assert resumed.values.keys() == structured_payload.keys()
    assert dict(resumed.values) == structured_payload
    assert resumed.values["raw"] == resumed.values["raw_conditional"]
    assert resumed.values["raw_unconditional"] == structured_payload[
        "raw_unconditional"
    ]
    assert resumed.values["raw_invalid_codeword_rate"] == structured_payload[
        "raw_invalid_codeword_rate"
    ]
    assert resumed.values["raw_invalid_codeword_shots"] == structured_payload[
        "raw_invalid_codeword_shots"
    ] == {"total_shots": 10, "accepted_shots": 8, "invalid_shots": 2}
    recompute.assert_not_called()
    assert artifact.read_bytes() == original_artifact
    assert sorted(path.name for path in run.iterdir()) == ["experiment.json"]


def test_resume_legacy_completed_result_loads_embedded_result_without_artifacts(
    tmp_path,
):
    backend = {"kind": "iqm", "name": "legacy-target", "version": "old"}
    values = {"raw": {"estimate": {"real": 2.5, "imag": -0.25}}}
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 1,
            "status": "completed",
            "backend": {"identity": backend},
            "source": {
                "encoding_artifact": "missing-encoding.json",
                "encoding_sha256": "not-checked",
            },
            "circuits": {
                "source": {
                    "artifact": "missing-source.qpy",
                    "sha256": "not-checked",
                }
            },
            "job_ids": ["legacy-job"],
            "result": values,
        },
    )

    loaded = resume_experiment(run)

    assert loaded.experiment_id == run.name
    assert loaded.status is ExperimentStatus.COMPLETED
    assert loaded.artifact_dir == run
    assert dict(loaded.values) == values
    assert dict(loaded.backend) == backend
    assert loaded.job_ids == ("legacy-job",)


def test_resume_unfinished_result_rejects_without_adapter_calls(tmp_path):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "running",
        },
    )
    adapter = Mock()

    with pytest.raises(ExperimentValidationError, match="unfinished.*not supported"):
        resume_experiment(run, adapter=adapter)

    assert adapter.mock_calls == []


def test_resume_completed_result_accepts_relative_run_path(tmp_path, monkeypatch):
    backend = {"kind": "custom", "name": "target"}
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": backend,
            "result": {"raw": {}},
        },
    )
    monkeypatch.chdir(tmp_path)

    loaded = resume_experiment(run.relative_to(tmp_path))

    assert loaded.artifact_dir == run


@pytest.mark.parametrize(
    "document",
    [
        {
            "schema_version": 3,
            "status": "completed",
            "result": {"raw": {}},
        },
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "job_ids": "job-1",
            "result": {"raw": {}},
        },
    ],
)
def test_resume_rejects_malformed_completed_result(tmp_path, document):
    run = _write_experiment(tmp_path, document)

    with pytest.raises(
        ExperimentPersistenceError, match="completed experiment JSON is invalid"
    ):
        resume_experiment(run)


def test_resume_rejects_experiment_id_mismatching_run_directory(tmp_path):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "experiment_id": "different-run",
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )

    with pytest.raises(
        ExperimentPersistenceError, match="completed experiment JSON is invalid"
    ):
        resume_experiment(run)


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_resume_accepts_explicit_supported_completed_schema_versions(
    tmp_path, schema_version
):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": schema_version,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )

    assert resume_experiment(run).status is ExperimentStatus.COMPLETED


@pytest.mark.parametrize(
    "schema_fields",
    [
        {},
        {"schema_version": True},
        {"schema_version": False},
        {"schema_version": 4},
        {"schema_version": "3"},
    ],
)
def test_resume_rejects_missing_or_unsupported_schema_version(
    tmp_path, schema_fields
):
    run = _write_experiment(
        tmp_path,
        {
            **schema_fields,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )

    with pytest.raises(
        ExperimentPersistenceError, match="unsupported experiment schema version"
    ):
        resume_experiment(run)


def test_resume_ignores_tagged_legacy_fields_without_decoding_them(
    tmp_path, monkeypatch
):
    run = _write_plain_experiment(
        tmp_path,
        {
            "schema_version": 1,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
            "ignored": [
                {
                    "__qoq_type__": "enum",
                    "module": "qudits_on_qubits.experiments.models",
                    "name": "COMPLETED",
                    "qualname": "ExperimentStatus",
                },
                {
                    "__qoq_type__": "numpy_array",
                    "data": [1],
                    "dtype": "<i8",
                    "shape": [1],
                },
            ],
        },
    )
    import_module = Mock(wraps=store_module.importlib.import_module)
    asarray = Mock(wraps=store_module.np.asarray)
    monkeypatch.setattr(store_module.importlib, "import_module", import_module)
    monkeypatch.setattr(store_module.np, "asarray", asarray)

    loaded = resume_experiment(run)

    assert loaded.status is ExperimentStatus.COMPLETED
    import_module.assert_not_called()
    asarray.assert_not_called()


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {
            "backend": {
                "kind": "custom",
                "name": "target",
                "metadata": {"access_token": "secret-value"},
            }
        },
        {"result": {"raw": {"access_token": "secret-value"}}},
        {"job_ids": ["../../unsafe"]},
        {"job_ids": ["token:secret-value"]},
    ],
)
def test_resume_rejects_unsafe_selected_fields(tmp_path, invalid_fields):
    document = {
        "schema_version": 3,
        "status": "completed",
        "backend": {"kind": "custom", "name": "target"},
        "result": {"raw": {}},
        **invalid_fields,
    }
    run = _write_experiment(tmp_path, document)

    with pytest.raises(
        ExperimentPersistenceError, match="completed experiment JSON is invalid"
    ):
        resume_experiment(run)


def test_resume_rejects_lexical_parent_traversal(tmp_path):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )
    traversing = run.parent / ".." / run.parent.name / run.name

    with pytest.raises(ExperimentPersistenceError, match="traversal"):
        resume_experiment(traversing)


def test_resume_rejects_symlinked_run_path(tmp_path):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )
    link = run.parent / "linked-run"
    try:
        link.symlink_to(run, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ExperimentPersistenceError, match="symlink or reparse"):
        resume_experiment(link)


def test_resume_rejects_non_regular_experiment_json(tmp_path):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )
    artifact = run / "experiment.json"
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ExperimentPersistenceError, match="regular file"):
        resume_experiment(run)


def test_resume_rejects_fifo_experiment_json_without_opening_it(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation unavailable")
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )
    artifact = run / "experiment.json"
    artifact.unlink()
    try:
        os.mkfifo(artifact)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"FIFO creation unavailable: {error}")

    with pytest.raises(ExperimentPersistenceError, match="regular file"):
        resume_experiment(run)


def test_resume_rejects_oversized_experiment_json(tmp_path, monkeypatch):
    run = _write_plain_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
            "ignored": "padding",
        },
    )
    size = (run / "experiment.json").stat().st_size
    monkeypatch.setattr(store_module, "_MAX_PLAIN_JSON_BYTES", size - 1, raising=False)

    with pytest.raises(ExperimentPersistenceError, match="JSON size limit"):
        resume_experiment(run)


@pytest.mark.parametrize(
    ("limit_name", "limit", "ignored"),
    [
        ("_MAX_PLAIN_JSON_DEPTH", 3, [[[[0]]]]),
        ("_MAX_PLAIN_JSON_NODES", 4, [0, 1, 2, 3]),
        ("_MAX_PLAIN_JSON_STRING_LENGTH", 3, "four"),
        ("_MAX_PLAIN_JSON_CONTAINER_ITEMS", 3, [0, 1, 2, 3]),
    ],
)
def test_resume_enforces_json_complexity_limits(
    tmp_path, monkeypatch, limit_name, limit, ignored
):
    run = _write_plain_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
            "ignored": ignored,
        },
    )
    monkeypatch.setattr(store_module, limit_name, limit, raising=False)

    with pytest.raises(ExperimentPersistenceError, match="JSON complexity limit"):
        resume_experiment(run)


@pytest.mark.parametrize("invalid_status", [None, True, 1, "unknown-status"])
def test_resume_rejects_invalid_status_as_persistence_error(tmp_path, invalid_status):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": invalid_status,
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )

    with pytest.raises(
        ExperimentPersistenceError, match="completed experiment JSON is invalid"
    ):
        resume_experiment(run)


@pytest.mark.parametrize(
    "unfinished_status", ["created", "submitted", "postprocessing", "failed"]
)
def test_resume_rejects_known_unfinished_status_as_validation_error(
    tmp_path, unfinished_status
):
    run = _write_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": unfinished_status,
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
        },
    )

    with pytest.raises(ExperimentValidationError, match="unfinished.*not supported"):
        resume_experiment(run)


def test_resume_limits_number_token_length_in_ignored_field(tmp_path, monkeypatch):
    run = _write_plain_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
            "ignored": 1234,
        },
    )
    monkeypatch.setattr(
        store_module, "_MAX_PLAIN_JSON_NUMBER_LENGTH", 3, raising=False
    )

    with pytest.raises(ExperimentPersistenceError, match="JSON complexity limit"):
        resume_experiment(run)


def test_resume_rejects_overflowing_float_in_ignored_field(tmp_path):
    run = _write_plain_experiment(
        tmp_path,
        {
            "schema_version": 3,
            "status": "completed",
            "backend": {"kind": "custom", "name": "target"},
            "result": {"raw": {}},
            "ignored": "OVERFLOW",
        },
    )
    artifact = run / "experiment.json"
    artifact.write_text(
        artifact.read_text(encoding="utf-8").replace('"OVERFLOW"', "1e10000"),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPersistenceError, match="JSON numbers must be finite"):
        resume_experiment(run)


def test_resume_rejects_nul_path_as_persistence_error():
    with pytest.raises(
        ExperimentPersistenceError,
        match="artifact_dir must identify an existing run directory",
    ):
        resume_experiment("\0")


def test_resume_rejects_unsafe_matching_experiment_id_as_persistence_error(
    tmp_path,
):
    run = tmp_path / "runs" / "2026-08-27" / "token=secret-value"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "experiment_id": run.name,
                "status": "completed",
                "backend": {"kind": "custom", "name": "target"},
                "result": {"raw": {}},
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentPersistenceError, match="completed experiment JSON is invalid"
    ):
        resume_experiment(run)


def test_resume_duplicate_key_error_does_not_echo_unsafe_key(tmp_path):
    unsafe_key = "token" + "=secret-value"
    run = tmp_path / "runs" / "2026-08-27" / "safe-run"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(
        "{"
        '"schema_version":3,'
        '"experiment_id":"safe-run",'
        '"status":"completed",'
        '"backend":{"kind":"custom","name":"target"},'
        '"result":{"raw":{}},'
        f'"ignored":{{"{unsafe_key}":1,"{unsafe_key}":2}}'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPersistenceError) as caught:
        resume_experiment(run)

    assert "duplicate key" in str(caught.value)
    assert unsafe_key not in str(caught.value)
