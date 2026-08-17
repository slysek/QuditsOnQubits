from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)

import qudits_on_qubits.experiments.store as store_module
from qudits_on_qubits.experiments.errors import ExperimentPersistenceError
from qudits_on_qubits.experiments.store import ExperimentStore


class SampleStatus(Enum):
    READY = "ready"


@dataclass(frozen=True)
class SampleRecord:
    name: str
    location: Path


class DictModel:
    def to_dict(self):
        return {"model": "safe", "value": 7}


def test_json_is_canonical_deterministic_and_utf8(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run("canonical")

    first = store.write_json(run, "first.json", {"z": 1, "accent": "zażółć", "a": [3, 2]})
    second = store.write_json(run, "second.json", {"a": [3, 2], "accent": "zażółć", "z": 1})

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert b"\\u017c" not in first.read_bytes()
    assert first.read_text(encoding="utf-8") == '{"a":[3,2],"accent":"zażółć","z":1}\n'


def test_json_round_trips_supported_tagged_values_and_model_values(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    ordered = OrderedDict([(("A", 1), Path("alpha")), (3, complex(-2.5, 4.0))])
    array = np.array([[1.5 + 2j, -3j], [0j, 4.25 - 1j]], dtype=np.complex128)
    value = {
        "path": Path("relative/żółć"),
        "enum": SampleStatus.READY,
        "complex": 1.25 - 0.5j,
        "scalar": np.int16(12),
        "array": array,
        "mapping": ordered,
        "tuple": ("setting", 2),
        "dataclass": SampleRecord("sample", Path("record")),
        "model": DictModel(),
    }

    store.write_json(run, "values.json", value)
    restored = store.read_json(run, "values.json")

    assert restored["path"] == value["path"]
    assert restored["enum"] is SampleStatus.READY
    assert restored["complex"] == value["complex"]
    assert isinstance(restored["scalar"], np.int16)
    assert restored["scalar"] == value["scalar"]
    np.testing.assert_array_equal(restored["array"], array)
    assert restored["array"].dtype == array.dtype
    assert list(restored["mapping"].items()) == list(ordered.items())
    assert restored["tuple"] == ("setting", 2)
    assert restored["dataclass"] == {"location": Path("record"), "name": "sample"}
    assert restored["model"] == {"model": "safe", "value": 7}

    encoded = json.loads((run / "values.json").read_text(encoding="utf-8"))
    assert encoded["complex"]["__qoq_type__"] == "complex"
    assert encoded["mapping"]["__qoq_type__"] == "mapping"


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), complex(float("nan"), 1), np.float32(np.inf), np.array([1.0, np.nan])],
)
def test_json_rejects_nonfinite_values(tmp_path, value):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()

    with pytest.raises(ExperimentPersistenceError, match="finite"):
        store.write_json(run, "invalid.json", {"value": value})


def test_json_rejects_unsupported_objects(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()

    with pytest.raises(ExperimentPersistenceError, match="unsupported"):
        store.write_json(run, "invalid.json", {"value": object()})


def test_json_read_wraps_missing_invalid_and_nonfinite_documents(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()

    with pytest.raises(ExperimentPersistenceError, match="read"):
        store.read_json(run, "missing.json")
    (run / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ExperimentPersistenceError, match="decode"):
        store.read_json(run, "broken.json")
    (run / "nan.json").write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ExperimentPersistenceError, match="finite"):
        store.read_json(run, "nan.json")


def test_create_run_uses_utc_date_unique_component_and_never_overwrites(tmp_path, monkeypatch):
    fixed = datetime(2026, 8, 17, 12, 34, 56, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(store_module, "_utc_now", lambda: fixed)
    monkeypatch.setattr(store_module.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    store = ExperimentStore(tmp_path / "runs")

    first = store.create_run("trial")
    marker = first / "marker"
    marker.write_text("keep", encoding="utf-8")

    assert first.parent.name == "2026-08-17"
    assert first.name == "20260817T123456.123456Z-trial-aaaaaaaaaaaa"
    with pytest.raises(ExperimentPersistenceError, match="already exists"):
        store.create_run("trial")
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "a\\b", ".", "..", "C:\\escape", "/absolute", ""])
def test_create_run_rejects_unsafe_run_ids(tmp_path, run_id):
    store = ExperimentStore(tmp_path / "runs")
    with pytest.raises(ExperimentPersistenceError, match="run_id"):
        store.create_run(run_id)


def test_store_rejects_runs_and_filenames_outside_root(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ExperimentPersistenceError, match="root"):
        store.write_json(outside, "leak.json", {})
    with pytest.raises(ExperimentPersistenceError, match="relative"):
        store.write_json(run, "../leak.json", {})
    with pytest.raises(ExperimentPersistenceError, match="relative"):
        store.write_json(run, tmp_path / "absolute.json", {})


def test_store_rejects_symlink_escape(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = run / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ExperimentPersistenceError, match="run directory"):
        store.write_json(run, "link/leak.json", {})
    assert not (outside / "leak.json").exists()


def test_atomic_replacement_preserves_old_file_and_cleans_temp_on_failure(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    path = store.write_json(run, "state.json", {"version": 1})
    before = path.read_bytes()

    monkeypatch.setattr(store_module.os, "replace", lambda source, destination: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(ExperimentPersistenceError, match="state.json"):
        store.write_json(run, "state.json", {"version": 2})

    assert path.read_bytes() == before
    assert list(run.glob(".state.json.*.tmp")) == []


def test_experiment_document_preserves_all_caller_fields(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    document = {
        "spec": {"state": "ghz3", "shots": 100},
        "metadata": {"device": "garnet"},
        "status": SampleStatus.READY,
        "timestamps": {"created": "2026-08-17T12:00:00Z"},
        "history": [{"status": "created"}, {"status": "ready"}],
        "job_ids": ["job-1", "job-2"],
        "attempts": 2,
    }

    path = store.write_experiment(run, document)

    assert path == run / "experiment.json"
    assert store.read_experiment(run) == document


def test_circuit_qpy_round_trip_and_hash_exact_file_bytes(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    bell = QuantumCircuit(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    measured = bell.measure_all(inplace=False)

    digest = store.write_circuits(run, (bell, measured))
    persisted = (run / "circuits.qpy").read_bytes()
    restored = store.read_circuits(run)

    assert digest == hashlib.sha256(persisted).hexdigest()
    assert isinstance(restored, tuple)
    assert restored == (bell, measured)


def test_circuit_store_rejects_empty_and_corrupt_batches(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()

    with pytest.raises(ExperimentPersistenceError, match="empty"):
        store.write_circuits(run, ())
    (run / "circuits.qpy").write_bytes(b"not a qpy file")
    with pytest.raises(ExperimentPersistenceError, match="QPY"):
        store.read_circuits(run)


def test_counts_round_trip_raw_and_quasi_values_preserving_setting_order(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    raw = OrderedDict([
        (("A0", "B0"), {"00": 10, "11": 6}),
        (("A0", "B1"), {"01": 4, "10": 12}),
    ])
    quasi = OrderedDict([
        (("A1", "B0"), {"00": 0.75, "01": -0.125, "11": 0.375}),
        (("A1", "B1"), {"00": -1.5, "11": 2.5}),
    ])

    raw_path = store.write_counts(run, 1, raw)
    quasi_path = store.write_counts(run, 3, quasi)

    assert raw_path.name == "counts-factor-1.json"
    assert quasi_path.name == "counts-factor-3.json"
    assert store.read_counts(run, 1) == raw
    restored_quasi = store.read_counts(run, 3)
    assert list(restored_quasi) == list(quasi)
    assert restored_quasi == quasi
    on_disk = json.loads(raw_path.read_text(encoding="utf-8"))
    assert [entry["setting"] for entry in on_disk["settings"]] == [
        {"__qoq_type__": "tuple", "items": ["A0", "B0"]},
        {"__qoq_type__": "tuple", "items": ["A0", "B1"]},
    ]


@pytest.mark.parametrize(
    "factor, counts",
    [
        (0, {("A",): {"0": 1}}),
        (1, {("A",): {"": 1}}),
        (1, {("A",): {0: 1}}),
        (1, {("A",): {"0": -1}}),
        (1, {("A",): {"0": float("nan")}}),
        (1, {("A",): {"0": float("inf")}}),
        (1, {("A",): {"0": True}}),
        (1, {("A",): {"0": object()}}),
        (1, {object(): {"0": 1}}),
    ],
)
def test_counts_reject_invalid_factors_keys_and_values(tmp_path, factor, counts):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    with pytest.raises(ExperimentPersistenceError):
        store.write_counts(run, factor, counts)


def test_counts_reader_rejects_corrupt_schema_and_values(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    run = store.create_run()
    store.write_json(
        run,
        "counts-factor-1.json",
        {"factor": 1, "settings": [{"setting": ("A",), "counts": {"0": -2}}]},
    )

    with pytest.raises(ExperimentPersistenceError, match="count"):
        store.read_counts(run, 1)
