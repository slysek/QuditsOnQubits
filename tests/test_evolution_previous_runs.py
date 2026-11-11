"""Check reuse of valid results when search settings change."""

import json

import numpy as np
import pytest

from qudits_on_qubits.evolution import discovery, oe_run
from qudits_on_qubits.evolution.artifacts import file_hash
from qudits_on_qubits.evolution.checkpoints import latest_checkpoint
from qudits_on_qubits.evolution.oe_agent import builder_source, with_builder
from qudits_on_qubits.evolution.oe_candidates import candidate_angles, skeleton_program
from qudits_on_qubits.evolution.records import Record, RecordStore, as_baseline, measure, seed_records
from qudits_on_qubits.evolution.skeleton_fit import known_angles, skeleton_of
from test_evolution_records import FIXED, _campaign, _source

BUILDER = "def build():\n    return 'previous construction'\n"


def _saved_campaign(directory, circuit, encodings, *, complete=True, **settings):
    if complete:
        _campaign(directory, circuit, encodings)
    else:
        directory.mkdir(parents=True)
    pairs, angles = known_angles(circuit) or (skeleton_of(circuit), None)
    code = with_builder(skeleton_program(6, pairs, angles), BUILDER)
    config = {"state": "ghz3", "method": "canonical_direct", "representation": "skeleton",
              "encoding_mode": "fixed", "objective": "preparation", "architecture": None,
              "iterations": 20, "workers": 4, "restarts": 16, **settings}
    (directory / "configuration.json").write_text(json.dumps(config))
    if complete:
        (directory / "best_program.py").write_text(code)
        files = {name: file_hash((directory / name).read_bytes())
                 for name in ("best.qpy", "best_encodings.npy", "best_program.py")}
        (directory / "summary.json").write_text(json.dumps({"files": files}))
    else:
        np.save(directory / "encoding.npy", encodings, allow_pickle=False)
        checkpoint = directory / "openevolve/checkpoints/checkpoint_2"
        (checkpoint / "programs").mkdir(parents=True)
        (checkpoint / "programs/best.json").write_text(json.dumps({
            "id": "best", "code": code, "metrics": {"is_valid": True}}))
        (checkpoint / "metadata.json").write_text(json.dumps({
            "last_iteration": 2, "islands": [["best"]], "best_program_id": "best"}))
    return code


def test_new_settings_use_the_best_result_from_a_paused_run(tmp_path, monkeypatch):
    circuit, encodings = _source(tmp_path)
    previous = tmp_path / "runs/old/round_1_family"
    previous.parent.mkdir(parents=True)
    code = _saved_campaign(previous, circuit, encodings)
    (previous.parent / "resume.json").write_text(json.dumps({"status": "paused"}))
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    selected = []

    def campaign(baseline, output, **options):
        selected.append(baseline)
        assert options["restarts"] == 1 and options["workers"] == 1
        seed = options["initial_program"].read_text()
        assert builder_source(seed) == BUILDER
        np.testing.assert_allclose(candidate_angles(options["initial_program"]), candidate_angles(previous / "best_program.py"))
        source, encoding, _, metrics = oe_run._source(baseline, "ghz3", "canonical_direct")
        _campaign(output, source, encodings)
        (output / "best_program.py").write_text(seed)
        return {"saved": metrics}

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    report = discovery.run_discovery(tmp_path / "baseline", tmp_path / "runs/new", method="canonical_direct",
                                     record_store=store, families=1, restarts=1, workers=1, polish_seconds=0)
    assert report["status"] == "completed"
    assert report["seed_record"] == str(previous)
    assert selected == [tmp_path / "runs/new/inputs/record"]
    assert (selected[0] / "initial_program.py").read_text() == code


def test_selection_uses_quality_and_rejects_invalid_or_incompatible_results(tmp_path):
    circuit, encodings = _source(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    best = root / "old_best"
    _saved_campaign(best, circuit, encodings)
    worse = circuit.copy()
    worse.cz(0, 1)
    worse.cz(0, 1)
    _saved_campaign(root / "newer_worse", worse, encodings)
    invalid = circuit.copy()
    invalid.x(0)
    _saved_campaign(root / "invalid", invalid, encodings)
    _saved_campaign(root / "other_encoding", circuit, encodings, encoding_mode="free")
    _saved_campaign(root / "other_processor", circuit, encodings, architecture="iqm_garnet")
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    seed, others = seed_records(store, "ghz3", None, {**FIXED, "method": "canonical_direct"}, run_roots=(root,))
    assert seed.directory == best and others == []
    loaded = as_baseline(seed, tmp_path / "selected", "ghz3", "canonical_direct")
    assert builder_source((loaded / "initial_program.py").read_text()) == BUILDER


def test_partial_population_supplies_a_valid_seed_without_more_search(tmp_path):
    circuit, encodings = _source(tmp_path)
    previous = tmp_path / "runs/paused/round_1_family"
    _saved_campaign(previous, circuit, encodings, complete=False)
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    seed, _ = seed_records(store, "ghz3", None, {**FIXED, "method": "canonical_direct"},
                           run_roots=(tmp_path / "runs",))
    assert seed.directory == previous and seed.metrics["is_valid"]
    assert seed.metrics["nfev"] == 0
    assert builder_source(seed.program) == BUILDER
    assert skeleton_of(seed.circuit) == skeleton_of(circuit)


def test_checkpoint_inspection_does_not_modify_the_previous_population(tmp_path):
    for index in (4, 1):
        directory = tmp_path / f"openevolve/checkpoints/checkpoint_{index}"
        (directory / "programs").mkdir(parents=True)
        (directory / "programs/best.json").write_text(json.dumps({"id": "best"}))
        (directory / "metadata.json").write_text(json.dumps({
            "last_iteration": index, "islands": [["best"]], "best_program_id": "best"}))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    latest_checkpoint(tmp_path, update_metadata=False)
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_processor_comparison_routes_both_the_record_and_previous_result(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import continuation
    circuit, encodings = _source(tmp_path)
    previous = tmp_path / "runs/previous"
    previous.parent.mkdir()
    _saved_campaign(previous, circuit, encodings, architecture="iqm_garnet")
    worse = circuit.copy()
    worse.cz(0, 1)
    worse.cz(0, 1)
    current = Record(tmp_path / "record", worse, encodings, None,
                     measure("ghz3", worse, encodings, None), FIXED)

    def routed(state, circuit, encodings, decoder, processor, objective):
        metrics = measure(state, circuit, encodings, decoder)
        return {**metrics, "routed_two_qubit_count": 2 * metrics["cz_count"],
                "routed_two_qubit_depth": 2 * metrics["two_qubit_depth"],
                "routed_depth": 2 * metrics["depth"], "routed_success_estimate": 0.5}

    monkeypatch.setattr(continuation, "measure", routed)
    seed = continuation.best_previous((previous.parent,), "ghz3", "iqm_garnet",
                                       {**FIXED, "method": "canonical_direct"}, current)
    assert seed.directory == previous


@pytest.mark.parametrize("flag", ["from_baseline", "blank_start"])
def test_explicit_fresh_start_skips_previous_campaigns(tmp_path, monkeypatch, flag):
    circuit, encodings = _source(tmp_path)
    previous = tmp_path / "runs/previous"
    previous.parent.mkdir()
    code = _saved_campaign(previous, circuit, encodings)
    (tmp_path / "baseline/initial_program.py").write_text(code)
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")

    def campaign(baseline, output, **options):
        assert baseline == tmp_path / "baseline"
        if flag == "blank_start":
            assert builder_source(options["initial_program"].read_text()) is None
        _campaign(output, circuit, encodings)
        (output / "best_program.py").write_text(code)
        return {"saved": measure("ghz3", circuit, encodings, None)}

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    report = discovery.run_discovery(tmp_path / "baseline", tmp_path / "runs/new", method="canonical_direct",
                                     record_store=store, families=1, **{flag: True})
    assert report["status"] == "completed" and report["seed_record"] is None
