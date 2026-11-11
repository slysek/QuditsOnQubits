"""Check the records in quantum_circuits/<state>/<processor>/: start, comparison, archive and history."""

import json

import numpy as np
from qiskit import qpy

from qudits_on_qubits.evolution import discovery, oe_run
from qudits_on_qubits.evolution.oe_candidates import skeleton_program
from qudits_on_qubits.evolution.records import RecordStore, as_baseline, read, seed_records, update
from qudits_on_qubits.evolution.skeleton_fit import skeleton_of
from qudits_on_qubits.evolution.state_workloads import party_encodings
from test_evolution_oe_run import _baseline

FIXED = {"encoding_mode": "fixed", "objective": "preparation"}


def _source(tmp_path):
    _baseline(tmp_path / "baseline")
    circuit, encoding, _, _ = oe_run._source(tmp_path / "baseline", "ghz3", "canonical_direct")
    return circuit, party_encodings("ghz3", encoding)


def _campaign(path, circuit, encodings):
    path.mkdir()
    with (path / "best.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)
    np.save(path / "best_encodings.npy", encodings, allow_pickle=False)
    return path


def _provenance(run):
    return {"source": {"run": run, "family": "f"}, "settings": dict(FIXED)}


def test_a_record_changes_only_for_a_better_circuit(tmp_path):
    circuit, encodings = _source(tmp_path)
    worse = circuit.copy()
    worse.cz(0, 1)
    worse.cz(0, 1)  # the same state with two more CZ
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    first = update(store, "ghz3", None, _campaign(tmp_path / "worse", worse, encodings),
                   objective="preparation", provenance=_provenance("r1"))
    again = update(store, "ghz3", None, _campaign(tmp_path / "again", worse, encodings),
                   objective="preparation", provenance=_provenance("r2"))
    better = update(store, "ghz3", None, _campaign(tmp_path / "better", circuit, encodings),
                    objective="preparation", provenance=_provenance("r3"))

    assert (first["accepted"], again["accepted"], better["accepted"]) == (True, False, True)
    record = read(store.directory("ghz3"), "ghz3")
    assert record.metrics["cz_count"] == better["new"]["cz_count"] == first["new"]["cz_count"] - 2
    results = json.loads((store.directory("ghz3") / "results.json").read_text())
    assert results["processor"] == "logical" and results["source"]["run"] == "r3" and "native" not in results
    archived = json.loads((tmp_path / better["archived"] / "results.json").read_text())
    assert archived["source"]["run"] == "r1"
    history = (store.archive / "ghz3" / "logical" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["accepted"] for line in history] == [True, False, True]


def test_a_processor_record_holds_the_native_circuit_and_keeps_its_settings(tmp_path):
    circuit, encodings = _source(tmp_path)
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    result = update(store, "ghz3", "iqm_garnet", _campaign(tmp_path / "run", circuit, encodings),
                    objective="preparation", provenance=_provenance("r1"))
    folder = store.directory("ghz3", "iqm_garnet")
    native = json.loads((folder / "circuit_native.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    assert result["accepted"] and (folder / "circuit_native.qpy").is_file()
    assert results["native"]["two_qubit_count"] == result["new"]["routed_two_qubit_count"]
    assert native["initial_physical_qubits"] == results["native"]["initial_physical_qubits"]
    assert {gate["name"] for gate in native["gates"]} <= set(native["basis_gates"])

    free = {"source": {"run": "r2"}, "settings": {"encoding_mode": "free", "objective": "preparation"}}
    skipped = update(store, "ghz3", "iqm_garnet", _campaign(tmp_path / "free", circuit, encodings),
                     objective="preparation", provenance=free)
    assert not skipped["accepted"] and "other settings" in skipped["skipped"]
    assert json.loads((folder / "results.json").read_text())["source"]["run"] == "r1"


def test_seed_records_fall_back_to_the_logical_record_and_skip_broken_ones(tmp_path, capsys):
    circuit, encodings = _source(tmp_path)
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    update(store, "ghz3", None, _campaign(tmp_path / "run", circuit, encodings),
           objective="preparation", provenance=_provenance("r1"))
    broken = store.directory("ghz3", "ibm_heron_r2")
    broken.mkdir()
    (broken / "results.json").write_text("{}")

    seed, others = seed_records(store, "ghz3", "iqm_garnet", FIXED)
    assert seed.directory == store.directory("ghz3") and others == []
    assert "ibm_heron_r2: skipped" in capsys.readouterr().out
    assert seed_records(store, "ghz3", None, {**FIXED, "encoding_mode": "free"}) == (None, [])
    baseline = as_baseline(seed, tmp_path / "as_baseline", "ghz3", "canonical_direct")
    loaded = oe_run._source(baseline, "ghz3", "canonical_direct")[0]
    assert skeleton_of(loaded) == skeleton_of(circuit)


def test_cost_records_stay_apart_and_start_from_the_preparation_record(tmp_path):
    circuit, encodings = _source(tmp_path)
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    cost = {**FIXED, "objective": "cost"}
    update(store, "ghz3", "iqm_garnet", _campaign(tmp_path / "run", circuit, encodings),
           objective="preparation", provenance=_provenance("r1"))

    seed, others = seed_records(store, "ghz3", "iqm_garnet", cost)
    assert seed.directory == store.directory("ghz3", "iqm_garnet") and others == []
    result = update(store, "ghz3", "iqm_garnet", _campaign(tmp_path / "cost", circuit, encodings),
                    objective="cost", provenance={"source": {"run": "r2"}, "settings": cost})
    folder = store.directory("ghz3", "iqm_garnet", "cost")
    assert result["accepted"] and folder.name == "iqm_garnet_cost" and result["record"] == str(folder)
    assert json.loads((folder / "results.json").read_text())["processor"] == "iqm_garnet"
    assert (store.archive / "ghz3" / "iqm_garnet_cost" / "history.jsonl").is_file()
    assert seed_records(store, "ghz3", "iqm_garnet", cost)[0].directory == folder
    assert seed_records(store, "ghz3", "iqm_garnet", FIXED)[0].directory == store.directory("ghz3", "iqm_garnet")


def test_discovery_starts_from_the_record_and_offers_its_winner(tmp_path, monkeypatch):
    _baseline(tmp_path / "baseline")
    store = RecordStore(tmp_path / "quantum_circuits", tmp_path / "archive")
    baselines = []

    def run(baseline, output, **kwargs):
        baselines.append(baseline)
        circuit, encoding, _, metrics = oe_run._source(baseline, "ghz3", "canonical_direct")
        _campaign(output, circuit, party_encodings("ghz3", encoding))
        (output / "best_program.py").write_text(skeleton_program(6, skeleton_of(circuit)))
        return {"saved": metrics}

    monkeypatch.setattr(discovery, "run_campaign", run)
    options = dict(state="ghz3", method="canonical_direct", families=1, record_store=store)
    first = discovery.run_discovery(tmp_path / "baseline", tmp_path / "first", **options)
    second = discovery.run_discovery(tmp_path / "baseline", tmp_path / "second", **options)
    third = discovery.run_discovery(tmp_path / "baseline", tmp_path / "third", **options, from_baseline=True)

    assert first["seed_record"] is None and [r["accepted"] for r in first["records"]] == [True]
    assert second["seed_record"] == str(store.directory("ghz3"))
    assert baselines[1] == tmp_path / "second" / "inputs" / "record"
    assert [r["accepted"] for r in second["records"]] == [False]
    assert baselines[2] == tmp_path / "baseline" and third["record_warm_starts"] == [str(store.directory("ghz3"))]
