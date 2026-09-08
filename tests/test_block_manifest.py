from concurrent.futures import ThreadPoolExecutor

import pytest

from qudits_on_qubits.experiments import ExperimentStore, ExperimentPersistenceError


def test_submission_marker_is_exclusive_and_never_overwritten(tmp_path):
    store = ExperimentStore(tmp_path)
    run = store.create_run()
    value = {"attempt_id": "one"}
    assert store.write_exclusive_json(run, "batches/0000/attempt.json", value)
    assert not store.write_exclusive_json(run, "batches/0000/attempt.json", {"attempt_id": "two"})
    assert store.read_plain_json(run, "batches/0000/attempt.json") == value
    assert store.artifact_exists(run, "batches/0000/attempt.json")
    assert len(store.artifact_sha256(run, "batches/0000/attempt.json")) == 64


def test_new_store_helpers_reject_escape(tmp_path):
    store = ExperimentStore(tmp_path)
    run = store.create_run()
    for method, args in ((store.artifact_exists, ()), (store.artifact_sha256, ()),
                         (store.write_exclusive_json, ({},))):
        with pytest.raises(ExperimentPersistenceError):
            method(run, "../escape.json", *args)


def test_concurrent_submitters_have_exactly_one_durable_marker(tmp_path):
    store = ExperimentStore(tmp_path)
    run = store.create_run()
    store.write_plain_json(run, "batch/request.json", {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda i: store.write_exclusive_json(run, "batch/attempt.json", {"attempt": i}), range(8)))
    assert sum(outcomes) == 1
    assert store.read_plain_json(run, "batch/attempt.json")["attempt"] == outcomes.index(True)


@pytest.mark.parametrize("change", ["outcomes", "catalogue", "width", "duplicate_block", "shots_float", "missing_hash"])
def test_hashes_do_not_replace_semantic_manifest_validation(tmp_path, change):
    from test_randomized_bell_runner import make_spec, all_settings_draw, RecordingAdapter
    from qudits_on_qubits.experiments.block_runner import run_randomized_experiment
    from qudits_on_qubits.experiments.block_manifest import verify_block_artifacts, value_hash
    result = run_randomized_experiment(make_spec(tmp_path), adapter=RecordingAdapter(), _randbelow=all_settings_draw())
    store, run = ExperimentStore.open_existing_run(result.artifact_dir)
    doc = store.read_plain_json(run, "experiment.json")
    if change in {"outcomes", "catalogue", "width"}:
        path = "catalog.json"
        artifact = store.read_plain_json(run, path)
        if change == "outcomes": artifact["outcome_map"]["3"] = 2
        elif change == "catalogue": artifact["catalog_index_by_block_id"][0] = 1
        else: artifact["classical_widths"][0] += 1
    elif change in {"duplicate_block", "shots_float"}:
        path = "batches/0000/request.json"
        artifact = store.read_plain_json(run, path)
        if change == "duplicate_block": artifact["positions"][1]["block_id"] = 0
        else: artifact["shots"] = 8.0
        doc["batches"][0]["request_hash"] = value_hash(artifact)
    else:
        doc["artifacts"].pop("compiled-catalog.qpy")
        with pytest.raises(ExperimentPersistenceError): verify_block_artifacts(store, run, doc)
        return
    store.write_plain_json(run, path, artifact)
    doc["artifacts"][path] = store.artifact_sha256(run, path)
    with pytest.raises(ExperimentPersistenceError): verify_block_artifacts(store, run, doc)
