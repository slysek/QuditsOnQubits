"""Check the state experiment harness."""

import json
from pathlib import Path

import pytest

from qudits_on_qubits.evolution import harness
from qudits_on_qubits.evolution.artifacts import main_checkout


def test_main_checkout_maps_a_linked_worktree_to_the_main_checkout(tmp_path):
    main, linked, sub = tmp_path / "main", tmp_path / "linked", tmp_path / "sub"
    (main / ".git/worktrees/linked").mkdir(parents=True)
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {main}/.git/worktrees/linked\n", encoding="utf-8")
    sub.mkdir()
    (sub / ".git").write_text("gitdir: ../main/.git/modules/sub\n", encoding="utf-8")

    assert main_checkout(linked) == main
    assert main_checkout(main) == main
    assert main_checkout(sub) == sub
    assert main_checkout(tmp_path) == tmp_path


def _result(cz):
    return {"saved": {"is_valid": True, "cz_count": cz, "two_qubit_depth": 3,
                      "depth": 4, "one_qubit_gate_count": 5, "F": 1.0, "L": 0.0}}


def test_batch_runs_each_state_and_seed_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    def run(baseline, output, **kwargs):
        calls.append((baseline, output, kwargs))
        return _result(9 if kwargs["state"] == "ghz3" else 19)
    monkeypatch.setattr(harness, "run_campaign", run)
    output = tmp_path / "batch"
    code = harness.run_batch(output, {"ghz3": tmp_path / "g", "ame43": tmp_path / "a"}, seeds=(1, 2))
    assert code == 0
    assert [(v[2]["state"], v[2]["seed"]) for v in calls] == [
        ("ghz3", 1), ("ghz3", 2), ("ame43", 1), ("ame43", 2)]
    assert [v[0] for v in calls] == [tmp_path / "g", tmp_path / "g", tmp_path / "a", tmp_path / "a"]
    assert len({v[1] for v in calls}) == 4
    report = json.loads((output / "batch.json").read_text())
    assert report["status"] == "completed"
    assert len(report["runs"]) == 4
    assert [row["cz_count"] for row in report["runs"]] == [9, 9, 19, 19]


def test_batch_stops_on_error_and_records_partial_results(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    def run(*args, **kwargs):
        calls.append(kwargs["seed"])
        if kwargs["seed"] == 2:
            raise TimeoutError("secret")
        return _result(9)
    monkeypatch.setattr(harness, "run_campaign", run)
    output = tmp_path / "batch"
    assert harness.run_batch(output, {"ghz3": tmp_path / "g"}, seeds=(1, 2, 3)) == 1
    report = json.loads((output / "batch.json").read_text())
    assert calls == [1, 2]
    assert report["status"] == "failed"
    assert [row["status"] for row in report["runs"]] == ["completed", "failed"]
    assert report["runs"][-1]["error"] == "TimeoutError"
    assert "secret" not in (output / "batch.json").read_text()


def test_batch_rejects_invalid_input_and_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    for seeds in [(), (-1,), (1, 1), (True,)]:
        with pytest.raises(ValueError):
            harness.run_batch(tmp_path / "batch", {"ghz3": tmp_path}, seeds=seeds)
        assert not (tmp_path / "batch").exists()
    with pytest.raises(ValueError):
        harness.run_batch(tmp_path / "batch", {"bad": tmp_path})
    with pytest.raises(ValueError, match="cost objective needs an architecture"):
        harness.run_batch(tmp_path / "batch", {"ghz3": tmp_path}, objective="cost")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        harness.run_batch(output, {"ghz3": tmp_path})
    assert (output / "keep").read_text() == "keep"


def test_cli_starts_the_batch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "run_batch", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    output = tmp_path / "out"
    assert harness.main(["run", "--output", str(output), "--state", "ghz3", "--seed", "7"]) == 0
    assert calls[-1][1]["seeds"] == (7,)
    assert list(calls[-1][0][1]) == ["ghz3"]


def test_cli_forwards_other_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(harness.subprocess, "run", lambda command, check: calls.append(command) or type("Result", (), {"returncode": 0})())
    assert harness.main(["baseline", "--output", "my result"]) == 0
    assert calls[0][-2:] == ["--output", "my result"]
    assert calls[0][2] == "qudits_on_qubits.evolution.baseline"


def test_series_deadline_caps_each_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    clock = iter([0, 1, 6])
    monkeypatch.setattr(harness, "monotonic", lambda: next(clock))
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["campaign_timeout_seconds"])
        return _result(9)
    monkeypatch.setattr(harness, "run_campaign", run)
    output = tmp_path / "batch"
    assert harness.run_batch(output, {"ghz3": tmp_path}, seeds=(1, 2),
                             campaign_timeout_seconds=10, series_timeout_seconds=5) == 1
    assert calls == [4]
    report = json.loads((output / "batch.json").read_text())
    assert report["status"] == "failed"
    assert report["error"] == "SeriesTimeout"
    assert len(report["runs"]) == 1


def test_invalid_series_deadline_does_not_create_output(tmp_path):
    for value in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            harness.run_batch(tmp_path / "batch", {"ghz3": tmp_path}, series_timeout_seconds=value)
    assert not (tmp_path / "batch").exists()


def test_readout_cli_saves_circuits_and_metrics(tmp_path, monkeypatch):
    import numpy as np
    from qiskit import QuantumCircuit, qpy
    from qudits_on_qubits.evolution import state_readout

    circuit = tmp_path / "source.qpy"
    with circuit.open("wb") as stream:
        qpy.dump(QuantumCircuit(6), stream)
    encoding = tmp_path / "encoding.npy"
    np.save(encoding, np.eye(4, 3), allow_pickle=False)
    captured = {}
    def compile_readout(name, source, actual_encoding, edges, *, seeds):
        captured.update(name=name, edges=edges, seeds=seeds, width=source.num_qubits)
        return {"circuits": {"decoded": source, "routed_with_measurement": source},
                "metadata": {"decoded_metrics": {"is_valid": True}, "physical_backend_validated": False}}
    monkeypatch.setattr(state_readout, "compile_readout", compile_readout)
    output = tmp_path / "readout"
    assert harness.main(["readout", "--state", "ghz3", "--circuit", str(circuit),
                         "--encoding", str(encoding), "--output", str(output),
                         "--edge", "0,1", "--edge", "1,2", "--seed", "3"]) == 0
    assert captured == {"name": "ghz3", "edges": [(0, 1), (1, 2)], "seeds": (3,), "width": 6}
    assert (output / "decoded.qpy").is_file()
    assert (output / "routed_with_measurement.qpy").is_file()
    assert not json.loads((output / "summary.json").read_text())["physical_backend_validated"]
    with pytest.raises(SystemExit):
        harness.main(["readout", "--state", "ghz3", "--circuit", str(circuit),
                      "--encoding", str(encoding), "--output", str(output), "--edge", "0,1"])


def test_readout_rejects_bad_qpy_before_creating_output(tmp_path, capsys):
    source = tmp_path / "bad.qpy"
    source.write_bytes(b"bad")
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        harness.main(["readout", "--state", "ghz3", "--circuit", str(source),
                      "--encoding", str(tmp_path / "encoding.npy"), "--output", str(output), "--edge", "0,1"])
    assert "QPY file is invalid" in capsys.readouterr().err
    assert not output.exists()
