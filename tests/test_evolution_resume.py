"""Check saved operations after an interruption and a process restart."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from qudits_on_qubits.evolution import checkpoints, discovery, harness, oe_run
from qudits_on_qubits.evolution.checkpoints import latest_checkpoint, resume_operation, saved_run
from test_evolution_oe_run import _baseline, _socketpair_works


@pytest.fixture(autouse=True)
def _session_history(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "user_state"))
    monkeypatch.setattr(harness, "ROOT", tmp_path)


def _result():
    return {"saved": {"is_valid": True, "cz_count": 5, "two_qubit_depth": 3,
                      "depth": 7, "one_qubit_gate_count": 12, "F": 1.0, "L": 0.0}}


def test_batch_pause_and_resume_preserve_completed_campaigns(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    calls = []

    def campaign(baseline, output, **options):
        calls.append(options["seed"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _result()

    monkeypatch.setattr(harness, "run_campaign", campaign)
    output = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        harness.run_batch(output, {"ghz3": tmp_path}, seeds=(1, 2, 3))
    report = json.loads((output / "batch.json").read_text())
    assert report["status"] == "paused"
    assert [row["status"] for row in report["runs"]] == ["completed", "paused"]
    assert resume_operation(output) == 0
    assert calls == [1, 2, 2, 3]
    report = json.loads((output / "batch.json").read_text())
    assert report["status"] == "completed" and len(report["runs"]) == 3


def test_batch_resume_keeps_the_state_order(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    calls = []

    def campaign(base, output, **options):
        calls.append((options["state"], options["seed"]))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _result()

    monkeypatch.setattr(harness, "run_campaign", campaign)
    output = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        harness.run_batch(output, {"ghz3": tmp_path, "ame43": tmp_path}, seeds=(1, 2))
    assert resume_operation(output) == 0
    assert calls == [("ghz3", 1), ("ghz3", 2), ("ghz3", 2), ("ame43", 1), ("ame43", 2)]


def test_discovery_resume_keeps_ideation_inputs_and_family_results(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    calls = []
    pause = True

    def campaign(base, output, **options):
        nonlocal pause
        calls.append(output.name)
        if len(calls) == 2 and pause:
            pause = False
            raise KeyboardInterrupt
        output.mkdir()
        (output / "best_program.py").write_bytes(options["initial_program"].read_bytes())
        return _result()

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    output = tmp_path / "discovery"
    with pytest.raises(KeyboardInterrupt):
        discovery.run_discovery(baseline, output, method="canonical_direct", families=2,
                                 rounds=2, polish_seconds=0)
    report = json.loads((output / "discovery.json").read_text())
    assert report["status"] == "paused"
    idea = (output / "ideation.json").read_bytes()
    seed = (output / "inputs/round_1_1_hub/seed.py").read_bytes()
    monkeypatch.setattr(discovery, "offline_families", lambda *args: pytest.fail("Ideation must remain saved"))
    report = resume_operation(output)
    assert report["status"] == "completed" and len(report["runs"]) == 4
    assert calls == ["round_1_0_source", "round_1_1_hub", "round_1_1_hub",
                     "round_2_0_source", "round_2_1_hub"]
    assert (output / "ideation.json").read_bytes() == idea
    assert (output / "inputs/round_1_1_hub/seed.py").read_bytes() == seed


def test_campaign_resume_uses_saved_population_and_skips_diagnostics(tmp_path, monkeypatch):
    api = pytest.importorskip("openevolve.api")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "campaign"
    model = tmp_path / "model.yaml"
    model.write_text("llm:\n  models:\n    - name: test-model\n")

    def interrupted(initial, evaluator, **options):
        checkpoint = output / "openevolve/checkpoints/checkpoint_1"
        (checkpoint / "programs").mkdir(parents=True)
        (checkpoint / "programs/seed.json").write_text(json.dumps({"id": "seed", "iteration_found": 0}))
        (checkpoint / "metadata.json").write_text(json.dumps({
            "last_iteration": 1, "islands": [["seed"]], "best_program_id": "seed"}))
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "run_evolution", interrupted)
    with pytest.raises(KeyboardInterrupt):
        oe_run._run_campaign(baseline, output, method="canonical_direct", iterations=3,
                             model_config=model, timeout_seconds=20, cpu_seconds=20)
    initial = (output / "initial_program.py").read_bytes()
    monkeypatch.setattr(oe_run, "_diagnose", lambda *args: pytest.fail("Diagnostics must remain saved"))

    def continued(initial, evaluator, **options):
        assert Path(options["checkpoint_path"]).name == "checkpoint_1"
        assert options["iterations"] == 2
        return api.EvolutionResult(best_program=None, best_score=1.0, best_code=Path(initial).read_text(),
                                   metrics={"is_valid": True}, output_dir=None)

    monkeypatch.setattr(api, "run_evolution", continued)
    result = oe_run._run_campaign(baseline, output, method="canonical_direct", iterations=3,
                                  model_config=model, timeout_seconds=20, cpu_seconds=20,
                                  resume=True)
    assert result["saved"]["is_valid"]
    assert (output / "initial_program.py").read_bytes() == initial


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_openevolve_resumes_a_real_population(tmp_path, monkeypatch):
    api = pytest.importorskip("openevolve.api")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "campaign"
    evolve = api.run_evolution

    def pause_after_one(*args, **options):
        evolve(*args, **{**options, "iterations": 1})
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "run_evolution", pause_after_one)
    options = dict(method="canonical_direct", representation="skeleton", offline_model="mutations",
                   iterations=2, restarts=1, timeout_seconds=5, cpu_seconds=5)
    with pytest.raises(KeyboardInterrupt):
        oe_run._run_campaign(baseline, output, **options)
    checkpoint, completed = latest_checkpoint(output)
    assert completed == 1
    previous = {path.name for path in (checkpoint / "programs").glob("*.json")}
    monkeypatch.setattr(api, "run_evolution", evolve)
    result = oe_run._run_campaign(baseline, output, **options, resume=True)
    assert result["saved"]["is_valid"]
    checkpoint, completed = latest_checkpoint(output)
    assert completed == 2
    assert previous <= {path.name for path in (checkpoint / "programs").glob("*.json")}
    assert resume_operation(output) == result


def test_pause_during_source_loading_saves_the_operation(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    monkeypatch.setattr(oe_run, "_source", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    output = tmp_path / "campaign"
    with pytest.raises(KeyboardInterrupt):
        oe_run._run_campaign(tmp_path, output)
    assert json.loads((output / "resume.json").read_text())["status"] == "paused"


def test_resume_restores_agent_settings_without_saving_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)
    monkeypatch.setenv("QOQ_AGENT_MODEL", "gpt-6.1-terra")
    monkeypatch.setenv("QOQ_AGENT_SECONDS", "600")
    monkeypatch.setenv("QOQ_PROXY_KEY", "private-key")
    monkeypatch.setattr(harness, "run_campaign", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    output = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        harness.run_batch(output, {"ghz3": tmp_path})
    assert "private-key" not in (output / "resume.json").read_text()
    monkeypatch.setenv("QOQ_AGENT_MODEL", "different")
    monkeypatch.delenv("QOQ_AGENT_SECONDS")

    def resumed(*args, **options):
        import os
        assert os.environ["QOQ_AGENT_MODEL"] == "gpt-6.1-terra"
        assert os.environ["QOQ_AGENT_SECONDS"] == "600"
        return _result()

    monkeypatch.setattr(harness, "run_campaign", resumed)
    assert harness.main(["resume", str(output)]) == 0


@pytest.mark.parametrize("document", [{}, {"kind": "other", "options": {}}, [], {"version": 1, "kind": "run", "options": []}])
def test_resume_rejects_invalid_saved_options(tmp_path, document):
    (tmp_path / "resume.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        resume_operation(tmp_path)


def test_pause_stops_worker_and_child_processes(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "from qudits_on_qubits.evolution.checkpoints import saved_run\n"
        "output = Path(sys.argv[1])\n"
        "def stop(*args):\n"
        "    (output / 'stop_received').write_text('saved')\n"
        "    raise KeyboardInterrupt\n"
        "with saved_run(output, 'campaign', {'output': output}):\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "    if os.fork() == 0:\n"
        "        time.sleep(60)\n"
        "    else:\n"
        "        (output / 'ready').write_text('ready')\n"
        "        time.sleep(60)\n"
    )
    popen = subprocess.Popen
    processes = []
    children = []

    def start(args, **options):
        process = popen([sys.executable, str(worker), str(output)], **options)
        processes.append(process)
        return process

    def interrupt_when_ready(seconds):
        if (output / "ready").is_file():
            import psutil
            children.extend(psutil.Process(processes[0].pid).children(recursive=True))
            raise KeyboardInterrupt
        time.sleep(0.01)

    monkeypatch.setattr(oe_run.subprocess, "Popen", start)
    monkeypatch.setattr(oe_run, "sleep", interrupt_when_ready)
    with pytest.raises(KeyboardInterrupt):
        oe_run.run_campaign(tmp_path, output, campaign_timeout_seconds=20)
    assert (output / "stop_received").read_text() == "saved"
    assert json.loads((output / "resume.json").read_text())["status"] == "paused"
    assert processes[0].poll() is not None
    import psutil
    for _ in range(100):
        if all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in children):
            break
        time.sleep(0.01)
    else:
        pytest.fail("The paused campaign still has an active child process")


def test_saved_time_excludes_the_pause(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import checkpoints
    ticks = iter([10, 20, 1_000_000, 1_000_005])
    monkeypatch.setattr(checkpoints, "monotonic", lambda: next(ticks))
    output = tmp_path / "saved"
    with pytest.raises(KeyboardInterrupt), saved_run(output, "run", {"output": output}):
        raise KeyboardInterrupt
    with saved_run(output, "run", {"output": output}, resume=True) as document:
        assert document["elapsed_seconds"] == 10
    assert json.loads((output / "resume.json").read_text())["elapsed_seconds"] == 15


def test_running_operation_rejects_a_second_resume(tmp_path):
    output = tmp_path / "saved"
    with saved_run(output, "run", {"output": output}):
        with pytest.raises(ValueError, match="already running"), saved_run(output, "run", {"output": output}, resume=True):
            pytest.fail("A second process must not modify the saved run")


def test_checkpoint_uses_completion_count_and_survives_a_partial_write(tmp_path):
    parent = tmp_path / "openevolve/checkpoints"
    for index in (4, 1, 9):
        checkpoint = parent / f"checkpoint_{index}"
        (checkpoint / "programs").mkdir(parents=True)
        (checkpoint / "programs/seed.json").write_text(json.dumps({"id": "seed"}))
        (checkpoint / "metadata.json").write_text(json.dumps({
            "last_iteration": index, "islands": [["seed"]], "best_program_id": "seed"}))
        os.utime(checkpoint, (index if index != 1 else 5, index if index != 1 else 5))
    (parent / "checkpoint_9/metadata.json").write_text("{")
    checkpoint, completed = latest_checkpoint(tmp_path)
    assert checkpoint.name == "checkpoint_1" and completed == 2
    assert json.loads((checkpoint / "metadata.json").read_text())["last_iteration"] == 4


def test_resume_command_works_from_a_new_process_and_directory(tmp_path, monkeypatch):
    (tmp_path / "artifacts/evolution/openevolve").mkdir(parents=True)
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "batch"
    monkeypatch.setattr(harness, "run_campaign", lambda *args, **options: _result())
    assert harness.run_batch(output, {"ghz3": baseline}) == 0
    result = subprocess.run([sys.executable, "-m", "qudits_on_qubits.evolution.harness", "resume", "--last"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert str(output) in result.stdout


def test_menu_can_resume_a_saved_operation(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(harness, "choose", lambda *args: "resume")
    monkeypatch.setattr(harness, "ask", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(harness, "resume_operation", lambda output: opened.append(output) or 0)
    assert harness.main([]) == 0
    assert opened == [tmp_path]


def test_saved_sessions_find_existing_runs_and_hide_child_campaigns(tmp_path):
    root = tmp_path / "artifacts/evolution"
    older, newer = root / "older", root / "newer"
    for output in (older, newer, newer / "campaign"):
        with saved_run(output, "run", {"output": output, "baselines": {"ghz3": tmp_path}}):
            pass
    for entry in checkpoints.history_directory().glob("*.json"):
        entry.unlink()
    os.utime(older / "resume.json", (10, 10))
    os.utime(newer / "resume.json", (20, 20))
    os.utime(newer / "campaign/resume.json", (30, 30))

    sessions = checkpoints.saved_sessions(root)

    assert [session["output"] for session in sessions] == [newer, older]


def test_saved_sessions_keep_custom_paths_and_ignore_invalid_entries(tmp_path):
    output = tmp_path / "custom results"
    with saved_run(output, "run", {"output": output}):
        pass
    history = checkpoints.history_directory()
    (history / "invalid.json").write_text("{")
    (history / "wrong_type.json").write_text('{"output": []}')
    (history / "missing.json").write_text(json.dumps({"output": str(tmp_path / "missing")}))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "resume.json").write_text("{")
    (history / "broken.json").write_text(json.dumps({"output": str(broken)}))

    sessions = checkpoints.saved_sessions(tmp_path / "empty")

    assert [session["output"] for session in sessions] == [output]
    assert len(list(history.glob("*.json"))) == 5


@pytest.mark.parametrize("status", ["completed", "running"])
def test_resume_without_path_selects_the_latest_by_default(tmp_path, monkeypatch, capsys, status):
    output = tmp_path / "saved"
    with saved_run(output, "run", {"output": output, "baselines": {"ghz3": tmp_path}}):
        pass
    checkpoints.save(output / "resume.json", {**checkpoints.load(output), "status": status})
    opened = []
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(harness, "resume_operation", lambda output: opened.append(output) or 0)

    assert harness.main(["resume"]) == 0
    assert opened == [output]
    text = capsys.readouterr().out
    assert "ghz3" in text and status in text and str(output) in text
    if status == "running":
        assert "running or interrupted" in text


def test_resume_can_select_an_older_session(tmp_path, monkeypatch):
    outputs = [tmp_path / "older", tmp_path / "newer"]
    for timestamp, output in enumerate(outputs, 10):
        with saved_run(output, "run", {"output": output}):
            pass
        os.utime(output / "resume.json", (timestamp, timestamp))
    opened = []
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    monkeypatch.setattr(harness, "resume_operation", lambda output: opened.append(output) or 0)

    assert harness.main(["resume"]) == 0
    assert opened == [outputs[0]]


def test_resume_last_without_sessions_has_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("The last option must not request input"))
    with pytest.raises(SystemExit) as stopped:
        harness.main(["resume", "--last"])
    assert stopped.value.code == 2
    assert "No saved sessions" in capsys.readouterr().err


def test_resume_last_rejects_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "resume_operation", lambda output: pytest.fail("Conflicting options must not run"))
    with pytest.raises(SystemExit) as stopped:
        harness.main(["resume", str(tmp_path), "--last"])
    assert stopped.value.code == 2


def test_menu_lists_saved_sessions_without_requesting_a_path(tmp_path, monkeypatch):
    output = tmp_path / "saved"
    with saved_run(output, "run", {"output": output}):
        pass
    answers = iter(["3", ""])
    opened = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(harness, "ask", lambda *args, **kwargs: pytest.fail("Select a session without a path"))
    monkeypatch.setattr(harness, "resume_operation", lambda output: opened.append(output) or 0)

    assert harness.main([]) == 0
    assert opened == [output]


def test_resume_session_picker_keeps_the_manual_path_option(tmp_path, monkeypatch):
    with saved_run(tmp_path / "saved", "run", {"output": tmp_path / "saved"}):
        pass
    answers = iter(["2", str(tmp_path / "other")])
    opened = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(harness, "resume_operation", lambda output: opened.append(output) or 0)

    assert harness.main(["resume"]) == 0
    assert opened == [tmp_path / "other"]


def test_history_write_failure_preserves_the_saved_run(tmp_path, monkeypatch, capsys):
    unavailable = tmp_path / "not_a_directory"
    unavailable.write_text("existing")
    monkeypatch.setattr(checkpoints, "history_directory", lambda: unavailable)
    output = tmp_path / "saved"

    with saved_run(output, "run", {"output": output}):
        pass

    assert checkpoints.load(output)["status"] == "completed"
    assert "Cannot save session history" in capsys.readouterr().out


def test_session_history_survives_abrupt_process_exit(tmp_path):
    output = tmp_path / "interrupted"
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from qudits_on_qubits.evolution.checkpoints import saved_run\n"
        "output = Path(sys.argv[1])\n"
        "with saved_run(output, 'run', {'output': output}):\n"
        "    os._exit(0)\n"
    )

    result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)

    assert result.returncode == 0, result.stderr
    session, = checkpoints.saved_sessions(tmp_path / "empty")
    assert session["output"] == output and session["status"] == "running"
