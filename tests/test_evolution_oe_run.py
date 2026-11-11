"""Check the OpenEvolve state search with a local model."""

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from qiskit import qpy, transpile

from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_run import run_campaign
from qudits_on_qubits.evolution.state_workloads import build_workload, evaluate_workload

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "src/qudits_on_qubits/quantum_circuits/CZgate3.qpy"


def _socketpair_works():
    try:
        left, right = socket.socketpair()
        with left, right:
            left.send(b"x")
            return right.recv(1) == b"x"
    except OSError:
        return False


def _baseline(path):
    path.mkdir()
    with LEGACY.open("rb") as stream:
        circuit, = qpy.load(stream)
    circuit = transpile(circuit, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    for index in reversed(range(len(circuit.data))):
        if circuit.data[index].operation.name == "barrier":
            del circuit.data[index]
    encoding = canonical_encoding(3)
    state = build_workload("ghz3", encoding, circuit)
    with (path / "ghz.qpy").open("wb") as stream:
        qpy.dump(state, stream)
    np.save(path / "E.npy", encoding, allow_pickle=False)
    files = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
             for name in ("ghz.qpy", "E.npy")}
    (path / "manifest.json").write_text(json.dumps({"files": files}))
    (path / "summary.json").write_text(json.dumps({"results": [{
        "workload": "ghz3", "method": "canonical_direct", "is_valid": True,
        "circuit_artifact": "ghz.qpy", "encoding_artifact": "E.npy",
    }]}))


def test_invalid_options_do_not_create_output(tmp_path):
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="iteration"):
        run_campaign(tmp_path / "baseline", output, iterations=0)
    assert not output.exists()
    with pytest.raises(ValueError, match="one iteration"):
        run_campaign(tmp_path / "baseline", output, iterations=2)
    assert not output.exists()
    with pytest.raises(ValueError, match="cost objective needs an architecture"):
        run_campaign(tmp_path / "baseline", output, objective="cost")
    assert not output.exists()


@pytest.mark.parametrize("limit", [0, -1, True, float("nan"), float("inf"), "5", 10**400])
def test_invalid_campaign_time_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="time limit"):
        run_campaign(tmp_path / "baseline", tmp_path / "run", campaign_timeout_seconds=limit)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("preexisting", [False, True])
def test_campaign_timeout_kills_process_group(tmp_path, monkeypatch, preexisting):
    from qudits_on_qubits.evolution import oe_run
    output = tmp_path / "run"
    pidfile = tmp_path / "child.pid"
    program = tmp_path / "hang.py"
    if preexisting:
        output.mkdir()
        (output / "existing.txt").write_text("keep")
    program.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).mkdir(exist_ok=True)\n"
        "Path(sys.argv[1], 'other.txt').write_text('keep')\n"
        "if os.fork() == 0:\n"
        "    Path(sys.argv[2]).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    time.sleep(60)\n"
    )
    popen = subprocess.Popen
    def start_hanging_worker(_args, **kwargs):
        return popen([sys.executable, str(program), str(output), str(pidfile)], **kwargs)
    monkeypatch.setattr(oe_run.subprocess, "Popen", start_hanging_worker)
    with pytest.raises(TimeoutError, match="time limit"):
        run_campaign(tmp_path / "baseline", output, campaign_timeout_seconds=1)
    assert (output / "other.txt").read_text() == "keep"
    assert not (output / "campaign_status.json").exists()
    if preexisting:
        assert (output / "existing.txt").read_text() == "keep"
    child = int(pidfile.read_text())
    for _ in range(40):
        try:
            state = Path(f"/proc/{child}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("The campaign child process is still running")


def test_codex_quota_output_filters_fields(monkeypatch, capsys):
    from qudits_on_qubits.evolution import codex_quota, oe_run
    monkeypatch.setattr(codex_quota, "read_codex_quota", lambda _path: {
        "primary_window": {"used_percent": 20, "remaining_percent": 80, "secret": "do-not-print"},
    })
    oe_run._print_codex_quota("private", "before")
    output = capsys.readouterr().out
    assert '"used_percent": 20' in output
    assert '"remaining_percent": 80' in output
    assert "do-not-print" not in output
    assert "private" not in output
    monkeypatch.setattr(codex_quota, "read_codex_quota", lambda _path: None)
    oe_run._print_codex_quota("private", "after")
    assert "unavailable" in capsys.readouterr().out


def test_cli_passes_campaign_and_quota_options(tmp_path, monkeypatch, capsys):
    from qudits_on_qubits.evolution import oe_run
    received = {}
    def fake_run(*args, **kwargs):
        received.update(kwargs)
        return {"saved": {"cz_count": 0, "F": 1.0}}
    monkeypatch.setattr(oe_run, "run_campaign", fake_run)
    auth = tmp_path / "private.json"
    monkeypatch.setattr(sys, "argv", ["oe_run", "--baseline", "base", "--output", "out",
                                      "--campaign-timeout-seconds", "12", "--codex-auth-file", str(auth)])
    oe_run.main()
    assert received["campaign_timeout_seconds"] == 12
    assert received["codex_auth_file"] == auth
    assert "CZ=0" in capsys.readouterr().out


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_openevolve_mock_saves_checked_state(tmp_path, capfd):
    pytest.importorskip("openevolve")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "run"
    auth = tmp_path / "private-account.json"
    auth.write_text('{}')
    result = run_campaign(baseline, output, state="ghz3", method="canonical_direct",
                          iterations=1, seed=0, max_nfev=0, timeout_seconds=20, cpu_seconds=20,
                          codex_auth_file=auth)
    assert result["saved"]["is_valid"]
    captured = capfd.readouterr().out
    assert "Codex quota (before): unavailable" in captured
    assert "Codex quota (after): unavailable" in captured
    assert (output / "best_program.py").exists()
    assert (output / "diagnostic_program.py").exists()
    assert "Evaluated program" in (output / "campaign.log").read_text()
    assert "done in" in (output / "progress.log").read_text() and "Metrics:" not in captured
    readme = (output / "README.md").read_text()
    assert "OpenEvolve 0.4.0" in readme
    assert ".[evolution]" in readme
    assert "not an LLM campaign" in readme
    assert "--campaign-timeout-seconds 900" in readme
    assert str(auth) not in readme
    assert '$QOQ_CODEX_AUTH_FILE' in readme
    assert json.loads((output / "configuration.json").read_text())["campaign_timeout_seconds"] == 900
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["initial_program"]["metrics"]["is_valid"]
    assert diagnostics["mock_program"]["metrics"]["cz_count"] == result["source"]["cz_count"] - 1
    with (output / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    encoding = np.load(output / "encoding.npy", allow_pickle=False)
    measured = evaluate_workload("ghz3", circuit, encoding, tolerance=1e-10)
    assert measured["is_valid"]
    assert measured["cz_count"] == result["saved"]["cz_count"]
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    programs = list((output / "population" / "programs").glob("*.json"))
    assert len(programs) >= 2
    with pytest.raises(FileExistsError):
        run_campaign(baseline, output, state="ghz3", method="canonical_direct")


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_openevolve_mock_skeleton_with_free_encodings(tmp_path):
    pytest.importorskip("openevolve")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "run"
    result = run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=1,
                          representation="skeleton", encoding_mode="free", restarts=8, workers=2,
                          timeout_seconds=120, cpu_seconds=120)
    assert result["saved"]["is_valid"]
    # Shorter than the fit tails (3 CZ per qutrit). The warm start from the baseline gives a different
    # valid encoding than random restarts, so the exact decoder length is not fixed.
    assert result["saved"]["decoder_cz_count"] < 9
    prompt = (output / "prompts" / "full_rewrite_user.txt").read_text()
    assert "Schmidt rank" in prompt and "{artifacts}" in prompt
    assert json.loads((output / "configuration.json").read_text())["encoding_mode"] == "free"
    encodings = np.load(output / "best_encodings.npy", allow_pickle=False)
    with (output / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    assert encodings.shape == (3, 4, 3)
    assert evaluate_workload("ghz3", circuit, encodings, tolerance=1e-10)["is_valid"]
    with (output / "decoder.qpy").open("rb") as stream:
        decoder, = qpy.load(stream)
    decoded = evaluate_workload("ghz3", circuit.compose(decoder), canonical_encoding(3), tolerance=1e-10)
    assert decoded["is_valid"]
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["mock_program"]["metrics"]["cz_count"] == result["source"]["cz_count"] - 1
    assert "--representation skeleton --encoding-mode free" in (output / "README.md").read_text()


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_campaign_without_a_valid_candidate_keeps_its_best_for_migration(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from dataclasses import replace
    from qudits_on_qubits.evolution import oe_run
    real = oe_run.assess

    def final_check_fails(path, **options):
        result = real(path, **options)
        if options["timeout_seconds"] == oe_run.MAX_SECONDS:
            return replace(result, metrics={**result.metrics, "is_valid": False})
        return result

    monkeypatch.setattr(oe_run, "assess", final_check_fails)
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="did not pass"):
        oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=1,
                             timeout_seconds=20, cpu_seconds=20)
    saved = json.loads((output / oe_run.INVALID_BEST).read_text())
    assert saved["is_valid"] is False and saved["cz_count"] > 0 and 0 <= saved["F"] <= 1
    assert (output / "best_program.py").is_file()


def test_campaign_stops_with_sigterm_before_the_limit_and_keeps_the_result(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import oe_run
    output = tmp_path / "run"
    output.mkdir()  # an existing directory has no progress reader
    program = tmp_path / "graceful.py"
    program.write_text(
        "import json, signal, sys, time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        "    Path(sys.argv[1]).write_text(json.dumps({'result': {'saved': {'cz_count': 3}}}))\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    popen = subprocess.Popen
    monkeypatch.setattr(oe_run.subprocess, "Popen",
                        lambda args, **kwargs: popen([sys.executable, str(program), args[3]], **kwargs))
    monkeypatch.setattr(oe_run, "POLL_SECONDS", 0.05)  # the stop window of a 3 s campaign is 0.3 s
    started = time.monotonic()
    result = run_campaign(tmp_path / "baseline", output, campaign_timeout_seconds=3)
    assert result == {"saved": {"cz_count": 3}}
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "5", 10**9])
def test_invalid_polish_time_limit(tmp_path, value):
    with pytest.raises(ValueError, match="polish time limit"):
        run_campaign(tmp_path / "baseline", tmp_path / "run", polish_seconds=value)
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_polished_campaign_saves_a_checked_state(tmp_path):
    pytest.importorskip("openevolve")
    from qudits_on_qubits.evolution import oe_run
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "run"
    result = oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=1,
                                  representation="skeleton", timeout_seconds=20, cpu_seconds=20,
                                  polish_seconds=30)
    assert result["saved"]["is_valid"]
    assert result["saved"]["two_qubit_depth"] <= result["source"]["two_qubit_depth"]
    assert "polish" in result
    assert json.loads((output / "configuration.json").read_text())["polish_seconds"] == 30
    assert "--polish-seconds 30" in (output / "README.md").read_text()


@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5"])
def test_invalid_minimum_seed_fidelity_creates_no_output(tmp_path, value):
    from qudits_on_qubits.evolution import oe_run
    with pytest.raises(ValueError, match="seed fidelity"):
        oe_run._run_campaign(tmp_path / "baseline", tmp_path / "run", min_seed_fidelity=value)
    assert not (tmp_path / "run").exists()


def test_invalid_best_is_saved_from_its_search_metrics_without_a_new_fit(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    import openevolve.api
    from qudits_on_qubits.evolution import oe_run
    from qudits_on_qubits.evolution.oe_candidates import skeleton_program
    best = skeleton_program(6, [[0, 2], [1, 3], [0, 4]])
    metrics = {"is_valid": 0.0, "F": 0.25, "L": 0.5, "cz_count": 3.0, "decoder_cz_count": 0.0,
               "two_qubit_depth": 2.0, "depth": 5.0, "one_qubit_gate_count": 12.0}
    monkeypatch.setattr(openevolve.api, "run_evolution", lambda *a, **k: openevolve.api.EvolutionResult(
        best_program=None, best_score=0.125, best_code=best, metrics=metrics, output_dir=None))
    real = oe_run.assess

    def no_final_fit(path, **options):
        assert options["timeout_seconds"] != oe_run.MAX_SECONDS, "the invalid best got a new fit"
        return real(path, **options)

    monkeypatch.setattr(oe_run, "assess", no_final_fit)
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="did not pass"):
        oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=1,
                             representation="skeleton", timeout_seconds=20, cpu_seconds=20)
    saved = json.loads((output / oe_run.INVALID_BEST).read_text())
    assert saved["is_valid"] is False and saved["cz_count"] == 3 and saved["F"] == 0.25
    assert (output / "best_program.py").read_text() == best


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_seed_below_the_minimum_fidelity_is_replaced_by_the_known_structure(tmp_path):
    pytest.importorskip("openevolve")
    from qudits_on_qubits.evolution import oe_run
    from qudits_on_qubits.evolution.oe_candidates import load_skeleton, skeleton_program
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    seed = tmp_path / "seed.py"
    seed.write_text(skeleton_program(6, [[0, 1], [0, 1]]))  # no CZ across a party cut: the fit fails
    output = tmp_path / "run"
    result = oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=1,
                                  representation="skeleton", timeout_seconds=20, cpu_seconds=20,
                                  initial_program=seed, min_seed_fidelity=0.5)
    assert result["saved"]["is_valid"]
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["seed_check"] == {"F": 0.0, "min_F": 0.5, "replaced": True}
    assert "cut" in diagnostics["rejected_seed"]["error"]
    assert diagnostics["initial_program"]["metrics"]["is_valid"]
    source = oe_run.skeleton_of(oe_run._source(baseline, "ghz3", "canonical_direct")[0])
    assert tuple(load_skeleton(output / "initial_program.py", expected_qubits=6)) == tuple(source)
    assert "fidelity F=0.0000 in its first fit" in (output / "prompts" / "full_rewrite_user.txt").read_text()
