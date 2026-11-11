"""Check the agent loop: candidates with angles, the fit tool, the Codex session client and a campaign."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from qudits_on_qubits.evolution import agent_tools, harness, oe_agent
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_candidates import candidate_angles, load_skeleton, skeleton_program
from qudits_on_qubits.evolution.oe_evaluator import _cache_entry, _warm_start, assess
from qudits_on_qubits.evolution.skeleton_fit import cut_seed, cut_violations, fit_skeleton

from test_evolution_oe_run import _baseline, _socketpair_works

TWO_QUTRIT = ((0, 2), (0, 3), (1, 3))
GHZ3 = ((0, 1), (0, 2), (1, 3), (0, 4), (1, 5))
FAKE_CODEX = """
import json, os, subprocess, sys, time
from pathlib import Path
prompt = sys.stdin.read()
Path("args.json").write_text(json.dumps(sys.argv[1:]))
Path("prompt.txt").write_text(prompt)
mode = os.environ["QOQ_FAKE_MODE"]
if mode == "sleep":
    time.sleep(60)
if mode == "write":
    Path("candidate.py").write_text('CANDIDATE = {"num_qubits": 4, "cz": [[0, 2], [0, 3], [1, 3]]}\\n')
    Path("notes.txt").write_text("DISCOVERY: one\\nnoise\\nDISCOVERY: two\\nDISCOVERY: three\\nDISCOVERY: four\\n")
    Path("builder.py").write_text("CANDIDATE = {}\\n")
if mode == "tool":
    subprocess.run([sys.executable, "-m", "qudits_on_qubits.evolution.agent_tools", "fit", "--cz",
                    os.environ["QOQ_FAKE_CZ"], "--restarts", "8", "--write", "candidate.py"], check=True)
    Path("notes.txt").write_text("DISCOVERY: party 0 copies its superposition to the other parties\\n")
if mode == "tool":
    if Path("builder.py").exists():
        Path("inherited_builder.py").write_text(Path("builder.py").read_text())
    Path("builder.py").write_text("from qudits_on_qubits.evolution.agent_tools import fit\\n"
                                   "fit(" + os.environ["QOQ_FAKE_CZ"] + ", write='candidate.py')\\n")
"""


def _campaign(directory, state="two_qutrit", mode="free"):
    np.save(directory / "encoding.npy", canonical_encoding(3), allow_pickle=False)
    config = directory / "evaluation.json"
    config.write_text(json.dumps({"state": state, "encoding_path": str(directory / "encoding.npy"),
                                  "representation": "skeleton", "encoding_mode": mode}))
    return config


def _fake_codex(directory, monkeypatch, mode, seconds="30"):
    path = directory / "fake_codex"
    path.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    path.chmod(0o755)
    monkeypatch.setenv("QOQ_AGENT_BINARY", str(path))
    monkeypatch.setenv("QOQ_AGENT_SECONDS", seconds)
    monkeypatch.setenv("QOQ_FAKE_MODE", mode)


def _client(tmp_path, monkeypatch, mode, seconds="30"):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(campaign)))
    _fake_codex(tmp_path, monkeypatch, mode, seconds)
    return oe_agent.AgentClient(SimpleNamespace(name="qoq-codex-agent")), campaign


def _ask(client, text="# Current candidate\nPARENT"):
    return asyncio.run(client.generate_with_context("system", [{"role": "user", "content": text}]))


def test_skeleton_angles_round_trip(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text(skeleton_program(4, TWO_QUTRIT, [0.5, -1.0, 2.0]))
    assert load_skeleton(path, expected_qubits=4) == TWO_QUTRIT
    assert candidate_angles(path) == [0.5, -1.0, 2.0]
    path.write_text(skeleton_program(4, TWO_QUTRIT))
    assert candidate_angles(path) is None


@pytest.mark.parametrize("angles", ["[1e400]", "['x']", "[True]", "1.0"])
def test_bad_candidate_angles_are_rejected(tmp_path, angles):
    path = tmp_path / "candidate.py"
    path.write_text('CANDIDATE = {"num_qubits": 4, "cz": [[0, 2]], "angles": %s}\n' % angles)
    with pytest.raises(ValueError, match="angle"):
        candidate_angles(path)


def test_skeleton_with_an_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text('CANDIDATE = {"num_qubits": 4, "cz": [[0, 2]], "code": "x"}\n')
    with pytest.raises(ValueError, match="fields"):
        load_skeleton(path, expected_qubits=4)


def test_candidate_angles_are_only_verified(tmp_path, monkeypatch):
    fitted = fit_skeleton("two_qutrit", TWO_QUTRIT, None, restarts=8, seed=0)
    path = tmp_path / "candidate.py"
    path.write_text(skeleton_program(4, TWO_QUTRIT, fitted.angles))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(tmp_path / "evaluation.json"))
    x0 = _warm_start(path, {"state": "two_qutrit", "representation": "skeleton", "encoding_mode": "free"})
    result = assess(path, state="two_qutrit", encoding=canonical_encoding(3), representation="skeleton",
                    encoding_mode="free", x0=x0, timeout_seconds=60, cpu_seconds=60)
    assert result.metrics["is_valid"]
    assert result.metrics["nfev"] == 0


def test_cache_key_depends_on_the_angles(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(tmp_path / "evaluation.json"))
    paths = [tmp_path / f"{name}.py" for name in "abc"]
    for path, angles in zip(paths, (None, [1.0], [2.0])):
        path.write_text(skeleton_program(4, TWO_QUTRIT, angles))
    options = {"state": "two_qutrit", "representation": "skeleton"}
    assert len({_cache_entry(path, options) for path in paths}) == 3


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_cut_seed_meets_every_cut_without_a_known_circuit(state):
    pairs = cut_seed(state)
    assert len(pairs) >= 2
    assert cut_violations(state, pairs) == []


def test_fit_tool_writes_a_valid_candidate_and_keeps_the_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path)))
    monkeypatch.chdir(tmp_path)
    assert agent_tools.main(["fit", "--cz", json.dumps(TWO_QUTRIT), "--write", "candidate.py"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["cz_count"] == 3 and report["written"] == "candidate.py"
    assert load_skeleton(tmp_path / "candidate.py", expected_qubits=4) == TWO_QUTRIT
    assert len(candidate_angles(tmp_path / "candidate.py")) == 3 * 4 + 6 * (3 + 2 * 3)
    fits = json.loads((tmp_path / agent_tools.SESSION_FITS).read_text())
    assert fits[0]["pairs"] == [list(pair) for pair in TWO_QUTRIT]


def test_fit_tool_reports_cut_violations_without_a_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path)))
    monkeypatch.chdir(tmp_path)
    assert agent_tools.main(["fit", "--cz", "[[0, 1]]", "--write", "candidate.py"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"] and report["cut_violations"]
    assert not (tmp_path / "candidate.py").exists()


def test_fit_tool_rejects_code_in_the_cz_argument(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path)))
    assert agent_tools.main(["fit", "--cz", "__import__('os').system('true')"]) == 2
    assert "Error" in capsys.readouterr().err


def test_agent_session_returns_its_candidate_and_keeps_three_discoveries(tmp_path, monkeypatch):
    client, campaign = _client(tmp_path, monkeypatch, "write")
    answer = _ask(client)
    assert answer.startswith("```python\n") and answer.rstrip().endswith("```")
    assert oe_agent.builder_source(answer.removeprefix("```python\n").removesuffix("\n```")) == "CANDIDATE = {}\n"
    first, = (campaign / oe_agent.SESSIONS).iterdir()
    args = json.loads((first / "args.json").read_text())
    assert args[:2] == ["exec", "--json"]
    assert "workspace-write" in args and "sandbox_workspace_write.network_access=false" in args
    assert "PARENT" in (first / "prompt.txt").read_text()
    notes = (campaign / oe_agent.DISCOVERIES).read_text().splitlines()
    assert len(notes) == 3 and notes[0].endswith("DISCOVERY: one") and notes[2].endswith("DISCOVERY: three")
    _ask(client)
    second, = set((campaign / oe_agent.SESSIONS).iterdir()) - {first}
    assert "DISCOVERY: one" in (second / "prompt.txt").read_text()


def test_agent_session_without_a_candidate_fails(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, "nothing")
    with pytest.raises(RuntimeError, match="candidate.py"):
        _ask(client)


def test_agent_session_stops_at_its_time_limit(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, "sleep", seconds="1")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="time limit"):
        _ask(client)
    assert time.monotonic() - started < 30


@pytest.mark.parametrize("name, value", [("QOQ_AGENT_SECONDS", "0"), ("QOQ_AGENT_SECONDS", "x"),
                                         ("QOQ_AGENT_EFFORT", "huge"), ("QOQ_AGENT_MODEL", "a b"),
                                         ("QOQ_AGENT_FAST", "yes")])
def test_bad_agent_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        oe_agent.agent_settings()


def test_agent_max_effort_and_fast_tier_reach_the_codex_command(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_AGENT_BINARY", sys.executable)
    monkeypatch.setenv("QOQ_AGENT_EFFORT", "max")
    monkeypatch.setenv("QOQ_AGENT_FAST", "1")
    settings = oe_agent.agent_settings()
    argv = oe_agent._argv(settings, tmp_path, {})
    assert 'model_reasoning_effort="max"' in argv and 'service_tier="fast"' in argv
    assert 'service_tier="fast"' not in oe_agent._argv({**settings, "fast": False}, tmp_path, {})


def test_claude_session_runs_jailed_and_codex_gets_the_writable_roots(tmp_path):
    settings = {"binary": "/usr/bin/claude", "model": "claude-opus-5-5", "effort": "high", "fast": False}
    shared = tmp_path / "shared"
    argv = oe_agent._argv(settings, tmp_path, {}, writable=[shared])
    assert argv[:3] == ["bwrap", "--ro-bind", "/"] and ["--bind", str(shared), str(shared)] == argv[13:16]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--add-dir") + 1] == str(shared) and "--dangerously-skip-permissions" not in argv
    codex = oe_agent._argv({**settings, "binary": "codex"}, tmp_path, {}, writable=[shared])
    assert f"sandbox_workspace_write.writable_roots={json.dumps([str(shared)])}" in codex


def test_agent_session_does_not_get_the_proxy_key(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_PROXY_KEY", "secret")
    argv = [sys.executable, "-c", "import os; print(os.environ.get('QOQ_PROXY_KEY', 'none'))"]
    assert oe_agent._run(argv, "", tmp_path, 30, {"OMP_NUM_THREADS": "1"}) is False
    assert (tmp_path / "events.jsonl").read_text().strip() == "none"


def test_agent_network_reads_codex_and_claude_usage(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location("agent_network", Path(__file__).parents[1] / "scripts" / "agent_network.py")
    network = module_from_spec(spec)
    spec.loader.exec_module(network)
    events = tmp_path / "events.jsonl"
    events.write_text('{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}\nnot json\n'
                      '{"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}\n')
    assert network._session_usage(events) == {"input_tokens": 6, "output_tokens": 3}
    message = {"type": "assistant", "message": {"id": "m", "usage": {"input_tokens": 3, "output_tokens": 1}}}
    events.write_text(json.dumps(message) + "\n" + json.dumps(message) + "\n")
    assert network._session_usage(events) == {"input_tokens": 3, "output_tokens": 1}  # killed: per message
    result = {"type": "result", "usage": {"input_tokens": 4, "output_tokens": 9, "cache_creation": {"x": 1}}}
    events.write_text(json.dumps(message) + "\n" + json.dumps(result) + "\n")
    assert network._session_usage(events) == {"input_tokens": 4, "output_tokens": 9}


def test_vector_only_description_hides_the_state(monkeypatch):
    from qudits_on_qubits.evolution.state_workloads import describe

    monkeypatch.setenv("QOQ_VECTOR_ONLY", "/runs/target.npy")
    text, units = describe("ame43", "fixed")
    assert "/runs/target.npy" in text and "AME" not in text and "graph" not in text and units == "blocks"


def test_vector_only_description_supplies_the_amplitudes(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution.state_workloads import describe

    vector = np.zeros(16, dtype=complex)
    vector[2], vector[9] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    path = tmp_path / "target.npy"
    np.save(path, vector, allow_pickle=False)
    monkeypatch.setenv("QOQ_VECTOR_ONLY", str(path))
    monkeypatch.setenv("QOQ_VECTOR_INLINE", "1")
    text, units = describe("two_qutrit", "fixed")
    assert "[2, 0.7071067811865475, 0.0]" in text
    assert "[9, 0.0, 0.7071067811865475]" in text
    assert "index, real, imaginary" in text and "two_qutrit" not in text
    assert units == "blocks"
    np.save(path, np.ones(16), allow_pickle=False)
    with pytest.raises(ValueError, match="normalized"):
        describe("two_qutrit", "fixed")


def test_agent_network_blank_setup_has_no_source_or_known_angles(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location("agent_network", Path(__file__).parents[1] / "scripts/agent_network.py")
    network = module_from_spec(spec)
    spec.loader.exec_module(network)
    output = tmp_path / "network"
    options, pairs, prompt = network._setup("ame43", output, "fixed", 16, blank_start=True)
    assert pairs == cut_seed("ame43")
    assert json.loads((output / "warm_starts.json").read_text()) == []
    assert not (output / "source.qpy").exists()
    assert "known valid structure" not in prompt


def test_batch_passes_the_agent_and_the_blank_start(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)

    def run(baseline, output, **kwargs):
        calls.append(kwargs)
        return {"saved": {"is_valid": True, "cz_count": 5, "two_qubit_depth": 3, "depth": 4,
                          "one_qubit_gate_count": 5, "F": 1.0, "L": 0.0}}
    monkeypatch.setattr(harness, "run_campaign", run)
    assert harness.main(["run", "--output", str(tmp_path / "batch"), "--state", "ghz3",
                         "--baseline-ghz3", str(tmp_path), "--representation", "skeleton",
                         "--offline-model", "agent", "--blank-start", "--iterations", "2"]) == 0
    assert calls[0]["offline_model"] == "agent" and calls[0]["blank_start"] is True


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_agent_campaign_from_a_blank_start_saves_the_agent_circuit(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from qudits_on_qubits.evolution.oe_run import run_campaign
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    _fake_codex(tmp_path, monkeypatch, "tool", seconds="120")
    monkeypatch.setenv("QOQ_FAKE_CZ", json.dumps(GHZ3))
    output = tmp_path / "run"
    result = run_campaign(baseline, output, state="ghz3", method="canonical_direct", iterations=2,
                          representation="skeleton", encoding_mode="free", restarts=2,
                          offline_model="agent", blank_start=True, timeout_seconds=120, cpu_seconds=120,
                          campaign_timeout_seconds=600)
    assert result["saved"]["is_valid"] and result["saved"]["cz_count"] == 5
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["blank_start"] and configuration["model_mode"] == "codex_agent"
    assert configuration["agent"]["seconds"] == 120
    assert "known valid structure" not in (output / "prompts" / "full_rewrite_user.txt").read_text()
    assert json.loads((output / "warm_starts.json").read_text()) == []
    assert "copies its superposition" in (output / oe_agent.DISCOVERIES).read_text()
    builder = (output / "best_builder.py").read_text()
    assert "agent_tools import fit" in builder
    assert oe_agent.builder_source((output / "best_program.py").read_text()) == builder
    assert result["agent_search"]["sessions"] == 2
    for session in (output / oe_agent.SESSIONS).iterdir():
        source = oe_agent.builder_source((session / "parent_candidate.py").read_text())
        inherited = session / "inherited_builder.py"
        if source is None:
            assert not inherited.exists()
        else:
            assert inherited.read_text() == source
    assert json.loads((output / "agent_usage.json").read_text())["sessions"] == 2
    assert (output / "target.npy").is_file()
