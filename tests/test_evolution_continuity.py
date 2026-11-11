"""Check parameter and builder inheritance across search iterations."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from qudits_on_qubits.evolution import agent_tools, oe_agent, oe_evaluator
from qudits_on_qubits.evolution.oe_candidates import load_skeleton, skeleton_program
from test_evolution_agent import TWO_QUTRIT, _ask, _campaign, _client


def _metrics(fidelity):
    return {"is_valid": False, "F": fidelity, "cz_count": 3, "two_qubit_depth": 3, "depth": 7,
            "one_qubit_gate_count": 16, "error_norm": 0.9, "leakage_norm": 0.1, "fit_restarts": 1,
            "decoder_cz_count": 0, "evaluation_seconds": 1.0, "combined_score": fidelity / 2}


def test_invalid_fit_continues_and_keeps_the_best_angles(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path, mode="fixed")))
    candidate = tmp_path / "candidate.py"
    candidate.write_text(skeleton_program(4, TWO_QUTRIT))
    starts = []

    def assess(path, **options):
        starts.append(options["x0"])
        fidelity = (0.25, 0.5, 0.1)[len(starts) - 1]
        return SimpleNamespace(metrics=_metrics(fidelity), angles=np.full(30, len(starts)))

    monkeypatch.setattr(oe_evaluator, "assess", assess)
    results = [oe_evaluator.evaluate(candidate) for _ in range(4)]
    assert starts[0] is None
    np.testing.assert_array_equal(starts[1], np.ones(30))
    np.testing.assert_array_equal(starts[2], np.full(30, 2))
    assert [r.metrics["F"] for r in results] == [0.25, 0.5, 0.5, 0.5]
    assert [r.metrics["cache_hit"] for r in results] == [0, 0, 0, 1]
    cache, = (tmp_path / oe_evaluator.CACHE_DIRECTORY).glob("*.json")
    assert json.loads(cache.read_text())["angles"] == [2.0] * 30
    config = tmp_path / "evaluation.json"
    data = json.loads(config.read_text()) | {"fit_seed": 7}
    config.write_text(json.dumps(data))
    monkeypatch.setattr(oe_evaluator, "assess", lambda *a, **kw: SimpleNamespace(
        metrics=_metrics(0.75), angles=np.full(30, 3)))
    assert oe_evaluator.evaluate(candidate).metrics["F"] == 0.75


def test_final_save_uses_fitted_angles_instead_of_the_submitted_start(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path, mode="fixed")))
    candidate = tmp_path / "candidate.py"
    candidate.write_text(skeleton_program(4, TWO_QUTRIT, np.zeros(30)))
    monkeypatch.setattr(oe_evaluator, "assess", lambda *a, **kw: SimpleNamespace(
        metrics=_metrics(0.5), angles=np.ones(30)))
    oe_evaluator.evaluate(candidate)
    options = oe_evaluator._config()
    np.testing.assert_array_equal(oe_evaluator._warm_start(candidate, options), np.zeros(30))
    np.testing.assert_array_equal(oe_evaluator._warm_start(candidate, options, exact=True), np.ones(30))


def test_cache_keeps_the_smaller_error_when_fidelity_rounds_to_one(tmp_path):
    entry = tmp_path / "cache" / "fit.json"
    first = {**_metrics(1.0), "error_norm": 1e-8}
    worse = {**first, "error_norm": 2e-8}
    oe_evaluator._store(entry, first, "first", {"pairs": TWO_QUTRIT, "angles": [1.0] * 30})
    metrics, _ = oe_evaluator._store(entry, worse, "worse", {"pairs": TWO_QUTRIT, "angles": [2.0] * 30})
    assert metrics["error_norm"] == 1e-8
    assert json.loads(entry.read_text())["angles"] == [1.0] * 30


def test_valid_numeric_cache_entry_needs_no_new_fit(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path, mode="fixed")))
    candidate = tmp_path / "candidate.py"
    candidate.write_text(skeleton_program(4, TWO_QUTRIT))
    metrics = {**_metrics(1.0), "is_valid": True, "error_norm": 1e-15}
    calls = []
    monkeypatch.setattr(oe_evaluator, "assess", lambda *a, **kw: calls.append(1) or SimpleNamespace(
        metrics=metrics, angles=np.ones(30)))
    assert oe_evaluator.evaluate(candidate).metrics["cache_hit"] == 0
    assert oe_evaluator.evaluate(candidate).metrics["cache_hit"] == 1
    assert calls == [1]


def test_fit_tool_keeps_invalid_progress_without_publishing_it_as_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(_campaign(tmp_path, mode="fixed")))
    monkeypatch.chdir(tmp_path)
    starts = []

    def fit(*args, **options):
        starts.append(options["x0"])
        return SimpleNamespace(metrics=_metrics(0.5), angles=np.ones(30), restarts=1)

    monkeypatch.setattr(agent_tools, "fit_skeleton", fit)
    for _ in range(2):
        assert not agent_tools.fit(TWO_QUTRIT, write="candidate.py")["valid"]
    assert starts[0] is None
    np.testing.assert_array_equal(starts[1], np.ones(30))
    assert not (tmp_path / "candidate.py").exists()
    assert not (tmp_path / agent_tools.SESSION_FITS).exists()
    assert json.loads((tmp_path / agent_tools.SESSION_PROGRESS).read_text())[0]["F"] == 0.5
    (tmp_path / agent_tools.SESSION_PROGRESS).write_text('{}')
    with pytest.raises(ValueError, match="progress file"):
        agent_tools.fit(TWO_QUTRIT)
    assert len(starts) == 2


def test_agent_inherits_the_builder_of_the_selected_parent(tmp_path, monkeypatch):
    client, campaign = _client(tmp_path, monkeypatch, "nothing")
    builder = "def build():\n    return '```'\n"
    parent = oe_agent.with_builder(skeleton_program(4, TWO_QUTRIT), builder)
    seen = []

    def run(argv, task, session, seconds, env):
        seen.append((session / "builder.py").read_text())
        (session / "builder.py").write_text(builder + "# revised\n")
        (session / "candidate.py").write_text(skeleton_program(4, TWO_QUTRIT))
        return False

    monkeypatch.setattr(oe_agent, "_run", run)
    answer = _ask(client, "# Current candidate\n```python\n" + parent + "```\n")
    code = answer.removeprefix("```python\n").removesuffix("\n```")
    assert seen == [builder]
    assert oe_agent.builder_source(code) == builder + "# revised\n"
    candidate = tmp_path / "archived.py"
    candidate.write_text(code)
    assert load_skeleton(candidate, expected_qubits=4) == TWO_QUTRIT
    assert "Phase: construction" in next((campaign / oe_agent.SESSIONS).glob("*/TASK.md")).read_text()


def test_builder_source_is_data_and_has_a_size_limit(tmp_path):
    destination = tmp_path / "executed"
    builder = f"open({str(destination)!r}, 'w').write('bad')\n"
    code = oe_agent.with_builder(skeleton_program(4, TWO_QUTRIT), builder)
    assert oe_agent.builder_source(code) == builder
    assert oe_agent.builder_source(oe_agent.with_builder(code, "pass\n")) == "pass\n"
    described = oe_agent.with_builder('"A literal candidate."\n' + skeleton_program(4, TWO_QUTRIT), builder)
    path = tmp_path / "described.py"
    path.write_text(described)
    assert load_skeleton(path, expected_qubits=4) == TWO_QUTRIT
    assert not destination.exists()
    with pytest.raises(ValueError, match="too large"):
        oe_agent.with_builder(code, "x" * (oe_agent.MAX_BUILDER_BYTES + 1))
    with pytest.raises(ValueError, match="syntax"):
        oe_agent.with_builder(code, "def broken(")
    with pytest.raises(ValueError, match="empty"):
        oe_agent.with_builder(code, " ")


def test_agent_rejects_a_candidate_without_its_builder(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, "nothing")

    def run(argv, task, session, seconds, env):
        (session / "candidate.py").write_text(skeleton_program(4, TWO_QUTRIT))
        return False

    monkeypatch.setattr(oe_agent, "_run", run)
    with pytest.raises(RuntimeError, match="builder.py"):
        _ask(client)


def test_phase_requires_a_valid_result(tmp_path):
    assert oe_agent._phase(tmp_path) == "construction"
    cache = tmp_path / oe_evaluator.CACHE_DIRECTORY
    cache.mkdir()
    entry = cache / "fit.json"
    entry.write_text(json.dumps({"metrics": {"is_valid": False}}))
    assert oe_agent._phase(tmp_path) == "construction"
    entry.write_text(json.dumps({"metrics": {"is_valid": True}}))
    assert oe_agent._phase(tmp_path) == "compression"


def test_campaign_keeps_seed_angles_and_builder_through_final_verification(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import oe_run
    from qudits_on_qubits.evolution.oe_candidates import candidate_angles
    from qudits_on_qubits.evolution.skeleton_fit import known_angles
    from test_evolution_oe_run import _baseline
    import openevolve.api
    import sys

    baseline = tmp_path / "baseline"
    _baseline(baseline)
    circuit = oe_run._source(baseline, "ghz3", "canonical_direct")[0]
    pairs, angles = known_angles(circuit)
    initial = tmp_path / "seed.py"
    builder = "def build():\n    return 'seed'\n"
    initial.write_text(oe_agent.with_builder(skeleton_program(6, pairs, angles), builder))
    monkeypatch.setenv("QOQ_AGENT_BINARY", sys.executable)

    def evolution(path, evaluator, **options):
        np.testing.assert_allclose(candidate_angles(path), angles)
        assert oe_agent.builder_source(path.read_text()) == builder
        return SimpleNamespace(best_code=path.read_text(), metrics={"is_valid": True})

    monkeypatch.setattr(openevolve.api, "run_evolution", evolution)
    output = tmp_path / "run"
    result = oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct",
                                  representation="skeleton", initial_program=initial, blank_start=True,
                                  offline_model="agent", timeout_seconds=60, cpu_seconds=60)
    assert result["saved"]["is_valid"]
    assert result["best"]["nfev"] == 0
    assert (output / "best_builder.py").read_text() == builder
    assert oe_agent.builder_source((output / "best_program.py").read_text()) == builder


def test_cz_removal_keeps_all_other_gates_in_the_warm_start():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qudits_on_qubits.evolution.skeleton_fit import _circuits, warm_angles

    pairs = ((0, 1), (1, 2), (2, 3), (0, 3))
    angles = np.random.default_rng(5).uniform(-1, 1, 3 * 4 + 6 * len(pairs))
    keep = (pairs[0][::-1], pairs[2])
    start = warm_angles(keep, [(pairs, angles)], 4)
    actual = _circuits(4, keep, (), start, 0)[0]
    original = _circuits(4, pairs, (), angles, 0)[0]
    expected = QuantumCircuit(4)
    cz_index = 0
    for item in original.data:
        if item.operation.name == "cz":
            cz_index += 1
            if cz_index in (2, 4):
                continue
        expected.append(item.operation, [original.find_bit(q).index for q in item.qubits])
    assert Operator(actual).equiv(Operator(expected), atol=1e-12)


def test_discovery_keeps_the_full_program_between_rounds(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import discovery
    from qudits_on_qubits.evolution.oe_candidates import candidate_angles
    from test_evolution_oe_run import _baseline
    import sys

    baseline = tmp_path / "baseline"
    _baseline(baseline)
    angles = [0.125] * 48
    code = oe_agent.with_builder(skeleton_program(6, ((0, 1), (0, 2), (1, 3), (0, 4), (1, 5)), angles), "pass\n")
    seeds = []

    def campaign(base, output, **options):
        assert options["min_seed_fidelity"] == 0
        seeds.append(options["initial_program"].read_text())
        output.mkdir()
        (output / "best_program.py").write_text(code)
        return {"saved": {**_metrics(1.0), "is_valid": True}}

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    monkeypatch.setenv("QOQ_AGENT_BINARY", sys.executable)
    result = discovery.run_discovery(baseline, tmp_path / "discovery", state="ghz3",
                                     method="canonical_direct", families=1, rounds=2, offline_model="agent")
    assert result["status"] == "completed"
    assert seeds[1] == code
    seed = tmp_path / "next.py"
    seed.write_text(seeds[1])
    assert candidate_angles(seed) == angles
    assert oe_agent.builder_source(seeds[1]) == "pass\n"
