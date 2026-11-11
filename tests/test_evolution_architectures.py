"""Check the offline QPU architectures and their selection in the harness."""

import json

import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.evolution import architectures, harness
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_candidates import skeleton_program
from qudits_on_qubits.evolution.oe_evaluator import assess, rank_key, score_metrics

TWO_QUTRIT_FREE = ((0, 2), (0, 3), (1, 3))


def test_select_expands_groups_once_in_order():
    assert architectures.select(["ibm"]) == ("ibm_heron_r2", "ibm_heron_r3", "ibm_nighthawk")
    assert architectures.select(["iqm_garnet", "iqm", "all"]) == (
        "iqm_garnet", "iqm_emerald", "iqm_sirius", "ibm_heron_r2", "ibm_heron_r3", "ibm_nighthawk")
    assert architectures.select([]) == ()
    with pytest.raises(ValueError, match="Unknown architecture"):
        architectures.select(["ibm_eagle"])


@pytest.mark.parametrize("name", list(architectures.ARCHITECTURES))
def test_each_architecture_routes_a_triangle(name):
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(0, 2)
    routed = architectures.route(circuit, name)
    device = architectures.backend(name)
    costs = architectures.routed_costs(circuit, name)
    assert costs["routed_two_qubit_count"] >= 3 and 0 < costs["routed_success_estimate"] < 1
    if name == "iqm_sirius":
        assert "move" in routed.count_ops() and routed.num_qubits == device.num_qubits + 1  # plus the resonator
        return
    edges = {tuple(sorted(edge)) for edge in device.coupling_map.get_edges()}
    pairs = [tuple(sorted(routed.find_bit(q).index for q in item.qubits))
             for item in routed.data if item.operation.num_qubits == 2]
    assert routed.num_qubits == device.num_qubits and set(pairs) <= edges


def test_star_routing_puts_the_resonator_last_for_the_costs():
    # The IQM Star plugin can put its resonator register first (the order changes between processes).
    from qiskit import QuantumRegister

    device = architectures.backend("iqm_sirius")
    resonator, qubits = QuantumRegister(1, "resonators"), QuantumRegister(device.num_qubits, "q")
    routed = QuantumCircuit(resonator, qubits)
    routed.append(device.target_with_resonators.operation_from_name("move"), [qubits[0], resonator[0]])
    routed.cz(qubits[1], resonator[0])
    fixed = architectures._resonators_last(routed)
    assert fixed.find_bit(resonator[0]).index == device.num_qubits
    assert [item.operation.name for item in fixed.data] == ["move", "cz"]
    assert architectures._costs(fixed, device)["routed_success_estimate"] == pytest.approx(1 - 0.0128)
    assert architectures._resonators_last(fixed) is fixed


def _route_in_child(circuit):
    architectures.routed_costs(circuit, "ibm_heron_r3")


def test_forked_child_routes_after_the_parent(monkeypatch):
    # OpenEvolve forks its evaluators after the campaign process routed its diagnostics.
    import multiprocessing

    monkeypatch.delenv("QISKIT_IN_PARALLEL", raising=False)
    circuit = QuantumCircuit(4)
    for pair in ((0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (1, 3)):
        circuit.h(pair[0])
        circuit.cz(*pair)
    architectures.routed_costs(circuit, "ibm_heron_r3")
    child = multiprocessing.get_context("fork").Process(target=_route_in_child, args=(circuit,))
    child.start()
    child.join(60)
    hung = child.is_alive()
    if hung:
        child.kill()
    assert not hung and child.exitcode == 0


def test_snapshots_hold_the_named_processors():
    heron = architectures.backend("ibm_heron_r3")
    nighthawk = architectures.backend("ibm_nighthawk")
    assert (heron.num_qubits, len(heron.coupling_map.get_edges()) // 2) == (156, 176)
    assert (nighthawk.num_qubits, len(nighthawk.coupling_map.get_edges()) // 2) == (120, 218)


def test_routed_costs_rank_before_logical_costs():
    logical = {"is_valid": True, "cz_count": 3, "two_qubit_depth": 3, "depth": 5, "one_qubit_gate_count": 4,
               "decoder_cz_count": 0}
    few_swaps = {**logical, "cz_count": 4, "routed_two_qubit_count": 4, "routed_two_qubit_depth": 4,
                 "routed_depth": 9, "routed_success_estimate": 0.5}
    many_swaps = {**logical, "routed_two_qubit_count": 12, "routed_two_qubit_depth": 9, "routed_depth": 2000,
                  "routed_success_estimate": 0.9}
    assert score_metrics(few_swaps) > score_metrics(many_swaps) > 1
    assert score_metrics(logical) > score_metrics({**logical, "cz_count": 4})
    # The success estimate ranks after the routed gate count and depth, before the logical CZ count.
    likely = {**few_swaps, "cz_count": 9, "routed_success_estimate": 0.6}
    assert rank_key(likely) > rank_key(few_swaps) and score_metrics(likely) > score_metrics(few_swaps)
    assert rank_key({**few_swaps, "routed_two_qubit_depth": 3, "routed_success_estimate": 0.1}) > rank_key(likely)
    assert rank_key({**few_swaps, "decoder_cz_count": 0}) > rank_key({**few_swaps, "decoder_cz_count": 2})
    with pytest.raises(ValueError, match="success estimate"):
        rank_key({**few_swaps, "routed_success_estimate": float("nan")})


def test_cost_objective_ranks_the_success_estimate_first():
    logical = {"is_valid": True, "cz_count": 3, "two_qubit_depth": 3, "depth": 5, "one_qubit_gate_count": 4,
               "decoder_cz_count": 0}
    few_swaps = {**logical, "routed_two_qubit_count": 4, "routed_two_qubit_depth": 4, "routed_depth": 9,
                 "routed_success_estimate": 0.5}
    likely = {**few_swaps, "cz_count": 9, "routed_two_qubit_count": 12, "routed_success_estimate": 0.9}
    assert rank_key(few_swaps) > rank_key(likely)
    assert rank_key(likely, "cost") > rank_key(few_swaps, "cost")
    assert score_metrics(likely, "cost") > score_metrics(few_swaps, "cost") > 1
    # Equal success estimates (to 1e-4): the routed gate count decides.
    assert rank_key({**likely, "routed_two_qubit_count": 5}, "cost") > rank_key(likely, "cost")
    # Without error data, the cost objective ranks the logical gate counts.
    assert rank_key(logical, "cost") == rank_key(logical)


def test_routed_costs_take_the_best_seed_and_route_reproduces_it():
    circuit = QuantumCircuit(6)
    for pair in ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (2, 4), (3, 5), (1, 5)):
        circuit.h(pair[0])
        circuit.cz(*pair)
    costs = architectures.routed_costs(circuit, "ibm_heron_r2", seeds=4)
    first = architectures.routed_costs(circuit, "ibm_heron_r2", seeds=1)
    assert costs["routed_seed"] in range(4) and first["routed_seed"] == 0
    assert costs["routed_two_qubit_count"] <= first["routed_two_qubit_count"]
    routed = architectures.route(circuit, "ibm_heron_r2", costs["routed_seed"])
    assert sum(item.operation.num_qubits == 2 for item in routed.data) == costs["routed_two_qubit_count"]
    likely = architectures.routed_costs(circuit, "ibm_heron_r2", seeds=4, objective="cost")
    assert likely["routed_success_estimate"] >= costs["routed_success_estimate"]


def test_assessment_adds_routed_costs(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text(skeleton_program(4, TWO_QUTRIT_FREE), encoding="utf-8")
    options = dict(state="two_qutrit", encoding=canonical_encoding(3), representation="skeleton",
                   encoding_mode="free", restarts=8, timeout_seconds=60, cpu_seconds=60)
    result = assess(path, **options, architecture="iqm_garnet")
    assert result.metrics["is_valid"] and result.metrics["routed_two_qubit_count"] >= 3
    with pytest.raises(ValueError, match="Unknown architecture"):
        assess(path, **options, architecture="ibm_eagle")


def test_batch_runs_one_campaign_per_architecture(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "_check_baseline", lambda *args: None)

    def run(baseline, output, **kwargs):
        calls.append((output.name, kwargs["architecture"]))
        return {"saved": {"is_valid": True, "cz_count": 3, "two_qubit_depth": 3, "depth": 4,
                          "one_qubit_gate_count": 5, "F": 1.0, "L": 0.0},
                "best": {"routed_two_qubit_count": 7, "cz_count": 3}}

    monkeypatch.setattr(harness, "run_campaign", run)
    output = tmp_path / "batch"
    assert harness.run_batch(output, {"ghz3": tmp_path}, seeds=(1,), architectures=("iqm", "iqm_garnet")) == 0
    assert calls == [("ghz3_seed_1_iqm_garnet", "iqm_garnet"), ("ghz3_seed_1_iqm_emerald", "iqm_emerald"),
                     ("ghz3_seed_1_iqm_sirius", "iqm_sirius")]
    report = json.loads((output / "batch.json").read_text())
    assert report["architectures"] == ["iqm_garnet", "iqm_emerald", "iqm_sirius"]
    assert [row["routed_two_qubit_count"] for row in report["runs"]] == [7, 7, 7]
    with pytest.raises(ValueError, match="Unknown architecture"):
        harness.run_batch(tmp_path / "other", {"ghz3": tmp_path}, architectures=("bad",))
    assert not (tmp_path / "other").exists()


def test_cli_selects_architectures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "run_batch", lambda *args, **kwargs: calls.append(kwargs) or 0)
    output = tmp_path / "out"
    assert harness.main(["run", "--output", str(output), "--state", "ghz3",
                         "--architecture", "ibm", "--architecture", "iqm_garnet"]) == 0
    assert calls[-1]["architectures"] == ("ibm", "iqm_garnet")


def test_discovery_passes_the_processor_and_ranks_by_routed_costs(tmp_path, monkeypatch):
    from test_evolution_oe_run import _baseline

    from qudits_on_qubits.evolution import discovery
    from qudits_on_qubits.evolution.oe_candidates import load_skeleton

    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    hub = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]]
    answer = "```json\n" + json.dumps([{"name": "few", "idea": "x", "cz": hub},
                                       {"name": "many", "idea": "y", "cz": hub + [[2, 4]]}]) + "\n```"
    monkeypatch.setattr(discovery, "_ask_model", lambda *_: answer)
    seen = []

    def run(baseline, output, **kwargs):
        seed = load_skeleton(kwargs["initial_program"], expected_qubits=6)
        seen.append(kwargs["architecture"])
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(6, seed))
        cz = len(seed)
        # More logical CZ but fewer gates after routing: the routed count must decide.
        return {"saved": {"is_valid": True, "cz_count": cz, "two_qubit_depth": cz, "depth": 2 * cz,
                          "one_qubit_gate_count": 3 * cz, "F": 1.0, "L": 0.0},
                "best": {"routed_two_qubit_count": 40 - 5 * cz, "routed_two_qubit_depth": 10, "routed_depth": 20,
                         "routed_success_estimate": 0.5}}

    monkeypatch.setattr(discovery, "run_campaign", run)
    report = discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                     families=2, model_config=model, architecture="iqm_garnet")
    assert seen == ["iqm_garnet", "iqm_garnet"] and report["architecture"] == "iqm_garnet"
    assert report["best"]["family"] == "1_many" and report["best"]["routed_two_qubit_count"] == 10
    with pytest.raises(ValueError, match="Unknown architecture"):
        discovery.run_discovery(baseline, tmp_path / "bad", architecture="ibm_eagle")
    assert not (tmp_path / "bad").exists()
