"""Check the CZ-skeleton fit, free encodings, skeleton candidates and their evaluator."""

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import qpy, transpile
from qiskit.quantum_info import Statevector

from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_candidates import load_skeleton, skeleton_program
from qudits_on_qubits.evolution.oe_evaluator import assess
from qudits_on_qubits.evolution.skeleton_fit import (
    MAX_SKELETON_CZ, _Ansatz, _circuits, cut_requirements, fit_skeleton, skeleton_of,
)
from qudits_on_qubits.evolution.state_workloads import build_workload, evaluate_workload, layout, logical_state

ROOT = Path(__file__).resolve().parents[1]
GHZ3_FREE = ((0, 1), (0, 2), (1, 3), (0, 4), (1, 5))
TWO_QUTRIT_FREE = ((0, 2), (0, 3), (1, 3))


def _write(tmp_path, text):
    path = tmp_path / "candidate.py"
    path.write_text(text, encoding="utf-8")
    return path


def _legacy_ghz3():
    with (ROOT / "src/qudits_on_qubits/quantum_circuits/CZgate3.qpy").open("rb") as stream:
        gate, = qpy.load(stream)
    gate = transpile(gate, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    return build_workload("ghz3", canonical_encoding(3), gate)


def test_jacobian_and_state_match_qiskit():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    ansatz = _Ansatz(4, pairs)
    x = np.random.default_rng(1).uniform(0, 2 * np.pi, ansatz.parameters)
    full = ansatz.run(x, jacobian=True)
    circuit, _ = _circuits(4, pairs, (), x, 0.0)
    np.testing.assert_allclose(full[:, 0], Statevector.from_instruction(circuit).data, atol=1e-12)
    for j in (0, 7, ansatz.parameters - 1):
        step = np.zeros_like(x)
        step[j] = 1e-6
        numeric = (ansatz.run(x + step)[:, 0] - ansatz.run(x - step)[:, 0]) / 2e-6
        np.testing.assert_allclose(full[:, 1 + j], numeric, atol=1e-8)


def test_free_encodings_reach_ghz3_with_five_cz():
    fit = fit_skeleton("ghz3", GHZ3_FREE, restarts=8)
    assert fit.metrics["strict_valid"] and fit.metrics["cz_count"] == 5
    # The shortest decoders found need fewer CZs than the 3 per qutrit of the fit tail.
    assert fit.decoder.count_ops().get("cz", 0) == fit.metrics["decoder_cz_count"] <= 6
    assert fit.encodings.shape == (3, 4, 3)
    for encoding in fit.encodings:
        np.testing.assert_allclose(encoding.conj().T @ encoding, np.eye(3), atol=1e-12)
    # Independent checks: the stated encodings and the decoded canonical state.
    assert evaluate_workload("ghz3", fit.circuit, fit.encodings, tolerance=1e-10)["strict_valid"]
    decoded = fit.circuit.compose(fit.decoder)
    assert evaluate_workload("ghz3", decoded, canonical_encoding(3), tolerance=1e-10)["strict_valid"]
    assert set(fit.circuit.count_ops()) <= {"u", "cz"}


def test_fixed_encoding_refit_and_too_short_skeletons():
    pairs = skeleton_of(_legacy_ghz3())
    fixed = fit_skeleton("ghz3", pairs, canonical_encoding(3), restarts=4)
    assert fixed.metrics["strict_valid"] and fixed.metrics["decoder_cz_count"] == 0
    assert fixed.metrics["cz_count"] == len(pairs)
    short = fit_skeleton("two_qutrit", TWO_QUTRIT_FREE[:2], restarts=2)
    assert not short.metrics["is_valid"] and short.fit_error > 1e-3 and short.restarts == 2
    assert short.decoder is None and short.metrics["decoder_cz_count"] == 0


def test_two_qutrit_free_encoding_with_three_cz():
    fit = fit_skeleton("two_qutrit", TWO_QUTRIT_FREE, restarts=16)
    assert fit.metrics["strict_valid"] and fit.metrics["cz_count"] == 3


def test_cut_requirements_follow_the_graph_rank():
    assert cut_requirements("two_qutrit") == [((0,), 2)]
    assert dict(cut_requirements("ghz3")) == {(0,): 2, (1,): 2, (0, 1): 2}
    ame = dict(cut_requirements("ame43"))
    assert ame[(0, 1)] == ame[(0, 2)] == ame[(1, 2)] == 4 and ame[(0,)] == 2


def test_ame44_is_absolutely_maximally_entangled_and_fits_its_gf4_network():
    assert layout("ame44") == (4, 4, 2)
    tensor = logical_state("ame44").reshape((4,) * 4)
    for side in ((0,), (3,), (0, 1), (0, 2), (0, 3)):
        rest = tuple(k for k in range(4) if k not in side)
        matrix = tensor.transpose([3 - k for k in side + rest]).reshape(4 ** len(side), -1)
        assert np.allclose(matrix @ matrix.conj().T, np.eye(4 ** len(side)) / 4 ** len(side))
    cuts = dict(cut_requirements("ame44"))
    assert cuts[(0,)] == 2 and cuts[(0, 1)] == cuts[(0, 2)] == cuts[(1, 2)] == 4
    # c = a + b and d = a + w b over GF(4) as CNOTs; the free-encoding fit finds all angles.
    network = ((0, 4), (2, 4), (1, 5), (3, 5), (0, 6), (3, 6), (1, 7), (2, 7), (3, 7))
    assert fit_skeleton("ame44", network, None, restarts=4, seed=0).metrics["is_valid"]


def test_ame64_is_the_absolutely_maximally_entangled_hexacode_state():
    from itertools import combinations

    assert layout("ame64") == (6, 4, 2)
    tensor = logical_state("ame64").reshape((4,) * 6)
    for size in (1, 2, 3):
        for side in combinations(range(6), size):
            rest = tuple(k for k in range(6) if k not in side)
            matrix = tensor.transpose([5 - k for k in side + rest]).reshape(4 ** size, -1)
            assert np.allclose(matrix @ matrix.conj().T, np.eye(4 ** size) / 4 ** size)
    cuts = dict(cut_requirements("ame64"))
    assert cuts[(0,)] == 2 and cuts[(0, 1)] == 4 and cuts[(0, 1, 2)] == 6


@pytest.mark.parametrize("pairs,options", [
    ([(0, 0)], {}), ([(0, 6)], {}), ([(0, True)], {}), ([(0, 1, 2)], {}),
    ([(0, 1)] * (MAX_SKELETON_CZ + 1), {}), ([(0, 1)], {"restarts": 0}),
    ([(0, 1)], {"seed": -1}), ([(0, 1)], {"max_nfev": 0}),
])
def test_fit_rejects_invalid_input(pairs, options):
    with pytest.raises(ValueError):
        fit_skeleton("ghz3", pairs, **options)


def test_stacked_encodings_need_one_per_qutrit():
    with pytest.raises(ValueError, match="each of the 3"):
        evaluate_workload("ghz3", _legacy_ghz3(), np.stack([canonical_encoding(3)] * 2))


def test_skeleton_program_round_trip(tmp_path):
    text = skeleton_program(6, GHZ3_FREE)
    assert text.count("\n") == len(GHZ3_FREE) + 5 and "# 4" in text
    assert load_skeleton(_write(tmp_path, text), expected_qubits=6) == GHZ3_FREE


@pytest.mark.parametrize("source", [
    "CANDIDATE = {'num_qubits': 6, 'cz': [[0, 1]], 'phases': []}",
    "CANDIDATE = {'num_qubits': 8, 'cz': [[0, 1]]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': [[0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': [[0, 6]]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': [[0, 1, 2]]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': [[0, True]]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': [0, 1]}",
    "CANDIDATE = {'num_qubits': 6, 'cz': (lambda: 0)()}",
    f"CANDIDATE = {{'num_qubits': 6, 'cz': {[[0, 1]] * (MAX_SKELETON_CZ + 1)}}}",
])
def test_reject_invalid_skeletons(tmp_path, source):
    with pytest.raises(ValueError):
        load_skeleton(_write(tmp_path, source), expected_qubits=6)


def test_subprocess_assessment_checks_free_encodings(tmp_path):
    path = _write(tmp_path, skeleton_program(6, GHZ3_FREE))
    result = assess(path, state="ghz3", encoding=canonical_encoding(3), representation="skeleton",
                    encoding_mode="free", restarts=8, timeout_seconds=60, cpu_seconds=60)
    assert result.metrics["is_valid"] and result.metrics["combined_score"] > 1
    assert result.metrics["decoder_cz_count"] == result.decoder.count_ops().get("cz", 0) <= 6
    assert evaluate_workload("ghz3", result.circuit, result.encodings, tolerance=1e-10)["strict_valid"]
    with pytest.raises(ValueError, match="skeleton representation"):
        assess(path, state="ghz3", encoding=canonical_encoding(3), encoding_mode="free")


def test_openevolve_evaluate_returns_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from qudits_on_qubits.evolution.oe_evaluator import evaluate
    np.save(tmp_path / "E.npy", canonical_encoding(3), allow_pickle=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state": "ghz3", "encoding_path": str(tmp_path / "E.npy"),
                                  "representation": "skeleton", "encoding_mode": "free",
                                  "restarts": 8, "timeout_seconds": 60, "cpu_seconds": 60}))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(config))
    valid = evaluate(_write(tmp_path, skeleton_program(6, GHZ3_FREE)))
    assert valid.metrics["is_valid"] == 1.0 and valid.metrics["cz_count"] == 5.0
    assert valid.artifacts["evaluation"].startswith("VALID: cz=5")
    # Meets every cut condition, but no 4-CZ GHZ3 structure is known to work.
    invalid = evaluate(_write(tmp_path, skeleton_program(6, [(0, 2), (0, 2), (0, 4), (0, 4)])))
    assert invalid.metrics["is_valid"] == 0.0 and invalid.metrics["evaluation_failed"] == 0.0
    assert "change the structure" in invalid.artifacts["evaluation"]


def test_deadline_stops_inside_a_least_squares_call_and_keeps_the_best_point(monkeypatch):
    from qudits_on_qubits.evolution import skeleton_fit
    from qudits_on_qubits.evolution.state_workloads import encoded_target
    ansatz = _Ansatz(6, GHZ3_FREE)
    target, _ = encoded_target("ghz3", canonical_encoding(3))
    x0 = np.random.default_rng(3).uniform(0, 2 * np.pi, ansatz.parameters)
    start_error = skeleton_fit._error(ansatz, target, x0)[0]
    clock = iter(range(1000))
    # The clock passes the deadline at its sixth reading: the fit stops in the middle of the solver.
    monkeypatch.setattr(skeleton_fit, "perf_counter", lambda: 1e9 if next(clock) >= 5 else 0.0)
    x, error, _, nfev = skeleton_fit._fit(ansatz, target, x0, 10_000, deadline=1.0)
    assert 0 < nfev < 10_000
    assert np.isfinite(error) and error <= start_error
    assert np.isclose(error, skeleton_fit._error(ansatz, target, x)[0])


def test_time_budget_stops_restarts_and_is_validated():
    fit = fit_skeleton("two_qutrit", TWO_QUTRIT_FREE[:2], restarts=64, time_budget=1e-9)
    assert fit.restarts == 1 and not fit.metrics["is_valid"]
    with pytest.raises(ValueError, match="time_budget"):
        fit_skeleton("two_qutrit", TWO_QUTRIT_FREE, time_budget=0)


def test_mutator_reads_the_current_candidate_and_returns_a_valid_skeleton(tmp_path):
    import asyncio
    import random
    from types import SimpleNamespace
    from qudits_on_qubits.evolution.oe_mock import SkeletonMutator, mutate_skeleton
    rng = random.Random(3)
    for _ in range(200):
        pairs = mutate_skeleton(6, GHZ3_FREE, rng)
        assert all(a != b and 0 <= a < 6 and 0 <= b < 6 for a, b in pairs)
        assert abs(len(pairs) - len(GHZ3_FREE)) <= 1
    model = SkeletonMutator(SimpleNamespace(name="m", random_seed=1))
    prompt = "# Current candidate\n```python\n" + skeleton_program(6, GHZ3_FREE) + "```\n"
    reply = asyncio.run(model.generate_with_context("", [{"role": "user", "content": prompt}]))
    code = reply.removeprefix("```python\n").removesuffix("```")
    assert load_skeleton(_write(tmp_path, code), expected_qubits=6) != GHZ3_FREE
    with pytest.raises(ValueError, match="no current candidate"):
        asyncio.run(model.generate_with_context("", [{"role": "user", "content": "nothing"}]))


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
@pytest.mark.parametrize("mode", ["fixed", "free"])
def test_skeleton_prompt_formats_with_openevolve_fields(state, mode):
    from qudits_on_qubits.evolution.oe_run import _skeleton_template
    from qudits_on_qubits.evolution.oe_mock import _CURRENT
    fields = ("metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
              "evolution_history", "current_program", "language", "artifacts")
    values = {name: f"<{name}>" for name in fields} | {"language": "python"}
    text = _skeleton_template(state, mode).format(**values)
    assert "<artifacts>" in text and "Schmidt rank" in text and "{" not in text
    assert _CURRENT.search(text).group(1) == "<current_program>\n"


def test_decoders_are_short_and_exact():
    from qiskit.quantum_info import Operator
    from scipy.stats import unitary_group
    from qudits_on_qubits.evolution.decoders import block_decoder, decoder_circuit
    canonical = canonical_encoding(3)
    assert block_decoder(canonical).count_ops().get("cz", 0) == 0
    random_encoding = unitary_group.rvs(4, random_state=5)[:, :3]
    for encoding in (random_encoding, unitary_group.rvs(4, random_state=6)[:, :3]):
        decoder = block_decoder(encoding)
        mapped = Operator(decoder).data @ encoding
        phase = np.trace(canonical.conj().T @ mapped)
        np.testing.assert_allclose(mapped, phase / abs(phase) * canonical, atol=1e-10)
        assert decoder.count_ops().get("cz", 0) <= 3
    full = decoder_circuit(np.stack([canonical, random_encoding]))
    assert full.num_qubits == 4 and set(full.count_ops()) <= {"u", "cz"}
    with pytest.raises(ValueError):
        block_decoder(np.ones((4, 3)))


def test_decoder_near_a_cheaper_class_uses_an_exact_fallback():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qudits_on_qubits.evolution.decoders import _maps_to_canonical, block_decoder
    # An exact two-CZ circuit only 1e-7 from a local gate. Qiskit's synthesis snaps it.
    fallback = QuantumCircuit(2)
    fallback.u(0.3, 1.1, -0.4, 0)
    fallback.u(2.0, 0.2, 0.9, 1)
    fallback.cz(0, 1)
    fallback.u(1e-7, -np.pi / 2, np.pi / 2, 0)
    fallback.cz(0, 1)
    encoding = Operator(fallback).data.conj().T @ canonical_encoding(3)
    assert _maps_to_canonical(fallback, encoding)
    assert _maps_to_canonical(block_decoder(encoding, fallback), encoding)
    try:
        assert _maps_to_canonical(block_decoder(encoding), encoding)
    except ValueError as exc:
        assert "No exact decoder" in str(exc)


def test_total_objective_counts_the_decoder():
    from qudits_on_qubits.evolution.oe_evaluator import score_metrics
    def row(cz, decoder):
        return {"is_valid": True, "cz_count": cz, "decoder_cz_count": decoder, "two_qubit_depth": 3,
                "depth": 9, "one_qubit_gate_count": 12, "F": 1.0}
    assert score_metrics(row(5, 6)) > score_metrics(row(6, 0))
    assert score_metrics(row(5, 6), "total") < score_metrics(row(6, 0), "total")
    with pytest.raises(ValueError, match="objective"):
        score_metrics(row(5, 6), "decoder")
    from qudits_on_qubits.evolution.oe_run import _skeleton_template
    assert "decoder to the canonical encoding" in _skeleton_template("ghz3", "free", "total")
    cost = _skeleton_template("ghz3", "free", "cost", architecture="iqm_garnet")
    assert "a larger routed_success_estimate" in cost and "larger routed success estimate" in cost
    assert "Among valid candidates, fewer CZ gates" in _skeleton_template("ghz3", "free")


def test_cut_violations_reject_skeletons_before_the_fit(tmp_path):
    from qudits_on_qubits.evolution.skeleton_fit import cut_violations
    assert cut_violations("ghz3", GHZ3_FREE) == []
    assert cut_violations("two_qutrit", [(0, 2), (0, 1)]) == [((0,), 2, 1)]
    ame = cut_violations("ame43", [(0, 2), (2, 4), (4, 6), (6, 0)])
    assert ((0, 1), 4, 2) in ame
    path = _write(tmp_path, skeleton_program(4, [(0, 2), (0, 1)]))
    with pytest.raises(ValueError, match="needs at least 2"):
        assess(path, state="two_qutrit", encoding=canonical_encoding(3), representation="skeleton")


def test_prompt_notes_are_bounded_and_escaped(tmp_path):
    from qudits_on_qubits.evolution.oe_run import MAX_NOTES_BYTES, _skeleton_template, read_prompt_notes
    notes = tmp_path / "notes.txt"
    notes.write_text("Do not delete single CZs from {the 19-CZ skeleton}.\n", encoding="utf-8")
    text = read_prompt_notes(notes)
    fields = ("metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
              "evolution_history", "current_program", "language", "artifacts")
    prompt = _skeleton_template("ame43", "free", notes=text).format(**dict.fromkeys(fields, "x"))
    assert "# Notes from earlier searches\nDo not delete single CZs from {the 19-CZ skeleton}." in prompt
    assert read_prompt_notes(None) == ""
    notes.write_bytes(b"x" * (MAX_NOTES_BYTES + 1))
    with pytest.raises(ValueError, match="at most"):
        read_prompt_notes(notes)
    notes.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        read_prompt_notes(notes)


def test_skeleton_prompt_shows_the_known_structure_and_lets_invalid_candidates_grow():
    from qudits_on_qubits.evolution.oe_run import _skeleton_template
    fields = ("metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
              "evolution_history", "current_program", "language", "artifacts")
    known = ((0, 1), (0, 2), (1, 3))
    text = _skeleton_template("ghz3", "free", known=known).format(**dict.fromkeys(fields, "x"))
    assert "A known valid structure with 3 CZ: [[0, 1], [0, 2], [1, 3]]" in text
    assert "If the current candidate is INVALID, make it valid first" in text and "Add CZ gates" in text
    assert "A known valid structure" not in _skeleton_template("ghz3", "free")
