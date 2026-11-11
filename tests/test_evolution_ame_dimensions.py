"""Check the AME targets with three qubits per party."""

from itertools import combinations
import json

import numpy as np
import pytest

from qudits_on_qubits.evolution import discovery, harness, oe_run, state_workloads
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_candidates import load_candidate, skeleton_program
from qudits_on_qubits.evolution.skeleton_fit import (
    MAX_SKELETON_CZ, cut_requirements, cut_violations, fit_skeleton, known_angles, skeleton_of,
)


TARGETS = (("ame45", 5), ("ame47", 7), ("ame48", 8))


@pytest.mark.parametrize("state,d", TARGETS)
def test_all_two_party_reductions_are_maximally_mixed(state, d):
    assert state_workloads.layout(state) == (4, d, 3)
    vector = state_workloads.logical_state(state)
    assert vector.shape == (d**4,)
    assert not vector.flags.writeable
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-14)
    for pair in combinations(range(4), 2):
        axes = [3 - k for k in pair] + [3 - k for k in range(4) if k not in pair]
        matrix = vector.reshape((d,) * 4).transpose(axes).reshape(d**2, d**2)
        np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(d**2) / d**2, atol=1e-14)
    for side, needed in cut_requirements(state):
        assert needed == int(np.ceil(min(len(side), 4 - len(side)) * np.log2(d)))


@pytest.mark.parametrize("state,d", TARGETS)
def test_canonical_target_uses_the_required_levels_and_qubit_order(state, d):
    logical = state_workloads.logical_state(state)
    physical, encodings = state_workloads.encoded_target(state, canonical_encoding(d))
    assert physical.shape == (4096,)
    assert encodings.shape == (4, 8, d)
    valid = np.ones(4096, dtype=bool)
    indices = np.arange(4096)
    logical_indices = np.zeros(4096, dtype=int)
    for party in range(4):
        level = (indices >> (3 * party)) & 7
        valid &= level < d
        logical_indices += level * d**party
    assert np.count_nonzero(valid) == d**4
    np.testing.assert_allclose(physical[valid], logical[logical_indices[valid]], atol=1e-14)
    np.testing.assert_array_equal(physical[~valid], 0)


@pytest.mark.parametrize("state,d", TARGETS)
def test_reference_and_both_encoding_modes_pass_the_strict_limit(tmp_path, state, d):
    method = oe_run.method_for(state, "estar_direct")
    assert method == "canonical_direct"
    circuit, encoding, _, metrics = oe_run._source(tmp_path / state, state, method)
    assert circuit.num_qubits == 12
    assert metrics["strict_valid"]
    assert 2 <= metrics["cz_count"] <= MAX_SKELETON_CZ
    assert not cut_violations(state, skeleton_of(circuit))
    np.testing.assert_array_equal(encoding, canonical_encoding(d))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("CANDIDATE = " + repr(oe_run.candidate_from_circuit(circuit)))
    load_candidate(candidate, expected_qubits=12)
    for mode in ("fixed", "free"):
        pairs, angles = known_angles(circuit, np.asarray([encoding] * 4) if mode == "free" else None)
        fit = fit_skeleton(state, pairs, encoding=encoding if mode == "fixed" else None,
                           x0=angles, restarts=1, max_nfev=1, time_budget=15)
        assert fit.metrics["strict_valid"]
        assert state_workloads.evaluate_workload(
            state, fit.circuit.compose(fit.decoder), encoding, tolerance=1e-10)["strict_valid"]
    summary = json.loads((tmp_path / state / "summary.json").read_text())
    assert summary["results"][0]["method"] == "canonical_direct"


@pytest.mark.parametrize("state,d", TARGETS)
def test_targets_are_available_in_the_cli_and_prompts(state, d, monkeypatch):
    monkeypatch.delenv("QOQ_VECTOR_ONLY", raising=False)
    assert state in dict(harness.STATE_CHOICES)
    args = harness._parser().parse_args(["run", "--state", state, "--output", "unused"])
    assert getattr(args, f"baseline_{state}") == harness.BASELINES[state]
    for mode in ("fixed", "free"):
        text, units = state_workloads.describe(state, mode)
        assert f"AME(4,{d})" in text
        assert "12 qubits" in text
        assert units == "parties"
        assert "Necessary conditions" in oe_run._skeleton_template(state, mode)


def test_discovery_accepts_a_new_dimension_without_an_existing_baseline(tmp_path, monkeypatch):
    def campaign(baseline, output, **options):
        circuit, _, _, metrics = oe_run._source(baseline, options["state"], options["method"])
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(12, skeleton_of(circuit)))
        return {"saved": metrics}

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    report = discovery.run_discovery(tmp_path / "baseline", tmp_path / "run", state="ame48",
                                     encoding_mode="free", families=1, polish_seconds=0)
    assert report["status"] == "completed"
    assert report["method"] == "canonical_direct"
