"""Check the four qutrit targets and their harness inputs."""

from itertools import combinations
import json

import numpy as np
import pytest
from qiskit import qpy

from qudits_on_qubits.evolution import discovery, harness, oe_run, state_workloads
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.oe_candidates import load_candidate, skeleton_program
from qudits_on_qubits.evolution.skeleton_fit import (
    MAX_SKELETON_CZ, cut_requirements, cut_violations, fit_skeleton, known_angles, skeleton_of,
)


TARGETS = (("aharonov3", 3), ("dicke4", 4), ("cluster4", 4), ("aklt4", 4))


def _reduction(vector: np.ndarray, parties: tuple[int, ...], n: int) -> np.ndarray:
    axes = [n - 1 - k for k in parties] + [n - 1 - k for k in range(n) if k not in parties]
    matrix = vector.reshape((3,) * n).transpose(axes).reshape(3 ** len(parties), -1)
    return matrix @ matrix.conj().T


@pytest.mark.parametrize("state,n", TARGETS)
def test_targets_have_the_required_dimensions_and_full_local_rank(state, n):
    assert state_workloads.layout(state) == (n, 3, 2)
    target = state_workloads.logical_state(state)
    assert target.shape == (3**n,)
    assert not target.flags.writeable
    assert np.linalg.norm(target) == pytest.approx(1, abs=1e-14)
    for party in range(n):
        assert np.linalg.matrix_rank(_reduction(target, (party,), n)) == 3
    assert len(cut_requirements(state)) == 2 ** (n - 1) - 1


def test_aharonov_target_changes_sign_under_every_exchange():
    target = state_workloads.logical_state("aharonov3").reshape((3,) * 3)
    assert np.count_nonzero(target) == 6
    for a, b in combinations(range(3), 2):
        np.testing.assert_allclose(target.swapaxes(a, b), -target, atol=1e-14)
    np.testing.assert_allclose(_reduction(target.reshape(-1), (0,), 3), np.eye(3) / 3, atol=1e-14)


def test_dicke_target_has_the_required_occupations_and_symmetry():
    target = state_workloads.logical_state("dicke4").reshape((3,) * 4)
    words = np.argwhere(np.abs(target) > 1e-14)
    assert len(words) == 12
    for word in words:
        np.testing.assert_array_equal(np.bincount(word, minlength=3), (2, 1, 1))
        assert target[tuple(word)] == pytest.approx(1 / np.sqrt(12))
    for a, b in combinations(range(4), 2):
        np.testing.assert_allclose(target.swapaxes(a, b), target, atol=1e-14)
    np.testing.assert_allclose(_reduction(target.reshape(-1), (0,), 4), np.diag([0.5, 0.25, 0.25]), atol=1e-14)


def test_cluster_target_satisfies_the_four_graph_stabilizers():
    target = state_workloads.logical_state("cluster4").reshape((3,) * 4)
    omega = np.exp(2j * np.pi / 3)
    for party in range(4):
        transformed = np.roll(target, 1, axis=3 - party)
        for neighbor in (party - 1, party + 1):
            if 0 <= neighbor < 4:
                shape = [1] * 4
                shape[3 - neighbor] = 3
                transformed = transformed * (omega ** np.arange(3)).reshape(shape)
        np.testing.assert_allclose(transformed, target, atol=1e-14)


def test_aklt_target_has_no_spin_two_component_on_any_ring_bond():
    target = state_workloads.logical_state("aklt4")
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.diag([1, 0, -1])
    interaction = sum(np.kron(spin, spin) for spin in (sx, sy, sz))
    projector = interaction @ interaction / 6 + interaction / 2 + np.eye(9) / 3
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        assert abs(np.trace(_reduction(target, (a, b), 4) @ projector)) < 1e-14
    for party in range(4):
        np.testing.assert_allclose(_reduction(target, (party,), 4), np.eye(3) / 3, atol=1e-14)


@pytest.mark.parametrize("state,n", TARGETS)
def test_new_targets_work_in_prompts_menu_and_cli(state, n, monkeypatch):
    monkeypatch.delenv("QOQ_VECTOR_ONLY", raising=False)
    assert state in dict(harness.STATE_CHOICES)
    args = harness._parser().parse_args(["run", "--state", state, "--output", "unused"])
    assert getattr(args, f"baseline_{state}") == harness.BASELINES[state]
    for mode in ("fixed", "free"):
        text, units = state_workloads.describe(state, mode)
        assert f"{2 * n} qubits" in text
        assert units == "qutrits"
        if state != "cluster4":
            assert "graph state" not in text
        assert "Necessary conditions" in oe_run._skeleton_template(state, mode)


@pytest.mark.parametrize("state,n", TARGETS)
def test_menu_accepts_new_targets_without_a_saved_baseline(tmp_path, state, n, monkeypatch):
    monkeypatch.setattr(harness, "BASELINES", {state: tmp_path / "missing"})
    selections = iter((state, "free", "preparation", "none"))
    monkeypatch.setattr(harness, "choose", lambda *args: next(selections))
    monkeypatch.setattr(harness, "pick_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(harness, "_loop_options", lambda: (["--offline-model", "mutations"], {}))
    monkeypatch.setattr(harness, "_narrator_options", lambda: [])
    monkeypatch.setattr(harness, "_search_settings", lambda *args: ["--workers", "1"])
    monkeypatch.setattr(harness, "ask_int", lambda label, default, *args: default)
    monkeypatch.setattr(harness, "ask_new_path", lambda *args: str(tmp_path / "run"))
    monkeypatch.setattr(harness, "ask", lambda *args, **kwargs: pytest.fail("The target needs no external baseline"))
    monkeypatch.setattr(harness, "review", lambda *args: True)
    args = harness._discover_menu()
    assert args[args.index("--state") + 1] == state
    assert "--baseline" not in args


@pytest.mark.parametrize("state,n", TARGETS)
def test_reference_baseline_is_valid_bounded_and_preserves_existing_files(tmp_path, state, n):
    baseline = tmp_path / state
    method = oe_run.method_for(state, "estar_direct")
    circuit, encoding, _, metrics = oe_run._source(baseline, state, method)
    assert metrics["strict_valid"]
    assert circuit.num_qubits == 2 * n
    assert 2 <= metrics["cz_count"] <= MAX_SKELETON_CZ
    assert not cut_violations(state, skeleton_of(circuit))
    np.testing.assert_array_equal(encoding, canonical_encoding(3))
    row, = json.loads((baseline / "summary.json").read_text())["results"]
    candidate = tmp_path / "candidate.py"
    candidate.write_text("CANDIDATE = " + repr(oe_run.candidate_from_circuit(circuit)))
    load_candidate(candidate, expected_qubits=2 * n)
    with (baseline / row["circuit_artifact"]).open("rb") as stream:
        saved, = qpy.load(stream)
    assert state_workloads.evaluate_workload(state, saved, encoding, tolerance=1e-10)["strict_valid"]
    before = {path.name: path.read_bytes() for path in baseline.iterdir()}
    oe_run._source(baseline, state, method)
    assert before == {path.name: path.read_bytes() for path in baseline.iterdir()}
    for mode in ("fixed", "free"):
        encodings = np.asarray([encoding] * n) if mode == "free" else None
        pairs, angles = known_angles(circuit, encodings)
        fit = fit_skeleton(state, pairs, encoding=encoding if mode == "fixed" else None,
                           x0=angles, restarts=1, max_nfev=1)
        assert fit.metrics["strict_valid"]
        decoded = state_workloads.evaluate_workload(
            state, fit.circuit.compose(fit.decoder), encoding, tolerance=1e-10)
        assert decoded["strict_valid"]
    source = baseline / row["circuit_artifact"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        oe_run._source(baseline, state, method)


def test_existing_incomplete_baseline_is_preserved(tmp_path):
    from qudits_on_qubits.evolution.state_benchmark import ensure_qutrit_baseline

    baseline = tmp_path / "existing"
    baseline.mkdir()
    keep = baseline / "keep.txt"
    keep.write_text("Keep this file.")
    ensure_qutrit_baseline("aklt4", baseline)
    assert list(baseline.iterdir()) == [keep]
    assert keep.read_text() == "Keep this file."
    with pytest.raises(ValueError, match="supported"):
        ensure_qutrit_baseline("unsupported", tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_harness_and_discovery_create_their_missing_baselines(tmp_path, monkeypatch):
    calls = []

    def campaign(baseline, output, **options):
        method = oe_run.method_for(options["state"], options["method"])
        circuit, _, _, metrics = oe_run._source(baseline, options["state"], method)
        calls.append((options["state"], method))
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(circuit.num_qubits, skeleton_of(circuit)))
        return {"saved": metrics}

    monkeypatch.setattr(harness, "run_campaign", campaign)
    assert harness.main(["run", "--state", "cluster4", "--baseline-cluster4", str(tmp_path / "batch_base"),
                         "--output", str(tmp_path / "batch")]) == 0
    monkeypatch.setattr(discovery, "run_campaign", campaign)
    report = discovery.run_discovery(tmp_path / "discovery_base", tmp_path / "discovery",
                                     state="cluster4", encoding_mode="free", families=1, polish_seconds=0)
    assert report["status"] == "completed"
    assert calls == [("cluster4", "canonical_direct"), ("cluster4", "canonical_direct")]
    for name in ("batch_base", "discovery_base"):
        assert (tmp_path / name / "manifest.json").is_file()
