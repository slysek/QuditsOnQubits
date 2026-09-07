import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qudits_on_qubits.bell_measurements.postprocessing import evaluate_reference_bell_values_from_counts
from study import (STATES, workload, encodings, exact_probabilities, outcome_scores,
                   evaluate_counts, get_reference_experiment)


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("name", ["canonical_ez", "sup013_P102_ph012", "sup023_P210_ph021", "sup123_P201_ph002"])
def test_exact_bell_and_paired_f3_probabilities(state, name):
    outputs = []
    for variant in ("standard", "optimal"):
        circuits, meta = workload(state, name, variant)
        p = np.asarray([exact_probabilities(c) for c in circuits])
        scores, accepted = outcome_scores(meta, circuits[0].num_qubits)
        assert np.sum(p*scores) == pytest.approx(get_reference_experiment(state).expected.ideal_bell_value, abs=1e-9)
        assert np.sum(p*(~accepted)) == pytest.approx(0,abs=1e-10)
        outputs.append(p)
    np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-10)


def test_encoding_scope():
    assert len(encodings()) == 217
    for e in encodings().values():
        np.testing.assert_allclose(e.conj().T@e,np.eye(3),atol=1e-12)


def test_scores_agree_with_repo_for_noisy_counts():
    for state in STATES:
        circuits,meta = workload(state,"canonical_ez","optimal")
        rng = np.random.default_rng(91)
        counts = []
        for c in circuits:
            p = .8*exact_probabilities(c)+.2/(2**c.num_qubits)
            values = rng.multinomial(2048,p/p.sum())
            counts.append({format(i,f"0{c.num_qubits}b"):int(n) for i,n in enumerate(values) if n})
        result = evaluate_counts(counts,meta,samples=100)
        reference = evaluate_reference_bell_values_from_counts(state,dict(zip(meta["setting_by_circuit_index"],counts)),meta["qutrit_bit_indices_by_setting"])
        assert result["unconditional"] == pytest.approx(reference.unconditional.real)
        assert result["conditional"] == pytest.approx(reference.conditional.real)
        assert result["invalid_fraction"] == pytest.approx(reference.leakage_rate)
        assert result["conditional_se"] > 0


def test_physical_wires_and_classical_permutation_survive_qpy(tmp_path):
    qc = QuantumCircuit(9,2)
    qc.x(7)
    qc.measure(7,0)
    qc.measure(3,1)
    path=tmp_path/"test.qpy"
    with path.open("wb") as f:
        qpy.dump(qc,f)
    with path.open("rb") as f:
        restored=qpy.load(f)[0]
    np.testing.assert_allclose(exact_probabilities(restored),[0,1,0,0])


def test_atomic_json_retries_transient_windows_lock(tmp_path, monkeypatch):
    from pathlib import Path
    import study
    original = Path.replace
    attempts = []
    def locked_once(self, target):
        attempts.append(self)
        if len(attempts) == 1:
            raise PermissionError("transient destination lock")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(study.time, "sleep", lambda _: None)
    path = tmp_path / "receipt.json"
    study.save_json(path, {"job_id": "existing"})
    assert study.json.loads(path.read_text()) == {"job_id": "existing"}
    assert len(attempts) == 2
    assert list(tmp_path.glob("*.tmp")) == []
