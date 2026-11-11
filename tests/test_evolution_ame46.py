"""Check AME(4,6) support with a synthetic six-level state: import, cuts, decoders and warm starts.

The synthetic state has two qudit Bell pairs, A-C and B-D,
in levels 0 and 1, so a two-CZ circuit prepares it in a random local basis.
"""

import hashlib
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary

from qudits_on_qubits.evolution import ame46, skeleton_fit, state_workloads
from qudits_on_qubits.evolution.decoders import _maps_to_canonical, block_decoder
from qudits_on_qubits.evolution.discovery import offline_families, select_families
from qudits_on_qubits.evolution.oe_candidates import skeleton_program
from qudits_on_qubits.evolution.oe_evaluator import WARM_STARTS, _warm_start, assess
from qudits_on_qubits.evolution.oe_run import _source, _warm_pool, method_for

STATE = "HZ96"
PAIRS = [(0, 6), (3, 9)]
OPS = [["u", 0, np.pi / 2, 0.0, np.pi], ["cx", 0, 6], ["u", 3, np.pi / 2, 0.0, np.pi], ["cx", 3, 9]]


def _write_inputs(directory, seed=0):
    tensor = np.zeros((6,) * 4, dtype=complex)
    for a in range(2):
        for b in range(2):
            tensor[a, b, a, b] = 0.5
    unitaries = [random_unitary(6, seed=seed + k).data for k in range(4)]
    source = np.einsum("ia,jb,kc,ld,ijkl->abcd", *(u.conj() for u in unitaries), tensor)
    np.savez(directory / "targets.npz", **{STATE: source})
    circuits = directory / "circuits"
    circuits.mkdir()
    basis = {"format": "ame46-local-basis-v1", "direction": "source_to_output", "party_order": list("ABCD"),
             "source_sha256": hashlib.sha256(source.astype("<c16").tobytes()).hexdigest(),
             "unitaries": np.stack([np.stack([u.real, u.imag], -1) for u in unitaries]).tolist()}
    data = {"state_id": STATE, "local_basis": basis,
            "circuit": {"format": "ame46-c18-ops-v1", "num_qubits": 12, "global_phase": 0.0, "ops": OPS}}
    (circuits / f"{STATE}.json").write_text(json.dumps(data))
    return directory / "targets.npz", circuits


@pytest.fixture
def imported(tmp_path, monkeypatch):
    targets, circuits = _write_inputs(tmp_path)
    rows = ame46.import_ame46(targets, circuits, tmp_path / "ame46")
    monkeypatch.setenv(state_workloads.AME46_TARGETS_VARIABLE, str(tmp_path / "ame46" / "targets.npz"))
    return tmp_path / "ame46" / "baseline", rows


def test_import_checks_the_circuit_and_writes_a_baseline(imported):
    baseline, rows = imported
    assert [(r["workload"], r["method"], r["is_valid"], r["cz_count"]) for r in rows] == [(STATE, "adapted_direct", True, 2)]
    circuit, encoding, _, metrics = _source(baseline, STATE, method_for(STATE, "estar_direct"))
    assert metrics["is_valid"] and encoding.shape == (4, 8, 6) and circuit.num_qubits == 12
    assert state_workloads.layout(STATE) == (4, 6, 3)


def test_import_rejects_a_basis_of_another_tensor_and_keeps_existing_data(tmp_path):
    targets, circuits = _write_inputs(tmp_path)
    path = circuits / f"{STATE}.json"
    data = json.loads(path.read_text())
    data["local_basis"]["source_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different source"):
        ame46.import_ame46(targets, circuits, tmp_path / "out")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "targets.npz").write_bytes(b"old")
    with pytest.raises(ValueError, match="Remove it first"):
        ame46.import_ame46(targets, circuits, tmp_path / "out")
    assert (tmp_path / "out" / "targets.npz").read_bytes() == b"old"


def test_import_reads_several_tensor_files_and_rejects_a_state_in_two(tmp_path):
    targets, circuits = _write_inputs(tmp_path)
    other = tmp_path / "other.npz"
    np.savez(other, Casas36=np.zeros((6,) * 4))
    rows = ame46.import_ame46([other, targets], circuits, tmp_path / "out")
    assert [r["workload"] for r in rows] == [STATE]
    with np.load(tmp_path / "out" / "targets.npz") as data:
        assert sorted(data.files) == ["Casas36", STATE]
    with pytest.raises(ValueError, match="more than one"):
        ame46.import_ame46([targets, targets], circuits, tmp_path / "again")
    assert not (tmp_path / "again").exists()


def test_a_missing_tensor_file_names_the_import_command(tmp_path, monkeypatch):
    monkeypatch.setenv(state_workloads.AME46_TARGETS_VARIABLE, str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError, match="ame46-import"):
        state_workloads.logical_state(STATE)


def test_cuts_prompt_and_offline_families_use_three_qubit_blocks(imported):
    # Bell pairs A-C and B-D: rank 2 across one party, 4 across AB|CD and AD|BC, 1 across AC|BD.
    assert dict(skeleton_fit.cut_requirements(STATE)) == {(0,): 1, (1,): 1, (0, 1): 2, (2,): 1, (0, 2): 0,
                                                          (1, 2): 2, (0, 1, 2): 1}
    assert skeleton_fit.cut_violations(STATE, [(0, 1), (3, 9)]) == [((0,), 1, 0), ((0, 1), 2, 1), ((2,), 1, 0),
                                                                    ((1, 2), 2, 1)]
    text, units = state_workloads.describe(STATE, "free")
    assert units == "parties" and 'AME(4,6) state "HZ96" (complex amplitudes)' in text
    assert "Qubits 3k, 3k+1 and 3k+2 hold party k" in text and "6-dimensional subspace of its three" in text
    accepted, _ = select_families(offline_families(STATE, PAIRS), STATE, 3)
    hub = next(f for f in accepted if f["name"].endswith("hub"))
    assert hub["cz"][:2] == [[0, 1], [1, 2]] and [0, 3] in hub["cz"] and [2, 11] in hub["cz"]


def test_three_qubit_decoders_are_exact():
    canonical = np.eye(8, 6, dtype=complex)
    assert block_decoder(canonical).count_ops().get("cz", 0) == 0
    encoding = random_unitary(8, seed=3).data[:, :6]
    assert _maps_to_canonical(block_decoder(encoding), encoding)


def test_skeleton_angles_reproduce_the_circuit():
    circuit = QuantumCircuit(3)
    rng = np.random.default_rng(1)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        circuit.u(*rng.uniform(0, 6, 3), a)
        circuit.u(*rng.uniform(0, 6, 3), a)
        circuit.cz(a, b)
    circuit.u(*rng.uniform(0, 6, 3), 1)
    pairs, x = skeleton_fit.skeleton_angles(circuit)
    state = skeleton_fit._Ansatz(3, pairs).run(x)[:, 0]
    assert pairs == ((0, 1), (1, 2), (2, 0))
    assert abs(np.vdot(Statevector(circuit).data, state)) == pytest.approx(1, abs=1e-12)


def test_warm_angles_copy_common_cz_angles_and_flip_reversed_pairs():
    width = 2
    known_pairs, angles = [(0, 1), (1, 0)], np.arange(3 * width + 12, dtype=float)
    x = skeleton_fit.warm_angles([(1, 0), (0, 1), (0, 1)], [(known_pairs, angles), ([(0, 1)], [1.0])], width)
    assert x[:6].tolist() == angles[:6].tolist()
    # Candidate CZ 0 matches known CZ 0 in the other qubit order, CZ 1 matches known CZ 1; CZ 2 is new.
    assert x[6:12].tolist() == [9, 10, 11, 6, 7, 8] and x[12:18].tolist() == [15, 16, 17, 12, 13, 14]
    assert not x[18:].any()
    assert skeleton_fit.warm_angles([(0, 1)], [([(0, 1)], [np.nan] * 12), (7, "x"), ([["a", 1]], [0.0])], width) is None


@pytest.mark.parametrize("encoding_mode", ["fixed", "free"])
def test_warm_start_from_the_baseline_reaches_the_state_without_a_fit(imported, encoding_mode):
    baseline, _ = imported
    circuit, encoding, _, _ = _source(baseline, STATE, "adapted_direct")
    free = encoding_mode == "free"
    pool = _warm_pool(STATE, encoding_mode, circuit, encoding)
    assert len(pool) == 1
    tail = 4 * len(skeleton_fit.TAIL_PAIRS[3]) if free else 0
    x0 = skeleton_fit.warm_angles(PAIRS, [(p["pairs"], p["angles"]) for p in pool], 12, tail)
    fit = skeleton_fit.fit_skeleton(STATE, PAIRS, None if free else encoding, restarts=1, x0=x0)
    assert fit.metrics["is_valid"] and fit.nfev == 0 and fit.encodings.shape == (4, 8, 6)
    if free:
        assert fit.metrics["decoder_cz_count"] == tail  # the tail is the decoder of each party
    with pytest.raises(ValueError, match="finite angles"):
        skeleton_fit.fit_skeleton(STATE, PAIRS, None if free else encoding, x0=[0.0])


def test_evaluator_takes_the_warm_start_of_the_campaign_and_returns_the_angles(imported, tmp_path):
    baseline, _ = imported
    circuit, encoding, _, _ = _source(baseline, STATE, "adapted_direct")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / WARM_STARTS).write_text(json.dumps(_warm_pool(STATE, "fixed", circuit, encoding)))
    program = campaign / "candidate.py"
    program.write_text(skeleton_program(12, PAIRS))
    options = {"state": STATE, "encoding": encoding, "representation": "skeleton", "encoding_mode": "fixed",
               "restarts": 1, "timeout_seconds": 120, "cpu_seconds": 120}
    x0 = _warm_start(program, options, campaign)
    result = assess(program, **options, x0=x0)
    assert result.metrics["is_valid"] and result.metrics["nfev"] == 0 and result.metrics["cz_count"] == 2
    assert result.angles.shape == (3 * (12 + 2 * len(PAIRS)),)


def test_campaign_rechecks_the_best_skeleton_from_its_warm_start(imported, tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from qudits_on_qubits.evolution import oe_run
    real, final = oe_run.assess, []

    def recorded(path, **options):
        if options["timeout_seconds"] == oe_run.MAX_SECONDS:
            final.append(options.get("x0"))
        return real(path, **options)

    monkeypatch.setattr(oe_run, "assess", recorded)
    baseline, _ = imported
    result = oe_run._run_campaign(baseline, tmp_path / "run", state=STATE, method="estar_direct", iterations=1,
                                  representation="skeleton", restarts=1, timeout_seconds=120, cpu_seconds=120)
    assert final and final[0] is not None
    assert result["saved"]["is_valid"] and result["saved"]["cz_count"] == 2
    assert json.loads((tmp_path / "run" / "configuration.json").read_text())["method"] == "adapted_direct"


def test_warm_angles_prefer_the_entry_of_the_same_skeleton():
    width = 2
    longer = ([(0, 1), (0, 1), (1, 0)], np.zeros(3 * width + 18))
    same = ([(0, 1), (0, 1)], np.ones(3 * width + 12))
    assert skeleton_fit.warm_angles([(0, 1), (0, 1)], [longer, same], width).tolist() == same[1].tolist()
