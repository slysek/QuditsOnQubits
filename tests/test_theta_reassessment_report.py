from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, state_fidelity

from qudits_on_qubits.benchmarks.theta_continuation import reassessment_report as report
from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding
from qudits_on_qubits.benchmarks.direct_basis.circuits import build_direct_basis_graph_state_circuit


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def store(tmp_path):
    config = ThetaBenchmarkConfig(theta_points=2, cz3_tolerance=5e-4, states=("two_qutrit",))
    manifest = {"benchmark": "theta_threshold_reassessment_v1", "config": config.to_dict(),
                "source_fingerprint": "original-fingerprint", "source_cz3_tolerance": 1e-5,
                "provenance": {"versions": {"qiskit": "2.1.fixture"}}}
    manifest["fingerprint"] = fingerprint(manifest)
    store = RunStore.create(tmp_path / "run", manifest)
    store.write_bundle("baseline", metadata={"baseline_n_cz": 6})
    for index, theta in enumerate(config.grid()):
        f3 = QuantumCircuit(2)
        f3.u(0.23456789012345678, 0.12, -0.2, 0)
        f3.cz(0, 1)
        f3.u(-0.1, 0.4, 0.3, 1)
        f3.cz(0, 1)
        f3.global_phase = 0.12345678901234567
        control = f3.copy()
        control.cz(0, 1)
        cz3 = QuantumCircuit(4)
        cz3.u(0.2, 0.3, 0.4, 0)
        cz3.cz(0, 1)
        cz3.cz(2, 3)
        cz3.u(0.4, 0.3, 0.2, 1)
        cz3.cz(1, 2)
        prep = QuantumCircuit(2)
        if index:
            prep.u(0.3, 0, 0, 0)
            prep.cz(0, 1)
        metadata = {"index": index, "theta": theta, "theta_over_pi": theta / np.pi,
                    "correct": True, "status": "above_baseline", "f3_alpha": float(index),
                    "selected_method": "baseline" if not index else "bqskit",
                    "f3_metrics": {"E_norm": 1e-15, "L_norm": 2e-15},
                    "f3_alpha_zero_metrics": {"E_norm": 3e-15, "L_norm": 4e-15},
                    "cz3_metrics": {"E_norm": 0.00031252833076717365, "L_norm": 0.000012345678901234567,
                                    "compilation_method": "exact_u3_to_u" if index else "transpile"},
                    "f3_n_cz": 999, "cz3_n_cz": 999,
                    "reassessment": {"origin": "reconstructed" if index else "baseline",
                                     "source_status": "failed" if index else "ok", "source_correct": index == 0,
                                     "attempts": [{"id": f"attempt-{index}", "historical_valid": False, "valid_at_new_tolerance": False, "E_norm": 0.005206495797314588, "L_norm": 0.006}],
                                     "original_metrics_match": True}}
        complete = store.write_bundle(f"points/{index:05d}", metadata=metadata,
                                      circuits={"f3_optimal.qpy": f3, "f3_alpha_zero.qpy": control,
                                                "cz3_selected.qpy": cz3},
                                      documents={"reassessment.json": {"attempts": []}})
        reference = Statevector.from_instruction(build_direct_basis_graph_state_circuit("two_qutrit", theta_embedding(theta)))
        full = QuantumCircuit(4)
        full.append(StatePreparation(reference.data), range(4))
        full = transpile(full, basis_gates=["u", "cz"], optimization_level=0, approximation_degree=1.0, seed_transpiler=0)
        full.u(1e-6, 0, 0, 0)
        fidelity = float(state_fidelity(reference, Statevector.from_instruction(full)))
        row = {"index": index, "theta": theta, "state_name": "two_qutrit", "success": True,
               "fidelity": round(fidelity, 12), "gate_point_fingerprint": fingerprint(complete),
               "preparation_n_cz": 999, "two_qubit_gate_count": 999, "best_depth": 999}
        store.write_bundle(f"circuits/two_qutrit/{index:05d}", metadata=row,
                           circuits={"graph_state.qpy": full, "graph_state_transpiled.qpy": full,
                                     "zero_preparation.qpy": prep}, documents={"row.json": row})
    return store


def test_costs_compute_parallel_cz_depth_and_empty_circuit():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(2, 3)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.u(0.2, 0.3, 0.4, 0)
    circuit.cz(0, 2)
    assert report.circuit_costs(circuit) == {"n_cz": 3, "n_1q": 2, "depth": 4, "cz_depth": 2, "size": 5}
    assert report.circuit_costs(QuantumCircuit(2)) == {"n_cz": 0, "n_1q": 0, "depth": 0, "cz_depth": 0, "size": 0}


@pytest.mark.parametrize("kind", ["h", "measure", "parameter"])
def test_costs_reject_unexpected_or_unbound_instruction(kind):
    circuit = QuantumCircuit(1, 1) if kind == "measure" else QuantumCircuit(1)
    if kind == "parameter":
        from qiskit.circuit import Parameter
        circuit.u(Parameter("a"), 0, 0, 0)
    elif kind == "measure":
        circuit.measure(0, 0)
    else:
        circuit.h(0)
    with pytest.raises(ValueError):
        report.circuit_costs(circuit)


def test_report_uses_actual_qpy_costs_and_full_precision_provenance(store):
    output = report.generate_reassessment_report(store.root)
    assert output == store.root / "report.md"
    text = output.read_text(encoding="utf-8")
    assert "0.0005" in text and "1e-05" in text
    assert "original-fingerprint" in text
    assert "exact_u3_to_u" in text and "reconstructed" in text
    assert "previous threshold: **1/2**" in text
    assert "Historical continuation attempts: 2" in text
    assert "meeting the new threshold: 0" in text
    assert float(_csv(store.root / "attempts.csv")[0]["E_norm"]) == 0.005206495797314588
    assert "does not imply" in text and "continuation" in text
    rows = _csv(store.root / "points.csv")
    assert len(rows) == 2
    assert rows[0]["f3_n_cz"] == "2"
    assert rows[0]["cz3_n_cz"] == "3"
    assert rows[0]["cz3_cz_depth"] == "2"
    assert float(rows[0]["cz3_E_norm"]) == 0.00031252833076717365
    assert rows[1]["cz3_origin"] == "reconstructed"
    full = _csv(store.root / "full_circuits.csv")
    assert [row["preparation_n_cz"] for row in full] == ["0", "2"]
    assert [row["f3_blocks_n_cz"] for row in full] == ["4", "4"]
    assert [row["cz3_blocks_n_cz"] for row in full] == ["3", "3"]
    for row in full:
        bundle = store.read_bundle(f"circuits/two_qutrit/{int(row['index']):05d}")
        actual = bundle["circuits"]["graph_state_transpiled.qpy"]
        expected_fidelity = float(state_fidelity(
            Statevector.from_instruction(build_direct_basis_graph_state_circuit("two_qutrit", theta_embedding(float(row["theta"])))),
            Statevector.from_instruction(actual)))
        assert int(row["n_cz"]) == actual.count_ops()["cz"]
        assert float(row["fidelity"]) == expected_fidelity
        assert 0 < abs(float(row["fidelity"]) - float(row["saved_fidelity"])) < 1e-12
        assert f"{expected_fidelity:.17g}" in text
    parameters = _csv(store.root / "gate_parameters.csv")
    first = next(row for row in parameters if row["artifact_id"] == "f3_00000" and row["operation"] == "u")
    assert float(first["theta"]) == 0.23456789012345678
    assert float(first["global_phase"]) == 0.12345678901234567
    assert (store.root / "figures/reassessment_metrics.png").stat().st_size > 100
    assert (store.root / "figures/reassessment_metrics.pdf").stat().st_size > 100
    assert (store.root / "diagrams/f3_00000.png").stat().st_size > 100
    assert (store.root / "diagrams/cz3_00001.pdf").stat().st_size > 100
    assert (store.root / "diagrams/two_qutrit_00000.txt").stat().st_size > 100
    gallery = (store.root / "gallery.html").read_text(encoding="utf-8")
    links = re.findall(r'(?:href|src)="([^"]+)"', gallery)
    assert links and all((store.root / link.split("#")[0]).is_file() for link in links if not link.startswith("#"))
    for filename in ("points.csv", "full_circuits.csv", "gate_parameters.csv", "gallery.html", "report.md"):
        assert "\ufffd" not in (store.root / filename).read_text(encoding="utf-8")


def test_report_checks_all_source_hashes_before_publishing(store):
    (store.root / "circuits/two_qutrit/00001/metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        report.generate_reassessment_report(store.root)
    assert not (store.root / "report.md").exists()
    assert not (store.root / "points.csv").exists()


def test_report_rejects_changed_manifest_and_source_campaign(store):
    path = store.root / "manifest.json"
    manifest = store.manifest
    manifest["config"]["cz3_tolerance"] = 0.1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        report.generate_reassessment_report(store.root)
    manifest["benchmark"] = "theta_continuation_v1"
    manifest.pop("fingerprint")
    manifest["fingerprint"] = fingerprint(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reassessment"):
        report.generate_reassessment_report(store.root)


def _replace_metadata(store, relative, update):
    """Build a hash-valid fixture with semantically invalid metadata."""
    import hashlib
    root = store.root / relative
    metadata = json.loads((root / "metadata.json").read_text())
    update(metadata)
    complete = json.loads((root / "complete.json").read_text())
    for name in ("metadata.json", "row.json"):
        if name not in complete["files"]:
            continue
        content = json.dumps(metadata).encode()
        (root / name).write_bytes(content)
        complete["files"][name] = hashlib.sha256(content).hexdigest()
    (root / "complete.json").write_text(json.dumps(complete))


@pytest.mark.parametrize("problem", ["missing_norm", "negative_norm", "above_threshold", "index"])
def test_report_rejects_semantically_invalid_point_before_outputs(store, problem):
    def update(row):
        if problem == "index":
            row["index"] = 3
        else:
            row["cz3_metrics"]["E_norm"] = {"missing_norm": None, "negative_norm": -1, "above_threshold": 0.1}[problem]
    _replace_metadata(store, "points/00001", update)
    with pytest.raises(ValueError):
        report.generate_reassessment_report(store.root)
    assert not (store.root / "points.csv").exists()


def test_report_rejects_full_circuit_from_another_gate_bundle(store):
    _replace_metadata(store, "circuits/two_qutrit/00001", lambda row: row.update(gate_point_fingerprint="other"))
    with pytest.raises(ValueError, match="identity"):
        report.generate_reassessment_report(store.root)


def test_empty_report_marks_missing_results_and_draws_empty_circuit(tmp_path):
    config = ThetaBenchmarkConfig(theta_points=2, cz3_tolerance=5e-4)
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.md").write_text("previous report")
    manifest = {"benchmark": "theta_threshold_reassessment_v1", "config": config.to_dict(),
                "source_run": str(source)}
    manifest["fingerprint"] = fingerprint(manifest)
    store = RunStore.create(tmp_path / "empty", manifest)
    store.write_bundle("baseline", metadata={})
    result = report.generate_reassessment_report(store.root)
    text = result.read_text(encoding="utf-8")
    assert "missing points: [0, 1]" in text
    assert "**0/2**" in text
    assert (source / "report.md").as_posix() in text
    assert _csv(store.root / "points.csv") == []
    identity = report._Artifact("identity", "Empty", 0, QuantumCircuit(2), "unused.qpy")
    assert report._parameter_rows([identity])[0]["operation"] == "identity"
    report._draw_circuit(store.root, identity)
    assert (store.root / "diagrams/identity.png").stat().st_size > 100


def test_report_rejects_saved_fidelity_inconsistent_with_real_state(store):
    _replace_metadata(store, "circuits/two_qutrit/00001", lambda row: row.update(fidelity=0.2))
    with pytest.raises(ValueError, match="fidelity"):
        report.generate_reassessment_report(store.root)
    assert not (store.root / "report.md").exists()
