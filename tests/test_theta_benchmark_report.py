from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
from qudits_on_qubits.benchmarks.theta_continuation import report


def _template(template_id="branch-a"):
    return {
        "template_id": template_id,
        "parameter_metadata": [
            {"index": index, "instruction_index": 2, "qubit": 1, "angle": angle}
            for index, angle in enumerate(("theta", "phi", "lambda"))
        ],
        "operations": [{"name": "cz", "qubits": [0, 1]}],
    }


@pytest.fixture
def store(tmp_path):
    config = ThetaBenchmarkConfig(theta_points=5)
    store = RunStore.create(tmp_path / "run", {
        "fingerprint": "fixture", "config": config.to_dict(),
        "provenance": {"versions": {"python": "3.12.test", "qiskit": "2.1.test"},
                       "source_hashes": {"src/example.py": "abc123"}},
    })
    store.write_bundle("baseline", metadata={"baseline_n_cz": 6, "original_n_cz": 6},
                       documents={"template.json": _template()})
    for index, theta in enumerate(config.grid()):
        valid = index != 2
        metadata = {
            "index": index, "theta": theta, "theta_over_pi": theta / np.pi,
            "template_id": "branch-a", "status": "failed" if index == 2 else "above_baseline" if index == 3 else "success",
            "correct": valid, "continuation_valid": valid,
            "fallback_used": index in (2, 3), "fallback_reason": "fit_failed" if index == 2 else None,
            "selected_method": "bqskit" if index == 3 else "continuation" if valid else None,
            "within_baseline_budget": valid and index != 3,
            "warm_start_parent_id": f"p-{max(0, index - 1)}", "baseline_n_cz": 6,
            "cz3_n_cz": 8 if index == 3 else 6 if valid else None,
            "cz3_depth": 11 if valid else None, "cz3_n_1q": 11 if valid else None,
            "cz3_E_norm": 1e-6 if valid else None, "cz3_L_norm": 1e-7 if valid else None,
            "f3_alpha": 0.25 + index, "f3_n_cz": 2, "f3_alpha_zero_n_cz": 3,
            "f3_E_norm": 1e-12, "f3_L_norm": 1e-13,
            "continuation_seconds": 2.0, "fallback_seconds": 3.0 if index in (2, 3) else 0.0,
            "errors": [{"stage": "fallback", "error": "finite budget exhausted"}] if index == 2 else [],
        }
        # Keep parent angles on failed point: report must never use them as p(theta).
        arrays = {"branch_parameters.npy": np.array([8.0, 8.0, 8.0])}
        if valid:
            arrays["cz3_parameters.npy"] = np.array([0.2, 3.1 if index == 0 else -3.1, 0.3])
        store.write_bundle(f"points/{index:05d}", metadata=metadata, arrays=arrays, documents={
            "attempts.json": {"attempts": [{"id": f"attempt-{index}", "theta": theta,
                "requested_grid_point": True, "warm_start_parent_id": f"p-{max(0, index-1)}",
                "result": {"success": valid, "nfev": index + 1}}]},
        })
    store.write_bundle("circuits/two_qutrit/00000", metadata={
        "index": 0, "theta": 0.0, "state": "two_qutrit", "success": True,
        "n_cz": 6, "depth": 18, "fidelity": 1.0,
    })
    return store


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_exports_failures_provenance_and_scientific_figures(store):
    path = report.generate_report(store.root)
    assert path == store.root / "report.md"
    text = path.read_text(encoding="utf-8")
    assert "Nieudane punkty: 1" in text
    assert "Poprawne powyżej baseline: 1" in text
    assert "3.12.test" in text and "abc123" in text
    assert "1e-10" in text and "1e-05" in text
    assert "nie dowodzi" in text.lower()
    assert "unwrap" in text.lower() and "gładkości" in text
    rows = _csv(store.root / "points.csv")
    assert len(rows) == 5 and rows[2]["status"] == "failed"
    assert rows[2]["cz3_n_cz"] == "" and rows[3]["cz3_n_cz"] == "8"
    assert rows[0]["f3_alpha_zero_n_cz"] == "3"
    attempts = _csv(store.root / "attempts.csv")
    assert len(attempts) == 5 and attempts[2]["result.success"] == "False"
    assert _csv(store.root / "full_circuits.csv")[0]["state"] == "two_qutrit"
    figures = list((store.root / "figures").glob("*.png"))
    assert len(figures) >= 4
    for figure in figures:
        assert figure.stat().st_size > 100
        assert figure.with_suffix(".pdf").stat().st_size > 100


def test_parameter_series_leave_gaps_and_unwrap_only_contiguous_valid_runs(store):
    report.generate_report(store.root)
    rows = _csv(store.root / "parameters.csv")
    phi = [row for row in rows if row["angle"] == "phi"]
    assert len(phi) == 5
    assert float(phi[1]["raw_radians"]) == pytest.approx(-3.1)
    assert float(phi[1]["unwrapped_radians"]) == pytest.approx(2 * np.pi - 3.1)
    assert phi[2]["raw_radians"] == phi[2]["unwrapped_radians"] == ""
    assert phi[2]["continuation_valid"] == "False"
    assert float(phi[3]["unwrapped_radians"]) == pytest.approx(-3.1)
    distances = _csv(store.root / "operator_distances.csv")
    assert len(distances) == 5
    assert distances[0]["distance"] == distances[2]["distance"] == distances[3]["distance"] == ""
    assert float(distances[4]["distance"]) == pytest.approx(0.0, abs=1e-12)


def test_local_operator_distance_is_global_phase_and_u3_periodicity_invariant():
    angles = np.array([0.61, -0.2, 0.43])
    assert report.local_u3_distance(angles, angles + [2*np.pi, 0, 0]) < 1e-12
    assert report.local_u3_distance([0, 0.2, 0.3], [0, -0.4, 0.9]) < 1e-12
    assert report.local_u3_distance(angles, angles + [0.1, 0, 0]) > 0.01


def test_report_rejects_corrupt_point_before_publishing(store):
    path = store.root / "points/00002/metadata.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|digest|checksum"):
        report.generate_report(store.root)
    assert not (store.root / "report.md").exists()
    assert not (store.root / "points.csv").exists()


def test_report_rejects_corrupt_persisted_circuit_even_if_summary_supplied(store):
    (store.root / "circuits/two_qutrit/00000/metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="hash|digest|checksum"):
        report.generate_report(store.root, full_circuits=[{"state": "two_qutrit", "success": True}])


def test_report_rejects_invalid_configuration_before_outputs(store):
    manifest = json.loads((store.root / "manifest.json").read_text())
    manifest["config"]["theta_points"] = 0
    (store.root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        report.generate_report(store.root)
    assert not (store.root / "points.csv").exists()


def test_report_handles_empty_valid_run(tmp_path):
    store = RunStore.create(tmp_path / "empty", {
        "fingerprint": "empty", "config": ThetaBenchmarkConfig().to_dict(),
    })
    store.write_bundle("baseline", metadata={}, documents={"template.json": _template()})
    path = report.generate_report(store.root)
    assert "Nieudane punkty: 0" in path.read_text(encoding="utf-8")
    assert _csv(store.root / "points.csv") == []


def test_separate_templates_never_share_unwrap_or_operator_segments(tmp_path):
    config = ThetaBenchmarkConfig(theta_points=3)
    store = RunStore.create(tmp_path / "separate", {"fingerprint": "separate", "config": config.to_dict()})
    store.write_bundle("baseline", metadata={}, documents={"template.json": _template()})
    for index, template_id in enumerate(("branch-a", "branch-b", "branch-a")):
        store.write_bundle(f"points/{index:05d}", metadata={
            "index": index, "theta": config.grid()[index], "template_id": template_id,
            "correct": True, "continuation_valid": True, "status": "success",
        }, arrays={"cz3_parameters.npy": np.array([0.2, 3.1 if index == 0 else -3.1, 0.3])},
        documents={"template.json": _template(template_id)})
    report.generate_report(store.root)
    rows = _csv(store.root / "parameters.csv")
    assert {row["template_id"] for row in rows} == {"branch-a", "branch-b"}
    a_phi = [row for row in rows if row["template_id"] == "branch-a" and row["angle"] == "phi"]
    assert a_phi[1]["raw_radians"] == ""
    assert float(a_phi[2]["unwrapped_radians"]) == pytest.approx(-3.1)
    assert all(row["distance"] == "" for row in _csv(store.root / "operator_distances.csv"))


def test_real_pinned_template_exports_all_33_angles_without_fabricating_later_points(tmp_path):
    config = ThetaBenchmarkConfig(theta_points=2)
    template = load_cz3_template(Path(config.baseline_qpy))
    store = RunStore.create(tmp_path / "pinned", {"fingerprint": "pinned", "config": config.to_dict()})
    store.write_bundle("baseline", metadata={}, documents={"template.json": template.to_dict()})
    store.write_bundle("points/00000", metadata={
        "index": 0, "theta": 0.0, "theta_over_pi": 0.0, "template_id": template.template_id,
        "correct": True, "continuation_valid": True, "status": "ok",
    }, arrays={"cz3_parameters.npy": template.initial_parameters})
    text = report.generate_report(store.root).read_text(encoding="utf-8")
    assert "6 CZ, 11 U3, 33 kątów" in text
    rows = _csv(store.root / "parameters.csv")
    assert len(rows) == 66
    first = [row for row in rows if row["index"] == "0"]
    np.testing.assert_allclose([float(row["raw_radians"]) for row in first], template.initial_parameters)
    assert all(row["raw_radians"] == "" for row in rows if row["index"] == "1")
    assert {int(row["instruction_index"]) for row in first} == {item["instruction_index"] for item in template.parameter_metadata}


def test_report_rejects_changed_protocol_with_unchanged_run_fingerprint(store):
    manifest = store.manifest
    manifest["benchmark"] = "theta_continuation_v1"
    manifest.pop("fingerprint")
    manifest["fingerprint"] = fingerprint(manifest)
    manifest["config"]["f3_tolerance"] = 1e-8
    (store.root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        report.generate_report(store.root)
    assert not (store.root / "points.csv").exists()


def test_report_shows_component_cz_costs_before_fusion_and_preserves_1q_costs(store, monkeypatch):
    monkeypatch.setattr(report, "_figures", lambda *args: [])
    components = {
        "component_num_qutrits": 3, "component_num_cz3_blocks": 2,
        "preparation_n_cz": 0, "preparation_n_1q": 0,
        "f3_blocks_n_cz": 6, "f3_blocks_n_1q": 12,
        "cz3_blocks_n_cz": 12, "cz3_blocks_n_1q": 22,
        "unfused_n_cz": 18, "unfused_n_1q": 34,
    }
    store.write_bundle("circuits/ghz3/00000", metadata={
        "index": 0, "theta": 0.0, "theta_over_pi": 0.0, "state_name": "ghz3",
        "success": True, "status": "ok", "two_qubit_gate_count": 16,
        "one_qubit_gate_count": 27, **components,
    })
    text = report.generate_report(store.root).read_text(encoding="utf-8")
    assert "### Koszt składników przed łączeniem bloków" in text
    assert "| przygotowanie CZ | bloki F3 CZ | bloki CZ3 CZ | suma przed łączeniem CZ | pełny obwód CZ |" in text
    assert "| 0 | 0 | ghz3 | 0 | 6 | 12 | 18 | 16 |" in text
    assert "może różnić się" in text and "optymalizacji całego obwodu" in text
    assert "koszty 1q" in text and "full_circuits.csv" in text
    row = next(row for row in _csv(store.root / "full_circuits.csv") if row["state_name"] == "ghz3")
    for key, value in components.items():
        assert row[key] == str(value)
    assert row["two_qubit_gate_count"] == "16" and row["one_qubit_gate_count"] == "27"


def test_report_does_not_infer_missing_component_costs_for_older_bundles(store, monkeypatch):
    monkeypatch.setattr(report, "_figures", lambda *args: [])
    store.write_bundle("circuits/ghz3/00000", metadata={
        "index": 0, "theta": 0.0, "theta_over_pi": 0.0, "state_name": "ghz3",
        "success": True, "status": "ok", "two_qubit_gate_count": 16,
    })
    text = report.generate_report(store.root).read_text(encoding="utf-8")
    assert "| 0 | 0 | ghz3 | — | — | — | — | 16 |" in text
    assert "brak zapisanych danych" in text and "zerowego kosztu" in text
    row = next(row for row in _csv(store.root / "full_circuits.csv") if row["state_name"] == "ghz3")
    assert row.get("preparation_n_cz", "") == ""
    assert row.get("unfused_n_cz", "") == ""
