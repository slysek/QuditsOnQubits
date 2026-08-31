from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ghz3_bell_canonical_baseline.ipynb"


def _resolve_iqm_env_with_notebook_contract(monkeypatch) -> Path:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    setup_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def resolve_iqm_env_path" in "".join(cell["source"])
    )
    namespace = {
        "__name__": "__iqm_selector_smoke__",
        "__file__": str(NOTEBOOK_PATH),
    }
    monkeypatch.chdir(REPO_ROOT)
    exec(
        compile(
            "".join(setup_cell["source"]),
            str(NOTEBOOK_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["resolve_iqm_env_path"](REPO_ROOT)


@pytest.mark.skipif(
    os.environ.get("QOQ_RUN_IQM_SELECTOR_SMOKE") != "1",
    reason="set QOQ_RUN_IQM_SELECTOR_SMOKE=1 for live compile-only IQM smoke",
)
def test_iqm_selector_and_full_workload_compile_never_submit(
    tmp_path,
    monkeypatch,
):
    from qudits_on_qubits.experiments.artifacts import load_basis_artifacts
    from qudits_on_qubits.experiments.backends import IQMAdapter
    from qudits_on_qubits.experiments.models import (
        ExperimentSpec,
        IQMHardware,
        IQMQubitSelectorConfig,
        PathBasis,
        WorkloadOptimizationConfig,
    )
    from qudits_on_qubits.experiments.preparation import prepare_measurements
    from qudits_on_qubits.experiments.runner import (
        _compile_measurement_workload,
    )

    env_path = _resolve_iqm_env_with_notebook_contract(monkeypatch)
    basis = PathBasis(
        REPO_ROOT
        / "experiment_inputs"
        / "reference_bases"
        / "ghz3"
        / "canonical_ez"
    )
    artifacts = load_basis_artifacts(basis, "ghz3", REPO_ROOT)
    prepared = prepare_measurements(artifacts)
    settings = tuple(
        tuple(setting)
        for setting in prepared.metadata["setting_by_circuit_index"]
    )
    selector = IQMQubitSelectorConfig(top_k=2, num_trials=200)
    spec = ExperimentSpec(
        state="ghz3",
        basis=basis,
        backend=IQMHardware(
            "garnet",
            use_metrics=True,
            env_path=env_path,
        ),
        shots=1,
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=((0, 1, 2, 7, 3, 4),),
            seed_transpilers=(3,),
            iqm_qubit_selector=selector,
        ),
        output_root=tmp_path / "forbidden-runs",
    )
    adapter = IQMAdapter(spec.backend)
    identity = adapter.resolve()
    backend = adapter.backend
    submit_guard = Mock(side_effect=AssertionError("adapter.submit called"))
    backend_run_guard = Mock(side_effect=AssertionError("backend.run called"))
    monkeypatch.setattr(adapter, "submit", submit_guard)
    monkeypatch.setattr(backend, "run", backend_run_guard)

    selection = _compile_measurement_workload(
        adapter,
        prepared.circuits,
        settings,
        spec,
        expected_identity=identity,
    )

    assert selection.metadata["selector"]["provider"] == (
        "iqm-qubit-selector"
    )
    assert len(selection.metadata["selector"]["generated_layouts"]) >= 1
    assert len(selection.batch.circuits) == len(settings) == 12
    assert selection.metadata["selected_layout"]
    assert selection.metadata["selected_seed_transpiler"] == 3
    submit_guard.assert_not_called()
    backend_run_guard.assert_not_called()
    assert not spec.output_root.exists()
