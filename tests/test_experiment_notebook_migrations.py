import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_CASES = (
    (
        "notebooks/working/iqm/best_garnet_ghz.ipynb",
        "ghz3",
        "run_experiments",
        'IQMHardware(device="garnet", use_metrics=True)',
        "BenchmarkBasis(",
    ),
    (
        "notebooks/working/iqm/best_garnet_ame43.ipynb",
        "ame43",
        "run_experiments",
        'IQMHardware(device="garnet", use_metrics=True)',
        "BenchmarkBasis(",
    ),
    (
        "notebooks/working/aqt/aqt_two_qutrit.ipynb",
        "two_qutrit",
        "run_experiment",
        'PiastQHardware(mode="managed", owner="notebook")',
        "PathBasis(BASIS_DIRECTORY)",
    ),
)

FORBIDDEN_EXECUTION_DUPLICATES = (
    "def full_pipeline",
    "def build_readout_calibration_matrices",
    "def apply_mitigation",
    "def fold_cz_preserve_layout",
    "PiastQClient",
    "PiastQSampler",
    "compute_bell_value_from_counts_aqt",
    "run_sampler_circuits_to_counts_by_setting(",
    "load_piast_backend(",
    "IQMProvider(",
    "mthree",
    "np.polyfit",
    "qpy.load(",
    ".run(",
)


def _load_notebook(relative_path):
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _code_source(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize(
    ("relative_path", "state", "runner", "backend", "basis"),
    NOTEBOOK_CASES,
)
def test_notebooks_use_only_public_high_level_experiment_api(
    relative_path, state, runner, backend, basis
):
    notebook = _load_notebook(relative_path)
    source = _code_source(notebook)

    assert "from qudits_on_qubits.experiments import (" in source
    assert "ExperimentSpec(" in source
    assert f'state="{state}"' in source
    assert backend in source
    assert basis in source
    assert re.search(rf"\b{runner}\(", source)

    for forbidden in FORBIDDEN_EXECUTION_DUPLICATES:
        assert forbidden not in source


@pytest.mark.parametrize(
    ("relative_path", "state", "runner", "backend", "basis"),
    NOTEBOOK_CASES,
)
def test_runner_cells_have_no_stale_execution_state(
    relative_path, state, runner, backend, basis
):
    notebook = _load_notebook(relative_path)
    runner_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and re.search(rf"\b{runner}\(", "".join(cell["source"]))
    ]

    assert runner_cells
    for cell in runner_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


@pytest.mark.parametrize("relative_path", [case[0] for case in NOTEBOOK_CASES])
def test_notebooks_do_not_embed_credentials_or_user_paths(relative_path):
    source = _code_source(_load_notebook(relative_path))

    assert not re.search(r"(?i)(token|api[_-]?key|password)\s*=", source)
    assert "dashboard_api" not in source
    assert "os.environ" not in source
    assert not re.search(r"(?i)\b[A-Z]:\\", source)
    assert "\\Users\\" not in source


@pytest.mark.parametrize(
    "relative_path,state",
    [(NOTEBOOK_CASES[0][0], "ghz3"), (NOTEBOOK_CASES[1][0], "ame43")],
)
def test_iqm_batches_use_explicit_benchmark_specs_and_prior_mitigation(
    relative_path, state
):
    source = _code_source(_load_notebook(relative_path))

    assert "RUN_ID =" in source
    assert 'SELECTION = "exact"' in source
    assert "RANKS = (" in source
    assert "SPECS = tuple(" in source
    assert "run_experiments(SPECS)" in source
    assert not re.search(r"\brun_experiment\(", source)
    assert "itertools.product" not in source
    assert "product(" not in source
    assert "MitigationConfig(" in source
    assert "readout=True" in source
    assert "zne=True" in source
    assert "zne_factors=(1, 3, 5)" in source
    assert "BootstrapConfig(samples=2000" in source
    assert "result.values" in source
    assert "result.artifact_dir" in source


def test_aqt_uses_one_managed_piastq_experiment_and_relative_basis():
    source = _code_source(_load_notebook(NOTEBOOK_CASES[2][0]))

    assert 'state="two_qutrit"' in source
    assert 'mode="managed"' in source
    assert "shots=200" in source
    assert "run_experiment(SPEC)" in source
    assert "run_experiments(" not in source
    assert "BASIS_DIRECTORY = (" in source
    assert '"20260709_154209"' in source
    assert '"monomial_full__sup012_P120_ph011"' in source
    assert "MitigationConfig(" not in source
    assert "BootstrapConfig(" not in source
    assert "result.values" in source
    assert "result.artifact_dir" in source


def test_ame_notebook_never_substitutes_ghz_state_or_candidate():
    source = _code_source(_load_notebook(NOTEBOOK_CASES[1][0]))

    assert 'state="ghz3"' not in source
    assert 'candidate="ghz3"' not in source
