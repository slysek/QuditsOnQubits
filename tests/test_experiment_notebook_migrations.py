import ast
import json
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.experiments import (
    BenchmarkBasis,
    ExperimentSpec,
    IQMHardware,
    PathBasis,
    PiastQHardware,
)


NOTEBOOK_CASES = (
    (
        "notebooks/working/iqm/best_garnet_ghz.ipynb",
        "ghz3",
        "run_experiments",
        "IQMHardware",
        "BenchmarkBasis",
    ),
    (
        "notebooks/working/iqm/best_garnet_ame43.ipynb",
        "ame43",
        "run_experiments",
        "IQMHardware",
        "BenchmarkBasis",
    ),
    (
        "notebooks/working/aqt/aqt_two_qutrit.ipynb",
        "two_qutrit",
        "run_experiment",
        "PiastQHardware",
        "PathBasis",
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


def _code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _named_calls(source, names):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _runner_cells(notebook):
    runners = {"run_experiment", "run_experiments"}
    matches = []
    for cell in _code_cells(notebook):
        calls = _named_calls("".join(cell["source"]), runners)
        if calls:
            matches.append((cell, calls))
    return matches


def _execute_setup_cells(relative_path, monkeypatch, cwd):
    notebook = _load_notebook(relative_path)
    monkeypatch.chdir(cwd)
    namespace = {"__name__": "__notebook_setup__"}
    for cell in _code_cells(notebook):
        source = "".join(cell["source"])
        if _named_calls(source, {"run_experiment", "run_experiments"}):
            continue
        exec(compile(source, str(REPO_ROOT / relative_path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("relative_path", "state", "runner", "backend", "basis"),
    NOTEBOOK_CASES,
)
def test_notebooks_use_only_public_high_level_experiment_api(
    relative_path, state, runner, backend, basis
):
    notebook = _load_notebook(relative_path)
    source = _code_source(notebook)
    tree = ast.parse(source)
    public_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "qudits_on_qubits.experiments"
        for alias in node.names
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"ExperimentSpec", backend, basis, runner} <= public_imports
    assert {"ExperimentSpec", backend, basis, runner} <= calls
    assert "product" not in calls
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert {"values", "artifact_dir"} <= attributes
    spec_calls = _named_calls(source, {"ExperimentSpec"})
    assert spec_calls
    assert all(
        isinstance(state_keyword := next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "state"
        ), ast.Constant)
        and state_keyword.value == state
        for call in spec_calls
    )
    if state in {"ghz3", "ame43"}:
        for call in spec_calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            assert "uncertainty" in keyword_names
            assert "bootstrap" not in keyword_names

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
    runner_cells = _runner_cells(notebook)

    assert len(runner_cells) == 1
    for cell, calls in runner_cells:
        assert len(calls) == 1
        assert calls[0].func.id == runner
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


@pytest.mark.parametrize(
    ("relative_path", "state", "runner", "backend", "basis"),
    NOTEBOOK_CASES,
)
@pytest.mark.parametrize("launch_directory", ("repo", "notebook"))
def test_setup_cells_resolve_repo_and_build_valid_specs(
    relative_path,
    state,
    runner,
    backend,
    basis,
    launch_directory,
    monkeypatch,
):
    notebook_path = REPO_ROOT / relative_path
    cwd = REPO_ROOT if launch_directory == "repo" else notebook_path.parent
    namespace = _execute_setup_cells(relative_path, monkeypatch, cwd)

    assert namespace["REPO_ROOT"] == REPO_ROOT
    assert "__file__" not in namespace

    if state in {"ghz3", "ame43"}:
        specs = namespace["SPECS"]
        expected_run_id = "ghz3_qpy13" if state == "ghz3" else "ame43"
        assert namespace["RUN_ID"] == expected_run_id
        assert namespace["SELECTION"] == "exact"
        assert namespace["RANKS"] == tuple(range(1, 12))
        assert isinstance(specs, tuple)
        assert tuple(spec.basis.rank for spec in specs) == namespace["RANKS"]
        for spec in specs:
            assert isinstance(spec, ExperimentSpec)
            assert spec.state == state
            assert isinstance(spec.basis, BenchmarkBasis)
            assert spec.basis.run_kind == "iqm_runs"
            assert spec.basis.run_id == namespace["RUN_ID"]
            assert spec.basis.selection == namespace["SELECTION"]
            assert isinstance(spec.backend, IQMHardware)
            assert spec.backend.device == "garnet"
            assert spec.backend.use_metrics is True
            assert spec.shots == 20_480
            assert spec.mitigation.readout is True
            assert spec.mitigation.zne is True
            assert spec.mitigation.zne_factors == (1, 3, 5)
            assert spec.uncertainty.samples == 2000
            assert spec.bootstrap is spec.uncertainty
            assert spec.output_root == Path("artifacts/experiment_runs")
            assert not spec.output_root.is_absolute()
    else:
        spec = namespace["SPEC"]
        expected_basis = (
            REPO_ROOT
            / "artifacts"
            / "piast_runs"
            / "processed"
            / "transpiler_harness"
            / "20260709_154209"
            / "quantum_circuits"
            / "two_qutrit"
            / "monomial_full__sup012_P120_ph011"
        )
        assert isinstance(spec, ExperimentSpec)
        assert spec.state == "two_qutrit"
        assert isinstance(spec.basis, PathBasis)
        assert namespace["BASIS_DIRECTORY"] == expected_basis
        assert spec.basis.directory == expected_basis
        assert isinstance(spec.backend, PiastQHardware)
        assert spec.backend.mode == "managed"
        assert spec.backend.owner == "notebook"
        assert spec.shots == 200
        assert spec.output_root == Path("artifacts/experiment_runs")


@pytest.mark.parametrize("relative_path", [case[0] for case in NOTEBOOK_CASES])
def test_repo_root_helper_rejects_unrelated_directory(
    relative_path, monkeypatch, tmp_path
):
    namespace = _execute_setup_cells(relative_path, monkeypatch, REPO_ROOT)

    with pytest.raises(RuntimeError, match="repository root"):
        namespace["find_repo_root"](tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "state", "runner", "backend", "basis"),
    NOTEBOOK_CASES,
)
def test_runner_calls_anchor_relative_paths_to_repo_root(
    relative_path, state, runner, backend, basis
):
    notebook = _load_notebook(relative_path)
    runner_cells = _runner_cells(notebook)

    assert len(runner_cells) == 1
    call = runner_cells[0][1][0]
    keyword = next(
        (item for item in call.keywords if item.arg == "repo_root"), None
    )
    assert keyword is not None
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "REPO_ROOT"


@pytest.mark.parametrize("relative_path", [case[0] for case in NOTEBOOK_CASES])
def test_notebooks_do_not_embed_credentials_or_user_paths(relative_path):
    source = _code_source(_load_notebook(relative_path))

    assert not re.search(r"(?i)(token|api[_-]?key|password)\s*=", source)
    assert "dashboard_api" not in source
    assert "os.environ" not in source
    assert not re.search(r"(?i)\b[A-Z]:\\", source)
    assert "\\Users\\" not in source


def test_ame_notebook_never_substitutes_ghz_state_or_candidate():
    source = _code_source(_load_notebook(NOTEBOOK_CASES[1][0]))

    assert 'state="ghz3"' not in source
    assert 'candidate="ghz3"' not in source
