import ast
import json
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "two_qutrit_bell_canonical_baseline.ipynb"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def named_calls(cell_source, name):
    tree = ast.parse(cell_source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_notebook_has_three_separate_high_level_backend_runs():
    notebook = load_notebook()
    cells = code_cells(notebook)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]

    assert len(run_cells) == 3
    assert all(len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells)

    tree = ast.parse("\n".join(source(cell) for cell in cells))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "qudits_on_qubits.experiments"
        for alias in node.names
    }
    assert {
        "AerIdeal",
        "BootstrapConfig",
        "ExperimentSpec",
        "IQMHardware",
        "MitigationConfig",
        "PathBasis",
        "PiastQHardware",
        "run_experiment",
    } <= imported

    backend_order = ("AerIdeal", "IQMHardware", "PiastQHardware")
    assert [
        next(backend for backend in backend_order if backend in source(cell))
        for cell in run_cells
    ] == list(backend_order)

    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo_root_keywords = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo_root_keywords) == 1
        assert isinstance(repo_root_keywords[0].value, ast.Name)
        assert repo_root_keywords[0].value.id == "REPO_ROOT"
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_hardware_cells_are_false_by_default_and_separately_guarded():
    notebook = load_notebook()
    cells = code_cells(notebook)
    full_source = "\n".join(source(cell) for cell in cells)

    assert re.search(r"^RUN_IQM\s*=\s*False\s*$", full_source, re.MULTILINE)
    assert re.search(r"^RUN_PIASTQ\s*=\s*False\s*$", full_source, re.MULTILINE)

    iqm_cells = [cell for cell in cells if "IQMHardware" in source(cell)]
    piastq_cells = [cell for cell in cells if "PiastQHardware" in source(cell)]
    assert iqm_cells and any(re.search(r"if\s+RUN_IQM\b", source(cell)) for cell in iqm_cells)
    assert piastq_cells and any(
        re.search(r"if\s+RUN_PIASTQ\b", source(cell)) for cell in piastq_cells
    )


def test_notebook_has_no_secrets_user_paths_or_low_level_execution():
    notebook = load_notebook()
    full_source = "\n".join(source(cell) for cell in code_cells(notebook))

    assert not re.search(
        r"(?im)^\s*\w*(?:token|api[_ -]?key|password)\w*\s*=", full_source
    )
    assert "dashboard_api" not in full_source
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:\\\\", full_source)
    for forbidden in (
        "PiastQClient",
        "IQMProvider(",
        "compute_bell_value_from_counts",
        "build_sampler_circuits",
        ".run(",
    ):
        assert forbidden not in full_source
