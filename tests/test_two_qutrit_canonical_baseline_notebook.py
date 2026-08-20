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
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def runner_backend(cell_source, call):
    tree = ast.parse(cell_source)
    specs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "ExperimentSpec":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    specs[target.id] = node.value
    value = next((keyword.value for keyword in call.keywords if keyword.arg == "spec"), call.args[0] if call.args else None)
    if isinstance(value, ast.Name):
        value = specs.get(value.id)
    assert isinstance(value, ast.Call)
    backend = next(keyword.value for keyword in value.keywords if keyword.arg == "backend")
    assert isinstance(backend, ast.Call) and isinstance(backend.func, ast.Name)
    return backend.func.id


def test_notebook_has_three_separate_high_level_backend_runs():
    notebook = load_notebook()
    cells = code_cells(notebook)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 3
    assert all(len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells)
    tree = ast.parse("\n".join(source(cell) for cell in cells))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "qudits_on_qubits.experiments" for alias in node.names}
    assert {"AerIdeal", "BootstrapConfig", "ExperimentSpec", "IQMHardware", "MitigationConfig", "PathBasis", "PiastQHardware", "run_experiment"} <= imported
    expected = ["AerIdeal", "IQMHardware", "PiastQHardware"]
    assert [runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0]) for cell in run_cells] == expected
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)
    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo) == 1 and isinstance(repo[0].value, ast.Name) and repo[0].value.id == "REPO_ROOT"


def test_hardware_cells_are_false_by_default_and_separately_guarded():
    notebook = load_notebook()
    cells = code_cells(notebook)
    full_source = "\n".join(source(cell) for cell in cells)
    assert re.search(r"^RUN_IQM\s*=\s*False\s*$", full_source, re.MULTILINE)
    assert re.search(r"^RUN_PIASTQ\s*=\s*False\s*$", full_source, re.MULTILINE)
    for backend, flag in (("IQMHardware", "RUN_IQM"), ("PiastQHardware", "RUN_PIASTQ")):
        cell = next(cell for cell in cells if backend in source(cell))
        tree = ast.parse(source(cell))
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == flag]
        runner = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_experiment"]
        assert guards and len(runner) == 1
        assert any(runner[0] in ast.walk(guard) for guard in guards)
        assert runner_backend(source(cell), runner[0]) == backend


def test_notebook_has_no_secrets_user_paths_or_low_level_execution():
    notebook = load_notebook()
    full_source = "\n".join(source(cell) for cell in code_cells(notebook))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(rf"(?im)os\.environ(?:\.get)?\([^)]*{credential}", full_source)
    assert not re.search(rf"(?im)[\"']{credential}[\"']\s*:", full_source)
    assert "dashboard_api" not in full_source
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in ("PiastQClient", "IQMProvider(", "compute_bell_value_from_counts", "build_sampler_circuits", ".run("):
        assert forbidden not in full_source