# Canonical Two-Qutrit Bell Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reproducible notebook that prepares the canonical two-qutrit encoding input and runs the durable Bell pipeline independently on Aer, IQM Garnet, and PiastQ.

**Architecture:** Keep canonical source-input preparation inside reusable notebook functions, writing beneath `experiment_inputs/reference_bases/`. Keep experiment execution entirely on the public `ExperimentSpec`/`PathBasis`/`run_experiment` API. Hardware cells are separate, false-by-default guarded submission points; Aer is the only backend executed by `Run All` defaults.

**Tech Stack:** Python 3.10+, Jupyter notebook JSON, NumPy, Qiskit QPY/Statevector, Qiskit Aer, pytest, existing `qudits_on_qubits.experiments` pipeline.

---

## File Structure

- Create `experiment_inputs/README.md`: contract and extensible directory layout for reusable experiment inputs.
- Create `notebooks/two_qutrit_bell_canonical_baseline.ipynb`: canonical input preparation, three backend cells, and result summary.
- Create `tests/test_two_qutrit_canonical_baseline_notebook.py`: structural, safety, input-validation, idempotence, and Aer smoke tests.
- Do not modify pipeline implementation: notebook consumes current public API.

### Task 1: Lock Notebook Contract with Failing Static Tests

**Files:**
- Create: `tests/test_two_qutrit_canonical_baseline_notebook.py`

- [ ] **Step 1: Write notebook loader and failing structure test**

```python
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


def code_cells():
    return [cell for cell in load_notebook()["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def named_calls(cell_source, name):
    return [
        node
        for node in ast.walk(ast.parse(cell_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_notebook_has_three_separate_high_level_backend_runs():
    runner_cells = [cell for cell in code_cells() if named_calls(source(cell), "run_experiment")]
    assert len(runner_cells) == 3
    assert all(len(named_calls(source(cell), "run_experiment")) == 1 for cell in runner_cells)
    combined = "\n".join(source(cell) for cell in code_cells())
    tree = ast.parse(combined)
    public_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "qudits_on_qubits.experiments"
        for alias in node.names
    }
    assert {
        "AerIdeal", "BootstrapConfig", "ExperimentSpec", "IQMHardware",
        "MitigationConfig", "PathBasis", "PiastQHardware", "run_experiment",
    } <= public_imports
    assert "AerIdeal(" in source(runner_cells[0])
    assert "IQMHardware(" in source(runner_cells[1])
    assert "PiastQHardware(" in source(runner_cells[2])
    assert all("repo_root=REPO_ROOT" in source(cell) for cell in runner_cells)
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in runner_cells)
```

- [ ] **Step 2: Write failing hardware-guard and security tests**

```python
def test_hardware_cells_are_false_by_default_and_separately_guarded():
    combined = "\n".join(source(cell) for cell in code_cells())
    assert "RUN_IQM = False" in combined
    assert "RUN_PIASTQ = False" in combined
    iqm_cell = next(cell for cell in code_cells() if "IQMHardware(" in source(cell))
    piastq_cell = next(cell for cell in code_cells() if "PiastQHardware(" in source(cell))
    assert "if RUN_IQM:" in source(iqm_cell)
    assert "if RUN_PIASTQ:" in source(piastq_cell)


def test_notebook_has_no_secrets_user_paths_or_low_level_execution():
    combined = "\n".join(source(cell) for cell in code_cells())
    assert not re.search(r"(?i)(token|api[_-]?key|password)\s*=", combined)
    assert "dashboard_api" not in combined
    assert "\\Users\\" not in combined
    assert not re.search(r"(?i)\b[A-Z]:\\", combined)
    assert "PiastQClient" not in combined
    assert "IQMProvider(" not in combined
    assert "compute_bell_value_from_counts" not in combined
    assert "build_sampler_circuits" not in combined
    assert ".run(" not in combined
```

- [ ] **Step 3: Run tests to verify missing notebook failure**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py -q`

Expected: FAIL with `FileNotFoundError` for `notebooks/two_qutrit_bell_canonical_baseline.ipynb`.

- [ ] **Step 4: Commit failing contract tests**

```bash
git add tests/test_two_qutrit_canonical_baseline_notebook.py
git commit -m "test: define canonical Bell notebook contract"
```

### Task 2: Add Extensible Canonical Input Preparation

**Files:**
- Create: `experiment_inputs/README.md`
- Create: `notebooks/two_qutrit_bell_canonical_baseline.ipynb`
- Modify: `tests/test_two_qutrit_canonical_baseline_notebook.py`

- [ ] **Step 1: Add failing tests for setup functions**

Add helpers that execute only notebook imports and function-definition cells. Mark the materialization cell with `metadata.tags = ["canonical-input-materialization"]` so tests can skip it.

```python
def setup_namespace(cwd):
    namespace = {"__name__": "__notebook_test__"}
    for cell in code_cells():
        tags = cell.get("metadata", {}).get("tags", [])
        cell_source = source(cell)
        if "canonical-input-materialization" in tags:
            continue
        if named_calls(cell_source, "run_experiment"):
            continue
        if "summarize_results(" in cell_source and "def summarize_results(" not in cell_source:
            continue
        exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("launch_directory", (REPO_ROOT, NOTEBOOK_PATH.parent))
def test_setup_finds_repository_root(monkeypatch, launch_directory):
    monkeypatch.chdir(launch_directory)
    namespace = setup_namespace(launch_directory)
    assert namespace["find_repo_root"]() == REPO_ROOT


def test_prepare_canonical_basis_is_valid_and_idempotent(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    first = namespace["prepare_canonical_basis"](tmp_path)
    first_hashes = {path.name: namespace["sha256_file"](path) for path in first.iterdir()}
    second = namespace["prepare_canonical_basis"](tmp_path)
    second_hashes = {path.name: namespace["sha256_file"](path) for path in second.iterdir()}
    assert first == tmp_path / "experiment_inputs/reference_bases/two_qutrit/canonical_ez"
    assert first == second
    assert first_hashes == second_hashes
    assert {path.name for path in first.iterdir()} == {
        "E.npy", "graph_state_direct_basis.qpy", "metadata.json"
    }


def test_prepare_canonical_basis_rejects_modified_input(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    directory = namespace["prepare_canonical_basis"](tmp_path)
    (directory / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata"):
        namespace["prepare_canonical_basis"](tmp_path)
```

- [ ] **Step 2: Run setup tests to verify helper is absent**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py -q`

Expected: FAIL because notebook/setup functions do not exist.

- [ ] **Step 3: Document experiment-input layout**

Create `experiment_inputs/README.md` with this contract:

```markdown
# Experiment inputs

Reusable, deterministic source inputs for experiment runners. Runtime results
belong under `artifacts/`; source bases belong here.

Layout: `reference_bases/<state>/<encoding>/`.

Each basis directory contains:

- `graph_state_direct_basis.qpy`: one unmeasured source circuit;
- `E.npy`: finite isometric logical-to-physical encoding;
- `metadata.json`: schema, identity, dimensions, and SHA-256 hashes.

Canonical bundles are immutable after creation. If validation fails, inspect or
remove the exact damaged basis directory before regenerating it.
```

- [ ] **Step 4: Create notebook setup and validation cells**

Create a valid notebook with Markdown context plus code cells implementing:

```python
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector


def find_repo_root(start=Path.cwd()):
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/qudits_on_qubits").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root()
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits import get_encoding, get_reference_experiment
from qudits_on_qubits.benchmarks.direct_basis.circuits import build_direct_basis_graph_state_circuit
from qudits_on_qubits.experiments import (
    AerIdeal, BootstrapConfig, ExperimentSpec, IQMHardware, MitigationConfig,
    PathBasis, PiastQHardware, run_experiment,
)
```

Define `sha256_file`, `load_single_circuit`, `validate_canonical_basis`, and
`prepare_canonical_basis`. `prepare_canonical_basis(repo_root)` must:

1. derive `E` from `get_encoding("canonical_ez").as_array()`;
2. derive the circuit from
   `build_direct_basis_graph_state_circuit("two_qutrit", E)`;
3. validate an existing complete bundle without writing;
4. reject a partially existing bundle;
5. write QPY, NPY, then metadata through UUID-named sibling temporary files and
   `Path.replace`;
6. delete only its own known temporary files in `finally`;
7. validate hashes, metadata, isometry, four-qubit/no-classical-bit circuit,
   forbidden instructions, and `Statevector(...).equiv(expected_state)`.

Use this exact metadata shape:

```python
{
    "schema": "qoq-reference-basis-v1",
    "state": "two_qutrit",
    "encoding_id": "canonical_ez",
    "num_qubits": 4,
    "encoding_shape": [4, 3],
    "files": {
        "graph_state_direct_basis.qpy": {"sha256": sha256_file(qpy_path)},
        "E.npy": {"sha256": sha256_file(encoding_path)},
    },
}
```

Add a tagged materialization cell:

```python
CANONICAL_BASIS_DIRECTORY = prepare_canonical_basis(REPO_ROOT)
CANONICAL_BASIS_DIRECTORY
```

- [ ] **Step 5: Run setup and static tests**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py -q`

Expected: setup/idempotence/security tests PASS; backend-run count still fails until Task 3.

- [ ] **Step 6: Commit canonical input preparation**

```bash
git add experiment_inputs/README.md notebooks/two_qutrit_bell_canonical_baseline.ipynb tests/test_two_qutrit_canonical_baseline_notebook.py
git commit -m "feat: prepare canonical two-qutrit experiment input"
```

### Task 3: Add Three Independent Pipeline Runs and Summary

**Files:**
- Modify: `notebooks/two_qutrit_bell_canonical_baseline.ipynb`
- Modify: `tests/test_two_qutrit_canonical_baseline_notebook.py`

- [ ] **Step 1: Add failing semantic configuration assertions**

Extend the structure test with AST/setup assertions:

```python
def test_shared_and_backend_configuration_matches_baseline_contract():
    combined = "\n".join(source(cell) for cell in code_cells())
    assert "SHOTS = 20_480" in combined
    assert "BootstrapConfig(samples=2_000" in combined
    assert "MitigationConfig(readout=True, zne=True, zne_factors=(1, 3, 5))" in combined
    assert 'device="garnet"' in combined
    assert 'mode="auto"' in combined
    assert 'owner="notebook"' in combined
    assert 'state="two_qutrit"' in combined
    assert "PathBasis(CANONICAL_BASIS_DIRECTORY)" in combined
```

- [ ] **Step 2: Run test to verify missing execution cells**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py -q`

Expected: FAIL because shared constants and backend cells are absent.

- [ ] **Step 3: Add shared configuration cell**

```python
SHOTS = 20_480
UNCERTAINTY = BootstrapConfig(samples=2_000, seed=7)
HARDWARE_MITIGATION = MitigationConfig(
    readout=True,
    zne=True,
    zne_factors=(1, 3, 5),
)
REFERENCE = get_reference_experiment("two_qutrit")
RESULTS = {}
```

- [ ] **Step 4: Add Aer runner cell**

```python
AER_SPEC = ExperimentSpec(
    state="two_qutrit",
    basis=PathBasis(CANONICAL_BASIS_DIRECTORY),
    backend=AerIdeal(seed_simulator=11),
    shots=SHOTS,
    uncertainty=UNCERTAINTY,
    tags={"baseline": "canonical_ez", "backend": "aer_ideal"},
)
AER_RESULT = run_experiment(AER_SPEC, repo_root=REPO_ROOT)
RESULTS["aer_ideal"] = AER_RESULT
```

- [ ] **Step 5: Add IQM runner cell**

```python
RUN_IQM = False
if RUN_IQM:
    IQM_SPEC = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(CANONICAL_BASIS_DIRECTORY),
        backend=IQMHardware(device="garnet", use_metrics=True),
        shots=SHOTS,
        mitigation=HARDWARE_MITIGATION,
        uncertainty=UNCERTAINTY,
        tags={"baseline": "canonical_ez", "backend": "iqm_garnet"},
    )
    IQM_RESULT = run_experiment(IQM_SPEC, repo_root=REPO_ROOT)
    RESULTS["iqm_garnet"] = IQM_RESULT
else:
    print("IQM skipped: set RUN_IQM = True to submit.")
```

- [ ] **Step 6: Add PiastQ runner cell**

```python
RUN_PIASTQ = False
if RUN_PIASTQ:
    PIASTQ_SPEC = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(CANONICAL_BASIS_DIRECTORY),
        backend=PiastQHardware(mode="auto", owner="notebook"),
        shots=SHOTS,
        mitigation=HARDWARE_MITIGATION,
        uncertainty=UNCERTAINTY,
        tags={"baseline": "canonical_ez", "backend": "piastq"},
    )
    PIASTQ_RESULT = run_experiment(PIASTQ_SPEC, repo_root=REPO_ROOT)
    RESULTS["piastq"] = PIASTQ_RESULT
else:
    print("PiastQ skipped: set RUN_PIASTQ = True to submit.")
```

- [ ] **Step 7: Add result-summary cells**

Define `summarize_results(results, reference)` to return one JSON-safe row per
available backend with `status`, `raw`, optional `readout_mitigated`, `zne`,
`zne_readout_mitigated`, `diagnostics`, any `leakage_rate` key exposed by the
pipeline, `classical_bound`, `ideal_bell_value`, and `artifact_dir`. Add a final
display cell:

```python
SUMMARY = summarize_results(RESULTS, REFERENCE)
SUMMARY
```

- [ ] **Step 8: Run focused tests**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py -q`

Expected: all focused tests PASS.

- [ ] **Step 9: Commit execution cells**

```bash
git add notebooks/two_qutrit_bell_canonical_baseline.ipynb tests/test_two_qutrit_canonical_baseline_notebook.py
git commit -m "feat: run canonical Bell baseline on three backends"
```

### Task 4: Add Real Aer Smoke Coverage

**Files:**
- Modify: `tests/test_two_qutrit_canonical_baseline_notebook.py`

- [ ] **Step 1: Write small-shot Aer integration test**

```python
def test_generated_canonical_basis_runs_through_real_aer(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="two_qutrit",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=64,
        uncertainty=namespace["BootstrapConfig"](samples=20, seed=7),
        output_root=tmp_path / "runs",
    )
    result = namespace["run_experiment"](spec, repo_root=tmp_path)
    assert result.status.value == "completed"
    assert set(result.values) == {"raw", "config", "diagnostics"}
    assert result.values["raw"]["estimate"]["real"] == pytest.approx(6.0, abs=0.8)
    assert result.artifact_dir.is_relative_to(tmp_path / "runs")
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py::test_generated_canonical_basis_runs_through_real_aer -q`

Expected: PASS with one durable local run below pytest temporary directory.

- [ ] **Step 3: Run related experiment tests**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py tests/test_experiment_aer_integration.py tests/test_experiment_notebook_migrations.py -q`

Expected: all selected tests PASS.

- [ ] **Step 4: Commit Aer smoke coverage**

```bash
git add tests/test_two_qutrit_canonical_baseline_notebook.py
git commit -m "test: verify canonical Bell notebook on Aer"
```

### Task 5: Final Verification

**Files:**
- Verify: `experiment_inputs/README.md`
- Verify: `notebooks/two_qutrit_bell_canonical_baseline.ipynb`
- Verify: `tests/test_two_qutrit_canonical_baseline_notebook.py`

- [ ] **Step 1: Validate notebook JSON and code syntax**

Run:

```powershell
python -c "import ast,json,pathlib; p=pathlib.Path('notebooks/two_qutrit_bell_canonical_baseline.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); [ast.parse(''.join(c['source'])) for c in n['cells'] if c['cell_type']=='code']; print('notebook ok')"
```

Expected: `notebook ok`.

- [ ] **Step 2: Run focused and related tests again**

Run: `python -m pytest tests/test_two_qutrit_canonical_baseline_notebook.py tests/test_experiment_aer_integration.py tests/test_experiment_notebook_migrations.py -q`

Expected: all selected tests PASS.

- [ ] **Step 3: Run complete suite**

Run: `python -m pytest -q`

Expected: all tests PASS; any environment-only optional-backend skips remain skips.

- [ ] **Step 4: Review changes and repository cleanliness**

Run:

```powershell
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors; only planned files plus pre-existing user-owned
`.vscode/` and `M2.1_API_DEFINITIONS.md` changes.

- [ ] **Step 5: Commit any verification-only corrections**

```bash
git add experiment_inputs/README.md notebooks/two_qutrit_bell_canonical_baseline.ipynb tests/test_two_qutrit_canonical_baseline_notebook.py
git commit -m "fix: harden canonical Bell notebook"
```

Skip this commit when verification requires no correction.
