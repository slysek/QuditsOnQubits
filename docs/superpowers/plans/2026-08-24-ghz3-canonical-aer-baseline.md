# GHZ3 Canonical Aer Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, reproducible GHZ3 `canonical_ez` Aer baseline notebook plus an opt-in IQM Garnet full-mitigation smoke test using 50 shots per circuit.

**Architecture:** Mirror the proven two-qutrit canonical notebook contract in a separate GHZ3 notebook. Keep committed notebook execution local and deterministic; place connected IQM verification behind an explicit environment gate in its test module. Reuse existing experiment runner, reference registry, direct-basis circuit builder, and artifact store without changing production modules.

**Tech Stack:** Python 3.11+, Jupyter notebook JSON, Qiskit/QPY, NumPy, Qiskit Aer, pytest, project `qudits_on_qubits.experiments` runner, IQM Qiskit adapter.

---

## File map

- Create `tests/test_ghz3_canonical_baseline_notebook.py`: structural, safety, materialization, local E2E, and opt-in connected IQM coverage.
- Create `notebooks/ghz3_bell_canonical_baseline.ipynb`: GHZ3 canonical input creation, Aer execution, and one-row summary.
- Do not modify existing source modules or `notebooks/two_qutrit_bell_canonical_baseline.ipynb`.

### Task 1: Add notebook contract tests and establish RED

**Files:**
- Create: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Add notebook parsing and setup helpers**

Use the established AST-based notebook testing pattern. Target the new path and execute setup cells while skipping input materialization, experiment execution, and summary display:

```python
import ast
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ghz3_bell_canonical_baseline.ipynb"
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


def setup_namespace(cwd):
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for cell in code_cells(load_notebook()):
            cell_source = source(cell)
            if "canonical-input-materialization" in cell.get("metadata", {}).get("tags", []):
                continue
            if named_calls(cell_source, "run_experiment") or "SUMMARY =" in cell_source:
                continue
            exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace
```

- [ ] **Step 2: Add structural, configuration, summary, and safety tests**

Add tests with these exact contracts:

```python
def test_notebook_has_one_aer_run_and_clean_cells():
    cells = code_cells(load_notebook())
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 1
    assert len(named_calls(source(run_cells[0]), "run_experiment")) == 1
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)
    full_source = "\n".join(source(cell) for cell in cells)
    assert "AerIdeal" in full_source
    assert "IQMHardware" not in full_source
    assert "PiastQHardware" not in full_source


def test_configuration_and_empty_summary_are_complete():
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["SHOTS"] == 100
    assert namespace["UNCERTAINTY"].samples == 2_000
    assert namespace["UNCERTAINTY"].seed == 7
    assert namespace["REFERENCE"].candidate == "ghz3"
    summary = namespace["summarize_results"]({}, namespace["REFERENCE"])
    assert len(summary) == 1
    assert summary[0]["backend"] == "aer_ideal"
    assert summary[0]["status"] == "not_run"
    assert summary[0]["classical_bound"] == namespace["REFERENCE"].bell_functional.classical_bound
    assert summary[0]["ideal_bell_value"] == namespace["REFERENCE"].expected.ideal_bell_value
    json.dumps(summary)


def test_summary_preserves_runner_values():
    namespace = setup_namespace(REPO_ROOT)
    values = {"raw": {"estimate": 6.0}, "diagnostics": {"factors": [1]}}
    result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ghz3-aer",
        values=values,
    )
    row = namespace["summarize_results"](
        {"aer_ideal": result}, namespace["REFERENCE"]
    )[0]
    assert row["status"] == "completed"
    assert row["raw"] == values["raw"]
    assert row["diagnostics"] == values["diagnostics"]
    assert row["artifact_dir"] == str(result.artifact_dir)


def test_notebook_has_no_secrets_paths_or_low_level_execution():
    full_source = "\n".join(source(cell) for cell in code_cells(load_notebook()))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in (
        "PiastQClient",
        "IQMProvider(",
        "compute_bell_value_from_counts",
        "build_sampler_circuits",
        ".run(",
    ):
        assert forbidden not in full_source
```

Also parse the `ExperimentSpec` AST and assert `state="ghz3"`, `PathBasis(CANONICAL_BASIS_DIRECTORY)`, `AerIdeal(seed_simulator=11)`, `shots=SHOTS`, `uncertainty=UNCERTAINTY`, tags `{"baseline": "canonical_ez", "backend": "aer_ideal"}`, and `run_experiment(..., repo_root=REPO_ROOT)`.

- [ ] **Step 3: Add materialization and local E2E tests**

Add tests for both working directories, exact bundle contents, idempotent hashes, corrupt metadata rejection, `(4, 3)` isometry, one unmeasured 6-qubit circuit, cleanup after mocked `np.save` failure, and local Aer execution:

```python
@pytest.mark.parametrize("cwd", [REPO_ROOT, NOTEBOOK_PATH.parent])
def test_setup_discovers_repo_root(monkeypatch, tmp_path, cwd):
    monkeypatch.chdir(tmp_path)
    namespace = setup_namespace(cwd)
    assert namespace["REPO_ROOT"] == REPO_ROOT
    assert Path.cwd() == tmp_path


def test_prepare_canonical_basis_creates_exact_idempotent_bundle(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    assert target == tmp_path / "experiment_inputs" / "reference_bases" / "ghz3" / "canonical_ez"
    assert {path.name for path in target.iterdir()} == {
        "graph_state_direct_basis.qpy",
        "E.npy",
        "metadata.json",
    }
    first_hashes = {path.name: namespace["sha256_file"](path) for path in target.iterdir()}
    assert namespace["prepare_canonical_basis"](tmp_path) == target
    assert {path.name: namespace["sha256_file"](path) for path in target.iterdir()} == first_hashes


def test_canonical_notebook_runs_ghz3_on_aer(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="ghz3",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=64,
        uncertainty=namespace["BootstrapConfig"](samples=20, seed=7),
        output_root=tmp_path / "runs",
    )
    result = namespace["run_experiment"](spec, repo_root=tmp_path)
    assert result.status.value == "completed"
    assert result.values["raw"]["estimate"]["real"] == pytest.approx(
        namespace["REFERENCE"].expected.ideal_bell_value,
        abs=0.2,
    )
    circuit = namespace["load_single_circuit"](
        basis / "graph_state_direct_basis.qpy"
    )
    assert circuit.num_qubits == 6
    assert circuit.num_clbits == 0
```

Use an initial `abs=0.2` finite-shot tolerance; tighten only after recording deterministic seed-11 output from GREEN.

- [ ] **Step 4: Add opt-in connected IQM full-pipeline test**

```python
@pytest.mark.skipif(
    os.environ.get("QOQ_RUN_IQM_HARDWARE") != "1",
    reason="set QOQ_RUN_IQM_HARDWARE=1 to submit the IQM Garnet smoke run",
)
def test_ghz3_canonical_full_pipeline_on_iqm_garnet():
    from qudits_on_qubits.experiments import IQMHardware, MitigationConfig

    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](REPO_ROOT)
    spec = namespace["ExperimentSpec"](
        state="ghz3",
        basis=namespace["PathBasis"](basis),
        backend=IQMHardware(device="garnet", use_metrics=True),
        shots=50,
        mitigation=MitigationConfig(
            readout=True,
            zne=True,
            zne_factors=(1, 3, 5),
        ),
        uncertainty=namespace["BootstrapConfig"](samples=20, seed=7),
        tags={"baseline": "canonical_ez", "backend": "iqm_garnet_smoke"},
    )
    result = namespace["run_experiment"](spec, repo_root=REPO_ROOT)
    assert result.status.value == "completed"
    assert {"raw", "readout_mitigated", "zne", "zne_readout_mitigated", "diagnostics"} <= set(result.values)
    assert result.values["diagnostics"]["factors"] == [1, 3, 5]
```

Import `IQMHardware` and `MitigationConfig` directly from `qudits_on_qubits.experiments` inside the connected test. Do not add them to notebook imports or namespace assumptions.

- [ ] **Step 5: Run focused test and verify intended RED**

Run:

```powershell
python -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
```

Expected: FAIL because `notebooks/ghz3_bell_canonical_baseline.ipynb` does not exist. No unrelated import, syntax, or dependency failure counts as RED.

- [ ] **Step 6: Commit RED checkpoint**

```powershell
git add tests/test_ghz3_canonical_baseline_notebook.py
git commit -m "test: define GHZ3 canonical notebook contract"
```

Verify commit is reachable from current `HEAD` using `git log -1 --oneline`.

### Task 2: Implement GHZ3 canonical Aer notebook and reach GREEN

**Files:**
- Create: `notebooks/ghz3_bell_canonical_baseline.ipynb`
- Test: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Create clean notebook shell**

Create notebook format 4.5 with kernel metadata matching project notebooks. Use fixed cell IDs prefixed `ghz3-canonical-`. Every code cell starts with:

```json
"execution_count": null,
"outputs": []
```

Add markdown purpose text stating Run All performs only local Aer work and stores no credentials.

- [ ] **Step 2: Add imports and repository discovery**

Notebook imports only high-level local execution dependencies:

```python
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import qiskit.qpy as qpy
from qiskit.quantum_info import Statevector


def find_repo_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "qudits_on_qubits"
        ).is_dir():
            return candidate
    raise RuntimeError(
        "Cannot find repository root. Start from this repository or a descendant "
        "containing pyproject.toml and src/qudits_on_qubits."
    )


REPO_ROOT = find_repo_root()
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.reference_experiments import get_encoding, get_reference_experiment
from qudits_on_qubits.benchmarks.direct_basis.circuits import build_direct_basis_graph_state_circuit
from qudits_on_qubits.experiments import (
    AerIdeal,
    BootstrapConfig,
    ExperimentSpec,
    PathBasis,
    run_experiment,
)
```

- [ ] **Step 3: Add exact canonical bundle validation and materialization**

Adapt the existing two-qutrit helpers without changing their transaction pattern. Required GHZ3 differences:

```python
EXPECTED_STATE = "ghz3"
EXPECTED_QUBITS = 6

# validate_canonical_basis
if circuit.num_qubits != EXPECTED_QUBITS or circuit.num_clbits != 0:
    raise RuntimeError(
        "canonical basis QPY must contain one unmeasured six-qubit circuit"
    )

expected_metadata = {
    "schema": "qoq-reference-basis-v1",
    "state": EXPECTED_STATE,
    "encoding_id": "canonical_ez",
    "num_qubits": EXPECTED_QUBITS,
    "encoding_shape": [4, 3],
    "files": {
        "graph_state_direct_basis.qpy": {"sha256": sha256_file(circuit_path)},
        "E.npy": {"sha256": sha256_file(encoding_path)},
    },
}

# prepare_canonical_basis
expected_encoding = get_encoding("canonical_ez").as_array()
expected_circuit = build_direct_basis_graph_state_circuit(
    EXPECTED_STATE, expected_encoding
)
parent = repo_root / "experiment_inputs" / "reference_bases" / EXPECTED_STATE
directory = parent / "canonical_ez"
```

Keep all existing checks: exact file set, safe `np.load`, finite values, isometry, registry equality, single QPY circuit, no measure/reset/conditions/control flow, statevector equivalence, exact metadata, SHA-256, staging cleanup, and `FileExistsError` race handling. Tag the materialization cell metadata with `canonical-input-materialization`.

- [ ] **Step 4: Add Aer configuration and execution**

```python
SHOTS = 100
UNCERTAINTY = BootstrapConfig(samples=2_000, seed=7)
REFERENCE = get_reference_experiment("ghz3")
RESULTS = {}

AER_SPEC = ExperimentSpec(
    state="ghz3",
    basis=PathBasis(CANONICAL_BASIS_DIRECTORY),
    backend=AerIdeal(seed_simulator=11),
    shots=SHOTS,
    uncertainty=UNCERTAINTY,
    tags={"baseline": "canonical_ez", "backend": "aer_ideal"},
)
AER_RESULT = run_experiment(AER_SPEC, repo_root=REPO_ROOT)
RESULTS["aer_ideal"] = AER_RESULT
```

- [ ] **Step 5: Add one-row summary without recomputation**

```python
def summarize_results(results, reference):
    result = results.get("aer_ideal")
    return [
        {
            "backend": "aer_ideal",
            "status": "not_run" if result is None else result.status.value,
            "raw": None if result is None else result.values.get("raw"),
            "readout_mitigated": None if result is None else result.values.get("readout_mitigated"),
            "zne": None if result is None else result.values.get("zne"),
            "zne_readout_mitigated": None if result is None else result.values.get("zne_readout_mitigated"),
            "diagnostics": None if result is None else result.values.get("diagnostics"),
            "leakage_rate": None if result is None else result.values.get("leakage_rate"),
            "classical_bound": reference.bell_functional.classical_bound,
            "ideal_bell_value": reference.expected.ideal_bell_value,
            "artifact_dir": None if result is None else str(result.artifact_dir),
        }
    ]


SUMMARY = summarize_results(RESULTS, REFERENCE)
SUMMARY
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
```

Expected: all offline tests PASS; connected IQM test SKIP.

- [ ] **Step 7: Tighten local Aer tolerance**

Record deterministic `raw.estimate.real` from seed 11 and 64 shots. Replace `abs=0.2` with the smallest stable tolerance that covers only observed finite-shot deviation from ideal `6.0`, plus a short explanatory comment. Rerun focused tests; expected PASS with one connected test SKIP.

- [ ] **Step 8: Commit GREEN checkpoint**

```powershell
git add notebooks/ghz3_bell_canonical_baseline.ipynb tests/test_ghz3_canonical_baseline_notebook.py
git commit -m "feat: add GHZ3 canonical Aer baseline notebook"
```

Verify commit is reachable from current `HEAD` using `git log -1 --oneline`.

### Task 3: Review and offline regression verification

**Files:**
- Review: `notebooks/ghz3_bell_canonical_baseline.ipynb`
- Review: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Run notebook JSON and syntax checks**

```powershell
python -m json.tool notebooks/ghz3_bell_canonical_baseline.ipynb > $null
python -m py_compile tests/test_ghz3_canonical_baseline_notebook.py
```

Expected: both commands exit 0.

- [ ] **Step 2: Run focused notebook tests with verbose evidence**

```powershell
python -m pytest tests/test_ghz3_canonical_baseline_notebook.py -vv
```

Expected: all offline tests PASS; exactly one connected IQM test SKIP.

- [ ] **Step 3: Run relevant regression tests**

```powershell
python -m pytest tests/test_reference_regressions.py tests/test_reference_experiments.py tests/test_reference_measurement_integration.py tests/test_experiment_runner.py tests/test_experiment_readout.py tests/test_experiment_zne.py tests/test_experiment_iqm_adapter.py -q
```

Expected: all selected tests PASS. If current unrelated user edits cause failure, preserve them and report exact failing test rather than rewriting unrelated code.

- [ ] **Step 4: Review focused diff and notebook hygiene**

```powershell
git diff --check -- notebooks/ghz3_bell_canonical_baseline.ipynb tests/test_ghz3_canonical_baseline_notebook.py
git diff --stat
git status --short
```

Confirm only intended new files belong to this implementation, notebook has no outputs, credentials, absolute user paths, IQM imports, or PiastQ imports, and no existing dirty file was overwritten.

### Task 4: Execute connected IQM Garnet mitigation smoke run

**Files:**
- Test: `tests/test_ghz3_canonical_baseline_notebook.py`
- Generated artifacts: `artifacts/experiment_runs/<date>/<run-id>/`
- Generated canonical input: `experiment_inputs/reference_bases/ghz3/canonical_ez/`

- [ ] **Step 1: Confirm IQM dependency and configuration without exposing secrets**

```powershell
python -c "import importlib.util; print('iqm_provider:', bool(importlib.util.find_spec('qiskit_iqm'))); print('mthree:', bool(importlib.util.find_spec('mthree')))"
```

Expected: both lines report `True`. Do not print environment variable values or provider configuration contents.

- [ ] **Step 2: Run only connected hardware test**

```powershell
$env:QOQ_RUN_IQM_HARDWARE='1'
python -m pytest tests/test_ghz3_canonical_baseline_notebook.py::test_ghz3_canonical_full_pipeline_on_iqm_garnet -vv -s
Remove-Item Env:QOQ_RUN_IQM_HARDWARE
```

Expected: one PASS after IQM jobs complete. The run uses 50 shots per submitted circuit, readout calibration, and ZNE factors 1, 3, 5.

- [ ] **Step 3: Inspect persisted result and report evidence**

Find the newest GHZ3 IQM smoke artifact without printing credentials. Verify:

```python
assert result["status"] == "completed"
assert result["values"]["diagnostics"]["factors"] == [1, 3, 5]
assert result["values"]["raw"] is not None
assert result["values"]["readout_mitigated"] is not None
assert result["values"]["zne"] is not None
assert result["values"]["zne_readout_mitigated"] is not None
```

Report run/job identifier, artifact directory, status, raw estimate, readout-mitigated estimate, ZNE estimate, and ZNE+readout estimate. If provider configuration or hardware availability blocks submission, report shortest decisive error and retain all offline GREEN evidence.

- [ ] **Step 4: Final fresh verification**

Rerun focused offline tests after connected execution:

```powershell
python -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
```

Expected: all offline tests PASS and connected test SKIP with environment gate removed.
