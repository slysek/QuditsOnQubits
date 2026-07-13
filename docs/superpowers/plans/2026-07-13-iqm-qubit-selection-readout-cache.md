# IQM Qubit Selection and Readout Calibration Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Garnet GHZ notebook choose a fresh best layout per candidate and persist M3 readout matrices so only missing calibrations require hardware execution.

**Architecture:** Keep the workflow in the notebook, with one helper for IQM selection/transpilation and one group of helpers for calibration caching. Tests load only the relevant notebook cells and inject fake collaborators, so verification cannot connect to IQM or submit hardware jobs.

**Tech Stack:** Python 3.10+, Jupyter notebook JSON, NumPy, Qiskit, IQM Client 34, IQM Qubit Selector 1, `unittest`

---

## File Structure

- Modify: `notebooks/working/iqm/best_garnet_ghz.ipynb` - authentication, selector, cache, and per-candidate pipeline.
- Create: `tests/test_best_garnet_ghz_notebook.py` - isolated notebook-cell tests using fake backends.
- Modify: `pyproject.toml` - package dependency.
- Modify: `requirements.txt` - notebook environment dependency.

### Task 1: Readout cache contract

**Files:**
- Create: `tests/test_best_garnet_ghz_notebook.py`
- Modify: `notebooks/working/iqm/best_garnet_ghz.ipynb` (imports and calibration cells)

- [ ] **Step 1: Write the notebook-cell loader and failing cache-hit test**

Create this loader, which reads notebook JSON without executing the notebook:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "working" / "iqm" / "best_garnet_ghz.ipynb"


def load_cell_namespace(marker, **injected):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )
    namespace = {"np": np, "QuantumCircuit": QuantumCircuit, **injected}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace
```

Add `test_cache_hit_does_not_run_backend`. Write a versioned JSON cache for backend `garnet` and qubit `2`, call `build_readout_calibration_matrices`, then assert `backend.run.assert_not_called()` and `np.testing.assert_allclose(matrices[2], expected)`.

- [ ] **Step 2: Run the test and verify RED**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook.ReadoutCalibrationCacheTests.test_cache_hit_does_not_run_backend -v
```

Expected: FAIL because the notebook function does not accept `cache_path`.

- [ ] **Step 3: Implement cache loading and matrix validation**

Add imports `json`, `os`, and `datetime, timezone`. Define `READOUT_CACHE_VERSION = 1`, a default path under `repo_root / "artifacts" / "iqm_runs" / "calibration"`, and these helpers:

```python
def _backend_cache_key(backend):
    name = backend.name
    return str(name() if callable(name) else name)


def _load_readout_cache(cache_path):
    if not cache_path.exists():
        return {"version": READOUT_CACHE_VERSION, "backends": {}}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read readout calibration cache: {cache_path}") from exc
    if payload.get("version") != READOUT_CACHE_VERSION or not isinstance(payload.get("backends"), dict):
        raise ValueError(f"Invalid readout calibration cache: {cache_path}")
    return payload


def _matrix_from_cache(entry, backend_key, qubit):
    try:
        matrix = np.asarray(entry["matrix"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cached matrix for {backend_key} qubit {qubit}") from exc
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid cached matrix for {backend_key} qubit {qubit}")
    return matrix
```

Extend `build_readout_calibration_matrices` with `cache_path=None` and `force_recalibration=False`. Load valid requested entries before constructing calibration circuits.

- [ ] **Step 4: Run the cache-hit test and verify GREEN**

Run Step 2 again. Expected: PASS and zero backend calls.

- [ ] **Step 5: Write failing miss and cache-safety tests**

Use a fake result returning `[{'0': 90, '1': 10}, {'0': 20, '1': 80}]`. Add separate tests for cache miss, partial hit, different backend name, forced refresh, malformed JSON, and a wrong-shaped matrix. Assert calibrated matrices equal `[[0.9, 0.2], [0.1, 0.8]]`; malformed data must raise before `backend.run`.

- [ ] **Step 6: Run the cache suite and verify RED**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook.ReadoutCalibrationCacheTests -v
```

Expected: FAIL because new calibrations are not persisted.

- [ ] **Step 7: Implement missing-only calibration and atomic persistence**

Store each successful measurement as:

```python
backend_cache[str(q)] = {
    "matrix": matrix.tolist(),
    "shots": shots,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
```

Write only after successful calibration and atomically replace the cache:

```python
temporary_path = cache_path.with_name(f"{cache_path.name}.tmp")
temporary_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary_path, cache_path)
```

Keep the M3-compatible list return type and `verbose` output.

- [ ] **Step 8: Run the cache suite and verify GREEN**

Run Step 6 again. Expected: all tests PASS using only mock backends.

- [ ] **Step 9: Commit the cache behavior**

```powershell
git add -- tests/test_best_garnet_ghz_notebook.py notebooks/working/iqm/best_garnet_ghz.ipynb
git commit -m "feat: cache IQM readout calibration matrices"
```

### Task 2: Per-candidate IQM Qubit Selector

**Files:**
- Modify: `tests/test_best_garnet_ghz_notebook.py`
- Modify: `notebooks/working/iqm/best_garnet_ghz.ipynb` (imports, selector helper, pipeline, candidate loop)

- [ ] **Step 1: Write failing selector helper tests**

Inject fake `CostEvaluator` and `perform_backend_transpilation`. Assert the helper returns layout `[2, 5, 7, 8, 11, 13]` and cost `0.031`, scores the logical state circuit, reduces the coupling map with that layout, and transpiles the complete sampler batch with `qiskit_optim_level=3`. Add a no-layout case that raises `RuntimeError` containing the candidate name before transpilation.

- [ ] **Step 2: Run selector tests and verify RED**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook.QubitSelectorTests -v
```

Expected: FAIL because `select_and_transpile_candidate` does not exist.

- [ ] **Step 3: Add official selector imports and helper**

```python
from iqm.qubit_selector.qubit_selector import CostEvaluator
from iqm.qubit_selector.qiskit_utils import perform_backend_transpilation


def select_and_transpile_candidate(backend, logical_state_circuit, sampler_circuits, candidate):
    layouts, costs = CostEvaluator(
        backend=backend,
        quantum_circuit=logical_state_circuit,
    ).get_top_layouts(num_layouts=1)
    if not layouts:
        raise RuntimeError(f"IQM Qubit Selector returned no valid layout for {candidate}")
    best_layout = list(layouts[0])
    transpiled = perform_backend_transpilation(
        sampler_circuits,
        backend,
        best_layout,
        backend.coupling_map.reduce(mapping=best_layout),
        qiskit_optim_level=3,
    )
    return transpiled, best_layout, float(costs[0])
```

- [ ] **Step 4: Run selector tests and verify GREEN**

Run Step 2 again. Expected: all selector tests PASS.

- [ ] **Step 5: Write a failing static pipeline contract test**

Read all notebook source and assert the literal token is absent, provider construction has no `token=`, the pipeline calls `select_and_transpile_candidate`, rows include `selected_layout` and `selector_cost`, and the final loop calls `full_pipeline(backend_garnet, testqc, Esup` rather than `full_pipeline(garnet, qcsuptrans`.

- [ ] **Step 6: Run the contract test and verify RED**

```powershell
$env:PYTHONDWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook.NotebookPipelineContractTests -v
```

Expected: FAIL on the exposed token and fake-backend pipeline call.

- [ ] **Step 7: Rewire the pipeline without executing it**

Remove the token argument from `IQMProvider`. Make `full_pipeline` accept the logical `testqc`, `Esup`, and `candidate`. Build sampler circuits using logical qutrit pairs `((0, 1), (2, 3), (4, 5))`, then select and transpile the entire sampler batch once. Add candidate, layout, and selector cost to each row. Update the final loop to:

```python
rows = []
for candidate in ghz_best_list:
    testqc, _, F3sup, Esup = load_candidate(candidate)
    rows = full_pipeline(backend_garnet, testqc, Esup, candidate=candidate, rows=rows)
```

Do not execute this cell; it remains the user's explicit hardware action.

- [ ] **Step 8: Run notebook tests and verify GREEN**

```powershell
$env:PYTHONDWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook -v
```

Expected: all tests PASS without provider construction or hardware access.

- [ ] **Step 9: Commit selector integration**

```powershell
git add -- tests/test_best_garnet_ghz_notebook.py notebooks/working/iqm/best_garnet_ghz.ipynb
git commit -m "feat: select Garnet layout per GHZ candidate"
```

### Task 3: Dependencies and offline verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `tests/test_best_garnet_ghz_notebook.py`

- [ ] **Step 1: Write a failing dependency test**

Assert both dependency files contain `iqm-qubit-selector>=1,<2`.

- [ ] **Step 2: Run the test and verify RED**

```powershell
$env:PYTHONDWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook.DependencyContractTests -v
```

Expected: FAIL because the requirement is absent.

- [ ] **Step 3: Declare the dependency**

Add `"iqm-qubit-selector>=1,<2",` to `pyproject.toml` and `iqm-qubit-selector>=1,<2` to `requirements.txt`, after IQM Client.

- [ ] **Step 4: Run the dependency test and verify GREEN**

Run Step 2 again. Expected: PASS.

- [ ] **Step 5: Validate notebook JSON and compile code cells without execution**

```powershell
@'
import json
from pathlib import Path
path = Path("notebooks/working/iqm/best_garnet_ghz.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
print(f"validated {len(notebook['cells'])} cells")
'@ | python -
```

Expected: validation count with exit code 0. Source is compiled, never executed.

- [ ] **Step 6: Run focused regression tests**

```powershell
$env:PYTHONDWRITEBYTECODE='1'; python -m unittest tests.test_best_garnet_ghz_notebook tests.test_clean_repo_smoke -v
```

Expected: all tests PASS without network or hardware access.

- [ ] **Step 7: Inspect secrets and notebook churn**

```powershell
git diff --check
git diff --stat
git diff -- notebooks/working/iqm/best_garnet_ghz.ipynb pyproject.toml requirements.txt tests/test_best_garnet_ghz_notebook.py
```

Expected: no IQM token, unrelated output churn, or whitespace errors.

- [ ] **Step 8: Commit dependency and verification changes**

```powershell
git add -- pyproject.toml requirements.txt tests/test_best_garnet_ghz_notebook.py
git commit -m "test: cover Garnet notebook hardware safeguards"
```

## Safety Invariant

No step executes the notebook, constructs the real `IQMProvider`, calls `backend_garnet.run`, or submits circuits to Garnet. Every exercised `backend.run` belongs to a local `Mock`.
