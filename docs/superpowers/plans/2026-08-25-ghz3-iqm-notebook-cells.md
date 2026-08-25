# GHZ3 IQM Notebook Cells Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in IQM Garnet execution section with readout mitigation and ZNE to the GHZ3 canonical notebook in pull request #14.

**Architecture:** Follow the established two-qutrit notebook layout. Keep Aer unguarded, place IQM behind `RUN_IQM = False`, share shots, bootstrap, basis, and mitigation configuration, then report both backends through the existing summary function.

**Tech Stack:** Python 3.12, Jupyter notebook JSON, Qiskit, qudits_on_qubits experiment API, pytest, Python AST assertions.

---

## File map

- Modify `notebooks/ghz3_bell_canonical_baseline.ipynb`: add the user-facing IQM configuration, guarded execution, and combined summary.
- Modify `tests/test_ghz3_canonical_baseline_notebook.py`: define and verify the notebook contract without submitting hardware jobs.

### Task 1: Define the opt-in IQM notebook contract

**Files:**
- Modify: `tests/test_ghz3_canonical_baseline_notebook.py`
- Test: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Change the structural test to require Aer and IQM runs**

Replace the Aer-only expectations with:

```python
def test_notebook_has_aer_and_opt_in_iqm_runs_and_clean_cells():
    notebook = load_notebook()
    cells = code_cells(notebook)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 2
    assert {
        runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0])
        for cell in run_cells
    } == {"AerIdeal", "IQMHardware"}
    assert all(
        len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells
    )
    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo) == 1
        assert isinstance(repo[0].value, ast.Name)
        assert repo[0].value.id == "REPO_ROOT"

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
        "run_experiment",
    } <= imported
    assert "PiastQHardware" not in imported
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)
```

- [ ] **Step 2: Add an executable default-skip test**

```python
def test_iqm_cell_is_opt_in_and_does_not_submit_by_default():
    namespace = setup_namespace(REPO_ROOT)
    iqm_cell = next(
        cell
        for cell in code_cells(load_notebook())
        if any(
            runner_backend(source(cell), call) == "IQMHardware"
            for call in named_calls(source(cell), "run_experiment")
        )
    )
    submissions = []
    namespace["run_experiment"] = lambda *args, **kwargs: submissions.append(
        (args, kwargs)
    )

    exec(compile(source(iqm_cell), str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["RUN_IQM"] is False
    assert submissions == []
```

- [ ] **Step 3: Extend exact configuration assertions**

After the shared configuration assertions, require:

```python
    assert namespace["HARDWARE_MITIGATION"].readout is True
    assert namespace["HARDWARE_MITIGATION"].zne is True
    assert namespace["HARDWARE_MITIGATION"].zne_factors == (1, 3, 5)
```

Locate the IQM `ExperimentSpec` by its `IQM_SPEC` assignment and assert:

```python
    iqm_keywords = {keyword.arg: keyword.value for keyword in iqm_spec.keywords}
    assert ast.literal_eval(iqm_keywords["state"]) == "ghz3"
    assert isinstance(iqm_keywords["backend"], ast.Call)
    assert isinstance(iqm_keywords["backend"].func, ast.Name)
    assert iqm_keywords["backend"].func.id == "IQMHardware"
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in iqm_keywords["backend"].keywords
    } == {"device": "garnet", "use_metrics": True}
    assert isinstance(iqm_keywords["shots"], ast.Name)
    assert iqm_keywords["shots"].id == "SHOTS"
    assert isinstance(iqm_keywords["mitigation"], ast.Name)
    assert iqm_keywords["mitigation"].id == "HARDWARE_MITIGATION"
    assert isinstance(iqm_keywords["uncertainty"], ast.Name)
    assert iqm_keywords["uncertainty"].id == "UNCERTAINTY"
    assert ast.literal_eval(iqm_keywords["tags"]) == {
        "baseline": "canonical_ez",
        "backend": "iqm_garnet",
    }
```

Change the empty summary expectations to:

```python
    summary = namespace["summarize_results"]({}, namespace["REFERENCE"])
    assert [row["backend"] for row in summary] == ["aer_ideal", "iqm_garnet"]
    assert [row["status"] for row in summary] == ["not_run", "skipped"]
```

- [ ] **Step 4: Extend summary preservation to IQM**

Create distinct Aer and IQM result objects and assert each row preserves the
runner values:

```python
    aer_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ghz3-aer",
        values=values,
    )
    iqm_values = {
        **values,
        "raw": {"estimate": 4.3},
        "zne_readout_mitigated": {"estimate": 5.1},
    }
    iqm_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ghz3-iqm",
        values=iqm_values,
    )
    rows = namespace["summarize_results"](
        {"aer_ideal": aer_result, "iqm_garnet": iqm_result},
        namespace["REFERENCE"],
    )
    assert [row["backend"] for row in rows] == ["aer_ideal", "iqm_garnet"]
    assert rows[0]["raw"] == values["raw"]
    assert rows[1]["raw"] == iqm_values["raw"]
    assert rows[1]["zne_readout_mitigated"] == iqm_values[
        "zne_readout_mitigated"
    ]
    assert rows[1]["artifact_dir"] == str(iqm_result.artifact_dir)
```

- [ ] **Step 5: Update the safety assertion**

Rename the safety test to
`test_notebook_has_no_secrets_paths_provider_or_low_level_execution`. Keep
`IQMProvider(` forbidden, but remove the `IQMHardware` string from the
forbidden list because the high-level backend is now required.

- [ ] **Step 6: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Join-Path $PWD 'src')
& 'C:\Users\szymon\.conda\envs\QuditsOnQubitsEnv\python.exe' -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
Remove-Item Env:PYTHONPATH
```

Expected: failures showing one run instead of two, missing `IQMHardware`,
missing `HARDWARE_MITIGATION`, one-row summary, and no IQM run cell.

### Task 2: Add the guarded IQM Garnet cells

**Files:**
- Modify: `notebooks/ghz3_bell_canonical_baseline.ipynb`
- Test: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Update notebook introduction and imports**

Use this introduction:

```markdown
# Canonical GHZ3 Bell baseline

This notebook prepares a deterministic `canonical_ez` direct-basis input for
the three-qutrit GHZ Bell experiment. By default, Run All submits only the
local Aer baseline; IQM submits a remote job only after `RUN_IQM` is set to
`True`. Credentials remain provider/environment-only and are never embedded
or persisted by this notebook.
```

Add these imports from `qudits_on_qubits.experiments`:

```python
    IQMHardware,
    MitigationConfig,
```

- [ ] **Step 2: Define shared hardware mitigation**

Rename the configuration heading to `## Shared configuration` and use:

```python
SHOTS = 100
UNCERTAINTY = BootstrapConfig(samples=2_000, seed=7)
HARDWARE_MITIGATION = MitigationConfig(
    readout=True,
    zne=True,
    zne_factors=(1, 3, 5),
)
REFERENCE = get_reference_experiment("ghz3")
RESULTS = {}
```

- [ ] **Step 3: Add the opt-in IQM section after Aer**

Add a markdown cell:

```markdown
## IQM Garnet baseline

Submission is opt-in; the default keeps this hardware run skipped.
```

Add a clean code cell:

```python
RUN_IQM = False

if RUN_IQM:
    IQM_SPEC = ExperimentSpec(
        state="ghz3",
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
    print("IQM Garnet skipped; set RUN_IQM = True to submit.")
```

The cell must have `execution_count: null` and `outputs: []`.

- [ ] **Step 4: Expand the summary**

Replace the single-result implementation with:

```python
def summarize_results(results, reference):
    rows = []
    for backend, missing_status in (
        ("aer_ideal", "not_run"),
        ("iqm_garnet", "skipped"),
    ):
        result = results.get(backend)
        rows.append(
            {
                "backend": backend,
                "status": (
                    missing_status if result is None else result.status.value
                ),
                "raw": None if result is None else result.values.get("raw"),
                "readout_mitigated": (
                    None
                    if result is None
                    else result.values.get("readout_mitigated")
                ),
                "zne": None if result is None else result.values.get("zne"),
                "zne_readout_mitigated": (
                    None
                    if result is None
                    else result.values.get("zne_readout_mitigated")
                ),
                "diagnostics": (
                    None if result is None else result.values.get("diagnostics")
                ),
                "leakage_rate": (
                    None if result is None else result.values.get("leakage_rate")
                ),
                "classical_bound": (
                    reference.bell_functional.classical_bound
                ),
                "ideal_bell_value": reference.expected.ideal_bell_value,
                "artifact_dir": (
                    None if result is None else str(result.artifact_dir)
                ),
            }
        )
    return rows
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same focused command from Task 1.

Expected: `12 passed, 1 skipped`, with only the environment-gated hardware
test skipped.

- [ ] **Step 6: Commit the notebook change**

```powershell
git add notebooks/ghz3_bell_canonical_baseline.ipynb tests/test_ghz3_canonical_baseline_notebook.py
git commit -m "feat: add opt-in IQM cells to GHZ3 notebook"
```

### Task 3: Validate and update pull request #14

**Files:**
- Verify: `notebooks/ghz3_bell_canonical_baseline.ipynb`
- Verify: `tests/test_ghz3_canonical_baseline_notebook.py`

- [ ] **Step 1: Run the focused pipeline suite**

```powershell
$env:PYTHONPATH=(Join-Path $PWD 'src')
& 'C:\Users\szymon\.conda\envs\QuditsOnQubitsEnv\python.exe' -m pytest tests/test_experiment_preparation.py tests/test_experiment_readout.py tests/test_experiment_runner.py tests/test_two_qutrit_canonical_baseline_notebook.py tests/test_ghz3_canonical_baseline_notebook.py -q
Remove-Item Env:PYTHONPATH
```

Expected: zero failures; the IQM connected test remains skipped unless
`QOQ_RUN_IQM_HARDWARE=1`.

- [ ] **Step 2: Run repository regression tests**

```powershell
$env:PYTHONPATH=(Join-Path $PWD 'src')
& 'C:\Users\szymon\.conda\envs\QuditsOnQubitsEnv\python.exe' -m pytest tests -q
Remove-Item Env:PYTHONPATH
```

Expected: zero failures.

- [ ] **Step 3: Verify notebook cleanliness and diff**

```powershell
git diff --check origin/main...HEAD
git status --short
```

Verify every code cell has `execution_count: null`, every output list is
empty, and only the ignored generated reference-basis directory remains
untracked.

- [ ] **Step 4: Run independent review**

Ask the repository reviewer to inspect the complete updated diff against
`origin/main`. Fix every valid P0, P1, and P2 finding, rerun relevant tests,
and repeat review until `CLEAN` or the three-round quality-loop limit.

- [ ] **Step 5: Push the updated branch**

```powershell
git push origin codex/ghz3-canonical-aer-baseline-v2
gh pr view 14 --json url,state,mergeable,headRefName,baseRefName
```

Expected: pull request #14 remains open, targets `main`, and contains the
new design, plan, notebook, and test commits.
