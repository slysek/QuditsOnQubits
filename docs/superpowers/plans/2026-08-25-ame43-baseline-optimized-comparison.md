# AME43 Baseline vs Exact-Optimized Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a PR to `main` containing the exact AME43 circuit optimization and a new reproducible notebook comparing it with the canonical baseline.

**Architecture:** Keep the baseline and optimized builders/notebooks independent. Add one comparison notebook that materializes both bundles, proves equivalence, runs both through Aer, reports common-basis compiler metrics, and offers separately guarded IQM compilation and raw 50-shot hardware comparison. Rebase the scoped implementation onto current `origin/main`, then validate and review the complete diff before creating the PR.

**Tech Stack:** Python 3.12, NumPy, Qiskit/QPY, IQM Qiskit adapter, Jupyter/nbconvert, pytest, Git/GitHub CLI.

---

### Task 1: Stabilize standalone AME43 notebooks for worktrees

**Files:**
- Modify: `notebooks/ame43_canonical_baseline.ipynb`
- Modify: `notebooks/ame43_canonical_exact_optimized.ipynb`
- Modify: `tests/test_ame43_canonical_baseline_notebook.py`
- Modify: `tests/test_ame43_canonical_exact_optimized_notebook.py`

- [ ] **Step 1: Add failing worktree `.env` and balanced-seed tests**

Add assertions equivalent to:

```python
assert "resolve_iqm_env_path" in full_source
assert "env_path=IQM_ENV_PATH" in full_source
assert "TranspilationConfig(optimization_level=3, seed_transpiler=13)" in optimized_source
```

- [ ] **Step 2: Run RED**

Run with the worktree source forced first:

```powershell
$env:PYTHONPATH='<worktree>\src'
python -m pytest tests/test_ame43_canonical_baseline_notebook.py tests/test_ame43_canonical_exact_optimized_notebook.py -q
```

Expected: failures because neither notebook passes an explicit worktree-aware
`.env` path and optimized still uses seed `6`.

- [ ] **Step 3: Add the minimal resolver to both notebooks**

Use this behavior in a setup cell:

```python
def resolve_iqm_env_path(repo_root):
    repo_root = Path(repo_root).resolve()
    candidates = [repo_root / ".env"]
    if repo_root.parent.name == ".worktrees":
        candidates.append(repo_root.parent.parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("IQM .env file was not found in the checkout or owning repository")

IQM_ENV_PATH = resolve_iqm_env_path(REPO_ROOT)
```

Construct IQM hardware as:

```python
IQMHardware(device="garnet", use_metrics=True, env_path=IQM_ENV_PATH)
```

The resolver must execute only inside guarded IQM cells so offline `Run All`
does not require `.env`. Change the optimized shared seed to `13`.

- [ ] **Step 4: Run GREEN and local Aer**

Expected: both notebook suites pass; source notebooks remain unexecuted; Aer
exact-optimized estimate remains near `8`.

### Task 2: Create the comparison notebook with TDD

**Files:**
- Create: `notebooks/ame43_canonical_comparison.ipynb`
- Create: `tests/test_ame43_canonical_comparison_notebook.py`

- [ ] **Step 1: Write the failing structural test**

The test must require:

```python
assert NOTEBOOK_PATH.is_file()
assert "RUN_IQM_COMPILE = False" in full_source
assert "RUN_IQM_HARDWARE = False" in full_source
assert "HARDWARE_SHOTS = 50" in full_source
assert "IQM_SEED = 13" in full_source
assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/test_ame43_canonical_comparison_notebook.py -q
```

Expected: failure because the comparison notebook does not exist.

- [ ] **Step 3: Build the offline comparison path**

The notebook must create baseline and optimized circuits with:

```python
baseline = build_direct_basis_graph_state_circuit("ame43", encoding)
optimized = build_exact_optimized_direct_basis_graph_state_circuit("ame43", encoding)
fidelity = abs(Statevector.from_instruction(baseline).inner(
    Statevector.from_instruction(optimized)
)) ** 2
```

For each preparation and each of the 13 measured circuits, compile identically:

```python
transpile(circuit, basis_gates=["u", "cz"], optimization_level=3, seed_transpiler=0)
```

Produce comparison rows containing variant, logical depth, preparation CZ/depth,
measurement CZ/depth total/min/max, Aer Bell estimate, classical bound, ideal
value, and state fidelity.

- [ ] **Step 4: Add guarded IQM compilation and hardware cells**

Read-only compilation uses one Garnet target and seed `13`. Hardware specs use:

```python
ExperimentSpec(
    state="ame43",
    basis=PathBasis(bundle),
    backend=IQMHardware(device="garnet", use_metrics=True, env_path=IQM_ENV_PATH),
    shots=HARDWARE_SHOTS,
    mitigation=MitigationConfig(readout=False, zne=False),
    transpilation=TranspilationConfig(optimization_level=3, seed_transpiler=IQM_SEED),
)
```

Both guards are false in source. No low-level provider or `backend.run` call is
allowed.

- [ ] **Step 5: Add semantic and safety tests**

Tests must prove:

```python
assert Statevector.from_instruction(baseline).equiv(Statevector.from_instruction(optimized))
assert optimized_native.count_ops()["cz"] < baseline_native.count_ops()["cz"]
assert optimized_native.depth() < baseline_native.depth()
assert baseline_aer == pytest.approx(8.0, abs=0.3)
assert optimized_aer == pytest.approx(8.0, abs=0.3)
```

Also reject outputs, credentials, absolute user paths, low-level IQM clients,
and enabled hardware guards.

- [ ] **Step 6: Run GREEN and notebook `Run All`**

Execute source to an ignored artifact with IQM guards false. Expected: complete
without `.env`, both Aer values near `8`, and optimized relative cost lower.

### Task 3: Validate IQM comparison and commit scoped implementation

**Files:**
- Generated only under ignored `artifacts/`
- Commit only scoped source/notebook/test files

- [ ] **Step 1: Run read-only Garnet compilation**

Enable only `RUN_IQM_COMPILE` in a temporary executed copy. Compare both batches
with seed `13`; record per-circuit and aggregate CZ/depth without submitting jobs.

- [ ] **Step 2: Run optional raw hardware smoke**

If credentials and Garnet availability succeed, execute baseline and optimized
with `50` shots, no mitigation. Record job IDs/results only in ignored artifacts.
Immediately restore source guards false.

- [ ] **Step 3: Run focused regressions**

Run the baseline, optimized, comparison, preparation, reference, IQM adapter,
runner, and Aer integration tests with worktree `PYTHONPATH`. Expect all pass.

- [ ] **Step 4: Commit only implementation files**

Stage explicit paths; exclude `experiment_inputs/reference_bases/`, executed
notebooks, seed-sweep output, `.env`, and unrelated files. Commit:

```text
feat: compare baseline and optimized AME43 circuits
```

### Task 4: Rebase, final quality gate, and PR

**Files:**
- Complete branch diff against `origin/main`

- [ ] **Step 1: Fetch and rebase**

```powershell
git fetch origin main
git rebase origin/main
```

Resolve only conflicts in scoped files. Never discard unrelated repository
changes.

- [ ] **Step 2: Re-run complete validation after rebase**

Repeat focused suites and notebook `Run All`; run `git diff --check` and verify
source guards false/unexecuted.

- [ ] **Step 3: Mandatory reviewer loop**

Review the complete `origin/main...HEAD` diff. Fix valid P0-P2 findings, rerun
validation, and request a fresh full review until `CLEAN` or three rounds.

- [ ] **Step 4: Push and create PR**

```powershell
git push -u origin codex/ame43-exact-optimization
gh pr create --base main --head codex/ame43-exact-optimization
```

PR body must summarize exact equivalence, CZ/depth reductions, comparison
notebook, Aer results, optional IQM results, tests, and safety guards.
