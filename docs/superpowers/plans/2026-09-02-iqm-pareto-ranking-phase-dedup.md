# IQM Pareto Ranking and Global-Phase Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate projectively equivalent direct-basis candidates before IQM transpilation, then use post-transpilation seed statistics, Pareto layers, and the approved 50/30/20 ideal-point score to recommend the best compiled circuits.

**Architecture:** Keep `run_iqm_transpiler_harness` as the compilation boundary and preserve its three-value return contract. A new projective-equivalence module filters candidate matrices before the harness loop and stores its audit rows in `all_trials.attrs`. A pure DataFrame analysis module aggregates successful seeds per candidate and strategy, assigns Pareto layers and deterministic scores, and delegates QPY-based logical-state grouping to a focused state-equivalence module. The harness writer and a standalone post-processing CLI emit the new tables while preserving the existing CSVs and summary keys.

**Tech Stack:** Python 3.11+, NumPy, pandas, Qiskit/QPY, `unittest`-style pytest tests, PowerShell test commands.

---

## Working conventions

- Work only in `C:\Users\szymo\QuditsOnQubits\QuditsOnQubits\.worktrees\pareto-phase-dedup` on branch `feature/pareto-phase-dedup`.
- Use this Git form because the worktree needs an explicit safe-directory declaration:

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup status --short
```

- Disable unrelated third-party pytest plugin autoload for every focused test command:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_iqm_transpiler_harness.py
```

- Treat the approved design as normative: `docs/superpowers/specs/2026-09-02-iqm-pareto-ranking-phase-dedup-design.md`.
- Keep `best_by_candidate.csv` depth-first. The new two-qubit-first concrete-trial choice belongs only to the Pareto analysis.

## Task 1: Add projective matrix equivalence and deterministic candidate grouping

**Files:**

- Create: `src/qudits_on_qubits/benchmarks/direct_basis/phase_equivalence.py`
- Create: `tests/test_direct_basis_phase_equivalence.py`

- [ ] **Step 1: Write failing phase-equivalence unit tests**

Add tests for exact equality, arbitrary unit-modulus phase, a relative-phase mismatch, deterministic baseline preference, lexicographic fallback, unsupported candidates remaining independent, and stable audit rows:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.phase_equivalence import (
    deduplicate_candidates_up_to_global_phase,
    global_phase_between,
)


def _candidate(class_name: str, name: str, matrix: np.ndarray | None):
    return DirectBasisCandidate(
        name=name,
        candidate_type=class_name,
        matrix=matrix,
        source_class_name=class_name,
        source_candidate_name=name,
        error_message="unsupported" if matrix is None else "",
    )


class PhaseEquivalenceTests(unittest.TestCase):
    def test_global_phase_between_accepts_arbitrary_phase(self):
        reference = np.array([[1, 0], [0, 1j]], dtype=complex)
        phase = np.exp(0.37j)

        detected = global_phase_between(reference, phase * reference)

        self.assertIsNotNone(detected)
        self.assertAlmostEqual(abs(detected), 1.0)
        np.testing.assert_allclose(phase * reference, detected * reference)

    def test_global_phase_between_rejects_relative_phase_change(self):
        reference = np.eye(2, dtype=complex)
        changed = np.diag([1.0, 1j]).astype(complex)

        self.assertIsNone(global_phase_between(reference, changed))

    def test_dedup_prefers_baseline_and_emits_audit_row(self):
        matrix = np.eye(3, dtype=complex)
        result = deduplicate_candidates_up_to_global_phase(
            [
                _candidate("z_class", "copy", -1j * matrix),
                _candidate("baseline", "E_old", matrix),
                _candidate("unsupported", "missing", None),
            ]
        )

        identities = [(item.class_name, item.candidate_name) for item in result.representatives]
        self.assertEqual(
            identities,
            [("baseline", "E_old"), ("unsupported", "missing")],
        )
        self.assertEqual(len(result.duplicate_rows), 1)
        row = result.duplicate_rows[0]
        self.assertEqual(row["representative_class_name"], "baseline")
        self.assertEqual(row["representative_candidate_name"], "E_old")
        self.assertEqual(row["duplicate_class_name"], "z_class")
        self.assertEqual(row["duplicate_candidate_name"], "copy")
        self.assertEqual(row["reason"], "global_phase_equivalent_matrix")
        self.assertAlmostEqual(row["detected_phase_real"], 0.0)
        self.assertAlmostEqual(row["detected_phase_imag"], -1.0)

    def test_dedup_uses_lexicographically_smallest_nonbaseline_identity(self):
        matrix = np.eye(3, dtype=complex)
        result = deduplicate_candidates_up_to_global_phase(
            [
                _candidate("z_class", "later", matrix),
                _candidate("a_class", "first", -matrix),
            ]
        )

        self.assertEqual(len(result.representatives), 1)
        self.assertEqual(result.representatives[0].class_name, "a_class")
        self.assertEqual(result.representatives[0].candidate_name, "first")
```

- [ ] **Step 2: Run the new test and verify the import failure**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_phase_equivalence.py
```

Expected: collection fails with `ModuleNotFoundError` for `phase_equivalence`.

- [ ] **Step 3: Implement the projective-equivalence API**

Create these public types and constants:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate


DEFAULT_PHASE_ATOL = 1e-9
DEFAULT_PHASE_RTOL = 1e-7
DEFAULT_BUCKET_DECIMALS = 9

PHASE_DUPLICATE_COLUMNS = (
    "global_phase_group_id",
    "representative_class_name",
    "representative_candidate_name",
    "duplicate_class_name",
    "duplicate_candidate_name",
    "detected_phase_real",
    "detected_phase_imag",
    "reason",
)


@dataclass(frozen=True)
class CandidatePhaseDeduplication:
    representatives: Sequence[DirectBasisCandidate]
    duplicate_rows: Sequence[dict[str, object]]

    @property
    def removed_count(self) -> int:
        return len(self.duplicate_rows)
```

Implement `global_phase_between(reference, candidate, *, atol, rtol) -> complex | None` with these exact rules:

1. Convert the inputs with `np.asarray(reference, dtype=complex)` and `np.asarray(candidate, dtype=complex)`, then reject different shapes.
2. For two all-zero arrays return `1 + 0j`; if exactly one array is all-zero return `None`.
3. Use the largest-magnitude reference element as the pivot.
4. Compute `phase = candidate[pivot] / reference[pivot]`, normalize it to unit magnitude, and reject a zero candidate pivot.
5. Return the phase only when both `abs(abs(phase) - 1.0) <= atol + rtol` and `np.allclose(candidate, phase * reference, atol=atol, rtol=rtol)` hold.

Implement the canonical bucket key so it is only an acceleration hint:

```python
def _canonical_bucket_key(
    matrix: np.ndarray,
    *,
    atol: float,
    decimals: int,
) -> tuple[object, object]:
    value = np.asarray(matrix, dtype=complex)
    flat = value.reshape(-1)
    pivot_index = int(np.argmax(np.abs(flat)))
    pivot = flat[pivot_index]
    if abs(pivot) > atol:
        value = value * np.exp(-1j * np.angle(pivot))
    real = np.real(value).copy()
    imag = np.imag(value).copy()
    real[np.abs(real) <= atol] = 0.0
    imag[np.abs(imag) <= atol] = 0.0
    rounded = tuple(zip(np.round(real, decimals).flat, np.round(imag, decimals).flat))
    return (value.shape, rounded)
```

Implement `deduplicate_candidates_up_to_global_phase` by sorting supported candidates with this representative key before grouping:

```python
def _representative_key(candidate: DirectBasisCandidate) -> tuple[int, str, str]:
    return (
        0 if candidate.class_name == "baseline" else 1,
        candidate.class_name,
        candidate.candidate_name,
    )
```

Use the canonical key to find candidate groups, but confirm every merge with `global_phase_between`. Keep unsupported candidates as independent representatives. Number verified groups as `global_phase_0001`, `global_phase_0002`, and so on after sorting representatives by `_representative_key`. Emit one audit record for every removed candidate with phase defined by `duplicate = detected_phase * representative`.

- [ ] **Step 4: Add a generator-level regression for current monomial phase triples**

Extend the test file with a test that imports `generate_extended_legacy_candidates`, filters `class_name == "monomial_full"`, runs the deduplicator, and asserts:

```python
self.assertEqual(len(raw), 648)
self.assertEqual(len(result.representatives), 216)
self.assertEqual(result.removed_count, 432)
```

If the generator has one baseline-equivalent entry that is filtered elsewhere, keep this module-level test scoped strictly to the 648 `monomial_full` entries. The expected 216 equivalence classes follow from removing the three global cubic-root phases per projective matrix.

- [ ] **Step 5: Run focused tests to green**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_phase_equivalence.py
```

Expected: all phase-equivalence tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/phase_equivalence.py tests/test_direct_basis_phase_equivalence.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: deduplicate candidates up to global phase"
```

## Task 2: Apply phase deduplication before the IQM harness candidate loop

**Files:**

- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py`
- Modify: `tests/test_direct_basis_iqm_transpiler_harness.py`

- [ ] **Step 1: Write a failing harness integration test**

Add a test with baseline `I`, a phase-shifted copy, and a distinct matrix. Record the candidate identities received by the fake strategy runner and assert:

```python
self.assertEqual(len(all_trials), 2)
self.assertEqual(summary["candidate_count"], 3)
self.assertEqual(summary["representative_candidate_count"], 2)
self.assertEqual(summary["global_phase_duplicate_count"], 1)
self.assertEqual(len(all_trials.attrs["candidate_global_phase_duplicates"]), 1)
self.assertEqual(
    all_trials.attrs["candidate_global_phase_duplicates"][0]["duplicate_candidate_name"],
    "I_phase",
)
```

The fake runner should return a successful `_native_iqm_circuit()` for one strategy and one seed. The distinct candidate can be `np.diag([1.0, 1.0, -1.0]).astype(complex)`.

- [ ] **Step 2: Run the test and verify that all three candidates are currently transpiled**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_iqm_transpiler_harness.py -k global_phase
```

Expected: the new assertions fail because the harness currently loops over all inputs and has no duplicate metadata.

- [ ] **Step 3: Integrate the deduplicator without changing the return signature**

At the start of `run_iqm_transpiler_harness`, materialize raw candidates, deduplicate them, and loop over representatives:

```python
raw_candidates = list(config.candidates)
phase_deduplication = deduplicate_candidates_up_to_global_phase(raw_candidates)
candidates = list(phase_deduplication.representatives)
candidate_count = len(candidates)
```

After creating the DataFrames, attach the duplicate audit records to both frames so direct writers and copied result consumers can access them:

```python
all_trials = pd.DataFrame(rows)
best_by_candidate = pd.DataFrame(best_rows)
duplicate_rows = [dict(row) for row in phase_deduplication.duplicate_rows]
all_trials.attrs["candidate_global_phase_duplicates"] = duplicate_rows
best_by_candidate.attrs["candidate_global_phase_duplicates"] = duplicate_rows
```

Keep `candidate_count` backward-compatible as the number of requested candidates and add explicit new counts:

```python
summary["candidate_count"] = len(raw_candidates)
summary["representative_candidate_count"] = len(candidates)
summary["global_phase_duplicate_count"] = phase_deduplication.removed_count
```

Return the same tuple `(all_trials, best_by_candidate, summary)`. Preserve unsupported candidate rows and the legacy depth-first `_best_trial_rows` logic.

- [ ] **Step 4: Run harness and phase tests**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_phase_equivalence.py tests\test_direct_basis_iqm_transpiler_harness.py
```

Expected: both files pass; existing tests still observe unchanged behavior for nonduplicate inputs.

- [ ] **Step 5: Commit Task 2**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py tests/test_direct_basis_iqm_transpiler_harness.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: deduplicate IQM candidates before transpilation"
```

## Task 3: Aggregate post-transpilation seed statistics per strategy

**Files:**

- Create: `src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py`
- Create: `tests/test_direct_basis_pareto_selection.py`

- [ ] **Step 1: Write failing aggregation tests**

Create a `_trial` helper that always supplies these columns:

```python
def _trial(
    *,
    candidate_name: str,
    strategy_name: str,
    seed: int,
    depth: float | None,
    two_qubit: float | None,
    one_qubit: float | None = 4,
    size: float | None = 8,
    success: bool = True,
    state_name: str = "two_qutrit",
) -> dict[str, object]:
    return {
        "state_name": state_name,
        "iqm_backend_name": "fake_backend",
        "backend_calibration_set_id": "calibration-a",
        "selection_label": "exact",
        "class_name": "test_class",
        "candidate_name": candidate_name,
        "strategy_name": strategy_name,
        "seed_transpiler": seed,
        "success": success,
        "status": "ok" if success else "failed",
        "depth": depth,
        "two_qubit_gate_count": two_qubit,
        "one_qubit_gate_count": one_qubit,
        "size": size,
        "graph_state_transpiled_qpy": f"{candidate_name}-{strategy_name}-{seed}.qpy",
    }
```

Assert that two strategies for the same candidate produce two rows, successful and failed trials are counted without mixing, and population standard deviation is used:

```python
statistics = aggregate_strategy_statistics(pd.DataFrame(rows))
self.assertEqual(len(statistics), 2)
strategy_a = statistics[statistics["strategy_name"] == "strategy_a"].iloc[0]
self.assertEqual(strategy_a["successful_trial_count"], 2)
self.assertEqual(strategy_a["failed_trial_count"], 1)
self.assertAlmostEqual(strategy_a["success_rate"], 2 / 3)
self.assertEqual(strategy_a["mean_depth"], 12.0)
self.assertEqual(strategy_a["std_depth"], 2.0)
self.assertEqual(strategy_a["mean_two_qubit_gate_count"], 3.0)
```

Add a best-trial test in which a shallower row has more two-qubit gates. Assert that selection follows:

```python
(
    two_qubit_gate_count,
    depth,
    one_qubit_gate_count,
    size,
    seed_transpiler,
)
```

and therefore chooses the row with fewer two-qubit gates. Add tests that one success gives `std_depth == 0.0` and `insufficient_stability_samples is True`, no successes remain with `pareto_eligible is False`, duplicate trial identities raise `ValueError`, and a negative or nonfinite successful metric raises a message containing candidate, strategy, and column.

- [ ] **Step 2: Run aggregation tests and verify the missing module/API failure**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_selection.py -k "aggregate or best_trial or stability or invalid or duplicate"
```

Expected: import or attribute failure for `pareto_selection`.

- [ ] **Step 3: Define stable analysis schemas and boundary columns**

In `pareto_selection.py`, define:

```python
IDENTITY_COLUMNS = (
    "state_name",
    "class_name",
    "candidate_name",
    "strategy_name",
)
OPTIONAL_BOUNDARY_COLUMNS = (
    "iqm_backend_name",
    "backend_calibration_set_id",
    "selection_label",
)
TRIAL_ID_COLUMNS = (*IDENTITY_COLUMNS, "seed_transpiler")
OBJECTIVE_COLUMNS = (
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
)
BEST_TRIAL_ORDER = (
    "two_qubit_gate_count",
    "depth",
    "one_qubit_gate_count",
    "size",
    "seed_transpiler",
)
```

Only add an optional boundary column to a grouping key when it exists in the input. Before aggregation, reject more than one distinct nonempty IQM backend or calibration ID in a single file unless those columns are included in every group key. This ensures that statistics never cross an explicit hardware boundary.

- [ ] **Step 4: Implement metric validation and aggregation**

Implement `aggregate_strategy_statistics(all_trials: pd.DataFrame) -> pd.DataFrame` with these operations:

1. Empty input returns an empty DataFrame with the documented strategy-statistics columns.
2. Reject duplicated `TRIAL_ID_COLUMNS` plus all present optional boundary columns.
3. Group all rows by the present boundary columns followed by `IDENTITY_COLUMNS`.
4. Select successful rows with `success.astype(bool) & status.eq("ok")`.
5. Validate successful `depth`, `two_qubit_gate_count`, `one_qubit_gate_count`, and `size` using `pd.to_numeric`; each value must be finite and non-negative.
6. Compute count, mean, min, max, and population standard deviation (`ddof=0`) for depth and two-qubit count.
7. Select the concrete best successful row with a stable `sort_values(list(BEST_TRIAL_ORDER), kind="mergesort")`.
8. Copy `best_seed_transpiler`, `best_graph_state_transpiled_qpy`, `best_depth`, `best_two_qubit_gate_count`, `best_one_qubit_gate_count`, and `best_size` from that row.
9. For a zero-success group, set metric summaries and best-trial fields to null, `pareto_eligible=False`, and `analysis_status="no_successful_trials"`.

Use the following stability flags:

```python
"insufficient_stability_samples": successful_trial_count < 2,
"pareto_eligible": successful_trial_count > 0,
"analysis_status": "eligible" if successful_trial_count > 0 else "no_successful_trials",
```

- [ ] **Step 5: Run aggregation tests to green**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_selection.py -k "aggregate or best_trial or stability or invalid or duplicate"
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py tests/test_direct_basis_pareto_selection.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: aggregate IQM strategy seed statistics"
```

## Task 4: Add Pareto layers and the approved deterministic 50/30/20 ranking

**Files:**

- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py`
- Modify: `tests/test_direct_basis_pareto_selection.py`

- [ ] **Step 1: Write failing Pareto-layer tests**

Add a compact strategy-statistics fixture with:

- `balanced = (2Q=3, depth=10, std=2)`;
- `low_2q = (2Q=2, depth=14, std=3)`;
- `stable = (2Q=4, depth=12, std=1)`;
- `dominated = (2Q=5, depth=16, std=4)`;
- `deeply_dominated = (2Q=6, depth=18, std=5)`;
- `balanced_copy = (2Q=3, depth=10, std=2)`.

Assert that the first three and `balanced_copy` have `pareto_rank == 1`, `dominated` has rank 2, and `deeply_dominated` has rank 3. Assert that identical metrics share `pareto_metric_group_id` but remain separate rows.

- [ ] **Step 2: Write failing normalization and score tests**

Assert these exact score semantics:

```python
self.assertEqual(row["normalized_mean_two_qubit_gate_count"], expected_2q)
self.assertEqual(row["normalized_mean_depth"], expected_depth)
self.assertEqual(row["normalized_std_depth"], expected_std)
self.assertAlmostEqual(
    row["ideal_score"],
    0.50 * expected_2q + 0.30 * expected_depth + 0.20 * expected_std,
)
```

Add a constant-column case and assert its normalized values are exactly `0.0`. Add a deterministic-tie case and assert ordering by `class_name`, `candidate_name`, then `strategy_name`. Add two states with reversed metrics and assert normalization and Pareto ranks are computed independently per state.

- [ ] **Step 3: Run the new tests and verify failure**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_selection.py -k "pareto or normalize or score or constant or tie or states"
```

Expected: failures because Pareto ranking is not yet implemented.

- [ ] **Step 4: Implement Pareto dominance and iterative layers**

Add:

```python
DEFAULT_OBJECTIVE_WEIGHTS = {
    "mean_two_qubit_gate_count": 0.50,
    "mean_depth": 0.30,
    "std_depth": 0.20,
}


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))
```

Implement `assign_pareto_ranks(statistics: pd.DataFrame) -> pd.DataFrame` per analysis boundary. For each boundary group, repeatedly find rows not dominated by any remaining row, assign the next integer rank, remove them from the working set, and continue. Preserve ineligible diagnostic rows with null `pareto_rank` and `ideal_score`.

Create `pareto_metric_group_id` from exact objective tuples after numeric validation. Sort tuples deterministically and number them `pareto_metrics_0001`, `pareto_metrics_0002`, and so on independently per boundary.

- [ ] **Step 5: Implement min-max normalization and ideal score**

Add `rank_pareto_candidates(statistics, *, objective_weights=None)`. Validate that weights contain exactly the three objective keys, are finite and non-negative, and have a positive sum. Normalize supplied weights to sum to one so the default remains exactly 0.50/0.30/0.20.

For each state and explicit boundary, normalize each objective across all eligible rows, not separately inside each Pareto layer:

```python
span = maximum - minimum
normalized = 0.0 if span == 0.0 else (value - minimum) / span
```

Compute `ideal_score` and stable-sort eligible rows by:

```python
[
    "pareto_rank",
    "ideal_score",
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
    "class_name",
    "candidate_name",
    "strategy_name",
]
```

Append ineligible rows after eligible rows using identity columns as deterministic ties. Add `recommendation_order` as a one-based integer within each boundary.

- [ ] **Step 6: Run the complete pure-statistics test file**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_selection.py
```

Expected: all aggregation, Pareto, normalization, and score tests pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py tests/test_direct_basis_pareto_selection.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: rank compiled circuits with Pareto layers"
```

## Task 5: Reconstruct logical compiled states and group state-equivalent recommendations

**Files:**

- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py`
- Create: `src/qudits_on_qubits/benchmarks/direct_basis/state_equivalence.py`
- Create: `tests/test_direct_basis_state_equivalence.py`
- Modify: `tests/test_direct_basis_benchmark.py`

- [ ] **Step 1: Add failing tests for a public logical-output-state helper**

In `tests/test_direct_basis_benchmark.py`, add tests around the existing layout-aware fidelity fixtures. The new public helper must return a `DensityMatrix` in input-logical-qubit order and a diagnostic note:

```python
state, note = logical_output_density_matrix(
    transpiled_circuit,
    logical_qubit_count=reference_circuit.num_qubits,
    max_qubits=10,
)
self.assertIsNotNone(state)
self.assertGreater(state_fidelity(Statevector.from_instruction(reference_circuit), state), 1 - 1e-10)
```

Reuse the existing tests that cover final-layout restoration, stripped idle qubits, and extra active qubits. Add a measurement-bearing circuit case and assert `(None, note)` with `"measurement"` in the note.

- [ ] **Step 2: Run the helper tests and verify the missing symbol failure**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_benchmark.py -k logical_output_density_matrix
```

Expected: import or name failure for `logical_output_density_matrix`.

- [ ] **Step 3: Refactor existing fidelity reconstruction into the public helper**

Add this signature to `benchmark.py`:

```python
def logical_output_density_matrix(
    circuit,
    *,
    logical_qubit_count: int,
    max_qubits: int,
) -> tuple[DensityMatrix | None, str]:
```

Move the existing idle-qubit stripping, final-layout restoration, active-output mapping, and ancillary-qubit tracing logic from `_safe_fidelity` into this helper. Before statevector construction, reject any circuit instruction whose operation name is `measure`. Return `DensityMatrix(candidate_statevector)` for equal logical/active widths and `DensityMatrix(_density_matrix_in_input_order(candidate_statevector.data, input_to_output))` for extra active qubits. Retain all existing diagnostic notes.

Rewrite `_safe_fidelity` to call the helper for the candidate circuit and compare its result to `Statevector.from_instruction(reference_qc)`. Run all existing fidelity tests to prove no behavior regressed.

- [ ] **Step 4: Write failing state-equivalence grouping tests**

Create `tests/test_direct_basis_state_equivalence.py` with ranked rows for three candidate-strategy combinations. Use an injected loader returning:

```python
states = {
    "a.qpy": DensityMatrix(Statevector([1, 0])),
    "b.qpy": DensityMatrix(Statevector([-1j, 0])),
    "c.qpy": DensityMatrix(Statevector([0, 1])),
}
```

Assert that `a` and `b` share one `state_equivalence_group_id`, `c` is separate, and the compact recommendations retain the better-ranked member of the `a/b` group. Add cases for a missing path and a loader exception; each must remain a singleton recommendation with `state_equivalence_status` equal to `missing_qpy` or `state_reconstruction_failed` and a nonempty diagnostic message.

Add a real-QPY integration test that writes a one-qubit circuit using `qpy.dump`, invokes the default loader, and verifies that it reconstructs the expected state.

- [ ] **Step 5: Run state-equivalence tests and verify the import failure**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_state_equivalence.py
```

Expected: collection fails because `state_equivalence.py` does not exist.

- [ ] **Step 6: Implement QPY loading, phase-invariant grouping, and recommendation collapse**

Define:

```python
STATE_EQUIVALENCE_ATOL = 1e-9


def load_logical_state_from_qpy(
    path: str,
    logical_qubit_count: int,
    *,
    max_qubits: int,
) -> tuple[DensityMatrix | None, str]:
```

The loader must require one circuit in the QPY file and call `logical_output_density_matrix`. Cache successful loads by `(resolved_path, logical_qubit_count, max_qubits)` during one analysis.

Define:

```python
def group_state_equivalent_candidates(
    pareto_ranked: pd.DataFrame,
    *,
    max_qubits: int = 12,
    state_loader=load_logical_state_from_qpy,
    atol: float = STATE_EQUIVALENCE_ATOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

For eligible rows, load `best_graph_state_transpiled_qpy`, infer logical width from `n_qutrits * 2` when `n_qutrits` is available, otherwise use the QPY circuit input width, and compare states using `state_fidelity(left, right) >= 1.0 - atol`. Density-matrix fidelity is invariant to global phase.

Use union-find within each present state/device/calibration/selection boundary so transitive equivalent pairs share one group. Sort members by `(class_name, candidate_name, strategy_name)` before assigning IDs `state_equivalence_0001`, `state_equivalence_0002`, and so on. Unsafe or missing states get singleton IDs and explicit status/diagnostics.

Choose one recommendation per group by stable ordering:

```python
[
    "pareto_rank",
    "ideal_score",
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
    "class_name",
    "candidate_name",
    "strategy_name",
]
```

Add representative identity columns to every detailed row:

```python
"recommended_class_name"
"recommended_candidate_name"
"recommended_strategy_name"
"is_state_equivalence_recommendation"
```

- [ ] **Step 7: Run benchmark and state-equivalence tests to green**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_benchmark.py tests\test_direct_basis_state_equivalence.py
```

Expected: all tests pass, including pre-existing layout-aware fidelity coverage.

- [ ] **Step 8: Commit Task 5**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py src/qudits_on_qubits/benchmarks/direct_basis/state_equivalence.py tests/test_direct_basis_benchmark.py tests/test_direct_basis_state_equivalence.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: group state-equivalent compiled circuits"
```

## Task 6: Orchestrate analysis and write all new harness artifacts

**Files:**

- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py`
- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py`
- Modify: `tests/test_direct_basis_pareto_selection.py`
- Modify: `tests/test_direct_basis_iqm_transpiler_harness.py`

- [ ] **Step 1: Write a failing end-to-end analysis result test**

Add a result dataclass expectation and test:

```python
result = analyze_iqm_trials(all_trials, state_loader=fake_state_loader)

self.assertEqual(set(result.strategy_statistics["strategy_name"]), {"a", "b"})
self.assertEqual(result.pareto_ranked["pareto_rank"].min(), 1)
self.assertEqual(len(result.state_equivalence_groups), 2)
self.assertEqual(len(result.recommended_circuits), 1)
self.assertEqual(result.summary_counts["analyzed_strategy_combination_count"], 2)
self.assertEqual(result.summary_counts["pareto_front_count"], 2)
self.assertEqual(result.summary_counts["state_equivalence_group_count"], 1)
self.assertEqual(result.summary_counts["recommended_circuit_count"], 1)
```

Use two candidate-strategy combinations with two seeds each. Make both eligible and state-equivalent but Pareto trade-offs, so both are visible on the front while one compact recommendation is retained.

- [ ] **Step 2: Define and implement the orchestration API**

Add:

```python
@dataclass(frozen=True)
class ParetoAnalysisResult:
    strategy_statistics: pd.DataFrame
    pareto_ranked: pd.DataFrame
    state_equivalence_groups: pd.DataFrame
    recommended_circuits: pd.DataFrame
    summary_counts: dict[str, int]


def analyze_iqm_trials(
    all_trials: pd.DataFrame,
    *,
    objective_weights: dict[str, float] | None = None,
    max_state_qubits: int = 12,
    state_loader=load_logical_state_from_qpy,
) -> ParetoAnalysisResult:
```

The function must call aggregation, Pareto ranking, and state grouping in that order. For an empty input, return schema-correct empty frames and zero counts. Set `pareto_front_count` to the number of eligible rows whose `pareto_rank == 1`, not the number of metric groups.

- [ ] **Step 3: Write failing output-writer assertions**

Extend `test_write_iqm_transpiler_harness_outputs` to pass a complete harness-shaped `all_trials` fixture and assert all returned paths exist:

```python
expected_path_keys = {
    "all_trials_csv",
    "best_by_candidate_csv",
    "candidate_global_phase_duplicates_csv",
    "strategy_statistics_csv",
    "pareto_ranked_csv",
    "state_equivalence_groups_csv",
    "recommended_circuits_csv",
    "summary_json",
}
self.assertEqual(set(paths), expected_path_keys)
```

Set `all_trials.attrs["candidate_global_phase_duplicates"]` to one audit row and assert it appears in `candidate_global_phase_duplicates.csv`. Assert existing summary keys survive and the new counts are merged.

- [ ] **Step 4: Run the writer test and verify missing output failures**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_iqm_transpiler_harness.py -k write_iqm_transpiler_harness_outputs
```

Expected: assertions fail because only the legacy three files are currently written.

- [ ] **Step 5: Extend the writer while preserving legacy files and meanings**

Update `write_iqm_transpiler_harness_outputs` with optional analysis controls that keep existing callers valid:

```python
def write_iqm_transpiler_harness_outputs(
    output_dir: str | Path,
    *,
    all_trials: pd.DataFrame,
    best_by_candidate: pd.DataFrame,
    summary: dict[str, Any],
    objective_weights: dict[str, float] | None = None,
    max_state_qubits: int = 12,
    state_loader=load_logical_state_from_qpy,
) -> dict[str, str]:
```

Call `analyze_iqm_trials` before writing. Build the phase duplicate table with exact `PHASE_DUPLICATE_COLUMNS`, using `all_trials.attrs.get("candidate_global_phase_duplicates", ())`. Write:

```text
all_trials.csv
best_by_candidate.csv
candidate_global_phase_duplicates.csv
strategy_statistics.csv
pareto_ranked.csv
state_equivalence_groups.csv
recommended_circuits.csv
summary.json
```

Merge summary counts into a new dictionary so the caller's input is not mutated:

```python
merged_summary = {**summary, **analysis.summary_counts}
```

Do not rename, remove, or reinterpret existing `all_trials.csv`, `best_by_candidate.csv`, or summary fields. Write empty frames with headers, so every advertised output exists even when no candidate is eligible.

- [ ] **Step 6: Run analysis and harness tests to green**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_selection.py tests\test_direct_basis_state_equivalence.py tests\test_direct_basis_iqm_transpiler_harness.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 6**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py tests/test_direct_basis_pareto_selection.py tests/test_direct_basis_iqm_transpiler_harness.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: emit IQM Pareto analysis artifacts"
```

## Task 7: Add standalone post-processing and expose new harness outputs in the CLI

**Files:**

- Create: `scripts/analyze_iqm_transpiler_harness.py`
- Create: `tests/test_direct_basis_pareto_cli.py`
- Modify: `scripts/run_iqm_transpiler_harness.py`
- Modify: `tests/test_direct_basis_iqm_transpiler_harness_cli.py`

- [ ] **Step 1: Write failing parser and CLI tests**

Test the standalone parser defaults:

```python
args = _build_parser().parse_args(["--all-trials", "run/all_trials.csv"])
self.assertEqual(args.two_qubit_weight, 0.50)
self.assertEqual(args.depth_weight, 0.30)
self.assertEqual(args.std_depth_weight, 0.20)
self.assertEqual(args.max_state_qubits, 12)
```

In a temporary directory, write a complete successful `all_trials.csv`, write an existing `summary.json` containing `{"candidate_count": 2}`, invoke `main`, and assert the five analysis CSVs exist and `summary.json` contains both `candidate_count` and `recommended_circuit_count`.

Mock `analyze_iqm_trials` or use a one-qubit QPY fixture so this test remains local and does not load an IQM backend.

- [ ] **Step 2: Run CLI tests and verify the script import failure**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_cli.py
```

Expected: collection fails because the analysis script does not exist.

- [ ] **Step 3: Implement the standalone CLI**

The parser must expose:

```text
--all-trials PATH                  required
--output-dir PATH                  defaults to the CSV parent directory
--two-qubit-weight FLOAT           default 0.50
--depth-weight FLOAT               default 0.30
--std-depth-weight FLOAT           default 0.20
--max-state-qubits INTEGER         default 12
```

`main` must:

1. load the CSV with `pd.read_csv`;
2. build the three-key objective weight dictionary;
3. call `analyze_iqm_trials` without invoking any transpiler or backend;
4. write `strategy_statistics.csv`, `pareto_ranked.csv`, `state_equivalence_groups.csv`, and `recommended_circuits.csv`;
5. create an empty, header-only `candidate_global_phase_duplicates.csv` when it does not already exist, because raw candidate matrices are unavailable in `all_trials.csv`;
6. read an existing object-valued `summary.json` when present, merge analysis counts, and write it back; if absent, create it from the analysis counts;
7. print every output path and return `0`.

Factor the four common analysis-frame writes into `write_pareto_analysis_outputs` in `pareto_selection.py` so the harness writer and standalone CLI share filenames and serialization.

- [ ] **Step 4: Extend the harness CLI output reporting**

After `write_iqm_transpiler_harness_outputs`, print the five new returned paths. Update `tests/test_direct_basis_iqm_transpiler_harness_cli.py` to assert those files are passed through without changing existing parser defaults or harness configuration.

- [ ] **Step 5: Run both CLI suites**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_pareto_cli.py tests\test_direct_basis_iqm_transpiler_harness_cli.py
```

Expected: both suites pass without network access or IQM hardware submission.

- [ ] **Step 6: Commit Task 7**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add scripts/analyze_iqm_transpiler_harness.py scripts/run_iqm_transpiler_harness.py src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py tests/test_direct_basis_pareto_cli.py tests/test_direct_basis_iqm_transpiler_harness_cli.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "feat: add IQM Pareto post-processing CLI"
```

## Task 8: Document the workflow and run regression verification

**Files:**

- Modify: `README.md:430`
- Modify: `tests/test_direct_basis_iqm_transpiler_harness.py`

- [ ] **Step 1: Add a final compatibility regression**

Add one test that runs two strategies across two seeds for three raw candidates where one is a phase duplicate. Assert all of the following in one integration path:

```python
self.assertEqual(summary["candidate_count"], 3)
self.assertEqual(summary["representative_candidate_count"], 2)
self.assertEqual(summary["global_phase_duplicate_count"], 1)
self.assertEqual(len(all_trials), 8)
self.assertEqual(len(best_by_candidate), 2)
self.assertEqual(set(best_by_candidate["candidate_name"]), {"E_old", "distinct"})
```

After writing outputs, assert `best_by_candidate.csv` still chooses its historical depth-first winner, while `strategy_statistics.csv` chooses its `best_seed_transpiler` with the new two-qubit-first order.

- [ ] **Step 2: Run the compatibility regression and fix only observed integration defects**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_iqm_transpiler_harness.py -k "compatibility and phase"
```

Expected: the integration test passes. If it fails, make the smallest production change that restores the approved boundary; do not change the legacy selection order.

- [ ] **Step 3: Update the IQM Transpiler Harness README section**

Extend the output tree under `README.md:448` with the five new CSV files. Document:

- candidate matrices are deduplicated up to global phase before transpilation;
- seed statistics and Pareto metrics are computed after IQM transpilation from `all_trials.csv`;
- strategies remain separate statistical alternatives;
- rank 1 is the nondominated front over mean 2Q count, mean depth, and depth standard deviation;
- `ideal_score` uses 0.50/0.30/0.20 weights and cannot override Pareto rank;
- `recommended_circuits.csv` collapses compiled state-equivalent alternatives only after their physical costs are known;
- `best_by_candidate.csv` remains the legacy depth-first view;
- the standalone command is:

```powershell
python scripts/analyze_iqm_transpiler_harness.py --all-trials artifacts/iqm_runs/processed/transpiler_harness/20260902_120000/all_trials.csv
```

State explicitly that neither CLI submits hardware jobs.

- [ ] **Step 4: Run focused feature tests**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_phase_equivalence.py tests\test_direct_basis_pareto_selection.py tests\test_direct_basis_state_equivalence.py tests\test_direct_basis_iqm_transpiler_harness.py tests\test_direct_basis_iqm_transpiler_harness_cli.py tests\test_direct_basis_pareto_cli.py
```

Expected: all focused feature tests pass.

- [ ] **Step 5: Run the established direct-basis regression suite**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -q tests\test_direct_basis_iqm_transpiler_harness.py tests\test_direct_basis_rerun_selection.py tests\test_direct_basis_selection.py tests\test_direct_basis_benchmark.py
```

Expected: all tests pass. Compare against the pre-change baseline of `76 passed, 1 warning, 19 subtests passed` for the first three files, allowing the pass count to increase because new tests were added.

- [ ] **Step 6: Run formatting and repository checks**

```powershell
python -m compileall -q src\qudits_on_qubits\benchmarks\direct_basis scripts\analyze_iqm_transpiler_harness.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup diff --check
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup status --short
```

Expected: compilation succeeds, `diff --check` is silent, and status lists only intended Task 8 files before the commit.

- [ ] **Step 7: Commit Task 8**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup add README.md tests/test_direct_basis_iqm_transpiler_harness.py
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup commit -m "docs: explain IQM Pareto benchmark selection"
```

- [ ] **Step 8: Inspect the final branch history and cleanliness**

```powershell
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup log --oneline --decorate -9
git -c safe.directory=C:/Users/szymo/QuditsOnQubits/QuditsOnQubits/.worktrees/pareto-phase-dedup status --short
```

Expected: the design and plan commits are followed by the implementation commits, and the worktree is clean.
