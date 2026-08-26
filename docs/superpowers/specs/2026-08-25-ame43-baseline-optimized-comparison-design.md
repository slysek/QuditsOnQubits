# AME43 Baseline vs Exact-Optimized Comparison Design

## Goal

Add one reproducible notebook that compares the canonical AME(4,3) baseline
against the exact-optimized preparation while retaining both standalone
notebooks. The comparison must separate logical equivalence, compiler cost, and
hardware behavior.

## Files and responsibilities

- `notebooks/ame43_canonical_baseline.ipynb`: standalone canonical baseline.
- `notebooks/ame43_canonical_exact_optimized.ipynb`: standalone exact-optimized
  experiment.
- `notebooks/ame43_canonical_comparison.ipynb`: new side-by-side comparison.
- `tests/test_ame43_canonical_comparison_notebook.py`: structure, safety,
  correctness, and Aer regressions for the comparison notebook.

The comparison notebook consumes the existing baseline and optimized circuit
builders. It does not duplicate graph-state construction logic.

## Default offline comparison

The default `Run All` path must require no credentials and submit no remote
jobs. It will:

1. Materialize distinct baseline and optimized canonical bundles.
2. Validate both QPY circuits and the common `canonical_ez` encoding.
3. Prove statevector equivalence and report fidelity.
4. Evaluate both Bell experiments with Aer using the same shot count and seed.
5. Transpile both preparations and all 13 measurement circuits to the same
   exact all-to-all `u/cz` basis.
6. Display one comparison table containing logical depth, preparation CZ/depth,
   measurement-batch CZ/depth totals and ranges, Bell estimate, ideal value, and
   classical bound.

Expected exact preparation metrics in the current pinned environment are
baseline `44 CZ / depth 40` and optimized `36 CZ / depth 23`. Automated tests
assert relative improvement rather than exact Qiskit snapshot values.

## IQM comparison

IQM behavior is split into two explicit disabled guards:

- `RUN_IQM_COMPILE = False`: read-only Garnet target fetch and compilation of
  both 13-circuit batches with one shared seed.
- `RUN_IQM_HARDWARE = False`: two comparable raw hardware runs, baseline and
  optimized, with `HARDWARE_SHOTS = 50` by default and no ZNE/readout mitigation.

Raw runs avoid hiding circuit effects behind different folding or mitigation.
The user may raise shots to 100 after the smoke comparison.

Seed `13` is the balanced shared IQM seed. In the completed 20-seed optimized
sweep it gives `891` total CZ and maximum depth `85`; seed `6` gives fewer total
CZ (`879`) but a much worse maximum depth (`122`). The comparison favors bounded
worst-case depth over a 1.4% reduction in total CZ.

The notebook resolves `.env` in either a normal checkout (`REPO_ROOT/.env`) or
the repository owning a project-local `.worktrees/<name>` checkout. It passes
that path explicitly to `IQMHardware`; credentials are never printed or stored
in notebook output/spec artifacts.

## Safety and cache handling

- Source notebooks remain unexecuted with empty outputs and hardware guards
  false.
- Generated bundles and executed notebooks remain untracked artifacts.
- Bundle validation rejects stale/deoptimized circuits and unsafe symlink or
  reparse ancestors.
- Existing cross-process cache locking/recovery remains covered by tests.
- The notebook uses only the public experiment API for hardware execution.

## Validation

- TDD: comparison-notebook existence/config test fails before file creation.
- Structural tests: separate bundles, shared encoding, shared seed, disabled
  guards, no secrets or low-level clients, unexecuted source.
- Semantic tests: statevector equivalence and optimized relative CZ/depth
  reduction.
- Aer: both Bell estimates remain near the ideal value `8`.
- Notebook `Run All`: completes with IQM guards false.
- Optional IQM smoke: baseline and optimized at 50 shots, recorded only under
  ignored artifacts.
- Full relevant regression suite and mandatory independent reviewer loop.

## Git and PR

The implementation remains on `codex/ame43-exact-optimization`. Because local
`main` is ahead of the branch, all scoped changes are committed first, then the
branch is rebased onto current `main`, validated again, pushed, and opened as a
PR targeting `main`. Generated reference bundles and unrelated workspace files
are excluded from commits.
