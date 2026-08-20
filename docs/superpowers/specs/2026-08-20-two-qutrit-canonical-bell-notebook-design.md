# Canonical Two-Qutrit Bell Notebook Design

## Goal

Add one canonical notebook that prepares the standard `canonical_ez` encoding,
runs the existing durable Bell experiment pipeline for `two_qutrit`, and keeps
Aer, IQM Garnet, and PiastQ execution in separate cells.

The notebook is a reproducible baseline for later experiments. It does not
duplicate Bell measurement construction, execution, mitigation, persistence,
or postprocessing already owned by the experiment pipeline.

## Files and Layout

The notebook will be stored at:

```text
notebooks/two_qutrit_bell_canonical_baseline.ipynb
```

Reusable experiment inputs will be stored outside `artifacts/`:

```text
experiment_inputs/
  reference_bases/
    two_qutrit/
      canonical_ez/
        graph_state_direct_basis.qpy
        E.npy
        metadata.json
```

The hierarchy intentionally supports later sibling states and encodings, such
as `reference_bases/ghz3/canonical_ez/` or another encoding below
`reference_bases/two_qutrit/`.

## Notebook Structure

The notebook contains these sections in order:

1. Purpose, execution safety, and environment requirements.
2. Repository-root discovery and imports.
3. Deterministic canonical input preparation and validation.
4. Shared experiment parameters.
5. Aer ideal execution.
6. IQM Garnet execution guarded by `RUN_IQM = False`.
7. PiastQ execution guarded by `RUN_PIASTQ = False`.
8. Backend-independent result summary.

Each backend section contains its own `ExperimentSpec` and one separate
`run_experiment` call. Aer runs without mitigation. IQM and PiastQ enable
readout mitigation, linear ZNE with factors `(1, 3, 5)`, and 2,000 local
bootstrap resamples. Every backend uses 20,480 shots.

## Canonical Input Preparation

The setup derives the encoding from public
`get_encoding("canonical_ez")`. It builds the unmeasured four-qubit
`two_qutrit` graph-state circuit with the repository's direct-basis circuit
builder.

If the input directory is absent, setup creates the three files. Writes use
temporary sibling files followed by replacement so an interrupted write does
not leave a partial canonical input.

If the directory already exists, setup validates it and does not overwrite it.
Validation requires:

- `E.npy` has shape `(4, 3)`, finite entries, and satisfies
  `E.conj().T @ E == I`;
- the QPY contains exactly one four-qubit circuit;
- the circuit has no classical bits, measurements, resets, conditions, or
  control flow;
- `metadata.json` identifies `two_qutrit`, `canonical_ez`, and the input schema;
- recorded SHA-256 hashes match the QPY and NPY files.

Any missing, conflicting, or corrupt existing file raises a clear error. The
user must resolve the conflict explicitly; the notebook never silently repairs
or replaces an established baseline.

## Pipeline Use

All execution uses the public high-level API:

```python
ExperimentSpec(...)
PathBasis(...)
run_experiment(..., repo_root=REPO_ROOT)
```

`PathBasis` points at the canonical input directory. The durable runner remains
responsible for validation, measurement circuit generation, backend
compilation, execution, mitigation, bootstrap uncertainty, checkpointing,
postprocessing, and run artifacts.

No notebook cell calls a backend's `.run()` method, constructs Bell measurement
circuits, computes the Bell functional directly, or reimplements mitigation.
There is no backend fallback.

## Backend Configuration

### Aer

- Backend: `AerIdeal` with a fixed simulator seed.
- Shots: 20,480.
- Mitigation: disabled.
- Purpose: deterministic ideal reference using the same sampling and Bell
  postprocessing pipeline as hardware.

### IQM Garnet

- Backend: `IQMHardware(device="garnet")`.
- Guard: `RUN_IQM = False` by default.
- Shots: 20,480.
- Mitigation: readout plus linear ZNE `(1, 3, 5)`.
- Uncertainty: 2,000 bootstrap resamples with a fixed local seed.
- Credentials: provider/environment configuration only.

### PiastQ

- Backend: `PiastQHardware`, configured from its safe non-secret defaults and
  provider environment.
- Guard: `RUN_PIASTQ = False` by default.
- Shots: 20,480.
- Mitigation: readout plus linear ZNE `(1, 3, 5)`.
- Uncertainty: 2,000 bootstrap resamples with a fixed local seed.
- Credentials: provider/environment configuration only.

The hardware flags must be changed deliberately before their cells submit
jobs. Running all cells with defaults performs only the Aer experiment.

## Result Summary

Each successful backend stores its `ExperimentResult` in a shared result map.
The final cell builds a compact comparison from available entries. It shows:

- backend label and terminal status;
- raw Bell estimate;
- available mitigated estimates;
- component uncertainty and confidence intervals;
- leakage-related values exposed by the result;
- classical bound and ideal reference value from the frozen reference registry;
- durable `artifact_dir`.

Skipped hardware cells remain explicit and do not break the summary.

## Failure Handling

- Repository discovery fails with an actionable error when launched outside the
  repository tree.
- Missing optional backend dependencies remain explicit pipeline errors.
- Missing credentials remain provider/backend errors; no secrets are requested
  or persisted by the notebook.
- A backend failure is not caught and converted into another backend run.
- Existing canonical-input conflicts stop before any experiment submission.
- Hardware guards print a concise skipped status without constructing or
  submitting a remote job.

## Verification

Add focused tests that parse and execute notebook setup cells without executing
the three runner cells.

Tests will verify:

- exactly three separate `run_experiment` call cells exist;
- each call uses `repo_root=REPO_ROOT` and the intended backend;
- IQM and PiastQ calls are controlled by false-by-default guards;
- notebook source contains no embedded credentials, user-specific absolute
  paths, direct backend `.run()` calls, or duplicated Bell/mitigation logic;
- root discovery works from repository root and notebook directory;
- canonical input preparation works in a temporary root, is idempotent, and
  rejects modified input files;
- generated encoding and circuit satisfy pipeline contracts;
- a small-shot Aer smoke run completes through `run_experiment`.

Run focused notebook tests and the relevant experiment/Aer integration tests,
then review the final diff. Existing unrelated user changes remain untouched.
