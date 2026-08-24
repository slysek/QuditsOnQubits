# AME(4,3) canonical baseline notebook design

## Goal

Add a standalone executable notebook for the canonical AME(4,3) Bell baseline. It must follow the validated structure of `two_qutrit_bell_canonical_baseline.ipynb` without changing that notebook or its tests.

## Scope

Create `notebooks/ame43_canonical_baseline.ipynb` and a dedicated notebook test module. The notebook supports three separate high-level experiment runs:

- deterministic local Aer baseline;
- opt-in IQM Garnet hardware baseline;
- opt-in PiastQ managed baseline.

Only IQM Garnet is submitted during implementation verification. PiastQ remains present but disabled and untested against hardware.

## Canonical input bundle

The notebook materializes an AME(4,3) bundle under:

`experiment_inputs/reference_bases/ame43/canonical_ez`

The bundle contains exactly:

- `graph_state_direct_basis.qpy`;
- `E.npy`;
- `metadata.json`.

The expected encoding is `get_encoding("canonical_ez")`, with shape `(4, 3)`, finite numeric entries, and isometric columns. The expected state preparation is `build_direct_basis_graph_state_circuit("ame43", encoding)`. Its QPY circuit must contain one unmeasured 8-qubit circuit with no reset, classical condition, or control flow.

Metadata uses schema `qoq-reference-basis-v1`, state `ame43`, encoding ID `canonical_ez`, 8 qubits, encoding shape `[4, 3]`, and SHA-256 hashes for the QPY and NPY files. Existing bundles are validated before use. New bundles are written through a unique staging directory, validated, and atomically renamed. Failed writes clean their staging files.

## Notebook flow

The notebook uses repository-root discovery and imports only public experiment APIs. Configuration:

- `SHOTS = 100`;
- deterministic bootstrap seed;
- readout mitigation and ZNE factors `(1, 3, 5)` for hardware;
- canonical AME(4,3) reference via `get_reference_experiment("ame43")`.

Run All always executes Aer. Hardware cells are separate and guarded by `RUN_IQM = False` and `RUN_PIASTQ = False`. Credentials remain provider- or environment-managed; notebook stores no secret names, values, absolute user paths, or low-level provider clients.

The summary reports `aer_ideal`, `iqm_garnet`, and `piastq` in fixed order. Missing hardware results remain explicit `skipped` rows. Completed rows expose runner-produced raw, readout-mitigated, ZNE, combined mitigation, diagnostics, leakage, reference bounds, and artifact directories without recomputation.

## Error handling

Input validation fails with actionable `RuntimeError` messages for unavailable directories, unexpected files, malformed NPY/QPY/JSON data, incorrect circuit width, non-isometric encoding, state mismatch, or metadata mismatch. Hardware errors remain handled and persisted by the existing experiment runner.

## Tests

Dedicated tests verify:

- notebook cells are unexecuted and output-free in version control;
- exactly three separate `run_experiment` calls use Aer, IQM, and PiastQ public backends;
- hardware flags default to false;
- all specs use state `ame43`, canonical bundle, 100 shots, expected tags, and hardware mitigation;
- summary content and explicit skip behavior;
- absence of secrets, user paths, low-level execution, and provider clients;
- repository-root discovery from repository and notebook directories;
- exact, idempotent bundle creation and staging cleanup;
- rejection of corrupt metadata and invalid canonical content;
- an Aer integration run returns a completed AME(4,3) result near ideal Bell value `8.0` and writes artifacts.

Verification runs the dedicated tests, relevant experiment/reference regressions, then the broader suite if practical. Final hardware verification enables only IQM Garnet for one experiment run configured with 100 shots per circuit. PiastQ remains disabled.

## Non-goals

- Refactoring the existing two-qutrit notebook into shared helpers.
- Modifying the existing two-qutrit notebook or its pending changes.
- Benchmark-rank selection or replacing canonical encoding with a selected-best encoding.
- PiastQ hardware submission.
- Embedding credentials or provider configuration.
