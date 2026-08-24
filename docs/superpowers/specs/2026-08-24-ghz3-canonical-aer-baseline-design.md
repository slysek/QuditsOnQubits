# GHZ3 canonical Aer baseline notebook

## Goal

Add a reproducible notebook for the `ghz3` Bell experiment using the `canonical_ez` qutrit encoding and the local ideal Aer backend. Keep its workflow and artifact contract analogous to `notebooks/two_qutrit_bell_canonical_baseline.ipynb`. The committed notebook remains Aer-only, while verification may submit one low-shot connected smoke run to IQM Garnet.

## User journey

As an experiment author, I want to run one notebook from a clean checkout and obtain a validated GHZ3 canonical-basis Bell baseline, so I can compare later IQM and PiastQ runs against a deterministic local reference.

## Notebook

Create `notebooks/ghz3_bell_canonical_baseline.ipynb` with these sections:

1. Purpose and execution-safety note.
2. Imports and repository-root discovery.
3. Canonical input validation and idempotent materialization.
4. Shared Aer configuration.
5. Aer ideal execution.
6. Result summary.

All code cells must have empty outputs and `execution_count: null` in the committed notebook.

## Canonical input bundle

Materialize the input under:

`experiment_inputs/reference_bases/ghz3/canonical_ez/`

The bundle contains:

- `graph_state_direct_basis.qpy`: one unmeasured 6-qubit GHZ3 direct-basis circuit;
- `E.npy`: the `(4, 3)` `canonical_ez` isometry;
- `metadata.json`: state, encoding, shape, circuit size, and SHA-256 provenance.

Materialization must be idempotent. Existing valid data is reused. Invalid or incomplete data raises a clear `RuntimeError`. New data is written through a temporary staging directory, validated, then moved into place. Failed writes leave no staging directory or partial final bundle.

Validation checks:

- encoding shape equals `(4, 3)`;
- encoding equals registry `canonical_ez`;
- `E.conj().T @ E` equals the 3-dimensional identity;
- QPY contains exactly one circuit;
- circuit has 6 qubits, zero classical bits, and no measurements;
- circuit statevector is equivalent to `build_direct_basis_graph_state_circuit("ghz3", expected_encoding)`;
- metadata fields and stored SHA-256 hashes match the files.

## Experiment configuration

Use:

- `state="ghz3"`;
- `basis=PathBasis(CANONICAL_BASIS_DIRECTORY)`;
- `backend=AerIdeal(seed_simulator=11)`;
- `shots=100`;
- `BootstrapConfig(samples=2_000, seed=7)`;
- tags `{"baseline": "canonical_ez", "backend": "aer_ideal"}`.

The notebook invokes `run_experiment` once and passes `repo_root=REPO_ROOT`. IQM, PiastQ, credentials, low-level provider clients, and hardware submission flags are excluded from the committed notebook.

## Summary

Produce a JSON-serializable one-row summary for `aer_ideal`. Include:

- result status;
- runner-produced values without recomputation;
- reference classical bound;
- reference ideal Bell value;
- artifact directory.

If no result is present, status is explicitly `not_run`.

## Error handling

- Repository discovery fails with a clear error when project markers are missing.
- QPY load rejects zero or multiple circuits.
- Corrupt metadata, mismatched hashes, wrong encoding, wrong circuit size, measurements, or statevector mismatch fail before experiment execution.
- Staging cleanup runs after any materialization failure.

## Tests

Add `tests/test_ghz3_canonical_baseline_notebook.py` before creating the notebook.

Tests cover:

- expected notebook structure and exactly one high-level Aer run;
- exact experiment configuration and serializable empty summary;
- absence of secrets, user-specific paths, hardware providers, and low-level execution;
- repository discovery from project root and notebook directory;
- exact, idempotent canonical bundle materialization;
- rejection of corrupt metadata;
- isometry, 6-qubit QPY, and unmeasured-circuit validation;
- staging cleanup after a simulated write failure;
- end-to-end local Aer execution with 64 shots and a small deterministic bootstrap sample count.

After local tests pass, verification may run one connected IQM Garnet smoke experiment through the same high-level `run_experiment` pipeline with:

- the materialized GHZ3 `canonical_ez` basis;
- `IQMHardware(device="garnet", use_metrics=True)`;
- `shots=50`;
- minimal uncertainty work suitable for smoke verification;
- no readout mitigation or ZNE, to keep hardware usage bounded;
- provider/environment credentials only, never credentials stored in the repository.

The connected run is conditional on an available IQM provider configuration. Its job identifier, final status, artifact directory, and shortest decisive failure message are reported. It is not a deterministic unit test and is not required in offline CI.

The focused notebook test must demonstrate RED because the notebook is absent, then GREEN after implementation. Relevant existing regression tests are rerun afterward.

## Out of scope

- PiastQ execution;
- readout mitigation or ZNE;
- selected-best/ranked encoding bases;
- changes to existing two-qutrit notebook or experiment-runner behavior.
