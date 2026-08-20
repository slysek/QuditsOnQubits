# SZY-43 Unified Provider Contracts Design

**Date:** 2026-08-20
**Issue:** [SZY-43](https://linear.app/szymonplaner/issue/SZY-43/m31-ujednolicic-kontrakty-aer-iqm-i-piastq)
**Repository:** `slysek/QuditsOnQubits`
**Target branch:** `main`

## Goal

Give Aer, IQM, and PiastQ one explicit experiment and manifest contract. The same scientific experiment configuration must run through the real Aer adapter and mocked IQM and PiastQ provider boundaries, produce durable circuit and result artifacts, and load as one public, immutable `RunManifest` model.

## Current State

- `ExperimentSpec` is an immutable backend-neutral experiment configuration except for its explicit backend specification.
- The real Aer integration test exercises the complete runner and durable artifacts.
- IQM and PiastQ have detailed adapter unit tests, but no provider-specific end-to-end tests that pass through the complete runner and load a shared manifest type.
- The runner persists a versioned `experiment.json` mapping with `schema_version = 1`; there is no public `RunManifest` model.
- Backend identity and capabilities are recorded after resolution, but the execution mode is not serialized explicitly.
- `CustomBackend` cannot distinguish hardware, ideal simulation, and noisy simulation from its current fields.
- The clean `origin/main` baseline passes: 703 tests passed, 3 skipped, and 315 subtests passed.

## Scope

This change will:

- add a public `ExecutionMode` enum;
- add a public immutable `RunManifest` model and loader;
- introduce manifest schema version 2;
- serialize one explicit execution mode in every backend specification;
- require `CustomBackend` callers to provide an explicit execution mode;
- normalize supported schema-v1 manifests in memory;
- validate every runner checkpoint before it is atomically persisted;
- add full Aer, mocked IQM, and mocked PiastQ provider contract tests;
- document the public manifest and migration behavior.

## Non-Goals

- No general `QuditCircuit` abstraction or universal compiler.
- No migration to the broader `QuditEncoding` and `CircuitSpec` architecture proposed separately for M2.1.
- No real IQM or PiastQ network requests, credentials, queue waits, or hardware execution.
- No Pydantic or JSON Schema runtime dependency.
- No change to the QPY, counts, calibration, or result artifact formats.
- No change to the public `ExperimentResult` shape.
- No automatic rewrite of a manifest merely because it was loaded.

## File Structure

### New production files

- `src/qudits_on_qubits/experiments/execution.py`
  - Defines `ExecutionMode`.
  - Contains no provider imports.

- `src/qudits_on_qubits/experiments/manifest.py`
  - Defines immutable `RunManifest`.
  - Validates and deeply freezes manifest snapshots.
  - Normalizes schema-v1 built-in backend documents to schema v2.
  - Loads `experiment.json` through `ExperimentStore` path-safety rules.

### Modified production files

- `src/qudits_on_qubits/experiments/models.py`
  - Adds serialized execution modes to all backend specifications.
  - Requires explicit `execution_mode` for `CustomBackend`.

- `src/qudits_on_qubits/experiments/runner.py`
  - Creates schema-v2 documents.
  - Validates every checkpoint with `RunManifest` before writing.
  - Normalizes schema-v1 documents before resume logic consumes them.
  - Checks configured backend kind, execution mode, and resolved backend identity for consistency before compilation or submission.

- `src/qudits_on_qubits/experiments/__init__.py`
  - Exports `ExecutionMode` and `RunManifest`.

- `src/qudits_on_qubits/__init__.py`
  - Adds lazy top-level exports for `ExecutionMode` and `RunManifest`.

- `README.md`
  - Documents schema v2, `RunManifest.load()`, execution modes, custom backend requirements, and schema-v1 compatibility.

### New test files

- `tests/test_experiment_manifest.py`
  - Covers model validation, immutability, round trips, loading, migration, and invalid data.

- `tests/test_experiment_provider_contracts.py`
  - Exercises one scientific experiment across Aer, mocked IQM, and mocked PiastQ through the complete runner.

### Modified test files

- Existing tests that construct `CustomBackend` will pass an explicit `ExecutionMode`.
- Existing public-surface and README assertions will include the new public API.
- Existing schema-version assertions and manually assembled manifest fixtures will use schema v2 or explicit schema-v1 migration cases as appropriate.

## Execution Mode Contract

`ExecutionMode` is a string enum with exactly three values:

```python
class ExecutionMode(str, Enum):
    IDEAL_SIMULATOR = "ideal_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    HARDWARE = "hardware"
```

Built-in backend specifications have fixed modes:

- `AerIdeal`: `ideal_simulator`;
- `NoisySimulator`: `noisy_simulator`;
- `IQMHardware`: `hardware`;
- `PiastQHardware`: `hardware`.

`CustomBackend` gains a required keyword-only `execution_mode: ExecutionMode` field. It has no default. Inferring the mode from whether an adapter is local would be unsound because local execution does not distinguish ideal and noisy simulation.

Every backend specification includes `execution_mode` in `to_safe_dict()`. This nested value at `spec.backend.execution_mode` is the only serialized source of truth. `RunManifest.execution_mode` exposes it as a typed convenience property. The resolved `backend` record does not duplicate the value.

## RunManifest Contract

`RunManifest` is a frozen dataclass representing one complete or intermediate runner checkpoint. It covers these top-level fields:

- `schema_version`;
- `experiment_id`;
- `spec`;
- `status`;
- `timestamps`;
- `status_history`;
- `attempts`;
- `backend`;
- `jobs` and `job_ids`;
- `source`, `circuits`, and `counts`;
- `postprocessing` and `calibration`;
- `result` and `result_artifact`;
- `failure`.

Nested mappings and sequences are copied and deeply frozen during construction. Caller mutation after construction cannot alter the model. `to_safe_dict()` returns a fresh strict-JSON-compatible tree, so serialization cannot mutate the model.

The public construction and loading API is:

```python
manifest = RunManifest.from_safe_dict(document)
document = manifest.to_safe_dict()
manifest = RunManifest.load(artifact_dir)
mode = manifest.execution_mode
```

`RunManifest.load()` accepts a run artifact directory, not an arbitrary JSON filename. It delegates containment and reparse-point checks to `ExperimentStore`, reads `experiment.json`, normalizes supported legacy input, and returns a schema-v2 model. Loading alone never rewrites the source file.

## Manifest Invariants

The model enforces:

- input is a mapping with the exact supported schema after normalization;
- `experiment_id` is a safe non-empty identifier;
- `status` is an `ExperimentStatus` value;
- timestamps and status-history entries have the expected safe shape;
- attempts, jobs, counts, circuits, and artifact references have their required container shapes;
- all persisted values are finite, strict-JSON-compatible, and free of credential material;
- `spec.backend.execution_mode` is one of the three public modes;
- a built-in backend kind uses its fixed execution mode;
- resolved backend identity is compatible with the configured backend kind;
- `job_ids` agrees with durable job records when both are present;
- terminal snapshots have the corresponding result or failure state required by the existing runner lifecycle.

The model must accept every legitimate intermediate checkpoint emitted by the current runner, including the initial `created` snapshot where the resolved `backend` record is still `None`.

## Runner Integration

The runner may keep a mutable working document to avoid a broad state-machine rewrite. The persistence boundary becomes strict:

```text
mutable runner document
    -> RunManifest.from_safe_dict(document)
    -> deeply validated immutable snapshot
    -> snapshot.to_safe_dict()
    -> ExperimentStore.write_experiment(...)
```

This validation runs before every checkpoint write. Invalid state never replaces the last valid on-disk snapshot.

At backend resolution, the runner checks the configured backend kind and execution mode against the resolved adapter identity before compile or submit. Existing compile, submission, and result identity checks remain in force. There is no fallback from hardware to any simulator.

Resume reads the stored mapping, normalizes it through `RunManifest`, then gives the schema-v2 dictionary to existing resume logic. A nonterminal schema-v1 built-in run is upgraded on disk only when resume writes its next normal checkpoint. A completed schema-v1 run remains unchanged on disk because completed resume is idempotent.

## Schema-v1 Migration

Schema-v1 input is normalized in memory by copying the document, adding `spec.backend.execution_mode`, and setting `schema_version` to 2.

The mapping is deterministic for built-in backend kinds:

- `aer_ideal` becomes `ideal_simulator`;
- `noisy_simulator` becomes `noisy_simulator`;
- `iqm_hardware` becomes `hardware`;
- `piastq_hardware` becomes `hardware`.

A schema-v1 custom backend manifest is rejected with `ExperimentPersistenceError`. Its existing fields cannot identify hardware, ideal simulation, or noisy simulation, so migration must not guess. Unknown schema versions are also rejected.

## Provider Contract Tests

One base `ExperimentSpec` defines the `two_qutrit` scientific configuration, basis, shots, uncertainty settings, output location, and tags. The provider cases use `dataclasses.replace(base_spec, backend=...)`, so only the backend specification changes.

### Aer case

- Uses the real `AerAdapter` and local `qiskit-aer` simulator.
- Executes the complete runner without network access.

### IQM case

- Uses the real `IQMAdapter`.
- Injects a fake backend, pass manager, and durable job at the provider boundary.
- Exercises adapter resolution, compilation, submission, result extraction, and manifest persistence without contacting IQM.

### PiastQ case

- Uses the real `PiastQAdapter`.
- Injects a fake client, sampler, backend, and durable job at the provider boundary.
- Replaces only provider-specific compilation behavior that requires the unavailable remote backend plugin.
- Exercises compilation, submission, result extraction, and manifest persistence without importing credentials or waiting for a real queue.

Each case must:

1. prepare the same logical measurement circuits;
2. compile through its selected adapter;
3. complete submit and result stages;
4. finish with `ExperimentStatus.COMPLETED`;
5. load the saved document through `RunManifest.load()`;
6. expose schema version 2 and the same top-level manifest contract;
7. record the correct execution mode and provider identity;
8. preserve compiled QPY, counts, result artifacts, and their SHA-256 references;
9. make no network request;
10. prove that PiastQ produced a durable compiled circuit package before the mocked result completed.

Provider-varying values such as identity, job ID, local/remote status transitions, and timestamps are asserted per case. Shared structure and scientific configuration are asserted once through common helpers.

## Error Handling

- Invalid caller data passed directly to `RunManifest.from_safe_dict()` raises `ExperimentValidationError`.
- Runner and loader validation failures are exposed as `ExperimentPersistenceError`.
- Error messages identify the invalid field or invariant but never echo full payloads, credentials, provider exceptions, or unsafe paths.
- A mismatch among configured backend kind, execution mode, and resolved identity fails before compilation or submission.
- Existing provider-specific typed errors remain unchanged after the manifest boundary validates successfully.

## Testing Strategy

Development follows TDD:

1. add focused failing manifest tests;
2. implement the smallest model and migration behavior that passes them;
3. add failing checkpoint-integration tests;
4. integrate runner validation;
5. add failing provider contract cases one provider at a time;
6. implement only the seams required by those contracts;
7. run focused tests after each task;
8. run the complete suite before completion.

Coverage target: at least 80% line coverage for new `execution.py`, new `manifest.py`, and newly changed runner paths. Coverage is evidence that tests execute the new behavior, not a substitute for assertions about valid, invalid, migration, security, and provider-specific paths.

Required focused coverage includes:

- safe round trip and deep immutability;
- public loader behavior and path safety;
- every built-in schema-v1 migration;
- schema-v1 custom rejection;
- unknown schema rejection;
- invalid status, timestamp, execution mode, identity, artifact shape, and secret-bearing data;
- validation before each runner write;
- completed Aer, mocked IQM, and mocked PiastQ manifests;
- public exports and README examples.

The final verification command is the complete repository test suite. No live provider credential or network access is permitted.

## Documentation

README changes will document:

- the three `ExecutionMode` values;
- explicit `execution_mode=` for `CustomBackend`;
- `RunManifest.from_safe_dict()`, `to_safe_dict()`, and `load()`;
- schema version 2 and built-in schema-v1 normalization;
- rejection of ambiguous schema-v1 custom manifests;
- the fact that loading does not rewrite artifacts;
- the absence of hardware fallback and network access in contract tests.

## Rollout and Compatibility

- The feature is implemented on the Linear branch `szymonlysek123/szy-43-m31-ujednolicic-kontrakty-aer-iqm-i-piastq` created from `origin/main`.
- Existing real artifacts retain the filename `experiment.json`.
- Existing built-in schema-v1 artifacts remain loadable and resumable under the stated lifecycle rules.
- Existing `CustomBackend` construction is intentionally source-incompatible until the caller supplies an explicit execution mode.
- No external service is modified by implementation or tests.

## Acceptance Criteria

- `ExecutionMode` and `RunManifest` are public from both experiment and top-level package APIs.
- Every new manifest serializes schema version 2 and exactly one execution mode at `spec.backend.execution_mode`.
- `RunManifest` is immutable, validates persisted state, round-trips to safe dictionaries, and loads an artifact directory safely.
- Every runner checkpoint is validated before atomic persistence.
- Built-in schema-v1 manifests normalize in memory without an automatic write.
- Ambiguous schema-v1 custom manifests fail explicitly without guessing.
- `CustomBackend` requires an explicit execution mode.
- The same scientific experiment completes with real Aer, mocked IQM, and mocked PiastQ adapters.
- All three provider cases produce the shared manifest contract and durable circuit, counts, and result artifacts.
- PiastQ contract evidence includes a ready compiled circuit package without a real queue wait.
- Tests perform no provider network request and persist no credential material.
- New and changed manifest paths meet the 80% line-coverage target.
- The complete test suite passes.
