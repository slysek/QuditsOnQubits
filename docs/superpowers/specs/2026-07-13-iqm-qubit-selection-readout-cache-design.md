# IQM Qubit Selection and Readout Calibration Cache

## Scope

Update `notebooks/working/iqm/best_garnet_ghz.ipynb` so that each candidate circuit gets its own layout selected with IQM Qubit Selector. Add persistent readout-calibration caching so repeated notebook runs do not submit calibration jobs for qubits already present in the cache.

The implementation and its verification must not submit any job to `backend_garnet`. Connecting to the backend for metrics and transpiling against it are allowed. Hardware execution remains an explicit action performed later by the notebook user.

## Backend and Authentication

The notebook will construct `backend_garnet` with `IQMProvider` and rely on the `IQM_TOKEN` environment variable. The token currently embedded in the notebook will be removed. The exposed token must be revoked outside this code change.

The real backend will be used for current calibration metrics, layout selection, and transpilation. `IQMFakeGarnet` may remain available for local experiments, but it will not be passed accidentally as the production backend in the main candidate loop.

## Per-Candidate Qubit Selection

For every candidate:

1. Load its logical state circuit and associated data.
2. Pass the logical circuit and `backend_garnet` to `CostEvaluator`.
3. Select the first result from `get_top_layouts`.
4. Use IQM's `perform_backend_transpilation` with that layout and its reduced coupling map.
5. Build all sampler circuits for that candidate from the selected, transpiled state circuit.
6. Preserve the selected layout for all measurement settings and ZNE scale factors belonging to that candidate.

The result row will include the selected physical-qubit layout and selector cost. Different candidates may therefore run on different layouts, as explicitly requested.

If the selector produces no layout, the pipeline will fail before hardware execution with a clear error identifying the candidate.

## Readout Calibration Cache

`build_readout_calibration_matrices` will accept a cache path and `force_recalibration=False`. Its public return value remains the M3-compatible list whose indices correspond to backend qubit indices and whose unrequested entries are `None`.

The on-disk JSON document will be a dictionary partitioned by backend identity. Each qubit entry will contain:

- the 2x2 calibration matrix represented as JSON lists,
- the number of calibration shots,
- the UTC creation timestamp.

On each call, the function will:

1. Load and validate the JSON cache if it exists.
2. Reuse valid entries for requested qubits.
3. Build and execute calibration circuits only for missing qubits, or for every requested qubit when `force_recalibration=True`.
4. Update the cache atomically after successful calibration.
5. Convert cached matrices back to `numpy.float32` arrays for M3.

A malformed cache will raise a descriptive error instead of silently spending hardware time. Cache entries are reused regardless of their shot count because the requested behavior is to avoid reruns whenever a saved matrix exists. Manual refresh is available through `force_recalibration`.

The default cache location will be under `artifacts/iqm_runs/calibration/` and will be resolved from `repo_root`, so notebook working-directory changes do not redirect it unexpectedly.

## Dependencies

Add `iqm-qubit-selector>=1,<2` to both `pyproject.toml` and `requirements.txt`. The notebook will import `CostEvaluator` and `perform_backend_transpilation` from their documented IQM modules.

## Verification

Tests will exercise extracted notebook code with fake backend objects and temporary cache files. They will cover:

- a cache miss submits calibration circuits and persists matrices,
- a cache hit submits no job,
- a partial hit calibrates only missing qubits,
- backend identities do not share matrices,
- forced recalibration replaces cached entries,
- malformed cache data fails before backend execution,
- selector results are passed to transpilation for each candidate.

Notebook syntax and JSON structure will be validated after editing. No verification step will invoke `backend_garnet.run` or otherwise submit work to IQM hardware.
