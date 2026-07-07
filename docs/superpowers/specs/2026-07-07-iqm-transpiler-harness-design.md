# IQM Transpiler Harness Design

Date: 2026-07-07

## Context

The direct-basis IQM rerun CSVs show unusable Garnet circuits for selected two-qutrit candidates:

- Legacy IBM-style direct-basis CSV for the same candidate names reports about depth 41-44 and 16-18 two-qubit gates.
- `artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_from_old_csv_runs20_20260706_204350.csv` reports about depth 274-290 and 150-159 `cz` gates.
- A fresh compile in the `qudityD3_laptop` environment against the current real Garnet backend produced about depth 22-35 and 13-18 `cz` gates for the same 11 candidates.

This suggests the bad CSV is likely caused by transpiler version/configuration drift or by a suboptimal IQM scheduling path, not by an unavoidable Garnet hardware cost.

## Goal

Add an IQM transpiler harness that compares several IQM-aware transpilation strategies on a selected set of direct-basis candidates, records all metrics and version metadata, and selects the best strategy per candidate.

The first intended use is to take candidates selected by previous benchmark CSVs, especially `from-old-csv` rerun inputs, and validate which IQM transpilation path yields usable circuits.

## Non-Goals

- Do not submit jobs to IQM hardware.
- Do not rerun the full direct-basis candidate search.
- Do not immediately replace the main direct-basis benchmark default.
- Do not depend on network access in unit tests.

## Architecture

Add a separate harness next to the existing direct-basis benchmark pipeline. The harness will reuse existing candidate loading and circuit construction code, then run multiple IQM transpilation strategies over the same input circuits.

The main benchmark script remains stable. A later implementation can add an option to consume harness-selected strategies, but this design focuses on diagnostics, comparison, and regression protection.

## Components

### IQM Transpiler Strategies

Add a small strategy layer that exposes named transpilation paths:

- `preset_default`: `generate_preset_pass_manager(backend=backend, optimization_level=3, scheduling_method=None)`.
- `preset_exact`: `generate_preset_pass_manager(..., scheduling_method="move_routing_exact_global_phase")`.
- `transpile_to_iqm_default`: IQM helper `transpile_to_IQM(..., remove_final_rzs=True)`.
- `transpile_to_iqm_exact`: IQM helper `transpile_to_IQM(..., remove_final_rzs=False)`.

Each strategy returns either a transpiled `QuantumCircuit` or a structured failure with exception type and message.

### IQM Transpiler Harness

Add an orchestration module that:

- Loads candidate definitions using the current direct-basis candidate loaders.
- Builds direct-basis graph-state circuits using the current circuit builder.
- Runs selected strategies across one or more seeds.
- Computes metrics for each trial.
- Selects the best successful trial per candidate.
- Flags suspicious results and all-strategy failures.

The default ranking is `(depth, cz_count, r_count, size)`.

### CLI

Add `scripts/run_iqm_transpiler_harness.py` with options aligned with the existing benchmark script:

- `--state`
- `--n-qutrits`
- `--candidate-set`
- `--old-csv`
- `--iqm-backend`
- `--iqm-use-metrics`
- `--n-transpile-runs`
- `--strategy`, repeatable, defaulting to all built-in strategies
- `--output-dir`
- `--max-depth-warning`, default `100`
- `--max-cz-warning`, default `50`

The CLI will only transpile circuits. It will not run circuits on hardware.

## Data Flow

1. CLI parses state, candidate source, backend, strategy, and output settings.
2. Candidate loading reuses the same code path as `run_direct_basis_benchmarks.py`.
3. One IQM backend is loaded through the existing `load_iqm_backend`.
4. For each supported candidate, the direct-basis graph-state circuit is built.
5. For each seed and selected strategy, the harness runs transpilation.
6. The harness records trial metrics:
   - `depth`
   - `size`
   - `cz_count`
   - `r_count`
   - `one_qubit_gate_count`
   - `two_qubit_gate_count`
   - `count_ops_json`
   - `compile_time_seconds`
7. The harness records reproducibility metadata:
   - Python executable
   - Python version
   - Qiskit version
   - IQM client version
   - IQM Qiskit adapter import path
   - backend name
   - calibration set id
   - backend operation names
   - backend coupling-map size
8. All trials are written to `all_trials.csv`.
9. The best successful trial per candidate is written to `best_by_candidate.csv`.
10. Warnings are emitted when all strategies fail or metrics exceed warning thresholds.

## Output Layout

Default output root:

```text
artifacts/iqm_runs/processed/transpiler_harness/<run_id>/
  all_trials.csv
  best_by_candidate.csv
  summary.json
```

`run_id` defaults to a timestamp and can be overridden by the CLI if needed during implementation.

## Error Handling

- Missing `.env` or missing IQM values should fail early with the existing clear IQM environment error.
- Unsupported candidates from old CSVs should produce skipped rows with status `unsupported_candidate`, not crash the entire run.
- A failing strategy should create a failed trial row and continue with other strategies and candidates.
- If every strategy fails for a candidate, `best_by_candidate.csv` should contain a diagnostic row with status `failed_all_strategies`.
- CSV writes should be atomic enough for normal local use: create the output directory first and write complete CSV files after in-memory collection.

## Testing

Unit tests should not require real IQM credentials or network access.

Coverage:

- Strategy registry contains the expected strategy names.
- Strategy execution works with `IQMFakeGarnet` for a small direct-basis circuit.
- Harness records success rows with expected metric columns.
- Harness records failure rows when a strategy raises.
- Best-trial selection ranks by `(depth, cz_count, r_count, size)`.
- Warning flags are set for depth and CZ thresholds.
- CLI parser supports `from-old-csv`, backend name, strategy selection, and threshold options.

Optional manual verification with real credentials:

- Run the harness on the existing 11-row two-qutrit IQM rerun CSV.
- Confirm current Garnet results are in the expected range, about depth 22-35 and 13-18 `cz` for those candidates in the current environment.

## Success Criteria

- The harness can compare built-in IQM strategies for candidates loaded from previous CSVs.
- `all_trials.csv` clearly shows which IQM path produced high or low depth.
- `best_by_candidate.csv` identifies the best strategy for each candidate.
- The harness records enough package/backend metadata to explain future transpiler drift.
- Tests pass without real IQM credentials.

## Implementation Boundary

The implementation should be limited to the new harness, strategy helpers, CLI, tests, and README usage notes. The main benchmark default should not be changed in this iteration.
