# IQM Pareto Selection Design

## Goal

Create reproducible Pareto-front CSVs from completed IQM direct-basis benchmark CSVs without rerunning transpilation. Produce one independent output file per state. The first run covers `two_qutrit` and `ghz3`; the same tool will support `ame43` after its benchmark CSV is complete.

Score calculation and Top-K selection are explicitly out of scope. They will consume Pareto outputs in a later change.

## Inputs

The command accepts one or more raw benchmark CSV paths. Each state must have exactly one complete candidate row per `(class_name, candidate_name)` after filtering to `selection_label=exact`. Duplicate candidate rows for a state are rejected because combining distinct runs or calibration snapshots would make the metrics incomparable.

Required columns:

- `state_name`
- `selection_label`
- `class_name`
- `candidate_name`
- `status`
- `success`
- `fidelity`
- `mean_two_qubit_gate_count`
- `mean_depth`
- `std_depth`

The loader records `source_csv` on every row.

## Eligibility Filters

Candidates enter Pareto analysis only when all conditions hold:

1. `selection_label == "exact"`.
2. `status == "ok"`.
3. `success` is true using the repository's existing boolean normalization rules.
4. `fidelity` is finite and within `1e-9` of `1.0`.
5. The row is not the baseline reference.
6. The encoding is not baseline-equivalent.
7. All Pareto metrics are finite and non-negative.

Baseline equivalence is derived with the existing direct-basis candidate metadata logic. An unresolved candidate is an error: it must not silently enter or disappear from the front.

Missing columns, duplicate exact rows, missing baseline-equivalence resolution, or invalid metrics produce a clear `ValueError` naming the state and offending candidate where applicable.

## Pareto Definition

All three objectives are minimized independently:

- `mean_two_qubit_gate_count`
- `mean_depth`
- `std_depth`

Candidate A dominates candidate B when A is no worse on every objective and strictly better on at least one objective. A candidate belongs to the Pareto front when no eligible candidate dominates it.

Rows with identical objective values do not dominate one another. All remain in the output because their encodings may differ physically. They receive the same deterministic `pareto_group_id`.

No weighted score, Pareto rank, or arbitrary Top-K tie-breaking is added.

## Components

### Core module

Add `src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py` with focused operations for:

- loading and validating input CSVs;
- filtering eligible exact candidates;
- calculating a minimization Pareto mask;
- assigning deterministic metric-group identifiers;
- selecting a front for one state;
- writing independent files for all states present in the input.

The numerical Pareto function accepts a DataFrame plus metric column names, making dominance behavior testable without filesystem or IQM dependencies.

### CLI

Add `scripts/select_pareto_candidates.py` with:

- repeatable `--input-csv`;
- optional `--output-root`, defaulting to `artifacts/iqm_runs/processed/pareto`;
- optional `--run-id`, defaulting to a timestamp.

The CLI always selects `exact`; it exposes no label override. This prevents accidental mixing with `fid090`, `fid095`, or `fid099` rows.

### Outputs

Each state receives a separate file:

```text
artifacts/iqm_runs/processed/pareto/<run_id>/
  direct_basis_two_qutrit_<run_id>_pareto_exact.csv
  direct_basis_ghz3_<run_id>_pareto_exact.csv
```

When a complete `ame43` CSV is supplied later, the same command additionally writes:

```text
direct_basis_ame43_<run_id>_pareto_exact.csv
```

Each output preserves source benchmark columns and adds:

- `source_csv`
- `pareto_front` set to true
- `pareto_group_id`, formatted `P001`, `P002`, and so on
- `pareto_group_size`

Rows are sorted deterministically by the three Pareto metrics, then `class_name` and `candidate_name`. The CLI reports eligible row count, front row count, and metric-group count per state.

If no completed CSV for a state is supplied, no placeholder file is created. Therefore the first run writes only `two_qutrit` and `ghz3` outputs.

## Testing

Use test-first development. Tests cover:

1. A dominated point is excluded.
2. Trade-off points remain on the front.
3. Identical metric points remain and share one group ID.
4. Exact, status, success, fidelity, baseline, and baseline-equivalence filters.
5. Missing or non-finite metrics fail clearly.
6. Duplicate candidates for one state fail clearly.
7. Multiple states produce independent output files.
8. CLI smoke behavior with repeated inputs.

Focused Pareto tests run first, followed by the full relevant direct-basis test suite. The generated `two_qutrit` and `ghz3` CSVs are then checked for schema, row counts, deterministic ordering, and absence of dominated rows.

## Non-Goals

- No benchmark or transpilation rerun.
- No `ame43` reconstruction from incomplete logs or QPY files.
- No score or Top-K selection.
- No copying of selected circuit artifacts.
- No cross-state aggregation.
- No mixing of calibration snapshots within one state's front.
