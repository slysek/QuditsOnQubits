# Direct Basis Rerun Selection Design

## Goal

Create a reproducible second-stage selection workflow for preliminary direct-basis benchmark CSV files. The workflow must select rerun candidates separately for each state, always include the baseline for comparison, exclude baseline-equivalent candidates from the Top-K rerun set, and still record those excluded baseline-equivalent rows for diagnostics.

The immediate use case is taking preliminary IQM CSV files such as `two_qutrit`, `ghz3`, and `ame43`, selecting promising candidates per state, and rerunning each selected per-state set with a larger transpilation count such as `--n-transpile-runs 20`.

## Scope

In scope:

- Read one or more preliminary benchmark CSV files.
- Group rows by `state_name`.
- Write one selection CSV per state.
- Select Top-K non-baseline, non-baseline-equivalent candidates per state.
- Always include the baseline row in each per-state selection CSV.
- Include baseline-equivalent rows in the same per-state CSV as excluded diagnostic rows.
- Annotate each row with baseline comparison fields.
- Keep reruns compatible with the existing `run_direct_basis_benchmarks.py --candidate-set from-old-csv --old-csv ...` path.
- Make the `from-old-csv` candidate loader ignore diagnostic rows when a selector CSV contains `selection_role`.

Out of scope:

- Computing Bell values.
- Running the second-stage benchmark automatically inside the selector.
- Changing the existing raw benchmark result schema unless later implementation needs a small metadata addition.
- Combining multiple states into one rerun output CSV.

## Architecture

Add a small selection module and a thin CLI:

```text
src/qudits_on_qubits/benchmarks/direct_basis/rerun_selection.py
scripts/select_top_rerun_candidates.py
```

`rerun_selection.py` owns CSV loading, validation, baseline discovery, baseline-equivalence annotation, ranking, and per-state file writing.

`select_top_rerun_candidates.py` only parses arguments, calls the module, and prints the generated per-state CSV paths and warnings.

The existing benchmark CLI remains responsible for rerunning circuits:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state two_qutrit `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/processed/rerun_selection/<run_id>/two_qutrit_top10_plus_baseline.csv `
  --iqm-backend garnet `
  --n-transpile-runs 20
```

To keep one CSV per state while also storing diagnostic rows, `candidates_from_old_csv()` should preserve its current behavior for old CSV files without `selection_role`. When `selection_role` exists, it should rerun only rows where `selection_role` is `baseline` or `candidate`; rows such as `baseline_equivalent_excluded` and `unresolved_candidate` stay in the file for inspection but are not benchmarked in stage 2.

## Output Layout

The selector writes a separate directory per selection run:

```text
artifacts/iqm_runs/processed/rerun_selection/<run_id>/
  two_qutrit_top10_plus_baseline.csv
  ghz3_top10_plus_baseline.csv
  ame43_top10_plus_baseline.csv
```

The second-stage benchmark should also be run separately per state so output stays split like the current preliminary files:

```text
artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_top10_plus_baseline_runs20_<timestamp>.csv
artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ghz3_top10_plus_baseline_runs20_<timestamp>.csv
artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ame43_top10_plus_baseline_runs20_<timestamp>.csv
```

## Selection Rules

For each `state_name`:

1. Find baseline rows where `class_name == "baseline"`.
2. Prefer `candidate_name == "E_old"` when more than one baseline row exists.
3. If more than one baseline row remains, choose the best by the same ranking used for candidates.
4. Build the candidate pool from rows with `status == "ok"` and `success == true`.
5. Exclude the selected baseline row from the candidate pool.
6. Exclude baseline-equivalent candidates from the Top-K candidate pool.
7. Rank the remaining candidate pool and take Top-K.
8. Write one per-state CSV containing:
   - baseline row first with `selection_role = "baseline"` and `selection_rank = 0`
   - Top-K selected rows with `selection_role = "candidate"` and ranks `1..K`
   - excluded baseline-equivalent rows with `selection_role = "baseline_equivalent_excluded"` and no Top-K rank
   - unresolved rows with `selection_role = "unresolved_candidate"` and no Top-K rank

Baseline-equivalent rows are intentionally saved, especially for states like `ghz3`, where many low-depth rows can be equivalent to baseline. They must not consume Top-K rerun slots.

## Ranking

Top-K candidates are sorted ascending by:

```text
best_depth
best_two_qubit_gate_count
mean_depth
std_depth
best_one_qubit_gate_count
best_size
candidate_name
```

This ranking prioritizes low best depth but uses mean and standard deviation to reduce the chance that a single lucky transpiler seed dominates the rerun set.

After the 20-run rerun, final analysis should compare both `best_depth` and stability fields such as `mean_depth` and `std_depth` against the rerun baseline.

## Baseline Equivalence

The selector must determine whether each row is baseline-equivalent before Top-K ranking.

Rules:

- If source CSV already contains `is_baseline_equivalent` and `is_baseline_reference`, use those fields.
- If those fields are missing, regenerate known candidates by `class_name` and `candidate_name`, convert the direct-basis matrix to the physical encoding with `encoding_embedding()` when needed, and use the existing baseline-equivalence logic from `qudits_on_qubits.encoding_search.triviality`.
- If a candidate cannot be regenerated, keep it out of Top-K and write it with `selection_role = "unresolved_candidate"`.

The baseline reference itself remains included as `selection_role = "baseline"` even though it is baseline-equivalent by definition.

## Output Columns

Each per-state CSV must include at least:

```text
state_name
selection_role
selection_rank
class_name
candidate_name
is_baseline_reference
is_baseline_equivalent
baseline_equivalence_reason
best_depth
mean_depth
std_depth
best_two_qubit_gate_count
best_one_qubit_gate_count
best_size
baseline_best_depth
depth_delta_vs_baseline
depth_ratio_vs_baseline
baseline_relation
source_csv
```

`baseline_relation` values:

```text
baseline
better
equal
worse
excluded_baseline_equivalent
unresolved
```

The CSV may preserve additional source columns after these required columns, but the required columns should appear first for easy inspection.

## CLI

Proposed selector command:

```powershell
python scripts/select_top_rerun_candidates.py `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_all_qutrit_u3_runs4_20260706_101248.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ghz3_all_qutrit_u3_runs4_20260706_101411.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ame43_all_qutrit_u3_runs1_20260706_102324.csv `
  --top-k 10 `
  --run-id stage2_20260706
```

Default output root:

```text
artifacts/iqm_runs/processed/rerun_selection/<run_id>/
```

Optional arguments:

- `--output-root` to override the processed selection directory.
- `--top-k` with default `10`.
- `--run-id` with default timestamp.
- `--include-label exact` if future CSV files contain approximation labels and only one label should be selected.

## Error Handling

The selector fails with a clear `ValueError` when:

- An input CSV is missing `state_name`, `class_name`, `candidate_name`, or `best_depth`.
- A state has no baseline row.
- `--top-k` is less than 1.
- No input CSV path is provided.

The selector completes with warnings when:

- A state has fewer than Top-K eligible non-equivalent candidates.
- More than one baseline row exists and the selector picks the best baseline row.
- A candidate cannot be regenerated for baseline-equivalence detection.

Warnings should be printed and included in a small summary object returned by the module so tests can assert them.

## Testing

Unit tests should cover:

- Per-state selection with baseline, better, equal, worse, and baseline-equivalent candidates.
- Baseline is always rank `0` and does not count toward Top-K.
- Baseline-equivalent candidates are written with `selection_role = "baseline_equivalent_excluded"` and do not count toward Top-K.
- Separate output files are written per state.
- Missing baseline raises a clear error.
- Multiple baseline rows produce a warning and deterministic baseline choice.
- CLI writes per-state files from temporary CSV inputs.
- Generated per-state CSV files contain `class_name` and `candidate_name` so they remain compatible with `--candidate-set from-old-csv`.
- `from-old-csv` keeps legacy behavior for CSV files without `selection_role`.
- `from-old-csv` reruns only `baseline` and `candidate` rows for selector CSV files with `selection_role`.

## Acceptance Criteria

- Running the selector on preliminary `two_qutrit`, `ghz3`, and `ame43` CSV inputs writes three separate selection CSV files.
- Each per-state file contains exactly one baseline row, up to Top-K selected non-equivalent candidates, and any excluded baseline-equivalent diagnostic rows.
- Top-K selection never includes baseline-equivalent candidates.
- Selector CSV files may include excluded diagnostic rows, but `from-old-csv` must not rerun those rows when `selection_role` is present.
- Existing rerun command with `--candidate-set from-old-csv --old-csv <state_selection.csv> --n-transpile-runs 20` can rerun each state separately.
- The final rerun outputs remain one CSV per state.
