# Artifacts

This directory is prepared for local experiment outputs.

- `iqm_runs/raw`: raw IQM simulation outputs.
- `iqm_runs/processed`: processed IQM summaries.
- `iqm_runs/processed/rerun_selection`: per-state Top-K rerun input CSVs generated from preliminary IQM results.
- `iqm_runs/selected_best`: selected best IQM-derived circuits/results.
- `direct_basis_runs/raw`: raw direct-basis benchmark outputs.
- `direct_basis_runs/processed`: processed direct-basis summaries.
- `direct_basis_runs/selected_best`: selected best direct-basis circuits/results.

Historical bulk QPY/NPY dumps are intentionally not copied here during the initial cleanup.

## Selected Direct-Basis Circuits

`direct_basis_runs/selected_best` contains timestamped selected-circuit runs:

```text
direct_basis_runs/selected_best/<state>/<run_id>/<selection_label>/rankNN_<class>__<candidate>/
  graph_state_direct_basis.qpy
  graph_state_direct_basis_transpiled.qpy
  E.npy
  W.npy                    # present when the raw candidate was a 3x3 direct-basis W
  F3_W.qpy
  CZ3_W.qpy
```

`<selection_label>` is `exact`, `fid099`, `fid095`, or `fid090`.

Each selected run has one shared manifest:

```text
direct_basis_runs/selected_best/<state>/<run_id>/manifest.csv
```

Manifest paths are relative to the repository root. The manifest is the handoff table for later Bell inequality computation: it records the selected state, run id, label, rank, candidate identifiers, benchmark metrics, and artifact paths.

## Rerun Selection CSVs

`iqm_runs/processed/rerun_selection/<run_id>/` contains one CSV per state produced by `scripts/select_top_rerun_candidates.py`:

```text
iqm_runs/processed/rerun_selection/<run_id>/direct_basis_<state>_<run_id>_top10_rerun_candidates.csv
```

These files are intended for `scripts/run_direct_basis_benchmarks.py --candidate-set from-old-csv --old-csv <state-csv>`. Only rows with `selection_role` equal to `baseline` or `candidate` are rerun; diagnostic rows such as `baseline_equivalent_excluded` and `unresolved_candidate` remain in the CSV for auditability.
