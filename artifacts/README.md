# Artifacts

This directory is prepared for local experiment outputs.

- `iqm_runs/raw`: raw IQM simulation outputs.
- `iqm_runs/processed`: processed IQM summaries.
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
