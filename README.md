# QuditsOnQubits

Clean working repository for qutrit-on-qubit graph-state circuits, direct-basis encoding benchmarks, and qutrit Bell measurement pipelines.

This repo intentionally starts without historical bulk benchmark dumps. The previous repository should be kept locally as `QuditsOnQubits_legacy` for archival lookup. Future IQM/direct-basis simulation outputs should go under `artifacts/`, with selected best circuits copied into the relevant `selected_best/` folder.

## Layout

```text
src/qudits_on_qubits/
  core/              # graph-state and AME circuit helpers
  benchmarks/        # direct-basis benchmark code
  bell_measurements/ # qutrit Bell measurement pipeline
  encoding_search/   # candidate generation/search helpers
scripts/             # runnable entry points
notebooks/working/   # active research notebooks
artifacts/           # local result folders and manifest template
tests/               # smoke and regression tests
```

## Setup

```powershell
conda activate qudityD3_laptop
pip install -r requirements.txt
pip install -e .
```

Run smoke tests:

```powershell
python -m unittest discover -s tests -v
```

