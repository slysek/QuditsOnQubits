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

## IQM Direct-Basis Transpilation

Create `.env` in the repository root:

```env
IQM_SERVER_URL=https://resonance.iqm.tech/
IQM_TOKEN=replace-with-your-iqm-api-token
```

Run a small IQM-backed direct-basis benchmark:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet
```

The `--iqm-backend` value is the IQM quantum computer name or alias. `garnet` is only an example. When this flag is present, the script loads one IQM backend for the whole run and compiles each candidate with Qiskit's preset pass manager using that backend:

```python
generate_preset_pass_manager(
    backend=backend,
    optimization_level=3,
    seed_transpiler=trial,
)
```

Optional transpiler controls:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --layout-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --routing-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-use-metrics
```

IQM output defaults to `artifacts/iqm_runs/raw`, and QPY exports default to `artifacts/iqm_runs/raw/quantum_circuits/<backend>/`.

