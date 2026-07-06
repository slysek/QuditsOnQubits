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

For a fresh Codespace or virtual environment, prefer installing the package and
its pinned dependency set in one resolver run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -c "from iqm.qiskit_iqm import IQMProvider; print('iqm qiskit adapter ok')"
```

If the IQM adapter import still fails in a reused environment, remove stale IQM
packages first and reinstall the project:

```bash
python -m pip uninstall -y qiskit-iqm cirq-iqm iqm-client
python -m pip install --force-reinstall -e .
```

Run smoke tests:

```powershell
python -m unittest discover -s tests -v
```

## Direct-Basis Top-K Selection

Run a full direct-basis benchmark for one Bell-supported state and copy selected circuits:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state ghz3 `
  --candidate-set all-qutrit-u3 `
  --n-transpile-runs 20 `
  --jobs 4 `
  --approximation-thresholds 0.99,0.95,0.90 `
  --select-top-k 5
```

This runs `exact`, `fid099`, `fid095`, and `fid090`. Threshold labels pass `approximation_degree` into Qiskit transpilation; selected threshold rows must also satisfy `fidelity >= threshold`. `--jobs` runs independent candidates concurrently while keeping each candidate's exact/threshold exports serialized. Selected circuits are written under `artifacts/direct_basis_runs/selected_best/<state>/<run_id>/`.

For a fast smoke run:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state two_qutrit `
  --candidate-set sanity `
  --limit-candidates 3 `
  --n-transpile-runs 1 `
  --jobs 2 `
  --local-line-coupling `
  --approximation-thresholds 0.99,0.95,0.90 `
  --select-top-k 2
```

Small smoke candidate sets may warn that a threshold label selected fewer than `top-k` rows; that means the measured fidelity did not pass that threshold. The `exact` label still selects the best depth-ranked circuits.

Load the rank-1 transpiled circuit from a selected run:

```powershell
python scripts/load_best_circuit.py `
  --run-kind direct_basis_runs `
  --state two_qutrit `
  --run-id <printed_run_id> `
  --selection-label exact `
  --rank 1
```

## IQM Direct-Basis Transpilation

Create `.env` in the repository root:

```env
IQM_SERVER_URL=https://resonance.iqm.tech/
IQM_TOKEN=replace-with-your-iqm-api-token
```

Run a small IQM-backed direct-basis benchmark:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --jobs 4
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
