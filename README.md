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

## Frozen reference experiments

Use the frozen registry to inspect a Bell experiment without a notebook or
provider:

```python
from qudits_on_qubits import get_encoding, get_reference_experiment

spec = get_reference_experiment("ghz3")
statevector = spec.state.statevector()
measurement_settings = spec.measurement_settings()
encoding = get_encoding(spec.default_encoding_id)

print(spec.experiment_id)
print(statevector)
print(measurement_settings)
print(encoding.encoding_id)
print(spec.expected.ideal_bell_value)
print(spec.bell_functional.classical_bound)
print(spec.leakage_policy)
print(spec.stable_hash())
```

Canonical experiment IDs are `two_qutrit`, `ghz3`, and `ame43`; `2qutrit` is
an alias for `two_qutrit`. Every reference uses the default encoding ID
`canonical_ez`. Backend adapters normalize physical results to logical outcomes
`0`, `1`, `2`, or leakage. Analysis reports leakage before postselection and
both unconditional and conditional Bell values. The stable `spec.stable_hash()`
is available for backend metadata and regression tests.

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

## Direct-Basis Rerun Candidate Selection

Use preliminary benchmark CSVs to create per-state rerun inputs:

```powershell
python scripts/select_top_rerun_candidates.py `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_all_qutrit_u3_runs4_<timestamp>.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ghz3_all_qutrit_u3_runs4_<timestamp>.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ame43_all_qutrit_u3_runs1_<timestamp>.csv `
  --top-k 10 `
  --run-id stage2_20260706
```

By default this writes one CSV per `state_name` under `artifacts/iqm_runs/processed/rerun_selection/<run_id>/`. The `candidate` rows are the unique Top-K non-baseline-equivalent candidates by depth ranking. Baseline-equivalent and unresolved rows are still kept in the same file as diagnostics with `selection_role` values such as `baseline_equivalent_excluded` and `unresolved_candidate`; they are not rerun by `from-old-csv`.

Rerun one state with the selected baseline plus candidates:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state ghz3 `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/processed/rerun_selection/stage2_20260706/direct_basis_ghz3_stage2_20260706_top10_rerun_candidates.csv `
  --iqm-backend garnet `
  --n-transpile-runs 20 `
  --jobs 4
```

Repeat the rerun command for each generated state CSV. The rerun selector always includes the chosen baseline row, so each state is compared against its own rerun baseline.

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

The `--iqm-backend` value is the IQM quantum computer name or alias. `garnet` is only an example. When this flag is present, the script loads one IQM backend for the whole run and, by default, compiles each candidate with the same IQM strategy set used by the transpiler harness:

```text
preset_default
preset_exact
transpile_to_iqm_default
transpile_to_iqm_exact
```

For each candidate and seed, the benchmark tries the selected strategies and keeps the best transpiled circuit by `(depth, two_qubit_gate_count, one_qubit_gate_count, size)`. The output CSV records the winning `iqm_transpiler_strategy` and `iqm_transpiler_seed`.

Optional transpiler controls:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --layout-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --routing-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-use-metrics
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-strategy preset_default
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-legacy-pass-manager
```

IQM output defaults to `artifacts/iqm_runs/raw`, and QPY exports default to `artifacts/iqm_runs/raw/quantum_circuits/<backend>/`.

## IQM Transpiler Harness

Use the harness to compare IQM-aware transpilation strategies for candidates
selected by earlier benchmark CSVs:

```powershell
python scripts/run_iqm_transpiler_harness.py `
  --state two_qutrit `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_from_old_csv_runs20_20260706_204350.csv `
  --iqm-backend garnet `
  --n-transpile-runs 3
```

The harness only transpiles circuits. It does not submit jobs to IQM hardware.
It writes:

```text
artifacts/iqm_runs/processed/transpiler_harness/<run_id>/
  all_trials.csv
  best_by_candidate.csv
  summary.json
  quantum_circuits/<state>/<class>__<candidate>/
    F3_W.qpy
    CZ3_W.qpy
    graph_state_direct_basis.qpy
    graph_state_direct_basis_transpiled_<strategy>_seed<seed>.qpy
    E.npy
    W.npy
```

Pass `--quantum-circuits-dir` to override the artifact directory, or
`--no-export-quantum-circuits` to write only CSV/JSON outputs.

Built-in strategies:

```text
preset_default
preset_exact
transpile_to_iqm_default
transpile_to_iqm_exact
```

`best_by_candidate.csv` chooses the best successful trial by
`(depth, cz_count, r_count, size)` and flags warning thresholds such as
`depth_gt_100` and `cz_gt_50`.

## PiastQ AQT Bell Execution

Install the optional PiastQ integration in the environment used by this
project:

```powershell
python -m pip install -e ".[piastq]"
```

Choose `auto`, `managed`, or `direct` when constructing `PiastQClient`. The Bell
helper receives `client.backend` unchanged and does not choose or override the
execution mode. Credentials must come from environment variables; do not place
PCSS tokens or dashboard API keys in notebooks or source files.

This explicit smoke example prepares the zero state in the two-qutrit encoding,
builds every Bell-setting circuit required by the existing pipeline, and
submits one PiastQ job containing every generated circuit:

```python
import os

from qiskit import QuantumCircuit

from cft_piastq import PiastQClient
from qudits_on_qubits.bell_measurements import (
    build_sampler_circuits_for_candidate,
    canonical_Ez,
    compute_bell_value_from_counts_aqt,
)

state_circuit = QuantumCircuit(4)
sampler_circuits, metadata = build_sampler_circuits_for_candidate(
    candidate="two_qutrit",
    state_circuit=state_circuit,
    E=canonical_Ez(),
    qutrit_qubits=((0, 1), (2, 3)),
)

client = PiastQClient(
    mode=os.environ.get("CFT_PIASTQ_MODE", "auto"),
    owner=os.environ["CFT_PIASTQ_OWNER"],
    token=os.environ.get("PCSS_TOKEN") or os.environ.get("PCSS_QAPI_TOKEN"),
    dashboard_api_url=os.environ.get("CFT_PIASTQ_DASHBOARD_API_URL"),
    dashboard_api_key=os.environ.get("CFT_PIASTQ_DASHBOARD_API_KEY"),
)

bell_value, execution = compute_bell_value_from_counts_aqt(
    sampler_circuits,
    metadata,
    backend=client.backend,
    shots=20_480,
    sampler_options={"cft_job_name": "two-qutrit-bell-smoke"},
    timeout=900.0,
    poll_interval=5.0,
)

print("Bell value:", bell_value)
print("PiastQ job:", execution["job"].job_id())
```

`job.result()` remains available in `execution["result"]` as a Qiskit
`SamplerResult`. Bell postprocessing uses the estimated integer dictionaries
returned by `PiastQJob.counts()`; this project does not independently multiply
or round the quasi probabilities.

The example can contact the managed dashboard or direct AQT provider and can
consume real hardware shots. Run it only as an intentional manual smoke test.
