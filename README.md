# QuditsOnQubits

QuditsOnQubits is a Python library for experiments with qudits encoded on qubit architectures. It provides circuit construction, direct-basis benchmarks, Bell measurements, backend adapters, durable artifacts, and local uncertainty analysis. It includes no dashboard, no web application, and no server.

## Install

Python 3.11--3.13 is supported. Install core library and ideal Aer backend:

```bash
python -m pip install -e .
```

M3 readout mitigation is optional:

```bash
python -m pip install -e ".[mitigation]"
```

PiastQ support requires the private `cft-piastq` package. Install `cft-piastq`
separately from a source available to you. QuditsOnQubits intentionally declares
no PiastQ package dependency and stores no private repository URL.

## Two-qutrit Bell vertical slice

After installation, run the public clean-room reference:

```bash
qoq-two-qutrit-bell --shots 2048 --seed 42 --output-root artifacts/vertical_slice_runs
```

The command starts from a logical two-qutrit Bell specification, applies the explicit `canonical_ez` encoding, builds and executes nine measured qubit circuits on ideal Aer, decodes the result, and writes an integrity-linked `run-manifest-v1`. Two-qutrit Bell is the reference benchmark, not the framework's functional boundary.

Run [the executable notebook](notebooks/two_qutrit_bell_vertical_slice.ipynb) for an annotated walk-through of the logical qutrit specification, explicit encoding, generated qubit circuits, execution, decoded Bell result, and verified manifest.

See [Two-Qutrit Bell Vertical Slice](docs/two_qutrit_bell_vertical_slice.md) for isolated installation, expected output, artifacts, verification, and troubleshooting.


## Unified experiment runner

`PathBasis` points to a directory containing an unmeasured `graph_state_direct_basis.qpy` and an isometric `(4, 3)` `E.npy` for `two_qutrit` (with state circuit widths adjusted for other states). Minimal ideal Aer run:

```python
from pathlib import Path

from qudits_on_qubits import (
    AerIdeal,
    BootstrapConfig,
    ExperimentSpec,
    PathBasis,
    run_experiment,
)

ideal = ExperimentSpec(
    state="two_qutrit",
    basis=PathBasis(Path("artifacts/bases/two_qutrit")),
    backend=AerIdeal(seed_simulator=11),
    shots=20_480,
    uncertainty=BootstrapConfig(samples=2000, seed=7),
)
result = run_experiment(ideal)
print(result.status, result.artifact_dir, result.values["raw"])
```

Use a structured `BenchmarkBasis` instead of manually locating a selected candidate:

```python
from dataclasses import replace
from qudits_on_qubits import BenchmarkBasis

selected = replace(
    ideal,
    basis=BenchmarkBasis(
        run_kind="direct_basis_runs",
        run_id="20260817-production",
        selection="exact",
        rank=1,
    ),
)
```

Backend choices keep simulation and hardware targets explicit:

```python
import os
from qudits_on_qubits import (
    CustomBackend,
    ExecutionMode,
    IQMHardware,
    NoisySimulator,
    PiastQHardware,
    TranspilationConfig,
)

# Local Aer execution using current IQM Garnet calibration profile and Garnet as compile target.
noisy_garnet = replace(
    selected,
    backend=NoisySimulator(source=IQMHardware(device="garnet")),
)

# Real IQM Garnet hardware with an explicit logical-to-physical layout.
real_garnet = replace(
    selected,
    backend=IQMHardware(device="garnet"),
    transpilation=TranspilationConfig(initial_layout=(16, 17, 18, 19)),
)

# PiastQ managed hardware. Credentials remain in environment/provider configuration.
piastq_managed = replace(
    selected,
    backend=PiastQHardware(
        mode="managed",
        owner=os.environ.get("CFT_PIASTQ_OWNER"),
    ),
)

# User-supplied backend object. Execution mode remains explicit.
custom = replace(
    selected,
    backend=CustomBackend(
        instance=my_backend,
        identity="laboratory-backend",
        execution_mode=ExecutionMode.HARDWARE,
    ),
)
```

Never put tokens, passwords, or API keys inline. Supply credentials only through environment variables or provider configuration. IQM uses its provider environment. PiastQ managed execution reads `CFT_PIASTQ_DASHBOARD_API_URL` and `CFT_PIASTQ_DASHBOARD_API_KEY`.

Run a batch in order, load a completed result, or finish saved postprocessing:

```python
from qudits_on_qubits import resume_experiment, run_experiments

results = run_experiments((ideal, noisy_garnet))
loaded = resume_experiment(results[0].artifact_dir)
```

`resume_experiment` loads completed schema-v3 direct results and completed legacy schema-v1/schema-v2 experiments without an adapter or backend call. A fresh schema-v3 run also publishes a `postprocessing` checkpoint after all requested counts, job metadata, workload selection, and optional calibration are durable. If bootstrap or final persistence is interrupted, `resume_experiment(checkpoint_dir, spec=matching_spec, ...)` recomputes postprocessing from those saved counts; it never retrieves or resubmits backend work. Custom/noisy specs require the matching `spec`. Runs using injected evaluators or mitigation strategies are intentionally not resumable. Other unfinished runs, including failures before complete counts, are rejected.

### IQM automatic layout selection

Configure IQM's calibration-aware selector through the public experiment API:

```python
from qudits_on_qubits import (
    IQMQubitSelectorConfig,
    WorkloadOptimizationConfig,
)

workload_optimization = WorkloadOptimizationConfig(
    initial_layouts=((0, 1, 2, 3, 4, 7),),
    seed_transpilers=(3, 7, 13),
    iqm_qubit_selector=IQMQubitSelectorConfig(
        top_k=10,
        num_trials=2000,
        cost_function="cz",
        readout_mode="none",
    ),
)
```

The IQM selector is a pipeline-level candidate source. With the tested `iqm-qubit-selector` 1.1.2 API, each returned value is an unordered physical routing subgraph, not an ordered logical-to-physical map. A subgraph may therefore contain more physical qubits than the logical circuit width. The pipeline sorts each subgraph, deduplicates candidates as sets, and keeps the first associated selector cost. While `iqm_qubit_selector` is enabled, explicit `initial_layouts` use the same routing-subgraph semantics; the sorted `(0, 1, 2, 3, 4, 7)` baseline above remains in the comparison. Outside selector mode, `TranspilationConfig(initial_layout=...)` remains an ordered logical-to-physical Qiskit mapping.

For each routing-subgraph×seed candidate, the IQM adapter calls `iqm.qiskit_iqm.transpile_to_IQM(..., restrict_to_qubits=list(subgraph))`. IQM returns a circuit indexed locally within that restriction, so the adapter inflates it to the backend's full width and restores real provider qubit indices before ranking, transforms, persistence, or submission. The pipeline evaluates every candidate against the complete Bell measurement workload and ranks the complete candidates before submission. Active physical qubits must stay inside the selected routing subgraph; with `require_exact_physical_qubit_set=True`, their union must equal it. All selector evaluation, candidate validation, and compilation happens before submission. Candidate-specific validation or compilation failures are recorded and skipped; fatal selector errors or a candidate set with no accepted compilation stop the run before any hardware job is submitted. Aer and PiastQ specifications reject IQM automatic layout selection instead of silently ignoring it.

### Direct pipeline and final artifact

Fresh runs use this pipeline:

1. Load the source basis and prepare all Bell measurement circuits in memory.
2. With workload optimization enabled, compile every configured layout×seed candidate across the complete Bell measurement workload and select the best candidate by calibrated or structural metrics. Without it, compile one batch. In IQM selector mode, each candidate is compiled with the official `iqm.qiskit_iqm.transpile_to_IQM` wrapper using `restrict_to_qubits`; other IQM paths use their configured transpilation options normally.
3. Submit the selected compiler-returned circuit objects directly through the adapter to `backend.run`. Optional readout calibration runs first; ZNE factor batches follow in order.
4. Keep counts in memory, ordered by ZNE factor and measurement setting, then run readout mitigation, ZNE, and bootstrap postprocessing.
5. After every requested job succeeds, atomically publish one schema-v3 `postprocessing` checkpoint. Run bootstrap, then atomically replace it with the completed `experiment.json`.

Each successful run gets a distinct UTC/UUID directory containing one file. An interrupted postprocessing run uses the same path and filename with `status: "postprocessing"` until resumed:

```text
artifacts/experiment_runs/YYYY-MM-DD/<experiment-id>/
  experiment.json
```

Schema-v3 `experiment.json` has this shape:

```text
experiment.json
  schema_version: 3
  experiment_id
  status: "completed"
  completed_at
  spec
  source
    provenance
    paths
  backend
  transpilation
  job_ids
  counts_by_factor
    "1"
      - setting
        counts
    "3"                         # only when requested by ZNE configuration
      - setting
        counts
  calibration                   # object with readout mitigation; otherwise null
  result
    raw                           # legacy alias of raw_conditional
    raw_conditional
    raw_unconditional
    raw_invalid_codeword_rate
    raw_invalid_codeword_shots
    readout_mitigated             # conditional alias; only when enabled
    readout_mitigated_conditional
    readout_mitigated_unconditional
    readout_effective_invalid_codeword_weight
    zne                           # conditional alias; only when enabled
    zne_conditional
    zne_unconditional
    zne_readout_mitigated         # conditional alias; only with both
    zne_readout_mitigated_conditional
    zne_readout_mitigated_unconditional
    config
    diagnostics
```

Fresh runs do not write separate compiled QPY files, source SHA-256 manifests, or multi-file status artifacts. The runner does not call a separate availability check or runner-level preflight before submission; adapter validation and the provider's `backend.run` boundary remain authoritative. A submit or result failure before complete counts leaves no artifact. A later postprocessing failure retains the inline checkpoint but no completed result. There is no silent fallback to ideal Aer or another target.

`RunManifest` is retained only as the immutable boundary for legacy schema-v1/schema-v2 checkpoint manifests. Its `from_safe_dict()`, `to_safe_dict()`, and `load()` methods do not model fresh schema-v3 results. Use `resume_experiment()` to load completed schema-v3 results and completed historical schema-v1/schema-v2 results.

The active checkpoint contains complete local counts, so `resume_experiment()` never retrieves a remote job and never resubmits work. Pre-count unfinished runs remain nonresumable. Preserve provider job IDs from the JSON for external audit or provider tooling.

### Bootstrap uncertainty

Default uncertainty is 2000 LOCAL resamples of saved counts, not 2000 backend experiments. Bootstrap never contacts a backend. Seeded runs calculate component-wise estimate, standard error, and confidence interval for every enabled conditional and unconditional result family, plus invalid-codeword evidence:

- `raw_conditional`, `raw_unconditional`, `raw_invalid_codeword_rate`, and `raw_invalid_codeword_shots`
- `readout_mitigated_conditional`, `readout_mitigated_unconditional`, and `readout_effective_invalid_codeword_weight`
- `zne_conditional` and `zne_unconditional`
- `zne_readout_mitigated_conditional` and `zne_readout_mitigated_unconditional`

The legacy `raw`, `readout_mitigated`, `zne`, and `zne_readout_mitigated` keys remain conditional aliases. Only enabled mitigation combinations appear. Intervals reflect finite-shot sampling and optional calibration resampling. They do not model hardware drift or ZNE model bias.

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

## PiastQ managed Bell execution

Install `cft-piastq` separately in the environment used by this project. The
QuditsOnQubits package metadata intentionally contains no private repository URL
and does not install `cft-piastq`.

The QuditsOnQubits integration is managed-only. `PiastQHardware` accepts
`mode="managed"`; `auto` and `direct` are rejected before any provider import or
network action. Configure `CFT_PIASTQ_DASHBOARD_API_URL`,
`CFT_PIASTQ_DASHBOARD_API_KEY`, and optionally `CFT_PIASTQ_OWNER` through the
environment. Do not place dashboard credentials in notebooks or source files.

Direct PCSS/AQT experiments require a separate environment and are not installed
by QuditsOnQubits.

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
    mode="managed",
    owner=os.environ["CFT_PIASTQ_OWNER"],
    dashboard_api_url=os.environ["CFT_PIASTQ_DASHBOARD_API_URL"],
    dashboard_api_key=os.environ["CFT_PIASTQ_DASHBOARD_API_KEY"],
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

The example contacts the managed dashboard and can consume real hardware shots.
Run it only as an intentional manual smoke test.
