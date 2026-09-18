# Experiment runner

QuditsOnQubits is a Python library. It includes no dashboard, no web application, and no server.


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

## IQM automatic layout selection

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

## Direct pipeline and final artifact

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

## Bootstrap uncertainty

Default uncertainty is 2000 LOCAL resamples of saved counts, not 2000 backend experiments. Bootstrap never contacts a backend. Seeded runs calculate component-wise estimate, standard error, and confidence interval for every enabled conditional and unconditional result family, plus invalid-codeword evidence:

- `raw_conditional`, `raw_unconditional`, `raw_invalid_codeword_rate`, and `raw_invalid_codeword_shots`
- `readout_mitigated_conditional`, `readout_mitigated_unconditional`, and `readout_effective_invalid_codeword_weight`
- `zne_conditional` and `zne_unconditional`
- `zne_readout_mitigated_conditional` and `zne_readout_mitigated_unconditional`

The legacy `raw`, `readout_mitigated`, `zne`, and `zne_readout_mitigated` keys remain conditional aliases. Only enabled mitigation combinations appear. Intervals reflect finite-shot sampling and optional calibration resampling. They do not model hardware drift or ZNE model bias.

IQM/direct-basis simulation outputs belong under `artifacts/`, with selected best circuits copied into the relevant `selected_best/` folder. Curated hardware evidence and the Bell benchmark archive are retained for reproducibility.


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


## Independent local Bell settings in raw blocks

[The runnable notebook](../notebooks/bell_randomized_raw.ipynb) prepares portable
reference-state inputs for `two_qutrit`, `ghz3`, and `ame43` in two encodings,
runs local Aer comparisons, and includes disabled-by-default IBM, IQM, and
PIAST/AQT examples. Each comparison arm generates its own setting schedule.

```python
from qudits_on_qubits.experiments import (
    AerIdeal, ExperimentSpec, PathBasis, RandomizedBlocks,
    run_experiment, resume_experiment,
)

spec = ExperimentSpec(
    state="ame43",
    basis=PathBasis("path/to/basis"),
    backend=AerIdeal(seed_simulator=17),
    measurement=RandomizedBlocks(
        setting_draws=64,
        shots_per_draw=16,
        max_setting_draws=512,
        max_circuits_per_job=100,
        confidence_level=0.95,
    ),
)
result = run_experiment(spec)
print(result.values["raw"], result.values["conditional"])
restored = resume_experiment(result.artifact_dir)
```

The basis directory contains `graph_state_direct_basis.qpy` and `E.npy`; the
notebook can create these inputs locally. The example budget demonstrates the
API and is not a hardware-budget recommendation. `measurement=None` retains
the existing all-settings path and its default `shots=20480`. With
`RandomizedBlocks`, omit legacy `shots`, `uncertainty`, and `bootstrap`.
Raw v1 rejects readout mitigation, circuit twirling, ZNE, and forced
recalibration. Hardware examples require intentional opt-in and existing
provider account configuration.

`setting_draws` is the minimum number of blocks. Each block independently
draws one uniform setting per party using Python `secrets`, then records
`shots_per_draw` joint shots. System randomness is not a physical QRNG.
Repeated settings retain distinct block IDs. Drawing continues after the
minimum until all required Bell patterns are covered. Reaching
`max_setting_draws` first saves `coverage_limit_reached` without submitting any
circuits. Actual raw cost is `N_actual * shots_per_draw`; increasing shots per
block does not increase the number of independent blocks.

| Scenario | Full configurations | Required patterns | Configurations contributing to a term |
| --- | ---: | ---: | ---: |
| `two_qutrit` | 9 | 9 | 9 |
| `ghz3` | 18 | 12 | 12 |
| `ame43` | 36 | 13 | 15 |

An AME identity (`None`) is a wildcard in a required pattern: that party still
receives a setting and is measured. Its outcome is marginalized, while its
leakage invalidates the joint shot. One block can support multiple patterns.
Configurations with no matching term remain in raw data, total shot cost, and
global leakage statistics.

The main `raw` value gives leakage zero contribution. The additional
`conditional` value normalizes each correlator by its own accepted shots; it
is not the raw value divided by global acceptance. No accepted shots for a
required correlator gives `conditional=None` with `no_accepted_shots`, while
complete acquisition can still finish successfully. Results retain coverage,
budgets, per-block and per-pattern leakage, and AME diagnostics per full
measurement context.

Intervals use `conditional_schedule_block_hoeffding_v1`, conditional on the
saved setting schedule. They assume independent blocks, allow correlated
shots within a block, and can remain wide even for constant observations or
large shot counts. Classical-bound comparisons are diagnostic: faithful local
observables and stable measurements are needed for a stationary Bell-value
interpretation, with additional assumptions after postselection. This is not
a loophole-closing test and does not produce a nonlocality p-value.

Schema 4 artifacts keep the schedule, unique circuit catalogue, ordered batch
requests, job IDs, raw counts, and derived report. Completed runs reload
offline through `resume_experiment`; interrupted runs retrieve confirmed jobs
and continue untouched batches. Ambiguous submissions remain
`submission_unknown` rather than being silently repeated. Failed analysis
preserves raw data for local reanalysis. Existing all-settings artifacts and
the historical hardware benchmark keep their original formats.

`recover_randomized_job(path, batch_index=..., job_id=..., adapter=...)`
can attach an uncertain job only when the adapter verifies the provider's
saved circuits, backend, shot count, and raw options. IBM supports this proof;
the current IQM/PIAST clients do not expose enough evidence for unknown-job
attachment. Their already confirmed job IDs remain resumable. Lost local Aer
job handles cannot be restored; complete saved Aer counts remain usable offline.
For a lost local handle, explicitly call
`replay_randomized_aer_batch(path, batch_index=...)` to repeat that batch with
the saved QPY and seed. It retains the original attempt and extra shot budget,
rejects hardware targets and refuses to replace any existing raw data.
No run, including a local run, silently submits an uncertain block again.
