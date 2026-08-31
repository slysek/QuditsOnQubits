# IQM Qubit Selector Workload Layout Design

## Goal

Select a calibration-aware physical layout across the complete 20-qubit IQM
Garnet device without enumerating logical-to-physical permutations in this
repository. Use IQM's supported `iqm-qubit-selector` package to generate a
small set of promising layouts, then use the existing workload optimizer to
compile and rank those layouts over every GHZ Bell measurement circuit.

The selected layout must be reproducible, auditable, and fixed consistently
for Bell settings, readout calibration, circuit twirling, and ZNE. No hardware
job may be submitted when automatic layout selection fails or when every
candidate violates the physical-layout contract.

## Scope

This change applies automatic layout generation only to `IQMHardware` runs.
Explicit `initial_layouts` remain supported for deterministic tests, manual
experiments, and as comparison candidates. Aer and PIAST-Q behavior remains
unchanged.

The GHZ3 notebook enables automatic IQM selection. It searches the full Garnet
device rather than only permuting the existing physical set
`{0, 1, 2, 3, 4, 7}`. The existing ordered layout `(0, 1, 2, 7, 3, 4)` is kept
as an explicit baseline candidate, but only after the IQM selector completes
successfully.

## Selection Strategy

Selection has two stages.

### Stage 1: IQM calibration-aware preselection

The IQM adapter invokes `iqm.qubit_selector.qubit_selector.CostEvaluator` with
the resolved live backend and one deterministic representative circuit from
the complete logical Bell workload. The representative is the circuit with
the largest two-qubit-gate count, then largest depth, then lowest original
index. GHZ Bell measurement circuits share the same entangling preparation;
the deterministic rule avoids dependence on setting order when their costs
tie.

Default selector parameters are:

- `num_layouts=10`;
- `num_trials=2000`;
- `CostFunction.GATE_COST_CZ`, matching Garnet's native entangling gate;
- `ReadoutMode.NONE`, because the experiment performs readout mitigation;
- no removed qubits.

The selector returns ordered Qiskit layouts and costs. The adapter validates
their width, integer indices, uniqueness within each layout, uniqueness across
layouts, finite non-negative costs, and backend bounds. Invalid provider
output is a compatibility error.

The configured explicit baseline layouts are appended after generated layouts
and deduplicated without changing first occurrence. Selector costs remain
attached only to generated layouts.

### Stage 2: complete-workload compilation and ranking

The existing workload optimizer compiles every accepted layout with every
configured transpiler seed across all 12 GHZ Bell measurement circuits. The
default seeds remain `(3, 7, 13)`.

Each layout-seed candidate is rejected before submission when:

- transpilation fails;
- the compiled batch has the wrong circuit count;
- a physical measurement mapping is missing, partial, duplicated, or has the
  wrong width;
- any compiled circuit uses a physical-qubit set different from the requested
  candidate set.

Accepted candidates use the existing deterministic full-workload rank:

1. total calibrated instruction error cost, then duration, when complete
   target metrics exist for every candidate;
2. maximum and total two-qubit-gate count;
3. maximum and total depth;
4. transpiler seed;
5. layout tuple.

This makes the selector a candidate generator, not the final authority. A
layout with an attractive single-circuit selector cost can still lose or be
rejected after all Bell settings are compiled.

## Configuration Model

Add an immutable `IQMQubitSelectorConfig` with JSON-safe serialization:

- `top_k: int = 10`;
- `num_trials: int = 2000`;
- `cost_function: str = "cz"`, allowed values `"cz"` and `"clifford"`;
- `readout_mode: str = "none"`, allowed values `"none"`, `"fidelity"`, and
  `"qndness"`;
- `remove_qubits: tuple[int, ...] = ()`.

`WorkloadOptimizationConfig` gains
`iqm_qubit_selector: IQMQubitSelectorConfig | None = None`.
`initial_layouts` may be empty only when the selector configuration is present;
at least one explicit or generated source is always required. Existing safe
payloads without the new field continue to round-trip unchanged.

Supplying `iqm_qubit_selector` for a non-IQM adapter is rejected before
compilation or submission. No IQM enum or optional provider type leaks into
the serialized experiment specification.

## Component Boundaries

### Models

`experiments/models.py` owns validation and safe serialization of selector
configuration. It contains no import from `iqm-qubit-selector`.

### IQM adapter

`experiments/backends/iqm.py` owns the optional dependency import, translation
from safe string configuration to IQM enums, `CostEvaluator` invocation, and
normalization of provider output. A callable injection seam keeps unit tests
offline and deterministic.

The adapter exposes a focused `suggest_layouts(circuit, config)` method. Other
adapters do not implement it.

### Runner

`experiments/runner.py` selects the representative circuit, requests generated
layouts only when configured, merges them with explicit candidates, and passes
the merged set into the existing full-workload candidate loop. It remains
responsible for exact-set validation, final ranking, checkpoint metadata, and
ensuring that only the already-compiled winning batch is submitted.

## Metadata and Reproducibility

The `workload_optimization` artifact records:

- selector provider `iqm-qubit-selector` and installed version;
- calibration-set identifier from the resolved backend identity;
- normalized selector configuration;
- representative circuit index and name;
- every generated layout and selector cost in returned order;
- explicit baseline layouts and deduplication results;
- every layout-seed candidate status and workload metrics;
- selected layout, seed, ranking basis, and selected workload metrics.

Selector metadata is checkpointed before any hardware submission. Resume uses
the persisted selected compiled workload/job information and does not silently
run a fresh selector against a different calibration snapshot for an existing
submission. A new experiment run performs fresh selection.

No authentication token, server URL containing credentials, raw exception
message, or calibration payload is persisted.

## Failure Handling

When automatic selection is requested, these conditions fail closed before
submission:

- `iqm-qubit-selector` is unavailable or incompatible;
- calibration retrieval or selector evaluation fails;
- no generated layouts are returned;
- selector output is malformed;
- all merged layout-seed candidates fail full-workload validation.

Errors expose a stable category and backend identity but redact provider
exception text. `KeyboardInterrupt`, `SystemExit`, and `MemoryError` propagate.
The explicit baseline is a comparison candidate, not a silent fallback for a
failed selector.

## Dependency Compatibility

Use `iqm-qubit-selector>=1.1,<2` with `iqm-client[qiskit]>=35,<36`, matching the
IQM OS 4.6 client family used by the project. Tests verify the public selector
symbols and signatures relied upon by the adapter. Environment diagnostics
must detect user-site shadowing by an older IQM client rather than accepting an
inconsistent package set.

## Testing

Development follows red-green-refactor. Tests cover:

1. selector configuration normalization, validation, and safe round trips;
2. compatibility with legacy `WorkloadOptimizationConfig` payloads;
3. IQM enum translation and exact arguments passed to `CostEvaluator`;
4. normalization and rejection of malformed layouts or costs;
5. deterministic representative-circuit selection;
6. generated-plus-explicit candidate merge and stable deduplication;
7. Top-K layouts multiplied by every configured transpiler seed;
8. final selection based on the complete workload rather than selector cost;
9. exact physical-set rejection before submission;
10. selector failure aborting with zero submissions;
11. Aer and PIAST-Q rejection of IQM-only automatic selection;
12. JSON-safe selector metadata without credential or exception leakage;
13. resume behavior without a fresh selector call after submission;
14. GHZ3 notebook configuration using full-device automatic selection and the
    existing layout as a baseline;
15. an offline fake-selector pipeline test with no IQM hardware access;
16. an opt-in live IQM selector/compile smoke test that submits zero shots.

After focused tests, run the complete project test suite and an independent
review of the full diff. A later explicit hardware test may run the selected
layout with 100 shots; it is not part of automatic unit or CI execution.

## Non-Goals

- No in-repository brute-force permutation or topology optimizer.
- No selector-only Top-1 execution without full-workload validation.
- No automatic hardware submission during layout discovery.
- No change to Bell statistics, ZNE, twirling, or readout algorithms.
- No claim that structural or calibration cost guarantees Bell violation.
- No support for IQM Star/resonator architectures in this change; the target is
  Garnet's Crystal topology.
