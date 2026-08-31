# IQM Qubit Selector Workload Layout Design

## Goal

Select a calibration-aware physical routing subgraph across the complete
20-qubit IQM Garnet device without enumerating logical-to-physical permutations
in this repository. Use IQM's supported `iqm-qubit-selector` package to generate
a small set of promising routing subgraphs, then use the existing workload
optimizer to compile and rank those subgraphs over every GHZ Bell measurement
circuit.

The selected routing subgraph must be reproducible, auditable, and enforced
consistently for Bell settings, readout calibration, circuit twirling, and ZNE.
The measured mapping for every original circuit must also survive twirling and
ZNE unchanged. No hardware job may be submitted when automatic selection fails
or when every candidate violates the physical-routing contract.

## Scope

This change applies automatic routing-subgraph generation only to `IQMHardware`
runs.
Explicit `initial_layouts` remain supported for deterministic tests, manual
experiments, and as comparison candidates. Aer and PIAST-Q behavior remains
unchanged.

The GHZ3 notebook enables automatic IQM selection. It searches the full Garnet
device rather than only permuting the existing physical set. The existing
physical set `(0, 1, 2, 3, 4, 7)` is kept as an explicit routing-subgraph
baseline candidate, but only after the IQM selector completes successfully.
Despite the `initial_layouts` field name, explicit candidates use
routing-subgraph semantics whenever `iqm_qubit_selector` is enabled. Outside
selector mode, `TranspilationConfig(initial_layout=...)` retains its ordered
logical-to-physical Qiskit semantics.

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

`CostEvaluator.get_top_layouts` returns list-like qubit collections plus costs,
but with the tested `iqm-qubit-selector` 1.1.2 API each collection semantically
represents an unordered physical routing subgraph. It is not a
`qiskit.transpiler.Layout`, and its order does not map logical qubits to physical
qubits. A returned subgraph may be wider than the representative logical
circuit because routing may need an additional physical qubit. The adapter
requires at least the logical width and at most the backend capacity, validates
integer indices, uniqueness inside each subgraph, removed-qubit exclusions,
finite non-negative costs, and backend bounds, then sorts each subgraph and
deduplicates permutations as the same set. The first selector cost for a set is
preserved. Invalid provider output is a compatibility error.

The configured explicit baseline subgraphs are canonical-sorted, appended after
generated subgraphs, and deduplicated as sets without changing first occurrence.
Selector costs remain attached only to generated subgraphs.

### Stage 2: complete-workload compilation and ranking

The existing workload optimizer compiles every accepted routing subgraph with
every configured transpiler seed across all 12 GHZ Bell measurement circuits.
For each candidate, the IQM adapter calls
`transpile_to_IQM(..., restrict_to_qubits=list(subgraph))`; it does not pass the
candidate as `initial_layout`. The restricted transpiler output uses local
indices, so the adapter composes it into a full backend-width circuit whose
qubit indices are the real provider indices before ranking or submission. The
default seeds remain `(3, 7, 13)`.

Each routing-subgraph-seed candidate is rejected before submission when:

- transpilation fails;
- the compiled batch has the wrong circuit count;
- a physical measurement mapping is missing, partial, duplicated, or has the
  wrong width;
- active physical qubits escape the requested routing subgraph;
- `require_exact_physical_qubit_set=True` and the aggregate active-qubit union
  does not equal the requested routing subgraph.

Accepted candidates use the existing deterministic full-workload rank:

1. total calibrated instruction error cost, then duration, when complete
   target metrics exist for every candidate;
2. maximum and total two-qubit-gate count;
3. maximum and total depth;
4. transpiler seed;
5. canonical routing-subgraph tuple.

This makes the selector a candidate generator, not the final authority. A
subgraph with an attractive single-circuit selector cost can still lose or be
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

The adapter exposes a focused `suggest_layouts(circuit, config)` method and a
`compile_restricted(circuits, config, physical_qubits)` method that applies the
IQM restriction and restores provider indices. Other adapters do not implement
these selector-specific methods.

### Runner

`experiments/runner.py` selects the representative circuit, requests generated
routing subgraphs only when configured, merges them with explicit candidates,
and passes the merged set into the existing full-workload candidate loop. It
remains responsible for active-subgraph validation, final ranking, checkpoint
metadata, measurement-mapping invariants across twirling and ZNE, and ensuring
that only the already-compiled winning batch is submitted.

## Metadata and Reproducibility

The `workload_optimization` artifact records:

- selector provider `iqm-qubit-selector` and installed version;
- calibration-set identifier from the resolved backend identity;
- normalized selector configuration;
- representative circuit index and name;
- `layout_semantics: "routing_subgraph"`;
- every canonical generated subgraph and selector cost in returned order;
- canonical explicit baseline subgraphs and set-deduplication results;
- every subgraph-seed candidate status, plus active-qubit unions and workload
  metrics for accepted candidates;
- selected routing subgraph, active-qubit union, seed, ranking basis, and
  selected workload metrics.

Selector metadata is constructed, normalized, and validated before any
hardware submission. It is then included in the runner's existing
postprocessing checkpoint and final experiment artifact. Resume from an
existing checkpoint uses its persisted selection metadata and does not run a
fresh selector against a different calibration snapshot. A new experiment run
performs fresh selection. Changing the runner's current artifact-staging order
is outside this feature.

No authentication token, server URL containing credentials, raw exception
message, or calibration payload is persisted.

## Failure Handling

When automatic selection is requested, these conditions fail closed before
submission:

- `iqm-qubit-selector` is unavailable or incompatible;
- calibration retrieval or selector evaluation fails;
- no generated routing subgraphs are returned;
- selector output is malformed;
- all merged routing-subgraph-seed candidates fail full-workload validation.

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
4. canonicalization, set deduplication, and rejection of malformed routing
   subgraphs or costs;
5. deterministic representative-circuit selection;
6. generated-plus-explicit candidate merge and stable deduplication;
7. Top-K routing subgraphs multiplied by every configured transpiler seed;
8. final selection based on the complete workload rather than selector cost;
9. routing-subgraph containment and optional exact active-union rejection before
   submission;
10. selector failure aborting with zero submissions;
11. Aer and PIAST-Q rejection of IQM-only automatic selection;
12. JSON-safe selector metadata without credential or exception leakage;
13. resume behavior without a fresh selector call after submission;
14. GHZ3 notebook configuration using full-device automatic selection and the
    existing physical set as a baseline routing subgraph;
15. an offline fake-selector pipeline test with no IQM hardware access;
16. an opt-in zero-submit live IQM selector/compile smoke test that creates no
    hardware job.

After focused tests, run the complete project test suite and an independent
review of the full diff. A later explicit hardware test may run the selected
routing subgraph with 100 shots; it is not part of automatic unit or CI
execution.

## Non-Goals

- No in-repository brute-force permutation or topology optimizer.
- No selector-only Top-1 execution without full-workload validation.
- No automatic hardware submission during layout discovery.
- No change to Bell statistics, ZNE, twirling, or readout algorithms.
- No claim that structural or calibration cost guarantees Bell violation.
- No support for IQM Star/resonator architectures in this change; the target is
  Garnet's Crystal topology.
