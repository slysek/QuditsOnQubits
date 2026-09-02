# IQM Pareto Ranking and Global-Phase Deduplication Design

## Goal

Extend the existing IQM transpiler harness with reproducible post-transpilation
statistics that identify the best compiled circuits without hiding trade-offs.
The result combines non-dominated Pareto layers with a deterministic ranking
inside each layer. It also removes genuine global-phase duplicates while
retaining an audit trail from every removed candidate to its representative.

The initial scope covers the direct-basis IQM harness and its currently
supported benchmark states. Analysis is performed independently for each
state. Results from different devices, calibration snapshots, or harness runs
must not be mixed.

## Decisions

- All quality metrics and rankings are calculated from circuits produced by
  the IQM transpiler harness, never from logical pre-transpilation gate counts.
- The unit of statistical analysis is one
  `(state_name, class_name, candidate_name, strategy_name)` combination.
- Repeated transpiler seeds provide the samples used for means and standard
  deviations. Different strategies are not averaged together.
- The three minimized objectives are:
  `mean_two_qubit_gate_count`, `mean_depth`, and `std_depth`.
- Ranking inside a Pareto layer uses normalized weights 50/30/20 for those
  objectives respectively.
- Global-phase-equivalent candidate matrices are canonicalized and
  deduplicated before transpilation. State-equivalent compiled circuits from
  otherwise distinct candidates are grouped after transpilation, so an
  alternative implementation is not discarded before its physical cost is
  known.

## Existing Harness Boundary

`run_iqm_transpiler_harness` remains responsible for building logical
circuits, running every selected IQM strategy and seed, exporting transpiled
QPY artifacts, and returning `all_trials` plus the legacy
`best_by_candidate` table.

The new analysis consumes successful rows from `all_trials`. The existing
files remain available for backward compatibility. No backend submission or
hardware execution is introduced; the feature analyzes compilation output.

## Pre-Transpilation Candidate Deduplication

Two candidate embeddings are projectively equivalent when there is one
complex scalar `gamma`, with `abs(gamma) == 1`, such that
`candidate == gamma * reference` within configured numerical tolerances.
`E_new=None` is interpreted as the canonical baseline embedding.

Canonicalization uses the largest-magnitude matrix element as a stable pivot,
removes its phase, normalizes signed zeros, and rounds only for an initial
bucket key. Every bucket match is verified with a phase-aware `allclose`
comparison before candidates are merged. This avoids treating a hash or
rounding collision as equivalence.

Each equivalence group chooses a deterministic representative:

1. the explicit `baseline/E_old` candidate, when present;
2. otherwise the lexicographically smallest `(class_name, candidate_name)`.

The harness transpiles only representatives. A duplicate table records the
state-independent group identifier, representative identity, duplicate
identity, detected phase, and reason. Candidate counts in the summary report
raw, representative, and removed totals separately.

This step safely removes cases such as phase triples generated from one
monomial embedding. It does not merge arbitrary circuits merely because they
prepare the same state for one workload.

## Post-Transpilation State-Equivalence Grouping

Distinct logical constructions may prepare the same benchmark state but
compile differently. They are therefore allowed through the harness. After
transpilation, the best concrete trial for each candidate-strategy combination
is compared within the same state using the repository's layout-aware
statevector reconstruction and phase-invariant state equivalence. Comparisons
never cross state, device, calibration, run, or selection-label boundaries.

State-equivalent compiled alternatives receive one deterministic
`state_equivalence_group_id`. They remain visible in the detailed statistics.
For the compact recommendation table, the member with the best
`(pareto_rank, ideal_score, mean_two_qubit_gate_count, mean_depth, std_depth)`
tuple is retained and the other members point to it. This preserves the chance
for two equivalent logical preparations to compile differently while still
removing duplicate recommendations.

Circuits that cannot be reconstructed safely, including incompatible
measurement-bearing artifacts, are not silently merged. They receive a clear
equivalence status and remain independent.

## Statistical Aggregation

For every candidate-strategy combination, successful seed rows produce:

- `successful_trial_count` and `failed_trial_count`;
- `success_rate`;
- mean, minimum, maximum, and population standard deviation for depth;
- mean, minimum, maximum, and population standard deviation for two-qubit
  gate count;
- the identity and QPY path of the best concrete trial.

The new best concrete trial is selected by
`(two_qubit_gate_count, depth, one_qubit_gate_count, size, seed_transpiler)`.
This ordering follows the approved preference for two-qubit cost. It does not
change the legacy `best_by_candidate.csv`, whose historical depth-first
selection remains backward compatible.

One successful trial is allowed. Its population standard deviation is zero,
but `successful_trial_count=1` and `insufficient_stability_samples=true` make
the limitation explicit. Combinations with no successful trials remain in a
diagnostic table and cannot enter Pareto analysis.

Eligible metrics must be finite and non-negative. Invalid or missing metrics
produce a descriptive validation error naming the state, candidate, strategy,
and offending column.

## Pareto Layers

All three primary objectives are minimized independently. A row A dominates B
when A is no worse on every objective and strictly better on at least one.
Rank 1 contains the non-dominated front. After removing rank 1, the same rule
is repeated to assign rank 2 and subsequent layers.

Rows with identical objective values do not dominate each other. They share a
deterministic `pareto_metric_group_id` and remain available until the separate
state-equivalence recommendation step.

Pareto calculations are independent per state and harness run. No weighted
score can move a row to a better Pareto layer.

## Ranking Inside a Layer

Each objective is min-max normalized within the eligible rows for one state:

`normalized = (value - minimum) / (maximum - minimum)`

A constant objective contributes zero. The distance from the ideal point is a
weighted Manhattan distance:

`ideal_score = 0.50 * normalized_2q + 0.30 * normalized_depth + 0.20 * normalized_std_depth`

Lower scores are better. Deterministic ordering uses:

1. `pareto_rank`;
2. `ideal_score`;
3. `mean_two_qubit_gate_count`;
4. `mean_depth`;
5. `std_depth`;
6. `class_name`, `candidate_name`, and `strategy_name`.

The raw and normalized metrics are always exported, making the ranking
auditable and allowing future reweighting without retranspilation.

## Components

### Projective candidate equivalence

Add a focused direct-basis utility containing phase extraction,
phase-invariant matrix comparison, deterministic grouping, and duplicate
metadata construction. The harness calls it once before its candidate loop.

### Pareto analysis

Add `src/qudits_on_qubits/benchmarks/direct_basis/pareto_selection.py` with
pure DataFrame operations for validation, seed aggregation, Pareto masks and
layers, normalization, ideal scoring, state-equivalence grouping, and final
selection.

### Harness integration

Extend harness output writing without removing or changing the meaning of
existing output files. The analysis is deterministic and uses the in-memory
`all_trials` frame produced by the same run.

### CLI/post-processing

Provide a CLI capable of analyzing an existing harness `all_trials.csv`
without rerunning transpilation. This is the recovery path when compilation
has already completed or ranking weights change later.

## Outputs

The harness output directory gains:

- `candidate_global_phase_duplicates.csv`;
- `strategy_statistics.csv`;
- `pareto_ranked.csv`;
- `state_equivalence_groups.csv`;
- `recommended_circuits.csv`.

`summary.json` gains counts for raw candidates, representatives, removed
global-phase duplicates, analyzed strategy combinations, Pareto-front rows,
state-equivalence groups, and final recommendations. Existing keys remain
unchanged.

## Error Handling

- Empty inputs produce schema-correct empty outputs and zero counts.
- Duplicate trial identities
  `(state, class, candidate, strategy, seed_transpiler)` are rejected rather
  than double-counted.
- Mixed states or run metadata are partitioned only when the boundary is
  explicit; ambiguous mixed calibration/device data is rejected.
- Failed transpilation rows remain diagnostic and do not enter metric
  aggregation.
- Missing QPY files disable post-transpilation state grouping for the affected
  row but do not invalidate its numerical Pareto statistics.
- Numerical phase comparison uses explicit absolute and relative tolerances.

## Testing

Development follows red-green-refactor. Tests cover:

1. matrices equal exactly and up to arbitrary global phase;
2. matrices with relative phase differences remaining distinct;
3. deterministic representative selection and duplicate diagnostics;
4. the current monomial phase triples collapsing correctly;
5. aggregation across seeds without mixing strategies;
6. failed trials and single-sample stability diagnostics;
7. dominated, trade-off, and identical-metric Pareto cases;
8. multiple Pareto layers;
9. normalization, constant columns, 50/30/20 scoring, and deterministic ties;
10. state-equivalent compiled circuits selecting the best compiled member;
11. missing or unsafe QPY artifacts remaining ungrouped with diagnostics;
12. independent outputs for multiple states;
13. harness and post-processing CLI integration;
14. backward compatibility of `all_trials.csv`, `best_by_candidate.csv`, and
    existing summary keys.

Focused tests run first. The existing IQM harness, direct-basis selection, and
rerun-selection suites then run as regression coverage, followed by the wider
relevant project suite.

## Non-Goals

- No quantum-hardware job submission or execution.
- No comparison across devices, calibration snapshots, or benchmark states.
- No learned or Bayesian ranking model in this iteration.
- No deletion of raw trial rows or transpiled artifacts.
- No replacement of Pareto rank with the weighted score.
- No deduplication based only on identical metric values.
