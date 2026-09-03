# Optimal F3 Leakage-Phase Benchmark Design

## Context

The direct-basis benchmark currently prepares each local qutrit `|+>` state with
`StatePreparation`. It exports an encoded `F3_W` gate, but that gate is not part
of the graph-state circuit whose compiled metrics determine the benchmark row.
Consequently, changing the phase assigned to the unused fourth state can change
the standalone decomposition of `F3_W` without changing the measured full
graph-state circuit.

For a monomial qutrit encoding

```text
E = B_s D P,
```

the unused physical state may instead receive a basis-dependent phase which
reduces the compiled cost of the encoded qutrit Fourier gate. The benchmark must
compare this extension with the historical identity extension in otherwise
identical full graph-state circuits.

## Goals

- Compute the analytic optimal unused-state phase independently for every
  monomial encoding.
- Build graph-state preparation circuits in which every local `F3` is explicit.
- Compare full baseline and optimal graph-state circuits with paired transpiler
  seeds and backend strategies.
- Preserve the current direct-basis ranking and output when the comparison is
  not requested.
- Record enough phase, metric, and artifact data to audit and reproduce the
  comparison.

## Non-goals

- Optimizing the leakage completion of `CZ3` or any other gate.
- Applying the analytic formula to dense or otherwise non-monomial encodings.
- Replacing the existing direct `StatePreparation` ranking circuit.
- Submitting circuits to quantum hardware as part of this change.
- Rewriting existing generated notebooks or historical benchmark artifacts.

## Mathematical Convention

The benchmark accepts either a `3 x 3` direct-basis unitary or a `4 x 3`
encoding isometry. For phase optimization it first resolves the physical
encoding `E` and verifies that it is monomial: every logical column has one
nonzero entry, and those entries occupy three distinct physical rows.

Let `support` be those three physical rows in ascending physical basis order and
let

```text
S = E[support, :].
```

For an encoding generated as `B_s D P`, this removes the support embedding
`B_s` while retaining its effective physical ordering. The qutrit block used by
the formula is

```text
C = S F3 S†.
```

Using mathematical one-based indices, define

```text
p  = C_12 C_21
z* = p / (det(C) conjugate(p))
φ* = arg(z*) mod 2π.
```

In zero-based NumPy notation, `p = C[0, 1] * C[1, 0]`. The implementation
normalizes `z*` to unit modulus to remove floating-point drift before extracting
the angle. A numerically zero `p` is rejected because the formula would be
undefined.

For the generated `monomial_full` family, diagonal `D` phases cancel. The six
effective permutations therefore produce only

```text
φ* = π/2 or φ* = 11π/6.
```

This two-value property is an invariant tested across every support,
permutation, and generated diagonal phase.

The physical encoded Fourier operator is

```text
U_F3(φ) = E F3 E† + exp(iφ) (I4 - E E†).
```

The baseline uses `φ = 0`; the optimized variant uses the analytic `φ*`.
Both have exactly the same logical action.

## Considered Approaches

### 1. Paired full-circuit comparison (selected)

Add an opt-in benchmark path that constructs two complete graph-state circuits.
Both prepare the encoded logical `|0>`, apply an explicit encoded `F3` to each
qutrit, and then apply the same encoded `CZ3` edge gates. Their only difference
is `φ = 0` versus `φ = φ*`.

This directly measures the optimization requested, keeps identical compilation
conditions, and does not invalidate historical ranking columns. Its cost is an
additional pair of transpilation runs for each eligible candidate.

### 2. Replace the primary benchmark circuit

The primary graph-state circuit could be changed from direct `|+>` preparation
to the optimized `F3` construction. This is simpler at runtime, but changes the
meaning of existing ranking columns and prevents direct comparison with stored
results. It also removes the baseline arm unless another circuit is added.

### 3. Compare only standalone F3 gates

This is inexpensive and useful as a diagnostic, but it does not measure the
compiled graph-state circuit. It is therefore insufficient as the main result.

## Architecture

### Phase analysis

`benchmarks/direct_basis/math_utils.py` will own the pure numerical operations:

- validation and extraction of the effective monomial `S` matrix;
- construction of `C = S F3 S†`;
- calculation of `z*` and `φ*`;
- phase-aware single-qutrit physical embedding.

The phase helper returns a small immutable result containing the angle, complex
phase, physical support, and effective qutrit block. Keeping this logic free of
Qiskit makes it fast and exhaustively testable.

### Graph-state circuits

`benchmarks/direct_basis/circuits.py` will retain the historical
`build_direct_basis_graph_state_circuit` unchanged. A dedicated Fourier-based
builder will:

1. prepare `E |0>` on each two-qubit qutrit block;
2. append `U_F3(φ)` to every block;
3. append the same encoded `CZ3` gates and use the same logical-to-physical edge
   ordering as the existing builder.

For monomial encodings, `E |0>` is a computational basis state up to global
phase. Using the existing `StatePreparation` abstraction keeps the builder valid
for both `3 x 3` and `4 x 3` representations and gives the two comparison arms
identical pre-Fourier operations.

### Paired compilation

`benchmarks/direct_basis/benchmark.py` will add a focused comparison routine.
For each transpiler trial and, where applicable, each IQM strategy, it compiles
the baseline and optimized graph circuits with the same seed and options. A
trial contributes metrics only if both arms compile successfully. Pairing avoids
mistaking seed or strategy variation for an effect of the leakage phase.

The comparison ranks each arm with the same graph-state metric order already
used by the active backend path. It reports best and mean depth, size, one-qubit
gate count, and two-qubit gate count, along with optimal-minus-baseline deltas.
The primary benchmark's existing ranking circuit and columns remain unchanged.

### CLI and applicability

`scripts/run_direct_basis_benchmarks.py` will expose
`--compare-optimal-f3-leakage`. The flag is disabled by default because a full
monomial search contains hundreds of candidates and the paired comparison adds
substantial transpilation work.

When enabled:

- monomial encodings are compared automatically;
- non-monomial rows receive `not_monomial` comparison status and are otherwise
  benchmarked normally;
- invalid monomial data or an undefined formula produces a row-level comparison
  error rather than aborting the candidate suite.

No user-supplied phase is accepted by this path. The benchmark records the phase
derived from the candidate itself.

## Result Contract

Every row will contain stable comparison fields. Important fields are:

- `f3_graph_comparison_status`: `not_requested`, `not_monomial`, `ok`,
  `analysis_error`, or `all_transpile_failed`;
- `f3_optimal_leakage_phase` and `f3_optimal_leakage_phase_over_pi`;
- `f3_optimal_leakage_phase_real` and `f3_optimal_leakage_phase_imag` for `z*`;
- `f3_graph_successful_pairs` and `f3_graph_failed_pairs`;
- `f3_graph_baseline_best_*` and `f3_graph_optimal_best_*` metrics;
- `f3_graph_baseline_mean_*` and `f3_graph_optimal_mean_*` metrics;
- `f3_graph_*_delta` fields, always defined as optimal minus baseline;
- `f3_graph_optimal_is_better` using the benchmark's metric ordering;
- best seed and strategy for each arm.

If circuit export is enabled, each eligible candidate also receives uncompiled
source artifacts for the two comparison arms and their local Fourier gates:

- `graph_state_f3_baseline.qpy`;
- `graph_state_f3_optimal.qpy`;
- `F3_W_phi0.qpy`;
- `F3_W_phi_optimal.qpy`.

Historical artifact names remain unchanged.

## Error Handling

- Shape, isometry, and monomial checks raise specific `ValueError` messages in
  the pure analysis helper.
- The candidate benchmark converts phase-analysis failures into
  `f3_graph_comparison_status = analysis_error` and stores a short error string.
- Paired transpilation re-raises `KeyboardInterrupt`, `SystemExit`, and
  `MemoryError`; ordinary backend/transpiler failures are counted per pair.
- A candidate with zero successful pairs records `all_transpile_failed` and does
  not fabricate metric deltas.
- Artifact output is written only after both source circuits have been built.

## Testing Strategy

The implementation follows red-green-refactor cycles.

1. Unit-test monomial detection, effective support ordering, the analytic
   formula, phase normalization, invalid inputs, and invariance under `D`.
2. Exhaustively test all generated `monomial_full` candidates and assert that
   the angle is only `π/2` or `11π/6`.
3. Test that baseline and optimized Fourier embeddings have identical logical
   action and differ only on the one-dimensional leakage complement.
4. Test that the two full graph-state circuits are state-equivalent and contain
   one explicit `F3_W` per qutrit.
5. Use a deterministic fake compiler to prove paired seeds, metric prefixes,
   deltas, statuses, and failure handling.
6. Test CLI forwarding and QPY artifact names without running hardware.
7. Run the full `direct_basis` test selection. The repository-wide suite is also
   run and any pre-existing dependency or historical-QPY failures are reported
   separately from feature regressions.

## Compatibility and Rollout

With the new flag absent, APIs, ranking behavior, runtime, CSV values, and
historical artifacts remain unchanged. Enabling the flag adds columns and
artifacts but does not change candidate ordering or which candidate is selected
by the current benchmark. This allows new results to be compared with existing
runs while the optimized circuit data is evaluated.

## Acceptance Criteria

- Every generated monomial encoding gets an analytic phase from its actual
  matrix, not from its name.
- Generated `monomial_full` candidates yield only `π/2` or `11π/6`, with
  diagonal phase choices leaving the result unchanged.
- The baseline and optimal full graph-state circuits prepare equivalent ideal
  logical states.
- The comparison changes only the unused-state phase of each local `F3`.
- Baseline and optimal circuits use identical transpiler seeds and strategy sets.
- CSV output exposes phase provenance, paired metrics, deltas, and explicit
  status values.
- Existing behavior is unchanged when the comparison flag is omitted.
- Targeted `direct_basis` tests and all newly added tests pass.
