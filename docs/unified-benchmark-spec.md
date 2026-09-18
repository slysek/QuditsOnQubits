# Unified encoding benchmark: implementation contract

The user approved implementation of the agreed pipeline on 2026-09-17. The notebook is a subsequent task. This contract records the implementation choices from the plan.

## Public API

The public package is `qudits_on_qubits.benchmarks`. `run_benchmark(circuit, *, families, backend, config=None, synthesis=None, baseline_synthesis=None, saved_candidates=(), output_dir=None, progress=None)` accepts an explicit logical circuit, explicit backend, and one or more encoding families. It returns concrete trials, statistics, the complete first Pareto front, and an artifact directory.

`LogicalCircuit` describes an ordered sequence of logical F3/CZ3 operations with qutrit zero least significant. The first example is the two-qutrit graph state. One encoding is shared by all qutrits. Independent logical simulation supplies the reference. The initial public domain is finite state-preparation circuits with these gates.

`LocalSU2` and `SchmidtTheta` provide immutable isometries, stable IDs and generation parameters. Canonical is included in every synthesis comparison. Identical matrices may remain separate labelled alternatives so provenance is not erased.

## Synthesis and reference circuits

Default synthesis uses the existing optimized F3/CZ3 library. The explicit `ExactSynthesis` control uses existing dense logical embeddings, useful for quick end-to-end validation without BQSKit. No silent substitution of strategies occurs.

A single synthesis strategy applies to all families, or an explicit mapping selects a strategy per family. Results from different synthesis protocols have separate Pareto boundaries. Every cohort includes canonical. `ThetaContinuationSynthesis` runs the existing sequential theta continuation/fallback pipeline or consumes an existing hash-verified campaign. Source configuration and grid must match.

`SavedGateSynthesis` consumes actual saved E/F3/CZ3. To reuse a complete saved preparation verbatim, the caller explicitly binds it to the logical workload hash; independent state validation still applies. Saved noncanonical data cannot synthesize canonical: `baseline_synthesis` supplies an explicit reference policy (default optimized), recorded in the protocol.

Gate code-space action and leakage are independently checked over all columns, with one global phase. Default F3 tolerance is 1e-10, CZ3 tolerance 1e-5; caller may explicitly set CZ3 to 5e-4 for the historical relaxed campaign. Default full-state minimum fidelity is 1-1e-6 and leakage maximum is 1e-6. These are distinct criteria, not inferred from transpiler approximation degree.

## Backend and routing

Backend is mandatory: a local synthetic target, IQM Garnet/Emerald, or a named IBM backend. Provider-specific adapters compile using a frozen target and common layout/routing policy. No hardware jobs are submitted. Hardware target loading is read-only; supplied backend objects enable offline validation.

All final metrics are measured after native compilation and routing. All native two-qubit gates count, including decomposed routing operations. Initial/final mappings, active physical wires, target data and compilation options are persisted. Validation simulates only active wires plus logical output wires, then accounts for output permutation and traces out routing ancillas. A configurable finite active-wire limit prevents accidental exponential allocation; exceeding it is a validation failure, never an accepted unverified candidate.

## Pareto and evidence

Default transpiler seeds are 0,1,2. Reference runs can explicitly choose twenty seeds. A candidate is eligible only when every planned seed succeeds and at least two seeds are present for a meaningful stability estimate. Partial results remain diagnostics.

Existing Pareto routines are reused. Objectives are mean 2q count, mean depth and population depth standard deviation. Boundaries include workload, target snapshot and complete comparison protocol. Different provider targets, tolerances and synthesis policies never silently share a front. Results are sets of nondominated alternatives, not a claim of global optimality.

A new run requires a new or empty directory. Manifest and immutable candidate/trial bundles use JSON, NPY and QPY with SHA-256 integrity checks. Final indexing is hash-linked. Reanalysis verifies artifacts and recomputes statistics without synthesis. Interrupted runs retain evidence; general computation resume is not claimed. Existing theta continuation retains its own resume mechanics.

Implementation must preserve unrelated working changes. No commits, pushes or QPU submissions. Tests use real mathematical and routing checks, isolated expensive-synthesis substitutes where necessary, at least 80% coverage of the new layer, and up to three independent full-diff reviewer rounds.
