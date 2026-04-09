# Encoded AME Circuit Basis-Change Design

**Date:** 2026-04-09

**Scope:** Update `QuditsOnQubits/QuditsOnQubits/create_ame_circuit.py` so that basis-changed AME circuits are assembled from explicit circuit blocks instead of pre-multiplied transformed gate matrices.

## Context

`create_ame_circuit` currently supports an optional encoding change through `E_new` for `dim=3`. The existing implementation:

- builds the encoding-change unitary `W`,
- keeps an initial layer of `W` on every encoded qutrit when `E_new` is provided,
- computes transformed matrices for `Fgate` and `CZgate`,
- wraps those transformed matrices back into `UnitaryGate` instances.

The desired behavior is to preserve the initial `W` layer and replace matrix conjugation with explicit gate sequences on the circuit:

- single-qutrit block: `W -> Fgate -> Wdag`,
- two-qutrit block: `(W on qutrit A, W on qutrit B) -> CZgate -> (Wdag on qutrit A, Wdag on qutrit B)`.

## Goals

- Keep the initial `W` layer whenever `E_new` is used.
- Stop building transformed `Fgate` and `CZgate` matrices for the new encoding.
- Represent the encoding change directly as explicit circuit structure.
- Build `W` and `Wdag` as decomposed two-qubit circuits using `TwoQubitWeylDecomposition(...).circuit()`.
- Keep the public API of `create_ame_circuit` unchanged.

## Non-Goals

- No change to the `basis` argument behavior.
- No change to `dim=4` behavior.
- No broad refactor of unrelated circuit-building logic.
- No introduction of a new test framework beyond what is needed to verify this change.

## Design

### 1. Encoding-change building blocks

When `E_new` is present, the implementation should compute:

- `W` from `build_encoding_change_unitary(E_new)`,
- `W_qc = TwoQubitWeylDecomposition(W).circuit()`,
- `Wdag_qc = TwoQubitWeylDecomposition(W.conj().T).circuit()`.

These objects should remain circuit blocks and be appended directly with `qc.append(...)` in the same style already used for the optional `basis` transformation.

### 2. Initial state preparation semantics

The initial `W` layer stays in place for every use of `E_new`.

This layer is not an optimization artifact. It is part of the intended circuit semantics and represents preparation of logical `|0>` states in the new encoding basis before the AME entangling structure is applied.

### 3. Helper-based circuit assembly

The basis-changed circuit assembly should be expressed through helper functions:

- one helper for appending the encoded single-qutrit `Fgate` block,
- one helper for appending the encoded two-qutrit `CZgate` block.

Expected helper behavior:

- encoded `Fgate` helper appends `W_qc`, then the original `Fgate`, then `Wdag_qc` on the same two-qubit pair,
- encoded `CZgate` helper appends `W_qc` on each endpoint pair, then the original `CZgate`, then `Wdag_qc` on each endpoint pair.

This keeps `_create_circuit_from_graph` focused on graph traversal and high-level block ordering.

### 4. Main circuit flow

For `E_new is None`, behavior stays unchanged.

For `E_new is not None`, `_create_circuit_from_graph` should:

1. select the original base gates for the requested dimension,
2. build `W_qc` and `Wdag_qc`,
3. append the initial `W_qc` layer to every qutrit pair,
4. append encoded `Fgate` blocks for every qutrit pair,
5. append encoded `CZgate` blocks for every graph edge.

The old `_transform_gates_to_new_encoding` path should be removed because it is no longer part of the target behavior.

## File Changes

Primary file:

- `QuditsOnQubits/QuditsOnQubits/create_ame_circuit.py`

Expected updates:

- revise docstrings/comments to describe explicit circuit composition instead of transformed matrices,
- replace the helper that returns transformed gates with a helper that returns decomposed `W` and `Wdag` circuits,
- add helper functions for encoded `Fgate` and encoded `CZgate` blocks,
- simplify `_create_circuit_from_graph` so the control flow reflects the intended circuit structure.

## Verification

Verification should cover both correctness and basic integration:

1. import/syntax check for the edited module,
2. a small operator-level equivalence check showing that:
   - `W -> Fgate -> Wdag` matches the previous `W @ F @ Wdag`,
   - `(W,W) -> CZ -> (Wdag,Wdag)` matches the previous `kron(W, W) @ CZ @ kron(Wdag, Wdag)`,
3. a smoke check that `create_ame_circuit(..., E_new=...)` still returns a valid Qiskit circuit for a small graph.

## Risks And Mitigations

### Repeated `W` layers increase circuit depth

This is intentional for now because the goal is to make the basis change explicit in the circuit rather than compressed into synthesized transformed gates.

### Qiskit append compatibility with decomposed subcircuits

The design uses raw circuits from `TwoQubitWeylDecomposition(...).circuit()` because the codebase already uses this append style for basis changes. Verification will confirm that the same pattern works for `W` and `Wdag`.

### Behavioral drift from prior matrix-based implementation

The operator-level equivalence check is included specifically to confirm the new circuit blocks preserve the same logical action as the old conjugated-matrix construction.
