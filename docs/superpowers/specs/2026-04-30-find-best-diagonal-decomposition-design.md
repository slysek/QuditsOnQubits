# Find Best Diagonal Decomposition Design

## Goal

Provide a single, self-contained Qiskit-based tool that, given a diagonal unitary `D_diag`
(1-D NumPy array of unit-modulus complex numbers, length a power of two), tries multiple
exact decomposition strategies and returns the best resulting `QuantumCircuit` according
to a configurable cost metric. Primary metric is the number of two-qubit gates, then
circuit depth, then total gate count.

The deliverable is a standalone script at `QuditsOnQubits/find_best_diagonal_decomposition.py`
(repo root, alongside `README.md` and `requirements.txt`), plus a pytest suite at
`QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`. The script does not
join the `QuditsOnQubits` package - it is intentionally importable as a top-level
module via `import find_best_diagonal_decomposition` when pytest's rootdir is in
`sys.path` (default behavior when no `conftest.py` / `pyproject.toml` is present).

The tool must be honest about what it does: it finds the best candidate among the
strategies and seeds it actually tried; it does not claim mathematical optimality.

## Architecture

Single-file script structured into clearly separated internal sections. The public
surface is small:

- `find_best_diagonal_decomposition(...)` - top-level driver.
- `unitaries_equal_up_to_global_phase(U, V, atol=1e-8)` - validation helper.
- `z_phase_coefficients_from_diag(D_diag, atol=1e-12)` - phase polynomial / Z-product
  coefficient extraction with diagnostics.

Internal helpers are private (leading underscore). The script also has a
`if __name__ == "__main__"` block that runs the canonical 6-qubit example from the
user spec and prints a comparison table.

The script imports only NumPy and Qiskit (`qiskit`, `qiskit.circuit.library.DiagonalGate`,
`qiskit.transpile`, `qiskit.quantum_info.Operator`). No PennyLane.

## Components

### 1. Input validation

`_validate_diagonal(D_diag, atol)`:

- assert `isinstance(D_diag, np.ndarray)` and `D_diag.ndim == 1`,
- assert `len(D_diag) >= 2` and `len(D_diag) == 2**num_qubits` for some integer
  `num_qubits >= 1`,
- assert `np.allclose(np.abs(D_diag), 1.0, atol=atol)`,
- return `num_qubits`.

Errors are `ValueError` with informative messages.

### 2. Up-to-global-phase equality

`unitaries_equal_up_to_global_phase(U, V, atol=1e-8)`:

- both inputs are square arrays of equal shape,
- find the entry of `V` with the largest magnitude (call its position `idx`),
- if that magnitude is `< atol`, both `V` and `U` should be effectively zero - check
  `np.allclose(U, 0, atol)`,
- otherwise compute `lambda = U[idx] / V[idx]`, reject if `abs(abs(lambda) - 1) > atol`,
- return `np.allclose(U, lambda * V, atol=atol)`.

Used both for full unitaries and for diagonals (the function does not care about the
internal structure - it uses entrywise comparison).

### 3. Phase polynomial / Z-product coefficients

`z_phase_coefficients_from_diag(D_diag, atol=1e-12)`:

- compute `theta = np.angle(D_diag)` (length `N = 2**n`); choose principal branch
  `(-pi, pi]`,
- for each subset `S` of `{0, ..., n-1}` (encoded as a bitmask `m`), compute

  `c_S = (1 / N) * sum_k (-1)^{popcount(m AND k)} * theta_k`

  This is a Walsh-Hadamard transform of `theta` (sign-pattern per subset).
- The diagonal is reconstructed up to global phase as
  `D_diag[k] = exp( i * sum_S c_S * (-1)^{popcount(S AND k)} )`.
- Return:

```python
{
    "num_qubits": n,
    "constant": c_0,                    # global phase (subset = empty)
    "coefficients": {frozenset_S: c_S}, # only entries with |c_S| > atol, S nonempty
    "num_nonzero": ...,
    "total_terms": 2**n - 1,
    "sparsity": num_nonzero / (2**n - 1),
    "max_weight": max(|S| for nonzero S, 0 otherwise),
    "weight_histogram": {k: count of nonzero S of size k},
}
```

### 4. Strategy: DiagonalGate + transpile

#### 4.1 Qubit-permutation semantics

The qubit-permutation knob does not aim to leave the diagonal invariant - it aims
to find the cheapest circuit among all qubit relabelings of the same diagonal. For
a permutation `perm = (p_0, p_1, ..., p_{n-1})`, the candidate circuit implements

  `D_diag_perm[k] = D_diag[ bit_permute(k, perm) ]`

i.e. the diagonal indexed by basis states whose bit positions have been permuted
according to `perm`. Because the original diagonal can equally be regarded as
"the same physical operation, with logical wires relabeled", the user can apply
the resulting cheaper circuit to qubits in the relabeled order. The chosen
permutation is recorded in the candidate's metadata.

The validation in Section 7 compares each candidate's unitary against
`diag(D_diag_perm)` for **its own** permutation - so each candidate is verified
to exactly implement its declared diagonal.

#### 4.2 The strategy itself

`_strategy_diagonal_gate(D_diag_perm, *, normalize_phase, perm, basis_gates, backend,
optimization_level, seed)`:

- if `normalize_phase`, divide `D_diag_perm` by `D_diag_perm[0]` (so the first
  phase is `1`); record the original `D_diag_perm[0]` as the global phase delta
  in metadata.
- build `qc = QuantumCircuit(n)` and `qc.append(DiagonalGate(D_diag_perm), range(n))`.
  Note: the gate is always appended to the standard wire order `[0, ..., n-1]`. The
  permutation has already been baked into `D_diag_perm` itself.
- run `transpile(qc, backend=backend, basis_gates=basis_gates,
  optimization_level=optimization_level, seed_transpiler=seed)`,
- return `(qc_transpiled, metadata)`.

### 5. Strategy: sparse phase polynomial

`_strategy_sparse_phase_poly(D_diag_perm, *, normalize_phase, perm, basis_gates,
backend, optimization_level, seed)`:

- inputs already account for permutation. Recompute Z-product coefficients **on
  `D_diag_perm`** (not on the original `D_diag`), since permuting qubits changes
  which subsets `S` show up: a coefficient `c_S` of the original becomes
  `c_{perm(S)}` of `D_diag_perm`. Recomputing is cheap (Walsh-Hadamard, O(N log N))
  and avoids index-tracking bugs.
- if `normalize_phase`, drop the `c_0` (constant) term so the resulting circuit's
  unitary equals `diag(D_diag_perm) / D_diag_perm[0]`. Otherwise, set
  `qc.global_phase = c_0`.
- for each nonzero subset `S` with coefficient `c_S` (with `|S| >= 1`):
  - the contribution to the diagonal phase is `exp(i * c_S * Z_{i_1} ... Z_{i_k})`
    where `S = {i_1, ..., i_k}`,
  - synthesize this as a CX ladder onto a designated target qubit (the largest index
    of `S`), an `RZ(-2 * c_S)` on the target (sign chosen so the resulting unitary
    matches the convention `RZ(theta) = diag(exp(-i theta/2), exp(i theta/2))`), and
    the reverse CX ladder.
  - For weight-1 terms, no CXs are needed - just an `RZ` on the single qubit.
- run `transpile(qc, ...)` with the same arguments as Section 4. The transpiler at
  `optimization_level=3` may merge adjacent CX runs across consecutive terms.

To give the transpiler more room to cancel CXs, the iteration order over subsets `S`
is **Gray-code-like**: sort by Hamming weight first (so weight-2 terms run before
weight-3, etc.), then within a weight by the subset's bitmask. This is a simple
ordering that often produces useful CX adjacencies; we don't claim it's optimal.

The synthesis is exact by construction (each block is `exp(i c_S Z_S)` and they all
commute because they are diagonal). Validation in Section 7 still runs.

### 6. Driver

`find_best_diagonal_decomposition(D_diag, backend=None, basis_gates=None,
optimization_level=3, seeds=range(100), try_qubit_permutations=True,
try_global_phase_normalization=True, metric="two_qubit_then_depth",
atol=1e-8, verbose=True)`:

- validate input,
- pick default `basis_gates`: if `backend is None and basis_gates is None`, use
  `["rz", "sx", "x", "cx"]` (a realistic IBM-style native set that includes single-qubit
  primitives the transpiler can use),
- compute `coeffs = z_phase_coefficients_from_diag(D_diag)` for the diagnostics
  block and for the sparse strategy,
- enumerate permutations:
  - if `try_qubit_permutations is False`: only `identity`,
  - elif `num_qubits <= 4`: all `n!` permutations,
  - elif `num_qubits <= 6`: `identity` plus 24 random permutations (seeded by
    `np.random.default_rng(0)` for reproducibility),
  - else: only `identity`, with a warning logged when `verbose=True`,
- enumerate strategies: `{diagonal_gate, sparse_phase_poly}`,
- enumerate normalizations: `{False}` plus `{True}` if `try_global_phase_normalization`,
- for each `(strategy, normalize, perm, seed)` build candidate and:
  1. run the chosen strategy,
  2. validate (Section 7),
  3. compute metrics (Section 8),
  4. score (Section 9).
- track best, return result dict matching the spec.

`verbose=True` prints a tqdm-free progress hint at the start ("trying X candidates")
and the final comparison table at the end.

### 7. Validation

For each candidate transpiled circuit `qc`:

- if `num_qubits <= 12`:
  - compute `U = Operator(qc).data` (a `2**n x 2**n` complex array),
  - the candidate is supposed to implement `diag(D_diag_perm)` (Section 4.1). We
    therefore call
    `unitaries_equal_up_to_global_phase(U, np.diag(D_diag_perm), atol=atol)`.
  - When `normalize_phase=True`, the comparison target is the phase-normalized
    `diag(D_diag_perm) / D_diag_perm[0]`. Both check shapes are accepted because
    the helper compares up to global phase.
- else:
  - skip exact validation, attach `validation: "skipped"` to the candidate, and
    emit a single user-facing `UserWarning` (only on the first such candidate of
    the run, to avoid spam).

Candidates that fail validation are dropped (they cannot be the "best") and recorded
under `failed_candidates` in the diagnostics.

Note: there is no cheaper "diagonal-only" check available - `Operator(qc)` already
materializes the full unitary, so extracting only its diagonal saves nothing. If a
faster validation becomes important for `n` between 12 and ~16, we can revisit using
state-vector simulation on selected basis states; that's listed as a non-goal here.

### 8. Metrics

`_count_metrics(qc)`:

- iterate `qc.data` once and tally:
  - `total_gates`: count of all instructions whose op is not a barrier/measure/reset,
  - `two_qubit`: count of instructions with `len(qargs) == 2`,
  - `breakdown`: `{op.name: count}` from `qc.count_ops()` cast to a plain dict.
- `depth = qc.depth()`.
- We also keep an explicit allow-list of canonical 2-qubit gate names
  (`{"cx", "cz", "ecr", "iswap", "swap", "rzz", "rxx", "ryy", "csx", "dcx"}`) used
  only for documentation in the printed table - the **count** itself is taken from
  `len(qargs) == 2` so any non-canonical 2-qubit op is still counted correctly.

### 9. Scoring

`_score(metrics, metric)`:

- only `"two_qubit_then_depth"` is implemented now; raises `ValueError` for any other
  string.
- returns a tuple `(two_qubit, depth, total)` so it sorts lexicographically.

### 10. Result object

```python
{
    "best_circuit": QuantumCircuit,
    "best_score": (two_qubit, depth, total),
    "best_metadata": {
        "strategy": "diagonal_gate" | "sparse_phase_poly",
        "seed": int,
        "permutation": tuple[int, ...],
        "normalize_phase": bool,
        "two_qubit": int,
        "depth": int,
        "total_gates": int,
        "ops_breakdown": {gate_name: count},
        "validation": "ok" | "skipped",
    },
    "all_candidates": [{...same shape as best_metadata, plus a 'circuit' key...}],
    "diagnostics": {
        "num_qubits": int,
        "phase_coefficients": <dict from Section 3>,
        "num_candidates_attempted": int,
        "num_candidates_failed_validation": int,
        "failed_candidates": [...],
    },
}
```

### 11. Comparison table

When `verbose=True`, `find_best_diagonal_decomposition` prints a plain-text comparison
table sorted by score, with these columns: `strategy`, `seed`, `permutation`,
`normalize`, `rz`, `cx`, `2q`, `total`, `depth`, `validation`. Implemented with
`f"{...:<10}"` style formatting - no extra dependency.

The `__main__` block prints a header banner and the table for the canonical 6-qubit
example exactly as specified.

## Data flow

```
D_diag -> _validate_diagonal -> num_qubits
       -> z_phase_coefficients_from_diag -> coeffs
       -> for (strategy, normalize, perm, seed):
            build_candidate -> transpile -> validate -> metrics -> score
       -> rank candidates -> result dict
```

## Error handling

- Bad input (`ndim != 1`, length not power of two, non-unit moduli) raises
  `ValueError` from `_validate_diagonal` with messages naming the offending property.
- Unknown `metric` raises `ValueError` from `_score`.
- Transpiler exceptions on a single candidate are caught, recorded in
  `diagnostics["failed_candidates"]` with `reason="transpile_error: ..."`, and the
  candidate is skipped. The driver does not fail unless **every** candidate fails.
- Validation failures are recorded the same way with `reason="validation_failed"`.
- For `num_qubits > 12`, validation is skipped with a clear warning printed once.
- `try_qubit_permutations=True` with `num_qubits > 6` produces a warning and falls
  back to identity-only.

## Testing

Tests live in `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`.
Pytest. Use `qiskit.quantum_info.Operator` for ground-truth unitaries.

Cases:

1. `unitaries_equal_up_to_global_phase`:
   - identity matches itself,
   - `U` and `e^{i pi/4} U` match up to global phase,
   - `U` and a different unitary do not match,
   - all-zero edge case.
2. `_validate_diagonal`:
   - rejects 2-D arrays,
   - rejects length 3 (not power of two),
   - rejects entries with magnitude != 1,
   - accepts the canonical 6-qubit example and returns 6.
3. `z_phase_coefficients_from_diag`:
   - on `D = diag(1, 1, 1, 1)` (n=2), returns no nonzero coefficients,
   - on `D = diag(1, -1, 1, -1)` (Z_0 phase pattern), returns exactly one nonzero
     coefficient at `frozenset({0})` with value `pi/2` (or `-pi/2` depending on sign
     convention - the test checks the absolute value and that reconstruction matches),
   - reconstruction round-trip: build `D` from random nonzero subset of coefficients,
     run the function, recover the same coefficients to numerical precision.
4. End-to-end on the 6-qubit example:
   - `find_best_diagonal_decomposition(D_diag, seeds=range(5),
     try_qubit_permutations=False, verbose=False)` returns a result whose
     `best_circuit` reproduces `D_diag` up to global phase,
   - the result has at least one candidate per strategy, all marked `validation: "ok"`.
5. Sparse strategy correctness on a hand-crafted sparse diagonal:
   - construct a diagonal with exactly two nonzero Z-product coefficients,
   - run only the sparse strategy and confirm the resulting circuit is exact.

Tests are kept deterministic: `np.random.default_rng(0)` for any randomness, and
`seeds=range(5)` instead of the default 100 for speed.

## Non-goals

- No claim of mathematical optimality. Docstrings explicitly say "best among
  candidates tried".
- No additional metrics beyond `two_qubit_then_depth` in this iteration. The `metric`
  argument exists and is validated, but only one value is supported.
- No Pauli-Y or Pauli-X polynomial decompositions - only Z-products (the diagonal is
  already in the computational basis).
- No backend-specific tweaks beyond what `transpile(backend=...)` already does.
- No multi-process parallelism. The candidate space is small enough for a single thread
  on the canonical 6-qubit example (typically a few hundred to a few thousand
  candidates).
- Not integrated into the existing `QuditsOnQubits` package or notebooks - it is a
  standalone script as requested.
