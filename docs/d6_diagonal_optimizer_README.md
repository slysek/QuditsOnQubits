# `d6_diagonal_optimizer.py` — README

A standalone Python module that synthesizes a 6-qubit realization of a
diagonal gate `D[Lambda]` acting on two ququits of dimension `d = 6`,
exploiting the fact that 28 of the 64 basis states fall outside the
qudit code space and can have *any* phase. The free phases are
optimized so that the resulting `CNOT + RZ` circuit is as cheap as
possible.

* **File:** `QuditsOnQubits/QuditsOnQubits/d6_diagonal_optimizer.py`
* **Entry point:** `optimize_d6_diagonal(diag36, ...)`
* **Companion (general-purpose, no don't-care optimization):**
  `find_best_diagonal_decomposition.py`

The rest of this document explains the math, the optimizer pipeline,
the API, and the empirical results.

---

## 1. Problem

We want to implement a diagonal unitary

```
D[Lambda] = diag(exp(i theta_0), exp(i theta_1), ..., exp(i theta_{35}))
```

on two qudits of dimension `d = 6`.  Each qudit level `a in {0, ..., 5}`
is encoded into 3 physical qubits via plain binary encoding:

```
0 -> |000>
1 -> |001>
2 -> |010>
3 -> |011>
4 -> |100>
5 -> |101>
```

The basis states `|110>` and `|111>` of each 3-qubit block are *not* in
the code space.  In total, 36 of the 64 basis states of the
6-qubit Hilbert space are *legal* and 28 are *illegal*
("don't-care").

`D[Lambda]` only needs to act correctly on the 36 legal basis states.
On the 28 illegal states it can do anything; in particular, we are free
to *choose* a phase on each of them. Done well, that choice can
dramatically simplify the resulting circuit.

The module's job is to (1) pick those 28 phases, (2) synthesize a
diagonal unitary for the resulting 64-vector using `CNOT + RZ`
phase gadgets, (3) compare it with a Qiskit `DiagonalGate` baseline,
and (4) return both circuits with their resource counts.

---

## 2. Conventions

Everything below is fixed and used consistently throughout the module.

### 2.1 Endianness

The 6-qubit basis-state index of a pair `(a, b)` is

```
index(a, b) = (a << 3) | b,         a, b in {0, ..., 5}.
```

So qubits **5, 4, 3** carry the 3-bit binary code of qudit `a` (qubit 5
= MSB of `a`) and qubits **2, 1, 0** carry the 3-bit binary code of
qudit `b`. In Qiskit, basis-state index `k` corresponds to physical
qubit `i` taking value `(k >> i) & 1`.

The 36 legal indices are exactly the integers `(a, b) -> (a << 3) | b`
for `a, b in {0..5}`. The complementary 28 indices are illegal.

```python
from d6_diagonal_optimizer import legal_indices, illegal_indices
assert len(legal_indices()) == 36
assert len(illegal_indices()) == 28
assert set(legal_indices()).isdisjoint(illegal_indices())
```

### 2.2 Order of `diag36`

`diag36` is a flat length-36 vector of the legal phases. Two
conventions are supported:

* `order="row-major"` (default): `diag36[6 * a + b]` is the phase of
  `|a, b>`. This matches the standard tabular reading of the AME(4,6)
  paper (rows over `a`, columns over `b`).
* `order="col-major"`: `diag36[a + 6 * b]` is the phase of `|a, b>`.

### 2.3 Input format

`diag36` may be given either as

* a length-36 array of **complex unit-modulus** numbers
  (e.g. `[1, omega, omega.conjugate(), ...]`); the module takes
  `np.angle(...)`, or
* a length-36 array of **real phases** in radians.

Auto-detection: complex dtype or any nonzero imaginary part triggers
the complex path; otherwise, if all entries have `|x| == 1` within
tolerance the input is treated as `+/-1` complex; otherwise as raw
phase angles. See `_phases_from_diag36`.

### 2.4 Phase normalization

All phases are wrapped into the half-open interval `(-pi, pi]` before
the Walsh transform via `normalize_phase`. The lift `theta + 2 pi k(x)`
for any integer-valued `k` does not change `exp(i theta(x))` and is a
free choice; this module always uses the canonical principal-value
lift.

### 2.5 RZ convention

`Qiskit's RZ(phi) = exp(-i phi Z / 2) = diag(exp(-i phi/2), exp(i phi/2))`.
For `exp(i c Z)` to equal `RZ(phi)`, we need `phi = -2 c`. Multi-qubit
`exp(i c Z_S)` is implemented via a CNOT ladder onto the last qubit of
the support, an `RZ(-2 c)` on that target, and the reversed CNOT
ladder. The 1-qubit case skips the ladder. This is verified
numerically in `_self_tests` against `qiskit.circuit.library.DiagonalGate`.

---

## 3. Walsh / phase-polynomial expansion

For a diagonal unitary `D = diag(exp(i theta_0), ..., exp(i theta_{N-1}))`
on `n` qubits with `N = 2**n`, write each phase angle as

```
theta(x) = sum_{S in {0,1}^n} c_S * (-1)^{popcount(S AND x)},
                                   for x in {0, ..., N-1}.
```

The coefficients `c_S` are the Walsh-Hadamard transform of `theta`:

```
c_S = (1 / N) * sum_x theta(x) * (-1)^{popcount(S AND x)}.
```

Equivalently,

```
D = exp(i c_0 * I) * prod_{S != 0} exp(i c_S * Z^S),
```

where `Z^S = product_{i in S} Z_i`. Each non-zero `c_S` corresponds to
one *phase gadget*: a CNOT ladder that computes the parity bit, an
`RZ(-2 c_S)` on the target, and the reversed CNOT ladder that
uncomputes the parity. The constant term `c_0` is the global phase and
is folded into `qc.global_phase`, never synthesized.

Implementation: in-place butterfly Walsh-Hadamard transform with the
sign convention `h[m] = sum_k (-1)^{popcount(m AND k)} theta_k`,
followed by a `1/N` rescaling. Numerical tolerance is exposed as
`tol=1e-10`. Round-trip is verified in `_self_tests`.

---

## 4. Why optimizing the don't-care phases helps

Both Qiskit's `DiagonalGate` synthesis and the `CNOT + RZ` phase-gadget
synthesis pay one `RZ` (and a fixed amount of `CX`) for every nonzero
Walsh coefficient `c_S`. Concretely:

* Phase gadgets: `2 * (popcount(S) - 1)` CNOTs per non-trivial term
  (plus 1 RZ).
* `DiagonalGate`: Qiskit uses the Shende-style construction
  (~ `2**n - 2` CNOTs in the worst case) but the transpiler eliminates
  rotations whose angle is exactly zero, often saving a substantial
  amount of `CX` work.

Either way: **fewer nonzero `c_S` means a cheaper circuit**.

The phases of the 36 legal entries are fixed by `D[Lambda]`. The 28
don't-care phases are free real parameters. Each Walsh coefficient
`c_S` is *affine* in those 28 parameters:

```
c_S = const_S + sum_{i = 0..27} B[S, i] * theta_illegal[i],
B[S, i] = (-1)^{popcount(S AND illegal_idx[i])} / N.
```

So the search space is a 28-dimensional affine subspace of
`R^64` (or, after restriction to a discrete alphabet, a 3^28 lattice
for the natural `[0, 2 pi/3, -2 pi/3]` alphabet). The optimizer's job
is to find a point on this set with as few nonzero `c_S` as possible
(equivalently, a sparse point in the affine subspace).

This is a hard combinatorial / sparse-recovery problem in general, so
the module attacks it with a stack of heuristics — random search,
multi-restart simulated annealing, an LP relaxation, greedy
coordinate descent, and pairwise (2-opt) local search — and then
optionally polishes with an objective that calls Qiskit's `transpile`
directly.

---

## 5. The optimizer pipeline

Default `method="best_of"` runs the stages in this order, each
keeping a `(best_phases, best_cost)` running record:

```
random_search                (RS)
  ->  multi_restart_simulated_annealing (SA, n_restarts=6)
  ->  lp_relax_walsh_l1   + project to alphabet
  ->  lp(rounded) -> greedy_coordinate_descent
  ->  multi_restart_greedy           (n_restarts=6)
  ->  two_opt_descent                (final pairwise local search)
  ->  optional polish_target with transpile()
```

Each stage gets the **full** `max_iters` budget; adding more stages
only ever makes the search stronger (and slower), never weaker. Below
is what each stage does and why.

### 5.1 Random search

`random_search(legal_phases, ..., alphabet, max_iters, seed)`.
i.i.d. samples of the 28 don't-care phases from the discrete
alphabet, evaluates `phase_polynomial_cost`, keeps the best. Cheap and
hard to beat as a baseline; on the natural `[0, 2 pi/3, -2 pi/3]`
alphabet, RS explores `3^28 ~= 2.3e13` configurations
non-redundantly, so even 4000 random draws find moderately good
basins.

### 5.2 Multi-restart simulated annealing

`multi_restart_simulated_annealing(...)` runs
`simulated_annealing` `n_restarts` times. The first restart warm-starts
from the best phases found so far (typically the RS best); the
remaining restarts begin from a fresh random sample. Each restart
performs single-flip Metropolis moves with an exponential cooling
schedule from `T_start = max(2, 0.5 * cost)` down to `T_end = 0.05`.
Adding restarts is the main fix for the original SA's tendency to get
trapped in the same local optimum as the RS warm-start.

### 5.3 LP relaxation `min ||c||_1`

`lp_relax_walsh_l1(legal_phases, legal_idx, illegal_idx)` solves the
continuous convex relaxation

```
min sum_{S != 0} |const_S + B[S, :] @ theta_illegal|
        s.t.  theta_illegal in [-pi, pi]^28.
```

This is an LP in `28 + 63 = 91` variables and `2 * 63 = 126`
inequality constraints, solved with HiGHS via
`scipy.optimize.linprog`. L1 minimization is the standard convex
relaxation of the L0 (number-of-nonzero) objective and tends to
produce sparse solutions. The continuous answer is then projected to
the nearest alphabet entry per coordinate (`_project_to_alphabet`,
circular distance) and polished with greedy coordinate descent.

For the example vectors of section 7, the *rounded* LP point is not
the discrete optimum (the legal phases live in a `3rd`-roots-of-unity
sublattice and the LP doesn't see the discrete structure), but it
provides a useful different starting point, particularly when the
random/SA stages have clustered onto one basin.

### 5.4 Greedy coordinate descent (1-opt)

`greedy_coordinate_descent(legal_phases, ..., init_phases, eval_cost,
max_passes=12)`. For each pass, walks the 28 don't-care indices in
order; for each one, tries every alphabet value in turn and accepts
the lowest-cost choice. Stops when no index improves in a full pass.
With `|alphabet| = 3` and 28 indices, each pass is `28 * 3 = 84`
cost evaluations; convergence is typically 3-5 passes. Cheap, fully
deterministic, and reliably reaches a local minimum.

`multi_restart_greedy(...)` runs `n_restarts=6` greedy descents from
random initializations (and optionally one warm-start from the
running best), keeping the lowest cost.

### 5.5 Pairwise 2-opt local search

`two_opt_descent(legal_phases, ..., init_phases, eval_cost,
max_passes=3)`. For every unordered pair `(i, j)` of don't-care
indices and every joint assignment in `alphabet x alphabet`, accepts
the lowest-cost choice. Costs `28 * 27 / 2 * 9 = 3402` cost
evaluations per pass at `|alphabet| = 3`.

**This is the stage that produces the largest empirical jump** on
`D_Lambda_23` (Walsh terms `45 -> 33`, baseline depth `94 -> 80`).
1-opt greedy reaches a local minimum where every single coordinate
flip increases the cost; many such minima can still be escaped by
flipping two coordinates simultaneously. 2-opt covers exactly those.
3-opt and higher are not implemented (cubic blow-up rarely pays).

### 5.6 Optional transpile-based polish

`polish_target` (default `"cx_then_depth"`) makes the optimizer run
one final greedy coordinate-descent pass with an objective evaluated
by *actually transpiling* the circuit:

```python
def polish_eval(theta64):
    return transpile_metric(
        theta64,
        target=polish_target,           # "depth" / "cx" / "weighted" / "cx_then_depth"
        via=polish_via,                 # "diagonal_gate" or "phase_gadget"
        basis_gates=qiskit_basis,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
```

This is much slower (one `transpile` per evaluation, ~50 ms each), but
optimizes exactly the metric the user cares about. In practice, on the
two example vectors, the polish doesn't find an improvement *beyond*
the Walsh-cost optimum because the Walsh-cost surrogate is already
very tight. It's left on by default because it's a safety net for
cases where the surrogate is loose. Disable with `polish_target=None`.

### 5.7 Cost function

`phase_polynomial_cost(theta64, ...)` returns a `CostBreakdown` with
four scalars:

| Component | Meaning |
|---|---|
| `num_nonzero_terms` | number of nonzero Walsh coefficients excluding `S = 0` |
| `cx_estimate` | naive CNOT cost of phase gadgets: `sum_S 2 * (popcount(S) - 1)` |
| `weighted_terms` | `sum_S popcount(S)` over nonzero `S != 0` |
| `max_support` | largest popcount over nonzero `S != 0` |

The combined scalar score (default weights) is

```
cost = 1.0 * cx_estimate + 0.1 * num_nonzero_terms + 0.05 * max_support
```

Override via `cost_weights={"cx": ..., "terms": ..., "support": ...,
"weighted": ...}`. The optimizer minimizes `cost`; the polish stage
optionally minimizes a transpile-derived metric instead.

---

## 6. Phase-gadget synthesis

`synthesize_phase_gadgets(theta64, tol=1e-10, num_qubits=6)` builds a
`QuantumCircuit(6)` containing only `CX` and `RZ` gates plus a global
phase. The implementation is straightforward:

1. Compute the Walsh coefficients `{S: c_S}` with tolerance `tol`.
2. Set `qc.global_phase = c_0` (do not synthesize this term).
3. Sort the remaining `(S, c_S)` by `(popcount(S), S)` for readability.
4. For each term, compute the support `[q_0, ..., q_{k-1}]`. Set the
   target `t = q_{k-1}`. Apply the CNOT ladder
   `cx(q_0, t), cx(q_1, t), ..., cx(q_{k-2}, t)`, then `rz(-2 c_S, t)`,
   then the reversed ladder. For `k = 1`, only the `rz` is emitted.

The result is a strictly diagonal unitary; this is double-checked in
`_self_tests` and `extract_diagonal_from_circuit` raises if the
resulting circuit's unitary is not diagonal within tolerance.

---

## 7. API

Importing the module exposes the following public names:

```python
from d6_diagonal_optimizer import (
    # encoding / indexing
    encode_qudit_level, pair_to_qubit_index, qubit_index_to_pair,
    legal_indices, illegal_indices,
    # phase-vector handling
    diag36_to_theta64, normalize_phase,
    # phase polynomial / cost
    walsh_coefficients, phase_polynomial_cost,
    # synthesis
    synthesize_phase_gadgets,
    # validation
    verify_on_code_space, extract_diagonal_from_circuit,
    # optimizer building blocks
    random_search, simulated_annealing,
    multi_restart_simulated_annealing,
    greedy_coordinate_descent, multi_restart_greedy,
    two_opt_descent, lp_relax_walsh_l1, transpile_metric,
    # main entry point
    optimize_d6_diagonal,
)
```

### 7.1 `optimize_d6_diagonal`

```python
optimize_d6_diagonal(
    diag36,
    phase_alphabet="default",      # "default" | None | iterable of float
    max_iters=4000,
    seed=12345,
    qiskit_basis=None,             # default ["rz","sx","x","cx"]
    optimization_level=3,
    order="row-major",             # or "col-major"
    method="best_of",              # "random"|"sa"|"greedy"|"lp"|"both"|"best_of"
    cost_weights=None,
    tol=1e-10,
    verbose=False,
    n_restarts=6,
    polish_target="cx_then_depth", # None | "depth" | "cx" | "weighted" | "cx_then_depth"
    polish_via="diagonal_gate",    # or "phase_gadget"
    polish_max_passes=6,
) -> dict
```

`phase_alphabet`:

* `"default"` -> `[0, 2 pi/3, -2 pi/3]`. Natural for `omega_3`-valued
  inputs (AME(4,6)).
* `None` -> the unique phases that appear in the legal input
  (rounded to 12 decimals). Stays inside the input's alphabet.
* iterable of floats -> used as-is.

Returns a dictionary with all artifacts:

| key | meaning |
|---|---|
| `best_theta64` | length-64 real array of phases (legal entries from `diag36`, illegal entries from optimizer) |
| `best_diag64` | `np.exp(1j * best_theta64)` |
| `best_illegal_phases` | length-28 list of the optimizer-chosen don't-care phases |
| `legal_indices`, `illegal_indices` | the 36 / 28 index lists |
| `walsh_coeffs` | `{mask: c_S}` after the synthesis tolerance |
| `num_nonzero_terms`, `cx_estimate`, `max_support` | summary numbers from `phase_polynomial_cost` |
| `alphabet` | the alphabet actually used |
| `method`, `polish_target`, `polished` | what ran and whether the polish helped |
| `phase_gadget_circuit` | naive `CX + RZ` circuit (untranspiled) |
| `transpiled_circuit` | the same after Qiskit `transpile` to the requested basis |
| `baseline_circuit` | Qiskit `DiagonalGate(64)` -> transpile, with the *optimizer-chosen* don't-care phases |
| `metrics` | depth / cx / rz / per-op breakdown for each of the three circuits |

### 7.2 Validation helpers

* `verify_on_code_space(qc, diag36, order="row-major")`: returns the
  largest absolute deviation between the qubit-circuit's diagonal on
  the 36 legal indices and the target `diag36`, after a global-phase
  fix at the largest-magnitude reference index. Don't-care indices are
  ignored. Use `< 1e-7` for a stringent assertion.
* `extract_diagonal_from_circuit(qc, atol=1e-9)`: returns the diagonal
  of `Operator(qc).data`. Raises `ValueError` if the unitary is not
  diagonal within `atol`.

### 7.3 Self-tests

`_self_tests()`, run from `__main__`, exercises:

1. encoding/index helpers (legal/illegal disjoint, sum equals
   `range(64)`, every legal index decomposes into `(a, b)` with
   `a, b < 6`),
2. round-trip Walsh-Hadamard on a random vector,
3. agreement of `synthesize_phase_gadgets` with
   `qiskit.DiagonalGate` up to a global phase on a random diagonal,
4. `verify_on_code_space < 1e-7` on a random `diag36`.

---

## 8. Results

All numbers below are with default parameters
(`phase_alphabet="default"`, `max_iters=4000`, `seed=2024`,
`method="best_of"`, `polish_target="cx_then_depth"`,
`basis_gates=["rz","sx","x","cx"]`, `optimization_level=3`) on Qiskit
2.3.1.

### 8.1 Vector A — original `D[Lambda_{2,3}]` (`_example_d_lambda_2_3`)

```python
omega3 = np.exp(2j * np.pi / 3); omega3_bar = np.conj(omega3)
diag = np.array([
     1, omega3, omega3_bar, omega3, omega3_bar, 1,
     1,      1,      omega3,      1, omega3_bar, 1,
     1, omega3,           1, omega3_bar,         1, 1,
     1, omega3_bar,       1, omega3, omega3, omega3,
     omega3_bar, omega3_bar, omega3_bar, omega3_bar, 1, 1,
     omega3, omega3, omega3_bar, 1, 1, omega3_bar,
])
```

| circuit | depth | cx | rz | nonzero Walsh terms |
|---|---:|---:|---:|---:|
| naive baseline (don't-care = 1) | — | — | — | — |
| phase-gadget (raw) | 173 | 150 | 41 | 41 |
| phase-gadget after `transpile` | 125 | 102 | 41 | 41 |
| `DiagonalGate` baseline (with optimized don't-care) | **85** | **50** | 41 | 41 |

`verify_on_code_space` ~= `1.5e-15` on all three. The optimizer's stages
all converged to the same Walsh cost (cost = `154.350`,
`cx_estimate = 150`, `terms = 41`, `max_support = 5`); 2-opt and the
transpile polish did not find improvements. Local optimum looks tight
for this input.

### 8.2 Vector B — user-provided `D_Lambda_23`

```python
D_Lambda_23 = np.array([
    1,
    omega3_bar, omega3_bar, omega3_bar, omega3_bar,
    1, 1, 1,
    omega3, 1, omega3, 1, 1, omega3, 1, omega3,
    1, 1, 1, omega3, 1,
    omega3_bar, omega3, omega3_bar, omega3_bar,
    omega3, omega3, omega3_bar,
    1, 1,
    omega3, omega3, omega3_bar,
    1, 1, omega3_bar,
])
```

| variant | depth | cx | rz | nonzero Walsh terms |
|---|---:|---:|---:|---:|
| naive baseline (don't-care = 1, no optimizer) | 107 | 60 | 53 | 53 |
| `method="both"` (RS + multi-restart SA) | 94 | 56 | 45 | 45 |
| `method="best_of"` (adds LP, greedy, 2-opt) | **80** | **52** | 33 | **33** |
| phase-gadget after `transpile` (best_of) | 98 | 92 | 33 | 33 |
| phase-gadget raw (best_of) | 139 | 130 | 33 | 33 |

`verify_on_code_space` ~= `1.4e-15` on all three. The breakdown of who
found what (default budget):

```
random_search                cost = 190.600  cx_est=186  terms=43  max_supp=6
sa(x6)                       cost = 186.750  cx_est=182  terms=45  max_supp=5
lp(continuous->alphabet)     cost = 264.600  cx_est=258  terms=63  max_supp=6
lp + greedy                  cost = 192.650  cx_est=188  terms=44  max_supp=5
greedy(x6)                   cost = 186.750  cx_est=182  terms=45  max_supp=5
two_opt(walsh)               cost = 133.550  cx_est=130  terms=33  max_supp=5  <-- winner
polish_target='cx_then_depth': no improvement
```

The big jump is 2-opt: SA, greedy and rounded-LP all settle around
~45 Walsh terms; 2-opt finds the 33-term basin in a single pass. Net
result on this vector:

```
depth   107  ->   80     (-25%)
cx       60  ->   52     (-13%)
rz       53  ->   33     (-38%)
Walsh terms 53 -> 33     (-38%)
```

### 8.3 Reproducing both result tables

```python
import numpy as np
from d6_diagonal_optimizer import (
    optimize_d6_diagonal, verify_on_code_space, _example_d_lambda_2_3
)

omega3 = np.exp(2j * np.pi / 3); omega3_bar = np.conjugate(omega3)
D_Lambda_23 = np.array([
    1,
    omega3_bar, omega3_bar, omega3_bar, omega3_bar,
    1, 1, 1,
    omega3, 1, omega3, 1, 1, omega3, 1, omega3,
    1, 1, 1, omega3, 1,
    omega3_bar, omega3, omega3_bar, omega3_bar,
    omega3, omega3, omega3_bar,
    1, 1,
    omega3, omega3, omega3_bar,
    1, 1, omega3_bar,
])

for label, diag in [("vector A", _example_d_lambda_2_3()),
                    ("vector B", D_Lambda_23)]:
    res = optimize_d6_diagonal(
        diag,
        phase_alphabet="default",
        max_iters=4000,
        seed=2024,
        order="row-major",
        method="best_of",
        verbose=False,
    )
    m = res["metrics"]
    err = verify_on_code_space(res["baseline_circuit"], diag)
    print(f"{label:9s}  baseline depth={m['baseline_depth']:>3} "
          f"cx={m['baseline_cx']:>3} rz={m['baseline_rz']:>3}  "
          f"walsh_terms={res['num_nonzero_terms']:>2}  err={err:.1e}")
```

Expected output:

```
vector A   baseline depth= 85 cx= 50 rz= 41  walsh_terms=41  err=1.5e-15
vector B   baseline depth= 80 cx= 52 rz= 33  walsh_terms=33  err=1.4e-15
```

---

## 9. Practical guidance

* For ω₃-valued inputs (AME(4,6) and friends), keep
  `phase_alphabet="default"` (`[0, 2 pi/3, -2 pi/3]`). Widening the
  alphabet usually *introduces* new Walsh content rather than removing
  any.
* Increase `max_iters` and/or `n_restarts` for harder instances.
* Set `polish_target=None` if you don't need the final transpile-based
  polish (it adds ~1 second; useful only when the Walsh surrogate is
  loose).
* If you want a `CX + RZ`-only circuit (no transpile), use
  `result["phase_gadget_circuit"]`. Note that on the example vectors
  the Qiskit `DiagonalGate` baseline is *cheaper* than the naive phase
  gadget circuit, because Qiskit's `DiagonalGate` synthesis applies
  Gray-code-style merging that this module's straight phase-gadget
  implementation does not.

---

## 10. Open ideas / future work

1. **2-opt with a transpile metric.** The current polish is 1-opt over
   `transpile()`. 2-opt would be `28 * 27 / 2 * 9 ~ 3.4k` transpile
   calls per pass (~3 minutes), which sometimes squeezes out a few
   more `cx`/`depth`. Could be limited to pairs with the largest
   Walsh-cost contribution.
2. **Wider alphabet experiments.** Try `[0, ±π/3, ±2π/3, π]` for both
   vectors. Expectation: usually worse for `ω_3`-structured inputs,
   but could help for inputs with finer phase structure.
3. **Tighter lower bound.** With 28 free real variables and 63 nonzero
   Walsh coefficients, the naive lower bound on nonzero terms is ~35.
   Vector B's 33-term result already beats that, so the matrix `B`
   has effective rank `< 28` for ω₃ inputs (known column dependencies).
   Computing the exact discrete optimum via a small MIP (28 ternary
   variables) would close the gap definitively; CPLEX/Gurobi/SCIP
   should solve in seconds.
4. **LP -> MIP warm start.** Use the LP-continuous solution as a
   warm-start for a MIP encoding the discrete alphabet constraints.
5. **Better phase-gadget synthesis.** The naive `CNOT` ladder per
   term costs `2(k-1)` CNOTs. A Gray-code-merged or `ParityNetwork`
   synthesis (Amy/Maslov) would substantially reduce the raw
   phase-gadget circuit's `cx` count, plausibly to within a few
   percent of the `DiagonalGate` baseline.

---

## 11. File layout & status

```
QuditsOnQubits/QuditsOnQubits/
├── d6_diagonal_optimizer.py                # the module
├── docs/
│   ├── d6_diagonal_optimizer_README.md     # this document
│   └── d6_diagonal_optimizer_notes.md      # session-memory snapshot
└── find_best_diagonal_decomposition.py     # related, more general module
                                            # (no don't-care optimization)
```

Run `python d6_diagonal_optimizer.py` to execute self-tests and the
default example. Lints clean; module passes its self-tests; both
example vectors give the metrics above on Qiskit 2.3.1, NumPy with
`bitwise_count` (or its slow fallback), and SciPy 1.x.
