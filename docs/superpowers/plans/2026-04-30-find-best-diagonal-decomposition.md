# Find Best Diagonal Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python script that, given a diagonal unitary as a 1-D NumPy array, returns the cheapest exact `QuantumCircuit` it can find by trying multiple Qiskit-based and custom phase-polynomial decomposition strategies, validated up to global phase.

**Architecture:** Single self-contained script `QuditsOnQubits/find_best_diagonal_decomposition.py` with three public symbols (`find_best_diagonal_decomposition`, `unitaries_equal_up_to_global_phase`, `z_phase_coefficients_from_diag`) and private helpers. Pytest test suite at `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`. Strategies tried: Qiskit `DiagonalGate` + transpile (with seeds, qubit permutations, optional global-phase normalization), and a custom sparse phase-polynomial synthesis using CX parity ladders + RZ.

**Tech Stack:** Python 3, NumPy, Qiskit (`qiskit`, `qiskit.circuit.library.DiagonalGate`, `qiskit.transpile`, `qiskit.quantum_info.Operator`), pytest.

**Workspace note:** This workspace is currently not a git repo. The plan therefore omits explicit `git commit` steps; each task ends with a verification step (running tests / running the script) instead. If git is later initialized, commits can be added per task.

**Spec:** [`docs/superpowers/specs/2026-04-30-find-best-diagonal-decomposition-design.md`](../specs/2026-04-30-find-best-diagonal-decomposition-design.md)

---

## File Map

| Path | Responsibility |
|------|----------------|
| `QuditsOnQubits/find_best_diagonal_decomposition.py` | All implementation: validation helpers, phase-polynomial extraction, strategies, driver, `__main__` block. |
| `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py` | Pytest tests for every public and key private helper, including end-to-end on the canonical 6-qubit `D_diag`. |

All tests run from `QuditsOnQubits/` as cwd via `pytest tests/test_find_best_diagonal_decomposition.py -v`.

---

## Task 1: Module scaffolding

**Files:**
- Create: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Create: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Create the empty module with docstring and imports**

Write `QuditsOnQubits/find_best_diagonal_decomposition.py`:

```python
"""Find the best available exact decomposition of a diagonal unitary.

Public API:
    find_best_diagonal_decomposition(D_diag, ...) -> dict
    unitaries_equal_up_to_global_phase(U, V, atol=1e-8) -> bool
    z_phase_coefficients_from_diag(D_diag, atol=1e-12) -> dict

The driver tries multiple strategies (Qiskit DiagonalGate + transpile, custom
sparse phase-polynomial synthesis), each combined with optional qubit
permutations, optional global-phase normalization, and a sweep over
seed_transpiler values. It returns the best candidate it found among the ones
it actually tried; it makes no claim of mathematical optimality.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Operator

__all__ = [
    "find_best_diagonal_decomposition",
    "unitaries_equal_up_to_global_phase",
    "z_phase_coefficients_from_diag",
]
```

- [ ] **Step 2: Create the empty test module**

Write `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`:

```python
"""Tests for find_best_diagonal_decomposition.

The script under test lives at the repo root (`QuditsOnQubits/`), not inside
the QuditsOnQubits package, so we import it as a top-level module. Pytest's
default rootdir-based sys.path injection makes this work when run from
`QuditsOnQubits/`.
"""

import numpy as np
import pytest

import find_best_diagonal_decomposition as fbdd
```

- [ ] **Step 3: Verify the test module loads**

Run from `c:\Users\szymo\QuditsOnQubits\QuditsOnQubits`:

```
pytest tests/test_find_best_diagonal_decomposition.py --collect-only -q
```

Expected: pytest reports "0 tests collected" with no import errors. If pytest fails to import the test module, fix imports before continuing.

---

## Task 2: `unitaries_equal_up_to_global_phase`

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py` (add function)
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_find_best_diagonal_decomposition.py`:

```python
class TestUnitariesEqualUpToGlobalPhase:
    def test_identity_matches_itself(self):
        I = np.eye(4, dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(I, I)

    def test_global_phase_factor_matches(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        phase = np.exp(1j * np.pi / 7)
        assert fbdd.unitaries_equal_up_to_global_phase(A, phase * A)

    def test_negated_diagonal_matches_up_to_phase(self):
        D = np.diag([1.0, 1j, -1.0, -1j])
        assert fbdd.unitaries_equal_up_to_global_phase(D, -D)

    def test_different_unitaries_do_not_match(self):
        I = np.eye(4, dtype=complex)
        Z_diag = np.diag([1.0, -1.0, 1.0, 1.0])
        assert not fbdd.unitaries_equal_up_to_global_phase(I, Z_diag)

    def test_zero_matrices_match(self):
        Z = np.zeros((4, 4), dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(Z, Z)

    def test_zero_vs_nonzero_does_not_match(self):
        I = np.eye(4, dtype=complex)
        Z = np.zeros((4, 4), dtype=complex)
        assert not fbdd.unitaries_equal_up_to_global_phase(I, Z)

    def test_atol_controls_strictness(self):
        I = np.eye(4, dtype=complex)
        perturbed = I + 1e-10 * np.ones((4, 4), dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(I, perturbed, atol=1e-8)
        assert not fbdd.unitaries_equal_up_to_global_phase(I, perturbed, atol=1e-12)
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestUnitariesEqualUpToGlobalPhase -v
```

Expected: AttributeError because `fbdd.unitaries_equal_up_to_global_phase` does not exist yet.

- [ ] **Step 3: Implement the function**

Append to `find_best_diagonal_decomposition.py`:

```python
def unitaries_equal_up_to_global_phase(
    U: np.ndarray,
    V: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Return True iff U and V are equal up to a multiplicative global phase.

    Compares entrywise: finds the largest-magnitude entry of V, derives the
    candidate phase lambda = U[idx] / V[idx], and checks U == lambda * V.

    Both inputs must have the same shape. Works for diagonal matrices,
    full unitaries, and any other complex array of equal shape.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    if U.shape != V.shape:
        return False

    abs_V = np.abs(V)
    idx_flat = int(np.argmax(abs_V))
    max_mag = float(abs_V.flat[idx_flat])

    if max_mag < atol:
        # V is effectively zero; U must be too.
        return bool(np.allclose(U, 0.0, atol=atol))

    lam = U.flat[idx_flat] / V.flat[idx_flat]
    if abs(abs(lam) - 1.0) > max(atol, 1e-6):
        # If lam is not a unit-magnitude number, U and V are not the same
        # operator up to global phase (one of them isn't a scaled version
        # of the other).
        return False

    return bool(np.allclose(U, lam * V, atol=atol))
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestUnitariesEqualUpToGlobalPhase -v
```

Expected: 7 passed.

---

## Task 3: `_validate_diagonal`

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestValidateDiagonal:
    def test_accepts_canonical_6_qubit_example(self):
        D = np.array([1, -1, 1j, -1j] * 16, dtype=complex)
        n = fbdd._validate_diagonal(D, atol=1e-8)
        assert n == 6

    def test_rejects_2d_array(self):
        with pytest.raises(ValueError, match="1-D"):
            fbdd._validate_diagonal(np.eye(4, dtype=complex), atol=1e-8)

    def test_rejects_non_power_of_two_length(self):
        with pytest.raises(ValueError, match="power of two"):
            fbdd._validate_diagonal(np.array([1, 1, 1], dtype=complex), atol=1e-8)

    def test_rejects_length_one(self):
        with pytest.raises(ValueError, match="at least 2"):
            fbdd._validate_diagonal(np.array([1.0], dtype=complex), atol=1e-8)

    def test_rejects_non_unit_modulus(self):
        D = np.array([1, 1, 1, 0.5], dtype=complex)
        with pytest.raises(ValueError, match="magnitude"):
            fbdd._validate_diagonal(D, atol=1e-8)

    def test_rejects_non_array(self):
        with pytest.raises(ValueError, match="ndarray"):
            fbdd._validate_diagonal([1, -1, 1, -1], atol=1e-8)
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestValidateDiagonal -v
```

Expected: AttributeError - `_validate_diagonal` not defined.

- [ ] **Step 3: Implement the function**

Append to `find_best_diagonal_decomposition.py`:

```python
def _validate_diagonal(D_diag: np.ndarray, atol: float) -> int:
    """Validate D_diag and return num_qubits.

    Raises ValueError with informative messages on:
    - non-ndarray input,
    - non-1-D input,
    - length < 2 or length not a power of two,
    - any entry whose magnitude differs from 1 by more than atol.
    """
    if not isinstance(D_diag, np.ndarray):
        raise ValueError(
            "D_diag must be a numpy.ndarray, got "
            f"{type(D_diag).__name__}"
        )
    if D_diag.ndim != 1:
        raise ValueError(
            f"D_diag must be 1-D, got ndim={D_diag.ndim} with shape {D_diag.shape}"
        )
    n_entries = D_diag.shape[0]
    if n_entries < 2:
        raise ValueError(
            f"D_diag must have at least 2 entries, got {n_entries}"
        )
    num_qubits = int(round(math.log2(n_entries)))
    if 2**num_qubits != n_entries:
        raise ValueError(
            f"D_diag length must be a power of two, got {n_entries} "
            f"(closest power of two is 2**{num_qubits} = {2**num_qubits})"
        )
    magnitudes = np.abs(D_diag)
    if not np.allclose(magnitudes, 1.0, atol=atol):
        bad_idx = int(np.argmax(np.abs(magnitudes - 1.0)))
        raise ValueError(
            f"All entries of D_diag must have magnitude 1 (within atol={atol}); "
            f"index {bad_idx} has magnitude {magnitudes[bad_idx]:.6g}"
        )
    return num_qubits
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestValidateDiagonal -v
```

Expected: 6 passed.

---

## Task 4: `z_phase_coefficients_from_diag`

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestZPhaseCoefficients:
    def test_constant_diagonal_has_no_nonzero_subsets(self):
        D = np.ones(4, dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        assert result["num_qubits"] == 2
        assert result["coefficients"] == {}
        assert result["num_nonzero"] == 0
        assert result["total_terms"] == 3
        assert result["max_weight"] == 0
        assert result["weight_histogram"] == {}
        assert abs(result["constant"]) < 1e-10

    def test_z0_pattern_has_single_nonzero_coefficient(self):
        # diag(1, -1, 1, -1) on 2 qubits = exp(i pi/2 * Z_0) up to global phase
        D = np.array([1, -1, 1, -1], dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        assert result["num_qubits"] == 2
        assert set(result["coefficients"].keys()) == {frozenset({0})}
        c = result["coefficients"][frozenset({0})]
        assert abs(abs(c) - np.pi / 2) < 1e-10

    def test_z0z1_pattern(self):
        # diag(1, -1, -1, 1) corresponds to exp(i pi/2 * Z_0 Z_1) up to global phase
        D = np.array([1, -1, -1, 1], dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        nonzero = result["coefficients"]
        assert set(nonzero.keys()) == {frozenset({0, 1})}

    def test_round_trip_random_sparse(self):
        # Construct a 4-qubit diagonal from a random sparse coefficient set
        rng = np.random.default_rng(0)
        n = 4
        chosen_subsets = [
            frozenset({0, 1}),
            frozenset({2}),
            frozenset({1, 3}),
            frozenset({0, 1, 2, 3}),
        ]
        chosen_coeffs = {S: rng.uniform(-np.pi, np.pi) for S in chosen_subsets}

        N = 2**n
        theta = np.zeros(N)
        for k in range(N):
            for S, cS in chosen_coeffs.items():
                sign = 1 - 2 * (bin(_bitmask(S) & k).count("1") % 2)
                theta[k] += cS * sign
        D = np.exp(1j * theta)

        result = fbdd.z_phase_coefficients_from_diag(D)
        recovered = result["coefficients"]
        for S, cS in chosen_coeffs.items():
            assert S in recovered
            assert abs(recovered[S] - cS) < 1e-10
        assert set(recovered.keys()) == set(chosen_subsets)


def _bitmask(S):
    m = 0
    for i in S:
        m |= 1 << i
    return m
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestZPhaseCoefficients -v
```

Expected: AttributeError - `z_phase_coefficients_from_diag` not defined.

- [ ] **Step 3: Implement the function**

Append to `find_best_diagonal_decomposition.py`:

```python
def z_phase_coefficients_from_diag(
    D_diag: np.ndarray,
    atol: float = 1e-12,
) -> dict:
    """Express phases of D_diag as coefficients of Z-product Pauli operators.

    For a diagonal unitary on n qubits, write each phase angle theta_k as

        theta_k = sum_{S subset of {0,...,n-1}} c_S * (-1)^{popcount(S AND k)}

    The coefficient c_S is the Walsh-Hadamard transform of theta:

        c_S = (1 / 2**n) * sum_k (-1)^{popcount(S AND k)} * theta_k.

    The empty subset coefficient c_emptyset (the global phase) is returned
    separately as "constant"; only nonempty subsets with |c_S| > atol are
    returned in "coefficients".

    Parameters
    ----------
    D_diag : np.ndarray
        1-D length-2**n array of unit-modulus complex numbers.
    atol : float
        Threshold for treating a coefficient as zero.

    Returns
    -------
    dict with keys:
        num_qubits, constant, coefficients (dict[frozenset, float]),
        num_nonzero, total_terms, sparsity, max_weight, weight_histogram.
    """
    num_qubits = _validate_diagonal(D_diag, atol=1e-8)
    N = 1 << num_qubits

    theta = np.angle(D_diag).astype(float, copy=False)

    # Walsh-Hadamard transform (Hadamard transform with sign convention
    # (-1)^popcount(S AND k)). In-place butterfly.
    h = theta.copy()
    step = 1
    while step < N:
        for start in range(0, N, step * 2):
            for i in range(start, start + step):
                a = h[i]
                b = h[i + step]
                h[i] = a + b
                h[i + step] = a - b
        step *= 2
    h /= N

    constant = float(h[0])
    coefficients: dict[frozenset, float] = {}
    weight_histogram: dict[int, int] = {}

    for m in range(1, N):
        c = float(h[m])
        if abs(c) > atol:
            S = frozenset(i for i in range(num_qubits) if (m >> i) & 1)
            coefficients[S] = c
            w = len(S)
            weight_histogram[w] = weight_histogram.get(w, 0) + 1

    num_nonzero = len(coefficients)
    total_terms = N - 1
    sparsity = num_nonzero / total_terms if total_terms > 0 else 0.0
    max_weight = max(weight_histogram.keys()) if weight_histogram else 0

    return {
        "num_qubits": num_qubits,
        "constant": constant,
        "coefficients": coefficients,
        "num_nonzero": num_nonzero,
        "total_terms": total_terms,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "weight_histogram": weight_histogram,
    }
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestZPhaseCoefficients -v
```

Expected: 4 passed.

---

## Task 5: Permutation helpers

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestPermutationHelpers:
    def test_bit_permute_identity_is_identity(self):
        n = 3
        perm = (0, 1, 2)
        for k in range(8):
            assert fbdd._bit_permute(k, perm, n) == k

    def test_bit_permute_swap_first_two(self):
        # perm=(1,0,2): bit at position i of new int comes from position perm[i] of old.
        # k=0b001 (bit 0 set) -> new bit 0 = old bit perm[0]=1 = 0; new bit 1 = old bit perm[1]=0 = 1.
        # so 0b001 -> 0b010
        n = 3
        perm = (1, 0, 2)
        assert fbdd._bit_permute(0b001, perm, n) == 0b010
        assert fbdd._bit_permute(0b010, perm, n) == 0b001
        assert fbdd._bit_permute(0b101, perm, n) == 0b110

    def test_permute_diagonal_identity_returns_copy(self):
        D = np.array([1, -1, 1j, -1j], dtype=complex)
        out = fbdd._permute_diagonal(D, (0, 1))
        np.testing.assert_array_equal(out, D)

    def test_permute_diagonal_swap_qubits(self):
        # n=2, swap qubits: index k=0b01 (qubit 0 set) maps to 0b10
        D = np.array([10, 20, 30, 40], dtype=complex)
        out = fbdd._permute_diagonal(D, (1, 0))
        # new[0b00] = D[0b00] = 10
        # new[0b01] = D[bit_permute(0b01,(1,0),2)] = D[0b10] = 30
        # new[0b10] = D[bit_permute(0b10,(1,0),2)] = D[0b01] = 20
        # new[0b11] = D[0b11] = 40
        np.testing.assert_array_equal(out, np.array([10, 30, 20, 40], dtype=complex))

    def test_enumerate_permutations_identity_only(self):
        perms = list(fbdd._enumerate_permutations(num_qubits=4, try_perms=False, rng_seed=0))
        assert perms == [(0, 1, 2, 3)]

    def test_enumerate_permutations_full_for_n_le_4(self):
        perms = list(fbdd._enumerate_permutations(num_qubits=4, try_perms=True, rng_seed=0))
        assert len(perms) == 24
        assert (0, 1, 2, 3) in perms

    def test_enumerate_permutations_sampled_for_n_5_or_6(self):
        perms = list(fbdd._enumerate_permutations(num_qubits=6, try_perms=True, rng_seed=0))
        # identity + 24 sampled (deduplicated)
        assert (0, 1, 2, 3, 4, 5) in perms
        assert 1 < len(perms) <= 25
        assert all(sorted(p) == [0, 1, 2, 3, 4, 5] for p in perms)

    def test_enumerate_permutations_identity_only_for_n_gt_6(self):
        with pytest.warns(UserWarning, match="permutation"):
            perms = list(fbdd._enumerate_permutations(num_qubits=8, try_perms=True, rng_seed=0))
        assert perms == [tuple(range(8))]
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestPermutationHelpers -v
```

Expected: AttributeError on `_bit_permute`, `_permute_diagonal`, `_enumerate_permutations`.

- [ ] **Step 3: Implement the helpers**

Append to `find_best_diagonal_decomposition.py`:

```python
def _bit_permute(k: int, perm: Sequence[int], num_qubits: int) -> int:
    """Permute the bits of integer k according to perm.

    The result has bit i taken from bit perm[i] of k:
        result_bit_i = k_bit_{perm[i]}.

    This matches Qiskit's wiring convention: when a gate is appended to
    qubits perm = [p_0, ..., p_{n-1}], the gate's qubit i is wired to
    circuit qubit p_i. The diagonal entry seen by the gate at circuit
    basis state |k> is therefore D_diag[bit_permute(k, perm, n)].
    """
    result = 0
    for i in range(num_qubits):
        bit = (k >> perm[i]) & 1
        result |= bit << i
    return result


def _permute_diagonal(D_diag: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Return D_diag re-indexed by bit-permutation perm.

    out[k] = D_diag[bit_permute(k, perm, n)] for k in range(2**n).
    """
    n = int(round(math.log2(len(D_diag))))
    N = 1 << n
    out = np.empty_like(D_diag)
    for k in range(N):
        out[k] = D_diag[_bit_permute(k, perm, n)]
    return out


def _enumerate_permutations(
    num_qubits: int,
    try_perms: bool,
    rng_seed: int = 0,
    sampled_n_5_6: int = 24,
) -> list[tuple[int, ...]]:
    """Enumerate qubit permutations to try, per the spec.

    - try_perms=False: only the identity.
    - num_qubits <= 4: all n! permutations.
    - 5 <= num_qubits <= 6: identity + sampled_n_5_6 random distinct permutations.
    - num_qubits > 6: identity only, with a UserWarning.
    """
    identity = tuple(range(num_qubits))
    if not try_perms:
        return [identity]

    if num_qubits <= 4:
        return [tuple(p) for p in itertools.permutations(range(num_qubits))]

    if num_qubits <= 6:
        rng = np.random.default_rng(rng_seed)
        perms = {identity}
        attempts = 0
        max_attempts = sampled_n_5_6 * 8
        while len(perms) < sampled_n_5_6 + 1 and attempts < max_attempts:
            arr = rng.permutation(num_qubits)
            perms.add(tuple(int(x) for x in arr))
            attempts += 1
        return [identity] + [p for p in perms if p != identity]

    warnings.warn(
        f"try_qubit_permutations=True with num_qubits={num_qubits} > 6 is "
        "skipped (combinatorial explosion). Falling back to identity only.",
        UserWarning,
        stacklevel=2,
    )
    return [identity]
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestPermutationHelpers -v
```

Expected: 8 passed.

---

## Task 6: Metrics + scoring

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestMetrics:
    def test_count_metrics_simple_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.rz(0.5, 0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.7, 2)
        m = fbdd._count_metrics(qc)
        assert m["total_gates"] == 4
        assert m["two_qubit"] == 2
        assert m["depth"] == qc.depth()
        assert m["breakdown"]["rz"] == 2
        assert m["breakdown"]["cx"] == 2

    def test_count_metrics_empty_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        m = fbdd._count_metrics(qc)
        assert m["total_gates"] == 0
        assert m["two_qubit"] == 0
        assert m["depth"] == 0
        assert m["breakdown"] == {}

    def test_count_metrics_excludes_barriers(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(1, 0)
        m = fbdd._count_metrics(qc)
        # barrier is not counted as a gate
        assert m["two_qubit"] == 2
        assert m["total_gates"] == 2


class TestScore:
    def test_score_two_qubit_then_depth_returns_tuple(self):
        m = {"two_qubit": 5, "depth": 12, "total_gates": 30}
        assert fbdd._score(m, "two_qubit_then_depth") == (5, 12, 30)

    def test_score_orders_by_two_qubit_first(self):
        a = {"two_qubit": 3, "depth": 100, "total_gates": 100}
        b = {"two_qubit": 4, "depth": 1, "total_gates": 1}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_orders_by_depth_when_tied(self):
        a = {"two_qubit": 3, "depth": 5, "total_gates": 100}
        b = {"two_qubit": 3, "depth": 6, "total_gates": 1}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_orders_by_total_when_tied(self):
        a = {"two_qubit": 3, "depth": 5, "total_gates": 9}
        b = {"two_qubit": 3, "depth": 5, "total_gates": 10}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            fbdd._score({"two_qubit": 1, "depth": 1, "total_gates": 1}, "unknown")
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestMetrics tests/test_find_best_diagonal_decomposition.py::TestScore -v
```

Expected: AttributeError on `_count_metrics`, `_score`.

- [ ] **Step 3: Implement metrics + scoring**

Append to `find_best_diagonal_decomposition.py`:

```python
# Canonical 2-qubit gate names used only for documentation in the printed
# table; the actual two-qubit count is taken from len(qargs) == 2 so any
# non-canonical 2-qubit op is still counted correctly.
_CANONICAL_TWO_QUBIT_GATE_NAMES = frozenset(
    {"cx", "cz", "ecr", "iswap", "swap", "rzz", "rxx", "ryy", "csx", "dcx"}
)
_NON_GATE_OPS = frozenset({"barrier", "measure", "reset", "delay"})


def _count_metrics(qc: QuantumCircuit) -> dict:
    """Count gates, two-qubit gates, depth, and per-op breakdown."""
    total = 0
    two_q = 0
    breakdown: dict[str, int] = {}

    for instr in qc.data:
        op = instr.operation
        name = op.name
        qargs = instr.qubits
        if name in _NON_GATE_OPS:
            continue
        total += 1
        if len(qargs) == 2:
            two_q += 1
        breakdown[name] = breakdown.get(name, 0) + 1

    return {
        "total_gates": total,
        "two_qubit": two_q,
        "depth": qc.depth(),
        "breakdown": breakdown,
    }


def _score(metrics: dict, metric: str) -> tuple:
    """Return a sortable score tuple for the given metric name."""
    if metric != "two_qubit_then_depth":
        raise ValueError(
            f"Unknown metric {metric!r}. Supported metrics: "
            "{'two_qubit_then_depth'}."
        )
    return (metrics["two_qubit"], metrics["depth"], metrics["total_gates"])
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestMetrics tests/test_find_best_diagonal_decomposition.py::TestScore -v
```

Expected: 8 passed.

---

## Task 7: Default basis gates + DiagonalGate strategy

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestDiagonalGateStrategy:
    def test_default_basis_gates_returned_when_neither_supplied(self):
        bg = fbdd._resolve_basis_gates(backend=None, basis_gates=None)
        assert bg == ["rz", "sx", "x", "cx"]

    def test_explicit_basis_gates_passes_through(self):
        bg = fbdd._resolve_basis_gates(backend=None, basis_gates=["rz", "cz"])
        assert bg == ["rz", "cz"]

    def test_diagonal_gate_strategy_is_exact(self):
        D = np.array([1, -1, 1j, -1j], dtype=complex)  # n=2
        qc = fbdd._strategy_diagonal_gate(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        target = np.diag(D)
        assert fbdd.unitaries_equal_up_to_global_phase(U, target, atol=1e-8)

    def test_diagonal_gate_strategy_normalize_phase(self):
        D = 1j * np.array([1, -1, 1j, -1j], dtype=complex)  # n=2, scaled by 1j
        qc = fbdd._strategy_diagonal_gate(
            D_diag_perm=D,
            normalize_phase=True,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        # When normalized, the circuit implements D / D[0]; up to global phase
        # this is the same operator as diag(D).
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)


# Make Operator available to tests.
from qiskit.quantum_info import Operator
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestDiagonalGateStrategy -v
```

Expected: AttributeError on `_resolve_basis_gates`, `_strategy_diagonal_gate`.

- [ ] **Step 3: Implement basis gates resolver and the strategy**

Append to `find_best_diagonal_decomposition.py`:

```python
_DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]


def _resolve_basis_gates(backend, basis_gates):
    """Return the basis_gates list to pass to transpile.

    If backend is provided, return None (let transpile use the backend).
    Otherwise, use basis_gates if given, else the default IBM-style set.
    """
    if backend is not None:
        return None  # transpile uses backend's native instructions
    if basis_gates is not None:
        return list(basis_gates)
    return list(_DEFAULT_BASIS_GATES)


def _strategy_diagonal_gate(
    D_diag_perm: np.ndarray,
    *,
    normalize_phase: bool,
    basis_gates,
    backend,
    optimization_level: int,
    seed: int,
) -> QuantumCircuit:
    """Build a candidate circuit via Qiskit's DiagonalGate + transpile.

    The diagonal D_diag_perm is the *already-permuted* diagonal that the
    candidate is supposed to implement (qubit-permutation has been baked in
    upstream, see docstring of find_best_diagonal_decomposition).
    """
    n = int(round(math.log2(len(D_diag_perm))))

    if normalize_phase:
        D_to_use = D_diag_perm / D_diag_perm[0]
    else:
        D_to_use = D_diag_perm

    qc = QuantumCircuit(n)
    qc.append(DiagonalGate(list(D_to_use)), range(n))

    transpile_kwargs = dict(
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    if backend is not None:
        transpile_kwargs["backend"] = backend
    if basis_gates is not None:
        transpile_kwargs["basis_gates"] = basis_gates

    return transpile(qc, **transpile_kwargs)
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestDiagonalGateStrategy -v
```

Expected: 4 passed.

---

## Task 8: Sparse phase-polynomial strategy

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestSparsePhasePolyStrategy:
    def test_sparse_strategy_z0_diagonal(self):
        # diag(1, -1, 1, -1) = exp(i pi/2 * Z_0) up to a global phase.
        D = np.array([1, -1, 1, -1], dtype=complex)
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)

    def test_sparse_strategy_two_term_diagonal(self):
        # Hand-crafted: c_{0,1} = pi/3, c_{0,1,2} = -pi/5, no others, no global phase.
        n = 3
        N = 1 << n
        c = {frozenset({0, 1}): np.pi / 3, frozenset({0, 1, 2}): -np.pi / 5}
        theta = np.zeros(N)
        for k in range(N):
            for S, cS in c.items():
                m = 0
                for i in S:
                    m |= 1 << i
                sign = 1 - 2 * (bin(m & k).count("1") % 2)
                theta[k] += cS * sign
        D = np.exp(1j * theta)
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)

    def test_sparse_strategy_canonical_6_qubit_is_exact(self):
        D = _canonical_d_diag()
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-7)


def _canonical_d_diag():
    """The 6-qubit example from the user spec."""
    return np.array([
         1,  1,  1j,  1, -1j, -1,  1j,  1,
        -1j, -1, -1, -1,  1j, -1j,  1, -1,
         1j,  1j, -1j, -1j,  1, -1j, -1j,  1j,
        -1j,  1, -1,  1j, -1,  1,  1j, -1,
        -1,  1, -1,  1j, -1,  1,  1j, -1j,
         1,  1, -1j, -1, -1,  1,  1j, -1j,
        -1, -1, -1, -1, -1,  1, -1j,  1,
        -1, -1, -1, -1,  1j,  1j, -1j, -1
    ], dtype=complex)
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestSparsePhasePolyStrategy -v
```

Expected: AttributeError on `_strategy_sparse_phase_poly`.

- [ ] **Step 3: Implement the sparse strategy**

Append to `find_best_diagonal_decomposition.py`:

```python
def _append_z_product_term(
    qc: QuantumCircuit,
    subset_sorted: list[int],
    coefficient: float,
) -> None:
    """Append exp(i * coefficient * Z_S) onto qc, where S = subset_sorted.

    For weight 1 (|S| = 1):
        Z_S = Z_t. exp(i c Z_t) = RZ(-2c) on t (up to a c-independent
        identity sign coming from RZ's global-phase convention).
    For weight >= 2:
        CX ladder onto target t = subset_sorted[-1] computes parity into t,
        RZ(-2c) on t applies the phase, then the reverse CX ladder uncomputes.

    The block applied here equals exp(i * coefficient * Z_S) up to an
    irrelevant global phase (RZ has global phase -1 sometimes); since we
    validate up to global phase the irrelevant factor is absorbed.
    """
    if not subset_sorted:
        return  # constant term handled separately via qc.global_phase

    target = subset_sorted[-1]

    if len(subset_sorted) == 1:
        qc.rz(-2.0 * coefficient, target)
        return

    controls = subset_sorted[:-1]
    for c in controls:
        qc.cx(c, target)
    qc.rz(-2.0 * coefficient, target)
    for c in reversed(controls):
        qc.cx(c, target)


def _build_sparse_phase_poly_circuit(
    D_diag_perm: np.ndarray,
    *,
    normalize_phase: bool,
) -> QuantumCircuit:
    """Build an un-transpiled circuit implementing diag(D_diag_perm).

    Uses the Z-product expansion: per nonempty subset S with coefficient c_S,
    append a CX-ladder + RZ + reverse-ladder block.
    """
    coeffs = z_phase_coefficients_from_diag(D_diag_perm)
    n = coeffs["num_qubits"]

    qc = QuantumCircuit(n, name="sparse_phase_poly")
    if not normalize_phase:
        qc.global_phase = coeffs["constant"]

    # Iterate subsets sorted by weight first, then by their bitmask within a weight.
    def subset_sort_key(S):
        bitmask = 0
        for i in S:
            bitmask |= 1 << i
        return (len(S), bitmask)

    for S in sorted(coeffs["coefficients"].keys(), key=subset_sort_key):
        c_S = coeffs["coefficients"][S]
        subset_sorted = sorted(S)
        _append_z_product_term(qc, subset_sorted, c_S)

    return qc


def _strategy_sparse_phase_poly(
    D_diag_perm: np.ndarray,
    *,
    normalize_phase: bool,
    basis_gates,
    backend,
    optimization_level: int,
    seed: int,
) -> QuantumCircuit:
    """Build a candidate via custom sparse phase-poly synthesis + transpile."""
    raw = _build_sparse_phase_poly_circuit(
        D_diag_perm, normalize_phase=normalize_phase
    )

    transpile_kwargs = dict(
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    if backend is not None:
        transpile_kwargs["backend"] = backend
    if basis_gates is not None:
        transpile_kwargs["basis_gates"] = basis_gates

    return transpile(raw, **transpile_kwargs)
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestSparsePhasePolyStrategy -v
```

Expected: 3 passed.

---

## Task 9: Driver `find_best_diagonal_decomposition`

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestDriver:
    def test_driver_canonical_example_validates_and_returns_best(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=False,
            verbose=False,
        )
        assert "best_circuit" in result
        assert "best_score" in result
        assert "best_metadata" in result
        assert "all_candidates" in result
        assert "diagnostics" in result
        # best circuit must reproduce D up to global phase
        U = Operator(result["best_circuit"]).data
        assert fbdd.unitaries_equal_up_to_global_phase(
            U, np.diag(D), atol=1e-7
        )
        # at least one candidate from each strategy must have validated
        strategies_seen = {c["strategy"] for c in result["all_candidates"]}
        assert "diagonal_gate" in strategies_seen
        assert "sparse_phase_poly" in strategies_seen
        assert all(c["validation"] == "ok" for c in result["all_candidates"])

    def test_driver_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fbdd.find_best_diagonal_decomposition(
                np.array([1, 1, 1], dtype=complex), verbose=False
            )

    def test_driver_score_orders_correctly(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=False,
            verbose=False,
        )
        # all_candidates is sorted ascending by score
        scores = [
            (c["two_qubit"], c["depth"], c["total_gates"])
            for c in result["all_candidates"]
        ]
        assert scores == sorted(scores)
        # best_score equals the first
        assert result["best_score"] == scores[0]

    def test_driver_diagnostics_contain_phase_coefficients(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(1),
            try_qubit_permutations=False,
            verbose=False,
        )
        diag = result["diagnostics"]
        assert diag["num_qubits"] == 6
        assert "phase_coefficients" in diag
        assert "num_candidates_attempted" in diag

    def test_driver_handles_failed_candidate_without_crashing(self, monkeypatch):
        # Force every candidate to fail validation by patching the equality helper.
        D = _canonical_d_diag()
        original = fbdd.unitaries_equal_up_to_global_phase
        monkeypatch.setattr(
            fbdd, "unitaries_equal_up_to_global_phase",
            lambda U, V, atol=1e-8: False
        )
        with pytest.raises(RuntimeError, match="all candidates failed"):
            fbdd.find_best_diagonal_decomposition(
                D, seeds=range(1), try_qubit_permutations=False, verbose=False
            )
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestDriver -v
```

Expected: AttributeError - `find_best_diagonal_decomposition` not yet defined.

- [ ] **Step 3: Implement the driver**

Append to `find_best_diagonal_decomposition.py`:

```python
_VALIDATION_SKIP_THRESHOLD_QUBITS = 12


def find_best_diagonal_decomposition(
    D_diag,
    backend=None,
    basis_gates=None,
    optimization_level: int = 3,
    seeds: Iterable[int] = range(100),
    try_qubit_permutations: bool = True,
    try_global_phase_normalization: bool = True,
    metric: str = "two_qubit_then_depth",
    atol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Find the best exact decomposition of a diagonal unitary.

    Tries multiple strategies, qubit permutations, optional global-phase
    normalization, and a sweep over seed_transpiler values. Validates each
    candidate up to global phase (skipped when num_qubits > 12). Returns
    the best candidate found among the candidates actually attempted.

    This function does NOT prove mathematical optimality - it returns the
    best of the candidates it tried.

    Parameters
    ----------
    D_diag : np.ndarray
        1-D length-2**n complex array of unit-modulus phases.
    backend : optional
        If provided, transpile against this backend (basis_gates ignored).
    basis_gates : optional
        Basis gates list. Defaults to ["rz", "sx", "x", "cx"] if backend is
        None and basis_gates is None.
    optimization_level : int
        Passed to qiskit.transpile.
    seeds : iterable of int
        seed_transpiler values to sweep.
    try_qubit_permutations : bool
        Whether to try non-identity qubit relabelings (see spec).
    try_global_phase_normalization : bool
        Whether to also try dividing the input by D_diag[0].
    metric : str
        Currently only "two_qubit_then_depth" is supported.
    atol : float
        Absolute tolerance for validation.
    verbose : bool
        If True, prints a comparison table after the search.

    Returns
    -------
    dict with keys: best_circuit, best_score, best_metadata, all_candidates,
    diagnostics. See spec section 10 for details.
    """
    num_qubits = _validate_diagonal(np.asarray(D_diag), atol=atol)
    D_diag = np.asarray(D_diag, dtype=complex).copy()

    resolved_basis_gates = _resolve_basis_gates(backend, basis_gates)
    phase_coefficients = z_phase_coefficients_from_diag(D_diag)

    # Validate the metric early.
    _ = _score(
        {"two_qubit": 0, "depth": 0, "total_gates": 0}, metric
    )

    perms = _enumerate_permutations(
        num_qubits, try_perms=try_qubit_permutations, rng_seed=0
    )

    normalizations: list[bool] = [False]
    if try_global_phase_normalization:
        normalizations.append(True)

    seeds_list = list(seeds)
    strategies = ("diagonal_gate", "sparse_phase_poly")

    do_validate = num_qubits <= _VALIDATION_SKIP_THRESHOLD_QUBITS
    if not do_validate and verbose:
        warnings.warn(
            f"Skipping exact validation: num_qubits={num_qubits} exceeds "
            f"the threshold {_VALIDATION_SKIP_THRESHOLD_QUBITS}. Candidates "
            "will be returned with validation='skipped'.",
            UserWarning,
            stacklevel=2,
        )

    total_candidates = (
        len(perms) * len(normalizations) * len(seeds_list) * len(strategies)
    )
    if verbose:
        print(
            f"[find_best_diagonal_decomposition] trying "
            f"{total_candidates} candidates "
            f"({len(perms)} perms x {len(normalizations)} norm x "
            f"{len(seeds_list)} seeds x {len(strategies)} strategies)..."
        )

    candidates: list[dict] = []
    failed: list[dict] = []

    for perm in perms:
        D_perm = _permute_diagonal(D_diag, perm)
        for normalize_phase in normalizations:
            for strategy_name in strategies:
                for seed in seeds_list:
                    try:
                        if strategy_name == "diagonal_gate":
                            qc_t = _strategy_diagonal_gate(
                                D_diag_perm=D_perm,
                                normalize_phase=normalize_phase,
                                basis_gates=resolved_basis_gates,
                                backend=backend,
                                optimization_level=optimization_level,
                                seed=seed,
                            )
                        else:
                            qc_t = _strategy_sparse_phase_poly(
                                D_diag_perm=D_perm,
                                normalize_phase=normalize_phase,
                                basis_gates=resolved_basis_gates,
                                backend=backend,
                                optimization_level=optimization_level,
                                seed=seed,
                            )
                    except Exception as exc:  # pragma: no cover - defensive
                        failed.append({
                            "strategy": strategy_name,
                            "seed": seed,
                            "permutation": perm,
                            "normalize_phase": normalize_phase,
                            "reason": f"transpile_error: {exc!r}",
                        })
                        continue

                    if do_validate:
                        U = Operator(qc_t).data
                        target = np.diag(D_perm)
                        if normalize_phase:
                            target = target / D_perm[0]
                        ok = unitaries_equal_up_to_global_phase(
                            U, target, atol=atol
                        )
                        if not ok:
                            failed.append({
                                "strategy": strategy_name,
                                "seed": seed,
                                "permutation": perm,
                                "normalize_phase": normalize_phase,
                                "reason": "validation_failed",
                            })
                            continue
                        validation = "ok"
                    else:
                        validation = "skipped"

                    metrics = _count_metrics(qc_t)
                    candidates.append({
                        "strategy": strategy_name,
                        "seed": seed,
                        "permutation": perm,
                        "normalize_phase": normalize_phase,
                        "two_qubit": metrics["two_qubit"],
                        "depth": metrics["depth"],
                        "total_gates": metrics["total_gates"],
                        "ops_breakdown": metrics["breakdown"],
                        "validation": validation,
                        "circuit": qc_t,
                    })

    if not candidates:
        raise RuntimeError(
            f"all candidates failed (attempted={total_candidates}, "
            f"failed={len(failed)}). See diagnostics for details. "
            f"Sample failures: {failed[:3]!r}"
        )

    candidates.sort(key=lambda c: _score(c, metric))
    best = candidates[0]

    best_metadata = {k: v for k, v in best.items() if k != "circuit"}
    diagnostics = {
        "num_qubits": num_qubits,
        "phase_coefficients": phase_coefficients,
        "num_candidates_attempted": total_candidates,
        "num_candidates_failed_validation": len(failed),
        "failed_candidates": failed,
    }

    if verbose:
        _print_comparison_table(candidates, best=best)

    return {
        "best_circuit": best["circuit"],
        "best_score": _score(best, metric),
        "best_metadata": best_metadata,
        "all_candidates": candidates,
        "diagnostics": diagnostics,
    }
```

- [ ] **Step 4: Add a stub for `_print_comparison_table` so the driver runs**

Append to `find_best_diagonal_decomposition.py`:

```python
def _print_comparison_table(candidates: list[dict], best: dict) -> None:
    """Placeholder; full implementation in Task 10."""
    print(f"[find_best_diagonal_decomposition] best: {best['strategy']} "
          f"(seed={best['seed']}, perm={best['permutation']}, "
          f"normalize={best['normalize_phase']}) "
          f"-> 2q={best['two_qubit']}, depth={best['depth']}, "
          f"total={best['total_gates']}, validation={best['validation']}")
```

- [ ] **Step 5: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestDriver -v
```

Expected: 5 passed.

---

## Task 10: Comparison table + `__main__` block

**Files:**
- Modify: `QuditsOnQubits/find_best_diagonal_decomposition.py`
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
class TestComparisonTable:
    def test_table_contains_required_columns(self, capsys):
        D = _canonical_d_diag()
        fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        for col in ("strategy", "seed", "perm", "norm", "rz", "cx",
                    "2q", "total", "depth", "validation"):
            assert col in out, f"missing column header: {col!r}"

    def test_table_contains_best_marker(self, capsys):
        D = _canonical_d_diag()
        fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        assert "BEST" in out

    def test_table_rows_match_candidate_count(self, capsys):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        # exactly len(all_candidates) data rows printed; we just check that
        # at least len(all_candidates) lines start with one of the strategy
        # names.
        candidate_count = len(result["all_candidates"])
        rows = sum(
            1 for line in out.splitlines()
            if line.lstrip().startswith(("diagonal_gate", "sparse_phase_poly"))
        )
        assert rows == candidate_count
```

- [ ] **Step 2: Run tests, verify they fail**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestComparisonTable -v
```

Expected: assertions fail because the placeholder `_print_comparison_table` from Task 9 doesn't print column headers / per-row data / `BEST` marker.

- [ ] **Step 3: Replace the placeholder with the full table printer**

In `find_best_diagonal_decomposition.py`, replace `_print_comparison_table` with:

```python
def _print_comparison_table(candidates: list[dict], best: dict) -> None:
    """Print a plain-text comparison table sorted by score.

    Columns: strategy | seed | perm | norm | rz | cx | 2q | total | depth |
    validation. The row matching `best` is annotated with a leading 'BEST'
    column.
    """
    header = (
        f"{'best':<5} {'strategy':<20} {'seed':>5} "
        f"{'perm':<24} {'norm':<5} "
        f"{'rz':>5} {'cx':>5} {'2q':>5} {'total':>6} {'depth':>6} "
        f"{'validation':<12}"
    )
    print()
    print(header)
    print("-" * len(header))

    for cand in candidates:
        is_best = cand is best
        marker = "BEST " if is_best else "     "
        rz_count = cand["ops_breakdown"].get("rz", 0)
        cx_count = cand["ops_breakdown"].get("cx", 0)
        perm_str = ",".join(str(x) for x in cand["permutation"])
        if len(perm_str) > 23:
            perm_str = perm_str[:20] + "..."
        norm_str = "yes" if cand["normalize_phase"] else "no"
        row = (
            f"{marker:<5} {cand['strategy']:<20} {cand['seed']:>5} "
            f"{perm_str:<24} {norm_str:<5} "
            f"{rz_count:>5} {cx_count:>5} {cand['two_qubit']:>5} "
            f"{cand['total_gates']:>6} {cand['depth']:>6} "
            f"{cand['validation']:<12}"
        )
        print(row)
    print()
    print(
        f"BEST -> strategy={best['strategy']}, seed={best['seed']}, "
        f"perm={best['permutation']}, normalize={best['normalize_phase']}: "
        f"2q={best['two_qubit']}, depth={best['depth']}, "
        f"total={best['total_gates']}, validation={best['validation']}"
    )
    print(
        "Note: result is best among candidates tried, NOT a proof of "
        "mathematical optimality."
    )
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestComparisonTable -v
```

Expected: 3 passed.

- [ ] **Step 5: Add `__main__` block**

Append to `find_best_diagonal_decomposition.py`:

```python
if __name__ == "__main__":
    D_diag = np.array([
         1,  1,  1j,  1, -1j, -1,  1j,  1,
        -1j, -1, -1, -1,  1j, -1j,  1, -1,
         1j,  1j, -1j, -1j,  1, -1j, -1j,  1j,
        -1j,  1, -1,  1j, -1,  1,  1j, -1,
        -1,  1, -1,  1j, -1,  1,  1j, -1j,
         1,  1, -1j, -1, -1,  1,  1j, -1j,
        -1, -1, -1, -1, -1,  1, -1j,  1,
        -1, -1, -1, -1,  1j,  1j, -1j, -1
    ], dtype=complex)

    print("=" * 72)
    print("find_best_diagonal_decomposition: 6-qubit example from spec")
    print("=" * 72)

    coeffs = z_phase_coefficients_from_diag(D_diag)
    print(
        f"Phase polynomial: num_qubits={coeffs['num_qubits']}, "
        f"nonzero Z-product terms={coeffs['num_nonzero']}/"
        f"{coeffs['total_terms']} "
        f"(sparsity={coeffs['sparsity']:.3f}, "
        f"max weight={coeffs['max_weight']})"
    )
    print(f"Weight histogram: {coeffs['weight_histogram']}")
    print()

    result = find_best_diagonal_decomposition(
        D_diag,
        seeds=range(20),
        try_qubit_permutations=False,
        verbose=True,
    )
```

- [ ] **Step 6: Verify the script runs end-to-end**

From `c:\Users\szymo\QuditsOnQubits\QuditsOnQubits`:

```
python find_best_diagonal_decomposition.py
```

Expected output: header banner, phase-polynomial summary, the comparison table with at least 80 candidate rows (20 seeds x 2 norm x 2 strategies = 80), one row marked `BEST`, final summary line, optimality disclaimer.

The best candidate row should reach `validation = ok` (the script raises if all candidates fail). If the run takes more than ~60 seconds, that's fine - it's a one-shot demonstration; document the wall time in the run log if convenient.

---

## Task 11: Final smoke test on the canonical example with permutations

**Files:**
- Modify: `QuditsOnQubits/tests/test_find_best_diagonal_decomposition.py`

- [ ] **Step 1: Add an end-to-end test that exercises permutations**

Append to the test file:

```python
class TestEndToEndWithPermutations:
    @pytest.mark.slow
    def test_canonical_with_permutations_validates(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=True,
            verbose=False,
        )
        # All accepted candidates must be marked validation == "ok"
        assert all(c["validation"] == "ok" for c in result["all_candidates"])
        # The best candidate's circuit, applied to the qubits implied by its
        # permutation, must reproduce D up to global phase.
        best = result["best_metadata"]
        D_perm = fbdd._permute_diagonal(D, best["permutation"])
        if best["normalize_phase"]:
            target = np.diag(D_perm) / D_perm[0]
        else:
            target = np.diag(D_perm)
        U = Operator(result["best_circuit"]).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, target, atol=1e-7)
```

- [ ] **Step 2: Run the slow test (takes ~30-60 s)**

```
pytest tests/test_find_best_diagonal_decomposition.py::TestEndToEndWithPermutations -v
```

Expected: 1 passed (the `slow` marker is just a label - pytest still runs it by default).

- [ ] **Step 3: Run the entire test suite for this file**

```
pytest tests/test_find_best_diagonal_decomposition.py -v
```

Expected: every test passes (running totals: 7 + 6 + 4 + 8 + 8 + 4 + 3 + 5 + 3 + 1 = ~49 tests, depending on exact splits).

- [ ] **Step 4: Sanity-check: nothing else in the repo regressed**

```
pytest tests -v --ignore=tests/test_find_best_diagonal_decomposition.py -x --maxfail=1
```

Expected: pre-existing tests still pass (or at least, no new failure introduced by our additions). If pre-existing tests were already failing, that's not in scope to fix.

---

## Self-Review Checklist (run after writing the plan)

**Spec coverage:** every spec section maps to at least one task:
- Section 1 (validation) -> Task 3
- Section 2 (up-to-global-phase equality) -> Task 2
- Section 3 (Z-product coefficients) -> Task 4
- Section 4 (DiagonalGate strategy) -> Task 7
- Section 5 (sparse phase-poly strategy) -> Task 8
- Section 6 (driver) -> Task 9
- Section 7 (validation logic) -> embedded in Task 9
- Section 8 (metrics) -> Task 6
- Section 9 (scoring) -> Task 6
- Section 10 (result object) -> Task 9
- Section 11 (comparison table) -> Task 10
- Tests (spec section "Testing") -> Tasks 2-11
- Permutation enumeration -> Task 5

**Placeholder scan:** zero TBDs / TODOs / "implement later". Each step has actual code or a precise command. The Task 9 step that uses a placeholder `_print_comparison_table` is replaced inside the same plan in Task 10.

**Type and signature consistency:** strategy functions both take the same keyword-only arguments (`D_diag_perm`, `normalize_phase`, `basis_gates`, `backend`, `optimization_level`, `seed`). The driver calls them by exactly those names. The result dict keys match what the tests assert.
