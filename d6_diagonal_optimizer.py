"""Optimize a 6-qubit realization of a diagonal gate D[Lambda] for d=6.

This module implements the construction described in the AME(4,6) context:

    * Two ququits of dimension d = 6 are encoded into 6 physical qubits.
    * Each qudit level a in {0, ..., 5} is encoded into 3 qubits via
      its plain binary representation:

          0 -> 000
          1 -> 001
          2 -> 010
          3 -> 011
          4 -> 100
          5 -> 101

      Levels 6 and 7 (i.e. binary strings 110 and 111) are *outside* the
      qudit code space.

    * The diagonal gate D[Lambda] acts on (a, b) for a, b in {0,...,5}
      and therefore has 36 legal phases.  The remaining 28 basis states
      of the 64-dimensional 6-qubit Hilbert space are "don't-care": the
      optimizer is free to choose their phases to make the resulting
      circuit cheaper.

    * The implementation builds a phase-polynomial / Walsh-Hadamard
      expansion of the 64-vector of phases and synthesizes the
      corresponding diagonal unitary as a sequence of ``CNOT`` ladders
      and ``RZ`` rotations (so-called *phase gadgets*).

The public entry point is :func:`optimize_d6_diagonal` which performs
random search and simulated annealing over the don't-care phases and
returns a dictionary of artifacts including:

    - the chosen 64-vector of phases,
    - the Walsh coefficients,
    - a phase-gadget circuit (only ``CX`` and ``RZ`` gates),
    - the same circuit transpiled to a Qiskit basis,
    - a baseline ``DiagonalGate``-based circuit for comparison,
    - resource metrics (depth / CX count / RZ count) for each variant.

Conventions
-----------

* Endianness for the qubit register.  In Qiskit, basis-state index ``k``
  corresponds to bit ``i`` taking value ``(k >> i) & 1`` on physical
  qubit ``i``.  In this module we adopt the convention

      index(a, b) = (a << 3) | b,        for a, b in {0, ..., 5}.

  Equivalently, qubits 5, 4, 3 carry the 3-bit binary code of qudit
  ``a`` (with qubit 5 the most-significant bit of ``a``) and qubits
  2, 1, 0 carry the code of qudit ``b``.  This is documented and used
  consistently throughout.

* Order of entries in the input ``diag36``.  We support two readings:

      - ``order="row-major"`` (default) means
        ``diag36[6 * a + b]`` is the phase of state ``|a, b>``.
      - ``order="col-major"`` means
        ``diag36[a + 6 * b]`` is the phase of state ``|a, b>``.

  The example vector from the AME(4,6) paper is laid out row-major,
  so that is what the module defaults to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Operator

__all__ = [
    "encode_qudit_level",
    "pair_to_qubit_index",
    "qubit_index_to_pair",
    "legal_indices",
    "illegal_indices",
    "diag36_to_theta64",
    "normalize_phase",
    "walsh_coefficients",
    "synthesize_phase_gadgets",
    "phase_polynomial_cost",
    "verify_on_code_space",
    "extract_diagonal_from_circuit",
    "random_search",
    "simulated_annealing",
    "multi_restart_simulated_annealing",
    "greedy_coordinate_descent",
    "multi_restart_greedy",
    "two_opt_descent",
    "lp_relax_walsh_l1",
    "transpile_metric",
    "optimize_d6_diagonal",
]


# ---------------------------------------------------------------------------
# Index / encoding helpers
# ---------------------------------------------------------------------------

QUDIT_DIM = 6
NUM_QUBITS_PER_QUDIT = 3
NUM_QUBITS = 2 * NUM_QUBITS_PER_QUDIT  # 6
NUM_LEGAL_STATES = QUDIT_DIM * QUDIT_DIM  # 36
NUM_TOTAL_STATES = 1 << NUM_QUBITS  # 64
NUM_ILLEGAL_STATES = NUM_TOTAL_STATES - NUM_LEGAL_STATES  # 28


def encode_qudit_level(a: int) -> int:
    """Return the 3-bit binary encoding of qudit level ``a`` (0..5).

    Levels 0..5 are encoded as the integers 0..5 (binary 000..101).
    Levels 6 and 7 are *not* legal qudit levels but are returned as 6/7
    for completeness.

    Raises
    ------
    ValueError
        If ``a`` is outside the range ``{0, ..., 7}``.
    """
    if not (0 <= a < 8):
        raise ValueError(f"qudit level must be in 0..7, got {a}")
    return int(a)


def pair_to_qubit_index(a: int, b: int) -> int:
    """Map a pair ``(a, b)`` of qudit levels to a 6-qubit basis index.

    Convention: ``index = (a << 3) | b``.  Qubits 5,4,3 carry ``a``'s
    3-bit code, qubits 2,1,0 carry ``b``'s 3-bit code.
    """
    if not (0 <= a < 8) or not (0 <= b < 8):
        raise ValueError(f"qudit levels must be in 0..7, got ({a}, {b})")
    return (encode_qudit_level(a) << NUM_QUBITS_PER_QUDIT) | encode_qudit_level(b)


def qubit_index_to_pair(k: int) -> tuple[int, int]:
    """Inverse of :func:`pair_to_qubit_index` (no legality check)."""
    if not (0 <= k < NUM_TOTAL_STATES):
        raise ValueError(f"qubit index must be in 0..63, got {k}")
    a = (k >> NUM_QUBITS_PER_QUDIT) & 0b111
    b = k & 0b111
    return a, b


def legal_indices() -> list[int]:
    """Return the sorted list of 36 legal 6-qubit basis indices."""
    return sorted(
        pair_to_qubit_index(a, b)
        for a in range(QUDIT_DIM)
        for b in range(QUDIT_DIM)
    )


def illegal_indices() -> list[int]:
    """Return the sorted list of 28 illegal (don't-care) 6-qubit indices."""
    legal = set(legal_indices())
    return sorted(k for k in range(NUM_TOTAL_STATES) if k not in legal)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _phases_from_diag36(diag36) -> np.ndarray:
    """Convert a length-36 input to a real array of 36 phases (radians).

    Auto-detection rules:

    * Complex dtype, or any non-zero imaginary component -> interpret as
      complex unit-modulus values; phases = ``np.angle(...)``.
    * Real dtype: if every entry is unit-modulus (i.e. all ``|x| = 1``
      within tol), interpret as complex; otherwise treat the values as
      raw phase angles in radians.
    """
    arr = np.asarray(diag36)
    if arr.shape != (NUM_LEGAL_STATES,):
        raise ValueError(
            f"diag36 must have length {NUM_LEGAL_STATES}, got {arr.shape}"
        )

    if np.iscomplexobj(arr):
        magnitudes = np.abs(arr)
        if not np.allclose(magnitudes, 1.0, atol=1e-6):
            bad = int(np.argmax(np.abs(magnitudes - 1.0)))
            raise ValueError(
                "Complex input must have unit modulus; entry "
                f"{bad} has |z| = {magnitudes[bad]:.6g}."
            )
        return np.angle(arr).astype(float, copy=False)

    arr_real = arr.astype(float)
    if np.allclose(np.abs(arr_real), 1.0, atol=1e-9):
        # Looks like real ±1, treat as unit-modulus complex.
        return np.angle(arr_real.astype(complex))
    return arr_real


def _diag36_complex(diag36) -> np.ndarray:
    """Return the canonical complex unit-modulus form of ``diag36``."""
    phases = _phases_from_diag36(diag36)
    return np.exp(1j * phases)


def diag36_to_theta64(
    diag36,
    illegal_phases: Optional[Sequence[float]] = None,
    order: str = "row-major",
) -> np.ndarray:
    """Embed a 36-dimensional diagonal into a 64-dimensional phase vector.

    Parameters
    ----------
    diag36 : array-like, length 36
        Phases or unit-modulus complex numbers describing D[Lambda] on
        the qudit code space.
    illegal_phases : sequence of 28 reals, optional
        Phases to place on the 28 don't-care indices.  Defaults to zero
        on every illegal index (i.e. ``D = +1`` there).
    order : {"row-major", "col-major"}
        How the entries of ``diag36`` are laid out.  See the module
        docstring.

    Returns
    -------
    np.ndarray, shape (64,)
        Real phase vector ``theta64`` such that ``D|k> = exp(i * theta64[k]) |k>``.
    """
    if order not in ("row-major", "col-major"):
        raise ValueError("order must be 'row-major' or 'col-major'")

    phases36 = _phases_from_diag36(diag36)

    theta64 = np.zeros(NUM_TOTAL_STATES, dtype=float)
    for a in range(QUDIT_DIM):
        for b in range(QUDIT_DIM):
            if order == "row-major":
                src = a * QUDIT_DIM + b
            else:  # col-major
                src = a + QUDIT_DIM * b
            dst = pair_to_qubit_index(a, b)
            theta64[dst] = phases36[src]

    if illegal_phases is not None:
        ill = illegal_indices()
        if len(illegal_phases) != len(ill):
            raise ValueError(
                f"illegal_phases must have length {len(ill)}, "
                f"got {len(illegal_phases)}"
            )
        for k, phi in zip(ill, illegal_phases):
            theta64[k] = float(phi)

    return theta64


def normalize_phase(theta) -> np.ndarray:
    """Wrap phases into the half-open interval ``(-pi, pi]``.

    Numerically the closed boundary at ``+pi`` is included to avoid
    spurious off-by-one wrapping when the input is essentially real
    valued and equal to ``+pi``.
    """
    arr = np.asarray(theta, dtype=float)
    out = np.mod(arr, 2.0 * np.pi)
    out = np.where(out > np.pi + 1e-15, out - 2.0 * np.pi, out)
    out = np.where(out < -np.pi + 1e-15, out + 2.0 * np.pi, out)
    return out


# ---------------------------------------------------------------------------
# Walsh-Hadamard / phase polynomial
# ---------------------------------------------------------------------------

def walsh_coefficients(
    theta64,
    tol: float = 1e-10,
    num_qubits: int = NUM_QUBITS,
) -> dict[int, float]:
    """Return the nonzero Walsh-Hadamard coefficients of ``theta64``.

    For an ``N = 2**n`` real vector ``theta``, returns a ``dict``
    mapping integer mask ``S`` (``0 <= S < N``) to coefficient ``c_S``
    such that

        theta(x) = sum_S c_S * (-1)^{popcount(S AND x)}.

    Coefficients with absolute value at most ``tol`` are dropped.  The
    global / "constant" term ``c_0`` is included in the dictionary if
    nonzero (callers should treat ``S = 0`` as the global phase and
    *not* synthesize it as a phase gadget).
    """
    N = 1 << num_qubits
    arr = np.asarray(theta64, dtype=float)
    if arr.shape != (N,):
        raise ValueError(f"theta vector must have length {N}, got {arr.shape}")

    h = arr.copy()
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

    return {m: float(h[m]) for m in range(N) if abs(h[m]) > tol}


# ---------------------------------------------------------------------------
# Phase-gadget synthesis
# ---------------------------------------------------------------------------

def _support_from_mask(mask: int, num_qubits: int) -> list[int]:
    """Return the sorted list of qubit indices in the support of ``mask``."""
    return [i for i in range(num_qubits) if (mask >> i) & 1]


def synthesize_phase_gadgets(
    theta64,
    tol: float = 1e-10,
    num_qubits: int = NUM_QUBITS,
    name: str = "phase_gadgets",
) -> QuantumCircuit:
    """Synthesize ``diag(exp(i * theta64))`` as CNOT ladders + RZ rotations.

    For every nonzero Walsh coefficient ``c_S`` with support
    ``S = (q_0, ..., q_{k-1})`` the circuit appends, with target
    ``q_{k-1}``:

        CX(q_0, q_{k-1}); CX(q_1, q_{k-1}); ...; CX(q_{k-2}, q_{k-1});
        RZ(-2 c_S, q_{k-1});
        CX(q_{k-2}, q_{k-1}); ...; CX(q_0, q_{k-1}).

    For ``|S| = 1`` only the ``RZ(-2 c_S)`` is used (no CNOTs).

    The constant Walsh term ``c_0`` is folded into ``qc.global_phase``.

    The conversion ``RZ(phi) = exp(-i phi Z / 2) <=> exp(i c Z) requires
    phi = -2 c`` matches the Qiskit convention and was double-checked
    against ``DiagonalGate`` numerically.
    """
    coeffs = walsh_coefficients(theta64, tol=tol, num_qubits=num_qubits)

    qc = QuantumCircuit(num_qubits, name=name)

    if 0 in coeffs:
        qc.global_phase = coeffs[0]

    nonzero = [(m, c) for m, c in coeffs.items() if m != 0]
    # Synthesize light gadgets first (low support size) for readability;
    # this also tends to give the transpiler more room to merge gates.
    nonzero.sort(key=lambda pair: (bin(pair[0]).count("1"), pair[0]))

    for mask, c in nonzero:
        support = _support_from_mask(mask, num_qubits)
        target = support[-1]
        controls = support[:-1]
        for q in controls:
            qc.cx(q, target)
        qc.rz(-2.0 * c, target)
        for q in reversed(controls):
            qc.cx(q, target)

    return qc


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """A small bag of cost components for a phase-polynomial expansion."""

    num_nonzero_terms: int
    cx_estimate: int
    weighted_terms: float
    max_support: int
    cost: float


def phase_polynomial_cost(
    theta64,
    tol: float = 1e-10,
    num_qubits: int = NUM_QUBITS,
    weights: Optional[dict] = None,
) -> CostBreakdown:
    """Compute several cost figures for ``theta64`` and a combined score.

    Components
    ----------
    num_nonzero_terms
        Number of nonzero Walsh coefficients ``c_S`` excluding ``S = 0``.
    cx_estimate
        Naive CNOT cost of synthesizing the phase gadgets:
        ``sum_S 2 * (popcount(S) - 1)`` over nonzero ``S`` with
        popcount >= 2.
    weighted_terms
        ``sum_S popcount(S)`` over nonzero ``S != 0``: penalizes high-
        weight terms more.
    max_support
        Maximum ``popcount(S)`` over nonzero ``S != 0`` (or ``0`` if the
        polynomial is empty).

    The default combined cost is

        cost = cx_estimate + 0.1 * num_nonzero_terms + 0.05 * max_support

    Pass ``weights`` (a dict with keys ``cx``, ``terms``, ``support``,
    ``weighted``) to use a different convex combination.
    """
    coeffs = walsh_coefficients(theta64, tol=tol, num_qubits=num_qubits)
    nonzero_masks = [m for m in coeffs if m != 0]

    num_nonzero = len(nonzero_masks)
    cx_est = 0
    weighted = 0.0
    max_supp = 0
    for m in nonzero_masks:
        k = bin(m).count("1")
        if k >= 2:
            cx_est += 2 * (k - 1)
        weighted += k
        max_supp = max(max_supp, k)

    w = {"cx": 1.0, "terms": 0.1, "support": 0.05, "weighted": 0.0}
    if weights is not None:
        w.update(weights)

    cost = (
        w["cx"] * cx_est
        + w["terms"] * num_nonzero
        + w["support"] * max_supp
        + w["weighted"] * weighted
    )

    return CostBreakdown(
        num_nonzero_terms=num_nonzero,
        cx_estimate=cx_est,
        weighted_terms=weighted,
        max_support=max_supp,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Don't-care optimizers
# ---------------------------------------------------------------------------

def _resolve_alphabet(
    phase_alphabet,
    legal_phases: np.ndarray,
) -> list[float]:
    """Resolve the alphabet used to assign phases to don't-care states.

    * ``"default"`` -> ``[0, 2 pi / 3, -2 pi / 3]`` (the natural choice
      for the omega_3 example).
    * ``None``      -> the unique phases that already appear in
      ``legal_phases`` (rounded to 12 decimals).  This keeps the
      optimizer in the "input alphabet" without exposing continuous
      freedom.  Always falls back to ``[0]`` if the input is identically
      zero.
    * iterable of floats -> used as-is (cast to ``list[float]``).
    """
    if phase_alphabet == "default":
        return [0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0]
    if phase_alphabet is None:
        unique = sorted({round(float(x), 12) for x in legal_phases})
        if not unique:
            return [0.0]
        return list(unique)
    return [float(x) for x in phase_alphabet]


def _build_theta64(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    illegal_phases: Sequence[float],
) -> np.ndarray:
    """Assemble the 64-vector by placing legal/illegal phases at their slots."""
    theta = np.zeros(NUM_TOTAL_STATES, dtype=float)
    for k, phi in zip(legal_idx, legal_phases):
        theta[k] = phi
    for k, phi in zip(illegal_idx, illegal_phases):
        theta[k] = phi
    return theta


def random_search(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    max_iters: int,
    seed: int,
    cost_weights: Optional[dict] = None,
) -> tuple[list[float], CostBreakdown]:
    """Random sampling over discrete phase alphabets for the 28 don't-cares."""
    rng = np.random.default_rng(seed)
    n_illegal = len(illegal_idx)
    n_alpha = len(alphabet)
    alpha_arr = np.asarray(alphabet, dtype=float)

    # Initial seed: zeros (no phase on don't-care states)
    best_phases: list[float] = [0.0] * n_illegal
    theta = _build_theta64(legal_phases, legal_idx, illegal_idx, best_phases)
    best_cost_obj = phase_polynomial_cost(
        normalize_phase(theta), weights=cost_weights
    )

    for _ in range(max_iters):
        choices = rng.integers(0, n_alpha, size=n_illegal)
        candidate = alpha_arr[choices].tolist()
        theta = _build_theta64(legal_phases, legal_idx, illegal_idx, candidate)
        cost_obj = phase_polynomial_cost(
            normalize_phase(theta), weights=cost_weights
        )
        if cost_obj.cost < best_cost_obj.cost:
            best_cost_obj = cost_obj
            best_phases = candidate

    return best_phases, best_cost_obj


def simulated_annealing(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    max_iters: int,
    seed: int,
    cost_weights: Optional[dict] = None,
    init_phases: Optional[Sequence[float]] = None,
    T_start: Optional[float] = None,
    T_end: float = 0.05,
) -> tuple[list[float], CostBreakdown]:
    """Single-flip simulated annealing over the discrete phase alphabet.

    A "move" picks one don't-care index at random and replaces its phase
    by another value drawn uniformly from ``alphabet``.  The Metropolis
    acceptance probability uses an exponential cooling schedule from
    ``T_start`` to ``T_end`` over ``max_iters`` iterations.
    """
    rng = np.random.default_rng(seed)
    n_illegal = len(illegal_idx)
    n_alpha = len(alphabet)
    alpha_arr = np.asarray(alphabet, dtype=float)

    if init_phases is None:
        phases = list(alpha_arr[rng.integers(0, n_alpha, size=n_illegal)])
    else:
        if len(init_phases) != n_illegal:
            raise ValueError(
                f"init_phases must have length {n_illegal}, "
                f"got {len(init_phases)}"
            )
        phases = [float(x) for x in init_phases]

    theta = _build_theta64(legal_phases, legal_idx, illegal_idx, phases)
    cost_obj = phase_polynomial_cost(
        normalize_phase(theta), weights=cost_weights
    )

    best_phases = list(phases)
    best_cost_obj = cost_obj

    if T_start is None:
        T_start = max(2.0, 0.5 * float(cost_obj.cost))

    if max_iters <= 1:
        decay = 1.0
    else:
        decay = (T_end / T_start) ** (1.0 / (max_iters - 1))

    T = T_start
    for _ in range(max_iters):
        i = int(rng.integers(0, n_illegal))
        old_val = phases[i]
        new_val = float(alpha_arr[int(rng.integers(0, n_alpha))])
        if new_val == old_val and n_alpha > 1:
            # Force a real move when possible; otherwise just continue.
            new_val = float(alpha_arr[(int(rng.integers(0, n_alpha)) + 1) % n_alpha])

        phases[i] = new_val
        theta = _build_theta64(legal_phases, legal_idx, illegal_idx, phases)
        cand_cost = phase_polynomial_cost(
            normalize_phase(theta), weights=cost_weights
        )
        delta = cand_cost.cost - cost_obj.cost

        accept = (delta <= 0.0) or (rng.random() < math.exp(-delta / max(T, 1e-9)))
        if accept:
            cost_obj = cand_cost
            if cost_obj.cost < best_cost_obj.cost:
                best_cost_obj = cost_obj
                best_phases = list(phases)
        else:
            phases[i] = old_val

        T *= decay

    return best_phases, best_cost_obj


def multi_restart_simulated_annealing(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    max_iters_per_run: int,
    n_restarts: int,
    seed: int,
    cost_weights: Optional[dict] = None,
    init_phases: Optional[Sequence[float]] = None,
) -> tuple[list[float], CostBreakdown]:
    """Run :func:`simulated_annealing` ``n_restarts`` times, return best.

    The first run may be warm-started with ``init_phases`` (when given);
    every subsequent restart starts from a fresh random sample.  Restart
    seeds are derived deterministically from ``seed``.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    rng = np.random.default_rng(seed)
    best_phases: Optional[list[float]] = None
    best_cost: Optional[CostBreakdown] = None

    for r in range(n_restarts):
        run_seed = int(rng.integers(0, 2**31 - 1))
        run_init = init_phases if r == 0 else None
        phases, cost_obj = simulated_annealing(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            max_iters=max_iters_per_run,
            seed=run_seed,
            cost_weights=cost_weights,
            init_phases=run_init,
        )
        if best_cost is None or cost_obj.cost < best_cost.cost:
            best_phases = phases
            best_cost = cost_obj

    assert best_phases is not None and best_cost is not None
    return best_phases, best_cost


def greedy_coordinate_descent(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    init_phases: Sequence[float],
    eval_cost: "callable",
    max_passes: int = 12,
) -> tuple[list[float], float]:
    """Greedy coordinate descent (hill climbing) over don't-care phases.

    For each pass, walks the 28 don't-care indices in order; for each
    index it tries every alphabet value and keeps the one with lowest
    ``eval_cost(theta64)``.  Stops when no index improves in a full
    pass, or after ``max_passes``.

    ``eval_cost`` is a callable taking a 64-vector of phases and
    returning a real number to minimize.  This is the hook that lets
    callers swap in a transpile-based metric for final polishing.
    """
    phases = list(init_phases)
    n = len(illegal_idx)
    if len(phases) != n:
        raise ValueError(
            f"init_phases must have length {n}, got {len(phases)}"
        )

    theta = _build_theta64(legal_phases, legal_idx, illegal_idx, phases)
    best_cost = float(eval_cost(theta))

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            old_val = phases[i]
            best_val = old_val
            best_local = best_cost
            for v in alphabet:
                if v == old_val:
                    continue
                phases[i] = v
                theta = _build_theta64(
                    legal_phases, legal_idx, illegal_idx, phases
                )
                c = float(eval_cost(theta))
                if c < best_local - 1e-12:
                    best_local = c
                    best_val = v
            phases[i] = best_val
            if best_val != old_val:
                best_cost = best_local
                improved = True
        if not improved:
            break

    return phases, best_cost


def multi_restart_greedy(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    eval_cost: "callable",
    n_restarts: int,
    seed: int,
    init_phases: Optional[Sequence[float]] = None,
    max_passes: int = 12,
) -> tuple[list[float], float]:
    """Run greedy coordinate descent from several random starts; keep best."""
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    rng = np.random.default_rng(seed)
    n = len(illegal_idx)
    alpha_arr = np.asarray(alphabet, dtype=float)

    best_phases: Optional[list[float]] = None
    best_cost = float("inf")

    for r in range(n_restarts):
        if r == 0 and init_phases is not None:
            start = list(init_phases)
        elif r == 0:
            start = [0.0] * n  # all-don't-care = 1; matches the user's naive baseline
        else:
            start = list(alpha_arr[rng.integers(0, len(alpha_arr), size=n)])
        phases, cost = greedy_coordinate_descent(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            init_phases=start,
            eval_cost=eval_cost,
            max_passes=max_passes,
        )
        if cost < best_cost:
            best_cost = cost
            best_phases = phases

    assert best_phases is not None
    return best_phases, best_cost


def two_opt_descent(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    alphabet: list[float],
    init_phases: Sequence[float],
    eval_cost: "callable",
    max_passes: int = 4,
) -> tuple[list[float], float]:
    """Pairwise local search over don't-care phases.

    For every unordered pair ``(i, j)`` of don't-care indices, tries every
    joint assignment in ``alphabet x alphabet`` (skipping the current
    one) and accepts the best.  Costs ``O(28 * 27 / 2 * |alphabet|^2)``
    cost evaluations per pass; useful as a stronger local search after
    1-opt greedy converges.
    """
    phases = list(init_phases)
    n = len(illegal_idx)
    if len(phases) != n:
        raise ValueError(
            f"init_phases must have length {n}, got {len(phases)}"
        )

    theta = _build_theta64(legal_phases, legal_idx, illegal_idx, phases)
    best_cost = float(eval_cost(theta))

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                old_i, old_j = phases[i], phases[j]
                best_pair = (old_i, old_j)
                best_local = best_cost
                for vi in alphabet:
                    for vj in alphabet:
                        if vi == old_i and vj == old_j:
                            continue
                        phases[i], phases[j] = vi, vj
                        theta = _build_theta64(
                            legal_phases, legal_idx, illegal_idx, phases
                        )
                        c = float(eval_cost(theta))
                        if c < best_local - 1e-12:
                            best_local = c
                            best_pair = (vi, vj)
                phases[i], phases[j] = best_pair
                if best_pair != (old_i, old_j):
                    best_cost = best_local
                    improved = True
        if not improved:
            break

    return phases, best_cost


def _walsh_full_no_threshold(
    theta: np.ndarray,
    num_qubits: int = NUM_QUBITS,
) -> np.ndarray:
    """Return the full ``2**n`` vector of Walsh coefficients (no tol)."""
    N = 1 << num_qubits
    h = np.asarray(theta, dtype=float).copy()
    if h.shape != (N,):
        raise ValueError(f"theta must have length {N}")
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
    return h


def lp_relax_walsh_l1(
    legal_phases: np.ndarray,
    legal_idx: list[int],
    illegal_idx: list[int],
    num_qubits: int = NUM_QUBITS,
) -> np.ndarray:
    """Solve a continuous L1 relaxation for the don't-care phases.

    For a fixed set of legal phases, the Walsh coefficients are
    *affine* in the don't-care phases:

        c_S = const_S + sum_i B[S, i] * theta_illegal[i],

    where ``B[S, i] = (-1)^{popcount(S AND illegal_idx[i])} / 2**n``.
    Minimizing the L1 norm of ``(c_S)_{S != 0}`` is the standard convex
    relaxation of "minimize the number of nonzero Walsh coefficients";
    it's a small linear program (91 vars, 126 constraints for n=6) and
    is solved exactly with HiGHS via ``scipy.optimize.linprog``.

    Returns a length-28 array of continuous don't-care phases that
    minimize ``||c||_1``.  Project these to the discrete alphabet (or
    use as continuous values) before feeding into synthesis.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - scipy is in requirements
        raise RuntimeError(
            "lp_relax_walsh_l1 requires scipy; install scipy to use it."
        ) from exc

    N = 1 << num_qubits
    n_illegal = len(illegal_idx)

    # Constant Walsh contribution from the legal entries (don't-care = 0).
    legal_full = np.zeros(N, dtype=float)
    for k, ph in zip(legal_idx, legal_phases):
        legal_full[k] = float(ph)
    h_const = _walsh_full_no_threshold(legal_full, num_qubits=num_qubits)

    # B[S, i] = (-1)^popcount(S AND illegal_idx[i]) / N.
    inv_N = 1.0 / float(N)
    masks = np.asarray(illegal_idx, dtype=np.int64)
    S_arr = np.arange(N, dtype=np.int64)[:, None]
    parity = np.bitwise_count(S_arr & masks[None, :]) if hasattr(np, "bitwise_count") else None
    if parity is None:  # numpy < 2.0 fallback
        parity = np.zeros((N, n_illegal), dtype=np.int64)
        for S in range(N):
            for i in range(n_illegal):
                parity[S, i] = bin(S & illegal_idx[i]).count("1")
    B = inv_N * (1 - 2 * (parity & 1).astype(float))

    # Variables: theta (n_illegal) + t (N - 1, one per S != 0).
    n_vars = n_illegal + (N - 1)
    c_obj = np.zeros(n_vars)
    c_obj[n_illegal:] = 1.0

    # Inequality constraints: B[S,:] theta - t_S <= -h_const[S]
    #                        -B[S,:] theta - t_S <=  h_const[S]
    n_ineq = 2 * (N - 1)
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    row = 0
    for S in range(1, N):
        col = n_illegal + (S - 1)
        A_ub[row, :n_illegal] = B[S, :]
        A_ub[row, col] = -1.0
        b_ub[row] = -h_const[S]
        row += 1
        A_ub[row, :n_illegal] = -B[S, :]
        A_ub[row, col] = -1.0
        b_ub[row] = h_const[S]
        row += 1

    bounds = [(-np.pi, np.pi)] * n_illegal + [(0.0, None)] * (N - 1)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:  # pragma: no cover - scipy should always succeed here
        raise RuntimeError(f"LP relaxation failed: {res.message}")

    return np.asarray(res.x[:n_illegal], dtype=float)


def _project_to_alphabet(
    values: Sequence[float],
    alphabet: Sequence[float],
) -> list[float]:
    """Project each value to the nearest alphabet entry (mod 2*pi distance)."""
    out: list[float] = []
    alpha_arr = np.asarray(alphabet, dtype=float)
    for v in values:
        # Use circular distance.
        d = np.abs(((alpha_arr - v + np.pi) % (2.0 * np.pi)) - np.pi)
        out.append(float(alpha_arr[int(np.argmin(d))]))
    return out


def transpile_metric(
    theta64,
    target: str = "depth",
    via: str = "diagonal_gate",
    basis_gates: Optional[Sequence[str]] = None,
    optimization_level: int = 3,
    seed_transpiler: int = 0,
) -> float:
    """Build a circuit for ``theta64`` and return a transpiled metric.

    Parameters
    ----------
    target : {"depth", "cx", "weighted", "cx_then_depth"}
        - "depth": ``tqc.depth()``
        - "cx":    CNOT count
        - "weighted":   ``cx + 0.1 * depth``
        - "cx_then_depth":  ``1000 * cx + depth`` (lex order)
    via : {"diagonal_gate", "phase_gadget"}
        Which synthesis to feed into ``transpile``.
    basis_gates : iterable of str, optional
        Defaults to ``["rz", "sx", "x", "cx"]``.
    optimization_level : int
        Forwarded to ``transpile``.
    seed_transpiler : int
        Forwarded to ``transpile``.
    """
    if basis_gates is None:
        basis_gates = ["rz", "sx", "x", "cx"]

    diag64 = np.exp(1j * np.asarray(theta64, dtype=float))
    if via == "diagonal_gate":
        qc = QuantumCircuit(NUM_QUBITS)
        qc.append(DiagonalGate(list(diag64)), range(NUM_QUBITS))
    elif via == "phase_gadget":
        qc = synthesize_phase_gadgets(theta64)
    else:
        raise ValueError("via must be 'diagonal_gate' or 'phase_gadget'")

    tqc = transpile(
        qc,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    ops = dict(tqc.count_ops())
    cx = int(ops.get("cx", 0))
    depth = int(tqc.depth())
    if target == "depth":
        return float(depth)
    if target == "cx":
        return float(cx)
    if target == "weighted":
        return float(cx) + 0.1 * float(depth)
    if target == "cx_then_depth":
        return 1000.0 * cx + depth
    raise ValueError(f"unknown target {target!r}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def extract_diagonal_from_circuit(
    qc: QuantumCircuit,
    atol: float = 1e-9,
) -> np.ndarray:
    """Return the diagonal of ``qc`` as a 1-D array.

    Raises ``ValueError`` if the unitary of ``qc`` is not diagonal up to
    tolerance ``atol``.  This is a fully classical check that simulates
    the unitary of the (typically small) circuit.
    """
    U = Operator(qc).data
    diag = np.diag(U).copy()
    off = np.abs(U - np.diag(diag))
    np.fill_diagonal(off, 0.0)
    max_off = float(np.max(off))
    if max_off > atol:
        raise ValueError(
            f"Circuit is not diagonal (max off-diagonal entry = {max_off:.3e})"
        )
    return diag


def verify_on_code_space(
    qc: QuantumCircuit,
    diag36,
    order: str = "row-major",
    atol: float = 1e-8,
) -> float:
    """Return ``max_error_on_code_space`` between ``qc`` and the target.

    The circuit is allowed to differ on the 28 don't-care indices and is
    compared up to a global phase on the 36 legal indices.

    Returns
    -------
    float
        The largest absolute deviation between ``lambda * D_circuit[k]``
        and ``D_target[k]`` over the 36 legal indices, where ``lambda``
        is the global-phase fix obtained from one reference index.  Use
        the value to assert ``< atol`` in tests.
    """
    target_complex = _diag36_complex(diag36)
    if order == "col-major":
        target_complex = target_complex.reshape(QUDIT_DIM, QUDIT_DIM).T.reshape(-1)
    elif order != "row-major":
        raise ValueError("order must be 'row-major' or 'col-major'")

    legal = legal_indices()
    diag64 = extract_diagonal_from_circuit(qc, atol=max(1e-6, atol))

    # Reorder: walk legal indices in the same (a, b) order as target.
    extracted = np.array(
        [diag64[pair_to_qubit_index(a, b)]
         for a in range(QUDIT_DIM)
         for b in range(QUDIT_DIM)],
        dtype=complex,
    )

    if extracted.shape != target_complex.shape:
        raise AssertionError("internal: extracted shape mismatch")

    # Global-phase fix using the largest-magnitude target entry.
    abs_t = np.abs(target_complex)
    idx_ref = int(np.argmax(abs_t))
    if abs(extracted[idx_ref]) < 1e-12:
        return float("inf")
    lam = target_complex[idx_ref] / extracted[idx_ref]

    err = float(np.max(np.abs(target_complex - lam * extracted)))
    return err


# ---------------------------------------------------------------------------
# Baselines and the main entry point
# ---------------------------------------------------------------------------

def _baseline_diagonal_circuit(
    theta64: np.ndarray,
    basis_gates: Sequence[str],
    optimization_level: int,
    seed_transpiler: int,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Build and transpile a baseline ``DiagonalGate`` realization."""
    diag64 = np.exp(1j * np.asarray(theta64, dtype=float))
    qc = QuantumCircuit(NUM_QUBITS, name="diagonal_baseline")
    qc.append(DiagonalGate(list(diag64)), range(NUM_QUBITS))
    tqc = transpile(
        qc,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    return qc, tqc


def _circuit_metrics(qc: QuantumCircuit) -> dict:
    ops = dict(qc.count_ops())
    return {
        "depth": int(qc.depth()),
        "cx": int(ops.get("cx", 0)),
        "rz": int(ops.get("rz", 0)),
        "ops": ops,
    }


def optimize_d6_diagonal(
    diag36,
    phase_alphabet: object = "default",
    max_iters: int = 4000,
    seed: int = 12345,
    qiskit_basis: Optional[Sequence[str]] = None,
    optimization_level: int = 3,
    order: str = "row-major",
    method: str = "best_of",
    cost_weights: Optional[dict] = None,
    tol: float = 1e-10,
    verbose: bool = False,
    n_restarts: int = 6,
    polish_target: Optional[str] = "cx_then_depth",
    polish_via: str = "diagonal_gate",
    polish_max_passes: int = 6,
) -> dict:
    """Optimize the don't-care phases and synthesize a ``CX``+``RZ`` circuit.

    Parameters
    ----------
    diag36 : array-like, length 36
        The 36 phases / unit-modulus complex numbers of D[Lambda] on the
        qudit code space.  See :func:`_phases_from_diag36` for the
        auto-detection rules.
    phase_alphabet : {"default", None} or iterable of float
        Alphabet from which to draw don't-care phases.  ``"default"``
        is ``[0, 2 pi / 3, -2 pi / 3]`` (matches omega_3 inputs).
        ``None`` reuses the unique phases that occur in the legal input.
    max_iters : int
        Total optimizer budget; split equally between random search and
        simulated annealing when ``method = "both"``.
    seed : int
        Master seed; the sub-optimizers use ``seed`` and ``seed + 1``.
    qiskit_basis : iterable of str, optional
        Basis gates for the post-synthesis Qiskit ``transpile`` call.
        Defaults to ``["rz", "sx", "x", "cx"]``.
    optimization_level : int
        Forwarded to :func:`qiskit.transpile`.
    order : {"row-major", "col-major"}
        Layout of ``diag36``.  Default is row-major, which matches the
        AME(4,6) example reproduced in this module's ``__main__`` block.
    method : {"random", "sa", "greedy", "both", "best_of"}
        Which optimizer(s) to run before the optional polish.

        * ``"random"`` -- pure random search.
        * ``"sa"``     -- multi-restart simulated annealing.
        * ``"greedy"`` -- multi-restart greedy coordinate descent on the
          Walsh cost.
        * ``"both"``   -- random search + multi-restart simulated
          annealing (legacy).
        * ``"best_of"`` (default) -- random search + multi-restart SA +
          multi-restart greedy, picking the best-by-cost configuration.
    cost_weights : dict, optional
        Weights for the cost components; see
        :func:`phase_polynomial_cost`.
    tol : float
        Numerical tolerance used in ``walsh_coefficients`` and the cost.
    verbose : bool
        Print progress / final metrics if True.
    n_restarts : int
        Number of random restarts for both the SA stage and the greedy
        coordinate-descent stage (when active).
    polish_target : {None, "depth", "cx", "weighted", "cx_then_depth"}
        If not None, runs a final greedy coordinate descent on the
        actual transpiled metric (``DiagonalGate`` baseline) to squeeze
        out the last few CX/depth.  This is slower (one ``transpile``
        per evaluation) but optimizes the metric the user actually
        cares about.  Default ``"cx_then_depth"`` minimizes CX first,
        then depth as tie-breaker.
    polish_via : {"diagonal_gate", "phase_gadget"}
        Which synthesis to feed into ``transpile`` during the polish.
    polish_max_passes : int
        Max coordinate-descent passes during polish.

    Returns
    -------
    dict
        See module docstring; all keys are listed below for clarity.
    """
    if qiskit_basis is None:
        qiskit_basis = ["rz", "sx", "x", "cx"]
    if method not in {"random", "sa", "greedy", "lp", "both", "best_of"}:
        raise ValueError(
            "method must be 'random', 'sa', 'greedy', 'lp', 'both', or 'best_of'"
        )

    legal_idx = legal_indices()
    illegal_idx = illegal_indices()

    # legal_phases follows the canonical (a, b) walk; we will index it by
    # the same walk used in _build_theta64 below.
    target_phases = _phases_from_diag36(diag36)
    if order == "col-major":
        target_phases = (
            target_phases.reshape(QUDIT_DIM, QUDIT_DIM).T.reshape(-1)
        )
    elif order != "row-major":
        raise ValueError("order must be 'row-major' or 'col-major'")

    # legal_phases[i] is the phase of state |a, b> where (a, b) is the i-th
    # pair under the (a, b) walk a in 0..5, b in 0..5; legal_idx is sorted
    # ascending.  These two orderings match because (a << 3) | b is
    # monotone in (a, b) under the lexicographic walk on {0..5}x{0..5}.
    legal_phases = target_phases.copy()

    if verbose:
        print(
            f"[optimize_d6_diagonal] legal indices: {len(legal_idx)} "
            f"of {NUM_TOTAL_STATES} (illegal: {len(illegal_idx)})"
        )

    alphabet = _resolve_alphabet(phase_alphabet, legal_phases)
    if verbose:
        print(f"[optimize_d6_diagonal] phase alphabet: {alphabet}")
        print(f"[optimize_d6_diagonal] running method='{method}', "
              f"max_iters={max_iters}, seed={seed}")

    # Baseline cost: zeros on illegal indices.
    baseline_phases = [0.0] * len(illegal_idx)
    baseline_theta = normalize_phase(
        _build_theta64(legal_phases, legal_idx, illegal_idx, baseline_phases)
    )
    baseline_cost = phase_polynomial_cost(
        baseline_theta, tol=tol, weights=cost_weights
    )

    best_phases = baseline_phases
    best_cost = baseline_cost

    # Each stage gets the full ``max_iters`` budget; "best_of" runs
    # several stages in sequence, but they don't share/divide the
    # budget, so adding stages only makes the search stronger (and
    # slower), never weaker.
    if method == "random":
        stages = ("rs",)
    elif method == "sa":
        stages = ("sa",)
    elif method == "greedy":
        stages = ("greedy",)
    elif method == "lp":
        stages = ("lp",)
    elif method == "both":
        stages = ("rs", "sa")
    else:  # "best_of"
        stages = ("rs", "sa", "lp", "greedy")

    def walsh_eval(theta64: np.ndarray) -> float:
        return phase_polynomial_cost(
            normalize_phase(theta64), tol=tol, weights=cost_weights
        ).cost

    def maybe_take(label: str, phases: list[float]):
        """Re-evaluate ``phases`` against the structured cost; keep if better."""
        nonlocal best_phases, best_cost
        theta = normalize_phase(
            _build_theta64(legal_phases, legal_idx, illegal_idx, phases)
        )
        cb = phase_polynomial_cost(theta, tol=tol, weights=cost_weights)
        if verbose:
            print(
                f"[optimize_d6_diagonal] {label:<28} cost = {cb.cost:.3f} "
                f"(cx_est={cb.cx_estimate}, terms={cb.num_nonzero_terms}, "
                f"max_supp={cb.max_support})"
            )
        if cb.cost < best_cost.cost - 1e-12:
            best_phases = list(phases)
            best_cost = cb
            return True
        return False

    if "rs" in stages:
        rs_phases, _ = random_search(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            max_iters=max_iters,
            seed=seed,
            cost_weights=cost_weights,
        )
        maybe_take("random_search", rs_phases)

    if "sa" in stages:
        per_run = max(1, max_iters // n_restarts)
        sa_phases, _ = multi_restart_simulated_annealing(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            max_iters_per_run=per_run,
            n_restarts=n_restarts,
            seed=seed + 1,
            cost_weights=cost_weights,
            init_phases=best_phases,
        )
        maybe_take(f"sa(x{n_restarts})", sa_phases)

    if "lp" in stages:
        # Continuous L1 relaxation of "minimize number of nonzero Walsh
        # coefficients", then projected to the discrete alphabet.  This
        # is what most reliably escapes the local minima random
        # search / single-flip SA get stuck in.
        try:
            cont = lp_relax_walsh_l1(
                legal_phases=legal_phases,
                legal_idx=legal_idx,
                illegal_idx=illegal_idx,
            )
            lp_phases = _project_to_alphabet(cont, alphabet)
            maybe_take("lp(continuous->alphabet)", lp_phases)
            # Polish the rounded LP point with 1-opt greedy.
            lp_polished, _ = greedy_coordinate_descent(
                legal_phases=legal_phases,
                legal_idx=legal_idx,
                illegal_idx=illegal_idx,
                alphabet=alphabet,
                init_phases=lp_phases,
                eval_cost=walsh_eval,
                max_passes=12,
            )
            maybe_take("lp + greedy", lp_polished)
        except RuntimeError as exc:
            if verbose:
                print(f"[optimize_d6_diagonal] lp stage skipped: {exc}")

    if "greedy" in stages:
        gr_phases, _ = multi_restart_greedy(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            eval_cost=walsh_eval,
            n_restarts=n_restarts,
            seed=seed + 2,
            init_phases=best_phases,
            max_passes=12,
        )
        maybe_take(f"greedy(x{n_restarts})", gr_phases)

        # Pairwise (2-opt) local search from the best so far.  Cheap on
        # the Walsh cost (~few thousand evaluations per pass) and often
        # uncovers swaps the 1-opt greedy can't see.
        two_opt_phases, _ = two_opt_descent(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            init_phases=best_phases,
            eval_cost=walsh_eval,
            max_passes=3,
        )
        maybe_take("two_opt(walsh)", two_opt_phases)

    # Optional polish: greedy descent on the *actual* transpiled metric.
    polished = False
    if polish_target is not None:
        def polish_eval(theta64: np.ndarray) -> float:
            return transpile_metric(
                theta64,
                target=polish_target,
                via=polish_via,
                basis_gates=qiskit_basis,
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )

        # Polish from the current best; one pass is usually sufficient.
        polished_phases, polished_cost = greedy_coordinate_descent(
            legal_phases=legal_phases,
            legal_idx=legal_idx,
            illegal_idx=illegal_idx,
            alphabet=alphabet,
            init_phases=best_phases,
            eval_cost=polish_eval,
            max_passes=polish_max_passes,
        )
        # Compare to the unpolished best on the same transpile metric.
        prepol_theta = normalize_phase(
            _build_theta64(legal_phases, legal_idx, illegal_idx, best_phases)
        )
        prepol_cost = polish_eval(prepol_theta)
        if polished_cost < prepol_cost - 1e-12:
            best_phases = polished_phases
            polished = True
        if verbose:
            print(
                f"[optimize_d6_diagonal] polish_target={polish_target!r}: "
                f"before={prepol_cost:.3f} after={polished_cost:.3f} "
                f"({'kept' if polished else 'no improvement'})"
            )
        # Refresh the structured CostBreakdown for the (possibly polished)
        # configuration so the returned dict is consistent.
        final_theta = normalize_phase(
            _build_theta64(legal_phases, legal_idx, illegal_idx, best_phases)
        )
        best_cost = phase_polynomial_cost(
            final_theta, tol=tol, weights=cost_weights
        )

    # Build best_theta64 and synthesize.
    best_theta64 = normalize_phase(
        _build_theta64(legal_phases, legal_idx, illegal_idx, best_phases)
    )
    best_diag64 = np.exp(1j * best_theta64)
    coeffs = walsh_coefficients(best_theta64, tol=tol)
    cost_info = phase_polynomial_cost(best_theta64, tol=tol, weights=cost_weights)

    qc = synthesize_phase_gadgets(best_theta64, tol=tol)
    tqc = transpile(
        qc,
        basis_gates=list(qiskit_basis),
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    baseline_qc, baseline_tqc = _baseline_diagonal_circuit(
        best_theta64,
        basis_gates=qiskit_basis,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    pg_metrics = _circuit_metrics(qc)
    tqc_metrics = _circuit_metrics(tqc)
    baseline_metrics = _circuit_metrics(baseline_tqc)

    if verbose:
        print(
            f"[optimize_d6_diagonal] phase-gadget circuit: "
            f"cx={pg_metrics['cx']}, rz={pg_metrics['rz']}, "
            f"depth={pg_metrics['depth']}"
        )
        print(
            f"[optimize_d6_diagonal] transpiled circuit:    "
            f"cx={tqc_metrics['cx']}, depth={tqc_metrics['depth']}"
        )
        print(
            f"[optimize_d6_diagonal] DiagonalGate baseline: "
            f"cx={baseline_metrics['cx']}, depth={baseline_metrics['depth']}"
        )

    return {
        "best_theta64": best_theta64,
        "best_diag64": best_diag64,
        "legal_indices": legal_idx,
        "illegal_indices": illegal_idx,
        "walsh_coeffs": coeffs,
        "num_nonzero_terms": cost_info.num_nonzero_terms,
        "cx_estimate": cost_info.cx_estimate,
        "max_support": cost_info.max_support,
        "alphabet": alphabet,
        "best_illegal_phases": list(best_phases),
        "method": method,
        "polish_target": polish_target,
        "polished": polished if polish_target is not None else False,
        "phase_gadget_circuit": qc,
        "transpiled_circuit": tqc,
        "baseline_circuit": baseline_tqc,
        "metrics": {
            "phase_gadget_depth": pg_metrics["depth"],
            "phase_gadget_cx": pg_metrics["cx"],
            "phase_gadget_rz": pg_metrics["rz"],
            "transpiled_depth": tqc_metrics["depth"],
            "transpiled_cx": tqc_metrics["cx"],
            "transpiled_rz": tqc_metrics["rz"],
            "baseline_depth": baseline_metrics["depth"],
            "baseline_cx": baseline_metrics["cx"],
            "baseline_rz": baseline_metrics["rz"],
            "phase_gadget_ops": pg_metrics["ops"],
            "transpiled_ops": tqc_metrics["ops"],
            "baseline_ops": baseline_metrics["ops"],
        },
    }


# ---------------------------------------------------------------------------
# Tests / sanity checks (run on import as part of __main__)
# ---------------------------------------------------------------------------

def _self_tests() -> None:
    """Lightweight assertions covering the API surface."""
    # Encoding and index helpers.
    assert pair_to_qubit_index(0, 0) == 0b000_000
    assert pair_to_qubit_index(5, 5) == 0b101_101
    assert pair_to_qubit_index(2, 3) == (0b010 << 3) | 0b011 == 0b010_011
    assert qubit_index_to_pair(pair_to_qubit_index(4, 1)) == (4, 1)

    legal = legal_indices()
    illegal = illegal_indices()
    assert len(legal) == 36
    assert len(illegal) == 28
    assert sorted(legal + illegal) == list(range(64))
    assert set(legal).isdisjoint(set(illegal))

    # Sanity: the illegal set is exactly indices where one of the qudit
    # halves has top bit set (>=110) but isn't a legal level.  All
    # legal pairs (a, b) have a < 6 and b < 6.
    for k in legal:
        a, b = qubit_index_to_pair(k)
        assert 0 <= a < 6 and 0 <= b < 6, (k, a, b)
    for k in illegal:
        a, b = qubit_index_to_pair(k)
        assert (a >= 6) or (b >= 6), (k, a, b)

    # Walsh transform self-consistency: round-trip of a random phase
    # vector recovers it exactly.
    rng = np.random.default_rng(0)
    theta = rng.normal(size=64)
    coeffs = walsh_coefficients(theta, tol=0.0)
    recon = np.zeros(64)
    for m, c in coeffs.items():
        for x in range(64):
            sign = 1 - 2 * (bin(m & x).count("1") & 1)
            recon[x] += c * sign
    assert np.allclose(recon, theta, atol=1e-10)

    # Phase-gadget synthesis matches DiagonalGate up to global phase
    # for a small random diagonal.
    theta_small = rng.normal(size=64) * 0.7
    diag = np.exp(1j * theta_small)
    qc_pg = synthesize_phase_gadgets(theta_small)
    qc_dg = QuantumCircuit(6)
    qc_dg.append(DiagonalGate(list(diag)), range(6))
    U_pg = Operator(qc_pg).data
    U_dg = Operator(qc_dg).data
    # Compare on the diagonal (both are diagonal); allow a single global
    # phase factor.
    d_pg = np.diag(U_pg)
    d_dg = np.diag(U_dg)
    lam = d_dg[0] / d_pg[0]
    assert np.allclose(d_dg, lam * d_pg, atol=1e-9), (
        "phase-gadget synthesis disagrees with DiagonalGate"
    )

    # extract_diagonal_from_circuit + verify_on_code_space should yield
    # ~0 error on a circuit built from the same diag36.
    diag36_test = np.exp(1j * rng.normal(size=NUM_LEGAL_STATES))
    theta64_test = diag36_to_theta64(diag36_test)
    qc_test = synthesize_phase_gadgets(theta64_test)
    err = verify_on_code_space(qc_test, diag36_test)
    assert err < 1e-7, f"verify_on_code_space gave err={err:.3e}"


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def _example_d_lambda_2_3() -> np.ndarray:
    """Return the 36-entry D[Lambda_{2,3}] vector from the AME(4,6) paper.

    Convention used here: ``order = "row-major"``, i.e. the entry at
    position ``6 * a + b`` is the phase of state ``|a, b>``.  This is
    the natural reading order of the paper's tabular display.
    """
    omega3 = np.exp(2j * np.pi / 3.0)
    omega3_bar = np.conj(omega3)
    return np.array(
        [
            1, omega3, omega3_bar, omega3, omega3_bar, 1,
            1, 1, omega3, 1, omega3_bar, 1,
            1, omega3, 1, omega3_bar, 1, 1,
            1, omega3_bar, 1, omega3, omega3, omega3,
            omega3_bar, omega3_bar, omega3_bar, omega3_bar, 1, 1,
            omega3, omega3, omega3_bar, 1, 1, omega3_bar,
        ],
        dtype=complex,
    )


def _print_metrics(name: str, m: dict, prefix: str = "") -> None:
    print(
        f"{prefix}{name:<24} "
        f"cx={m['cx']:>3} rz={m['rz']:>3} depth={m['depth']:>3} "
        f"ops={dict(m.get('ops', {}))}"
    )


def main() -> None:
    print("=" * 72)
    print("d=6 diagonal optimizer: AME(4,6) example D[Lambda_{2,3}]")
    print("=" * 72)
    _self_tests()
    print("[ok] self-tests passed")

    diag36 = _example_d_lambda_2_3()
    print(f"diag36 length     : {len(diag36)}")
    unique_phases = sorted({round(float(p), 6) for p in np.angle(diag36)})
    print(f"unique phases     : {unique_phases}")

    print()
    print("--- optimize with default alphabet [0, 2pi/3, -2pi/3] ---")
    result = optimize_d6_diagonal(
        diag36,
        phase_alphabet="default",
        max_iters=4000,
        seed=2024,
        order="row-major",
        method="best_of",
        verbose=True,
    )

    print()
    print("--- summary ---")
    print(f"num_nonzero Walsh terms (excl. global phase): "
          f"{result['num_nonzero_terms']}")
    print(f"cx_estimate (phase-gadget naive)            : "
          f"{result['cx_estimate']}")
    print(f"max_support across nonzero coefficients     : "
          f"{result['max_support']}")
    pg_m = {
        "cx": result["metrics"]["phase_gadget_cx"],
        "rz": result["metrics"]["phase_gadget_rz"],
        "depth": result["metrics"]["phase_gadget_depth"],
        "ops": result["metrics"]["phase_gadget_ops"],
    }
    tqc_m = {
        "cx": result["metrics"]["transpiled_cx"],
        "rz": result["metrics"]["transpiled_rz"],
        "depth": result["metrics"]["transpiled_depth"],
        "ops": result["metrics"]["transpiled_ops"],
    }
    base_m = {
        "cx": result["metrics"]["baseline_cx"],
        "rz": result["metrics"]["baseline_rz"],
        "depth": result["metrics"]["baseline_depth"],
        "ops": result["metrics"]["baseline_ops"],
    }
    _print_metrics("phase_gadget_circuit", pg_m)
    _print_metrics("transpiled_circuit  ", tqc_m)
    _print_metrics("baseline_circuit    ", base_m)

    err_pg = verify_on_code_space(
        result["phase_gadget_circuit"], diag36, order="row-major"
    )
    err_t = verify_on_code_space(
        result["transpiled_circuit"], diag36, order="row-major"
    )
    err_b = verify_on_code_space(
        result["baseline_circuit"], diag36, order="row-major"
    )
    print()
    print(f"verify_on_code_space (phase-gadget) : {err_pg:.3e}")
    print(f"verify_on_code_space (transpiled)   : {err_t:.3e}")
    print(f"verify_on_code_space (baseline)     : {err_b:.3e}")

    assert err_pg < 1e-7, f"phase-gadget circuit failed code-space check ({err_pg:.3e})"
    assert err_t < 1e-7, f"transpiled circuit failed code-space check ({err_t:.3e})"
    assert err_b < 1e-7, f"baseline circuit failed code-space check ({err_b:.3e})"

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
