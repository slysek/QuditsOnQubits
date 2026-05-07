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
    "enc3",
    "enc3_bits",
    "pair_to_qubit_index",
    "qubit_index_to_pair",
    "legal_indices",
    "illegal_indices",
    "logical_indices",
    "invalid_indices",
    "build_embedded_diag64",
    "embedded_diag64_matrix",
    "trace_fidelity_64",
    "logical_trace_fidelity_36",
    "leakage_norms",
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
    "build_optimized_circuit",
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
# Explicit two-quhex code-space embedding
# ---------------------------------------------------------------------------
#
# A single quhex (d = 6) is embedded into 3 qubits via the fixed encoding
#
#     |0> -> |000>, |1> -> |001>, |2> -> |010>, |3> -> |011>,
#     |4> -> |100>, |5> -> |101>,
#
# while |110> and |111> are *invalid* (outside the code space).
#
# Two quhexes therefore live in a 36-dim code space embedded into the
# 64-dim (C^2)^6 physical Hilbert space.  The embedding is a *product*
# embedding C^6 (x) C^6 -> (C^2)^3 (x) (C^2)^3, so the 28 invalid
# basis states of (C^2)^6 are NOT the last 28 indices of the basis: they
# are interleaved with the logical states (e.g. indices 6 and 7 are
# invalid and sit between legal indices 5 and 8).  Naively concatenating
# `[lambda36] + [1]*28` to obtain a 64-vector is therefore *incorrect*;
# the helpers below build the embedding explicitly and are covered by
# regression tests in :func:`_self_tests`.
#
# Endianness convention.  We use Qiskit's per-qubit little-endian layout
# throughout: the basis index of a 6-qubit computational state is
#
#     idx = sum_{q=0}^{5} bit_q * 2**q,
#
# matching ``Operator(qc).data``.  With this convention,
# ``pair_to_qubit_index(a, b) = (a << 3) | b`` puts ``b`` on qubits
# 0, 1, 2 (LSB first) and ``a`` on qubits 3, 4, 5.

def enc3(level: int) -> int:
    """Return the 3-bit binary encoding of a single quhex level.

    Mapping (matches the AME(4,6) reference)::

        0 -> 000, 1 -> 001, 2 -> 010, 3 -> 011,
        4 -> 100, 5 -> 101.

    Levels 6 and 7 are *not* in the quhex code space and are rejected.
    The returned integer is in 0..5; combined with :func:`enc3_bits`
    or :func:`pair_to_qubit_index` it gives the physical-basis index.
    """
    if not (0 <= level < QUDIT_DIM):
        raise ValueError(
            f"quhex level must be in 0..{QUDIT_DIM - 1}, got {level}"
        )
    return int(level)


def enc3_bits(level: int) -> tuple[int, int, int]:
    """Return ``(b0, b1, b2)`` of :func:`enc3`, with ``b0`` the LSB.

    For example::

        enc3_bits(5) == (1, 0, 1)   # because 5 = 0b101 (b2 b1 b0)
    """
    v = enc3(level)
    return (v & 1, (v >> 1) & 1, (v >> 2) & 1)


def logical_indices() -> list[int]:
    """Return the sorted list of 36 physical-basis indices in the code space.

    Alias of :func:`legal_indices` provided for clarity in the
    "logical vs. invalid" terminology used by the embedding helpers.
    """
    return legal_indices()


def invalid_indices() -> list[int]:
    """Return the sorted list of 28 physical-basis indices NOT in the code space.

    Alias of :func:`illegal_indices`.
    """
    return illegal_indices()


def build_embedded_diag64(
    lambda36,
    endian: str = "qiskit",
    order: str = "row-major",
) -> np.ndarray:
    """Embed a 36-dim diagonal D[Lambda] into the 64-dim two-quhex space.

    Implements the product embedding

        C^6 (x) C^6  -->  (C^2)^3 (x) (C^2)^3

    by writing ``lambda36[a, b]`` at the physical-basis index of the
    logical state ``enc(a) (x) enc(b)`` and leaving every invalid index
    at ``+1`` (zero phase).  This is the *correct* target for the
    diagonal gate D[Lambda_{2,3}] from the AME(4,6) construction; using
    ``np.concatenate([lambda36, np.ones(28)])`` instead would put the
    36 logical phases on indices ``0..35`` of the 64-vector, which is
    wrong because invalid indices 6, 7, 14, 15, ... are interleaved
    between logical ones.

    Parameters
    ----------
    lambda36 : array-like, length 36
        Diagonal entries of D[Lambda].  May be unit-modulus complex
        numbers (e.g. roots of unity) or real phase angles in radians;
        :func:`_phases_from_diag36` handles both.
    endian : {"qiskit"}
        Endianness convention; only Qiskit's per-qubit little-endian
        layout is supported.  Calling with any other string raises.
    order : {"row-major", "col-major"}
        Layout of ``lambda36``.  ``"row-major"`` (default) means
        ``lambda36[6 * a + b]`` is the entry for state ``|a, b>``.

    Returns
    -------
    np.ndarray, shape (64,), dtype complex
        The 64-dim diagonal vector ``diag64`` such that
        ``diag64[k] = lambda36[a, b]`` on logical indices and
        ``diag64[k] = 1`` on invalid indices.
    """
    if endian != "qiskit":
        raise ValueError(
            f"only endian='qiskit' is supported, got endian={endian!r}"
        )
    if order not in ("row-major", "col-major"):
        raise ValueError("order must be 'row-major' or 'col-major'")

    lam = _diag36_complex(lambda36)

    diag64 = np.ones(NUM_TOTAL_STATES, dtype=complex)
    for a in range(QUDIT_DIM):
        for b in range(QUDIT_DIM):
            src = a * QUDIT_DIM + b if order == "row-major" else a + QUDIT_DIM * b
            dst = pair_to_qubit_index(a, b)
            diag64[dst] = lam[src]
    return diag64


def embedded_diag64_matrix(
    lambda36,
    endian: str = "qiskit",
    order: str = "row-major",
) -> np.ndarray:
    """Same as :func:`build_embedded_diag64` but returns a 64x64 matrix."""
    return np.diag(build_embedded_diag64(lambda36, endian=endian, order=order))


def trace_fidelity_64(
    U_circuit: np.ndarray,
    D64_embedded,
) -> float:
    """Trace fidelity ``F = |Tr(U^dagger D)| / 64`` on the full 64-dim space.

    ``D64_embedded`` may be either the 64-vector returned by
    :func:`build_embedded_diag64` or the corresponding 64x64 diagonal
    matrix.  The fidelity is invariant under a global phase on
    ``U_circuit`` and is ``1`` iff ``U_circuit`` matches the embedded
    target up to a global phase.

    Note: the don't-care phases on the 28 invalid indices are part of
    this fidelity.  When the optimizer is free to pick those phases to
    reduce gate count, the full-64 fidelity is generally < 1 even
    though the logical-block fidelity is exactly 1.  In that regime use
    :func:`logical_trace_fidelity_36` for the strict correctness check.
    """
    D = np.asarray(D64_embedded)
    if D.ndim == 1:
        if D.shape != (NUM_TOTAL_STATES,):
            raise ValueError(
                f"D64 vector must have length {NUM_TOTAL_STATES}, got {D.shape}"
            )
        D_mat = np.diag(D)
    elif D.ndim == 2:
        if D.shape != (NUM_TOTAL_STATES, NUM_TOTAL_STATES):
            raise ValueError(
                f"D64 matrix must be {NUM_TOTAL_STATES}x{NUM_TOTAL_STATES}, "
                f"got {D.shape}"
            )
        D_mat = D
    else:
        raise ValueError(f"D64 must be 1-D or 2-D, got ndim={D.ndim}")

    if U_circuit.shape != (NUM_TOTAL_STATES, NUM_TOTAL_STATES):
        raise ValueError(
            f"U_circuit must be {NUM_TOTAL_STATES}x{NUM_TOTAL_STATES}, "
            f"got {U_circuit.shape}"
        )

    return float(np.abs(np.trace(U_circuit.conj().T @ D_mat)) / NUM_TOTAL_STATES)


def logical_trace_fidelity_36(
    U_circuit: np.ndarray,
    lambda36,
    order: str = "row-major",
) -> float:
    """Trace fidelity on the 36-dim quhex code space.

    Restricts ``U_circuit`` to the 36 logical indices and compares with
    ``diag(lambda36)``::

        F = |Tr(U_logical^dagger * diag(lambda36))| / 36.

    ``F = 1`` iff the synthesized circuit matches D[Lambda] on the code
    space up to a global phase.  This is the cost-function variant
    *option 2* in the spec (item 7); use it for the strict correctness
    check whenever the optimizer is free to set don't-care phases on
    the 28 invalid indices.
    """
    if U_circuit.shape != (NUM_TOTAL_STATES, NUM_TOTAL_STATES):
        raise ValueError(
            f"U_circuit must be {NUM_TOTAL_STATES}x{NUM_TOTAL_STATES}, "
            f"got {U_circuit.shape}"
        )
    log_idx = np.asarray(logical_indices(), dtype=int)
    U_log = U_circuit[np.ix_(log_idx, log_idx)]

    lam = _diag36_complex(lambda36)
    if order == "col-major":
        lam = lam.reshape(QUDIT_DIM, QUDIT_DIM).T.reshape(-1)
    elif order != "row-major":
        raise ValueError("order must be 'row-major' or 'col-major'")

    # logical_indices is sorted ascending and pair_to_qubit_index(a, b) =
    # (a << 3) | b is monotone in (a, b) under the lex walk a outer / b
    # inner, so U_log is already arranged in row-major (a, b) order
    # consistent with diag(lam).
    return float(np.abs(np.vdot(np.diag(U_log), lam)) / NUM_LEGAL_STATES)


def leakage_norms(U_circuit: np.ndarray) -> tuple[float, float]:
    """Return Frobenius norms of the off-block ``U_circuit`` couplings.

    ``leakage_out = ||U_circuit[invalid_indices, logical_indices]||_F``
    is the amplitude leaving the code space; ``leakage_in`` is the
    amplitude that ends *inside* the code space starting from invalid
    states.  For a correct realization of D[Lambda_{2,3}] (which is
    diagonal in the computational basis) both should be ~0.
    """
    if U_circuit.shape != (NUM_TOTAL_STATES, NUM_TOTAL_STATES):
        raise ValueError(
            f"U_circuit must be {NUM_TOTAL_STATES}x{NUM_TOTAL_STATES}, "
            f"got {U_circuit.shape}"
        )
    log_idx = np.asarray(logical_indices(), dtype=int)
    inv_idx = np.asarray(invalid_indices(), dtype=int)
    out_block = U_circuit[np.ix_(inv_idx, log_idx)]
    in_block = U_circuit[np.ix_(log_idx, inv_idx)]
    return float(np.linalg.norm(out_block)), float(np.linalg.norm(in_block))


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

    This is the *real-phase* twin of :func:`build_embedded_diag64`:
    instead of returning ``exp(i theta)`` directly, it returns the
    ``theta`` vector so the caller can do Walsh / phase-gadget
    arithmetic on it.  The placement is identical to
    :func:`build_embedded_diag64` -- the 36 logical phases live at
    ``pair_to_qubit_index(a, b)``, the 28 invalid indices are zero
    (i.e. ``D = +1`` there) unless ``illegal_phases`` is supplied.
    The "naive" recipe ``theta64 = list(theta36) + [0] * 28`` is
    *wrong* -- see the comment block above ``enc3``.

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
    pin_invalid_phases: bool = True,
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
    pin_invalid_phases : bool, default True
        If True (default), the 28 don't-care phases are pinned to 0
        so that ``D = +1`` on every invalid 6-qubit basis state.  The
        synthesized circuit then realizes the *strict* embedded gate
        ``D64_embedded = diag([D36 on logical] + [1 on invalid])`` --
        i.e. acts as identity on the invalid subspace -- and its full
        64-dim diagonal matches :func:`build_embedded_diag64` exactly.
        This is the natural semantics for D[Lambda_{2,3}] from
        AME(4,6).  Internally it overrides ``phase_alphabet`` with
        ``[0.0]`` so every random / SA / greedy / LP stage is a no-op
        and only the deterministic synthesis runs.

        Pass ``pin_invalid_phases=False`` to re-enable don't-care
        freedom: the optimizer is then allowed to assign non-zero
        phases to invalid indices in order to shrink the synthesized
        circuit's CX count.  The logical action stays identical
        (``F36 = 1``, no leakage), but the full-64 diagonal will
        differ from ``D64_embedded`` on the 28 invalid indices.

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

    if pin_invalid_phases:
        phase_alphabet = [0.0]

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
# Convenience wrapper for notebooks / interactive use
# ---------------------------------------------------------------------------

def build_optimized_circuit(
    diag36,
    which: str = "best",
    return_info: bool = False,
    verify: bool = True,
    verify_atol: float = 1e-7,
    order: str = "row-major",
    phase_alphabet: object = "default",
    max_iters: int = 4000,
    seed: int = 2024,
    qiskit_basis: Optional[Sequence[str]] = None,
    optimization_level: int = 3,
    method: str = "best_of",
    n_restarts: int = 6,
    polish_target: Optional[str] = "cx_then_depth",
    polish_via: str = "diagonal_gate",
    polish_max_passes: int = 6,
    cost_weights: Optional[dict] = None,
    tol: float = 1e-10,
    verbose: bool = False,
    pin_invalid_phases: bool = True,
):
    """One-shot helper: optimize the don't-care phases and return a circuit.

    Convenience wrapper around :func:`optimize_d6_diagonal` for use in
    notebooks.  Runs the full optimizer pipeline and returns a single
    ``QuantumCircuit`` ready to use.

    Parameters
    ----------
    diag36 : array-like, length 36
        The 36 phases / unit-modulus complex numbers of D[Lambda] on the
        qudit code space.  Same input format as
        :func:`optimize_d6_diagonal`.
    which : {"best", "baseline", "transpiled", "phase_gadget"}
        Which of the synthesized circuits to return.

        * ``"best"`` (default) -- pick the cheapest of the three by
          ``(cx, depth)`` (lex order).  Usually equals
          ``"baseline"``, but stays robust if the relative cost
          changes for a given input.
        * ``"baseline"`` -- the Qiskit ``DiagonalGate`` synthesis,
          transpiled to ``qiskit_basis``.  Typically the cheapest.
        * ``"transpiled"`` -- the phase-gadget ``CX``+``RZ`` circuit
          after Qiskit ``transpile``.
        * ``"phase_gadget"`` -- the raw, untranspiled ``CX``+``RZ``
          circuit (useful when you want to keep gates in that basis).
    return_info : bool, default False
        If True, return ``(circuit, info)`` where ``info`` is a dict
        with diagnostic data:

            - ``metrics``           : depth / cx / rz of the chosen circuit
            - ``which``             : the variant that was returned
            - ``best_theta64``      : 64-vector of phases used
            - ``best_illegal_phases``: 28 chosen don't-care phases
            - ``num_nonzero_terms`` : nonzero Walsh terms after optimize
            - ``cx_estimate``       : naive CX cost of the phase gadgets
            - ``alphabet``          : alphabet used by the optimizer
            - ``polished``          : whether the transpile polish helped
            - ``verify_error``      : ``verify_on_code_space`` result
            - ``all_metrics``       : metrics for *all* three variants
    verify : bool, default True
        If True, verify that the returned circuit reproduces ``diag36``
        on the code space within ``verify_atol``.  Raises
        ``AssertionError`` on failure.
    verify_atol : float, default 1e-7
        Tolerance for the code-space verification.
    order : {"row-major", "col-major"}
        Layout of ``diag36``.  See :func:`optimize_d6_diagonal`.
    phase_alphabet, max_iters, seed, qiskit_basis, optimization_level,
    method, n_restarts, polish_target, polish_via, polish_max_passes,
    cost_weights, tol, verbose, pin_invalid_phases
        Forwarded to :func:`optimize_d6_diagonal`; see that function for
        the full description.  By default ``pin_invalid_phases=True``
        forces ``D = +1`` on every invalid 6-qubit basis state, so the
        circuit's full 64-dim diagonal matches
        :func:`build_embedded_diag64` exactly (strict embedding).
        Pass ``pin_invalid_phases=False`` to opt into the legacy
        don't-care optimization (smaller CX count, but non-trivial
        phases on the 28 invalid indices).

    Returns
    -------
    QuantumCircuit, or (QuantumCircuit, dict)
        The optimized circuit, or ``(circuit, info)`` if
        ``return_info=True``.

    Examples
    --------
    Minimal usage in a notebook::

        from d6_diagonal_optimizer import build_optimized_circuit
        import numpy as np

        omega3 = np.exp(2j * np.pi / 3)
        diag36 = np.array([1, omega3, ..., 1], dtype=complex)  # length 36

        qc = build_optimized_circuit(diag36)
        qc.draw("mpl")

    To inspect the resource counts as well::

        qc, info = build_optimized_circuit(diag36, return_info=True)
        print(info["metrics"])             # {"depth": 80, "cx": 52, ...}
        print(info["num_nonzero_terms"])   # e.g. 33
        print(f"err = {info['verify_error']:.2e}")

    To get a pure ``CX``+``RZ`` circuit (no transpile)::

        qc = build_optimized_circuit(diag36, which="phase_gadget")
    """
    valid = {"best", "baseline", "transpiled", "phase_gadget"}
    if which not in valid:
        raise ValueError(
            f"which must be one of {sorted(valid)}, got {which!r}"
        )

    result = optimize_d6_diagonal(
        diag36,
        phase_alphabet=phase_alphabet,
        max_iters=max_iters,
        seed=seed,
        qiskit_basis=qiskit_basis,
        optimization_level=optimization_level,
        order=order,
        method=method,
        cost_weights=cost_weights,
        tol=tol,
        verbose=verbose,
        n_restarts=n_restarts,
        polish_target=polish_target,
        polish_via=polish_via,
        polish_max_passes=polish_max_passes,
        pin_invalid_phases=pin_invalid_phases,
    )

    metrics = result["metrics"]
    candidates = {
        "baseline": (
            result["baseline_circuit"],
            {
                "depth": metrics["baseline_depth"],
                "cx": metrics["baseline_cx"],
                "rz": metrics["baseline_rz"],
                "ops": metrics.get("baseline_ops", {}),
            },
        ),
        "transpiled": (
            result["transpiled_circuit"],
            {
                "depth": metrics["transpiled_depth"],
                "cx": metrics["transpiled_cx"],
                "rz": metrics["transpiled_rz"],
                "ops": metrics.get("transpiled_ops", {}),
            },
        ),
        "phase_gadget": (
            result["phase_gadget_circuit"],
            {
                "depth": metrics["phase_gadget_depth"],
                "cx": metrics["phase_gadget_cx"],
                "rz": metrics["phase_gadget_rz"],
                "ops": metrics.get("phase_gadget_ops", {}),
            },
        ),
    }

    if which == "best":
        # Lex order by (cx, depth) so we always pick the cheapest in
        # CX first; ties broken by depth.  Stable: among equal-cost
        # candidates, prefer baseline > transpiled > phase_gadget.
        ranking = ["baseline", "transpiled", "phase_gadget"]
        chosen_key = min(
            ranking,
            key=lambda k: (candidates[k][1]["cx"], candidates[k][1]["depth"]),
        )
    else:
        chosen_key = which

    qc, chosen_metrics = candidates[chosen_key]

    verify_err = float("nan")
    if verify:
        verify_err = verify_on_code_space(qc, diag36, order=order)
        assert verify_err < verify_atol, (
            f"build_optimized_circuit: returned circuit deviates from "
            f"target diag36 by {verify_err:.3e} > atol={verify_atol:.1e} "
            f"(which={chosen_key!r})"
        )

    if verbose:
        print(
            f"[build_optimized_circuit] returning '{chosen_key}' "
            f"cx={chosen_metrics['cx']} depth={chosen_metrics['depth']} "
            f"rz={chosen_metrics['rz']}  walsh_terms={result['num_nonzero_terms']} "
            f"err={verify_err:.2e}"
        )

    if not return_info:
        return qc

    info = {
        "which": chosen_key,
        "metrics": chosen_metrics,
        "all_metrics": {k: v[1] for k, v in candidates.items()},
        "best_theta64": result["best_theta64"],
        "best_illegal_phases": result["best_illegal_phases"],
        "num_nonzero_terms": result["num_nonzero_terms"],
        "cx_estimate": result["cx_estimate"],
        "max_support": result["max_support"],
        "alphabet": result["alphabet"],
        "polished": result["polished"],
        "verify_error": verify_err,
    }
    return qc, info


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

    # ------------------------------------------------------------------
    # Two-quhex code-space embedding tests (item 6 of the spec).
    # ------------------------------------------------------------------
    log_idx = logical_indices()
    inv_idx = invalid_indices()

    # (a) State counts and disjointness.
    assert len(log_idx) == 36, f"expected 36 logical indices, got {len(log_idx)}"
    assert len(inv_idx) == 28, f"expected 28 invalid indices, got {len(inv_idx)}"
    assert set(log_idx).isdisjoint(set(inv_idx)), (
        "logical and invalid indices must be disjoint"
    )
    assert sorted(log_idx + inv_idx) == list(range(NUM_TOTAL_STATES)), (
        "logical | invalid must cover the full 64-dim physical basis"
    )

    # Sanity: invalid indices are NOT all on the high end -- proves the
    # naive `[lambda36] + [1]*28` layout is wrong.
    assert 6 in inv_idx and 7 in inv_idx, (
        "indices 6 and 7 must be invalid (single-quhex |110>, |111> on b)"
    )
    assert max(log_idx) == pair_to_qubit_index(5, 5), (
        f"max logical index should be {pair_to_qubit_index(5, 5)}, "
        f"got {max(log_idx)}"
    )

    # (b) D64_embedded[idx] == 1 for every invalid idx.
    lam36 = np.exp(1j * rng.normal(size=NUM_LEGAL_STATES))
    D64_emb = build_embedded_diag64(lam36)
    assert D64_emb.shape == (NUM_TOTAL_STATES,)
    for k in inv_idx:
        assert np.isclose(D64_emb[k], 1.0 + 0j, atol=1e-12), (
            f"D64[{k}] = {D64_emb[k]} on invalid index, expected 1+0j"
        )

    # (c) Logical 36x36 block of diag(D64_embedded) equals diag(lambda36).
    D64_mat = embedded_diag64_matrix(lam36)
    log_arr = np.asarray(log_idx, dtype=int)
    block = D64_mat[np.ix_(log_arr, log_arr)]
    # logical_indices is sorted and pair_to_qubit_index(a, b) = (a<<3)|b is
    # monotone in (a, b) under the lex walk a outer / b inner, so the
    # block diagonal equals lam36 in row-major order.
    expected36 = _diag36_complex(lam36)
    assert np.allclose(np.diag(block), expected36, atol=1e-12), (
        "logical 36x36 block does not equal diag(lambda36)"
    )
    # All off-diagonal entries of the logical block are zero.
    assert np.max(np.abs(block - np.diag(np.diag(block)))) < 1e-12

    # (d) The naive [lambda36] + [1]*28 layout is NOT the same embedding.
    naive64 = np.concatenate(
        [expected36, np.ones(NUM_ILLEGAL_STATES, dtype=complex)]
    )
    assert naive64.shape == D64_emb.shape
    diff = float(np.max(np.abs(naive64 - D64_emb)))
    # The two arrays disagree at every interior index where naive has a
    # logical phase but the embedding has 1, or vice versa.  For a random
    # phase vector the difference is order 1.
    assert diff > 1e-3, (
        f"naive concatenation accidentally agrees with product embedding "
        f"(max diff = {diff:.3e})"
    )

    # (e) Trace fidelity against the embedded target is ~1 for a
    # circuit synthesized from the same lambda36 with all don't-care
    # phases pinned to 0.  Both the full-64 and logical-36 fidelities
    # should hit 1 here; the logical-36 number stays at 1 even when the
    # don't-care phases are chosen freely (item 8 covers the full-64
    # mismatch case).
    theta64_emb = diag36_to_theta64(lam36)
    qc_emb = synthesize_phase_gadgets(theta64_emb)
    U_emb = Operator(qc_emb).data
    fid64 = trace_fidelity_64(U_emb, D64_emb)
    fid36 = logical_trace_fidelity_36(U_emb, lam36)
    assert fid64 > 1.0 - 1e-9, f"full-64 trace fidelity = {fid64:.6f}, expected ~1"
    assert fid36 > 1.0 - 1e-9, f"logical-36 trace fidelity = {fid36:.6f}, expected ~1"

    # Robustness: setting non-zero don't-care phases must NOT change the
    # logical-36 fidelity (it cancels out on the logical block) but
    # WILL change the full-64 fidelity.  This pins down the spec
    # distinction between "compare against D64_embedded" and "compare
    # only on the logical subspace".
    rand_dont_care = list(rng.normal(size=NUM_ILLEGAL_STATES) * 0.7)
    theta64_dc = diag36_to_theta64(lam36, illegal_phases=rand_dont_care)
    qc_dc = synthesize_phase_gadgets(theta64_dc)
    U_dc = Operator(qc_dc).data
    fid36_dc = logical_trace_fidelity_36(U_dc, lam36)
    fid64_dc = trace_fidelity_64(U_dc, D64_emb)
    assert fid36_dc > 1.0 - 1e-9, (
        f"logical-36 fidelity broke when don't-cares moved: {fid36_dc:.6f}"
    )
    assert fid64_dc < 0.999, (
        "full-64 fidelity should drop when don't-cares are scrambled "
        f"(got {fid64_dc:.6f})"
    )

    # No leakage: the synthesized unitary is diagonal in the
    # computational basis, so logical<->invalid coupling blocks are 0.
    leak_out, leak_in = leakage_norms(U_emb)
    assert leak_out < 1e-10, f"leakage_out = {leak_out:.3e}"
    assert leak_in < 1e-10, f"leakage_in  = {leak_in:.3e}"

    # Cross-check: the diagonal of U_emb agrees with build_embedded_diag64
    # entry-by-entry up to a global phase on the logical block.  This
    # locks down the convention "Operator(qc).data uses Qiskit's
    # little-endian bit ordering" against build_embedded_diag64 / pair_to_qubit_index.
    diag_full = np.diag(U_emb)
    lam_ref = D64_emb[log_arr][0] / diag_full[log_arr][0]
    assert np.allclose(
        diag_full[log_arr] * lam_ref, D64_emb[log_arr], atol=1e-9
    ), "logical block of synthesized circuit disagrees with embedded target"

    # ------------------------------------------------------------------
    # pin_invalid_phases regression test.
    # When the flag is True, the 28 don't-care indices must remain at
    # phase 0 (i.e. diag = 1+0j) throughout the optimizer pipeline.
    # ------------------------------------------------------------------
    pinned = optimize_d6_diagonal(
        lam36,
        phase_alphabet="default",
        max_iters=200,
        seed=0,
        method="best_of",
        polish_target=None,
        verbose=False,
        pin_invalid_phases=True,
    )
    pinned_illegal = np.asarray(pinned["best_illegal_phases"])
    assert pinned_illegal.shape == (NUM_ILLEGAL_STATES,)
    assert np.allclose(pinned_illegal, 0.0, atol=1e-12), (
        f"pin_invalid_phases=True did not zero all 28 don't-cares "
        f"(max |phi| = {float(np.max(np.abs(pinned_illegal))):.3e})"
    )
    # The full-64 diagonal of the synthesized circuit must equal
    # build_embedded_diag64 entry-by-entry (up to global phase + atol).
    U_pinned = Operator(pinned["phase_gadget_circuit"]).data
    diag_pinned = np.diag(U_pinned)
    target_full = D64_emb
    lam_full = target_full[0] / diag_pinned[0]
    assert np.allclose(diag_pinned * lam_full, target_full, atol=1e-9), (
        "pin_invalid_phases=True: synthesized full diagonal does not "
        "match build_embedded_diag64 up to a global phase"
    )
    # And in particular, every invalid index has |D[k]| close to 1 with
    # phase 0 after removing the global phase factor.
    for k in inv_idx:
        v = diag_pinned[k] * lam_full
        assert np.isclose(v, 1.0 + 0j, atol=1e-9), (
            f"pin_invalid_phases=True: diag[{k}] = {v} (expected ~1+0j)"
        )


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

    # Trace fidelity + leakage against the *explicit* embedded target.
    # By default pin_invalid_phases=True, so the full 64-dim diagonal
    # of every synthesized circuit must equal build_embedded_diag64 up
    # to a global phase.  Both fidelities should hit 1 and there must
    # be zero leakage between logical and invalid subspaces.
    D64_target = build_embedded_diag64(diag36, order="row-major")
    print()
    print("--- explicit embedded-target checks (strict mode) ---")
    print("  F64 = |Tr(U^H D64_embedded)|/64   (full embedding)")
    print("  F36 = |Tr(U_logical^H diag(lambda36))|/36   (logical block)")
    for label, qc_kind in (
        ("phase-gadget", "phase_gadget_circuit"),
        ("transpiled  ", "transpiled_circuit"),
        ("baseline    ", "baseline_circuit"),
    ):
        U = Operator(result[qc_kind]).data
        F64 = trace_fidelity_64(U, D64_target)
        F36 = logical_trace_fidelity_36(U, diag36, order="row-major")
        leak_out, leak_in = leakage_norms(U)
        print(
            f"  {label}: F64={F64:.10f} F36={F36:.10f} "
            f"leak_out={leak_out:.2e} leak_in={leak_in:.2e}"
        )
        assert F64 > 1.0 - 1e-9, (
            f"{label.strip()} circuit full trace_fidelity = {F64:.6f} "
            f"is not ~1 against D64_embedded (pin_invalid_phases default broken?)"
        )
        assert F36 > 1.0 - 1e-9, (
            f"{label.strip()} circuit logical trace_fidelity = {F36:.6f} "
            f"is not ~1 against diag(lambda36)"
        )
        assert leak_out < 1e-10 and leak_in < 1e-10, (
            f"{label.strip()} circuit shows code-space leakage "
            f"(out={leak_out:.2e}, in={leak_in:.2e})"
        )

    # Briefly also exercise the legacy don't-care mode for documentation
    # (smaller CX, but F64 < 1 and diag[invalid] != 1).
    print()
    print("--- legacy don't-care mode (pin_invalid_phases=False) ---")
    legacy = optimize_d6_diagonal(
        diag36,
        phase_alphabet="default",
        max_iters=4000,
        seed=2024,
        order="row-major",
        method="best_of",
        pin_invalid_phases=False,
        verbose=False,
    )
    legacy_metrics = legacy["metrics"]
    legacy_qc = legacy["transpiled_circuit"]
    U_legacy = Operator(legacy_qc).data
    F64_legacy = trace_fidelity_64(U_legacy, D64_target)
    F36_legacy = logical_trace_fidelity_36(U_legacy, diag36, order="row-major")
    print(
        f"  transpiled (legacy): cx={legacy_metrics['transpiled_cx']} "
        f"depth={legacy_metrics['transpiled_depth']} "
        f"F64={F64_legacy:.10f} F36={F36_legacy:.10f}"
    )
    assert F36_legacy > 1.0 - 1e-9, (
        f"legacy mode broke logical action (F36={F36_legacy:.6f})"
    )

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
