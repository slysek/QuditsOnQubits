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


def unitaries_equal_up_to_global_phase(
    U: np.ndarray,
    V: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Return True iff U and V are equal up to a multiplicative global phase.

    Compares entrywise: finds the largest-magnitude entry of V, derives the
    candidate phase ``lambda = U[idx] / V[idx]``, and checks
    ``U == lambda * V`` within ``atol``.

    Both inputs must have the same shape. Works for diagonal matrices, full
    unitaries, and any other complex array of equal shape.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    if U.shape != V.shape:
        return False

    abs_V = np.abs(V)
    idx_flat = int(np.argmax(abs_V))
    max_mag = float(abs_V.flat[idx_flat])

    if max_mag < atol:
        return bool(np.allclose(U, 0.0, atol=atol))

    lam = U.flat[idx_flat] / V.flat[idx_flat]
    if abs(abs(lam) - 1.0) > max(atol, 1e-6):
        return False

    return bool(np.allclose(U, lam * V, atol=atol))


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
    separately as ``constant``; only nonempty subsets with ``|c_S| > atol``
    are returned in ``coefficients``.

    Parameters
    ----------
    D_diag : np.ndarray
        1-D length-2**n array of unit-modulus complex numbers.
    atol : float
        Threshold for treating a coefficient as zero.

    Returns
    -------
    dict
        Keys: ``num_qubits``, ``constant``, ``coefficients``,
        ``num_nonzero``, ``total_terms``, ``sparsity``, ``max_weight``,
        ``weight_histogram``.
    """
    num_qubits = _validate_diagonal(D_diag, atol=1e-8)
    N = 1 << num_qubits

    theta = np.angle(D_diag).astype(float, copy=False)

    # In-place Walsh-Hadamard butterfly with sign convention
    # h[m] = sum_k (-1)^{popcount(m AND k)} * theta_k.
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


def _bit_permute(k: int, perm: Sequence[int], num_qubits: int) -> int:
    """Permute the bits of integer k according to perm.

    The result has bit ``i`` taken from bit ``perm[i]`` of k:

        result_bit_i = k_bit_{perm[i]}.

    This matches Qiskit's wiring convention: when a gate is appended to
    qubits ``perm = [p_0, ..., p_{n-1}]``, the gate's qubit ``i`` is wired to
    circuit qubit ``p_i``. The diagonal entry seen by the gate at circuit
    basis state ``|k>`` is therefore ``D_diag[bit_permute(k, perm, n)]``.
    """
    result = 0
    for i in range(num_qubits):
        bit = (k >> perm[i]) & 1
        result |= bit << i
    return result


def _permute_diagonal(D_diag: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Return D_diag re-indexed by bit-permutation perm.

    ``out[k] = D_diag[bit_permute(k, perm, n)]`` for ``k in range(2**n)``.
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

    - ``try_perms=False``: only the identity.
    - ``num_qubits <= 4``: all ``n!`` permutations.
    - ``5 <= num_qubits <= 6``: identity + ``sampled_n_5_6`` random distinct
      permutations.
    - ``num_qubits > 6``: identity only, with a UserWarning.
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
        "skipped (combinatorial explosion). Falling back to identity-only "
        "permutation.",
        UserWarning,
        stacklevel=2,
    )
    return [identity]


# Canonical 2-qubit gate names used only for documentation in the printed
# table; the actual two-qubit count is taken from len(qargs) == 2 so any
# non-canonical 2-qubit op is still counted correctly.
_CANONICAL_TWO_QUBIT_GATE_NAMES = frozenset(
    {"cx", "cz", "ecr", "iswap", "swap", "rzz", "rxx", "ryy", "csx", "dcx"}
)
_NON_GATE_OPS = frozenset({"barrier", "measure", "reset", "delay"})


def _count_metrics(qc: QuantumCircuit) -> dict:
    """Count gates, two-qubit gates, depth, and per-op breakdown.

    Barriers, measurements, resets, and delays are excluded from the gate
    counts and the breakdown.
    """
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


_DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]


def _resolve_basis_gates(backend, basis_gates):
    """Return the basis_gates list to pass to transpile.

    If a backend is provided, return None (transpile will use the backend's
    native instructions). Otherwise, use ``basis_gates`` if given, else the
    default IBM-style set ``["rz", "sx", "x", "cx"]``.
    """
    if backend is not None:
        return None
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

    ``D_diag_perm`` is the *already-permuted* diagonal that the candidate is
    supposed to implement; qubit-permutation has been baked in upstream by
    the driver via ``_permute_diagonal``.
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


def _append_z_product_term(
    qc: QuantumCircuit,
    subset_sorted: list,
    coefficient: float,
) -> None:
    """Append ``exp(i * coefficient * Z_S)`` onto qc, with ``S = subset_sorted``.

    The block is exact up to an irrelevant global phase coming from RZ's
    convention; since validation is up-to-global-phase, this is fine.

    For weight 1 (``|S| = 1``) the block is just ``RZ(-2c)`` on the single
    qubit. For weight >= 2 it is a CX ladder onto target ``subset_sorted[-1]``,
    one ``RZ(-2c)``, then the reverse CX ladder.
    """
    if not subset_sorted:
        return  # the constant term is handled via qc.global_phase

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
    """Build an un-transpiled circuit implementing ``diag(D_diag_perm)``.

    Iterates the Z-product expansion's nonzero subsets, in a Gray-code-like
    order (sort by Hamming weight first, then by bitmask within a weight),
    appending one CX-ladder + RZ + reverse-ladder block per subset.
    """
    coeffs = z_phase_coefficients_from_diag(D_diag_perm)
    n = coeffs["num_qubits"]

    qc = QuantumCircuit(n, name="sparse_phase_poly")
    if not normalize_phase:
        qc.global_phase = coeffs["constant"]

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


_VALIDATION_SKIP_THRESHOLD_QUBITS = 12


def _print_comparison_table(candidates: list, best: dict) -> None:
    """Print a plain-text comparison table sorted by score.

    Columns: best | strategy | seed | perm | norm | rz | cx | 2q | total |
    depth | validation. The row that matches ``best`` is annotated with a
    leading ``BEST`` marker.
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

    Tries multiple strategies (Qiskit ``DiagonalGate`` + transpile, custom
    sparse phase-polynomial synthesis), each combined with optional qubit
    permutations, optional global-phase normalization, and a sweep over
    ``seed_transpiler`` values. Validates each candidate up to global phase
    (skipped when ``num_qubits > 12``). Returns the best candidate found
    among the candidates actually attempted.

    This function does NOT prove mathematical optimality - it returns the
    best of the candidates it tried.

    Parameters
    ----------
    D_diag : np.ndarray
        1-D length-2**n complex array of unit-modulus phases.
    backend : optional
        If provided, transpile against this backend (``basis_gates`` is then
        ignored).
    basis_gates : optional
        Basis gates list. Defaults to ``["rz", "sx", "x", "cx"]`` if both
        ``backend`` and ``basis_gates`` are None.
    optimization_level : int
        Passed to :func:`qiskit.transpile`.
    seeds : iterable of int
        ``seed_transpiler`` values to sweep.
    try_qubit_permutations : bool
        Whether to try non-identity qubit relabelings of the diagonal.
    try_global_phase_normalization : bool
        Whether to also try dividing the input by ``D_diag[0]`` (drops the
        constant term of the Z-product expansion).
    metric : str
        Currently only ``"two_qubit_then_depth"`` is supported.
    atol : float
        Absolute tolerance for validation.
    verbose : bool
        If True, prints a comparison table after the search.

    Returns
    -------
    dict
        Keys: ``best_circuit``, ``best_score``, ``best_metadata``,
        ``all_candidates``, ``diagnostics``.
    """
    num_qubits = _validate_diagonal(np.asarray(D_diag), atol=atol)
    D_diag = np.asarray(D_diag, dtype=complex).copy()

    resolved_basis_gates = _resolve_basis_gates(backend, basis_gates)
    phase_coefficients = z_phase_coefficients_from_diag(D_diag)

    # Validate the metric early so we fail fast on bad input.
    _ = _score(
        {"two_qubit": 0, "depth": 0, "total_gates": 0}, metric
    )

    perms = _enumerate_permutations(
        num_qubits, try_perms=try_qubit_permutations, rng_seed=0
    )

    normalizations: list = [False]
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

    candidates: list = []
    failed: list = []

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
