"""Build the unitary encoding-change matrix W (4×4) for a qutrit on two qubits.

The fixed base encoding maps |0>_L to |00>, |1>_L to |01>, and |2>_L to |10>:
    E_old = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
The caller supplies the new encoding isometry E_new (4×3).

W @ E_old = E_new and W†W = I₄, using
    W = E_new @ E_old† + |n_new><n_old|
where n_old and n_new are normalized vectors in the null spaces of E_old†
and E_new†, respectively (orthogonal complements of the code spaces)."""

import numpy as np


# ─────────────────────────── constants ───────────────────────

E_OLD = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]],
    dtype=complex,
)

# Unused direction of the old encoding: |11⟩
N_OLD = np.array([0, 0, 0, 1], dtype=complex)


# ──────────────────────── nullspace (SVD) ────────────────────

def _nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis of the null space of A using SVD.

    A has shape (m, n); tol is the singular-value threshold.
    The returned array has shape (n, k), with null-space basis vectors as columns."""
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    # Rows of Vh corresponding to singular values ≈ 0
    null_mask = s < tol
    # Add any missing dimensions (when m < n)
    extra = Vh.shape[0] - len(s)
    if extra > 0:
        null_mask = np.concatenate([null_mask, np.ones(extra, dtype=bool)])
    return Vh[null_mask].conj().T


# ──────────────────────── validation ─────────────────────────

def validate_encoding_map(E, tol: float = 1e-10) -> dict:
    """Check whether E is a valid encoding of one qutrit into two qubits.

    E must be a numpy.ndarray of shape (4, 3), mapping C³ to C⁴, with rank 3.
    Its columns must be normalized and orthogonal: E†E = I₃.
    The projector P = EE† must be Hermitian (P† = P) and idempotent (P² = P).

    Return a dict with is_valid, correct_type, correct_shape, columns_normalized,
    columns_orthogonal, is_isometry, projector_hermitian, projector_idempotent,
    rank, and message."""
    result = {
        "is_valid":             False,
        "correct_type":         False,
        "correct_shape":        False,
        "columns_normalized":   False,
        "columns_orthogonal":   False,
        "is_isometry":          False,
        "projector_hermitian":  False,
        "projector_idempotent": False,
        "rank":                 None,
        "message":              "",
    }

    # --- type ---
    if not isinstance(E, np.ndarray):
        result["message"] = (
            f"E must be numpy.ndarray; received {type(E).__name__}."
        )
        return result
    result["correct_type"] = True

    # --- shape ---
    if E.shape != (4, 3):
        result["message"] = (
            f"Invalid matrix shape: {E.shape}. Expected (4, 3)."
        )
        return result
    result["correct_shape"] = True

    # --- rank ---
    result["rank"] = int(np.linalg.matrix_rank(E, tol=tol))

    # --- Gram matrix G = E†E  (should be I₃) ---
    G = E.conj().T @ E

    # column norms
    col_norms_ok = np.allclose(np.diag(G), 1.0, atol=tol)
    result["columns_normalized"] = bool(col_norms_ok)

    # orthogonality (off-diagonal elements ≈ 0)
    off_diag = G - np.diag(np.diag(G))
    cols_ortho_ok = np.allclose(off_diag, 0.0, atol=tol)
    result["columns_orthogonal"] = bool(cols_ortho_ok)

    # overall isometry check
    isometry_ok = np.allclose(G, np.eye(3), atol=tol)
    result["is_isometry"] = bool(isometry_ok)

    # --- projector P = EE† ---
    P = E @ E.conj().T

    hermitian_ok = np.allclose(P, P.conj().T, atol=tol)
    result["projector_hermitian"] = bool(hermitian_ok)

    idempotent_ok = np.allclose(P @ P, P, atol=tol)
    result["projector_idempotent"] = bool(idempotent_ok)

    # --- final assessment ---
    all_ok = (
        isometry_ok
        and hermitian_ok
        and idempotent_ok
        and result["rank"] == 3
    )
    result["is_valid"] = bool(all_ok)

    if all_ok:
        result["message"] = "E is a valid encoding map (isometry)."
    else:
        issues = []
        if not col_norms_ok:
            issues.append("columns are not normalized")
        if not cols_ortho_ok:
            issues.append("columns are not orthogonal")
        if not hermitian_ok:
            issues.append("P = EE† is not Hermitian")
        if not idempotent_ok:
            issues.append("P = EE† is not idempotent (P² ≠ P)")
        if result["rank"] != 3:
            issues.append(f"rank = {result['rank']}; expected 3")
        result["message"] = "Invalid encoding map: " + "; ".join(issues) + "."

    return result


# ──────────────────── constructing W ─────────────────────────

def build_encoding_change_unitary(
    E_new: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Build a complex unitary encoding-change matrix W of shape (4, 4).

    The result satisfies W @ E_old = E_new and W†W = I₄.
    E_new is a real or complex ndarray of shape (4, 3), an isometry C³ to C⁴.
    tol is the numerical tolerance.

    Raise ValueError if E_new fails validation, W is not unitary, or
    W @ E_old = E_new is not satisfied."""

    # ── 1. validate E_new ──
    vres = validate_encoding_map(E_new, tol=tol)
    if not vres["is_valid"]:
        raise ValueError(
            f"E_new is not a valid encoding map.\n"
            f"Validation details:\n"
            + "\n".join(f"  {k}: {v}" for k, v in vres.items())
        )

    # ── 2. unused direction of the new encoding ──
    # n_new ∈ ker(E_new†), i.e.  E_new† @ n_new = 0
    ns = _nullspace(E_new.conj().T, tol=tol)       # shape (4, 1)
    if ns.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one null-space vector for E_new†; "
            f"found {ns.shape[1]}."
        )
    n_new = ns[:, 0]
    # normalize defensively (it should already be normalized)
    n_new = n_new / np.linalg.norm(n_new)

    # ── 3. construct W ──
    # W = E_new @ E_old† + |n_new⟩⟨n_old|
    W = (
        E_new @ E_OLD.conj().T
        + np.outer(n_new, N_OLD.conj())
    )
    W = W.astype(complex)

    # ── 4. verify unitarity ──
    if not np.allclose(W.conj().T @ W, np.eye(4), atol=tol):
        raise ValueError("The constructed matrix W is not unitary (W†W ≠ I₄).")

    # ── 5. verify W @ E_old == E_new ──
    if not np.allclose(W @ E_OLD, E_new, atol=tol):
        raise ValueError("Encoding-change verification failed: W @ E_old ≠ E_new.")

    return W


# ══════════════════════ examples ═════════════════════════════

if __name__ == "__main__":

    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    sep = "=" * 60

    # ─── printing helper ───
    def _print_check(label: str, ok: bool):
        print(f"  {label}: {'✓' if ok else '✗'}")

    # ─────────────────────────────────────────────────────────
    # Example 1: trivial change — E_new = E_old → W = I₄
    # ─────────────────────────────────────────────────────────
    print(sep)
    print("Example 1: E_new = E_old; W should be I₄")
    print(sep)

    E_trivial = E_OLD.copy()
    W1 = build_encoding_change_unitary(E_trivial)

    print("W =")
    print(W1)
    _print_check("W is unitary (W†W = I₄)",
                 np.allclose(W1.conj().T @ W1, np.eye(4)))
    _print_check("W @ E_old = E_new",
                 np.allclose(W1 @ E_OLD, E_trivial))
    _print_check("W ≈ I₄",
                 np.allclose(W1, np.eye(4)))

    # ─────────────────────────────────────────────────────────
    # Example 2: nontrivial change — rotation in the code subspace
    # (Hadamard-like on states |0⟩_L and |1⟩_L)
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Example 2: E_new with a Hadamard rotation on |0⟩_L, |1⟩_L")
    print(sep)

    a = 1.0 / np.sqrt(2)
    # |0⟩_L → a|00⟩ + a|01⟩,  |1⟩_L → a|00⟩ - a|01⟩,  |2⟩_L → |10⟩
    E_had = np.array(
        [[ a,  a, 0],
         [ a, -a, 0],
         [ 0,  0, 1],
         [ 0,  0, 0]],
        dtype=complex,
    )

    vres2 = validate_encoding_map(E_had)
    print("E_new validation:")
    for k, v in vres2.items():
        print(f"  {k}: {v}")

    W2 = build_encoding_change_unitary(E_had)
    print("\nW =")
    print(W2)
    _print_check("W is unitary", np.allclose(W2.conj().T @ W2, np.eye(4)))
    _print_check("W @ E_old = E_new", np.allclose(W2 @ E_OLD, E_had))

    # ─────────────────────────────────────────────────────────
    # Example 3: invalid encoding map → ValueError
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Example 3: an invalid encoding map raises an exception")
    print(sep)

    E_bad = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 1]],
        dtype=complex,
    )

    vres3 = validate_encoding_map(E_bad)
    print("E_bad validation:")
    for k, v in vres3.items():
        print(f"  {k}: {v}")

    try:
        build_encoding_change_unitary(E_bad)
    except ValueError as exc:
        print(f"\nCaught ValueError:\n{exc}")
