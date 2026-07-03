"""
encoding_change_unitary.py

Buduje unitarną macierz zmiany kodowania W (4×4) dla jednego qutrytu
zakodowanego na dwóch qubitach.

Stare (bazowe) kodowanie jest ustalone:
    |0⟩_L → |00⟩,  |1⟩_L → |01⟩,  |2⟩_L → |10⟩

    E_old = [[1,0,0],
             [0,1,0],
             [0,0,1],
             [0,0,0]]

Nowe kodowanie E_new (4×3, izometria) jest podawane jako argument.

Wynikowa macierz W spełnia:
    W @ E_old = E_new
    W†W = I₄

Matematyka:
    W = E_new @ E_old† + |n_new⟩⟨n_old|

gdzie n_old i n_new to znormalizowane wektory z jąder E_old† i E_new†
(dopełnienia ortogonalne podprzestrzeni kodowych).
"""

import numpy as np


# ─────────────────────────── stałe ───────────────────────────

E_OLD = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]],
    dtype=complex,
)

# Nieużywany kierunek starego kodowania: |11⟩
N_OLD = np.array([0, 0, 0, 1], dtype=complex)


# ──────────────────────── nullspace (SVD) ────────────────────

def _nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Zwraca ortonormalną bazę jądra macierzy A (null space) wyznaczoną z SVD.

    Parametry
    ---------
    A : ndarray, shape (m, n)
    tol : float – próg singularny

    Zwraca
    ------
    ndarray, shape (n, k) – kolumny to baza jądra A
    """
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    # Wiersze Vh odpowiadające wartościom singularnym ≈ 0
    null_mask = s < tol
    # Dodaj ewentualne brakujące wymiary (gdy m < n)
    extra = Vh.shape[0] - len(s)
    if extra > 0:
        null_mask = np.concatenate([null_mask, np.ones(extra, dtype=bool)])
    return Vh[null_mask].conj().T


# ──────────────────────── walidacja ──────────────────────────

def validate_encoding_map(E, tol: float = 1e-10) -> dict:
    """
    Sprawdza, czy macierz E jest poprawną mapą kodowania qutrytu w 2 kubity.

    Warunki poprawności
    -------------------
    E : C³ → C⁴   (wymiar 4×3)

    1. Izometria:  E†E = I₃
       – każda kolumna znormalizowana do 1
       – kolumny wzajemnie ortogonalne

    2. Projektor:  P = EE†
       – hermitowski:   P† = P
       – idempotentny:  P² = P

    3. rank(E) = 3

    Zwraca
    ------
    dict z polami: is_valid, correct_type, correct_shape,
        columns_normalized, columns_orthogonal, is_isometry,
        projector_hermitian, projector_idempotent, rank, message
    """
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

    # --- typ ---
    if not isinstance(E, np.ndarray):
        result["message"] = (
            f"E musi być numpy.ndarray, a jest {type(E).__name__}."
        )
        return result
    result["correct_type"] = True

    # --- kształt ---
    if E.shape != (4, 3):
        result["message"] = (
            f"Niepoprawny wymiar macierzy: {E.shape}. Oczekiwano (4, 3)."
        )
        return result
    result["correct_shape"] = True

    # --- rząd ---
    result["rank"] = int(np.linalg.matrix_rank(E, tol=tol))

    # --- Gram matrix G = E†E  (powinna być I₃) ---
    G = E.conj().T @ E

    # normy kolumn
    col_norms_ok = np.allclose(np.diag(G), 1.0, atol=tol)
    result["columns_normalized"] = bool(col_norms_ok)

    # ortogonalność (elementy poza diagonalą ≈ 0)
    off_diag = G - np.diag(np.diag(G))
    cols_ortho_ok = np.allclose(off_diag, 0.0, atol=tol)
    result["columns_orthogonal"] = bool(cols_ortho_ok)

    # izometria łącznie
    isometry_ok = np.allclose(G, np.eye(3), atol=tol)
    result["is_isometry"] = bool(isometry_ok)

    # --- projektor P = EE† ---
    P = E @ E.conj().T

    hermitian_ok = np.allclose(P, P.conj().T, atol=tol)
    result["projector_hermitian"] = bool(hermitian_ok)

    idempotent_ok = np.allclose(P @ P, P, atol=tol)
    result["projector_idempotent"] = bool(idempotent_ok)

    # --- ocena końcowa ---
    all_ok = (
        isometry_ok
        and hermitian_ok
        and idempotent_ok
        and result["rank"] == 3
    )
    result["is_valid"] = bool(all_ok)

    if all_ok:
        result["message"] = "E jest poprawną mapą kodowania (izometrią)."
    else:
        issues = []
        if not col_norms_ok:
            issues.append("kolumny nie są znormalizowane")
        if not cols_ortho_ok:
            issues.append("kolumny nie są ortogonalne")
        if not hermitian_ok:
            issues.append("P = EE† nie jest hermitowskie")
        if not idempotent_ok:
            issues.append("P = EE† nie jest idempotentne (P² ≠ P)")
        if result["rank"] != 3:
            issues.append(f"rząd = {result['rank']}, oczekiwano 3")
        result["message"] = "Niepoprawna mapa: " + "; ".join(issues) + "."

    return result


# ──────────────────── budowanie W ────────────────────────────

def build_encoding_change_unitary(
    E_new: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Buduje unitarną macierz zmiany kodowania W (4×4, complex).

    W @ E_old = E_new
    W†W = I₄

    Parametry
    ---------
    E_new : ndarray, shape (4, 3), complex/real
        Nowa mapa kodowania (izometria C³ → C⁴).
    tol : float
        Tolerancja numeryczna.

    Zwraca
    ------
    W : ndarray, shape (4, 4), dtype complex

    Rzuca
    -----
    ValueError – jeśli E_new nie przechodzi walidacji lub W nie
                 jest unitarna / nie spełnia W @ E_old = E_new.
    """

    # ── 1. walidacja E_new ──
    vres = validate_encoding_map(E_new, tol=tol)
    if not vres["is_valid"]:
        raise ValueError(
            f"E_new nie jest poprawną mapą kodowania.\n"
            f"Szczegóły walidacji:\n"
            + "\n".join(f"  {k}: {v}" for k, v in vres.items())
        )

    # ── 2. nieużywany kierunek nowego kodowania ──
    # n_new ∈ ker(E_new†), tzn.  E_new† @ n_new = 0
    ns = _nullspace(E_new.conj().T, tol=tol)       # shape (4, 1)
    if ns.shape[1] != 1:
        raise ValueError(
            f"Oczekiwano dokładnie 1 wektora jądra E_new†, "
            f"znaleziono {ns.shape[1]}."
        )
    n_new = ns[:, 0]
    # znormalizuj (powinien już być, ale dla pewności)
    n_new = n_new / np.linalg.norm(n_new)

    # ── 3. budujemy W ──
    # W = E_new @ E_old† + |n_new⟩⟨n_old|
    W = (
        E_new @ E_OLD.conj().T
        + np.outer(n_new, N_OLD.conj())
    )
    W = W.astype(complex)

    # ── 4. weryfikacja unitarności ──
    if not np.allclose(W.conj().T @ W, np.eye(4), atol=tol):
        raise ValueError("Zbudowana macierz W nie jest unitarna (W†W ≠ I₄).")

    # ── 5. weryfikacja W @ E_old == E_new ──
    if not np.allclose(W @ E_OLD, E_new, atol=tol):
        raise ValueError("W @ E_old ≠ E_new — coś poszło nie tak.")

    return W


# ══════════════════════ przykłady ════════════════════════════

if __name__ == "__main__":

    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    sep = "=" * 60

    # ─── helper do wypisywania ───
    def _print_check(label: str, ok: bool):
        print(f"  {label}: {'✓' if ok else '✗'}")

    # ─────────────────────────────────────────────────────────
    # Przykład 1: trywialna zmiana — E_new = E_old → W = I₄
    # ─────────────────────────────────────────────────────────
    print(sep)
    print("Przykład 1: E_new = E_old  →  W powinno być I₄")
    print(sep)

    E_trivial = E_OLD.copy()
    W1 = build_encoding_change_unitary(E_trivial)

    print("W =")
    print(W1)
    _print_check("W unitarna (W†W = I₄)",
                 np.allclose(W1.conj().T @ W1, np.eye(4)))
    _print_check("W @ E_old = E_new",
                 np.allclose(W1 @ E_OLD, E_trivial))
    _print_check("W ≈ I₄",
                 np.allclose(W1, np.eye(4)))

    # ─────────────────────────────────────────────────────────
    # Przykład 2: nietrywialna zmiana — obrót w podprzestrzeni
    # kodowej (Hadamard-like na stanach |0⟩_L i |1⟩_L)
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Przykład 2: E_new z obrotem Hadamarda na |0⟩_L, |1⟩_L")
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
    print("Walidacja E_new:")
    for k, v in vres2.items():
        print(f"  {k}: {v}")

    W2 = build_encoding_change_unitary(E_had)
    print("\nW =")
    print(W2)
    _print_check("W unitarna", np.allclose(W2.conj().T @ W2, np.eye(4)))
    _print_check("W @ E_old = E_new", np.allclose(W2 @ E_OLD, E_had))

    # ─────────────────────────────────────────────────────────
    # Przykład 3: niepoprawna mapa kodowania → ValueError
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Przykład 3: niepoprawna mapa kodowania → wyjątek")
    print(sep)

    E_bad = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 1]],
        dtype=complex,
    )

    vres3 = validate_encoding_map(E_bad)
    print("Walidacja E_bad:")
    for k, v in vres3.items():
        print(f"  {k}: {v}")

    try:
        build_encoding_change_unitary(E_bad)
    except ValueError as exc:
        print(f"\nZłapano ValueError:\n{exc}")
