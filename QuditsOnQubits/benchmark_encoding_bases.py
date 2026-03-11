"""
benchmark_encoding_bases.py

Automatyczny benchmark różnych baz kodowania qutrytu (C³→C⁴)
pod kątem głębokości obwodu po transpilacji na konkretną topologię.

Klasy testowanych baz — STARE (E_new = E_old @ S, w starej code space):
  1. baseline              – klasyczne E_old
  2. monomial              – E_old @ D @ P  (permutacja + fazy)
  3. fourier_like          – E_old @ D1 @ F3 @ D2
  4. householder_random    – E_old @ losowa unitarna 3×3 (Haar)
  5. clifford_wh           – E_old @ X3^a Z3^b F3^c

Klasy testowanych baz — NOWE (ogólne izometrie 4×3, pełna C⁴):
  6. haar_random_isometry       – losowa izometria 4×3 (Haar)
  7. perturbed_isometry         – E_old + mała perturbacja → reortonormalizacja
  8. entangling_isometry        – W_random(4×4) @ E_old
  9. structured_entangling      – W z lokalnych rotacji + bramka splątująca

Wyniki zapisywane do CSV i wypisywane w terminalu.
"""

import numpy as np
import pandas as pd
import traceback
import time
from itertools import product as iter_product

from qiskit import transpile
from QuditsOnQubits import create_ame_circuit
from encoding_change_unitary import validate_encoding_map

# ═══════════════════════════ STAŁE ═══════════════════════════

E_OLD = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]],
    dtype=complex,
)

# Projektor na starą code space: span{|00>,|01>,|10>}
P_OLD_CODESPACE = E_OLD @ E_OLD.conj().T  # 4×4

OMEGA = np.exp(2j * np.pi / 3)

BASIS_GATES = ["cz", "id", "rx", "rz", "rzz", "sx", "x"]

COUPLING_MAP = [
    # poziome
    [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],
    [16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],
    [32,33],[33,34],[34,35],[35,36],[36,37],[37,38],[38,39],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],
    [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,61],[61,62],[62,63],
    [64,65],[65,66],[66,67],[67,68],[68,69],[69,70],[70,71],[71,72],[72,73],[73,74],[74,75],[75,76],[76,77],[77,78],[78,79],
    [80,81],[81,82],[82,83],[83,84],[84,85],[85,86],[86,87],[87,88],[88,89],[89,90],[90,91],[91,92],[92,93],[93,94],[94,95],
    [96,97],[97,98],[98,99],[99,100],[100,101],[101,102],[102,103],[103,104],[104,105],[105,106],[106,107],[107,108],[108,109],[109,110],[110,111],
    [112,113],[113,114],[114,115],[115,116],[116,117],[117,118],[118,119],[119,120],[120,121],[121,122],[122,123],[123,124],[124,125],[125,126],[126,127],
    # pionowe
    [3,19],[7,23],[11,27],[15,31],
    [17,33],[21,37],[25,41],[29,45],
    [35,51],[39,55],[43,59],[47,63],
    [49,65],[53,69],[57,73],[61,77],
    [67,83],[71,87],[75,91],[79,95],
    [81,97],[85,101],[89,105],[93,109],
    [99,115],[103,119],[107,123],[111,127],
]


# ════════════════════ OPERATORY QUTRYTOWE (3×3) ══════════════

def qutrit_X():
    """Cykliczny shift X₃:  |k⟩ → |k+1 mod 3⟩."""
    return np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]],
        dtype=complex,
    )


def qutrit_Z():
    """Operator fazowy Z₃:  |k⟩ → ω^k |k⟩."""
    return np.diag([1, OMEGA, OMEGA**2]).astype(complex)


def qutrit_fourier():
    """Macierz Fouriera F₃ dla qutrytu."""
    return np.array(
        [[1, 1,        1       ],
         [1, OMEGA,    OMEGA**2],
         [1, OMEGA**2, OMEGA**4]],
        dtype=complex,
    ) / np.sqrt(3)


def get_E_old():
    """Zwraca bazowe kodowanie E_old (4×3)."""
    return E_OLD.copy()


# ════════════════════ POMOCNICZE ═════════════════════════════

def _random_unitary(n, rng):
    """Losowa macierz unitarna n×n (Haar) przez QR."""
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    Q = Q @ np.diag(d / np.abs(d))
    return Q


def _random_unitary_3x3(rng):
    """Losowa macierz unitarna 3×3 (Haar)."""
    return _random_unitary(3, rng)


def _random_unitary_4x4(rng):
    """Losowa macierz unitarna 4×4 (Haar)."""
    return _random_unitary(4, rng)


def _perm_matrix(perm):
    """Macierz permutacji 3×3 z krotki np. (1,2,0)."""
    P = np.zeros((3, 3), dtype=complex)
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P


def _phase_diag(phases):
    """Diagonalna macierz faz z listy/krotki 3 wartości."""
    return np.diag(np.array(phases, dtype=complex))


_PERMS_3 = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
]

_PHASES_3 = [1, OMEGA, OMEGA**2]


# ────────────── metryki kodowania ──────────────

def _codeword_entanglement(col):
    """
    Entropia splątania (von Neumann) 2-qubitowego stanu czystego |ψ⟩.

    col : ndarray, shape (4,) — znormalizowany wektor stanu
    Zwraca float ∈ [0, 1] (w bitach).
    """
    # Reshape do 2×2 (qubit_A ⊗ qubit_B)
    psi = col.reshape(2, 2)
    # Zredukowana macierz gęstości ρ_A = Tr_B(|ψ⟩⟨ψ|)
    rho_A = psi @ psi.conj().T
    eigvals = np.linalg.eigvalsh(rho_A)
    # Wytnij numeryczne szumy < 0
    eigvals = eigvals[eigvals > 1e-14]
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log2(eigvals)))


def compute_encoding_metadata(E_new):
    """
    Oblicza dodatkowe metryki charakteryzujące mapę kodowania.

    Zwraca dict:
      uses_old_codespace_only : bool
          True jeśli E_new mieści się w span{|00>,|01>,|10>}
      avg_codeword_entanglement : float
          Średnia entropia splątania (bity) trzech codewordów
      overlap_with_old_codespace : float
          ||P_old @ E_new||_F² / 3  ∈ [0, 1]
    """
    if E_new is None:
        return {
            "uses_old_codespace_only": True,
            "avg_codeword_entanglement": 0.0,
            "overlap_with_old_codespace": 1.0,
        }

    # Czy czwarty wiersz (|11⟩) jest zerowy?
    row11_norm = np.linalg.norm(E_new[3, :])
    uses_old = bool(row11_norm < 1e-10)

    # Średnie splątanie codewordów
    ents = [_codeword_entanglement(E_new[:, j]) for j in range(3)]
    avg_ent = float(np.mean(ents))

    # Overlap z starą code space
    proj = P_OLD_CODESPACE @ E_new  # 4×3
    overlap = np.real(np.trace(proj.conj().T @ proj)) / 3.0

    return {
        "uses_old_codespace_only": uses_old,
        "avg_codeword_entanglement": round(avg_ent, 6),
        "overlap_with_old_codespace": round(float(overlap), 6),
    }


# ════════════════ GENERATORY BAZ — STARA CODE SPACE ═════════

def generate_baseline():
    """Klasa 0: baseline – samo E_old."""
    return [("baseline", "E_old", None)]  # E_new=None → domyślne kodowanie


def generate_monomial_bases(max_candidates=120):
    """
    Klasa 1: E_new = E_old @ D @ P
    Permutacja + diagonalne fazy z {1, ω, ω²}.
    """
    candidates = []
    for perm in _PERMS_3:
        P = _perm_matrix(perm)
        for ph in iter_product(_PHASES_3, repeat=3):
            D = _phase_diag(ph)
            S = D @ P
            E_new = E_OLD @ S
            name = f"P{''.join(map(str,perm))}_ph{''.join(str(int(round(np.angle(p)*3/(2*np.pi)))%3) for p in ph)}"
            candidates.append(("monomial", name, E_new))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def generate_fourier_like_bases(max_candidates=80):
    """
    Klasa 2: E_new = E_old @ D1 @ F3 @ D2
    Fourier z fazami po obu stronach.
    """
    F3 = qutrit_fourier()
    candidates = []
    small_phases = [1, OMEGA]
    for ph1 in iter_product(small_phases, repeat=3):
        D1 = _phase_diag(ph1)
        for ph2 in iter_product(small_phases, repeat=3):
            D2 = _phase_diag(ph2)
            S = D1 @ F3 @ D2
            E_new = E_OLD @ S
            name = (
                f"D{''.join(str(int(round(np.angle(p)*3/(2*np.pi)))%3) for p in ph1)}"
                f"_F3_"
                f"D{''.join(str(int(round(np.angle(p)*3/(2*np.pi)))%3) for p in ph2)}"
            )
            candidates.append(("fourier_like", name, E_new))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def generate_householder_bases(n_samples=20, seed=42):
    """
    Klasa 3: E_new = E_old @ losowa_unitarna_3x3
    Małe losowe próbkowanie z U(3).  Zostaje w starej code space.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_samples):
        S = _random_unitary_3x3(rng)
        E_new = E_OLD @ S
        candidates.append(("householder_random", f"rand_{i:03d}", E_new))
    return candidates


def generate_clifford_wh_bases():
    """
    Klasa 4: E_new = E_old @ X3^a @ Z3^b @ F3^c
    Kombinacje generatorów Weyl-Heisenberg.
    """
    X3 = qutrit_X()
    Z3 = qutrit_Z()
    F3 = qutrit_fourier()
    candidates = []
    for a, b, c in iter_product(range(3), repeat=3):
        S = np.linalg.matrix_power(X3, a) @ np.linalg.matrix_power(Z3, b) @ np.linalg.matrix_power(F3, c)
        E_new = E_OLD @ S
        name = f"X{a}Z{b}F{c}"
        candidates.append(("clifford_wh", name, E_new))
    return candidates


# ════════════ GENERATORY BAZ — OGÓLNE IZOMETRIE 4×3 ═════════

def generate_haar_random_isometries(n_samples=20, seed=100):
    """
    Klasa 5: Losowa izometria 4×3 (Haar).
    Bierzemy pierwsze 3 kolumny losowej macierzy unitarnej 4×4.
    Pełna C⁴ — nie ograniczona do starej code space.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_samples):
        U = _random_unitary_4x4(rng)
        E_new = U[:, :3].copy()
        candidates.append(("haar_random_isometry", f"haar_{i:03d}", E_new))
    return candidates


def generate_perturbed_isometries(n_samples_per_eps=8, seed=200):
    """
    Klasa 6: E_old + mała perturbacja → reortonormalizacja (polar/QR).
    Testuje kodowania „blisko" klasycznego, z różnym poziomem perturbacji.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    epsilons = [0.01, 0.05, 0.1, 0.3]

    for eps in epsilons:
        for i in range(n_samples_per_eps):
            noise = (rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))) * eps
            E_pert = E_OLD + noise
            # Reortonormalizacja kolumn przez QR
            Q, R = np.linalg.qr(E_pert, mode='reduced')
            # Ustal orientację faz (żeby QR nie flipował znaków)
            signs = np.diag(np.sign(np.diag(R)))
            E_new = Q @ signs
            name = f"pert_eps{eps:.2f}_{i:02d}"
            candidates.append(("perturbed_isometry", name, E_new))
    return candidates


def generate_entangling_isometries(n_samples=20, seed=300):
    """
    Klasa 7: E_new = W @ E_old   gdzie W jest losową unitarną 4×4.
    Codewordy mogą mieć amplitudę na |11⟩ — ogólna zmiana code space.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_samples):
        W = _random_unitary_4x4(rng)
        E_new = W @ E_OLD
        candidates.append(("entangling_isometry", f"ent_{i:03d}", E_new))
    return candidates


def generate_structured_entangling_isometries():
    """
    Klasa 8: W = (Ry(θ)⊗Ry(φ)) @ CZ @ (Rx(α)⊗I)  → E_new = W @ E_old
    Prosta parametryczna rodzina z 1 bramką splątującą (CZ) i lokalnymi rotacjami.
    Mały grid parametrów.
    """
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator as QOp

    # CZ jako macierz 4×4
    cz_mtx = QOp(CZGate()).data

    def _ry(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rx(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    candidates = []

    for theta in angles:
        for phi in angles:
            for alpha in angles:
                local_pre = np.kron(_rx(alpha), np.eye(2, dtype=complex))
                local_post = np.kron(_ry(theta), _ry(phi))
                W = local_post @ cz_mtx @ local_pre
                E_new = W @ E_OLD
                name = f"t{theta:.2f}_p{phi:.2f}_a{alpha:.2f}"
                candidates.append(("structured_entangling", name, E_new))

    return candidates


# ═══════ GENERATORY BAZ — ROZSZERZONY SEARCH (nowe klasy) ════

def generate_local_ry_only(n_grid=10):
    """
    Klasa 10: W = Ry(θ) ⊗ Ry(φ) — wyłącznie rotacje lokalne.
    ZERO dodatkowych bramek 2-qubitowych z samego W.
    Gęsty grid kątów.
    """
    def _ry(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    angles = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    candidates = []
    for theta in angles:
        for phi in angles:
            if abs(theta) < 1e-10 and abs(phi) < 1e-10:
                continue  # pomiń tożsamość
            W = np.kron(_ry(theta), _ry(phi))
            E_new = W @ E_OLD
            name = f"ry_{theta:.3f}_{phi:.3f}"
            candidates.append(("local_ry_only", name, E_new))
    return candidates


def generate_local_general_su2(n_samples=30, seed=600):
    """
    Klasa 11: W = U₁ ⊗ U₂  gdzie U₁, U₂ ∈ SU(2) losowe (Haar).
    Zero dodatkowych bramek 2-qubitowych z W.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_samples):
        U1 = _random_unitary(2, rng)
        U2 = _random_unitary(2, rng)
        W = np.kron(U1, U2)
        E_new = W @ E_OLD
        candidates.append(("local_general_su2", f"lsu2_{i:03d}", E_new))
    return candidates


def generate_real_orthogonal_isometries(n_samples=20, seed=400):
    """
    Klasa 12: Losowa realna ortogonalna izometria 4×3.
    Macierze czysto rzeczywiste mogą dawać prostsze dekompozycje.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_samples):
        Z = rng.standard_normal((4, 4))
        Q, R = np.linalg.qr(Z)
        d = np.diag(np.sign(np.diag(R)))
        Q = Q @ d
        E_new = Q[:, :3].astype(complex)
        candidates.append(("real_orthogonal", f"real_{i:03d}", E_new))
    return candidates


def generate_near_identity_isometries(n_samples_per_eps=10, seed=500):
    """
    Klasa 13: W = expm(i·ε·H) dla małego ε i losowej macierzy hermitowskiej H.
    Bardzo blisko macierzy identyczności — minimalny overhead.
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    candidates = []
    epsilons = [0.01, 0.03, 0.05, 0.1]

    for eps in epsilons:
        for i in range(n_samples_per_eps):
            A = (rng.standard_normal((4, 4))
                 + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
            H = (A + A.conj().T) / 2
            W = expm(1j * eps * H)
            E_new = W @ E_OLD
            name = f"nearid_eps{eps:.2f}_{i:02d}"
            candidates.append(("near_identity", name, E_new))
    return candidates


def generate_finer_structured_grid():
    """
    Klasa 14: Dokładniejszy grid wokół najlepszych parametrów
    z klasy structured_entangling.
    Najlepsze wyniki były przy θ≈0, φ≈π/2…π, α≈0.5…1.2.
    """
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator as QOp

    cz_mtx = QOp(CZGate()).data

    def _ry(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rx(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    thetas = np.linspace(0, 0.5, 4)
    phis = np.linspace(1.2, 1.7, 5)
    alphas = np.linspace(0.5, 1.1, 5)

    candidates = []
    for theta in thetas:
        for phi in phis:
            for alpha in alphas:
                local_pre = np.kron(_rx(alpha), np.eye(2, dtype=complex))
                local_post = np.kron(_ry(theta), _ry(phi))
                W = local_post @ cz_mtx @ local_pre
                E_new = W @ E_OLD
                name = f"fine_t{theta:.2f}_p{phi:.2f}_a{alpha:.2f}"
                candidates.append(("finer_structured", name, E_new))
    return candidates


def generate_two_cz_ansatz(n_samples=50, seed=700):
    """
    Klasa 15: W = (Ry(θ₂)⊗Ry(φ₂)) @ CZ @ (Ry(θ₁)⊗Ry(φ₁)) @ CZ @ (Rx(α)⊗I)
    Dwie warstwy splątujące CZ — większa ekspresywność.
    Losowe próbkowanie 5 parametrów.
    """
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator as QOp

    cz_mtx = QOp(CZGate()).data
    rng = np.random.default_rng(seed)

    def _ry(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rx(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    candidates = []
    for i in range(n_samples):
        params = rng.uniform(0, 2 * np.pi, size=5)
        t1, p1, t2, p2, alpha = params

        local_pre = np.kron(_rx(alpha), np.eye(2, dtype=complex))
        mid = np.kron(_ry(t1), _ry(p1))
        local_post = np.kron(_ry(t2), _ry(p2))
        W = local_post @ cz_mtx @ mid @ cz_mtx @ local_pre
        E_new = W @ E_OLD
        name = f"2cz_{i:03d}"
        candidates.append(("two_cz_ansatz", name, E_new))
    return candidates


# ════════════════════ BENCHMARK JEDNEGO PRZYPADKU ════════════

def benchmark_basis(E_new, class_name, candidate_name,
                    n_qutrits=3, coupling_map=None, basis_gates=None,
                    n_transpile_runs=20):
    """
    Buduje obwód, transpiluje n_transpile_runs razy.
    Zbiera pełne statystyki: best, mean, std.

    Zwraca dict z wynikami.
    """
    if coupling_map is None:
        coupling_map = COUPLING_MAP
    if basis_gates is None:
        basis_gates = BASIS_GATES

    # ── metryki kodowania ──
    meta = compute_encoding_metadata(E_new)

    row = {
        "class_name":                  class_name,
        "candidate_name":              candidate_name,
        "is_valid":                    True,
        "uses_old_codespace_only":     meta["uses_old_codespace_only"],
        "avg_codeword_entanglement":   meta["avg_codeword_entanglement"],
        "overlap_with_old_codespace":  meta["overlap_with_old_codespace"],
        "best_depth":                  None,
        "mean_depth":                  None,
        "std_depth":                   None,
        "best_size":                   None,
        "mean_size":                   None,
        "best_two_qubit_gate_count":   None,
        "mean_two_qubit_gate_count":   None,
        "num_qubits":                  None,
        "best_count_ops":              None,
        "n_transpile_runs":            n_transpile_runs,
        "successful_trials":           0,
        "failed_trials":               0,
        "status":                      "ok",
        "error_message":               "",
    }

    # ── walidacja ──
    if E_new is not None:
        vres = validate_encoding_map(E_new)
        if not vres["is_valid"]:
            row["is_valid"] = False
            row["status"] = "invalid_encoding"
            row["error_message"] = vres["message"]
            return row

    # ── budowa obwodu ──
    try:
        qc, _ = create_ame_circuit(
            n=n_qutrits, dim=3, graph_type='star', E_new=E_new,
        )
    except Exception:
        row["status"] = "build_error"
        row["error_message"] = traceback.format_exc()
        return row

    # ── transpilacja × n_transpile_runs — zbieramy pełne statystyki ──
    TWO_Q_GATES = {"cz", "rzz", "cx", "ecr", "swap"}
    depths = []
    sizes = []
    two_q_counts = []
    best = None  # (depth, two_q_count, size, ops_dict)

    for trial in range(n_transpile_runs):
        try:
            qc_t = transpile(
                qc,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=3,
                seed_transpiler=trial,
            )
            ops = qc_t.count_ops()
            depth = qc_t.depth()
            size = qc_t.size()
            two_q = sum(v for k, v in ops.items() if k in TWO_Q_GATES)

            depths.append(depth)
            sizes.append(size)
            two_q_counts.append(two_q)
            row["successful_trials"] += 1

            if best is None or (depth, two_q, size) < (best[0], best[1], best[2]):
                best = (depth, two_q, size, dict(ops), qc_t.num_qubits)

        except Exception:
            row["failed_trials"] += 1

    if row["successful_trials"] == 0:
        row["status"] = "all_transpile_failed"
        row["error_message"] = "Wszystkie próby transpilacji zakończyły się błędem."
        return row

    # ── statystyki ──
    row["best_depth"] = best[0]
    row["mean_depth"] = round(float(np.mean(depths)), 2)
    row["std_depth"] = round(float(np.std(depths)), 2)
    row["best_size"] = best[2]
    row["mean_size"] = round(float(np.mean(sizes)), 2)
    row["best_two_qubit_gate_count"] = best[1]
    row["mean_two_qubit_gate_count"] = round(float(np.mean(two_q_counts)), 2)
    row["num_qubits"] = best[4]
    row["best_count_ops"] = best[3]

    return row


# ══════════════════════════ MAIN ═════════════════════════════

def run_benchmark(n_qutrits=3, n_transpile_runs=20,
                   csv_path="benchmark_encoding_bases_results.csv",
                   mode="full"):
    """
    Uruchamia benchmark i zapisuje wyniki do CSV.

    mode:
        "full"      — wszystkie generatory (oryginalne + rozszerzone)
        "original"  — tylko oryginalne generatory (klasy 0–9)
        "extended"  — tylko nowe rozszerzone generatory (klasy 10–15)
    """

    print("=" * 80)
    print(f"  Benchmark baz kodowania qutrytu  (n_qutrits={n_qutrits})  [mode={mode}]")
    print(f"  Transpilacja: {n_transpile_runs} prób na kandydata (best + mean ± std)")
    print("=" * 80)

    all_candidates = []

    if mode in ("full", "original"):
        # ── generuj kandydatów — stara code space ──
        all_candidates += generate_baseline()
        all_candidates += generate_monomial_bases(max_candidates=120)
        all_candidates += generate_fourier_like_bases(max_candidates=80)
        all_candidates += generate_householder_bases(n_samples=20, seed=42)
        all_candidates += generate_clifford_wh_bases()

        # ── ogólne izometrie (pełna C⁴) ──
        all_candidates += generate_haar_random_isometries(n_samples=20, seed=100)
        all_candidates += generate_perturbed_isometries(n_samples_per_eps=8, seed=200)
        all_candidates += generate_entangling_isometries(n_samples=20, seed=300)
        all_candidates += generate_structured_entangling_isometries()

    n_orig = len(all_candidates)

    if mode in ("full", "extended"):
        # ── ROZSZERZONY BENCHMARK — nowe klasy (10–15) ──
        all_candidates += generate_local_ry_only(n_grid=10)
        all_candidates += generate_local_general_su2(n_samples=30, seed=600)
        all_candidates += generate_real_orthogonal_isometries(n_samples=20, seed=400)
        all_candidates += generate_near_identity_isometries(n_samples_per_eps=10, seed=500)
        all_candidates += generate_finer_structured_grid()
        all_candidates += generate_two_cz_ansatz(n_samples=50, seed=700)

    n_ext = len(all_candidates) - n_orig

    print(f"\n  Kandydaci (oryginalne):     {n_orig}")
    print(f"  Kandydaci (rozszerzone):    {n_ext}")
    print(f"  Razem:                      {len(all_candidates)}")
    print("-" * 80)

    # ── benchmark ──
    results = []
    t0 = time.time()

    for idx, (cls, name, E_new) in enumerate(all_candidates):
        print(
            f"  [{idx+1:4d}/{len(all_candidates)}]  {cls:28s}  {name:30s}",
            end="  ", flush=True,
        )

        row = benchmark_basis(
            E_new, cls, name,
            n_qutrits=n_qutrits,
            coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES,
            n_transpile_runs=n_transpile_runs,
        )

        if row["status"] == "ok":
            cs = "old" if row["uses_old_codespace_only"] else "NEW"
            print(
                f"best_d={row['best_depth']:5d}  "
                f"mean_d={row['mean_depth']:7.1f}  "
                f"best_2q={row['best_two_qubit_gate_count']:5d}  "
                f"ent={row['avg_codeword_entanglement']:.3f}  "
                f"[{cs}]"
            )
        else:
            print(f"[{row['status']}]")

        results.append(row)

    elapsed = time.time() - t0
    print(f"\nCzas benchmarku: {elapsed:.1f} s")

    # ── dataframe ──
    df = pd.DataFrame(results)

    # ── zapis CSV ──
    df.to_csv(csv_path, index=False)
    print(f"Wyniki zapisane do: {csv_path}")

    # ── top 15 ──
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("\nŻaden przypadek nie zakończył się sukcesem.")
        return df

    df_ok = df_ok.sort_values(
        by=["best_depth", "best_two_qubit_gate_count", "best_size"],
        ascending=True,
    )

    print("\n" + "=" * 80)
    print("  TOP 15 (najniższa best_depth)")
    print("=" * 80)
    top = df_ok.head(15)
    for i, (_, r) in enumerate(top.iterrows()):
        cs = "old" if r["uses_old_codespace_only"] else "NEW"
        print(
            f"  {i+1:2d}.  best_d={r['best_depth']:5d}  "
            f"mean_d={r['mean_depth']:7.1f}±{r['std_depth']:5.1f}  "
            f"best_2q={r['best_two_qubit_gate_count']:5d}  "
            f"ent={r['avg_codeword_entanglement']:.3f}  "
            f"ovlp={r['overlap_with_old_codespace']:.3f}  "
            f"[{cs}]  "
            f"{r['class_name']:28s}  {r['candidate_name']}"
        )

    # ── statystyki klas ──
    print("\n" + "=" * 80)
    print("  Statystyki wg klasy (best_depth)")
    print("=" * 80)
    stats = df_ok.groupby("class_name")["best_depth"].agg(
        ["count", "min", "mean", "max"]
    )
    print(stats.to_string())

    # ── old vs new code space ──
    print("\n" + "=" * 80)
    print("  Porównanie: stara code space vs ogólne izometrie")
    print("=" * 80)
    for label, mask in [("OLD codespace", df_ok["uses_old_codespace_only"] == True),
                        ("NEW (general)", df_ok["uses_old_codespace_only"] == False)]:
        sub = df_ok[mask]
        if sub.empty:
            print(f"  {label:18s}  brak wyników")
        else:
            print(
                f"  {label:18s}  n={len(sub):4d}  "
                f"best_depth: min={sub['best_depth'].min():5d}  "
                f"mean={sub['best_depth'].mean():7.1f}  "
                f"best_2q: min={sub['best_two_qubit_gate_count'].min():5d}  "
                f"mean={sub['best_two_qubit_gate_count'].mean():7.1f}"
            )

    return df


if __name__ == "__main__":
    import sys

    # Domyślnie: tylko nowe klasy (extended); "full" lub "original" przez argument
    _mode = sys.argv[1] if len(sys.argv) > 1 else "extended"
    _csv = {
        "full":     "benchmark_encoding_bases_full_results.csv",
        "original": "benchmark_encoding_bases_results.csv",
        "extended": "benchmark_encoding_bases_extended_results.csv",
    }.get(_mode, f"benchmark_{_mode}_results.csv")

    run_benchmark(n_qutrits=3, mode=_mode, csv_path=_csv)
