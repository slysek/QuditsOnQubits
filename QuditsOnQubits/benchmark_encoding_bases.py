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

import os
import numpy as np
import pandas as pd
import traceback
import time
from functools import lru_cache
from itertools import combinations, product as iter_product

from igraph import Graph

from qiskit import qpy, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from QuditsOnQubits import create_ame_circuit
from QuditsOnQubits.create_ame_circuit import VALID_ENCODING_STRATEGIES
from QuditsOnQubits.project_paths import (
    benchmark_circuits_dir,
    benchmark_docs_dir,
    benchmark_results_path,
    benchmark_state_circuits_dir,
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
    prepared_w_benchmark_data_dir,
    prepared_w_benchmark_results_path,
)
from encoding_change_unitary import build_encoding_change_unitary, validate_encoding_map

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
DEFAULT_APPROXIMATION_VALUES = tuple(round(x, 2) for x in np.linspace(0.90, 0.99, 10))
DEFAULT_FIDELITY_THRESHOLDS = (0.85, 0.90, 0.95)
TWO_Q_GATES = {"cz", "rzz", "cx", "ecr", "swap"}
_DEFAULT_CIRCUITS_OUTPUT_DIR = object()
_EXPORTED_TRANSPILED_CIRCUIT_COUNT = 3

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

_GHZ_STAR_STATE_PREFIXES = ("ghz_star_", "ghz_n_")


def _parse_ghz_star_n_from_name(state_name):
    for prefix in _GHZ_STAR_STATE_PREFIXES:
        if state_name.startswith(prefix):
            suffix = state_name[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
    return None


def _validate_star_n(state_name, n_qutrits):
    n = int(n_qutrits)
    if n < 2:
        raise ValueError(f"{state_name} requires n_qutrits >= 2, got {n}.")
    return n


def _resolve_star_graph_n(state_name, n_qutrits=None):
    """Return n for states represented by a qutrit star graph, else None."""
    if state_name == "two_qutrit":
        if n_qutrits is not None and int(n_qutrits) != 2:
            raise ValueError(f"two_qutrit has fixed n_qutrits=2, got {n_qutrits}.")
        return 2
    if state_name == "ghz3":
        if n_qutrits is not None and int(n_qutrits) != 3:
            raise ValueError(f"ghz3 has fixed n_qutrits=3, got {n_qutrits}.")
        return 3
    if state_name in ("ghz_star", "ghz_n"):
        if n_qutrits is None:
            raise ValueError(f"{state_name} requires n_qutrits.")
        return _validate_star_n(state_name, n_qutrits)

    n_from_name = _parse_ghz_star_n_from_name(state_name)
    if n_from_name is not None:
        if n_qutrits is not None and int(n_qutrits) != n_from_name:
            raise ValueError(
                f"{state_name} already encodes n_qutrits={n_from_name}, got {n_qutrits}."
            )
        return _validate_star_n(state_name, n_from_name)

    return None


def _normalize_state_name(state_name, n_qutrits=None):
    """Return the stable benchmark state id used in result rows and paths."""
    n = _resolve_star_graph_n(state_name, n_qutrits)
    if state_name in ("ghz_star", "ghz_n"):
        return f"ghz_star_{n}"
    if _parse_ghz_star_n_from_name(state_name) is not None:
        return f"ghz_star_{n}"
    return state_name


def _state_family(state_name):
    if state_name == "ghz3" or _parse_ghz_star_n_from_name(state_name) is not None:
        return "ghz_star"
    return state_name


def _state_num_qutrits(state_name, n_qutrits=None):
    star_n = _resolve_star_graph_n(state_name, n_qutrits)
    if star_n is not None:
        return star_n
    if state_name == "ame43":
        return 4
    return n_qutrits


@lru_cache(maxsize=None)
def _get_state_graph(state_name, n_qutrits=None):
    """Return a cached graph object for a named benchmark state."""
    star_n = _resolve_star_graph_n(state_name, n_qutrits)
    if star_n is not None:
        return Graph(n=star_n, edges=[[0, i] for i in range(1, star_n)])
    if state_name == "ame43":
        return Graph(n=4, edges=[[0, 1], [0, 1], [1, 2], [2, 3], [3, 0]])
    raise ValueError(f"Unknown benchmark state: {state_name}")


def _get_ame43_graph():
    """Return the cached graph used for AME(4,3) benchmarks."""
    return _get_state_graph("ame43")


@lru_cache(maxsize=None)
def _get_cached_approximation_pass_manager(
    basis_gates,
    approximation_degree,
    seed_transpiler,
):
    """Reuse deterministic pass-manager instances across candidates."""
    return generate_preset_pass_manager(
        basis_gates=list(basis_gates),
        optimization_level=3,
        approximation_degree=approximation_degree,
        seed_transpiler=seed_transpiler,
    )


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


def _phase_label(phases):
    """Zakoduj fazy z {1, ω, ω²} jako cyfry 0/1/2 do nazwy kandydata."""
    return "".join(
        str(int(round(np.angle(phase) * 3 / (2 * np.pi))) % 3)
        for phase in phases
    )


# ────────────── metryki kodowania ──────────────

def _single_qubit_product_library():
    """Small discrete library of 1-qubit unitaries for product-basis candidates."""
    identity = np.eye(2, dtype=complex)
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    z_gate = np.diag([1, -1]).astype(complex)
    s_gate = np.diag([1, 1j]).astype(complex)
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    sx_gate = 0.5 * np.array(
        [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
        dtype=complex,
    )

    return [
        ("I", identity),
        ("X", x_gate),
        ("SX", sx_gate),
        ("SXdg", sx_gate.conj().T),
        ("H", hadamard),
        ("HS", hadamard @ s_gate),
        ("SH", s_gate @ hadamard),
        ("Z", z_gate),
    ]


def _single_qubit_product_angle_grid(angle_grid=None):
    """Resolve the phase/polar angle lists used by the optional SU(2) grid."""
    default_phase_angles = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    default_polar_angles = (0.0, np.pi / 2, np.pi)

    if angle_grid is None:
        return default_phase_angles, default_polar_angles

    if isinstance(angle_grid, dict):
        def _pick_angle_values(keys):
            for key in keys:
                if key in angle_grid and angle_grid[key] is not None:
                    return angle_grid[key]
            return None

        phase_angles = _pick_angle_values(
            ("phase_angles", "phases", "alpha_gamma", "rz")
        )
        polar_angles = _pick_angle_values(
            ("polar_angles", "betas", "beta", "rx")
        )
    else:
        try:
            phase_angles, polar_angles = angle_grid
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "angle_grid must be None, a dict, or a pair "
                "(phase_angles, polar_angles)."
            ) from exc

    if phase_angles is None or polar_angles is None:
        raise ValueError(
            "angle_grid must define both phase angles and polar angles."
        )

    return tuple(phase_angles), tuple(polar_angles)


def _single_qubit_rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _single_qubit_rz(theta):
    half_theta = theta / 2
    return np.diag(
        [np.exp(-1j * half_theta), np.exp(1j * half_theta)]
    ).astype(complex)


def _single_qubit_su2(alpha, beta, gamma):
    return _single_qubit_rz(alpha) @ _single_qubit_rx(beta) @ _single_qubit_rz(gamma)


def _format_product_angle(theta):
    if np.isclose(theta, 0.0, atol=1e-12):
        theta = 0.0
    return f"{theta:.2f}"


def _single_qubit_product_grid(angle_grid=None):
    """Return a named SU(2) grid based on the Rz-Rx-Rz parametrization."""
    phase_angles, polar_angles = _single_qubit_product_angle_grid(angle_grid)
    grid = []

    for alpha in phase_angles:
        for beta in polar_angles:
            for gamma in phase_angles:
                name = (
                    f"a{_format_product_angle(alpha)}"
                    f"_b{_format_product_angle(beta)}"
                    f"_g{_format_product_angle(gamma)}"
                )
                grid.append((name, _single_qubit_su2(alpha, beta, gamma)))

    return grid


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


def _benchmark_circuit_output_path(class_name, candidate_name, output_root=None, suffix=None):
    """Return a candidate-scoped output path inside the class-name folder."""
    if output_root is None:
        output_root = benchmark_circuits_dir()

    class_dir = os.path.join(output_root, class_name)
    os.makedirs(class_dir, exist_ok=True)

    stem = candidate_name if suffix is None else f"{candidate_name}__{suffix}"
    return os.path.join(class_dir, f"{stem}.qpy")


def _save_benchmark_circuit(qc, class_name, candidate_name, output_root=None, suffix=None):
    """Save a benchmark-related circuit under its class-name folder."""
    output_path = _benchmark_circuit_output_path(
        class_name,
        candidate_name,
        output_root=output_root,
        suffix=suffix,
    )
    with open(output_path, "wb") as fd:
        qpy.dump(qc, fd)

    return output_path


def _build_encoding_change_circuit(E_new):
    """Build a standalone 2-qubit circuit containing only the encoding-change W gate."""
    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W ma wymiar {W.shape}, oczekiwano (4, 4)"

    W_gate = UnitaryGate(W, label="W")
    W_gate.name = "W"

    qc = QuantumCircuit(2, name="W")
    qc.append(W_gate, [0, 1])
    return qc


def _build_state_circuit(state_name, E_new, encoding_strategy="append_w", n_qutrits=None):
    """Build the base circuit for the given benchmark state."""
    star_n = _resolve_star_graph_n(state_name, n_qutrits)
    if star_n is not None:
        return create_ame_circuit(n=star_n, dim=3, graph_type="star", E_new=E_new,
                                  encoding_strategy=encoding_strategy)
    if state_name == "ame43":
        return create_ame_circuit(dim=3, graph=_get_ame43_graph(), E_new=E_new,
                                  encoding_strategy=encoding_strategy)
    raise ValueError(f"Unknown benchmark state: {state_name}")


def _resolve_circuits_output_dir(state_name, circuits_output_dir):
    """Return the correct circuits output directory for a given state."""
    if circuits_output_dir is _DEFAULT_CIRCUITS_OUTPUT_DIR:
        return benchmark_state_circuits_dir(state_name)
    if circuits_output_dir is None:
        return None
    return os.path.join(circuits_output_dir, state_name)


def _count_two_qubit_gates(ops):
    """Count two-qubit gates from a count_ops dictionary."""
    return int(sum(v for k, v in ops.items() if k in TWO_Q_GATES))


def _strip_idle_qubits(qc):
    """Remove idle qubits so state construction only uses active wires."""
    qc_state = qc.remove_final_measurements(inplace=False)
    dag = circuit_to_dag(qc_state)
    idle_qubits = [wire for wire in dag.idle_wires() if wire in dag.qubits]
    if idle_qubits:
        dag.remove_qubits(*idle_qubits)
    return dag_to_circuit(dag)


def _fidelity_label(threshold):
    return f"fid{int(round(float(threshold) * 100)):03d}"


def _make_approximation_result_fields(fidelity_thresholds):
    fields = {
        "approx_ref_depth": None,
        "approx_ref_two_qubit_gate_count": None,
        "approx_status": "not_run",
        "approx_error_message": "",
    }

    for threshold in fidelity_thresholds:
        label = _fidelity_label(threshold)
        fields[f"{label}_best_approx_degree"] = None
        fields[f"{label}_best_fidelity"] = None
        fields[f"{label}_best_depth"] = None
        fields[f"{label}_best_two_qubit_gate_count"] = None

    return fields


def _benchmark_approximation_sweep(
    qc,
    basis_gates=None,
    coupling_map=None,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    seed_transpiler=0,
    keep_circuits=False,
):
    """Benchmark approximation_degree sweep against a max-fidelity reference."""
    if basis_gates is None:
        basis_gates = BASIS_GATES
    if approximation_values is None:
        approximation_values = DEFAULT_APPROXIMATION_VALUES
    if fidelity_thresholds is None:
        fidelity_thresholds = DEFAULT_FIDELITY_THRESHOLDS

    fidelity_thresholds = tuple(float(t) for t in fidelity_thresholds)
    approximation_values = tuple(float(a) for a in approximation_values)
    basis_gates = tuple(basis_gates)

    result = _make_approximation_result_fields(fidelity_thresholds)
    result["approx_status"] = "ok"

    pm_ref = _get_cached_approximation_pass_manager(
        basis_gates=basis_gates,
        approximation_degree=1.0,
        seed_transpiler=seed_transpiler,
    )
    qcmax = pm_ref.run(qc)
    ref_ops = qcmax.count_ops()
    result["approx_ref_depth"] = qcmax.depth()
    result["approx_ref_two_qubit_gate_count"] = _count_two_qubit_gates(ref_ops)
    rho_max = DensityMatrix(_strip_idle_qubits(qcmax))

    best_hits = {threshold: None for threshold in fidelity_thresholds}

    for approx_degree in approximation_values:
        pm = _get_cached_approximation_pass_manager(
            basis_gates=basis_gates,
            approximation_degree=approx_degree,
            seed_transpiler=seed_transpiler,
        )
        qctest = pm.run(qc)
        ops = qctest.count_ops()
        depth = qctest.depth()
        two_q = _count_two_qubit_gates(ops)
        fidelity = float(state_fidelity(DensityMatrix(_strip_idle_qubits(qctest)), rho_max))

        for threshold in fidelity_thresholds:
            if fidelity < threshold:
                continue

            best = best_hits[threshold]
            candidate = {
                "approx_degree": round(float(approx_degree), 2),
                "fidelity": round(fidelity, 12),
                "depth": depth,
                "two_q": two_q,
            }
            if keep_circuits:
                candidate["qc"] = qctest
            if best is None or (depth, two_q) < (best["depth"], best["two_q"]):
                best_hits[threshold] = candidate

    for threshold, best in best_hits.items():
        if best is None:
            continue

        label = _fidelity_label(threshold)
        result[f"{label}_best_approx_degree"] = best["approx_degree"]
        result[f"{label}_best_fidelity"] = best["fidelity"]
        result[f"{label}_best_depth"] = best["depth"]
        result[f"{label}_best_two_qubit_gate_count"] = best["two_q"]
        if keep_circuits and "qc" in best:
            result[f"_{label}_best_qc"] = best["qc"]

    return result


# ════════════════ GENERATORY BAZ — STARA CODE SPACE ═════════

def generate_baseline():
    """Klasa 0: baseline – samo E_old."""
    return [("baseline", "E_old", None)]  # E_new=None → domyślne kodowanie


def generate_monomial_old_codespace_bases(max_candidates=500):
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
            name = f"P{''.join(map(str, perm))}_ph{_phase_label(ph)}"
            candidates.append(("monomial_old_codespace", name, E_new))
            if max_candidates is not None and len(candidates) >= max_candidates:
                return candidates
    return candidates


def generate_monomial_bases(max_candidates=500):
    """Backward-compatible wrapper dla monomiali ograniczonych do E_old."""
    return generate_monomial_old_codespace_bases(max_candidates=max_candidates)


def generate_monomial_full_bases(max_candidates=500):
    """
    Pełna klasa monomialnych embeddingów qutrytu w 2 qubity.

    Wybiera dowolny support 3 z 4 stanów bazowych |00>, |01>, |10>, |11>,
    a następnie stosuje tę samą konwencję D @ P co w starej code space.
    """
    computational_basis = np.eye(4, dtype=complex)
    candidates = []

    for support in combinations(range(4), 3):
        B = computational_basis[:, support]
        support_label = "".join(map(str, support))
        for perm in _PERMS_3:
            P = _perm_matrix(perm)
            for ph in iter_product(_PHASES_3, repeat=3):
                D = _phase_diag(ph)
                S = D @ P
                E_new = B @ S
                name = f"sup{support_label}_P{''.join(map(str, perm))}_ph{_phase_label(ph)}"
                candidates.append(("monomial_full", name, E_new))
                if max_candidates is not None and len(candidates) >= max_candidates:
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

def generate_product_bases(
    max_candidates=500,
    e_base=None,
    mode="discrete",
    include_grid=False,
    angle_grid=None,
):
    """
    Klasa "product": E_new = (U ⊗ V) @ E_base, gdzie U i V są lokalne 1-qubit unitary.

    mode="discrete" używa małej, skończonej biblioteki bramek 1-qubitowych.
    mode="grid" używa prostego gridu parametrów SU(2) w postaci Rz-Rx-Rz.
    Jeśli include_grid=True przy mode="discrete", kandydaci gridowi są dopinani
    po skończonej bibliotece dyskretnej.
    """
    if mode not in {"discrete", "grid"}:
        raise ValueError("mode must be either 'discrete' or 'grid'.")

    e_base_arr = get_E_old() if e_base is None else np.array(e_base, dtype=complex, copy=True)
    vres = validate_encoding_map(e_base_arr)
    if not vres["is_valid"]:
        raise ValueError(f"Invalid e_base for generate_product_bases: {vres['message']}")

    candidates = []

    def _append_candidates(unitary_library, name_prefix=""):
        for (u_name, u_gate), (v_name, v_gate) in iter_product(unitary_library, repeat=2):
            e_new = np.kron(u_gate, v_gate) @ e_base_arr
            candidate_name = f"{name_prefix}U_{u_name}__V_{v_name}"
            candidates.append(("product", candidate_name, e_new))
            if max_candidates is not None and len(candidates) >= max_candidates:
                return True
        return False

    if mode == "discrete":
        if _append_candidates(_single_qubit_product_library()):
            return candidates
        if not include_grid:
            return candidates

    if _append_candidates(_single_qubit_product_grid(angle_grid), name_prefix="grid__"):
        return candidates

    return candidates


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
    Klasa 11: W = U₁ ⊗ U₂  gdzie U₁, U₂ ∈ SU(2) losowe.
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
                    state_name="ghz3",
                    n_qutrits=None, coupling_map=None, basis_gates=None,
                    n_transpile_runs=20, circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
                    approximation_values=None,
                    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
                    approximation_seed=0,
                    encoding_strategy="append_w"):
    """
    Buduje obwód, transpiluje n_transpile_runs razy.
    Zbiera pełne statystyki: best, mean, std.

    Zwraca dict z wynikami.
    """
    if coupling_map is None:
        coupling_map = COUPLING_MAP
    if basis_gates is None:
        basis_gates = BASIS_GATES
    if fidelity_thresholds is None:
        fidelity_thresholds = DEFAULT_FIDELITY_THRESHOLDS

    resolved_n_qutrits = _state_num_qutrits(state_name, n_qutrits)
    state_name = _normalize_state_name(state_name, resolved_n_qutrits)
    circuits_output_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)

    # ── metryki kodowania ──
    meta = compute_encoding_metadata(E_new)

    row = {
        "state_name":                  state_name,
        "state_family":                _state_family(state_name),
        "n_qutrits":                   resolved_n_qutrits,
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
    row.update(_make_approximation_result_fields(fidelity_thresholds))

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
        qc, _ = _build_state_circuit(state_name, E_new=E_new,
                                     encoding_strategy=encoding_strategy,
                                     n_qutrits=resolved_n_qutrits)
        if circuits_output_dir is not None:
            _save_benchmark_circuit(
                qc,
                class_name=class_name,
                candidate_name=candidate_name,
                output_root=circuits_output_dir,
            )
            if E_new is not None:
                _save_benchmark_circuit(
                    _build_encoding_change_circuit(E_new),
                    class_name=class_name,
                    candidate_name=candidate_name,
                    output_root=circuits_output_dir,
                    suffix="W",
                )
    except Exception:
        row["status"] = "build_error"
        row["error_message"] = traceback.format_exc()
        return row

    try:
        row.update(
            _benchmark_approximation_sweep(
                qc,
                basis_gates=basis_gates,
                approximation_values=approximation_values,
                fidelity_thresholds=fidelity_thresholds,
                seed_transpiler=approximation_seed,
                keep_circuits=(circuits_output_dir is not None),
            )
        )
    except Exception:
        row["approx_status"] = "error"
        row["approx_error_message"] = traceback.format_exc()

    # ── transpilacja × n_transpile_runs — zbieramy pełne statystyki ──
    depths = []
    sizes = []
    two_q_counts = []
    successful_circuits = []

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
            two_q = _count_two_qubit_gates(ops)

            depths.append(depth)
            sizes.append(size)
            two_q_counts.append(two_q)
            row["successful_trials"] += 1

            successful_circuits.append(
                {
                    "rank_key": (depth, two_q, size),
                    "depth": depth,
                    "two_q": two_q,
                    "size": size,
                    "ops": dict(ops),
                    "num_qubits": qc_t.num_qubits,
                    "qc": qc_t,
                }
            )

        except Exception:
            row["failed_trials"] += 1

    if row["successful_trials"] == 0:
        row["status"] = "all_transpile_failed"
        row["error_message"] = "Wszystkie próby transpilacji zakończyły się błędem."
        return row

    # ── statystyki ──
    ranked_circuits = sorted(successful_circuits, key=lambda item: item["rank_key"])
    best = ranked_circuits[0]

    if circuits_output_dir is not None:
        try:
            for rank, circuit_data in enumerate(
                ranked_circuits[:_EXPORTED_TRANSPILED_CIRCUIT_COUNT],
                start=1,
            ):
                _save_benchmark_circuit(
                    circuit_data["qc"],
                    class_name=class_name,
                    candidate_name=candidate_name,
                    output_root=circuits_output_dir,
                    suffix=f"transpiled_{rank}",
                )
        except Exception:
            row["status"] = "export_error"
            row["error_message"] = traceback.format_exc()
            return row

    row["best_depth"] = best["depth"]
    row["mean_depth"] = round(float(np.mean(depths)), 2)
    row["std_depth"] = round(float(np.std(depths)), 2)
    row["best_size"] = best["size"]
    row["mean_size"] = round(float(np.mean(sizes)), 2)
    row["best_two_qubit_gate_count"] = best["two_q"]
    row["mean_two_qubit_gate_count"] = round(float(np.mean(two_q_counts)), 2)
    row["num_qubits"] = best["num_qubits"]
    row["best_count_ops"] = best["ops"]

    return row


# ══════════════════════════ MARKDOWN REPORT ═══════════════════

def _markdown_table(headers, rows):
    """Build a markdown table from headers and row data."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_fidelity_cell(row, prefix, suffix):
    value = row.get(f"{prefix}_{suffix}")
    return "brak" if pd.isna(value) else value


def write_multi_state_benchmark_report(state_frames, output_path):
    """Generate a combined markdown report from per-state DataFrames."""
    lines = ["# Multi-State Encoding Benchmark Analysis", ""]
    comparison_rows = []

    for state_name in ("two_qutrit", "ghz3", "ame43"):
        df = state_frames[state_name]
        df_ok = df[df["status"] == "ok"].copy()
        top = df_ok.sort_values(
            by=["best_depth", "best_two_qubit_gate_count", "best_size"],
            ascending=True,
        ).head(10)
        per_class = (
            df_ok.sort_values(
                by=["best_depth", "best_two_qubit_gate_count", "best_size"],
                ascending=True,
            )
            .groupby("class_name", as_index=False)
            .first()
        )

        lines.extend([
            f"## {state_name}",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Rows", len(df)],
                    ["Successful rows", int((df["status"] == "ok").sum())],
                    ["Failed rows", int((df["status"] != "ok").sum())],
                ],
            ),
            "",
            "### Top 10 candidates",
            "",
            _markdown_table(
                ["Class", "Candidate", "best_depth", "best_2q", "mean_depth"],
                [
                    [
                        row["class_name"],
                        row["candidate_name"],
                        row["best_depth"],
                        row["best_two_qubit_gate_count"],
                        row["mean_depth"],
                    ]
                    for _, row in top.iterrows()
                ],
            ),
            "",
            "### Best per class",
            "",
            _markdown_table(
                ["Class", "Candidate", "best_depth", "best_2q", "mean_depth"],
                [
                    [
                        row["class_name"],
                        row["candidate_name"],
                        row["best_depth"],
                        row["best_two_qubit_gate_count"],
                        row["mean_depth"],
                    ]
                    for _, row in per_class.iterrows()
                ],
            ),
            "",
            "### Fidelity thresholds",
            "",
            _markdown_table(
                [
                    "Class",
                    "Candidate",
                    "fid085 approx",
                    "fid085 fidelity",
                    "fid085 depth",
                    "fid085 2Q",
                    "fid090 approx",
                    "fid090 fidelity",
                    "fid090 depth",
                    "fid090 2Q",
                    "fid095 approx",
                    "fid095 fidelity",
                    "fid095 depth",
                    "fid095 2Q",
                ],
                [
                    [
                        row["class_name"],
                        row["candidate_name"],
                        _format_fidelity_cell(row, "fid085", "best_approx_degree"),
                        _format_fidelity_cell(row, "fid085", "best_fidelity"),
                        _format_fidelity_cell(row, "fid085", "best_depth"),
                        _format_fidelity_cell(row, "fid085", "best_two_qubit_gate_count"),
                        _format_fidelity_cell(row, "fid090", "best_approx_degree"),
                        _format_fidelity_cell(row, "fid090", "best_fidelity"),
                        _format_fidelity_cell(row, "fid090", "best_depth"),
                        _format_fidelity_cell(row, "fid090", "best_two_qubit_gate_count"),
                        _format_fidelity_cell(row, "fid095", "best_approx_degree"),
                        _format_fidelity_cell(row, "fid095", "best_fidelity"),
                        _format_fidelity_cell(row, "fid095", "best_depth"),
                        _format_fidelity_cell(row, "fid095", "best_two_qubit_gate_count"),
                    ]
                    for _, row in per_class.iterrows()
                ],
            ),
            "",
        ])

        comparison_rows.extend([
            [
                state_name,
                row["class_name"],
                row["candidate_name"],
                row["best_depth"],
                row["best_two_qubit_gate_count"],
                _format_fidelity_cell(row, "fid095", "best_depth"),
            ]
            for _, row in per_class.iterrows()
        ])

    lines.extend([
        "## Cross-state comparison",
        "",
        _markdown_table(
            ["State", "Class", "Candidate", "best_depth", "best_2q", "fid095 depth"],
            comparison_rows,
        ),
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fd:
        fd.write("\n".join(lines).strip() + "\n")


# ══════════════════ TOP-3 FIDELITY CIRCUITS ═══════════════════

def _save_top3_fidelity_circuits(fidelity_circuits, output_root):
    """Save top-3-per-class transpiled circuits for each fidelity threshold."""
    from collections import defaultdict

    by_threshold = defaultdict(list)
    for entry in fidelity_circuits:
        by_threshold[entry["label"]].append(entry)

    saved = 0
    for label, entries in sorted(by_threshold.items()):
        by_class = defaultdict(list)
        for e in entries:
            by_class[e["class_name"]].append(e)

        for class_name, class_entries in sorted(by_class.items()):
            ranked = sorted(
                class_entries,
                key=lambda e: (
                    e["two_q"] if e["two_q"] is not None else float("inf"),
                    e["depth"] if e["depth"] is not None else float("inf"),
                ),
            )
            for rank, entry in enumerate(ranked[:3], start=1):
                try:
                    _save_benchmark_circuit(
                        entry["qc"],
                        class_name=class_name,
                        candidate_name=entry["candidate_name"],
                        output_root=output_root,
                        suffix=f"{label}_rank{rank}",
                    )
                    saved += 1
                except Exception:
                    pass

    if saved:
        print(f"  Zapisano {saved} obwodów fidelity → {output_root}")


# ══════════════════════ TOP-3 PER CLASS CSV ═══════════════════

def _save_top3_per_class_csvs(df, csv_path, fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS):
    """Generate top-3-per-class summary CSVs next to the main results file.

    Produces:
      *_top3_exact.csv           — sorted by (best_two_qubit_gate_count, best_depth)
      *_top3_fid{NNN}.csv        — one per threshold, sorted by (fidNNN 2Q, fidNNN depth)
    """
    if df.empty or "status" not in df.columns:
        return

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return

    base, ext = os.path.splitext(csv_path)

    # ── exact transpilation ──
    parts = []
    for _, group in df_ok.groupby("class_name", sort=True):
        top = group.sort_values(
            by=["best_two_qubit_gate_count", "best_depth"],
            ascending=True,
        ).head(3)
        parts.append(top)

    if parts:
        exact_df = pd.concat(parts, ignore_index=True)[
            ["class_name", "candidate_name", "best_depth",
             "best_size", "best_two_qubit_gate_count"]
        ].sort_values(
            by=["class_name", "best_two_qubit_gate_count", "best_depth"],
            ascending=True,
        )
        exact_path = f"{base}_top3_exact{ext}"
        exact_df.to_csv(exact_path, index=False)
        print(f"  Top-3 exact → {exact_path}")

    # ── fidelity thresholds ──
    for threshold in fidelity_thresholds:
        label = _fidelity_label(threshold)
        depth_col = f"{label}_best_depth"
        twoq_col = f"{label}_best_two_qubit_gate_count"
        fid_col = f"{label}_best_fidelity"

        cols_needed = [depth_col, twoq_col]
        if not all(c in df_ok.columns for c in cols_needed):
            continue

        df_fid = df_ok.dropna(subset=cols_needed)
        if df_fid.empty:
            continue

        parts = []
        for _, group in df_fid.groupby("class_name", sort=True):
            top = group.sort_values(
                by=[twoq_col, depth_col],
                ascending=True,
            ).head(3)
            parts.append(top)

        if not parts:
            continue

        keep_cols = ["class_name", "candidate_name"]
        if fid_col in df_fid.columns:
            keep_cols.append(fid_col)
        keep_cols.extend([depth_col, twoq_col])

        fid_df = pd.concat(parts, ignore_index=True)[keep_cols].sort_values(
            by=["class_name", twoq_col, depth_col],
            ascending=True,
        )
        fid_path = f"{base}_top3_{label}{ext}"
        fid_df.to_csv(fid_path, index=False)
        pct = int(round(float(threshold) * 100))
        print(f"  Top-3 fidelity>={pct}% → {fid_path}")


def _write_topk_tables_to_output_dir(df, output_dir, file_prefix,
                                     fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS):
    """Write top-3-per-class CSVs to a given output directory.

    Produces the same set of files as _save_top3_per_class_csvs but writes
    into an explicitly chosen directory with a configurable file prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{file_prefix}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Wyniki → {csv_path}")
    _save_top3_per_class_csvs(df, csv_path, fidelity_thresholds=fidelity_thresholds)


# ═══════════════ PRESELECTED CANDIDATES (second stage) ═══════

def _load_preselected_candidates(csv_path):
    """Load a CSV file with preselected candidates and return a set of
    (class_name, candidate_name) tuples.

    The CSV must contain at least 'class_name' and 'candidate_name' columns.
    Column values are stripped of leading/trailing whitespace.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Plik z preselekcja kandydatow nie istnieje: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for required_col in ("class_name", "candidate_name"):
        if required_col not in df.columns:
            raise ValueError(
                f"Plik preselekcji {csv_path!r} nie zawiera kolumny "
                f"'{required_col}'. Dostepne kolumny: {list(df.columns)}"
            )

    df["class_name"] = df["class_name"].astype(str).str.strip()
    df["candidate_name"] = df["candidate_name"].astype(str).str.strip()

    return set(zip(df["class_name"], df["candidate_name"]))


def _filter_candidates_by_preselection(all_candidates, preselected_set):
    """Filter candidate triples (class_name, name, E_new) to only those
    present in the preselected set of (class_name, candidate_name)."""
    filtered = [
        (cls, name, e_new)
        for cls, name, e_new in all_candidates
        if (cls, name) in preselected_set
    ]
    return filtered


def _validate_preselection_coverage(preselected_set, filtered_candidates, csv_path):
    """Warn if some preselected candidates were not found in the generator."""
    found = {(cls, name) for cls, name, _ in filtered_candidates}
    missing = preselected_set - found
    if missing:
        import warnings
        msg = (
            f"Nastepujace kandydaty z pliku preselekcji ({csv_path}) "
            f"nie zostaly znalezione w aktualnym generatorze:\n"
        )
        for cls, name in sorted(missing):
            msg += f"  - class_name={cls!r}, candidate_name={name!r}\n"
        msg += (
            "Upewnij sie, ze plik preselekcji odpowiada temu samemu "
            "stanowi / eksperymentowi i trybowi generowania kandydatow."
        )
        warnings.warn(msg, stacklevel=2)


def _run_prepared_w_benchmark(
    state_name="ghz3",
    n_qutrits=None,
    preselected_candidates_file=None,
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    class_filter=None,
    output_dir=None,
):
    """Run the second-stage benchmark using 'prepared_w_then_conjugated_entanglers'.

    This must be given a preselected_candidates_file (CSV from the first-stage
    append_w benchmark) so that only the best candidates are re-evaluated
    with the more expensive conjugated-entangler architecture.

    WARNING: The preselected_candidates_file must correspond to the same
    state_name / experiment.  This function does NOT assume that a file
    generated for GHZ3 is valid for AME(4,3) or any other state.
    """
    resolved_n_qutrits = _state_num_qutrits(state_name, n_qutrits)
    state_name = _normalize_state_name(state_name, resolved_n_qutrits)

    if preselected_candidates_file is None:
        raise ValueError(
            "Tryb 'prepared_w_then_conjugated_entanglers' wymaga podania "
            "--preselected-candidates-file z wynikami pierwszego etapu (append_w)."
        )

    preselected_set = _load_preselected_candidates(preselected_candidates_file)
    print(f"  Zaladowano {len(preselected_set)} preselekcjonowanych kandydatow "
          f"z: {preselected_candidates_file}")

    filter_set = None
    if class_filter is not None:
        if isinstance(class_filter, str):
            filter_set = {c.strip() for c in class_filter.split(",")}
        else:
            filter_set = set(class_filter)

    filter_label = ",".join(sorted(filter_set)) if filter_set else "all"
    print("=" * 80)
    print(f"  Benchmark [prepared_w_then_conjugated_entanglers]  "
          f"[state={state_name}]  [mode={mode}]  [class={filter_label}]")
    print(f"  Transpilacja: {n_transpile_runs} prób na kandydata (best + mean ± std)")
    print("=" * 80)

    all_candidates = []
    if mode in ("full", "original"):
        all_candidates += generate_baseline()
        all_candidates += generate_monomial_old_codespace_bases(max_candidates=None)
        all_candidates += generate_monomial_full_bases(max_candidates=None)
        all_candidates += generate_fourier_like_bases(max_candidates=80)
        all_candidates += generate_householder_bases(n_samples=20, seed=42)
        all_candidates += generate_clifford_wh_bases()
        all_candidates += generate_haar_random_isometries(n_samples=20, seed=100)
        all_candidates += generate_perturbed_isometries(n_samples_per_eps=8, seed=200)
        all_candidates += generate_entangling_isometries(n_samples=20, seed=300)
        all_candidates += generate_structured_entangling_isometries()

    if mode in ("full", "extended"):
        all_candidates += generate_product_bases(max_candidates=None)
        all_candidates += generate_local_ry_only(n_grid=10)
        all_candidates += generate_local_general_su2(n_samples=30, seed=600)
        all_candidates += generate_real_orthogonal_isometries(n_samples=20, seed=400)
        all_candidates += generate_near_identity_isometries(n_samples_per_eps=10, seed=500)
        all_candidates += generate_finer_structured_grid()
        all_candidates += generate_two_cz_ansatz(n_samples=50, seed=700)

    if filter_set:
        all_candidates = [(cls, n, e) for cls, n, e in all_candidates if cls in filter_set]

    filtered = _filter_candidates_by_preselection(all_candidates, preselected_set)
    _validate_preselection_coverage(preselected_set, filtered, preselected_candidates_file)

    if not filtered:
        print("  UWAGA: Zaden kandydat z preselekcji nie zostal znaleziony "
              "w aktualnym generatorze. Przerywam.")
        return pd.DataFrame(), None

    print(f"\n  Kandydaci (wygenerowani):     {len(all_candidates)}")
    print(f"  Kandydaci (po preselekcji):   {len(filtered)}")
    print("-" * 80)

    results = []
    fidelity_circuits = []
    t0 = time.time()

    resolved_circuits_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)

    for idx, (cls, name, E_new) in enumerate(filtered):
        print(
            f"  [{idx+1:4d}/{len(filtered)}]  {cls:28s}  {name:30s}",
            end="  ", flush=True,
        )

        row = benchmark_basis(
            E_new, cls, name,
            state_name=state_name,
            n_qutrits=resolved_n_qutrits,
            coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES,
            n_transpile_runs=n_transpile_runs,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            encoding_strategy="prepared_w_then_conjugated_entanglers",
        )

        for key in list(row.keys()):
            if key.startswith("_fid") and key.endswith("_best_qc"):
                label = key[1:].replace("_best_qc", "")
                fidelity_circuits.append({
                    "class_name": cls,
                    "candidate_name": name,
                    "label": label,
                    "two_q": row.get(f"{label}_best_two_qubit_gate_count"),
                    "depth": row.get(f"{label}_best_depth"),
                    "qc": row.pop(key),
                })

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
    print(f"\nCzas benchmarku [prepared_w, {state_name}]: {elapsed:.1f} s")

    df = pd.DataFrame(results)

    if output_dir is None:
        output_dir = prepared_w_benchmark_data_dir()
    os.makedirs(output_dir, exist_ok=True)

    if csv_path is None:
        csv_path = prepared_w_benchmark_results_path(state_name, mode)

    _write_topk_tables_to_output_dir(
        df, output_dir,
        file_prefix=f"benchmark_prepared_w_{state_name}_{mode}",
        fidelity_thresholds=fidelity_thresholds,
    )

    if resolved_circuits_dir is not None and fidelity_circuits:
        _save_top3_fidelity_circuits(fidelity_circuits, resolved_circuits_dir)

    _print_single_state_summary(df, f"{state_name} [prepared_w]")

    return df, csv_path


# ══════════════════════════ SINGLE-STATE BENCHMARK ═══════════

def _print_single_state_summary(df, state_name):
    """Print terminal summary for a single-state benchmark run."""
    if df.empty or "status" not in df.columns:
        print("\nBrak wyników do wyświetlenia.")
        return
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("\nŻaden przypadek nie zakończył się sukcesem.")
        return

    df_ok = df_ok.sort_values(
        by=["best_depth", "best_two_qubit_gate_count", "best_size"],
        ascending=True,
    )

    print("\n" + "=" * 80)
    print(f"  TOP 15 [{state_name}] (najniższa best_depth)")
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

    print("\n" + "=" * 80)
    print(f"  Statystyki wg klasy [{state_name}] (best_depth)")
    print("=" * 80)
    stats = df_ok.groupby("class_name")["best_depth"].agg(
        ["count", "min", "mean", "max"]
    )
    print(stats.to_string())

    print("\n" + "=" * 80)
    print(f"  Porównanie [{state_name}]: stara code space vs ogólne izometrie")
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


ALL_CLASS_NAMES = (
    "baseline",
    "monomial_old_codespace",
    "monomial_full",
    "fourier_like",
    "householder_random",
    "clifford_wh",
    "haar_random_isometry",
    "perturbed_isometry",
    "entangling_isometry",
    "structured_entangling",
    "product",
    "local_ry_only",
    "local_general_su2",
    "real_orthogonal",
    "near_identity",
    "finer_structured",
    "two_cz_ansatz",
)


def _run_single_state_benchmark(
    state_name="ghz3",
    n_qutrits=None,
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    class_filter=None,
):
    """Run the full benchmark pipeline for a single state and save results to CSV.

    class_filter : str or collection of str, optional
        If given, only candidates whose class_name matches are benchmarked.
        Accepts a single name or a comma-separated string or a collection.
    """
    resolved_n_qutrits = _state_num_qutrits(state_name, n_qutrits)
    state_name = _normalize_state_name(state_name, resolved_n_qutrits)

    filter_set = None
    if class_filter is not None:
        if isinstance(class_filter, str):
            filter_set = {c.strip() for c in class_filter.split(",")}
        else:
            filter_set = set(class_filter)

    filter_label = ",".join(sorted(filter_set)) if filter_set else "all"
    print("=" * 80)
    print(f"  Benchmark baz kodowania qutrytu  [state={state_name}]  [mode={mode}]  [class={filter_label}]")
    print(f"  Transpilacja: {n_transpile_runs} prób na kandydata (best + mean ± std)")
    print("=" * 80)

    if csv_path is None:
        csv_path = benchmark_state_results_path(state_name, mode)

    all_candidates = []

    if mode in ("full", "original"):
        all_candidates += generate_baseline()
        all_candidates += generate_monomial_old_codespace_bases(max_candidates=None)
        all_candidates += generate_monomial_full_bases(max_candidates=None)
        all_candidates += generate_fourier_like_bases(max_candidates=80)
        all_candidates += generate_householder_bases(n_samples=20, seed=42)
        all_candidates += generate_clifford_wh_bases()
        all_candidates += generate_haar_random_isometries(n_samples=20, seed=100)
        all_candidates += generate_perturbed_isometries(n_samples_per_eps=8, seed=200)
        all_candidates += generate_entangling_isometries(n_samples=20, seed=300)
        all_candidates += generate_structured_entangling_isometries()

    n_orig = len(all_candidates)

    if mode in ("full", "extended"):
        all_candidates += generate_product_bases(max_candidates=None)
        all_candidates += generate_local_ry_only(n_grid=10)
        all_candidates += generate_local_general_su2(n_samples=30, seed=600)
        all_candidates += generate_real_orthogonal_isometries(n_samples=20, seed=400)
        all_candidates += generate_near_identity_isometries(n_samples_per_eps=10, seed=500)
        all_candidates += generate_finer_structured_grid()
        all_candidates += generate_two_cz_ansatz(n_samples=50, seed=700)

    n_ext = len(all_candidates) - n_orig

    if filter_set:
        all_candidates = [(cls, n, e) for cls, n, e in all_candidates if cls in filter_set]

    print(f"\n  Kandydaci (oryginalne):     {n_orig}")
    print(f"  Kandydaci (rozszerzone):    {n_ext}")
    if filter_set:
        print(f"  Po filtrze ({filter_label}): {len(all_candidates)}")
    print(f"  Razem:                      {len(all_candidates)}")
    print("-" * 80)

    results = []
    fidelity_circuits = []
    t0 = time.time()

    resolved_circuits_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)

    for idx, (cls, name, E_new) in enumerate(all_candidates):
        print(
            f"  [{idx+1:4d}/{len(all_candidates)}]  {cls:28s}  {name:30s}",
            end="  ", flush=True,
        )

        row = benchmark_basis(
            E_new, cls, name,
            state_name=state_name,
            n_qutrits=resolved_n_qutrits,
            coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES,
            n_transpile_runs=n_transpile_runs,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
        )

        # Extract fidelity circuit objects before DataFrame creation
        for key in list(row.keys()):
            if key.startswith("_fid") and key.endswith("_best_qc"):
                label = key[1:].replace("_best_qc", "")
                fidelity_circuits.append({
                    "class_name": cls,
                    "candidate_name": name,
                    "label": label,
                    "two_q": row.get(f"{label}_best_two_qubit_gate_count"),
                    "depth": row.get(f"{label}_best_depth"),
                    "qc": row.pop(key),
                })

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
    print(f"\nCzas benchmarku [{state_name}]: {elapsed:.1f} s")

    df = pd.DataFrame(results)

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wyniki zapisane do: {csv_path}")

    _save_top3_per_class_csvs(df, csv_path, fidelity_thresholds=fidelity_thresholds)

    if resolved_circuits_dir is not None and fidelity_circuits:
        _save_top3_fidelity_circuits(fidelity_circuits, resolved_circuits_dir)

    _print_single_state_summary(df, state_name)

    return df, csv_path


# ══════════════════════════ MAIN ═════════════════════════════

def run_benchmark(n_qutrits=None, n_transpile_runs=20,
                   csv_path=None,
                   mode="full", circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
                   approximation_values=None,
                   fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
                   approximation_seed=0,
                   state_name="ghz3",
                   reuse_existing_ghz3=True,
                   combined_report_path=None,
                   class_filter=None,
                   encoding_strategy="append_w",
                   preselected_candidates_file=None,
                   output_dir=None):
    """
    Uruchamia benchmark i zapisuje wyniki do CSV.

    mode:
        "full"      — wszystkie generatory (oryginalne + rozszerzone)
        "original"  — tylko oryginalne generatory (klasy 0–9)
        "extended"  — tylko nowe rozszerzone generatory (klasy 10–15)

    state_name:
        "ghz3"       — 3-qutrytowy stan GHZ (star graph)
        "ghz_star"   — GHZ/star graph dla n_qutrits qutrytow
        "ghz_star_N" — to samo z N zakodowanym w nazwie wynikow
        "two_qutrit" — 2-qutrytowy stan (star graph, n=2)
        "ame43"      — stan AME(4,3) (specjalny graf z wielokrawędziami)
        "all"        — uruchom benchmark dla wszystkich trzech stanów
                       i wygeneruj wspólny raport markdown

    encoding_strategy:
        "append_w"  — standardowy obwod + lokalne W na koncu (domyslnie)
        "prepared_w_then_conjugated_entanglers"
                    — W|+> local preparation + (W⊗W)CZ(W†⊗W†) entanglery;
                      wymaga preselected_candidates_file

    preselected_candidates_file : str or None
        Sciezka do CSV z preselekcjonowanymi kandydatami z pierwszego etapu.
        Wymagana dla encoding_strategy="prepared_w_then_conjugated_entanglers".
        UWAGA: plik musi odpowiadac temu samemu state_name / eksperymentowi.

    output_dir : str or None
        Katalog wyjsciowy dla drugiego etapu. Jesli None, uzyty zostanie
        domyslny folder prepared_w_then_conjugated_entanglers_results/.

    class_filter:
        None         — wszystkie klasy kandydatów
        str          — jedna klasa lub wiele oddzielonych przecinkiem,
                       np. "monomial_full" lub
                       "monomial_old_codespace,baseline"
    """
    if encoding_strategy == "prepared_w_then_conjugated_entanglers":
        df, csv = _run_prepared_w_benchmark(
            state_name=state_name,
            n_qutrits=n_qutrits,
            preselected_candidates_file=preselected_candidates_file,
            n_transpile_runs=n_transpile_runs,
            csv_path=csv_path,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            class_filter=class_filter,
            output_dir=output_dir,
        )
        return df

    if state_name != "all":
        df, _ = _run_single_state_benchmark(
            state_name=state_name,
            n_qutrits=n_qutrits,
            n_transpile_runs=n_transpile_runs,
            csv_path=csv_path,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            class_filter=class_filter,
        )
        return df

    # ── tryb "all": trzy stany + wspólny raport ──
    common_kwargs = dict(
        n_transpile_runs=n_transpile_runs,
        mode=mode,
        circuits_output_dir=circuits_output_dir,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=approximation_seed,
        class_filter=class_filter,
    )

    state_frames = {}

    print("\n" + "#" * 80)
    print("  MULTI-STATE BENCHMARK: two_qutrit")
    print("#" * 80)
    state_frames["two_qutrit"], _ = _run_single_state_benchmark(
        state_name="two_qutrit", **common_kwargs,
    )

    print("\n" + "#" * 80)
    print("  MULTI-STATE BENCHMARK: ghz3")
    print("#" * 80)
    ghz_csv = benchmark_state_results_path("ghz3", mode)
    if reuse_existing_ghz3 and os.path.exists(ghz_csv):
        print(f"  Reusing existing GHZ3 results from: {ghz_csv}")
        state_frames["ghz3"] = pd.read_csv(ghz_csv)
    else:
        state_frames["ghz3"], _ = _run_single_state_benchmark(
            state_name="ghz3", **common_kwargs,
        )

    print("\n" + "#" * 80)
    print("  MULTI-STATE BENCHMARK: ame43")
    print("#" * 80)
    state_frames["ame43"], _ = _run_single_state_benchmark(
        state_name="ame43", **common_kwargs,
    )

    report_path = combined_report_path or multi_state_benchmark_report_path()
    write_multi_state_benchmark_report(state_frames, report_path)
    print(f"\nRaport markdown zapisany do: {report_path}")

    return state_frames


def _save_top3_per_class_csvs(df, csv_path, fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS):
    """Generate top-3-per-class summary CSVs next to the main results file.

    Produces:
      *_top3_by_depth.csv        - sorted by (best_depth, best_two_qubit_gate_count, best_size)
      *_top3_by_2q.csv           - sorted by (best_two_qubit_gate_count, best_depth)
      *_top3_exact.csv           - compatibility alias of *_top3_by_2q.csv
      *_top3_fid{NNN}.csv        - one per threshold, sorted by (fidNNN 2Q, fidNNN depth)
    """
    if df.empty or "status" not in df.columns:
        return

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        return

    base, ext = os.path.splitext(csv_path)

    parts = []
    for _, group in df_ok.groupby("class_name", sort=True):
        top = group.sort_values(
            by=["best_depth", "best_two_qubit_gate_count", "best_size"],
            ascending=True,
        ).head(3)
        parts.append(top)

    if parts:
        depth_df = pd.concat(parts, ignore_index=True)[
            ["class_name", "candidate_name", "best_depth",
             "best_size", "best_two_qubit_gate_count"]
        ].sort_values(
            by=["class_name", "best_depth", "best_two_qubit_gate_count", "best_size"],
            ascending=True,
        )
        depth_path = f"{base}_top3_by_depth{ext}"
        depth_df.to_csv(depth_path, index=False)
        print(f"  Top-3 by depth -> {depth_path}")

    parts = []
    for _, group in df_ok.groupby("class_name", sort=True):
        top = group.sort_values(
            by=["best_two_qubit_gate_count", "best_depth"],
            ascending=True,
        ).head(3)
        parts.append(top)

    if parts:
        twoq_df = pd.concat(parts, ignore_index=True)[
            ["class_name", "candidate_name", "best_depth",
             "best_size", "best_two_qubit_gate_count"]
        ].sort_values(
            by=["class_name", "best_two_qubit_gate_count", "best_depth"],
            ascending=True,
        )
        twoq_path = f"{base}_top3_by_2q{ext}"
        twoq_df.to_csv(twoq_path, index=False)
        print(f"  Top-3 by 2Q -> {twoq_path}")

        exact_path = f"{base}_top3_exact{ext}"
        twoq_df.to_csv(exact_path, index=False)
        print(f"  Top-3 exact -> {exact_path}")

    for threshold in fidelity_thresholds:
        label = _fidelity_label(threshold)
        depth_col = f"{label}_best_depth"
        twoq_col = f"{label}_best_two_qubit_gate_count"
        fid_col = f"{label}_best_fidelity"

        cols_needed = [depth_col, twoq_col]
        if not all(c in df_ok.columns for c in cols_needed):
            continue

        df_fid = df_ok.dropna(subset=cols_needed)
        if df_fid.empty:
            continue

        parts = []
        for _, group in df_fid.groupby("class_name", sort=True):
            top = group.sort_values(
                by=[twoq_col, depth_col],
                ascending=True,
            ).head(3)
            parts.append(top)

        if not parts:
            continue

        keep_cols = ["class_name", "candidate_name"]
        if fid_col in df_fid.columns:
            keep_cols.append(fid_col)
        keep_cols.extend([depth_col, twoq_col])

        fid_df = pd.concat(parts, ignore_index=True)[keep_cols].sort_values(
            by=["class_name", twoq_col, depth_col],
            ascending=True,
        )
        fid_path = f"{base}_top3_{label}{ext}"
        fid_df.to_csv(fid_path, index=False)
        pct = int(round(float(threshold) * 100))
        print(f"  Top-3 fidelity>={pct}% -> {fid_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark encoding bases for qutrit states",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="extended",
        choices=["full", "original", "extended"],
        help="which candidate generators to use (default: extended)",
    )
    parser.add_argument(
        "state",
        nargs="?",
        default="ghz3",
        help=(
            "which state to benchmark: ghz3, two_qutrit, ame43, all, "
            "or ghz_star/ghz_n with --n-qutrits"
        ),
    )
    parser.add_argument(
        "--n-qutrits",
        "--num-qutrits",
        dest="n_qutrits",
        type=int,
        default=None,
        help="number of qutrits for state ghz_star / ghz_n",
    )
    parser.add_argument(
        "--class",
        dest="class_filter",
        default=None,
        help=(
            "comma-separated list of class names to benchmark, e.g. "
            "'monomial_full' or 'monomial_old_codespace,baseline'. "
            f"Available: {', '.join(ALL_CLASS_NAMES)}"
        ),
    )
    parser.add_argument(
        "--encoding-strategy",
        dest="encoding_strategy",
        default="append_w",
        choices=list(VALID_ENCODING_STRATEGIES),
        help=(
            "Circuit build strategy: 'append_w' (default, stage 1) or "
            "'prepared_w_then_conjugated_entanglers' (stage 2, requires "
            "--preselected-candidates-file)"
        ),
    )
    parser.add_argument(
        "--preselected-candidates-file",
        dest="preselected_candidates_file",
        default=None,
        help=(
            "Path to CSV with preselected candidates from the first stage "
            "(append_w benchmark).  Required when --encoding-strategy is "
            "'prepared_w_then_conjugated_entanglers'.  The file MUST correspond "
            "to the same state_name / experiment — do NOT use GHZ3 rankings "
            "for AME(4,3)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=(
            "Custom output directory for second-stage results.  If not given, "
            "defaults to data/benchmarks/prepared_w_then_conjugated_entanglers_results/."
        ),
    )
    args = parser.parse_args()

    _csv = None
    run_benchmark(
        n_qutrits=args.n_qutrits,
        mode=args.mode,
        csv_path=_csv,
        state_name=args.state,
        class_filter=args.class_filter,
        encoding_strategy=args.encoding_strategy,
        preselected_candidates_file=args.preselected_candidates_file,
        output_dir=args.output_dir,
    )
