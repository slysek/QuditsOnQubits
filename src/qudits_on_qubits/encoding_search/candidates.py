from dataclasses import dataclass
from itertools import combinations, product as iter_product
from typing import Iterable, Optional, Sequence

import numpy as np


Candidate = tuple[str, str, object]

E_OLD = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]],
    dtype=complex,
)
OMEGA = np.exp(2j * np.pi / 3)
_PERMS_3 = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
]
_PHASES_3 = [1, OMEGA, OMEGA**2]


@dataclass(frozen=True)
class CandidateSearchConfig:
    """Controls the bounded candidate classes used by the v2 search."""

    max_monomial_full: Optional[int] = None
    max_product: Optional[int] = None
    include_product_grid: bool = False
    max_product_grid: Optional[int] = None
    product_grid_phase_steps: int = 4
    product_grid_polar_steps: int = 3
    include_near_identity: bool = False
    near_identity_samples_per_eps: int = 2
    near_identity_seed: int = 500
    limit_candidates: Optional[int] = None


def _product_angle_grid(config: CandidateSearchConfig):
    phase_angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(config.product_grid_phase_steps),
        endpoint=False,
    )
    polar_angles = np.linspace(
        0.0,
        np.pi,
        int(config.product_grid_polar_steps),
        endpoint=True,
    )
    return {"phase_angles": phase_angles, "polar_angles": polar_angles}


def _perm_matrix(perm):
    matrix = np.zeros((3, 3), dtype=complex)
    for index, source in enumerate(perm):
        matrix[index, source] = 1.0
    return matrix


def _phase_diag(phases):
    return np.diag(np.array(phases, dtype=complex))


def _phase_label(phases):
    return "".join(
        str(int(round(np.angle(phase) * 3 / (2 * np.pi))) % 3)
        for phase in phases
    )


def _generate_baseline():
    return [("baseline", "E_old", None)]


def _generate_monomial_full_bases(max_candidates=None):
    computational_basis = np.eye(4, dtype=complex)
    candidates: list[Candidate] = []

    for support in combinations(range(4), 3):
        basis = computational_basis[:, support]
        support_label = "".join(map(str, support))
        for perm in _PERMS_3:
            permutation = _perm_matrix(perm)
            for phases in iter_product(_PHASES_3, repeat=3):
                transform = _phase_diag(phases) @ permutation
                e_new = basis @ transform
                name = f"sup{support_label}_P{''.join(map(str, perm))}_ph{_phase_label(phases)}"
                candidates.append(("monomial_full", name, e_new))
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _single_qubit_product_library():
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


def _single_qubit_rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _single_qubit_rz(theta):
    half_theta = theta / 2
    return np.diag([np.exp(-1j * half_theta), np.exp(1j * half_theta)]).astype(complex)


def _single_qubit_su2(alpha, beta, gamma):
    return _single_qubit_rz(alpha) @ _single_qubit_rx(beta) @ _single_qubit_rz(gamma)


def _format_angle(theta):
    if np.isclose(theta, 0.0, atol=1e-12):
        theta = 0.0
    return f"{theta:.2f}"


def _single_qubit_product_grid(angle_grid):
    phase_angles = tuple(angle_grid["phase_angles"])
    polar_angles = tuple(angle_grid["polar_angles"])
    grid = []
    for alpha in phase_angles:
        for beta in polar_angles:
            for gamma in phase_angles:
                name = (
                    f"a{_format_angle(alpha)}"
                    f"_b{_format_angle(beta)}"
                    f"_g{_format_angle(gamma)}"
                )
                grid.append((name, _single_qubit_su2(alpha, beta, gamma)))
    return grid


def _generate_product_bases(max_candidates=None, mode="discrete", angle_grid=None):
    if mode == "discrete":
        library = _single_qubit_product_library()
        prefix = ""
    elif mode == "grid":
        library = _single_qubit_product_grid(angle_grid)
        prefix = "grid__"
    else:
        raise ValueError("mode must be 'discrete' or 'grid'.")

    candidates: list[Candidate] = []
    for (u_name, u_gate), (v_name, v_gate) in iter_product(library, repeat=2):
        e_new = np.kron(u_gate, v_gate) @ E_OLD
        candidates.append(("product", f"{prefix}U_{u_name}__V_{v_name}", e_new))
        if max_candidates is not None and len(candidates) >= max_candidates:
            return candidates
    return candidates


def _generate_near_identity_isometries(n_samples_per_eps=2, seed=500):
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    epsilons = [0.01, 0.03, 0.05, 0.1]
    candidates: list[Candidate] = []
    for eps in epsilons:
        for index in range(n_samples_per_eps):
            matrix = (
                rng.standard_normal((4, 4))
                + 1j * rng.standard_normal((4, 4))
            ) / np.sqrt(2)
            hermitian = (matrix + matrix.conj().T) / 2
            unitary = expm(1j * eps * hermitian)
            candidates.append(
                ("near_identity", f"nearid_eps{eps:.2f}_{index:02d}", unitary @ E_OLD)
            )
    return candidates


def _limit(candidates: list[Candidate], limit: Optional[int]) -> list[Candidate]:
    if limit is None:
        return candidates
    return candidates[: int(limit)]


def generate_stage1_candidates(config: CandidateSearchConfig | None = None) -> list[Candidate]:
    """Generate the stage-1 layered search pool.

    The default pool is intentionally bounded and interpretable:
    baseline, exhaustive monomial embeddings, and a finite product-unitary
    library. Optional product SU(2) grids and near-identity perturbations are
    opt-in because they can grow quickly.
    """
    config = config or CandidateSearchConfig()
    candidates: list[Candidate] = []

    candidates.extend(_generate_baseline())
    candidates.extend(_generate_monomial_full_bases(max_candidates=config.max_monomial_full))
    candidates.extend(
        _generate_product_bases(
            max_candidates=config.max_product,
            mode="discrete",
        )
    )

    if config.include_product_grid:
        candidates.extend(
            _generate_product_bases(
                max_candidates=config.max_product_grid,
                mode="grid",
                angle_grid=_product_angle_grid(config),
            )
        )

    if config.include_near_identity:
        candidates.extend(
            _generate_near_identity_isometries(
                n_samples_per_eps=config.near_identity_samples_per_eps,
                seed=config.near_identity_seed,
            )
        )

    return _limit(candidates, config.limit_candidates)


def filter_candidates_for_stage2(
    candidates: Iterable[Candidate],
    preselected: set[tuple[str, str]],
) -> list[Candidate]:
    """Keep candidates whose stable class/name pair appears in preselection."""
    return [
        (class_name, candidate_name, e_new)
        for class_name, candidate_name, e_new in candidates
        if (class_name, candidate_name) in preselected
    ]


def candidate_counts(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name, _, _ in candidates:
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def format_candidate_counts(candidates: Sequence[Candidate]) -> str:
    counts = candidate_counts(candidates)
    parts = [f"{class_name}={count}" for class_name, count in sorted(counts.items())]
    return ", ".join(parts)
