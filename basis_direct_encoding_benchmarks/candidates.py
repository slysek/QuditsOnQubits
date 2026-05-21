from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iter_product
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from basis_direct_encoding_benchmarks.math_utils import (
    OMEGA,
    extract_qutrit_unitary_from_embedding,
    qutrit_fourier,
    validate_isometry,
)


@dataclass(frozen=True)
class DirectBasisCandidate:
    """A qutrit encoding candidate for direct-basis encoding."""

    name: str
    candidate_type: str
    matrix: Optional[np.ndarray]
    source_class_name: str = ""
    source_candidate_name: str = ""
    notes: str = ""
    error_message: str = ""

    @property
    def is_supported(self) -> bool:
        return self.matrix is not None and not self.error_message

    @property
    def class_name(self) -> str:
        return self.source_class_name or self.candidate_type

    @property
    def candidate_name(self) -> str:
        return self.source_candidate_name or self.name


_PERMS_3 = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    return q @ np.diag(phases / np.abs(phases))


def _perm_matrix(perm: tuple[int, int, int]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=complex)
    for index, source in enumerate(perm):
        matrix[index, source] = 1.0
    return matrix


def _phase_diag(phase_digits: tuple[int, int, int]) -> np.ndarray:
    return np.diag([OMEGA**digit for digit in phase_digits]).astype(complex)


def _candidate(
    name: str,
    candidate_type: str,
    matrix: np.ndarray,
    *,
    source_class_name: str = "",
    source_candidate_name: str = "",
    notes: str = "",
) -> DirectBasisCandidate:
    return DirectBasisCandidate(
        name=name,
        candidate_type=candidate_type,
        matrix=np.asarray(matrix, dtype=complex),
        source_class_name=source_class_name,
        source_candidate_name=source_candidate_name,
        notes=notes,
    )


def generate_sanity_basis_candidates(
    random_count: int = 3,
    seed: int = 0,
) -> list[DirectBasisCandidate]:
    """Return a small, interpretable U(3) set for smoke tests and quick runs."""
    f3 = qutrit_fourier()
    candidates: list[DirectBasisCandidate] = [
        _candidate("I", "identity", np.eye(3, dtype=complex)),
        _candidate("F3", "fourier", f3),
        _candidate("F3dg", "fourier", f3.conj().T),
    ]

    for digits in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 2)):
        candidates.append(
            _candidate(f"D{''.join(map(str, digits))}", "diagonal_phase", _phase_diag(digits))
        )

    for perm in _PERMS_3:
        candidates.append(
            _candidate(f"P{''.join(map(str, perm))}", "permutation", _perm_matrix(perm))
        )

    rng = np.random.default_rng(seed)
    for index in range(int(random_count)):
        candidates.append(
            _candidate(f"haar_{index:03d}", "random_unitary", _random_unitary(3, rng))
        )

    return candidates


def _basis_type_from_legacy_class(class_name: str, candidate_name: str) -> str:
    if class_name == "baseline":
        return "identity"
    if class_name in {"householder_random", "haar_random"}:
        return "random_unitary"
    if class_name == "clifford_wh":
        return "clifford"
    if class_name == "fourier_like":
        return "fourier_like"
    if class_name in {"monomial_old_codespace", "monomial_full"}:
        return "monomial"
    if class_name == "product":
        return "product"
    if class_name == "haar_random_isometry":
        return "haar_random_isometry"
    if class_name == "perturbed_isometry":
        return "perturbed_isometry"
    if class_name == "entangling_isometry":
        return "entangling_isometry"
    if class_name == "structured_entangling":
        return "structured_entangling"
    if class_name == "local_ry_only":
        return "local_ry_only"
    if class_name == "local_general_su2":
        return "local_general_su2"
    if class_name == "real_orthogonal":
        return "real_orthogonal"
    if class_name == "near_identity":
        return "near_identity"
    if class_name == "finer_structured":
        return "finer_structured"
    if class_name == "two_cz_ansatz":
        return "two_cz_ansatz"
    return class_name


def direct_candidate_from_embedding(
    class_name: str,
    candidate_name: str,
    e_new,
) -> DirectBasisCandidate:
    """Convert a legacy E_new candidate into a direct encoding candidate."""
    try:
        matrix = extract_qutrit_unitary_from_embedding(e_new)
        notes = "Converted from legacy E_new = E_Z @ W candidate."
    except Exception:
        try:
            matrix = validate_isometry(e_new, 4, 3, name="E_new")
            notes = "Converted from legacy E_new 4x3 isometry candidate."
        except Exception as exc:
            return DirectBasisCandidate(
                name=candidate_name,
                candidate_type=_basis_type_from_legacy_class(class_name, candidate_name),
                matrix=None,
                source_class_name=class_name,
                source_candidate_name=candidate_name,
                notes="Not benchmarked: candidate is not a valid qutrit encoding isometry.",
                error_message=str(exc),
            )

    return DirectBasisCandidate(
        name=candidate_name,
        candidate_type=_basis_type_from_legacy_class(class_name, candidate_name),
        matrix=matrix,
        source_class_name=class_name,
        source_candidate_name=candidate_name,
        notes=notes,
    )


def generate_legacy_qutrit_u3_candidates(mode: str = "original") -> list[DirectBasisCandidate]:
    """Generate the legacy candidate classes that are actually E_Z @ W."""
    from QuditsOnQubits.benchmark_encoding_bases import (
        generate_baseline,
        generate_clifford_wh_bases,
        generate_fourier_like_bases,
        generate_householder_bases,
        generate_monomial_old_codespace_bases,
    )

    raw = []
    if mode in ("full", "original", "old_qutrit"):
        raw.extend(generate_baseline())
        raw.extend(generate_monomial_old_codespace_bases(max_candidates=None))
        raw.extend(generate_fourier_like_bases(max_candidates=80))
        raw.extend(generate_householder_bases(n_samples=20, seed=42))
        raw.extend(generate_clifford_wh_bases())
    else:
        raise ValueError("mode must be one of: full, original, old_qutrit.")

    return [
        direct_candidate_from_embedding(class_name, candidate_name, e_new)
        for class_name, candidate_name, e_new in raw
    ]


def generate_extended_legacy_candidates(
    *,
    include_monomial_full: bool = True,
    include_haar_random: bool = True,
    include_perturbed: bool = True,
    include_entangling: bool = True,
    include_structured_entangling: bool = True,
    include_product: bool = True,
    include_local_ry_only: bool = True,
    include_local_general_su2: bool = True,
    include_real_orthogonal: bool = True,
    include_near_identity: bool = True,
    include_finer_structured: bool = True,
    include_two_cz_ansatz: bool = True,
) -> list[DirectBasisCandidate]:
    """Generate the extended encoding-change classes (klasy 5-15) from benchmark_encoding_bases.

    These classes go beyond the old code space and include general 4x3 isometries,
    entangling unitaries, product bases, and structured parametric families:

      5.  monomial_full          – monomial embeddings over all 4 computational basis states
      6.  haar_random_isometry   – random 4x3 isometry (Haar)
      7.  perturbed_isometry     – E_old + small perturbation -> re-orthonormalisation
      8.  entangling_isometry    – W_random(4x4) @ E_old
      9.  structured_entangling  – (Ry x Ry) @ CZ @ (Rx x I) @ E_old
      10. product                – (U x V) @ E_old, local 1-qubit unitaries
      11. local_ry_only          – Ry(theta) x Ry(phi), dense grid
      12. local_general_su2      – random SU(2) x SU(2), no 2-qubit overhead
      13. real_orthogonal        – real orthogonal random 4x3 isometry
      14. near_identity          – W = expm(i*eps*H) for small eps
      15. finer_structured       – refined parameter grid around best structured_entangling results
      16. two_cz_ansatz          – two CZ layers with random rotation parameters
    """
    from QuditsOnQubits.benchmark_encoding_bases import (
        generate_entangling_isometries,
        generate_finer_structured_grid,
        generate_haar_random_isometries,
        generate_local_general_su2,
        generate_local_ry_only,
        generate_monomial_full_bases,
        generate_near_identity_isometries,
        generate_perturbed_isometries,
        generate_product_bases,
        generate_real_orthogonal_isometries,
        generate_structured_entangling_isometries,
        generate_two_cz_ansatz,
    )

    raw = []
    if include_monomial_full:
        raw.extend(generate_monomial_full_bases(max_candidates=None))
    if include_haar_random:
        raw.extend(generate_haar_random_isometries())
    if include_perturbed:
        raw.extend(generate_perturbed_isometries())
    if include_entangling:
        raw.extend(generate_entangling_isometries())
    if include_structured_entangling:
        raw.extend(generate_structured_entangling_isometries())
    if include_product:
        raw.extend(generate_product_bases(max_candidates=None))
    if include_local_ry_only:
        raw.extend(generate_local_ry_only())
    if include_local_general_su2:
        raw.extend(generate_local_general_su2())
    if include_real_orthogonal:
        raw.extend(generate_real_orthogonal_isometries())
    if include_near_identity:
        raw.extend(generate_near_identity_isometries())
    if include_finer_structured:
        raw.extend(generate_finer_structured_grid())
    if include_two_cz_ansatz:
        raw.extend(generate_two_cz_ansatz())

    return [
        direct_candidate_from_embedding(class_name, candidate_name, e_new)
        for class_name, candidate_name, e_new in raw
    ]


def generate_v2_stage1_direct_candidates(
    *,
    include_unsupported: bool = True,
    candidate_config=None,
) -> list[DirectBasisCandidate]:
    """Generate v2 candidates and keep direct-compatible U(3) embeddings."""
    from encoding_search_v2.candidates import CandidateSearchConfig, generate_stage1_candidates

    config = candidate_config or CandidateSearchConfig()
    raw = generate_stage1_candidates(config)
    converted = [
        direct_candidate_from_embedding(class_name, candidate_name, e_new)
        for class_name, candidate_name, e_new in raw
    ]
    if include_unsupported:
        return converted
    return [candidate for candidate in converted if candidate.is_supported]


def _deduplicate_by_source_key(
    candidates: Iterable[DirectBasisCandidate],
) -> list[DirectBasisCandidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[DirectBasisCandidate] = []
    for candidate in candidates:
        key = (candidate.class_name, candidate.candidate_name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def generate_all_qutrit_u3_candidates(candidate_config=None) -> list[DirectBasisCandidate]:
    """Generate direct encoding candidates from v2, legacy, and extended pools."""
    candidates: list[DirectBasisCandidate] = []
    candidates.extend(
        generate_v2_stage1_direct_candidates(
            include_unsupported=False,
            candidate_config=candidate_config,
        )
    )
    candidates.extend(
        candidate
        for candidate in generate_legacy_qutrit_u3_candidates("old_qutrit")
        # monomial_old_codespace is exactly the sup012 subset of monomial_full.
        if candidate.class_name != "monomial_old_codespace"
    )
    candidates.extend(generate_extended_legacy_candidates())
    return _deduplicate_by_source_key(candidates)


def candidates_from_old_csv(
    old_csv_path: str,
    *,
    include_unsupported: bool = True,
) -> list[DirectBasisCandidate]:
    """Regenerate known old candidates and return rows requested by an old CSV."""
    df = pd.read_csv(old_csv_path)
    if "class_name" not in df.columns or "candidate_name" not in df.columns:
        raise ValueError("old CSV must contain class_name and candidate_name columns.")

    requested = [
        (str(row["class_name"]), str(row["candidate_name"]))
        for _, row in df.drop_duplicates(["class_name", "candidate_name"]).iterrows()
    ]

    lookup: dict[tuple[str, str], DirectBasisCandidate] = {}
    for candidate in generate_legacy_qutrit_u3_candidates("original"):
        lookup[(candidate.class_name, candidate.candidate_name)] = candidate
    for candidate in generate_v2_stage1_direct_candidates(include_unsupported=True):
        lookup[(candidate.class_name, candidate.candidate_name)] = candidate

    selected: list[DirectBasisCandidate] = []
    for class_name, candidate_name in requested:
        candidate = lookup.get((class_name, candidate_name))
        if candidate is None:
            candidate = DirectBasisCandidate(
                name=candidate_name,
                candidate_type=_basis_type_from_legacy_class(class_name, candidate_name),
                matrix=None,
                source_class_name=class_name,
                source_candidate_name=candidate_name,
                notes="Not benchmarked: candidate was present in the old CSV but not regenerated.",
                error_message="candidate not found in direct-basis candidate lookup",
            )
        if include_unsupported or candidate.is_supported:
            selected.append(candidate)
    return selected


def limit_candidates(
    candidates: Iterable[DirectBasisCandidate],
    limit: Optional[int],
) -> list[DirectBasisCandidate]:
    values = list(candidates)
    if limit is None:
        return values
    return values[: int(limit)]
