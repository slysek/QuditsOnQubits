import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    DirectBasisCandidate,
    generate_extended_legacy_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.phase_equivalence import (
    PHASE_DUPLICATE_COLUMNS,
    deduplicate_candidates_up_to_global_phase,
    global_phase_between,
)


def candidate(cls, name, matrix, supported=True):
    return DirectBasisCandidate(name, cls, np.asarray(matrix) if supported else None,
                                source_class_name=cls, source_candidate_name=name,
                                error_message="" if supported else "unsupported")


def test_global_phase_between_exact_and_zero_cases():
    a = np.array([[1, 2j], [0, -1]], complex)
    assert global_phase_between(a, a) == 1 + 0j
    assert np.allclose(global_phase_between(a, -1j * a), -1j)
    assert global_phase_between(np.zeros((2, 2)), np.zeros((2, 2))) == 1 + 0j
    assert global_phase_between(np.zeros((2, 2)), a) is None
    assert global_phase_between(a, np.zeros((2, 2))) is None
    assert global_phase_between(a, np.zeros((3, 3))) is None


def test_global_phase_rejects_relative_phase_mismatch():
    a = np.array([1, 1], complex)
    b = np.array([1, 1j], complex)
    assert global_phase_between(a, b) is None


def test_dedup_prefers_baseline_then_lexicographic_and_audits_stably():
    base = np.eye(2, dtype=complex)
    candidates = [candidate("zclass", "z", -base), candidate("baseline", "late", base),
                  candidate("aclass", "a", 1j * base), candidate("baseline", "early", base)]
    result = deduplicate_candidates_up_to_global_phase(candidates)
    assert [(c.class_name, c.candidate_name) for c in result.representatives] == [("baseline", "early")]
    assert result.removed_count == 3
    assert tuple(result.duplicate_rows[0]) == PHASE_DUPLICATE_COLUMNS
    assert [(row["duplicate_class_name"], row["duplicate_candidate_name"])
            for row in result.duplicate_rows] == [("aclass", "a"), ("baseline", "late"), ("zclass", "z")]


def test_unsupported_candidates_remain_independent():
    items = [candidate("x", "one", None, False), candidate("x", "two", None, False)]
    result = deduplicate_candidates_up_to_global_phase(items)
    assert len(result.representatives) == 2
    assert result.removed_count == 0


def test_monomial_full_generator_counts():
    raw = generate_extended_legacy_candidates(
        include_haar_random=False, include_perturbed=False, include_entangling=False,
        include_structured_entangling=False, include_product=False, include_local_ry_only=False,
        include_local_general_su2=False, include_real_orthogonal=False, include_near_identity=False,
        include_finer_structured=False, include_two_cz_ansatz=False)
    assert len(raw) == 648
    result = deduplicate_candidates_up_to_global_phase(raw)
    assert len(result.representatives) == 216
    assert result.removed_count == 432
