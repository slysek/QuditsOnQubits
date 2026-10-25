import numpy as np
import pytest

from qudits_on_qubits.encoding_search.candidates import (
    candidate_counts,
    generate_stage1_candidates,
)
from qudits_on_qubits.encoding_search.triviality import (
    _filter_trivial_candidates,
    candidate_metadata_fields,
)


def test_default_pool_has_valid_encodings_and_stable_identities():
    candidates = generate_stage1_candidates()

    assert candidate_counts(candidates) == {"baseline": 1, "monomial_full": 648, "product": 64}
    assert len({(class_name, name) for class_name, name, _ in candidates}) == len(candidates)
    for _, _, encoding in candidates[1:]:
        np.testing.assert_allclose(encoding.conj().T @ encoding, np.eye(3), atol=1e-12, rtol=0)


@pytest.mark.parametrize("phase, trivial", [(1.0, True), (np.exp(0.4j), False)])
def test_baseline_equivalence_preserves_global_phase_metadata(phase, trivial):
    encoding = phase * np.eye(4, 3)
    metadata = candidate_metadata_fields("product", "same_state", encoding)

    assert metadata["is_baseline_equivalent"]
    assert metadata["is_trivial_identity"] is trivial
    kept, skipped = _filter_trivial_candidates(
        [("baseline", "E_old", None), ("product", "same_state", encoding)],
        state_name="two_qutrit", stage=1,
    )
    assert kept == [("baseline", "E_old", None)]
    assert len(skipped) == 1
    assert skipped[0]["status"] == "skipped_baseline_equivalent"


def test_relative_logical_phase_is_not_a_baseline_equivalence():
    encoding = np.eye(4, 3, dtype=complex)
    encoding[1, 1] = np.exp(0.4j)

    assert not candidate_metadata_fields("product", "relative_phase", encoding)["is_baseline_equivalent"]
