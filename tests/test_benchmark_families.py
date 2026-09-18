import json

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.families import LocalSU2, SchmidtTheta, canonical_candidate
from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding, theta_grid


def test_canonical_candidate_is_explicit_and_is_not_added_to_other_families():
    canonical = canonical_candidate()
    assert canonical.candidate_id == "canonical"
    assert canonical.family == "canonical"
    np.testing.assert_array_equal(canonical.encoding, np.eye(4, 3))
    assert len(tuple(LocalSU2(samples=3).generate())) == 3
    assert len(tuple(SchmidtTheta(points=3).generate())) == 3


def test_local_su2_sampling_is_reproducible_and_prefix_stable():
    first = tuple(LocalSU2(samples=5, seed=42).generate())
    again = tuple(LocalSU2(samples=5, seed=42).generate())
    prefix = tuple(LocalSU2(samples=2, seed=42).generate())
    other_seed = tuple(LocalSU2(samples=5, seed=43).generate())

    assert len({candidate.candidate_id for candidate in first}) == 5
    for index, (left, right) in enumerate(zip(first, again)):
        assert left.candidate_id == right.candidate_id
        assert left.to_dict() == right.to_dict()
        np.testing.assert_array_equal(left.encoding, right.encoding)
        assert left.parameters["seed"] == 42
        assert left.parameters["sample_index"] == index
        assert left.family == "local_su2"
        np.testing.assert_allclose(left.encoding.conj().T @ left.encoding, np.eye(3), atol=1e-14)
        # Each encoded basis vector remains a product of two single-qubit states.
        for column in left.encoding.T:
            assert abs(np.linalg.det(column.reshape(2, 2))) < 1e-14
    for left, right in zip(first, prefix):
        assert left.candidate_id == right.candidate_id
        np.testing.assert_array_equal(left.encoding, right.encoding)
    assert not np.allclose(first[0].encoding, other_seed[0].encoding)
    assert first[0].candidate_id != other_seed[0].candidate_id


def test_local_su2_configuration_and_numpy_integer_inputs():
    family = LocalSU2(samples=np.int64(2), seed=np.int64(7))
    assert json.loads(json.dumps(family.to_dict())) == {
        "family": "local_su2", "samples": 2, "seed": 7, "sampler": "haar_qr",
    }
    assert len(tuple(family.generate())) == 2


@pytest.mark.parametrize("field,value", [
    ("samples", 0), ("samples", -1), ("samples", True), ("samples", 1.5),
    ("samples", "2"), ("samples", np.bool_(True)),
    ("seed", -1), ("seed", True), ("seed", 42.0), ("seed", None), ("seed", np.nan),
])
def test_local_su2_rejects_invalid_configuration(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        LocalSU2(**{field: value})


def test_schmidt_theta_reuses_existing_grid_and_encodings():
    family = SchmidtTheta()
    candidates = tuple(family.generate())
    assert len(candidates) == 41
    assert len({candidate.candidate_id for candidate in candidates}) == 41
    assert family.to_dict() == {"family": "schmidt_theta", "points": 41}
    for index, (candidate, theta) in enumerate(zip(candidates, theta_grid())):
        assert candidate.family == "schmidt_theta"
        assert candidate.parameters == {"theta": theta, "grid_index": index, "points": 41}
        np.testing.assert_array_equal(candidate.encoding, theta_embedding(theta))
    np.testing.assert_array_equal(candidates[0].encoding, canonical_candidate().encoding)
    assert candidates[-1].parameters["theta"] == np.pi / 4
    assert tuple(SchmidtTheta(points=3).generate())[1].candidate_id != candidates[1].candidate_id


@pytest.mark.parametrize("points", [0, 1, -1, True, np.bool_(True), 3.0, "3", None])
def test_schmidt_theta_rejects_invalid_configuration(points):
    with pytest.raises((TypeError, ValueError), match="points"):
        SchmidtTheta(points=points)
