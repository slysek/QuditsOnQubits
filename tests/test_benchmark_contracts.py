from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.models import EncodingCandidate


def test_candidate_owns_immutable_encoding_and_json_metadata():
    encoding = np.eye(4, 3, dtype=complex)
    parameters = {"angles": [0.2, 0.3], "nested": {"seed": np.int64(42)}}
    candidate = EncodingCandidate("test-1", "custom", encoding, parameters)
    encoding[0, 0] = 0
    parameters["angles"][0] = 9
    parameters["nested"]["seed"] = 7

    np.testing.assert_array_equal(candidate.encoding, np.eye(4, 3))
    assert candidate.parameters["angles"] == (0.2, 0.3)
    assert candidate.parameters["nested"]["seed"] == 42
    with pytest.raises(ValueError):
        candidate.encoding[0, 0] = 2
    with pytest.raises(ValueError):
        candidate.encoding.setflags(write=True)
    with pytest.raises(TypeError):
        candidate.parameters["new"] = 1
    with pytest.raises(FrozenInstanceError):
        candidate.family = "changed"
    assert json.loads(json.dumps(candidate.to_dict())) == {
        "candidate_id": "test-1", "family": "custom",
        "parameters": {"angles": [0.2, 0.3], "nested": {"seed": 42}},
    }


@pytest.mark.parametrize("encoding", [
    np.eye(3), np.eye(4), np.zeros((4, 3)), np.ones((4, 3)),
    np.eye(4, 3) * (1 + 1e-6), np.eye(4, 3) * np.nan,
    np.full((4, 3), np.inf), [["bad"]],
])
def test_candidate_rejects_invalid_isometry(encoding):
    with pytest.raises((ValueError, TypeError), match="encoding"):
        EncodingCandidate("candidate", "custom", encoding)


@pytest.mark.parametrize("identifier", ["", " ", "../escape", "/absolute", "a/b", "a\\b", ".", "..", "has space", "a\n", None, 12])
@pytest.mark.parametrize("field", ["candidate_id", "family"])
def test_candidate_rejects_unstable_identifiers(identifier, field):
    arguments = {"candidate_id": "candidate", "family": "custom", "encoding": np.eye(4, 3)}
    arguments[field] = identifier
    with pytest.raises((ValueError, TypeError), match=field):
        EncodingCandidate(**arguments)


@pytest.mark.parametrize("parameters", [[], {1: "value"}, {"bad": np.nan}, {"bad": float("inf")}, {"bad": 1j}, {"bad": object()}])
def test_candidate_rejects_non_json_parameters(parameters):
    with pytest.raises((ValueError, TypeError), match="parameters"):
        EncodingCandidate("candidate", "custom", np.eye(4, 3), parameters)


def test_candidate_serialization_does_not_expose_internal_metadata():
    candidate = EncodingCandidate("candidate", "custom", np.eye(4, 3), {"values": [1, 2]})
    candidate.to_dict()["parameters"]["values"].append(3)
    assert candidate.parameters["values"] == (1, 2)
