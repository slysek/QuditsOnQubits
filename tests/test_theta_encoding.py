from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.theta_continuation.encoding import (
    make_theta_candidates,
    theta_embedding,
    theta_grid,
    theta_leakage,
)
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig


@pytest.mark.parametrize("theta", [0.0, np.pi / 16, np.pi / 8, np.pi / 4])
def test_embedding_and_leakage_form_orthogonal_complete_basis(theta):
    embedding = theta_embedding(theta)
    leakage = theta_leakage(theta)

    assert embedding.shape == (4, 3)
    assert leakage.shape == (4,)
    assert np.iscomplexobj(embedding)
    assert np.iscomplexobj(leakage)
    np.testing.assert_allclose(embedding.conj().T @ embedding, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(embedding.conj().T @ leakage, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(
        embedding @ embedding.conj().T + np.outer(leakage, leakage.conj()),
        np.eye(4),
        atol=1e-15,
    )


def test_encoding_endpoint_signs_and_basis_order():
    np.testing.assert_array_equal(theta_embedding(0), np.eye(4, 3))
    np.testing.assert_array_equal(theta_leakage(0), [0, 0, 0, 1])
    root_half = 1 / np.sqrt(2)
    np.testing.assert_allclose(
        theta_embedding(np.pi / 4),
        [[root_half, 0, 0], [0, 1, 0], [0, 0, 1], [-root_half, 0, 0]],
        atol=1e-15,
    )
    np.testing.assert_allclose(theta_leakage(np.pi / 4), [root_half, 0, 0, root_half])


@pytest.mark.parametrize(
    "theta",
    [True, False, np.bool_(False), "0.1", None, 0.2j, np.array([0.1]), -0.1,
     np.pi / 4 + 1e-10, np.nan, np.inf, -np.inf],
)
@pytest.mark.parametrize("function", [theta_embedding, theta_leakage])
def test_theta_functions_reject_invalid_angles(function, theta):
    with pytest.raises((TypeError, ValueError), match="theta"):
        function(theta)


def test_theta_grid_is_full_linspace_with_unchanged_prefix():
    full_grid = theta_grid()
    assert isinstance(full_grid, tuple)
    assert all(type(theta) is float for theta in full_grid)
    assert full_grid == tuple(np.linspace(0, np.pi / 4, 41))
    assert full_grid[0] == 0.0
    assert full_grid[-1] == np.pi / 4
    assert all(left < right for left, right in zip(full_grid, full_grid[1:]))
    assert theta_grid(limit_points=1) == full_grid[:1]
    assert theta_grid(limit_points=5) == full_grid[:5]
    assert theta_grid(limit_points=41) == full_grid
    assert theta_grid(np.int64(7), np.int64(3)) == tuple(np.linspace(0, np.pi / 4, 7))[:3]


@pytest.mark.parametrize("point_count", [True, np.bool_(True), 0, 1, -2, 2.0, 2.2, "3", None])
def test_theta_grid_rejects_invalid_point_counts(point_count):
    with pytest.raises((TypeError, ValueError), match="point_count"):
        theta_grid(point_count)


@pytest.mark.parametrize("limit", [True, np.bool_(True), 0, -1, 42, 2.0, 2.2, "3"])
def test_theta_grid_rejects_invalid_limits(limit):
    with pytest.raises((TypeError, ValueError), match="limit_points"):
        theta_grid(limit_points=limit)


def test_candidates_have_distinct_index_identities_and_own_matrices():
    candidates = make_theta_candidates(7)
    prefix = make_theta_candidates(7, 3)
    assert len(candidates) == 7
    assert [candidate.name for candidate in candidates] == [f"theta_{index:05d}" for index in range(7)]
    assert [candidate.name for candidate in prefix] == [candidate.name for candidate in candidates[:3]]
    assert len({candidate.candidate_name for candidate in candidates}) == 7
    assert len({id(candidate.matrix) for candidate in candidates}) == 7
    for theta, candidate in zip(theta_grid(7), candidates):
        assert candidate.candidate_type == "theta_continuation"
        assert candidate.class_name == "theta_continuation"
        assert candidate.is_supported
        assert "theta" in candidate.notes and "rad" in candidate.notes
        np.testing.assert_array_equal(candidate.matrix, theta_embedding(theta))
    candidates[0].matrix[0, 0] = 20
    np.testing.assert_array_equal(prefix[0].matrix, theta_embedding(0))


def test_config_defaults_and_grid():
    config = ThetaBenchmarkConfig()
    assert config.theta_points == 41
    assert config.limit_points is None
    assert config.max_nfev == 3000
    assert config.max_subdivisions == 4
    assert config.f3_tolerance == 1e-10
    assert config.cz3_tolerance == 1e-5
    assert config.transpiler_seeds == (0, 1, 2)
    assert config.optimization_level == 3
    assert config.states == ("two_qutrit", "ghz3", "ame43")
    assert config.baseline_sha256 == "a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791"
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template
    assert Path(config.baseline_qpy).is_file()
    assert load_cz3_template(config.baseline_qpy).baseline_sha256 == config.baseline_sha256
    assert config.grid() == theta_grid()
    assert ThetaBenchmarkConfig(limit_points=3).grid() == config.grid()[:3]


def test_config_normalizes_immutable_sequences_and_serializes_roundtrip():
    seeds = [np.int64(0), np.int64(2**32 - 1)]
    states = ["ghz3", "ame43"]
    config = ThetaBenchmarkConfig(
        theta_points=np.int64(5), limit_points=np.int64(2),
        max_nfev=np.int64(7), max_subdivisions=np.int64(0),
        f3_tolerance=np.float32(9e-11), cz3_tolerance=np.float64(1e-5),
        optimization_level=np.int64(0), transpiler_seeds=seeds, states=states,
    )
    seeds.append(10)
    states.append("two_qutrit")
    assert config.transpiler_seeds == (0, 2**32 - 1)
    assert config.states == ("ghz3", "ame43")
    with pytest.raises(FrozenInstanceError):
        config.theta_points = 10
    serialized = config.to_dict()
    assert serialized["transpiler_seeds"] == [0, 2**32 - 1]
    assert serialized["states"] == ["ghz3", "ame43"]
    assert "output_dir" not in serialized
    assert "mode" not in serialized
    assert ThetaBenchmarkConfig.from_dict(json.loads(json.dumps(serialized))) == config
    serialized["states"].append("two_qutrit")
    assert config.states == ("ghz3", "ame43")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("theta_points", True), ("theta_points", 1), ("theta_points", 5.0),
        ("limit_points", False), ("limit_points", 0), ("limit_points", 42), ("limit_points", 2.0),
        ("max_nfev", True), ("max_nfev", 0), ("max_nfev", -1), ("max_nfev", 1.5),
        ("max_subdivisions", True), ("max_subdivisions", -1), ("max_subdivisions", 2.0),
        ("max_subdivisions", 5), ("max_subdivisions", 100),
        ("optimization_level", True), ("optimization_level", -1), ("optimization_level", 4),
        ("optimization_level", 1.0),
        ("transpiler_seeds", []), ("transpiler_seeds", [0, 0]), ("transpiler_seeds", [-1]),
        ("transpiler_seeds", [2**32]), ("transpiler_seeds", [True]),
        ("transpiler_seeds", [np.bool_(True)]), ("transpiler_seeds", [1.0]),
        ("transpiler_seeds", "01"), ("transpiler_seeds", None),
        ("states", []), ("states", ["ghz3", "ghz3"]), ("states", ["unknown"]),
        ("states", "ghz3"), ("states", [1]), ("states", [["ghz3"]]),
        ("states", None), ("baseline_qpy", ""), ("baseline_qpy", "  "),
        ("baseline_qpy", None), ("baseline_qpy", 1),
        ("baseline_sha256", "a" * 63), ("baseline_sha256", "g" * 64),
        ("baseline_sha256", None), ("baseline_sha256", "a" * 65),
    ],
)
def test_config_rejects_invalid_fields(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        ThetaBenchmarkConfig(**{field: value})


@pytest.mark.parametrize("field", ["f3_tolerance", "cz3_tolerance"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 0, -1e-6, np.inf, -np.inf, np.nan, "0.1", 1j])
def test_config_rejects_nonpositive_or_nonfinite_tolerances(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        ThetaBenchmarkConfig(**{field: value})


def test_from_dict_validates_and_rejects_unknown_config_fields():
    assert ThetaBenchmarkConfig.from_dict({}) == ThetaBenchmarkConfig()
    with pytest.raises((TypeError, ValueError), match="theta_points"):
        ThetaBenchmarkConfig.from_dict({"theta_points": 1})
    with pytest.raises((TypeError, ValueError), match="output_dir"):
        ThetaBenchmarkConfig.from_dict({"output_dir": "artifacts"})


def test_config_accepts_uppercase_sha256_and_normalizes_it():
    config = ThetaBenchmarkConfig(baseline_sha256="ABCDEF01" * 8)
    assert config.baseline_sha256 == "abcdef01" * 8


@pytest.mark.parametrize("field,limit", [("f3_tolerance", 1e-10)])
def test_config_rejects_tolerances_looser_than_existing_synthesis(field, limit):
    for value in (2 * limit, np.nextafter(limit, np.inf)):
        with pytest.raises(ValueError, match=field + ".*must not exceed"):
            ThetaBenchmarkConfig(**{field: value})
        with pytest.raises(ValueError, match=field + ".*must not exceed"):
            ThetaBenchmarkConfig.from_dict({field: value})
    assert getattr(ThetaBenchmarkConfig(**{field: limit}), field) == limit
    assert getattr(ThetaBenchmarkConfig(**{field: limit / 2}), field) == limit / 2
