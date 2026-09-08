from dataclasses import FrozenInstanceError
import json

import pytest

from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.measurement import RandomizedBlocks
from qudits_on_qubits.experiments.models import (
    AerIdeal, BootstrapConfig, ExperimentSpec, IBMHardware, MitigationConfig, PathBasis,
)


def spec(**kwargs):
    return ExperimentSpec("ghz3", PathBasis("basis"), AerIdeal(), **kwargs)


def test_randomized_config_immutable_json_roundtrip():
    config = RandomizedBlocks(20, 50, 500)
    payload = config.to_safe_dict()
    assert payload["mode"] == "independent_local_uniform_blocks"
    assert payload["version"] == 1
    assert RandomizedBlocks.from_safe_dict(json.loads(json.dumps(payload))) == config
    with pytest.raises(FrozenInstanceError):
        config.setting_draws = 21


def test_one_block_minimum_and_ibm_spec_roundtrip():
    config = RandomizedBlocks(1, 1, 1)
    value = ExperimentSpec("ame43", PathBasis("basis"), IBMHardware("ibm_example"), measurement=config)
    assert ExperimentSpec.from_safe_dict(value.to_safe_dict()) == value


@pytest.mark.parametrize("name,value", [
    ("setting_draws", 0), ("setting_draws", True), ("setting_draws", 2.0),
    ("shots_per_draw", 0), ("shots_per_draw", False),
    ("max_setting_draws", 3), ("max_setting_draws", 10.0),
    ("max_circuits_per_job", 0), ("max_circuits_per_job", True),
    ("confidence_level", 0), ("confidence_level", 1),
    ("confidence_level", float("nan")), ("confidence_level", True),
    ("confidence_level", 10**1000),
])
def test_randomized_config_validation(name, value):
    kwargs = dict(setting_draws=4, shots_per_draw=10, max_setting_draws=40)
    kwargs[name] = value
    with pytest.raises(ExperimentValidationError, match=name):
        RandomizedBlocks(**kwargs)


def test_randomized_spec_defaults_and_legacy_payload_stability():
    legacy = spec()
    assert legacy.shots == 20480
    assert legacy.uncertainty == BootstrapConfig()
    assert "measurement" not in legacy.to_safe_dict()
    assert ExperimentSpec.from_safe_dict(legacy.to_safe_dict()) == legacy
    randomized = spec(measurement=RandomizedBlocks(10, 50, 100))
    assert randomized.shots is None
    assert randomized.uncertainty is None
    assert randomized.bootstrap is None
    payload = randomized.to_safe_dict()
    assert payload["shots"] is payload["uncertainty"] is None
    assert ExperimentSpec.from_safe_dict(json.loads(json.dumps(payload))) == randomized


@pytest.mark.parametrize("kwargs", [
    {"shots": 20480}, {"shots": None}, {"uncertainty": BootstrapConfig()},
    {"uncertainty": None}, {"bootstrap": BootstrapConfig()}, {"bootstrap": None},
])
def test_randomized_spec_rejects_explicit_legacy_controls(kwargs):
    with pytest.raises(ExperimentValidationError):
        spec(measurement=RandomizedBlocks(10, 50, 100), **kwargs)


@pytest.mark.parametrize("flag", ["readout", "zne", "circuit_twirling", "force_recalibration"])
def test_randomized_spec_rejects_active_mitigation(flag):
    with pytest.raises(ExperimentValidationError, match="mitigation"):
        spec(measurement=RandomizedBlocks(10, 50, 100), mitigation=MitigationConfig(**{flag: True}))


@pytest.mark.parametrize("field,value", [("shots", 50), ("uncertainty", {})])
def test_randomized_spec_rejects_non_null_persisted_legacy_controls(field, value):
    payload = spec(measurement=RandomizedBlocks(10, 50, 100)).to_safe_dict()
    payload[field] = value
    with pytest.raises(ExperimentValidationError):
        ExperimentSpec.from_safe_dict(payload)
