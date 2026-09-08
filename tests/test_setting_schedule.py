from copy import deepcopy
from itertools import product
import json

import pytest

from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.measurement import RandomizedBlocks
from qudits_on_qubits.experiments.setting_schedule import (
    SettingSchedule, generate_schedule, local_settings_for_reference, matches_pattern,
)
from qudits_on_qubits.reference_experiments import get_reference_experiment


def deterministic_draws(reference, settings):
    local = local_settings_for_reference(reference)
    indices = iter(local[i].index(label) for setting in settings for i, label in enumerate(setting))
    return lambda n: next(indices)


@pytest.mark.parametrize("state,total,patterns,useful", [
    ("two_qutrit", 9, 9, 9), ("ghz3", 18, 12, 12), ("ame43", 36, 13, 15),
])
def test_reference_cartesian_settings_and_wildcard_coverage(state, total, patterns, useful):
    reference = get_reference_experiment(state)
    settings = list(product(*local_settings_for_reference(reference)))
    assert len(settings) == total
    assert len(reference.measurement_settings()) == patterns
    assert sum(any(matches_pattern(setting, p) for p in reference.measurement_settings()) for setting in settings) == useful


def test_schedule_retains_duplicates_and_stops_at_first_complete_prefix():
    reference = get_reference_experiment("two_qutrit")
    all_settings = list(product(*local_settings_for_reference(reference)))
    draws = [all_settings[0]] * 3 + all_settings[1:] + [all_settings[0]]
    schedule = generate_schedule(reference, RandomizedBlocks(5, 20, 20), _randbelow=deterministic_draws(reference, draws), _source="test_sequence")
    assert schedule.complete
    assert len(schedule.blocks) == 11
    assert [b.block_id for b in schedule.blocks] == list(range(11))
    assert schedule.blocks[0].settings == schedule.blocks[1].settings
    assert schedule.missing_pattern_indices == ()
    assert SettingSchedule.from_safe_dict(json.loads(json.dumps(schedule.to_safe_dict()))) == schedule


def test_schedule_waits_for_minimum_after_early_coverage():
    reference = get_reference_experiment("two_qutrit")
    draws = list(product(*local_settings_for_reference(reference)))
    draws += [draws[0]] * 5
    schedule = generate_schedule(reference, RandomizedBlocks(12, 1, 20), _randbelow=deterministic_draws(reference, draws), _source="test_sequence")
    assert schedule.complete
    assert len(schedule.blocks) == 12


def test_schedule_cap_returns_incomplete_and_preserves_zero_blocks():
    reference = get_reference_experiment("ame43")
    unused = next(s for s in product(*local_settings_for_reference(reference)) if not any(matches_pattern(s, p) for p in reference.measurement_settings()))
    schedule = generate_schedule(reference, RandomizedBlocks(2, 7, 5), _randbelow=deterministic_draws(reference, [unused] * 5), _source="test_sequence")
    assert not schedule.complete
    assert len(schedule.blocks) == 5
    assert all(block.matched_pattern_indices == () for block in schedule.blocks)
    assert schedule.missing_pattern_indices == tuple(range(13))


def test_schedule_system_secrets_draws_each_party_independently(monkeypatch):
    from qudits_on_qubits.experiments import setting_schedule
    bounds = []
    monkeypatch.setattr(setting_schedule.secrets, "randbelow", lambda n: bounds.append(n) or 0)
    schedule = generate_schedule(get_reference_experiment("ghz3"), RandomizedBlocks(2, 4, 3))
    assert bounds == [len(s) for s in schedule.local_settings] * 3
    assert schedule.source == "system_secrets"
    assert schedule.seed is None


def test_production_rejects_seed_or_forged_randomness_provenance():
    reference = get_reference_experiment("two_qutrit")
    config = RandomizedBlocks(2, 5, 10)
    with pytest.raises(ExperimentValidationError, match="seed"):
        generate_schedule(reference, config, _seed=4)
    with pytest.raises(ExperimentValidationError, match="source"):
        generate_schedule(reference, config, _source="test_sequence")


@pytest.mark.parametrize("tamper", [
    lambda p: p["blocks"][0].update(block_id=99),
    lambda p: p["blocks"][0].update(matched_pattern_indices=[]),
    lambda p: p["blocks"][0].update(settings=["B0", "A0"]),
    lambda p: p.update(complete=True),
    lambda p: p.update(missing_pattern_indices=[]),
    lambda p: p.update(source="unknown"),
    lambda p: p.pop("source"),
    lambda p: p.pop("seed"),
    lambda p: p.update(input_probability=1.0),
    lambda p: p.update(source="system_secrets", seed=4),
    lambda p: p["local_settings"][0].append("A99"),
    lambda p: p["required_patterns"].pop(),
    lambda p: p["blocks"].pop(),
])
def test_schedule_deserialization_rejects_tampering(tamper):
    reference = get_reference_experiment("two_qutrit")
    schedule = generate_schedule(reference, RandomizedBlocks(2, 5, 3), _randbelow=lambda n: 0, _source="test_sequence")
    payload = deepcopy(schedule.to_safe_dict())
    tamper(payload)
    with pytest.raises(ExperimentValidationError):
        SettingSchedule.from_safe_dict(payload)


def test_ame_block_matches_two_patterns_and_seeded_source_roundtrips():
    reference = get_reference_experiment("ame43")
    setting = ("A0", "B0", "C1", "D0")
    schedule = generate_schedule(reference, RandomizedBlocks(1, 2, 1), _randbelow=deterministic_draws(reference, [setting]), _seed=19)
    assert len(schedule.blocks[0].matched_pattern_indices) == 2
    assert schedule.source == "test_seeded"
    assert schedule.seed == 19
    assert schedule.input_probability == pytest.approx(1 / 36)
    assert SettingSchedule.from_safe_dict(schedule.to_safe_dict()) == schedule


def test_deserialization_rejects_continuing_after_complete_prefix():
    reference = get_reference_experiment("two_qutrit")
    settings = list(product(*local_settings_for_reference(reference)))
    schedule = generate_schedule(reference, RandomizedBlocks(1, 1, 20), _randbelow=deterministic_draws(reference, settings))
    payload = schedule.to_safe_dict()
    extra = deepcopy(payload["blocks"][0])
    extra["block_id"] = len(payload["blocks"])
    payload["blocks"].append(extra)
    with pytest.raises(ExperimentValidationError, match="first complete prefix"):
        SettingSchedule.from_safe_dict(payload)
