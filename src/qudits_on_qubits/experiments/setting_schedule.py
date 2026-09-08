"""Immutable, replayable independent local setting draws and coverage checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import secrets
from typing import Any

from qudits_on_qubits.reference_experiments import ReferenceExperimentSpec, get_reference_experiment

from .errors import ExperimentValidationError
from .measurement import RandomizedBlocks


def local_settings_for_reference(reference: ReferenceExperimentSpec) -> tuple[tuple[str, ...], ...]:
    """Return each party's alphabet from all audited Bell factors."""
    return tuple(
        tuple(sorted({factor.setting_label for term in reference.bell_functional.terms for factor in term.factors if factor.party == party}))
        for party in reference.state.party_order
    )


def matches_pattern(settings: Sequence[str], pattern: Sequence[str | None]) -> bool:
    """Identity positions in required patterns match every physical setting."""
    return len(settings) == len(pattern) and all(label is None or label == actual for actual, label in zip(settings, pattern))


@dataclass(frozen=True)
class SettingBlock:
    block_id: int
    settings: tuple[str, ...]
    matched_pattern_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.block_id) is not int or self.block_id < 0:
            raise ExperimentValidationError("block_id must be a non-negative integer")
        if isinstance(self.settings, (str, bytes)) or not isinstance(self.settings, Sequence):
            raise ExperimentValidationError("block settings must be a sequence of labels")
        settings = tuple(self.settings)
        if not settings or any(not isinstance(label, str) or not label for label in settings):
            raise ExperimentValidationError("block settings must contain full local setting labels")
        if isinstance(self.matched_pattern_indices, (str, bytes)) or not isinstance(self.matched_pattern_indices, Sequence):
            raise ExperimentValidationError("matched_pattern_indices must be a sequence")
        indices = tuple(self.matched_pattern_indices)
        if any(type(index) is not int or index < 0 for index in indices) or tuple(sorted(set(indices))) != indices:
            raise ExperimentValidationError("matched_pattern_indices must be sorted distinct non-negative integers")
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "matched_pattern_indices", indices)

    def to_safe_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id, "settings": list(self.settings), "matched_pattern_indices": list(self.matched_pattern_indices)}


@dataclass(frozen=True)
class SettingSchedule:
    config: RandomizedBlocks
    state: str
    blocks: tuple[SettingBlock, ...]
    local_settings: tuple[tuple[str, ...], ...]
    required_patterns: tuple[tuple[str | None, ...], ...]
    missing_pattern_indices: tuple[int, ...]
    complete: bool
    source: str = "system_secrets"
    seed: int | None = None

    @property
    def input_probability(self) -> float:
        """The uniform pre-stop probability of each full setting draw."""
        return 1 / math.prod(len(labels) for labels in self.local_settings)

    def __post_init__(self) -> None:
        if not isinstance(self.config, RandomizedBlocks):
            raise ExperimentValidationError("schedule config must be RandomizedBlocks")
        if not isinstance(self.state, str):
            raise ExperimentValidationError("schedule state must be an audited reference")
        try:
            reference = get_reference_experiment(self.state)
        except (TypeError, ValueError, KeyError) as error:
            raise ExperimentValidationError("schedule state must be an audited reference") from error
        if reference.experiment_id != self.state:
            raise ExperimentValidationError("schedule state must use its canonical reference name")
        if self.source not in {"system_secrets", "test_sequence", "test_seeded"}:
            raise ExperimentValidationError("schedule source must identify system_secrets or a test generator")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ExperimentValidationError("schedule seed must be a non-negative integer or None")
        if self.source == "system_secrets" and self.seed is not None:
            raise ExperimentValidationError("system_secrets source cannot have a seed")
        if self.source == "test_sequence" and self.seed is not None:
            raise ExperimentValidationError("a seeded test source must use test_seeded")
        if self.source == "test_seeded" and self.seed is None:
            raise ExperimentValidationError("test_seeded source requires a seed")
        try:
            local = tuple(tuple(labels) for labels in self.local_settings)
            patterns = tuple(tuple(pattern) for pattern in self.required_patterns)
            blocks = tuple(self.blocks)
            missing = tuple(self.missing_pattern_indices)
        except TypeError as error:
            raise ExperimentValidationError("invalid schedule sequence fields") from error
        if local != local_settings_for_reference(reference):
            raise ExperimentValidationError("local_settings do not match the audited reference")
        if patterns != reference.measurement_settings():
            raise ExperimentValidationError("required_patterns do not match the audited reference")
        if not self.config.setting_draws <= len(blocks) <= self.config.max_setting_draws:
            raise ExperimentValidationError("schedule block count violates the configured stop rule")
        covered: set[int] = set()
        for index, block in enumerate(blocks):
            if not isinstance(block, SettingBlock) or block.block_id != index:
                raise ExperimentValidationError("block IDs must be contiguous and start at zero")
            if len(block.settings) != len(local) or any(label not in alphabet for label, alphabet in zip(block.settings, local)):
                raise ExperimentValidationError("block settings must use the full valid labels of each party")
            matched = tuple(i for i, pattern in enumerate(patterns) if matches_pattern(block.settings, pattern))
            if block.matched_pattern_indices != matched:
                raise ExperimentValidationError("block matched_pattern_indices disagree with required patterns")
            covered.update(matched)
            if index + 1 >= self.config.setting_draws and len(covered) == len(patterns) and index != len(blocks) - 1:
                raise ExperimentValidationError("schedule continues after the first complete prefix")
        expected_missing = tuple(index for index in range(len(patterns)) if index not in covered)
        if any(type(index) is not int for index in missing) or missing != expected_missing:
            raise ExperimentValidationError("missing_pattern_indices disagree with schedule coverage")
        if type(self.complete) is not bool or self.complete != (not expected_missing):
            raise ExperimentValidationError("schedule complete flag disagrees with coverage")
        if not self.complete and len(blocks) != self.config.max_setting_draws:
            raise ExperimentValidationError("incomplete schedule must reach max_setting_draws")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "local_settings", local)
        object.__setattr__(self, "required_patterns", patterns)
        object.__setattr__(self, "missing_pattern_indices", missing)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": self.config.to_safe_dict(),
            "state": self.state,
            "blocks": [block.to_safe_dict() for block in self.blocks],
            "local_settings": [list(labels) for labels in self.local_settings],
            "required_patterns": [list(pattern) for pattern in self.required_patterns],
            "missing_pattern_indices": list(self.missing_pattern_indices),
            "complete": self.complete,
            "source": self.source,
            "seed": self.seed,
            "input_probability": self.input_probability,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "SettingSchedule":
        if not isinstance(data, Mapping) or type(data.get("version")) is not int or data["version"] != 1:
            raise ExperimentValidationError("unsupported setting schedule version")
        if set(data) != {"version", "config", "state", "blocks", "local_settings", "required_patterns", "missing_pattern_indices", "complete", "source", "seed", "input_probability"}:
            raise ExperimentValidationError("invalid persisted setting schedule fields or missing randomness provenance")
        try:
            values = dict(data)
            values.pop("version")
            probability = values.pop("input_probability")
            values["config"] = RandomizedBlocks.from_safe_dict(values["config"])
            values["blocks"] = tuple(SettingBlock(**block) for block in values["blocks"])
            schedule = cls(**values)
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or probability != schedule.input_probability:
                raise ExperimentValidationError("input_probability does not match independent uniform local settings")
            return schedule
        except (KeyError, TypeError) as error:
            raise ExperimentValidationError("invalid persisted setting schedule") from error


def generate_schedule(
    reference: ReferenceExperimentSpec,
    config: RandomizedBlocks,
    *,
    _randbelow: Callable[[int], int] | None = None,
    _source: str = "system_secrets",
    _seed: int | None = None,
) -> SettingSchedule:
    """Draw independently and stop at the first covered prefix after the minimum.

    Test generators are explicitly marked in persisted provenance. Production
    uses a separate ``secrets.randbelow`` call for every party and every block.
    """
    if not isinstance(reference, ReferenceExperimentSpec):
        raise ExperimentValidationError("reference must be ReferenceExperimentSpec")
    if not isinstance(config, RandomizedBlocks):
        raise ExperimentValidationError("config must be RandomizedBlocks")
    if _randbelow is None:
        if _seed is not None:
            raise ExperimentValidationError("seed requires an injected test generator")
        if _source != "system_secrets":
            raise ExperimentValidationError("test source requires an injected generator")
        draw = secrets.randbelow
    else:
        if not callable(_randbelow):
            raise ExperimentValidationError("_randbelow must be callable")
        draw = _randbelow
        if _source == "system_secrets":
            _source = "test_seeded" if _seed is not None else "test_sequence"
    local = local_settings_for_reference(reference)
    patterns = reference.measurement_settings()
    blocks: list[SettingBlock] = []
    missing = set(range(len(patterns)))
    for block_id in range(config.max_setting_draws):
        selected: list[str] = []
        for alphabet in local:
            index = draw(len(alphabet))
            if type(index) is not int or not 0 <= index < len(alphabet):
                raise ExperimentValidationError("_randbelow must return an integer within its bound")
            selected.append(alphabet[index])
        settings = tuple(selected)
        matches = tuple(index for index, pattern in enumerate(patterns) if matches_pattern(settings, pattern))
        blocks.append(SettingBlock(block_id, settings, matches))
        missing.difference_update(matches)
        if len(blocks) >= config.setting_draws and not missing:
            break
    return SettingSchedule(config, reference.experiment_id, tuple(blocks), local, patterns, tuple(sorted(missing)), not missing, _source, _seed)
