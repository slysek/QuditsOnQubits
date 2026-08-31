from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.modules.pop("qudits_on_qubits", None)

from qudits_on_qubits.experiments.errors import ExperimentValidationError, JobResultError
from qudits_on_qubits.experiments.mitigation import (
    ReadoutCalibration,
    assignment_matrices_from_counts,
)
from qudits_on_qubits.experiments.mitigation import linear_zne_extrapolate
from qudits_on_qubits.experiments.models import BootstrapConfig
from qudits_on_qubits.experiments.uncertainty import (
    BootstrapBellResults,
    BootstrapInputs,
    _RawInvalidCodewordShots,
    _Task6ReadoutBootstrap,
    _resample_counts,
    bootstrap_bell_results,
)


def _single_probability(counts_by_setting: Mapping[object, Mapping[str, float]]) -> complex:
    counts = counts_by_setting["setting"]
    total = sum(counts.values())
    return complex(counts.get("1", 0.0) / total)


def _reference_inputs_with_invalid_counts() -> BootstrapInputs:
    setting = ("A0",)
    return BootstrapInputs(
        counts_by_factor={1: {setting: {"00": 8, "11": 2}}},
        terms=({"coeff": 1.0, "settings": setting, "powers": (0,)},),
        qutrit_bit_indices_by_setting={setting: ((0, 1),)},
        decoding_kwargs={
            "bit_order": "left-to-right",
            "outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
            "d": 3,
        },
    )


def _reference_inputs_for_factors(
    counts_by_factor: Mapping[int, Mapping[tuple[str, ...], Mapping[str, int]]],
    *,
    readout_calibration: ReadoutCalibration | None = None,
) -> BootstrapInputs:
    reference = _reference_inputs_with_invalid_counts()
    return BootstrapInputs(
        counts_by_factor=counts_by_factor,
        terms=reference.terms,
        qutrit_bit_indices_by_setting=reference.qutrit_bit_indices_by_setting,
        decoding_kwargs=reference.decoding_kwargs,
        readout_calibration=readout_calibration,
    )


def test_default_bootstrap_reports_conditional_unconditional_and_invalid_evidence() -> None:
    config = BootstrapConfig(samples=5, confidence_level=0.6, seed=21)

    result = bootstrap_bell_results(_reference_inputs_with_invalid_counts(), config)

    assert result.raw is result.raw_conditional
    assert result.raw.estimate.real == pytest.approx(1.0)
    assert result.raw_unconditional is not None
    assert result.raw_unconditional.estimate.real == pytest.approx(0.8)
    assert result.raw_invalid_codeword_rate is not None
    assert result.raw_invalid_codeword_rate.estimate == pytest.approx(0.2)
    assert result.raw_invalid_codeword_shots is not None
    assert result.raw_invalid_codeword_shots.to_safe_dict() == {
        "total_shots": 10,
        "accepted_shots": 8,
        "invalid_shots": 2,
    }
    assert result.raw_conditional is not result.raw_unconditional
    assert result.raw_conditional.standard_error.real == pytest.approx(0.0)

    rng = np.random.default_rng(21)
    invalid_replicates = np.asarray(
        [rng.multinomial(10, (0.8, 0.2))[1] / 10 for _ in range(5)]
    )
    assert result.raw_unconditional.standard_error.real == pytest.approx(
        np.std(invalid_replicates, ddof=1)
    )
    assert result.raw_invalid_codeword_rate.standard_error == pytest.approx(
        np.std(invalid_replicates, ddof=1)
    )
    assert result.raw_invalid_codeword_rate.confidence_interval.low == pytest.approx(
        np.quantile(invalid_replicates, 0.2)
    )
    assert result.raw_invalid_codeword_rate.confidence_interval.high == pytest.approx(
        np.quantile(invalid_replicates, 0.8)
    )
    payload = result.to_safe_dict()
    assert payload["raw"] == payload["raw_conditional"]


def test_raw_shot_summary_preserves_counts_above_float_precision() -> None:
    setting = ("A0",)
    valid_shots = 2**53

    result = bootstrap_bell_results(
        _reference_inputs_for_factors(
            {1: {setting: {"00": valid_shots, "11": 1}}}
        ),
        BootstrapConfig(samples=2, seed=3),
    )

    assert result.raw_invalid_codeword_shots is not None
    assert result.raw_invalid_codeword_shots.to_safe_dict() == {
        "total_shots": valid_shots + 1,
        "accepted_shots": valid_shots,
        "invalid_shots": 1,
    }


@pytest.mark.parametrize(
    "values",
    [
        (True, 0, 0),
        (1.0, 1, 0),
        (-1, 0, -1),
        (10, 9, 2),
    ],
)
def test_raw_invalid_codeword_shots_rejects_invalid_or_inconsistent_counts(
    values: tuple[object, object, object],
) -> None:
    with pytest.raises(ExperimentValidationError):
        _RawInvalidCodewordShots(*values)  # type: ignore[arg-type]


def test_raw_invalid_codeword_shots_is_frozen_and_serializes_only_counts() -> None:
    shots = _RawInvalidCodewordShots(10, 8, 2)

    assert shots.to_safe_dict() == {
        "total_shots": 10,
        "accepted_shots": 8,
        "invalid_shots": 2,
    }
    with pytest.raises(FrozenInstanceError):
        shots.invalid_shots = 3  # type: ignore[misc]


def test_raw_bootstrap_zero_variance_is_immutable_serializable_and_has_no_replicates() -> None:
    source_counts = {1: {"setting": {"0": 12}}}
    inputs = BootstrapInputs(
        counts_by_factor=source_counts,
        terms=(),
        qutrit_bit_indices_by_setting={},
    )

    result = bootstrap_bell_results(
        inputs,
        BootstrapConfig(samples=4, confidence_level=0.8, seed=7),
        _evaluator=_single_probability,
    )

    assert isinstance(result, BootstrapBellResults)
    assert result.raw.estimate.to_safe_dict() == {"real": 0.0, "imag": 0.0}
    assert result.raw.standard_error.to_safe_dict() == {"real": 0.0, "imag": 0.0}
    assert result.raw.confidence_interval.to_safe_dict() == {
        "real": {"low": 0.0, "high": 0.0},
        "imag": {"low": 0.0, "high": 0.0},
    }
    payload = result.to_safe_dict()
    assert payload["config"] == {
        "samples": 4,
        "confidence_level": 0.8,
        "seed": 7,
        "include_readout_calibration": True,
    }
    assert set(payload) == {"raw", "config", "diagnostics"}
    for field_name in (
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "readout_mitigated_conditional",
        "readout_mitigated_unconditional",
        "readout_effective_invalid_codeword_weight",
        "zne_conditional",
        "zne_unconditional",
        "zne_readout_mitigated_conditional",
        "zne_readout_mitigated_unconditional",
    ):
        assert getattr(result, field_name) is None
    assert "replicate" not in repr(payload).lower()
    json.dumps(payload, allow_nan=False)
    assert isinstance(inputs.counts_by_factor, MappingProxyType)
    source_counts[1]["setting"]["0"] = 1
    assert inputs.counts_by_factor[1]["setting"]["0"] == 12


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("raw_conditional", object()),
        ("raw_invalid_codeword_rate", object()),
        ("raw_invalid_codeword_shots", object()),
    ],
)
def test_results_reject_invalid_explicit_estimate_types(
    field_name: str, value: object
) -> None:
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": {"0": 1}}},
            terms=(),
            qutrit_bit_indices_by_setting={},
        ),
        BootstrapConfig(samples=2),
        _evaluator=_single_probability,
    )

    with pytest.raises(ExperimentValidationError, match=field_name):
        replace(result, **{field_name: value})


def _calibration() -> ReadoutCalibration:
    raw_counts = (
        {"0": 9, "1": 1},
        {"0": 2, "1": 8},
    )
    return ReadoutCalibration(
        backend_identity="backend-a",
        calibration_id="cal-1",
        qubit_mapping=(0,),
        timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
        shots=10,
        raw_counts=raw_counts,
        assignment_matrices=(((0.9, 0.2), (0.1, 0.8)),),
    )


def _reference_calibration() -> ReadoutCalibration:
    return ReadoutCalibration(
        backend_identity="backend-a",
        calibration_id="cal-reference",
        qubit_mapping=(0, 1),
        timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
        shots=10,
        raw_counts=(
            {"0": 10, "1": 0},
            {"0": 0, "1": 10},
            {"0": 10, "1": 0},
            {"0": 0, "1": 10},
        ),
        assignment_matrices=(
            ((1.0, 0.0), (0.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
    )


def _union_calibration() -> ReadoutCalibration:
    mapping = (10, 15, 16, 11)
    return ReadoutCalibration(
        backend_identity="backend-a",
        calibration_id="cal-union",
        qubit_mapping=mapping,
        timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
        shots=10,
        raw_counts=tuple(
            {"0": 10, "1": 0} if index % 2 == 0 else {"0": 0, "1": 10}
            for index in range(2 * len(mapping))
        ),
        assignment_matrices=tuple(((1.0, 0.0), (0.0, 1.0)) for _ in mapping),
    )


def test_default_readout_bootstrap_uses_each_setting_physical_mapping_for_all_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ((10, 15, 16, 11), (15, 16, 11, 10))
    seen_mappings: list[tuple[int, ...]] = []

    class RecordingMitigation:
        def apply_correction(
            self, counts: Mapping[str, int], qubits: tuple[int, ...]
        ) -> dict[str, float]:
            seen_mappings.append(qubits)
            total = sum(counts.values())
            return {outcome: count / total for outcome, count in counts.items()}

    mitigation = RecordingMitigation()
    uncertainty_module = sys.modules[bootstrap_bell_results.__module__]
    monkeypatch.setattr(
        uncertainty_module,
        "build_m3_mitigation",
        lambda calibration: mitigation,
    )
    counts = {
        1: {
            "setting-a": {"0000": 8, "1111": 2},
            "setting-b": {"0000": 7, "1111": 3},
        },
        3: {
            "setting-a": {"0000": 6, "1111": 4},
            "setting-b": {"0000": 5, "1111": 5},
        },
    }

    bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor=counts,
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_union_calibration(),
            physical_qubit_mappings=mappings,
        ),
        BootstrapConfig(samples=2, seed=4, include_readout_calibration=False),
        _evaluator=lambda _: 0j,
    )

    assert seen_mappings == list(mappings) * 6


@pytest.mark.parametrize(
    "physical_qubit_mappings",
    [
        ((10, 15, 16, 11),),
        ((10, 15, 16), (15, 16, 11)),
        ((10, 15, 16, 16), (15, 16, 11, 10)),
        ((10, 15, 16, 99), (15, 16, 11, 10)),
    ],
)
def test_bootstrap_inputs_reject_invalid_per_setting_physical_mappings(
    physical_qubit_mappings: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(ExperimentValidationError, match="physical"):
        BootstrapInputs(
            counts_by_factor={
                1: {
                    "setting-a": {"0000": 1},
                    "setting-b": {"0000": 1},
                }
            },
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_union_calibration(),
            physical_qubit_mappings=physical_qubit_mappings,
        )


class _RecordingReadout:
    def __init__(self) -> None:
        self.build_calls = 0
        self.resample_calls = 0
        self.rng_ids: list[int] = []
        self.applied_contexts: list[object] = []

    def build_context(self, calibration: ReadoutCalibration) -> object:
        self.build_calls += 1
        return ("fixed", self.build_calls)

    def resample_calibration(
        self, calibration: ReadoutCalibration, rng: np.random.Generator
    ) -> object:
        self.resample_calls += 1
        self.rng_ids.append(id(rng))
        for counts in calibration.raw_counts:
            rng.multinomial(
                calibration.shots,
                np.asarray((counts["0"], counts["1"])) / calibration.shots,
            )
        return ("resampled", self.resample_calls)

    def apply(
        self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
    ) -> Mapping[object, Mapping[str, float]]:
        self.applied_contexts.append(context)
        return {
            setting: {
                outcome: count / sum(counts.values()) for outcome, count in counts.items()
            }
            for setting, counts in counts_by_setting.items()
        }


class _SignedReferenceReadout(_RecordingReadout):
    def apply(
        self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
    ) -> Mapping[object, Mapping[str, float]]:
        self.applied_contexts.append(context)
        return {
            setting: {"00": 1.25, "11": -0.25}
            for setting in counts_by_setting
        }


def test_default_readout_summarizes_structured_metrics_from_one_correction() -> None:
    setting = ("A0",)
    samples = 3
    readout = _SignedReferenceReadout()
    result = bootstrap_bell_results(
        _reference_inputs_for_factors(
            {1: {setting: {"00": 8, "11": 2}}},
            readout_calibration=_reference_calibration(),
        ),
        BootstrapConfig(
            samples=samples,
            seed=9,
            include_readout_calibration=True,
        ),
        readout_strategy=readout,
    )

    assert result.raw is result.raw_conditional
    assert result.readout_mitigated is result.readout_mitigated_conditional
    assert result.readout_mitigated_unconditional is not None
    assert result.readout_mitigated_unconditional.estimate.real == pytest.approx(1.25)
    effective_invalid = result.readout_effective_invalid_codeword_weight
    assert effective_invalid is not None
    assert math.isfinite(effective_invalid.estimate)
    assert effective_invalid.estimate == pytest.approx(-0.25)
    assert readout.build_calls == 1
    assert readout.resample_calls == samples
    assert len(readout.applied_contexts) == (samples + 1)


class _RecordingZNE:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[complex, ...]]] = []

    def extrapolate(
        self, factors: tuple[int, ...], values: tuple[complex, ...]
    ) -> tuple[complex, object]:
        self.calls.append((tuple(factors), tuple(values)))
        return linear_zne_extrapolate(factors, values)


class _ConstantZNE:
    def __init__(self, value: complex) -> None:
        self.value = value
        self.calls: list[tuple[tuple[int, ...], tuple[complex, ...]]] = []

    def extrapolate(
        self, factors: tuple[int, ...], values: tuple[complex, ...]
    ) -> complex:
        self.calls.append((tuple(factors), tuple(values)))
        return self.value


def test_default_zne_fits_conditional_and_unconditional_sequences_separately() -> None:
    setting = ("A0",)
    samples = 3
    zne = _RecordingZNE()
    result = bootstrap_bell_results(
        _reference_inputs_for_factors(
            {
                1: {setting: {"00": 8, "11": 2}},
                3: {setting: {"00": 6, "11": 4}},
            }
        ),
        BootstrapConfig(samples=samples, seed=4),
        zne_strategy=zne,
    )

    assert result.raw is result.raw_conditional
    assert result.raw_unconditional is not None
    assert result.zne is result.zne_conditional
    assert result.zne_unconditional is not None
    assert zne.calls[0][1] == pytest.approx((1.0 + 0.0j, 1.0 + 0.0j))
    assert zne.calls[1][1] == pytest.approx((0.8 + 0.0j, 0.6 + 0.0j))
    assert len(zne.calls) == 2 * (samples + 1)
    assert result.diagnostics.zne_fit_calls == len(zne.calls)


def test_default_zne_with_readout_reports_all_conditional_and_unconditional_aliases() -> None:
    setting = ("A0",)
    samples = 2
    readout = _SignedReferenceReadout()
    zne = _RecordingZNE()
    result = bootstrap_bell_results(
        _reference_inputs_for_factors(
            {
                1: {setting: {"00": 8, "11": 2}},
                3: {setting: {"00": 6, "11": 4}},
            },
            readout_calibration=_reference_calibration(),
        ),
        BootstrapConfig(
            samples=samples,
            seed=7,
            include_readout_calibration=True,
        ),
        readout_strategy=readout,
        zne_strategy=zne,
    )

    assert result.raw is result.raw_conditional
    assert result.readout_mitigated is result.readout_mitigated_conditional
    assert result.zne is result.zne_conditional
    assert result.zne_readout_mitigated is result.zne_readout_mitigated_conditional
    assert result.raw_unconditional is not None
    assert result.readout_mitigated_unconditional is not None
    assert result.zne_unconditional is not None
    assert result.zne_readout_mitigated_unconditional is not None
    assert result.readout_effective_invalid_codeword_weight is not None
    assert result.readout_effective_invalid_codeword_weight.estimate == pytest.approx(-0.25)
    assert readout.resample_calls == samples
    assert len(readout.applied_contexts) == 2 * (samples + 1)
    assert len(zne.calls) == 4 * (samples + 1)
    assert result.diagnostics.zne_fit_calls == len(zne.calls)


def test_injected_zne_strategy_estimates_original_points_and_every_replicate() -> None:
    zne = _ConstantZNE(42.0 - 7.0j)
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={
                1: {"setting": {"0": 8, "1": 2}},
                3: {"setting": {"0": 6, "1": 4}},
            },
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_calibration(),
        ),
        BootstrapConfig(samples=2, seed=4, include_readout_calibration=False),
        readout_strategy=_RecordingReadout(),
        zne_strategy=zne,
        _evaluator=_single_probability,
    )

    assert result.zne is not None
    assert result.zne_readout_mitigated is not None
    for estimate in (result.zne, result.zne_readout_mitigated):
        assert estimate.estimate.real == 42.0
        assert estimate.estimate.imag == -7.0
        assert estimate.standard_error.real == 0.0
        assert estimate.standard_error.imag == 0.0
        assert estimate.confidence_interval.real.low == 42.0
        assert estimate.confidence_interval.real.high == 42.0
        assert estimate.confidence_interval.imag.low == -7.0
        assert estimate.confidence_interval.imag.high == -7.0
    assert len(zne.calls) == 2 * (2 + 1)
    assert result.diagnostics.zne_fit_calls == 2 * (2 + 1)


def test_full_plan_resamples_locally_and_produces_exactly_four_named_variants() -> None:
    inputs = BootstrapInputs(
        counts_by_factor={
            5: {"setting": {"0": 5, "1": 5}},
            1: {"setting": {"0": 8, "1": 2}},
            3: {"setting": {"0": 6, "1": 4}},
        },
        terms=(),
        qutrit_bit_indices_by_setting={},
        readout_calibration=_calibration(),
    )
    readout = _RecordingReadout()
    zne = _RecordingZNE()

    result = bootstrap_bell_results(
        inputs,
        BootstrapConfig(samples=4, seed=19, include_readout_calibration=True),
        readout_strategy=readout,
        zne_strategy=zne,
        _evaluator=_single_probability,
    )

    assert tuple(inputs.counts_by_factor) == (1, 3, 5)
    assert set(result.to_safe_dict()) == {
        "raw",
        "readout_mitigated",
        "zne",
        "zne_readout_mitigated",
        "config",
        "diagnostics",
    }
    assert result.diagnostics.calibration_resamples == 4
    assert result.diagnostics.zne_fit_calls == 2 * (4 + 1)
    expected_zne_point = linear_zne_extrapolate((1, 3, 5), (0.2, 0.4, 0.5))[0]
    assert result.raw.estimate.real == pytest.approx(0.2)
    assert result.readout_mitigated is not None
    assert result.readout_mitigated.estimate.real == pytest.approx(0.2)
    assert result.zne is not None
    assert result.zne.estimate.real == pytest.approx(expected_zne_point.real)
    assert result.zne_readout_mitigated is not None
    assert result.zne_readout_mitigated.estimate.real == pytest.approx(
        expected_zne_point.real
    )
    assert readout.build_calls == 1
    assert readout.resample_calls == 4
    assert len(set(readout.rng_ids)) == 1
    assert len(zne.calls) == 2 * (4 + 1)
    assert readout.applied_contexts[:3] == [("fixed", 1)] * 3
    for replicate in range(4):
        start = 3 + 3 * replicate
        assert readout.applied_contexts[start : start + 3] == [
            ("resampled", replicate + 1)
        ] * 3
    forbidden = {"backend", "adapter", "job", "job_id"}
    assert forbidden.isdisjoint(inspect.signature(bootstrap_bell_results).parameters)
    assert forbidden.isdisjoint(BootstrapInputs.__dataclass_fields__)


def test_fixed_calibration_context_is_built_once_and_reused_for_all_resamples() -> None:
    readout = _RecordingReadout()
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_calibration(),
        ),
        BootstrapConfig(samples=3, seed=4, include_readout_calibration=False),
        readout_strategy=readout,
        _evaluator=_single_probability,
    )

    assert result.readout_mitigated is not None
    assert result.zne is None
    assert result.zne_readout_mitigated is None
    assert result.diagnostics.calibration_resamples == 0
    assert readout.build_calls == 1
    assert readout.resample_calls == 0
    assert readout.applied_contexts == [("fixed", 1)] * 4


def test_zne_without_readout_produces_only_raw_and_zne_variants() -> None:
    zne = _RecordingZNE()
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={
                1: {"setting": {"0": 8, "1": 2}},
                3: {"setting": {"0": 6, "1": 4}},
            },
            terms=(),
            qutrit_bit_indices_by_setting={},
        ),
        BootstrapConfig(samples=3, seed=4),
        zne_strategy=zne,
        _evaluator=_single_probability,
    )

    assert set(result.to_safe_dict()) == {"raw", "zne", "config", "diagnostics"}
    assert result.readout_mitigated is None
    assert result.zne_readout_mitigated is None
    assert result.diagnostics.zne_fit_calls == 3 + 1
    assert len(zne.calls) == 3 + 1


def _two_components(counts_by_setting: Mapping[object, Mapping[str, float]]) -> complex:
    counts = counts_by_setting["setting"]
    total = sum(counts.values())
    return complex(counts.get("01", 0.0) / total, counts.get("10", 0.0) / total)


def test_component_summaries_use_ddof_one_and_separate_percentiles() -> None:
    config = BootstrapConfig(samples=5, confidence_level=0.6, seed=21)
    counts = {"00": 2, "01": 1, "10": 1}
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": counts}},
            terms=(),
            qutrit_bit_indices_by_setting={},
        ),
        config,
        _evaluator=_two_components,
    )
    rng = np.random.default_rng(21)
    draws = np.asarray([rng.multinomial(4, (0.5, 0.25, 0.25)) for _ in range(5)])
    expected_real = draws[:, 1] / 4
    expected_imag = draws[:, 2] / 4

    assert result.raw.estimate.real == 0.25
    assert result.raw.estimate.imag == 0.25
    assert result.raw.standard_error.real == pytest.approx(np.std(expected_real, ddof=1))
    assert result.raw.standard_error.imag == pytest.approx(np.std(expected_imag, ddof=1))
    assert result.raw.confidence_interval.real.low == pytest.approx(
        np.quantile(expected_real, 0.2)
    )
    assert result.raw.confidence_interval.real.high == pytest.approx(
        np.quantile(expected_real, 0.8)
    )
    assert result.raw.confidence_interval.imag.low == pytest.approx(
        np.quantile(expected_imag, 0.2)
    )
    assert result.raw.confidence_interval.imag.high == pytest.approx(
        np.quantile(expected_imag, 0.8)
    )


def test_seed_controls_replicates_but_never_the_original_count_point_estimate() -> None:
    inputs = BootstrapInputs(
        counts_by_factor={1: {"setting": {"0": 7, "1": 3}}},
        terms=(),
        qutrit_bit_indices_by_setting={},
    )
    first = bootstrap_bell_results(
        inputs, BootstrapConfig(samples=12, seed=1), _evaluator=_single_probability
    )
    repeated = bootstrap_bell_results(
        inputs, BootstrapConfig(samples=12, seed=1), _evaluator=_single_probability
    )
    changed = bootstrap_bell_results(
        inputs, BootstrapConfig(samples=20, seed=2), _evaluator=_single_probability
    )

    assert first == repeated
    assert first.raw.estimate == changed.raw.estimate
    assert first.raw.standard_error != changed.raw.standard_error


def test_standard_bell_evaluator_uses_saved_metadata_without_injected_callable() -> None:
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {(0,): {"00": 10}}},
            terms=({"coeff": 1.0, "settings": (0,), "powers": (1,)},),
            qutrit_bit_indices_by_setting={(0,): ((0, 1),)},
        ),
        BootstrapConfig(samples=3, seed=8),
    )

    assert result.raw.estimate.real == pytest.approx(1.0)
    assert result.raw.estimate.imag == pytest.approx(0.0)
    assert result.raw.standard_error.real == pytest.approx(0.0)


@pytest.mark.parametrize(
    "counts_by_factor",
    [
        {},
        {3: {"setting": {"0": 1}}},
        {1: {"setting": {"0": 1}}, 2: {"setting": {"0": 1}}},
        {True: {"setting": {"0": 1}}},
        {1: {"setting": {"0": 1}}, 3: {"other": {"0": 1}}},
        {1: {"setting-a": {"0": 1}, "setting-b": {"0": 1}}, 3: {"setting-b": {"0": 1}, "setting-a": {"0": 1}}},
    ],
)
def test_inputs_reject_invalid_factors_or_nonidentical_setting_order(
    counts_by_factor: object,
) -> None:
    with pytest.raises(ExperimentValidationError):
        BootstrapInputs(
            counts_by_factor=counts_by_factor,  # type: ignore[arg-type]
            terms=(),
            qutrit_bit_indices_by_setting={},
        )


@pytest.mark.parametrize(
    "factors",
    [(1,), (1, 7), (1, 3, 7), (15, 1, 9)],
)
def test_inputs_accept_and_sort_any_unique_positive_odd_factors_including_one(
    factors: tuple[int, ...],
) -> None:
    inputs = BootstrapInputs(
        counts_by_factor={factor: {"setting": {"0": 1}} for factor in factors},
        terms=(),
        qutrit_bit_indices_by_setting={},
    )

    assert tuple(inputs.counts_by_factor) == tuple(sorted(factors))


class _DuplicateFactorMapping(dict[int, dict[str, dict[str, int]]]):
    def __iter__(self) -> Iterator[int]:
        return iter((1, 3, 3))


def test_inputs_reject_duplicate_factors_from_mapping_contract_violation() -> None:
    with pytest.raises(ExperimentValidationError, match="unique"):
        BootstrapInputs(
            counts_by_factor=_DuplicateFactorMapping(
                {1: {"setting": {"0": 1}}, 3: {"setting": {"0": 1}}}
            ),
            terms=(),
            qutrit_bit_indices_by_setting={},
        )


@pytest.mark.parametrize(
    "setting_counts",
    [
        {},
        {"": 1},
        {"02": 1},
        {2: 1},
        {"0": -1},
        {"0": True},
        {"0": 1.0},
        {"0": 0, "1": 0},
    ],
)
def test_inputs_reject_invalid_bitstrings_counts_and_shot_totals(setting_counts: object) -> None:
    with pytest.raises(ExperimentValidationError):
        BootstrapInputs(
            counts_by_factor={1: {"setting": setting_counts}},  # type: ignore[dict-item]
            terms=(),
            qutrit_bit_indices_by_setting={},
        )


def test_inputs_accept_setting_total_at_multinomial_int64_boundary() -> None:
    max_shots = int(np.iinfo(np.int64).max)

    inputs = BootstrapInputs(
        counts_by_factor={1: {"setting": {"0": max_shots}}},
        terms=(),
        qutrit_bit_indices_by_setting={},
    )

    assert inputs.counts_by_factor[1]["setting"]["0"] == max_shots


@pytest.mark.parametrize(
    "setting_counts",
    [
        {"0": int(np.iinfo(np.int64).max) + 1},
        {"0": int(np.iinfo(np.int64).max), "1": 1},
    ],
)
def test_inputs_reject_setting_totals_above_multinomial_int64_boundary(
    setting_counts: Mapping[str, int],
) -> None:
    with pytest.raises(
        ExperimentValidationError,
        match="supported bootstrap multinomial range",
    ):
        BootstrapInputs(
            counts_by_factor={1: {"setting": setting_counts}},
            terms=(),
            qutrit_bit_indices_by_setting={},
        )


def test_count_resampling_failures_are_typed_and_sanitized() -> None:
    class FailingRng:
        def multinomial(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("api_key=provider-secret")

    with pytest.raises(JobResultError, match="bootstrap count resampling failed") as caught:
        _resample_counts(
            {1: {"setting": {"0": 1}}},
            FailingRng(),  # type: ignore[arg-type]
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_high_shot_resampling_preserves_rare_outcome_probability() -> None:
    total = 2**53 + 1

    sampled = _resample_counts(
        {1: {"setting": {"00": 2**53, "11": 1}}},
        np.random.default_rng(0),
    )

    draw = sampled[1]["setting"]
    assert draw["11"] > 0
    assert sum(draw.values()) == total


def test_ordinary_resampling_preserves_legacy_seeded_draw() -> None:
    observed = np.asarray((8, 2), dtype=np.int64)
    expected = np.random.default_rng(17).multinomial(10, observed / 10)

    sampled = _resample_counts(
        {1: {"setting": {"0": 8, "1": 2}}},
        np.random.default_rng(17),
    )

    assert tuple(sampled[1]["setting"].values()) == tuple(int(value) for value in expected)


def test_high_shot_readout_calibration_resampling_preserves_rare_outcome() -> None:
    class PassthroughReadout(_Task6ReadoutBootstrap):
        def build_context(self, calibration: ReadoutCalibration) -> ReadoutCalibration:
            return calibration

    total = 2**53 + 1
    raw_counts = (
        {"0": 2**53, "1": 1},
        {"0": 0, "1": total},
    )
    calibration = ReadoutCalibration(
        backend_identity="backend-a",
        calibration_id="cal-high-shots",
        qubit_mapping=(0,),
        timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
        shots=total,
        raw_counts=raw_counts,
        assignment_matrices=assignment_matrices_from_counts(
            (0,), raw_counts, shots=total
        ),
    )

    resampled = PassthroughReadout(None).resample_calibration(
        calibration,
        np.random.default_rng(0),
    )

    assert isinstance(resampled, ReadoutCalibration)
    assert resampled.raw_counts[0]["1"] > 0
    assert all(sum(counts.values()) == total for counts in resampled.raw_counts)


def test_inputs_deep_copy_mutable_bell_metadata() -> None:
    terms = [{"settings": ["setting"], "powers": [1], "nested": {"value": 2}}]
    indices = {"setting": [(0, 1)]}
    decoding = {"outcome_map": {0: 0}}
    inputs = BootstrapInputs(
        counts_by_factor={1: {"setting": {"00": 1}}},
        terms=terms,
        qutrit_bit_indices_by_setting=indices,
        decoding_kwargs=decoding,
    )

    terms[0]["nested"]["value"] = 9
    indices["setting"].append((2, 3))
    decoding["outcome_map"][0] = 1
    assert inputs.terms[0]["nested"]["value"] == 2
    assert inputs.qutrit_bit_indices_by_setting["setting"] == ((0, 1),)
    assert inputs.decoding_kwargs["outcome_map"][0] == 0


class _SignedReadout(_RecordingReadout):
    def apply(
        self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
    ) -> Mapping[object, Mapping[str, float]]:
        self.applied_contexts.append(context)
        return {setting: {"0": 1.25, "1": -0.25} for setting in counts_by_setting}


def test_signed_corrected_quasi_counts_are_passed_to_bell_evaluator_unchanged() -> None:
    seen_negative: list[bool] = []

    def evaluator(counts_by_setting: Mapping[object, Mapping[str, float]]) -> complex:
        seen_negative.append(counts_by_setting["setting"].get("1", 0.0) < 0)
        return complex(counts_by_setting["setting"].get("1", 0.0))

    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_calibration(),
        ),
        BootstrapConfig(samples=2, seed=2, include_readout_calibration=False),
        readout_strategy=_SignedReadout(),
        _evaluator=evaluator,
    )

    assert result.readout_mitigated is not None
    assert result.readout_mitigated.estimate.real == -0.25
    assert any(seen_negative)


def test_float32_readout_normalization_roundoff_is_accepted() -> None:
    class Float32Readout(_RecordingReadout):
        def apply(
            self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
        ) -> Mapping[object, Mapping[str, float]]:
            return {
                setting: {"0": np.float32(0.99999994)}
                for setting in counts_by_setting
            }

    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": {"0": 10}}},
            terms=(),
            qutrit_bit_indices_by_setting={},
            readout_calibration=_calibration(),
        ),
        BootstrapConfig(samples=2, seed=2, include_readout_calibration=False),
        readout_strategy=Float32Readout(),
        _evaluator=_single_probability,
    )

    assert result.readout_mitigated is not None


@pytest.mark.parametrize("method", ["build", "resample", "apply"])
def test_readout_strategy_failures_are_typed_and_sanitized(method: str) -> None:
    class FailingReadout(_RecordingReadout):
        def build_context(self, calibration: ReadoutCalibration) -> object:
            if method == "build":
                raise RuntimeError("token=provider-secret")
            return super().build_context(calibration)

        def resample_calibration(
            self, calibration: ReadoutCalibration, rng: np.random.Generator
        ) -> object:
            if method == "resample":
                raise RuntimeError("token=provider-secret")
            return super().resample_calibration(calibration, rng)

        def apply(
            self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
        ) -> Mapping[object, Mapping[str, float]]:
            if method == "apply":
                raise RuntimeError("token=provider-secret")
            return super().apply(counts_by_setting, context)

    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
                readout_calibration=_calibration(),
            ),
            BootstrapConfig(samples=2, seed=2),
            readout_strategy=FailingReadout(),
            _evaluator=_single_probability,
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "output",
    [
        {"setting": {}},
        {"setting": {"0": math.nan}},
        {"setting": {"0": True}},
        {"setting": {"2": 1.0}},
        {"other": {"0": 1.0}},
    ],
)
def test_malformed_readout_outputs_are_rejected_before_bell_evaluation(output: object) -> None:
    class MalformedReadout(_RecordingReadout):
        def apply(self, counts_by_setting: object, context: object) -> object:
            return output

    with pytest.raises(JobResultError, match="readout mitigation produced invalid counts"):
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
                readout_calibration=_calibration(),
            ),
            BootstrapConfig(samples=2, include_readout_calibration=False),
            readout_strategy=MalformedReadout(),
            _evaluator=_single_probability,
        )


def test_zne_strategy_failure_is_typed_sanitized_and_requires_two_factors() -> None:
    class FailingZNE:
        def extrapolate(self, factors: object, values: object) -> complex:
            raise RuntimeError("password=provider-secret")

    single = BootstrapInputs(
        counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
        terms=(),
        qutrit_bit_indices_by_setting={},
    )
    with pytest.raises(ExperimentValidationError, match="at least two"):
        bootstrap_bell_results(
            single,
            BootstrapConfig(samples=2),
            zne_strategy=FailingZNE(),
            _evaluator=_single_probability,
        )

    multiple = BootstrapInputs(
        counts_by_factor={
            1: {"setting": {"0": 8, "1": 2}},
            3: {"setting": {"0": 6, "1": 4}},
        },
        terms=(),
        qutrit_bit_indices_by_setting={},
    )
    with pytest.raises(JobResultError, match="ZNE fit") as caught:
        bootstrap_bell_results(
            multiple,
            BootstrapConfig(samples=2),
            zne_strategy=FailingZNE(),
            _evaluator=_single_probability,
        )
    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_bell_evaluator_failure_is_typed_and_sanitized() -> None:
    def failing_evaluator(counts: object) -> complex:
        raise RuntimeError("api_key=provider-secret")

    with pytest.raises(JobResultError, match="Bell evaluation") as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 1}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            _evaluator=failing_evaluator,
        )
    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


class _HostileComplex:
    def __complex__(self) -> complex:
        raise RuntimeError("api_key=provider-secret")


def test_custom_evaluator_hostile_complex_conversion_is_typed_and_sanitized() -> None:
    def evaluator(counts: object) -> complex:
        return _HostileComplex()  # type: ignore[return-value]

    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 1}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            _evaluator=evaluator,
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_zne_hostile_complex_conversion_is_typed_and_sanitized() -> None:
    class HostileZNE:
        def extrapolate(self, factors: object, values: object) -> complex:
            return _HostileComplex()  # type: ignore[return-value]

    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={
                    1: {"setting": {"0": 8, "1": 2}},
                    3: {"setting": {"0": 6, "1": 4}},
                },
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            zne_strategy=HostileZNE(),
            _evaluator=_single_probability,
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_zne_hostile_tuple_parsing_is_typed_and_sanitized() -> None:
    class HostileTuple(tuple[object, ...]):
        def __len__(self) -> int:
            raise RuntimeError("api_key=provider-secret")

    class HostileZNE:
        def extrapolate(self, factors: object, values: object) -> complex:
            return HostileTuple((1.0 + 0.0j, object()))  # type: ignore[return-value]

    with pytest.raises(JobResultError, match="bootstrap ZNE fit failed") as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={
                    1: {"setting": {"0": 8, "1": 2}},
                    3: {"setting": {"0": 6, "1": 4}},
                },
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            zne_strategy=HostileZNE(),
            _evaluator=_single_probability,
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "target",
    ["compute_bell_value_from_counts", "invalid_codeword_statistics"],
)
def test_default_reference_metric_failures_are_typed_and_sanitized(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    uncertainty_module = sys.modules[bootstrap_bell_results.__module__]

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("api_key=provider-secret")

    monkeypatch.setattr(uncertainty_module, target, fail)
    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(
            _reference_inputs_with_invalid_counts(),
            BootstrapConfig(samples=2),
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


class _HostileInvalidStatistics:
    @property
    def invalid_fraction(self) -> float:
        raise RuntimeError("api_key=provider-secret")


@pytest.mark.parametrize("statistics", [object(), _HostileInvalidStatistics()])
def test_malformed_invalid_statistics_results_are_typed_and_sanitized(
    statistics: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    uncertainty_module = sys.modules[bootstrap_bell_results.__module__]
    monkeypatch.setattr(
        uncertainty_module,
        "invalid_codeword_statistics",
        lambda *args, **kwargs: statistics,
    )

    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(
            _reference_inputs_with_invalid_counts(),
            BootstrapConfig(samples=2),
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_default_decoding_failure_is_typed_and_sanitized() -> None:
    setting = ("A0",)
    inputs = BootstrapInputs(
        counts_by_factor={1: {setting: {"00": 8, "11": 2}}},
        terms=({"coeff": 1.0, "settings": setting, "powers": (0,)},),
        qutrit_bit_indices_by_setting={setting: ((0, 2),)},
        decoding_kwargs={
            "bit_order": "left-to-right",
            "outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
            "d": 3,
        },
    )

    with pytest.raises(JobResultError) as caught:
        bootstrap_bell_results(inputs, BootstrapConfig(samples=2))

    assert "out of range" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_results_are_frozen_and_default_to_2000_local_samples() -> None:
    assert BootstrapConfig().samples == 2000
    result = bootstrap_bell_results(
        BootstrapInputs(
            counts_by_factor={1: {"setting": {"0": 1}}},
            terms=(),
            qutrit_bit_indices_by_setting={},
        ),
        BootstrapConfig(samples=2),
        _evaluator=_single_probability,
    )
    with pytest.raises(FrozenInstanceError):
        result.raw = result.raw  # type: ignore[misc]


@pytest.mark.parametrize("seed", [True, 1.5, [1, 2], -1])
def test_bootstrap_rejects_non_integer_or_negative_rng_seed(seed: object) -> None:
    with pytest.raises(ExperimentValidationError, match="seed"):
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 1}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2, seed=seed),  # type: ignore[arg-type]
            _evaluator=_single_probability,
        )


def test_count_resampling_memory_error_is_propagated() -> None:
    failure = MemoryError("bootstrap allocation failed")

    class FailingRng:
        def multinomial(self, *args: object, **kwargs: object) -> object:
            raise failure

    with pytest.raises(MemoryError) as caught:
        _resample_counts(
            {1: {"setting": {"0": 1}}},
            FailingRng(),  # type: ignore[arg-type]
        )

    assert caught.value is failure


@pytest.mark.parametrize("method", ["build", "resample", "apply"])
def test_readout_strategy_memory_errors_are_propagated(method: str) -> None:
    failure = MemoryError(f"readout {method} allocation failed")

    class FailingReadout(_RecordingReadout):
        def build_context(self, calibration: ReadoutCalibration) -> object:
            if method == "build":
                raise failure
            return super().build_context(calibration)

        def resample_calibration(
            self, calibration: ReadoutCalibration, rng: np.random.Generator
        ) -> object:
            if method == "resample":
                raise failure
            return super().resample_calibration(calibration, rng)

        def apply(
            self, counts_by_setting: Mapping[object, Mapping[str, int]], context: object
        ) -> Mapping[object, Mapping[str, float]]:
            if method == "apply":
                raise failure
            return super().apply(counts_by_setting, context)

    with pytest.raises(MemoryError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 8, "1": 2}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
                readout_calibration=_calibration(),
            ),
            BootstrapConfig(samples=2, seed=2),
            readout_strategy=FailingReadout(),
            _evaluator=_single_probability,
        )

    assert caught.value is failure


def test_zne_strategy_memory_error_is_propagated() -> None:
    failure = MemoryError("ZNE allocation failed")

    class FailingZNE:
        def extrapolate(self, factors: object, values: object) -> complex:
            raise failure

    with pytest.raises(MemoryError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={
                    1: {"setting": {"0": 8, "1": 2}},
                    3: {"setting": {"0": 6, "1": 4}},
                },
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            zne_strategy=FailingZNE(),
            _evaluator=_single_probability,
        )

    assert caught.value is failure


def test_custom_bell_evaluator_memory_error_is_propagated() -> None:
    failure = MemoryError("Bell evaluation allocation failed")

    def failing_evaluator(counts: object) -> complex:
        raise failure

    with pytest.raises(MemoryError) as caught:
        bootstrap_bell_results(
            BootstrapInputs(
                counts_by_factor={1: {"setting": {"0": 1}}},
                terms=(),
                qutrit_bit_indices_by_setting={},
            ),
            BootstrapConfig(samples=2),
            _evaluator=failing_evaluator,
        )

    assert caught.value is failure


@pytest.mark.parametrize(
    "target",
    [
        "compute_bell_value_from_counts",
        "invalid_codeword_statistics",
        "bitstring_to_qutrit_outcomes",
    ],
)
def test_structured_reference_memory_errors_are_propagated(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = MemoryError(f"{target} allocation failed")
    uncertainty_module = sys.modules[bootstrap_bell_results.__module__]

    def fail(*args: object, **kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(uncertainty_module, target, fail)
    with pytest.raises(MemoryError) as caught:
        bootstrap_bell_results(
            _reference_inputs_with_invalid_counts(),
            BootstrapConfig(samples=2),
        )

    assert caught.value is failure
