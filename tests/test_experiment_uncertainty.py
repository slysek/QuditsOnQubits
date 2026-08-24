from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
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
from qudits_on_qubits.experiments.mitigation import ReadoutCalibration
from qudits_on_qubits.experiments.mitigation import linear_zne_extrapolate
from qudits_on_qubits.experiments.models import BootstrapConfig
from qudits_on_qubits.experiments.uncertainty import (
    BootstrapBellResults,
    BootstrapInputs,
    bootstrap_bell_results,
)


def _single_probability(counts_by_setting: Mapping[object, Mapping[str, float]]) -> complex:
    counts = counts_by_setting["setting"]
    total = sum(counts.values())
    return complex(counts.get("1", 0.0) / total)


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
    assert "replicate" not in repr(payload).lower()
    json.dumps(payload, allow_nan=False)
    assert isinstance(inputs.counts_by_factor, MappingProxyType)
    source_counts[1]["setting"]["0"] = 1
    assert inputs.counts_by_factor[1]["setting"]["0"] == 12


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
