"""Local parametric bootstrap uncertainty for saved Bell-count evidence."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from ..bell_measurements import compute_bell_value_from_counts
from .errors import ExperimentValidationError, JobResultError
from .mitigation import (
    ReadoutCalibration,
    apply_readout_mitigation,
    assignment_matrices_from_counts,
    build_m3_mitigation,
    linear_zne_extrapolate,
    validate_zne_factors,
)
from .models import (
    BellEstimate,
    BootstrapConfig,
    ComplexComponents,
    ComplexConfidenceInterval,
    ConfidenceInterval,
)


Setting = Hashable
RawCounts = Mapping[Setting, Mapping[str, int]]
WeightedCounts = Mapping[Setting, Mapping[str, float]]
BellEvaluator = Callable[[WeightedCounts], complex]


class ReadoutBootstrapStrategy(Protocol):
    """Build fixed or resampled correction contexts, then apply them."""

    def build_context(self, calibration: ReadoutCalibration) -> object:
        """Build one correction context from original calibration evidence."""

    def resample_calibration(
        self, calibration: ReadoutCalibration, rng: np.random.Generator
    ) -> object:
        """Resample all calibration distributions and return one context."""

    def apply(self, counts_by_setting: RawCounts, context: object) -> WeightedCounts:
        """Correct every setting using the supplied shared context."""


class ZNEBootstrapStrategy(Protocol):
    """Extrapolate one set of factor values to its zero-noise intercept."""

    def extrapolate(
        self, factors: Sequence[int], values: Sequence[complex]
    ) -> tuple[complex, object] | complex:
        """Return a finite complex intercept, optionally with fit evidence."""


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    return value


def _validated_counts_by_factor(
    value: object,
) -> Mapping[int, Mapping[Setting, Mapping[str, int]]]:
    if not isinstance(value, Mapping) or not value:
        raise ExperimentValidationError("counts_by_factor must be a non-empty mapping")
    factors = validate_zne_factors(tuple(value))

    expected_settings: tuple[Setting, ...] | None = None
    normalized_factors: dict[int, Mapping[Setting, Mapping[str, int]]] = {}
    expected_width: int | None = None
    for factor in factors:
        factor_counts = value[factor]
        if not isinstance(factor_counts, Mapping) or not factor_counts:
            raise ExperimentValidationError("each factor must contain setting counts")
        settings = tuple(factor_counts)
        if expected_settings is None:
            expected_settings = settings
        elif settings != expected_settings:
            raise ExperimentValidationError("settings and their order must be identical across factors")

        normalized_settings: dict[Setting, Mapping[str, int]] = {}
        for setting, setting_counts in factor_counts.items():
            try:
                hash(setting)
            except TypeError as error:
                raise ExperimentValidationError("setting keys must be hashable") from error
            if not isinstance(setting_counts, Mapping) or not setting_counts:
                raise ExperimentValidationError("setting counts must be non-empty mappings")
            normalized_counts: dict[str, int] = {}
            for outcome, count in setting_counts.items():
                if (
                    not isinstance(outcome, str)
                    or not outcome
                    or any(bit not in "01" for bit in outcome)
                ):
                    raise ExperimentValidationError("count outcomes must be non-empty bitstrings")
                if expected_width is None:
                    expected_width = len(outcome)
                elif len(outcome) != expected_width:
                    raise ExperimentValidationError("all count bitstrings must have identical length")
                if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                    raise ExperimentValidationError("raw counts must be non-negative integers")
                normalized_counts[outcome] = int(count)
            if sum(normalized_counts.values()) <= 0:
                raise ExperimentValidationError("setting count total must be positive")
            normalized_settings[setting] = MappingProxyType(normalized_counts)
        normalized_factors[factor] = MappingProxyType(normalized_settings)
    return MappingProxyType(normalized_factors)


@dataclass(frozen=True)
class BootstrapInputs:
    """Saved count evidence and pure Bell metadata; never a backend or job."""

    counts_by_factor: Mapping[int, RawCounts]
    terms: Sequence[Mapping[str, object]]
    qutrit_bit_indices_by_setting: Mapping[Setting, Sequence[tuple[int, int]]]
    decoding_kwargs: Mapping[str, object] = field(default_factory=dict)
    readout_calibration: ReadoutCalibration | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "counts_by_factor", _validated_counts_by_factor(self.counts_by_factor)
        )
        if not isinstance(self.terms, Sequence) or isinstance(self.terms, (str, bytes)):
            raise ExperimentValidationError("terms must be a sequence")
        if any(not isinstance(term, Mapping) for term in self.terms):
            raise ExperimentValidationError("terms must contain mappings")
        object.__setattr__(self, "terms", tuple(_freeze_metadata(term) for term in self.terms))
        if not isinstance(self.qutrit_bit_indices_by_setting, Mapping):
            raise ExperimentValidationError("qutrit_bit_indices_by_setting must be a mapping")
        object.__setattr__(
            self,
            "qutrit_bit_indices_by_setting",
            _freeze_metadata(self.qutrit_bit_indices_by_setting),
        )
        if not isinstance(self.decoding_kwargs, Mapping):
            raise ExperimentValidationError("decoding_kwargs must be a mapping")
        object.__setattr__(self, "decoding_kwargs", _freeze_metadata(self.decoding_kwargs))
        if self.readout_calibration is not None and not isinstance(
            self.readout_calibration, ReadoutCalibration
        ):
            raise ExperimentValidationError("readout_calibration must be ReadoutCalibration")
        if self.readout_calibration is not None:
            first_setting = next(iter(next(iter(self.counts_by_factor.values())).values()))
            width = len(next(iter(first_setting)))
            if width != len(self.readout_calibration.qubit_mapping):
                raise ExperimentValidationError(
                    "count bitstring width must match readout calibration mapping"
                )


@dataclass(frozen=True)
class BootstrapDiagnostics:
    factors: tuple[int, ...]
    settings_per_factor: int
    calibration_resamples: int = 0
    zne_fit_calls: int = 0

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "factors": list(self.factors),
            "settings_per_factor": self.settings_per_factor,
            "calibration_resamples": self.calibration_resamples,
            "zne_fit_calls": self.zne_fit_calls,
        }


@dataclass(frozen=True)
class BootstrapBellResults:
    raw: BellEstimate
    config: BootstrapConfig
    diagnostics: BootstrapDiagnostics
    readout_mitigated: BellEstimate | None = None
    zne: BellEstimate | None = None
    zne_readout_mitigated: BellEstimate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, BellEstimate):
            raise ExperimentValidationError("raw must be BellEstimate")
        if not isinstance(self.config, BootstrapConfig):
            raise ExperimentValidationError("config must be BootstrapConfig")
        if not isinstance(self.diagnostics, BootstrapDiagnostics):
            raise ExperimentValidationError("diagnostics must be BootstrapDiagnostics")
        for name in ("readout_mitigated", "zne", "zne_readout_mitigated"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BellEstimate):
                raise ExperimentValidationError(f"{name} must be BellEstimate or None")

    def to_safe_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"raw": self.raw.to_safe_dict()}
        for name in ("readout_mitigated", "zne", "zne_readout_mitigated"):
            estimate = getattr(self, name)
            if estimate is not None:
                payload[name] = estimate.to_safe_dict()
        payload["config"] = self.config.to_safe_dict()
        payload["diagnostics"] = self.diagnostics.to_safe_dict()
        return payload


def _resample_counts(
    counts_by_factor: Mapping[int, RawCounts], rng: np.random.Generator
) -> dict[int, dict[Setting, dict[str, int]]]:
    sampled: dict[int, dict[Setting, dict[str, int]]] = {}
    for factor, factor_counts in counts_by_factor.items():
        sampled_settings: dict[Setting, dict[str, int]] = {}
        for setting, counts in factor_counts.items():
            outcomes = tuple(counts)
            observed = np.asarray(tuple(counts.values()), dtype=np.int64)
            total = int(observed.sum())
            draw = rng.multinomial(total, observed / total)
            sampled_settings[setting] = {
                outcome: int(count) for outcome, count in zip(outcomes, draw, strict=True)
            }
        sampled[factor] = sampled_settings
    return sampled


@dataclass(frozen=True)
class _Task6ReadoutContext:
    calibration: ReadoutCalibration
    mitigation: object


class _Task6ReadoutBootstrap:
    """Adapt Task 6 M3 helpers to the bootstrap-context protocol."""

    def build_context(self, calibration: ReadoutCalibration) -> _Task6ReadoutContext:
        return _Task6ReadoutContext(calibration, build_m3_mitigation(calibration))

    def resample_calibration(
        self, calibration: ReadoutCalibration, rng: np.random.Generator
    ) -> _Task6ReadoutContext:
        raw_counts: list[dict[str, int]] = []
        for counts in calibration.raw_counts:
            observed = np.asarray((counts["0"], counts["1"]), dtype=np.int64)
            draw = rng.multinomial(calibration.shots, observed / calibration.shots)
            raw_counts.append({"0": int(draw[0]), "1": int(draw[1])})
        matrices = assignment_matrices_from_counts(
            calibration.qubit_mapping, raw_counts, shots=calibration.shots
        )
        resampled = ReadoutCalibration(
            backend_identity=calibration.backend_identity,
            calibration_id=calibration.calibration_id,
            qubit_mapping=calibration.qubit_mapping,
            timestamp=calibration.timestamp,
            shots=calibration.shots,
            raw_counts=tuple(raw_counts),
            assignment_matrices=matrices,
        )
        return self.build_context(resampled)

    def apply(
        self, counts_by_setting: RawCounts, context: object
    ) -> dict[Setting, dict[str, float]]:
        if not isinstance(context, _Task6ReadoutContext):
            raise ExperimentValidationError("readout context is invalid")
        original_settings = tuple(counts_by_setting)
        encoded = {
            f"setting-{index}": counts_by_setting[setting]
            for index, setting in enumerate(original_settings)
        }
        corrected = apply_readout_mitigation(
            encoded,
            mapping=context.calibration.qubit_mapping,
            mitigation=context.mitigation,  # type: ignore[arg-type]
        )
        return {
            setting: corrected[f"setting-{index}"]
            for index, setting in enumerate(original_settings)
        }


class _Task6ZNEBootstrap:
    def extrapolate(
        self, factors: Sequence[int], values: Sequence[complex]
    ) -> tuple[complex, object]:
        return linear_zne_extrapolate(factors, values)


def _finite_complex(value: object, operation: str) -> complex:
    try:
        normalized = complex(value)
    except (TypeError, ValueError, OverflowError):
        raise JobResultError(f"bootstrap {operation} produced an invalid value") from None
    if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
        raise JobResultError(f"bootstrap {operation} produced an invalid value")
    return normalized


def _summary(point: complex, replicates: Sequence[complex], confidence: float) -> BellEstimate:
    values = np.asarray(replicates, dtype=complex)
    alpha = (1.0 - confidence) / 2.0
    real_low, real_high = np.quantile(values.real, (alpha, 1.0 - alpha))
    imag_low, imag_high = np.quantile(values.imag, (alpha, 1.0 - alpha))
    return BellEstimate(
        estimate=ComplexComponents(float(point.real), float(point.imag)),
        standard_error=ComplexComponents(
            float(np.std(values.real, ddof=1)), float(np.std(values.imag, ddof=1))
        ),
        confidence_interval=ComplexConfidenceInterval(
            real=ConfidenceInterval(float(real_low), float(real_high)),
            imag=ConfidenceInterval(float(imag_low), float(imag_high)),
        ),
    )


def _default_evaluator(inputs: BootstrapInputs) -> BellEvaluator:
    def evaluate(counts_by_setting: WeightedCounts) -> complex:
        return compute_bell_value_from_counts(
            counts_by_setting,
            inputs.terms,
            inputs.qutrit_bit_indices_by_setting,
            **dict(inputs.decoding_kwargs),
        )

    return evaluate


def _evaluate(evaluator: BellEvaluator, counts: WeightedCounts) -> complex:
    try:
        value = evaluator(counts)
    except Exception:
        raise JobResultError("bootstrap Bell evaluation failed") from None
    return _finite_complex(value, "Bell evaluation")


def _readout_context(
    strategy: ReadoutBootstrapStrategy,
    calibration: ReadoutCalibration,
    *,
    rng: np.random.Generator | None,
) -> object:
    try:
        if rng is None:
            return strategy.build_context(calibration)
        return strategy.resample_calibration(calibration, rng)
    except Exception:
        raise JobResultError("bootstrap readout calibration failed") from None


def _correct_counts(
    strategy: ReadoutBootstrapStrategy, counts: RawCounts, context: object
) -> WeightedCounts:
    try:
        corrected = strategy.apply(counts, context)
        if not isinstance(corrected, Mapping) or tuple(corrected) != tuple(counts):
            raise ValueError
        normalized: dict[Setting, dict[str, float]] = {}
        expected_width = len(next(iter(next(iter(counts.values())))))
        for setting, weights in corrected.items():
            if not isinstance(weights, Mapping) or not weights:
                raise ValueError
            normalized_weights: dict[str, float] = {}
            for outcome, weight in weights.items():
                if (
                    not isinstance(outcome, str)
                    or len(outcome) != expected_width
                    or any(bit not in "01" for bit in outcome)
                    or isinstance(weight, bool)
                    or not isinstance(weight, Real)
                    or not math.isfinite(weight)
                ):
                    raise ValueError
                normalized_weights[outcome] = float(weight)
            if not math.isclose(
                sum(normalized_weights.values()), 1.0, rel_tol=1e-8, abs_tol=1e-8
            ):
                raise ValueError
            normalized[setting] = normalized_weights
    except Exception:
        raise JobResultError("bootstrap readout mitigation produced invalid counts") from None
    return normalized


def _zne_intercept(
    strategy: ZNEBootstrapStrategy,
    factors: tuple[int, ...],
    values: tuple[complex, ...],
) -> complex:
    try:
        output = strategy.extrapolate(factors, values)
    except Exception:
        raise JobResultError("bootstrap ZNE fit failed") from None
    estimate = output[0] if isinstance(output, tuple) and len(output) == 2 else output
    return _finite_complex(estimate, "ZNE fit")


def bootstrap_bell_results(
    inputs: BootstrapInputs,
    config: BootstrapConfig = BootstrapConfig(),
    *,
    readout_strategy: ReadoutBootstrapStrategy | None = None,
    zne_strategy: ZNEBootstrapStrategy | None = None,
    _evaluator: BellEvaluator | None = None,
) -> BootstrapBellResults:
    """Summarize Bell uncertainty from local resamples of already-saved counts."""

    if not isinstance(inputs, BootstrapInputs):
        raise ExperimentValidationError("inputs must be BootstrapInputs")
    if not isinstance(config, BootstrapConfig):
        raise ExperimentValidationError("config must be BootstrapConfig")
    if type(config.seed) is not int or config.seed < 0:
        raise ExperimentValidationError("bootstrap seed must be a non-negative integer")
    try:
        rng = np.random.default_rng(config.seed)
    except (TypeError, ValueError):
        raise ExperimentValidationError("bootstrap seed is invalid") from None
    evaluator = _default_evaluator(inputs) if _evaluator is None else _evaluator
    factors = tuple(inputs.counts_by_factor)
    use_readout = inputs.readout_calibration is not None
    use_zne = len(factors) > 1
    if readout_strategy is not None and not use_readout:
        raise ExperimentValidationError("readout_strategy requires readout_calibration")
    if zne_strategy is not None and not use_zne:
        raise ExperimentValidationError("ZNE requires at least two factors")

    readout = _Task6ReadoutBootstrap() if readout_strategy is None else readout_strategy
    zne = _Task6ZNEBootstrap() if zne_strategy is None else zne_strategy
    raw_points = tuple(
        _evaluate(evaluator, inputs.counts_by_factor[factor]) for factor in factors
    )
    readout_context: object | None = None
    corrected_points: tuple[complex, ...] = ()
    if use_readout:
        calibration = inputs.readout_calibration
        assert calibration is not None
        readout_context = _readout_context(readout, calibration, rng=None)
        corrected_points = tuple(
            _evaluate(
                evaluator,
                _correct_counts(readout, inputs.counts_by_factor[factor], readout_context),
            )
            for factor in factors
        )

    raw_point = raw_points[0]
    corrected_point = corrected_points[0] if corrected_points else None
    zne_fit_calls = 0
    zne_point = None
    corrected_zne_point = None
    if use_zne:
        zne_point = _zne_intercept(zne, factors, raw_points)
        zne_fit_calls += 1
        if use_readout:
            corrected_zne_point = _zne_intercept(zne, factors, corrected_points)
            zne_fit_calls += 1

    raw_replicates: list[complex] = []
    corrected_replicates: list[complex] = []
    zne_replicates: list[complex] = []
    corrected_zne_replicates: list[complex] = []
    calibration_resamples = 0
    for _ in range(config.samples):
        sampled = _resample_counts(inputs.counts_by_factor, rng)
        raw_values = tuple(_evaluate(evaluator, sampled[factor]) for factor in factors)
        raw_replicates.append(raw_values[0])

        corrected_values: tuple[complex, ...] = ()
        if use_readout:
            calibration = inputs.readout_calibration
            assert calibration is not None
            if config.include_readout_calibration:
                replicate_context = _readout_context(readout, calibration, rng=rng)
                calibration_resamples += 1
            else:
                replicate_context = readout_context
            corrected_values = tuple(
                _evaluate(
                    evaluator,
                    _correct_counts(readout, sampled[factor], replicate_context),
                )
                for factor in factors
            )
            corrected_replicates.append(corrected_values[0])

        if use_zne:
            zne_replicates.append(_zne_intercept(zne, factors, raw_values))
            zne_fit_calls += 1
            if use_readout:
                corrected_zne_replicates.append(
                    _zne_intercept(zne, factors, corrected_values)
                )
                zne_fit_calls += 1
    diagnostics = BootstrapDiagnostics(
        factors=factors,
        settings_per_factor=len(inputs.counts_by_factor[1]),
        calibration_resamples=calibration_resamples,
        zne_fit_calls=zne_fit_calls,
    )
    return BootstrapBellResults(
        raw=_summary(raw_point, raw_replicates, config.confidence_level),
        readout_mitigated=(
            _summary(corrected_point, corrected_replicates, config.confidence_level)
            if corrected_point is not None
            else None
        ),
        zne=(
            _summary(zne_point, zne_replicates, config.confidence_level)
            if zne_point is not None
            else None
        ),
        zne_readout_mitigated=(
            _summary(
                corrected_zne_point,
                corrected_zne_replicates,
                config.confidence_level,
            )
            if corrected_zne_point is not None
            else None
        ),
        config=config,
        diagnostics=diagnostics,
    )


__all__ = [
    "BootstrapBellResults",
    "BootstrapDiagnostics",
    "BootstrapInputs",
    "ReadoutBootstrapStrategy",
    "ZNEBootstrapStrategy",
    "bootstrap_bell_results",
]
