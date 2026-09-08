"""Schedule-conditional Hoeffding intervals with blocks as sampling units.

Shots within a block may be arbitrarily dependent. The simultaneous bound
does not assume independence between terms, including overlapping AME terms.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


METHOD = "conditional_schedule_block_hoeffding_v1"


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _components(value: Mapping[str, Any], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"real", "imag"}:
        raise ValueError(f"{name} must have real and imag components")
    return {part: _finite_real(value[part], name) for part in ("real", "imag")}


def _interval(lower: float, upper: float) -> dict[str, float]:
    return {"lower": lower, "upper": upper, "width": upper - lower}


def _bounded_interval(center: float, radius: float, bound: float) -> dict[str, float]:
    return _interval(max(-bound, center - radius), min(bound, center + radius))


def _ratio_interval(numerator: Mapping[str, float], denominator: Mapping[str, float],
                    bound: float) -> dict[str, float]:
    if denominator["lower"] == 0:
        return _interval(-bound, bound)
    endpoints = [numerator[n] / denominator[d]
                 for n in ("lower", "upper") for d in ("lower", "upper")]
    return _interval(max(-bound, min(endpoints)), min(bound, max(endpoints)))


def _sum_intervals(terms: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    if any(term[key] is None for term in terms):
        return None
    return {part: _interval(
        math.fsum(term[key][part]["lower"] for term in terms),
        math.fsum(term[key][part]["upper"] for term in terms),
    ) for part in ("real", "imag")}


def block_hoeffding_intervals(
    term_statistics: Sequence[Mapping[str, Any]],
    confidence_level: float,
    *,
    raw_available: bool = True,
    conditional_available: bool = True,
) -> dict[str, Any]:
    """Return JSON-safe simultaneous real/imaginary Bell intervals.

    Each statistic supplies ``coefficient`` and ``raw`` as complex component
    mappings, nullable ``conditional``, ``matched_blocks`` and the matched
    blocks' mean ``acceptance_fraction``. ``raw`` is the mean block numerator,
    which already includes the sampling coefficient. Missing full-run evidence
    can suppress aggregate intervals while retaining observed-term diagnostics.
    """
    confidence = _finite_real(confidence_level, "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if type(raw_available) is not bool or type(conditional_available) is not bool:
        raise ValueError("availability indicators must be bool values")
    term_count = len(term_statistics)
    if term_count == 0:
        raise ValueError("term_statistics must contain at least one term")
    alpha = 1 - confidence
    log_factor = math.log(6 * term_count / alpha)
    intervals: list[dict[str, Any]] = []
    for index, statistic in enumerate(term_statistics):
        coefficient = _components(statistic["coefficient"], "coefficient")
        bound = math.hypot(coefficient["real"], coefficient["imag"])
        if not math.isfinite(bound):
            raise ValueError("coefficient magnitude must be finite")
        matched = statistic["matched_blocks"]
        if type(matched) is not int or matched < 0:
            raise ValueError("matched_blocks must be a non-negative integer")
        raw = statistic["raw"]
        conditional = statistic["conditional"]
        acceptance = statistic["acceptance_fraction"]
        entry: dict[str, Any] = {
            "term_index": index, "matched_blocks": matched,
            "coefficient_magnitude": bound, "numerator": None,
            "acceptance": None, "raw": None, "conditional": None,
        }
        if matched == 0:
            if raw is not None or conditional is not None or acceptance is not None:
                raise ValueError("zero matched blocks require unavailable term statistics")
            intervals.append(entry)
            continue
        raw_parts = _components(raw, "raw")
        mean_v = _finite_real(acceptance, "acceptance_fraction")
        if not 0 <= mean_v <= 1:
            raise ValueError("acceptance_fraction must be in [0, 1]")
        if any(abs(part) > bound + 1e-12 for part in raw_parts.values()):
            raise ValueError("raw numerator exceeds the coefficient support")
        if conditional is not None:
            conditional_parts = _components(conditional, "conditional")
            if mean_v == 0:
                raise ValueError("zero acceptance requires an unavailable conditional estimate")
            if any(abs(part) > bound + 1e-12 for part in conditional_parts.values()):
                raise ValueError("conditional estimate exceeds the coefficient support")
        radius_u = bound * math.sqrt(2 * log_factor / matched)
        radius_v = math.sqrt(log_factor / (2 * matched))
        numerator = {part: _bounded_interval(raw_parts[part], radius_u, bound)
                     for part in ("real", "imag")}
        denominator = _interval(max(0., mean_v - radius_v), min(1., mean_v + radius_v))
        entry.update(numerator=numerator, acceptance=denominator, raw=numerator)
        if conditional is not None:
            entry["conditional"] = {
                part: _ratio_interval(numerator[part], denominator, bound)
                for part in ("real", "imag")
            }
        intervals.append(entry)
    result = {
        "method": METHOD,
        "confidence_level": confidence,
        "alpha": alpha,
        "term_count": term_count,
        "log_factor": log_factor,
        "sampling_unit": "block",
        "scope": "expected means over the saved schedule and measured contexts",
        "assumptions": [
            "outcomes of different blocks are independent conditional on the saved schedule",
            "arbitrary dependence between shots within each block is allowed",
            "no independence between algebraic terms is required by the union bound",
        ],
        "interpretation": (
            "A stationary Bell-functional interpretation additionally requires stable "
            "measurements and faithful local observables. These intervals do not cover "
            "arbitrary memory between blocks or device communication and are not a "
            "loophole-closing test. The conditional point estimate is a ratio estimate; "
            "no unbiasedness claim is made."
        ),
        "raw": _sum_intervals(intervals, "raw") if raw_available else None,
        "conditional": _sum_intervals(intervals, "conditional") if conditional_available else None,
        "terms": intervals,
    }
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as error:
        raise ValueError("uncertainty exceeds finite numeric reporting limits") from error
    return result
