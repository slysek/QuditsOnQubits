"""Raw Bell analysis for independent local settings in separately saved blocks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from qudits_on_qubits.bell_measurements.basis import omega
from qudits_on_qubits.bell_measurements.postprocessing import bitstring_to_qutrit_outcomes
from qudits_on_qubits.reference_experiments import ReferenceExperimentSpec

from .block_uncertainty import block_hoeffding_intervals


def _complex_json(value: complex | None) -> dict[str, float] | None:
    if value is None:
        return None
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("Bell values and coefficients must be finite")
    return {"real": float(value.real), "imag": float(value.imag)}


def _sum_complex(values: Sequence[complex]) -> complex:
    return complex(math.fsum(value.real for value in values),
                   math.fsum(value.imag for value in values))


def _matches(pattern: tuple, setting: tuple) -> bool:
    return all(label is None or label == choice for label, choice in zip(pattern, setting))


def _leakage(total: int, accepted: int) -> dict[str, Any]:
    return {"total_shots": total, "accepted_shots": accepted,
            "invalid_shots": total - accepted,
            "leakage_rate": (total - accepted) / total if total else None}


def _validate_outcome_map(mapping: Mapping[int, int | None]) -> dict[int, int | None]:
    if not isinstance(mapping, Mapping) or set(mapping) != {0, 1, 2, 3}:
        raise ValueError("outcome_map must cover all four physical indices")
    if any(type(key) is not int for key in mapping):
        raise ValueError("outcome_map indices must be integers")
    outcomes = list(mapping.values())
    if any(value is not None and (type(value) is not int or value not in (0, 1, 2))
           for value in outcomes):
        raise ValueError("outcome_map values must be 0, 1, 2 or None")
    if sorted(value for value in outcomes if value is not None) != [0, 1, 2]:
        raise ValueError("outcome_map must assign every qutrit outcome exactly once")
    return dict(mapping)


def _validate_bit_indices(indices: Any, parties: int) -> tuple[tuple[int, int], ...]:
    try:
        pairs = tuple(tuple(pair) for pair in indices)
    except TypeError as error:
        raise ValueError("bit mapping must contain one bit pair per party") from error
    if len(pairs) != parties or any(len(pair) != 2 for pair in pairs):
        raise ValueError("bit mapping must contain one bit pair per party")
    flattened = [index for pair in pairs for index in pair]
    if any(type(index) is not int or index < 0 for index in flattened):
        raise ValueError("bit mapping indices must be non-negative integers")
    if len(set(flattened)) != len(flattened):
        raise ValueError("bit mapping indices must be distinct across all parties")
    return pairs


def _summary(records: Sequence[dict[str, Any]], term_index: int, shots: int) -> dict[str, Any]:
    total = len(records) * shots
    accepted = sum(record["accepted_shots"] for record in records)
    values = [record["numerators"][term_index] for record in records]
    numerator_sum = _sum_complex(values)
    raw = numerator_sum / len(records) if records else None
    conditional = numerator_sum / (accepted / shots) if accepted else None
    return {
        "matched_blocks": len(records),
        "block_ids": [record["block_id"] for record in records],
        **_leakage(total, accepted),
        "acceptance_fraction": accepted / total if total else None,
        "raw": _complex_json(raw),
        "conditional": _complex_json(conditional),
        "raw_reason": None if records else "missing_blocks",
        "conditional_reason": (None if accepted else "no_accepted_shots" if records else "missing_blocks"),
    }


def evaluate_blocks(
    reference: ReferenceExperimentSpec,
    schedule: Any,
    counts_by_block: Mapping[int, Mapping[str, int]],
    bit_indices_by_setting: Mapping[tuple[str, ...], Sequence[tuple[int, int]]],
    outcome_map: Mapping[int, int | None],
) -> dict[str, Any]:
    """Evaluate supplied complete blocks, preserving absent evidence as missing.

    Counts must contain exactly ``schedule.config.shots_per_draw`` shots for
    every supplied block ID. Repeated full settings remain separate blocks.
    Every party's leakage, including a term's spectators, zeros the complete
    joint outcome. Full values are unavailable until every scheduled block and
    every required pattern have evidence; observed-term/context diagnostics
    remain available on partial execution. No hardware or filesystem is used.
    """
    shots = schedule.config.shots_per_draw
    if type(shots) is not int or shots < 1:
        raise ValueError("shots_per_draw must be a positive integer")
    if not isinstance(counts_by_block, Mapping):
        raise ValueError("counts_by_block must be a mapping keyed by block ID")
    if schedule.state != reference.experiment_id:
        raise ValueError("schedule state does not match the Bell reference")
    outcome_mapping = _validate_outcome_map(outcome_map)
    parties = reference.state.num_parties
    patterns = tuple(tuple(pattern) for pattern in schedule.required_patterns)
    if len(set(patterns)) != len(patterns) or set(patterns) != set(reference.measurement_settings()):
        raise ValueError("schedule required patterns do not match the Bell reference")
    terms = tuple(reference.bell_functional.terms)
    term_patterns = tuple(reference.setting_for_term(term) for term in terms)
    powers = tuple(reference.powers_for_term(term) for term in terms)
    coefficients = tuple(complex(term.sampling_coefficient()) for term in terms)
    for coefficient in coefficients:
        _complex_json(coefficient)
    local_labels = [{pattern[i] for pattern in patterns if pattern[i] is not None}
                    for i in range(parties)]
    blocks = tuple(schedule.blocks)
    block_ids: set[int] = set()
    normalized: list[tuple[Any, tuple[str, ...], tuple[tuple[int, int], ...]]] = []
    planned_pattern_counts = [0] * len(patterns)
    for block in blocks:
        if type(block.block_id) is not int or block.block_id < 0 or block.block_id in block_ids:
            raise ValueError("block IDs must be unique non-negative integers")
        block_ids.add(block.block_id)
        setting = tuple(block.settings)
        if len(setting) != parties or any(label not in local_labels[i] for i, label in enumerate(setting)):
            raise ValueError("block settings must specify valid local labels for every party")
        matched = tuple(i for i, pattern in enumerate(patterns) if _matches(pattern, setting))
        if tuple(block.matched_pattern_indices) != matched:
            raise ValueError("block matched patterns disagree with its full settings")
        for i in matched:
            planned_pattern_counts[i] += 1
        if setting not in bit_indices_by_setting:
            if block.block_id in counts_by_block:
                raise ValueError("missing bit mapping for a completed block setting")
            indices = ()
        else:
            indices = _validate_bit_indices(bit_indices_by_setting[setting], parties)
        normalized.append((block, setting, indices))
    if any(type(key) is not int or key not in block_ids for key in counts_by_block):
        raise ValueError("counts contain an unknown block ID")
    planned_missing = [i for i, count in enumerate(planned_pattern_counts) if count == 0]
    if list(schedule.missing_pattern_indices) != planned_missing:
        raise ValueError("schedule coverage metadata does not match its blocks")
    records: list[dict[str, Any]] = []
    leakage_blocks: list[dict[str, Any]] = []
    missing_blocks: list[int] = []
    root = omega(3)
    for block, setting, indices in normalized:
        if block.block_id not in counts_by_block:
            missing_blocks.append(block.block_id)
            leakage_blocks.append({"block_id": block.block_id, "settings": list(setting),
                                   "status": "missing", "total_shots": None, "accepted_shots": None,
                                   "invalid_shots": None, "leakage_rate": None})
            continue
        counts = counts_by_block[block.block_id]
        if not isinstance(counts, Mapping):
            raise ValueError("raw counts for each block must be a mapping")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError("raw counts must be non-negative integers, excluding bool")
        if sum(counts.values()) != shots:
            raise ValueError(f"block {block.block_id} raw counts do not total shots_per_draw")
        matched_terms = [i for i, pattern in enumerate(term_patterns) if _matches(pattern, setting)]
        contributions: dict[int, list[complex]] = {i: [] for i in matched_terms}
        accepted = 0
        width = max(index for pair in indices for index in pair) + 1
        for bitstring, count in counts.items():
            if not isinstance(bitstring, str):
                raise ValueError("count bitstrings must be strings")
            compact = bitstring.replace(" ", "")
            if len(compact) != width or set(compact) - {"0", "1"}:
                raise ValueError("count bitstring width or alphabet disagrees with bit mapping")
            outcomes = bitstring_to_qutrit_outcomes(compact, indices, outcome_map=outcome_mapping)
            if any(outcome is None for outcome in outcomes):
                continue
            accepted += count
            for i in matched_terms:
                exponent = sum(power * outcome for power, outcome in zip(powers[i], outcomes)) % 3
                contributions[i].append((count / shots) * root**exponent)
        record = {"block_id": block.block_id, "settings": setting,
                  "accepted_shots": accepted,
                  "matched_pattern_indices": tuple(block.matched_pattern_indices),
                  "numerators": {i: coefficients[i] * _sum_complex(values)
                                 for i, values in contributions.items()}}
        records.append(record)
        leakage_blocks.append({"block_id": block.block_id, "settings": list(setting),
                               "status": "completed", **_leakage(shots, accepted)})
    term_diagnostics: list[dict[str, Any]] = []
    for i, term in enumerate(terms):
        relevant = [record for record in records if i in record["numerators"]]
        planned = [block for block, setting, _ in normalized if _matches(term_patterns[i], setting)]
        contexts = []
        for setting in dict.fromkeys(tuple(block.settings) for block in planned):
            context_records = [record for record in relevant if record["settings"] == setting]
            context_blocks = [block for block in planned if tuple(block.settings) == setting]
            contexts.append({"settings": list(setting), "scheduled_blocks": len(context_blocks),
                             "missing_block_ids": [b.block_id for b in context_blocks
                                                   if b.block_id not in counts_by_block],
                             **_summary(context_records, i, shots)})
        term_diagnostics.append({
            "term_index": i, "pattern_index": patterns.index(term_patterns[i]),
            "settings": list(term_patterns[i]), "powers": list(powers[i]),
            "coefficient": _complex_json(coefficients[i]),
            "scheduled_blocks": len(planned),
            "missing_block_ids": [b.block_id for b in planned if b.block_id not in counts_by_block],
            **_summary(relevant, i, shots), "contexts": contexts,
        })
    pattern_leakage = []
    for i, pattern in enumerate(patterns):
        relevant = [record for record in records if i in record["matched_pattern_indices"]]
        pattern_leakage.append({
            "pattern_index": i, "settings": list(pattern),
            "scheduled_blocks": planned_pattern_counts[i], "matched_blocks": len(relevant),
            **_leakage(len(relevant)*shots, sum(record["accepted_shots"] for record in relevant)),
        })
    missing_patterns = [entry["pattern_index"] for entry in pattern_leakage if entry["matched_blocks"] == 0]
    raw_reason = "missing_blocks" if missing_blocks else "missing_patterns" if missing_patterns else None
    conditional_reason = raw_reason or ("no_accepted_shots" if any(
        term["conditional"] is None for term in term_diagnostics) else None)
    raw = None if raw_reason else _complex_json(_sum_complex([
        complex(**term["raw"]) for term in term_diagnostics]))
    conditional = None if conditional_reason else _complex_json(_sum_complex([
        complex(**term["conditional"]) for term in term_diagnostics]))
    total_shots = len(records) * shots
    result = {
        "schema_version": "independent-local-bell-analysis-v1",
        "state": reference.experiment_id,
        "estimator": "sum_of_matched_block_correlators_v1",
        "raw": raw, "conditional": conditional,
        "raw_reason": raw_reason, "conditional_reason": conditional_reason,
        "quality_flags": list(dict.fromkeys(reason for reason in (raw_reason, conditional_reason) if reason)),
        "terms": term_diagnostics,
        "leakage": {
            "global": _leakage(total_shots, sum(record["accepted_shots"] for record in records)),
            "blocks": leakage_blocks, "patterns": pattern_leakage,
            "counting_note": "pattern and term shot totals overlap and are not hardware cost",
        },
        "budget": {
            "setting_draws": schedule.config.setting_draws,
            "max_setting_draws": schedule.config.max_setting_draws,
            "shots_per_draw": shots, "scheduled_blocks": len(blocks), "completed_blocks": len(records),
            "minimum_shots": schedule.config.setting_draws * shots,
            "maximum_shots": schedule.config.max_setting_draws * shots,
            "scheduled_shots": len(blocks) * shots, "completed_shots": total_shots,
        },
        "coverage": {
            "schedule_complete": schedule.complete,
            "required_pattern_count": len(patterns),
            "scheduled_covered_patterns": len(patterns) - len(planned_missing),
            "completed_covered_patterns": len(patterns) - len(missing_patterns),
            "missing_scheduled_pattern_indices": planned_missing,
            "missing_pattern_indices": missing_patterns,
            "missing_block_ids": missing_blocks,
        },
        "reference_comparison": {
            "status": "diagnostic",
            "functional_id": reference.bell_functional.functional_id,
            "normalization": reference.bell_functional.normalization,
            "classical_bound": reference.bell_functional.classical_bound,
            "classical_bound_source": reference.bell_functional.classical_bound_source,
            "scope": "saved schedule and observed full-setting contexts",
            "note": "Interpretation requires faithful local measurements; postselection adds measurement assumptions.",
        },
        "uncertainty": block_hoeffding_intervals(
            term_diagnostics, schedule.config.confidence_level,
            raw_available=raw is not None, conditional_available=conditional is not None,
        ),
    }
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as error:
        raise ValueError("analysis exceeds finite numeric reporting limits") from error
    return result
