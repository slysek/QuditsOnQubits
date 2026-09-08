"""Block estimators use matched shot denominators and keep physical blocks."""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from qudits_on_qubits.bell_measurements.postprocessing import (
    evaluate_reference_bell_values_from_counts,
)
from qudits_on_qubits.experiments.block_estimation import evaluate_blocks
from qudits_on_qubits.reference_experiments import get_reference_experiment


def example(state="two_qutrit", shots=10, repeats=1):
    reference = get_reference_experiment(state)
    patterns = reference.measurement_settings()
    local = [sorted({p[i] for p in patterns if p[i] is not None})
             for i in range(reference.state.num_parties)]
    settings = list(itertools.product(*local)) * repeats
    blocks = tuple(SimpleNamespace(
        block_id=i, settings=s,
        matched_pattern_indices=tuple(j for j, p in enumerate(patterns)
                                      if matches(p, s)),
    ) for i, s in enumerate(settings))
    schedule = SimpleNamespace(
        state=state, blocks=blocks, required_patterns=patterns, complete=True,
        missing_pattern_indices=(), config=SimpleNamespace(
            shots_per_draw=shots, setting_draws=len(blocks),
            max_setting_draws=len(blocks), confidence_level=.95,
        ),
    )
    maps = {s: tuple((2*i, 2*i+1) for i in range(len(s))) for s in settings}
    counts = {b.block_id: {"0" * (2*len(b.settings)): shots} for b in blocks}
    outcome_map = dict(reference.outcome_convention.measurement_basis_index_map)
    return reference, schedule, counts, maps, outcome_map


def matches(pattern, setting):
    return all(label is None or label == choice for label, choice in zip(pattern, setting))


def as_complex(value):
    return complex(value["real"], value["imag"])


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_exact_reference_agreement_and_json_safety(state):
    reference, schedule, counts, maps, outcome_map = example(state, repeats=2)
    result = evaluate_blocks(reference, schedule, counts, maps, outcome_map)
    legacy_counts = {s: {"0"*(2*len(s)): 10} for s in reference.measurement_settings()}
    legacy_maps = {s: tuple((2*i, 2*i+1) for i in range(len(s))) for s in legacy_counts}
    legacy = evaluate_reference_bell_values_from_counts(state, legacy_counts, legacy_maps)
    assert as_complex(result["raw"]) == pytest.approx(legacy.unconditional)
    assert as_complex(result["conditional"]) == pytest.approx(legacy.conditional)
    assert result["budget"]["completed_shots"] == len(schedule.blocks)*10
    assert result["leakage"]["global"]["accepted_shots"] == len(schedule.blocks)*10
    assert len(result["leakage"]["blocks"]) == len(schedule.blocks)
    json.dumps(result, allow_nan=False)


def test_conditional_uses_ratio_of_sums_with_unequal_acceptance():
    reference, schedule, counts, maps, outcome_map = example(repeats=2)
    first, second = schedule.blocks[0], schedule.blocks[9]
    counts[first.block_id] = {"0000": 1, "1111": 9}
    counts[second.block_id] = {"0001": 9, "1111": 1}
    result = evaluate_blocks(reference, schedule, counts, maps, outcome_map)
    term_index = next(i for i, t in enumerate(reference.bell_functional.terms)
                      if reference.setting_for_term(t) == first.settings)
    term = reference.bell_functional.terms[term_index]
    powers = reference.powers_for_term(term)
    # Measurement index 1 corresponds to bit pair (1,0) and outcome 1.
    phase = np.exp(2j*np.pi/3) ** (powers[0] % 3)
    coefficient = term.sampling_coefficient()
    diagnostic = result["terms"][term_index]
    assert as_complex(diagnostic["raw"]) == pytest.approx(coefficient*(1+9*phase)/20)
    assert as_complex(diagnostic["conditional"]) == pytest.approx(coefficient*(1+9*phase)/10)
    assert as_complex(diagnostic["conditional"]) != pytest.approx(coefficient*(1+phase)/2)
    assert diagnostic["matched_blocks"] == 2


def test_ame_overlapping_marginals_and_spectator_leakage():
    reference, schedule, counts, maps, outcome_map = example("ame43")
    selected = next(b for b in schedule.blocks if len(b.matched_pattern_indices) > 1)
    pattern_index = next(i for i in selected.matched_pattern_indices
                         if None in schedule.required_patterns[i])
    spectator = schedule.required_patterns[pattern_index].index(None)
    bits = ["0"]*8
    bits[7-2*spectator] = bits[6-2*spectator] = "1"
    counts[selected.block_id] = {"".join(bits): 10}
    result = evaluate_blocks(reference, schedule, counts, maps, outcome_map)
    assert result["leakage"]["global"]["invalid_shots"] == 10
    assert result["leakage"]["patterns"][pattern_index]["invalid_shots"] == 10
    for i in selected.matched_pattern_indices:
        assert result["leakage"]["patterns"][i]["invalid_shots"] == 10
    diagnostic = next(t for t in result["terms"] if t["pattern_index"] == pattern_index)
    context = next(c for c in diagnostic["contexts"] if c["settings"] == list(selected.settings))
    assert as_complex(context["raw"]) == 0
    assert context["conditional"] is None
    assert context["conditional_reason"] == "no_accepted_shots"
    assert len(diagnostic["contexts"]) == len({b.settings for b in schedule.blocks
                                              if pattern_index in b.matched_pattern_indices})


def test_all_leakage_is_zero_raw_but_null_conditional():
    args = list(example())
    args[2] = {b.block_id: {"1111": 10} for b in args[1].blocks}
    result = evaluate_blocks(*args)
    assert as_complex(result["raw"]) == 0
    assert result["conditional"] is None
    assert result["conditional_reason"] == "no_accepted_shots"
    assert result["uncertainty"]["conditional"] is None
    assert result["leakage"]["global"]["leakage_rate"] == 1
    json.dumps(result, allow_nan=False)


def test_missing_blocks_are_null_without_losing_partial_diagnostics():
    args = list(example())
    del args[2][0]
    result = evaluate_blocks(*args)
    assert result["raw"] is None
    assert result["conditional"] is None
    assert result["raw_reason"] == result["conditional_reason"] == "missing_blocks"
    assert result["coverage"]["missing_block_ids"] == [0]
    assert any(t["raw"] is not None for t in result["terms"])
    assert result["uncertainty"]["raw"] is None


def test_missing_unmatched_block_also_prevents_complete_result():
    args = list(example("ame43"))
    missing = next(b for b in args[1].blocks if not b.matched_pattern_indices)
    del args[2][missing.block_id]
    result = evaluate_blocks(*args)
    assert result["raw"] is None
    assert result["raw_reason"] == "missing_blocks"
    assert all(t["raw"] is not None for t in result["terms"])


def test_no_completed_evidence_has_no_fabricated_leakage_rate():
    args = list(example())
    args[2] = {}
    result = evaluate_blocks(*args)
    assert result["raw"] is result["conditional"] is None
    assert result["leakage"]["global"]["leakage_rate"] is None
    assert result["budget"]["completed_shots"] == 0
    assert all(t["raw"] is None for t in result["terms"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("counts", [
    {"0000": -1, "0001": 11}, {"0000": True, "0001": 9},
    {"0000": 10.0}, {"0000": float("nan")}, {"0000": float("inf")},
    {"0000": 9}, {"00000": 10}, {"000": 10}, {"000x": 10},
])
def test_rejects_malformed_raw_block(counts):
    args = list(example())
    args[2][0] = counts
    with pytest.raises(ValueError):
        evaluate_blocks(*args)


@pytest.mark.parametrize("mapping", [((0, 1), (1, 2)), ((0, 1), (2, True)),
                                         ((0, 1),), ((0, 1), (2, -1))])
def test_rejects_malformed_classical_bit_mapping(mapping):
    args = list(example())
    args[3][args[1].blocks[0].settings] = mapping
    with pytest.raises(ValueError):
        evaluate_blocks(*args)


@pytest.mark.parametrize("mapping", [{0: 0}, {0: 0, 1: 1, 2: 2, 3: False},
                                       {0: 0, 1: 1, 2: 3, 3: None}])
def test_rejects_malformed_outcome_map(mapping):
    args = list(example())
    args[4] = mapping
    with pytest.raises(ValueError):
        evaluate_blocks(*args)


def test_duplicate_block_ids_are_rejected_instead_of_collapsed():
    args = list(example())
    args[1].blocks[1].block_id = 0
    with pytest.raises(ValueError, match="block.*[Ii][Dd]"):
        evaluate_blocks(*args)


def test_decoder_called_once_per_raw_key(monkeypatch):
    import qudits_on_qubits.experiments.block_estimation as module
    original = module.bitstring_to_qutrit_outcomes
    seen = []
    def spy(*args, **kwargs):
        seen.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "bitstring_to_qutrit_outcomes", spy)
    args = example("ame43")
    evaluate_blocks(*args)
    assert len(seen) == len(args[1].blocks)


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_varying_joint_outcomes_match_shot_table_oracle_with_permuted_bits(state):
    reference, schedule, counts, maps, outcome_map = example(state, shots=37, repeats=3)
    rng = np.random.default_rng(602)
    tables = {}
    parties = reference.state.num_parties
    permutation = rng.permutation(2*parties).tolist()
    indices = tuple(tuple(permutation[2*i:2*i+2]) for i in range(parties))
    for setting in maps:
        maps[setting] = indices
    for block in schedule.blocks:
        outcomes = rng.integers(0, 4, size=(37, parties))
        tables[block.block_id] = outcomes
        histogram = {}
        for row in outcomes:
            word = ["0"]*(2*parties)
            for party, outcome in enumerate(row):
                low, high = indices[party]
                word[-1-low] = str(outcome % 2)
                word[-1-high] = str(outcome // 2)
            # Include Qiskit multi-register spaces without changing chronology.
            bitstring = "".join(word[:2]) + " " + "".join(word[2:])
            histogram[bitstring] = histogram.get(bitstring, 0) + 1
        counts[block.block_id] = histogram
    result = evaluate_blocks(reference, schedule, counts, maps, outcome_map)
    expected_raw = 0j
    expected_conditional = 0j
    for index, term in enumerate(reference.bell_functional.terms):
        pattern = reference.setting_for_term(term)
        rows = np.concatenate([tables[b.block_id] for b in schedule.blocks
                               if matches(pattern, b.settings)])
        accepted_rows = rows[np.all(rows < 3, axis=1)]
        phases = np.exp(2j*np.pi/3 * ((accepted_rows @ reference.powers_for_term(term)) % 3))
        numerator = term.sampling_coefficient() * phases.sum()
        expected_raw += numerator/len(rows)
        expected_conditional += numerator/len(accepted_rows)
        assert as_complex(result["terms"][index]["raw"]) == pytest.approx(numerator/len(rows))
        assert as_complex(result["terms"][index]["conditional"]) == pytest.approx(numerator/len(accepted_rows))
    assert as_complex(result["raw"]) == pytest.approx(expected_raw)
    assert as_complex(result["conditional"]) == pytest.approx(expected_conditional)
    assert result["leakage"]["global"]["invalid_shots"] == sum(
        int(np.any(table == 3, axis=1).sum()) for table in tables.values())


def test_incomplete_setting_coverage_is_unavailable_with_preserved_zero_blocks():
    args = list(example("ame43"))
    schedule = args[1]
    schedule.blocks = schedule.blocks[:1]
    schedule.config.setting_draws = schedule.config.max_setting_draws = 1
    covered = set(schedule.blocks[0].matched_pattern_indices)
    schedule.missing_pattern_indices = tuple(i for i in range(len(schedule.required_patterns)) if i not in covered)
    schedule.complete = False
    args[2] = {0: args[2][0]}
    result = evaluate_blocks(*args)
    assert result["raw"] is None
    assert result["raw_reason"] == "missing_patterns"
    assert result["budget"]["completed_shots"] == 10
    assert result["coverage"]["schedule_complete"] is False
