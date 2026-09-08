"""Uncertainty conditions on the schedule and counts independent blocks."""

import math
import random

import pytest

from qudits_on_qubits.experiments.block_estimation import evaluate_blocks
from qudits_on_qubits.experiments.block_uncertainty import block_hoeffding_intervals
from test_randomized_bell_estimation import example


def test_more_correlated_shots_per_block_do_not_narrow_intervals():
    small = evaluate_blocks(*example(shots=1, repeats=2))
    large = evaluate_blocks(*example(shots=1000000, repeats=2))
    assert small["uncertainty"] == large["uncertainty"]
    assert small["uncertainty"]["method"] == "conditional_schedule_block_hoeffding_v1"
    assert small["uncertainty"]["raw"]["real"]["width"] > 0
    assert "standard_error" not in str(small["uncertainty"])


def test_exact_union_bound_radii_and_positive_denominator_ratio():
    stats = [{"coefficient": {"real": 2., "imag": 0.}, "matched_blocks": 1000,
              "raw": {"real": .4, "imag": -.3}, "conditional": {"real": .5, "imag": -.375},
              "acceptance_fraction": .8}]
    result = block_hoeffding_intervals(stats, .95)
    log_factor = math.log(6/.05)
    radius_u = 2*math.sqrt(2*log_factor/1000)
    radius_v = math.sqrt(log_factor/2000)
    term = result["terms"][0]
    assert term["numerator"]["real"]["lower"] == pytest.approx(.4-radius_u)
    assert term["numerator"]["imag"]["upper"] == pytest.approx(-.3+radius_u)
    assert term["acceptance"]["lower"] == pytest.approx(.8-radius_v)
    assert term["conditional"]["real"]["lower"] == pytest.approx((.4-radius_u)/(.8+radius_v))
    assert term["conditional"]["real"]["upper"] == pytest.approx((.4+radius_u)/(.8-radius_v))


def test_one_block_full_conditional_range_and_nonzero_raw_width():
    result = evaluate_blocks(*example(shots=1))
    for term, interval in zip(result["terms"], result["uncertainty"]["terms"]):
        c = abs(complex(**term["coefficient"]))
        assert interval["acceptance"]["lower"] == 0
        assert interval["conditional"]["real"] == pytest.approx({"lower": -c, "upper": c, "width": 2*c})
        assert interval["numerator"]["real"]["width"] > 0


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 0, 1, True])
def test_nonfinite_or_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        block_hoeffding_intervals([], confidence)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_term_values_rejected(bad):
    stats = [{"coefficient": {"real": bad, "imag": 0}, "matched_blocks": 2,
              "raw": {"real": 0, "imag": 0}, "conditional": {"real": 0, "imag": 0},
              "acceptance_fraction": 1.}]
    with pytest.raises(ValueError):
        block_hoeffding_intervals(stats, .95)


def test_overlap_terms_use_total_algebraic_count_for_union_bound():
    result = evaluate_blocks(*example("ame43"))
    assert result["uncertainty"]["term_count"] == 26
    assert result["uncertainty"]["log_factor"] == pytest.approx(math.log(6*26/.05))
    assert len(result["uncertainty"]["terms"]) == 26


@pytest.mark.parametrize("field,value", [
    ("raw", {"real": float("nan"), "imag": 0}),
    ("conditional", {"real": 0, "imag": float("inf")}),
    ("acceptance_fraction", float("nan")),
    ("acceptance_fraction", 1.1), ("matched_blocks", True),
])
def test_invalid_observed_statistics_rejected(field, value):
    statistic = {"coefficient": {"real": 1., "imag": 0.}, "matched_blocks": 2,
                 "raw": {"real": .4, "imag": -.3}, "conditional": {"real": .5, "imag": -.375},
                 "acceptance_fraction": .8}
    statistic[field] = value
    with pytest.raises(ValueError):
        block_hoeffding_intervals([statistic], .95)


def test_stopped_schedule_simulation_with_perfectly_correlated_block_shots():
    from qudits_on_qubits.experiments.measurement import RandomizedBlocks
    from qudits_on_qubits.experiments.setting_schedule import generate_schedule
    from qudits_on_qubits.reference_experiments import get_reference_experiment

    reference = get_reference_experiment("two_qutrit")
    coefficient_sum = sum(term.sampling_coefficient() for term in reference.bell_functional.terms)
    config = RandomizedBlocks(setting_draws=10, shots_per_draw=1000, max_setting_draws=300)
    lengths = set()
    covered_runs = 0
    for seed in range(40):
        settings_rng, outcomes_rng = random.Random(seed), random.Random(seed+10000)
        schedule = generate_schedule(reference, config, _randbelow=settings_rng.randrange, _seed=seed)
        assert schedule.complete
        lengths.add(len(schedule.blocks))
        # Outcomes are independent between blocks, with arbitrary (here perfect)
        # dependence between all 1000 shots within a block.
        counts = {b.block_id: {("0000" if outcomes_rng.random() < .8 else "1111"): 1000}
                  for b in schedule.blocks}
        maps = {b.settings: ((0, 1), (2, 3)) for b in schedule.blocks}
        result = evaluate_blocks(reference, schedule, counts, maps,
                                 dict(reference.outcome_convention.measurement_basis_index_map))
        raw_interval = result["uncertainty"]["raw"]
        expected = .8*coefficient_sum
        contained = (raw_interval["real"]["lower"] <= expected.real <= raw_interval["real"]["upper"]
                     and raw_interval["imag"]["lower"] <= expected.imag <= raw_interval["imag"]["upper"])
        if result["conditional"] is not None:
            interval = result["uncertainty"]["conditional"]
            contained &= (interval["real"]["lower"] <= coefficient_sum.real <= interval["real"]["upper"]
                          and interval["imag"]["lower"] <= coefficient_sum.imag <= interval["imag"]["upper"])
        covered_runs += contained
    assert len(lengths) > 1
    assert covered_runs >= 38
