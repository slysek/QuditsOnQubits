from __future__ import annotations

import math

import pytest
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)


from qudits_on_qubits.bell_measurements.postprocessing import (
    compute_bell_value_from_counts,
    compute_complex_expectation,
    leakage_rate,
)


def test_complex_expectation_accepts_signed_float_weights():
    value = compute_complex_expectation(
        {"00": 2.5, "01": -0.5},
        powers=(1,),
        qutrit_bit_indices=((0, 1),),
        bit_order="left-to-right",
    )

    expected = (2.5 + (-0.5) * complex(-0.5, math.sqrt(3) / 2)) / 2.0
    assert value == pytest.approx(expected)


def test_complex_expectation_returns_zero_for_zero_signed_denominator():
    assert compute_complex_expectation(
        {"00": 1.0, "01": -1.0},
        powers=(1,),
        qutrit_bit_indices=((0, 1),),
        bit_order="left-to-right",
    ) == 0j


def test_bell_value_accepts_weighted_counts():
    value = compute_bell_value_from_counts(
        {("A",): {"00": 3.0, "01": -1.0}},
        terms=({"settings": ("A",), "powers": (1,), "coeff": 2},),
        qutrit_bit_indices_by_setting={("A",): ((0, 1),)},
        bit_order="left-to-right",
    )

    expected = 2 * (3.0 - complex(-0.5, math.sqrt(3) / 2)) / 2.0
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_complex_expectation_rejects_non_finite_weights(weight):
    with pytest.raises(ValueError, match="finite"):
        compute_complex_expectation(
            {"00": weight},
            powers=(1,),
            qutrit_bit_indices=((0, 1),),
            bit_order="left-to-right",
        )


def test_leakage_rate_rejects_negative_weighted_counts():
    with pytest.raises(ValueError, match="non-negative"):
        leakage_rate({"11": -0.5}, ((0, 1),), bit_order="left-to-right")
