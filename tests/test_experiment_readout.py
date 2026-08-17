from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import math
from types import MappingProxyType

import numpy as np
import pytest

from qudits_on_qubits.experiments.errors import (
    ExperimentValidationError,
    JobResultError,
    OptionalDependencyError,
)
from qudits_on_qubits.experiments.mitigation import (
    ReadoutCalibration,
    apply_readout_mitigation,
    assignment_matrices_from_counts,
    build_m3_mitigation,
    build_readout_calibration_circuits,
    calibration_cache_is_valid,
)


def _record(**overrides: object) -> ReadoutCalibration:
    values: dict[str, object] = {
        "backend_identity": "backend-a",
        "calibration_id": "cal-17",
        "qubit_mapping": (2, 0),
        "timestamp": datetime(2026, 8, 17, 10, tzinfo=timezone.utc),
        "shots": 100,
        "raw_counts": (
            {"0": 90, "1": 10},
            {"0": 20, "1": 80},
            {"0": 95, "1": 5},
            {"0": 15, "1": 85},
        ),
        "assignment_matrices": (
            ((0.9, 0.2), (0.1, 0.8)),
            ((0.95, 0.15), (0.05, 0.85)),
        ),
    }
    values.update(overrides)
    return ReadoutCalibration(**values)  # type: ignore[arg-type]


def test_build_readout_calibration_circuits_has_deterministic_mapping_order() -> None:
    circuits = build_readout_calibration_circuits([2, 0])

    assert tuple((c.metadata["physical_qubit"], c.metadata["prepared_state"]) for c in circuits) == (
        (2, 0),
        (2, 1),
        (0, 0),
        (0, 1),
    )
    assert [[item.operation.name for item in circuit.data] for circuit in circuits] == [
        ["measure"],
        ["x", "measure"],
        ["measure"],
        ["x", "measure"],
    ]
    assert all(circuit.num_qubits == 3 and circuit.num_clbits == 1 for circuit in circuits)


@pytest.mark.parametrize("qubits", [[], [0, 0], [-1], [True], [1.0]])
def test_build_readout_calibration_circuits_rejects_invalid_mapping(qubits: list[object]) -> None:
    with pytest.raises(ExperimentValidationError):
        build_readout_calibration_circuits(qubits)  # type: ignore[arg-type]


def test_assignment_matrices_use_rows_measured_columns_prepared_without_rounding() -> None:
    raw_counts = (
        {"0": 7, "1": 3},
        {"0": 2, "1": 8},
        {"0": 6, "1": 4},
        {"0": 1, "1": 9},
    )

    matrices = assignment_matrices_from_counts([2, 0], raw_counts, shots=10)

    assert matrices == (
        ((0.7, 0.2), (0.3, 0.8)),
        ((0.6, 0.1), (0.4, 0.9)),
    )


@pytest.mark.parametrize(
    ("counts", "shots"),
    [
        (({"0": 0, "1": 0}, {"0": 1, "1": 0}), 1),
        (({"0": 1, "1": -1}, {"0": 1, "1": 0}), 1),
        (({"0": True, "1": 0}, {"0": 1, "1": 0}), 1),
        (({"2": 1}, {"0": 1}), 1),
        (({"0": 1},), 1),
        (({"0": 1}, {"1": 1}), 0),
        (({"0": 1}, {"1": 1}), True),
        (({"0": 1}, {"1": 1}), 2),
    ],
)
def test_assignment_matrices_reject_malformed_counts_and_shots(
    counts: tuple[dict[str, object], ...], shots: object
) -> None:
    with pytest.raises(ExperimentValidationError):
        assignment_matrices_from_counts([0], counts, shots=shots)  # type: ignore[arg-type]


def test_readout_calibration_freezes_raw_evidence_and_validates_matrix_convention() -> None:
    calibration = _record()

    assert isinstance(calibration.raw_counts[0], MappingProxyType)
    with pytest.raises(TypeError):
        calibration.raw_counts[0]["0"] = 0  # type: ignore[index]
    assert calibration.assignment_matrices[0][0] == (0.9, 0.2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend_identity": "token=secret"},
        {"calibration_id": "password=bad"},
        {"qubit_mapping": (0, 0)},
        {"shots": 0},
        {"timestamp": datetime(2026, 8, 17, 10)},
        {"assignment_matrices": (((1.0, 0.0), (0.0, math.nan)), ((1.0, 0.0), (0.0, 1.0)))},
    ],
)
def test_readout_calibration_rejects_unsafe_or_invalid_metadata(overrides: dict[str, object]) -> None:
    with pytest.raises(ExperimentValidationError):
        _record(**overrides)


class _FakeMitigation:
    def __init__(self, outputs: list[object] | None = None) -> None:
        self.matrices: list[object] | None = None
        self.outputs = list(outputs or [])
        self.calls: list[tuple[dict[str, int], tuple[int, ...]]] = []

    def cals_from_matrices(self, matrices: list[object]) -> None:
        self.matrices = matrices

    def apply_correction(self, counts: dict[str, int], qubits: tuple[int, ...]) -> object:
        self.calls.append((counts, qubits))
        return self.outputs.pop(0)


def test_build_m3_mitigation_configures_sparse_physical_matrices_with_injected_object() -> None:
    fake = _FakeMitigation()

    returned = build_m3_mitigation(_record(), mitigation=fake)

    assert returned is fake
    assert fake.matrices is not None
    assert len(fake.matrices) == 3
    assert np.asarray(fake.matrices[2]) == pytest.approx(np.array([[0.9, 0.2], [0.1, 0.8]]))
    assert fake.matrices[1] is None
    assert np.asarray(fake.matrices[0]) == pytest.approx(np.array([[0.95, 0.15], [0.05, 0.85]]))


def test_build_m3_mitigation_loads_optional_dependency_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def missing_mthree(name: str, package: str | None = None) -> object:
        if name == "mthree":
            raise ModuleNotFoundError("No module named 'mthree'", name="mthree")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_mthree)

    with pytest.raises(OptionalDependencyError, match=r"pip install -e \.\[mitigation\]"):
        build_m3_mitigation(_record())


def test_apply_readout_mitigation_preserves_setting_order_mapping_and_signed_floats() -> None:
    fake = _FakeMitigation(
        outputs=[{"00": 1.1, "01": -0.1}, {"00": np.float64(0.4), "11": 0.6}]
    )
    counts = {"setting-b": {"00": 8, "01": 2}, "setting-a": {"00": 3, "11": 7}}

    corrected = apply_readout_mitigation(counts, mapping={0: 2, 1: 0}, mitigation=fake)

    assert list(corrected) == ["setting-b", "setting-a"]
    assert corrected == {
        "setting-b": {"00": 1.1, "01": -0.1},
        "setting-a": {"00": 0.4, "11": 0.6},
    }
    assert all(type(value) is float for setting in corrected.values() for value in setting.values())
    assert fake.calls == [(counts["setting-b"], (2, 0)), (counts["setting-a"], (2, 0))]


@pytest.mark.parametrize(
    ("counts", "mapping"),
    [
        ({"invalid": 1}, (0,)),
        ({"0a": 1}, (0, 1)),
        ({"0": 1}, (0, 1)),
        ({"00": 1}, (0,)),
        ({"00": 1, "1": 1}, (0, 1)),
    ],
)
def test_apply_readout_mitigation_rejects_invalid_raw_bitstring_shape(
    counts: dict[str, int], mapping: tuple[int, ...]
) -> None:
    with pytest.raises(ExperimentValidationError, match="binary bitstrings"):
        apply_readout_mitigation(
            {"setting": counts},
            mapping=mapping,
            mitigation=_FakeMitigation([{"0": 1.0}]),
        )


@pytest.mark.parametrize(
    "outputs",
    [
        [None],
        [[0.5, 0.5]],
        [{}],
        [{"0": math.nan}],
        [{"0": math.inf}],
        [{"0": 0.0, "1": 0.0}],
        [{"0": 2.0}],
        [{"0": True}],
        [{"invalid": 1.0}],
        [{"00": 1.0}],
    ],
)
def test_apply_readout_mitigation_rejects_malformed_corrections(outputs: list[object]) -> None:
    with pytest.raises(ExperimentValidationError):
        apply_readout_mitigation(
            {"setting": {"0": 1}}, mapping=(0,), mitigation=_FakeMitigation(outputs)
        )


@pytest.mark.parametrize("output", [{"0": 1.0}, {"0a": 1.0}, {"000": 1.0}])
def test_apply_readout_mitigation_rejects_short_mixed_or_long_corrected_keys(
    output: dict[str, float],
) -> None:
    with pytest.raises(ExperimentValidationError, match="binary bitstrings"):
        apply_readout_mitigation(
            {"setting": {"00": 1}},
            mapping=(0, 1),
            mitigation=_FakeMitigation([output]),
        )


def test_apply_readout_mitigation_sanitizes_m3_failures() -> None:
    class FailingMitigation:
        def apply_correction(self, counts: dict[str, int], qubits: tuple[int, ...]) -> object:
            raise RuntimeError("token=provider-secret")

    with pytest.raises(JobResultError, match="readout mitigation correction failed") as caught:
        apply_readout_mitigation(
            {"setting": {"0": 1}},
            mapping=(0,),
            mitigation=FailingMitigation(),  # type: ignore[arg-type]
        )

    assert "secret" not in str(caught.value).lower()
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "counts",
    [[{}], [{"0": 0}], [{"0": -1}], [{"0": True}], [{"0": math.nan}]],
)
def test_apply_readout_mitigation_rejects_zero_or_nonfinite_input_totals(
    counts: list[dict[str, object]],
) -> None:
    with pytest.raises(ExperimentValidationError):
        apply_readout_mitigation(
            {"setting": counts[0]},
            mapping=(0,),
            mitigation=_FakeMitigation([{"0": 1.0}]),
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("mapping", [{1: 2}, {0: 2, 2: 3}, {0: 2, 1: 2}])
def test_apply_readout_mitigation_rejects_malformed_logical_mapping(
    mapping: dict[int, int],
) -> None:
    with pytest.raises(ExperimentValidationError):
        apply_readout_mitigation(
            {"setting": {"0": 1}},
            mapping=mapping,
            mitigation=_FakeMitigation([{"0": 1.0}]),
        )


def test_calibration_cache_requires_exact_identity_id_mapping_and_fresh_age() -> None:
    calibration = _record()
    now = calibration.timestamp + timedelta(hours=2)

    assert calibration_cache_is_valid(
        calibration,
        backend_identity="backend-a",
        calibration_id="cal-17",
        qubit_mapping=(2, 0),
        now=now,
        max_age_hours=3,
    )
    for changed in (
        {"backend_identity": "backend-b"},
        {"calibration_id": "cal-18"},
        {"qubit_mapping": (0, 2)},
        {"now": calibration.timestamp + timedelta(hours=4)},
        {"now": calibration.timestamp - timedelta(seconds=1)},
    ):
        arguments: dict[str, object] = {
            "backend_identity": "backend-a",
            "calibration_id": "cal-17",
            "qubit_mapping": (2, 0),
            "now": now,
            "max_age_hours": 3,
        }
        arguments.update(changed)
        assert not calibration_cache_is_valid(calibration, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_age", [0, -1, math.inf, True])
def test_calibration_cache_rejects_invalid_max_age(max_age: object) -> None:
    with pytest.raises(ExperimentValidationError):
        calibration_cache_is_valid(
            _record(),
            backend_identity="backend-a",
            calibration_id="cal-17",
            qubit_mapping=(2, 0),
            now=datetime.now(timezone.utc),
            max_age_hours=max_age,  # type: ignore[arg-type]
        )
