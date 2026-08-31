from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from numbers import Real
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.workload_metrics import (
    WorkloadMetrics,
    choose_workload_ranking_basis,
    summarize_compiled_workload,
    workload_rank_key,
)


def _structural_circuits() -> tuple[QuantumCircuit, QuantumCircuit]:
    first = QuantumCircuit(2)
    first.cz(0, 1)

    second = QuantumCircuit(2)
    second.x(0)
    second.cz(0, 1)
    second.cz(1, 0)
    return first, second


def test_summarize_compiled_workload_covers_the_complete_structural_workload():
    metrics = summarize_compiled_workload(
        _structural_circuits(),
        settings=(("A0",), ("A1",)),
        physical_mappings=((4, 7), (7, 4)),
        requested_physical_qubits=(4, 7),
    )

    safe = metrics.to_safe_dict()

    assert safe["circuits"] == [
        {
            "circuit_index": 0,
            "setting": ["A0"],
            "depth": 1,
            "size": 1,
            "operation_counts": {"cz": 1},
            "two_qubit_gate_count": 1,
            "native_cz_count": 1,
            "physical_qubit_mapping": [4, 7],
            "instruction_error_cost": None,
            "instruction_duration": None,
        },
        {
            "circuit_index": 1,
            "setting": ["A1"],
            "depth": 3,
            "size": 3,
            "operation_counts": {"cz": 2, "x": 1},
            "two_qubit_gate_count": 2,
            "native_cz_count": 2,
            "physical_qubit_mapping": [7, 4],
            "instruction_error_cost": None,
            "instruction_duration": None,
        },
    ]
    assert safe["aggregate"] == {
        "circuit_count": 2,
        "maximum_depth": 3,
        "total_depth": 4,
        "maximum_two_qubit_gate_count": 2,
        "total_two_qubit_gate_count": 3,
        "maximum_native_cz_count": 2,
        "total_native_cz_count": 3,
        "maximum_size": 3,
        "total_size": 4,
        "physical_qubit_union": [4, 7],
        "uses_exact_physical_qubit_set": True,
        "total_instruction_error_cost": None,
        "total_instruction_duration": None,
    }


def test_summarize_compiled_workload_detects_a_layout_escape():
    metrics = summarize_compiled_workload(
        _structural_circuits(),
        settings=(("A0",), ("A1",)),
        physical_mappings=((4, 7), (4, 8)),
        requested_physical_qubits=(4, 7),
    )

    assert metrics.aggregate["physical_qubit_union"] == (4, 7, 8)
    assert metrics.aggregate["uses_exact_physical_qubit_set"] is False


def test_workload_metrics_defensively_deeply_freezes_content():
    circuit = {
        "setting": ["A0"],
        "operation_counts": {"cz": 1},
    }
    aggregate = {"physical_qubit_union": [4, 7]}

    metrics = WorkloadMetrics(circuits=(circuit,), aggregate=aggregate)
    circuit["setting"].append("mutated")
    circuit["operation_counts"]["x"] = 1
    aggregate["physical_qubit_union"].append(8)

    assert metrics.circuits[0]["setting"] == ("A0",)
    assert metrics.circuits[0]["operation_counts"] == {"cz": 1}
    assert metrics.aggregate["physical_qubit_union"] == (4, 7)
    with pytest.raises(TypeError):
        metrics.circuits[0]["operation_counts"]["x"] = 1
    with pytest.raises(FrozenInstanceError):
        metrics.aggregate = {}


def test_workload_metrics_safe_dict_is_fresh_and_json_serializable():
    metrics = WorkloadMetrics(
        circuits=({"setting": ("A0",), "available": True},),
        aggregate={"cost": 0.25, "missing": None},
    )

    first = metrics.to_safe_dict()
    first["circuits"][0]["setting"].append("changed")
    first["aggregate"]["cost"] = 99.0
    second = metrics.to_safe_dict()

    assert second == {
        "circuits": [{"setting": ["A0"], "available": True}],
        "aggregate": {"cost": 0.25, "missing": None},
    }
    assert first is not second
    assert first["circuits"] is not second["circuits"]
    json.dumps(second, allow_nan=False)


@pytest.mark.parametrize(
    "unsafe",
    [
        object(),
        b"opaque",
        {"not", "json"},
        float("nan"),
        float("inf"),
        {1: "non-string-key"},
    ],
)
def test_workload_metrics_rejects_non_json_content(unsafe):
    with pytest.raises(ExperimentValidationError, match="JSON-safe"):
        WorkloadMetrics(circuits=({"unsafe": unsafe},), aggregate={})


def test_workload_metrics_rejects_recursive_content():
    recursive = []
    recursive.append(recursive)

    with pytest.raises(ExperimentValidationError, match="recursive"):
        WorkloadMetrics(circuits=({"unsafe": recursive},), aggregate={})


def test_experiments_package_exports_workload_metrics_api():
    from qudits_on_qubits import experiments

    assert experiments.WorkloadMetrics is WorkloadMetrics
    assert experiments.choose_workload_ranking_basis is choose_workload_ranking_basis
    assert experiments.summarize_compiled_workload is summarize_compiled_workload
    assert experiments.workload_rank_key is workload_rank_key
    assert "WorkloadMetrics" in experiments.__all__
    assert "choose_workload_ranking_basis" in experiments.__all__
    assert "summarize_compiled_workload" in experiments.__all__
    assert "workload_rank_key" in experiments.__all__


@pytest.mark.parametrize(
    ("circuits", "settings", "physical_mappings"),
    [
        ((), (), ()),
        ((QuantumCircuit(1),), (), ((0,),)),
        ((QuantumCircuit(1),), (("A0",),), ()),
        ((QuantumCircuit(1), QuantumCircuit(1)), (("A0",),), ((0,), (0,))),
    ],
)
def test_summarize_compiled_workload_rejects_empty_or_mismatched_batches(
    circuits, settings, physical_mappings
):
    with pytest.raises(ExperimentValidationError, match="same non-zero length"):
        summarize_compiled_workload(
            circuits,
            settings=settings,
            physical_mappings=physical_mappings,
            requested_physical_qubits=(0,),
        )


@pytest.mark.parametrize(
    "settings",
    [
        "A0",
        ("A0",),
        ((),),
        (("",),),
        ((1,),),
    ],
)
def test_summarize_compiled_workload_rejects_invalid_settings(settings):
    with pytest.raises(ExperimentValidationError, match="settings"):
        summarize_compiled_workload(
            (QuantumCircuit(1),),
            settings=settings,
            physical_mappings=((0,),),
            requested_physical_qubits=(0,),
        )


@pytest.mark.parametrize(
    "physical_mappings",
    [
        "0",
        ((),),
        ((True,),),
        ((-1,),),
        ((1.0,),),
        ((0, 0),),
    ],
)
def test_summarize_compiled_workload_rejects_invalid_physical_mappings(
    physical_mappings,
):
    with pytest.raises(ExperimentValidationError, match="physical_mappings"):
        summarize_compiled_workload(
            (QuantumCircuit(1),),
            settings=(("A0",),),
            physical_mappings=physical_mappings,
            requested_physical_qubits=(0,),
        )


@pytest.mark.parametrize(
    "requested_physical_qubits",
    [
        "0",
        (),
        (True,),
        (-1,),
        (1.0,),
        (0, 0),
    ],
)
def test_summarize_compiled_workload_rejects_invalid_requested_qubits(
    requested_physical_qubits,
):
    with pytest.raises(ExperimentValidationError, match="requested_physical_qubits"):
        summarize_compiled_workload(
            (QuantumCircuit(1),),
            settings=(("A0",),),
            physical_mappings=((0,),),
            requested_physical_qubits=requested_physical_qubits,
        )


def test_summarize_compiled_workload_does_not_mutate_inputs():
    circuits = list(_structural_circuits())
    settings = [["A0"], ["A1"]]
    physical_mappings = [[4, 7], [7, 4]]
    requested = [4, 7]

    summarize_compiled_workload(
        circuits,
        settings=settings,
        physical_mappings=physical_mappings,
        requested_physical_qubits=requested,
    )

    assert settings == [["A0"], ["A1"]]
    assert physical_mappings == [[4, 7], [7, 4]]
    assert requested == [4, 7]


def _measured_cz_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.cz(0, 1)
    circuit.cz(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    return circuit


def test_summarize_compiled_workload_counts_all_calibrated_instructions():
    target = {
        "cz": {
            (0, 1): SimpleNamespace(error=0.01, duration=1e-8),
        },
        "measure": {
            (0,): SimpleNamespace(error=0.02, duration=1e-8),
            (1,): SimpleNamespace(error=0.03, duration=1e-8),
        },
    }

    metrics = summarize_compiled_workload(
        (_measured_cz_circuit(),),
        settings=(("A0",),),
        physical_mappings=((4, 7),),
        requested_physical_qubits=(4, 7),
        target=target,
    )

    expected_error = (
        2 * -math.log1p(-0.01)
        + -math.log1p(-0.02)
        + -math.log1p(-0.03)
    )
    assert metrics.circuits[0]["instruction_error_cost"] == pytest.approx(
        expected_error
    )
    assert metrics.aggregate["total_instruction_error_cost"] == pytest.approx(
        expected_error
    )
    assert metrics.circuits[0]["instruction_duration"] == pytest.approx(4e-8)
    assert metrics.aggregate["total_instruction_duration"] == pytest.approx(4e-8)


def test_directives_remain_evidence_but_are_not_ranked_or_calibrated():
    circuit = QuantumCircuit(2, 2)
    circuit.cz(0, 1)
    circuit.barrier(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    target = {
        "cz": {(0, 1): SimpleNamespace(error=0.01, duration=1e-8)},
        "measure": {
            (0,): SimpleNamespace(error=0.02, duration=1e-8),
            (1,): SimpleNamespace(error=0.03, duration=1e-8),
        },
    }

    metrics = summarize_compiled_workload(
        (circuit,),
        settings=(("A0",),),
        physical_mappings=((4, 7),),
        requested_physical_qubits=(4, 7),
        target=target,
    )

    expected_error = sum(
        -math.log1p(-error) for error in (0.01, 0.02, 0.03)
    )
    assert metrics.circuits[0]["operation_counts"] == {
        "barrier": 1,
        "cz": 1,
        "measure": 2,
    }
    assert metrics.circuits[0]["two_qubit_gate_count"] == 1
    assert metrics.circuits[0]["native_cz_count"] == 1
    assert metrics.circuits[0]["instruction_error_cost"] == pytest.approx(
        expected_error
    )
    assert metrics.circuits[0]["instruction_duration"] == pytest.approx(3e-8)


class _HostileReal:
    def __init__(self, exception_type=RuntimeError):
        self._exception_type = exception_type

    def __float__(self):
        raise self._exception_type("secret provider conversion failure")


Real.register(_HostileReal)


@pytest.mark.parametrize("hostile_field", ["error", "duration"])
def test_hostile_real_conversion_only_invalidates_its_metric(hostile_field):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    properties = {"error": 0.1, "duration": 1e-8}
    properties[hostile_field] = _HostileReal()
    target = {"x": {(0,): SimpleNamespace(**properties)}}

    metrics = summarize_compiled_workload(
        (circuit,),
        settings=(("A0",),),
        physical_mappings=((4,),),
        requested_physical_qubits=(4,),
        target=target,
    )

    if hostile_field == "error":
        assert metrics.circuits[0]["instruction_error_cost"] is None
        assert metrics.circuits[0]["instruction_duration"] == pytest.approx(1e-8)
    else:
        assert metrics.circuits[0]["instruction_error_cost"] == pytest.approx(
            -math.log1p(-0.1)
        )
        assert metrics.circuits[0]["instruction_duration"] is None


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_hostile_real_conversion_preserves_critical_exceptions(exception_type):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = {
        "x": {
            (0,): SimpleNamespace(
                error=_HostileReal(exception_type),
                duration=1e-8,
            )
        }
    }

    with pytest.raises(exception_type, match="secret provider conversion failure"):
        summarize_compiled_workload(
            (circuit,),
            settings=(("A0",),),
            physical_mappings=((4,),),
            requested_physical_qubits=(4,),
            target=target,
        )


class _HostileDirectiveOperation:
    name = "x"

    def __init__(self, exception_type):
        self._exception_type = exception_type

    @property
    def _directive(self):
        raise self._exception_type("secret directive failure")


def _circuit_with_hostile_directive_flag(exception_type):
    instruction = SimpleNamespace(
        operation=_HostileDirectiveOperation(exception_type),
        qubits=(),
    )
    return SimpleNamespace(
        data=(instruction,),
        count_ops=lambda: {"x": 1},
        depth=lambda: 1,
        size=lambda: 1,
    )


def test_directive_flag_failure_is_safely_reported():
    with pytest.raises(ExperimentValidationError, match="directive") as captured:
        summarize_compiled_workload(
            (_circuit_with_hostile_directive_flag(RuntimeError),),
            settings=(("A0",),),
            physical_mappings=((4,),),
            requested_physical_qubits=(4,),
        )

    assert "secret directive failure" not in str(captured.value)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_directive_flag_access_preserves_critical_exceptions(exception_type):
    with pytest.raises(exception_type, match="secret directive failure"):
        summarize_compiled_workload(
            (_circuit_with_hostile_directive_flag(exception_type),),
            settings=(("A0",),),
            physical_mappings=((4,),),
            requested_physical_qubits=(4,),
        )


def test_missing_instruction_error_invalidates_only_error_totals_without_partial_sum():
    valid = QuantumCircuit(1)
    valid.x(0)
    unavailable = QuantumCircuit(1)
    unavailable.h(0)
    target = {
        "x": {(0,): SimpleNamespace(error=0.1, duration=1e-8)},
        "h": {(0,): SimpleNamespace(duration=2e-8)},
    }

    metrics = summarize_compiled_workload(
        (valid, unavailable),
        settings=(("A0",), ("A1",)),
        physical_mappings=((4,), (4,)),
        requested_physical_qubits=(4,),
        target=target,
    )

    assert metrics.circuits[0]["instruction_error_cost"] == pytest.approx(
        -math.log1p(-0.1)
    )
    assert metrics.circuits[1]["instruction_error_cost"] is None
    assert metrics.aggregate["total_instruction_error_cost"] is None
    assert metrics.circuits[0]["instruction_duration"] == pytest.approx(1e-8)
    assert metrics.circuits[1]["instruction_duration"] == pytest.approx(2e-8)
    assert metrics.aggregate["total_instruction_duration"] == pytest.approx(3e-8)


@pytest.mark.parametrize(
    "invalid_error",
    [True, "0.1", float("nan"), float("inf"), -0.01, 1.0],
)
def test_invalid_error_property_does_not_hide_valid_duration(invalid_error):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = {
        "x": {(0,): SimpleNamespace(error=invalid_error, duration=1e-8)},
    }

    metrics = summarize_compiled_workload(
        (circuit,),
        settings=(("A0",),),
        physical_mappings=((4,),),
        requested_physical_qubits=(4,),
        target=target,
    )

    assert metrics.circuits[0]["instruction_error_cost"] is None
    assert metrics.aggregate["total_instruction_error_cost"] is None
    assert metrics.circuits[0]["instruction_duration"] == pytest.approx(1e-8)
    assert metrics.aggregate["total_instruction_duration"] == pytest.approx(1e-8)


@pytest.mark.parametrize(
    "invalid_duration",
    [True, "1e-8", float("nan"), float("inf"), -1e-8],
)
def test_invalid_duration_property_does_not_hide_valid_error(invalid_duration):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = {
        "x": {(0,): SimpleNamespace(error=0.1, duration=invalid_duration)},
    }

    metrics = summarize_compiled_workload(
        (circuit,),
        settings=(("A0",),),
        physical_mappings=((4,),),
        requested_physical_qubits=(4,),
        target=target,
    )

    assert metrics.circuits[0]["instruction_error_cost"] == pytest.approx(
        -math.log1p(-0.1)
    )
    assert metrics.aggregate["total_instruction_error_cost"] == pytest.approx(
        -math.log1p(-0.1)
    )
    assert metrics.circuits[0]["instruction_duration"] is None
    assert metrics.aggregate["total_instruction_duration"] is None


def test_calibration_lookup_failure_marks_metrics_unavailable():
    class FailingTarget:
        def __getitem__(self, _operation_name):
            raise RuntimeError("opaque provider failure")

    circuit = QuantumCircuit(1)
    circuit.x(0)

    metrics = summarize_compiled_workload(
        (circuit,),
        settings=(("A0",),),
        physical_mappings=((4,),),
        requested_physical_qubits=(4,),
        target=FailingTarget(),
    )

    assert metrics.circuits[0]["instruction_error_cost"] is None
    assert metrics.circuits[0]["instruction_duration"] is None
    assert metrics.aggregate["total_instruction_error_cost"] is None
    assert metrics.aggregate["total_instruction_duration"] is None


class _MemoryErrorTarget:
    def __getitem__(self, _operation_name):
        raise MemoryError("target lookup exhausted")


class _MemoryErrorProperties:
    def __init__(self, failure_site):
        self._failure_site = failure_site

    @property
    def error(self):
        if self._failure_site == "error":
            raise MemoryError("error getter exhausted")
        return 0.1

    @property
    def duration(self):
        if self._failure_site == "duration":
            raise MemoryError("duration getter exhausted")
        return 1e-8


@pytest.mark.parametrize(
    ("failure_site", "message"),
    [
        ("target", "target lookup exhausted"),
        ("error", "error getter exhausted"),
        ("duration", "duration getter exhausted"),
    ],
)
def test_calibration_access_preserves_memory_error(failure_site, message):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = (
        _MemoryErrorTarget()
        if failure_site == "target"
        else {"x": {(0,): _MemoryErrorProperties(failure_site)}}
    )

    with pytest.raises(MemoryError, match=message):
        summarize_compiled_workload(
            (circuit,),
            settings=(("A0",),),
            physical_mappings=((4,),),
            requested_physical_qubits=(4,),
            target=target,
        )


def _ranking_metrics(
    *,
    error=0.1,
    duration=1.0,
    maximum_two_qubit_gate_count=2,
    total_two_qubit_gate_count=3,
    maximum_depth=4,
    total_depth=5,
):
    return WorkloadMetrics(
        circuits=(),
        aggregate={
            "total_instruction_error_cost": error,
            "total_instruction_duration": duration,
            "maximum_two_qubit_gate_count": maximum_two_qubit_gate_count,
            "total_two_qubit_gate_count": total_two_qubit_gate_count,
            "maximum_depth": maximum_depth,
            "total_depth": total_depth,
        },
    )


def test_choose_workload_ranking_basis_is_calibration_all_or_nothing():
    first = _ranking_metrics(error=0.1, duration=1.0)
    second = _ranking_metrics(error=0.2, duration=2.0)

    assert choose_workload_ranking_basis(
        (first, second), prefer_calibration=True
    ) == (True, True)
    assert choose_workload_ranking_basis(
        (first, second), prefer_calibration=False
    ) == (False, False)

    for incomplete in (
        _ranking_metrics(error=None, duration=2.0),
        _ranking_metrics(error=0.2, duration=None),
        _ranking_metrics(error="unavailable", duration=2.0),
        _ranking_metrics(error=-0.1, duration=2.0),
        _ranking_metrics(error=0.2, duration=-1.0),
    ):
        assert choose_workload_ranking_basis(
            (first, incomplete), prefer_calibration=True
        ) == (False, False)


def test_workload_rank_key_has_exact_calibrated_and_structural_shapes():
    metrics = _ranking_metrics()

    assert workload_rank_key(
        metrics,
        use_error=True,
        use_duration=True,
        seed=7,
        layout=(4, 7),
    ) == (0.1, 1.0, 2, 3, 4, 5, 7, (4, 7))
    assert workload_rank_key(
        metrics,
        use_error=False,
        use_duration=False,
        seed=7,
        layout=(4, 7),
    ) == (2, 3, 4, 5, 7, (4, 7))


def test_calibrated_rank_key_orders_error_then_duration_then_structure():
    common = {
        "use_error": True,
        "use_duration": True,
        "seed": 7,
        "layout": (4, 7),
    }

    assert workload_rank_key(
        _ranking_metrics(
            error=0.1,
            duration=100.0,
            maximum_two_qubit_gate_count=99,
        ),
        **common,
    ) < workload_rank_key(
        _ranking_metrics(
            error=0.2,
            duration=0.0,
            maximum_two_qubit_gate_count=0,
        ),
        **common,
    )
    assert workload_rank_key(
        _ranking_metrics(
            error=0.1,
            duration=1.0,
            maximum_two_qubit_gate_count=99,
        ),
        **common,
    ) < workload_rank_key(
        _ranking_metrics(
            error=0.1,
            duration=2.0,
            maximum_two_qubit_gate_count=0,
        ),
        **common,
    )
    assert workload_rank_key(
        _ranking_metrics(maximum_two_qubit_gate_count=1),
        **common,
    ) < workload_rank_key(
        _ranking_metrics(maximum_two_qubit_gate_count=2),
        **common,
    )


def test_rank_key_uses_seed_then_layout_as_final_tie_breaks():
    metrics = _ranking_metrics()

    assert workload_rank_key(
        metrics,
        use_error=False,
        use_duration=False,
        seed=1,
        layout=(7, 4),
    ) < workload_rank_key(
        metrics,
        use_error=False,
        use_duration=False,
        seed=2,
        layout=(4, 7),
    )
    assert workload_rank_key(
        metrics,
        use_error=False,
        use_duration=False,
        seed=1,
        layout=(4, 7),
    ) < workload_rank_key(
        metrics,
        use_error=False,
        use_duration=False,
        seed=1,
        layout=(7, 4),
    )


@pytest.mark.parametrize(
    ("candidates", "prefer_calibration"),
    [
        ((), True),
        ((_ranking_metrics(),), 1),
        ((object(),), True),
        ("not metrics", True),
    ],
)
def test_choose_workload_ranking_basis_rejects_invalid_inputs(
    candidates, prefer_calibration
):
    with pytest.raises(ExperimentValidationError):
        choose_workload_ranking_basis(
            candidates,
            prefer_calibration=prefer_calibration,
        )


@pytest.mark.parametrize(
    ("use_error", "use_duration"),
    [(1, True), (True, 1), (True, False), (False, True)],
)
def test_workload_rank_key_requires_equal_strict_bool_flags(
    use_error, use_duration
):
    with pytest.raises(ExperimentValidationError, match="use_error"):
        workload_rank_key(
            _ranking_metrics(),
            use_error=use_error,
            use_duration=use_duration,
            seed=1,
            layout=(4, 7),
        )


@pytest.mark.parametrize("seed", [True, -1, 1.0])
def test_workload_rank_key_rejects_invalid_seed(seed):
    with pytest.raises(ExperimentValidationError, match="seed"):
        workload_rank_key(
            _ranking_metrics(),
            use_error=False,
            use_duration=False,
            seed=seed,
            layout=(4, 7),
        )


@pytest.mark.parametrize(
    "layout",
    [(), "47", (True,), (-1,), (4.0,), (4, 4)],
)
def test_workload_rank_key_rejects_invalid_layout(layout):
    with pytest.raises(ExperimentValidationError, match="layout"):
        workload_rank_key(
            _ranking_metrics(),
            use_error=False,
            use_duration=False,
            seed=1,
            layout=layout,
        )


def test_workload_rank_key_rejects_unavailable_calibration_metrics():
    for metrics in (
        _ranking_metrics(error=None),
        _ranking_metrics(duration=None),
        _ranking_metrics(error="unavailable"),
        _ranking_metrics(error=-0.1),
        _ranking_metrics(duration=-1.0),
    ):
        with pytest.raises(ExperimentValidationError, match="unavailable"):
            workload_rank_key(
                metrics,
                use_error=True,
                use_duration=True,
                seed=1,
                layout=(4, 7),
            )
