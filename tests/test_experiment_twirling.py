from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    OptionalDependencyError,
)


def measured_circuit(name: str) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1, name=name)
    circuit.measure(0, 0)
    return circuit


def test_twirling_module_import_is_lazy():
    from qudits_on_qubits.experiments.mitigation import twirling

    assert "iqm.error_reduction_tools.twirling.twirling_api" not in sys.modules
    assert callable(twirling.twirl_iqm_circuits)


def test_default_api_loader_resolves_official_module_lazily(monkeypatch):
    from qudits_on_qubits.experiments.mitigation import twirling

    configuration = object()
    circuit_twirler = object()
    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(
            TwirlingConfiguration=configuration,
            CircuitTwirler=circuit_twirler,
        )

    monkeypatch.setattr(twirling.importlib, "import_module", fake_import)

    assert twirling._load_iqm_twirling_api() == (configuration, circuit_twirler)
    assert imported == ["iqm.error_reduction_tools.twirling.twirling_api"]


def test_missing_iqm_twirling_dependency_uses_safe_installation_hint():
    from qudits_on_qubits.experiments.mitigation.twirling import (
        twirl_iqm_circuits,
    )

    def missing_api():
        raise ModuleNotFoundError("token=do-not-leak")

    with pytest.raises(OptionalDependencyError) as caught:
        twirl_iqm_circuits(
            (measured_circuit("compiled"),),
            instances=2,
            seed=7,
            _api_loader=missing_api,
        )

    assert "pip install -e .[mitigation]" in str(caught.value)
    assert "token=do-not-leak" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("circuits", "instances", "seed"),
    [
        ((), 2, None),
        ((object(),), 2, None),
        ((measured_circuit("compiled"),), 0, None),
        ((measured_circuit("compiled"),), True, None),
        ((measured_circuit("compiled"),), 2, -1),
        ((measured_circuit("compiled"),), 2, True),
    ],
)
def test_twirling_rejects_invalid_direct_inputs(circuits, instances, seed):
    from qudits_on_qubits.experiments.mitigation.twirling import (
        twirl_iqm_circuits,
    )

    with pytest.raises(BackendCompatibilityError, match="twirl"):
        twirl_iqm_circuits(
            circuits,
            instances=instances,
            seed=seed,
            _api_loader=lambda: (object, object),
        )


def test_twirling_sanitizes_provider_transformation_failure():
    from qudits_on_qubits.experiments.mitigation.twirling import (
        twirl_iqm_circuits,
    )

    class FailingConfiguration:
        def __init__(self, **_kwargs):
            raise RuntimeError("token=do-not-leak")

    with pytest.raises(BackendCompatibilityError) as caught:
        twirl_iqm_circuits(
            (measured_circuit("compiled"),),
            instances=2,
            seed=None,
            _api_loader=lambda: (FailingConfiguration, object),
        )

    assert str(caught.value) == "IQM circuit twirling failed (RuntimeError)"
    assert caught.value.__cause__ is None


def test_official_wrapper_configuration_and_ordered_flattening():
    from qudits_on_qubits.experiments.mitigation.twirling import (
        twirl_iqm_circuits,
    )

    compiled = (measured_circuit("compiled-0"), measured_circuit("compiled-1"))
    variants = tuple(
        tuple(measured_circuit(f"twirled-{original}-{instance}") for instance in range(3))
        for original in range(2)
    )
    recorded = {}

    class FakeConfiguration:
        def __init__(self, **kwargs):
            recorded["configuration"] = kwargs

    class FakeCircuitTwirler:
        def __init__(self, client=None, config=None):
            recorded["client"] = client
            recorded["config"] = config

        def twirl(self, circuits):
            recorded["circuits"] = circuits
            return self

        def get_twirled_circuits(self, return_qiskit=False):
            recorded["return_qiskit"] = return_qiskit
            return [list(group) for group in variants]

    result = twirl_iqm_circuits(
        compiled,
        instances=3,
        seed=12345,
        _api_loader=lambda: (FakeConfiguration, FakeCircuitTwirler),
    )

    assert recorded["configuration"] == {
        "readout_twirl_strategy": "NONE",
        "circuit_twirling": True,
        "num_twirling_instances": 3,
        "seed": 12345,
    }
    assert recorded["client"] is None
    assert recorded["circuits"] == list(compiled)
    assert recorded["return_qiskit"] is True
    assert result.circuits == variants[0] + variants[1]
    assert result.original_indices == (0, 0, 0, 1, 1, 1)
    assert result.instance_indices == (0, 1, 2, 0, 1, 2)
    assert result.metadata == {
        "provider": "iqm-error-reduction-tools",
        "method": "circuit_twirling",
        "readout_strategy": "NONE",
        "instances_per_circuit": 3,
        "seed": 12345,
    }


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [[measured_circuit("only-one-input")]],
        [
            [measured_circuit("0-0")],
            [measured_circuit("1-0"), measured_circuit("1-1")],
        ],
        [
            [measured_circuit("0-0"), object()],
            [measured_circuit("1-0"), measured_circuit("1-1")],
        ],
    ],
    ids=("no-groups", "missing-group", "wrong-size", "not-qiskit"),
)
def test_malformed_twirler_output_is_rejected(groups):
    from qudits_on_qubits.experiments.mitigation.twirling import (
        twirl_iqm_circuits,
    )

    class FakeConfiguration:
        def __init__(self, **_kwargs):
            pass

    class FakeCircuitTwirler:
        def __init__(self, **_kwargs):
            pass

        def twirl(self, _circuits):
            return self

        def get_twirled_circuits(self, return_qiskit=False):
            assert return_qiskit is True
            return groups

    with pytest.raises(BackendCompatibilityError, match="twirl"):
        twirl_iqm_circuits(
            (measured_circuit("compiled-0"), measured_circuit("compiled-1")),
            instances=2,
            seed=None,
            _api_loader=lambda: (FakeConfiguration, FakeCircuitTwirler),
        )
