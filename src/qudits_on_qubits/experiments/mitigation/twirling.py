"""Client-less IQM circuit-twirling transformation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Any

from qiskit import QuantumCircuit

from ..errors import BackendCompatibilityError, OptionalDependencyError


_INSTALL_HINT = "pip install -e .[mitigation]"


def _load_iqm_twirling_api() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module(
        "iqm.error_reduction_tools.twirling.twirling_api"
    )
    return module.TwirlingConfiguration, module.CircuitTwirler


@dataclass(frozen=True)
class TwirledBatch:
    """Flattened variants with their original-circuit membership."""

    circuits: tuple[QuantumCircuit, ...]
    original_indices: tuple[int, ...]
    instance_indices: tuple[int, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        circuits = tuple(self.circuits)
        original_indices = tuple(self.original_indices)
        instance_indices = tuple(self.instance_indices)
        if not circuits or any(
            not isinstance(circuit, QuantumCircuit) for circuit in circuits
        ):
            raise BackendCompatibilityError(
                "twirled circuit batch must contain Qiskit circuits"
            )
        if not (
            len(circuits) == len(original_indices) == len(instance_indices)
        ):
            raise BackendCompatibilityError(
                "twirled circuit membership does not match circuit count"
            )
        if any(type(index) is not int or index < 0 for index in original_indices):
            raise BackendCompatibilityError("twirled original indices are invalid")
        if any(type(index) is not int or index < 0 for index in instance_indices):
            raise BackendCompatibilityError("twirled instance indices are invalid")
        object.__setattr__(self, "circuits", circuits)
        object.__setattr__(self, "original_indices", original_indices)
        object.__setattr__(self, "instance_indices", instance_indices)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def twirl_iqm_circuits(
    circuits: Sequence[QuantumCircuit],
    *,
    instances: int,
    seed: int | None,
    _api_loader: Callable[[], tuple[type[Any], type[Any]]] = _load_iqm_twirling_api,
) -> TwirledBatch:
    """Generate ordered IQM circuit-twirled variants without submission."""

    batch = tuple(circuits)
    if not batch or any(not isinstance(circuit, QuantumCircuit) for circuit in batch):
        raise BackendCompatibilityError(
            "IQM circuit twirling requires a non-empty Qiskit circuit batch"
        )
    if type(instances) is not int or instances <= 0:
        raise BackendCompatibilityError(
            "twirling instances must be a positive integer"
        )
    if seed is not None and (type(seed) is not int or seed < 0):
        raise BackendCompatibilityError(
            "twirling seed must be a non-negative integer or None"
        )

    try:
        configuration_type, twirler_type = _api_loader()
    except (ImportError, AttributeError):
        raise OptionalDependencyError(
            "IQM circuit twirling requires iqm-error-reduction-tools; "
            f"install it with `{_INSTALL_HINT}`"
        ) from None

    try:
        configuration = configuration_type(
            readout_twirl_strategy="NONE",
            circuit_twirling=True,
            num_twirling_instances=instances,
            seed=seed,
        )
        twirler = twirler_type(config=configuration)
        groups_value = twirler.twirl(list(batch)).get_twirled_circuits(
            return_qiskit=True
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except OptionalDependencyError:
        raise
    except Exception as error:
        raise BackendCompatibilityError(
            f"IQM circuit twirling failed ({type(error).__name__})"
        ) from None

    if isinstance(groups_value, (str, bytes)):
        raise BackendCompatibilityError("IQM twirler returned malformed groups")
    try:
        groups = tuple(groups_value)
    except TypeError:
        raise BackendCompatibilityError(
            "IQM twirler returned malformed groups"
        ) from None
    if len(groups) != len(batch):
        raise BackendCompatibilityError(
            "IQM twirler group count does not match input circuit count"
        )

    flattened: list[QuantumCircuit] = []
    original_indices: list[int] = []
    instance_indices: list[int] = []
    for original_index, group_value in enumerate(groups):
        if isinstance(group_value, (str, bytes)):
            raise BackendCompatibilityError("IQM twirler returned malformed group")
        try:
            group = tuple(group_value)
        except TypeError:
            raise BackendCompatibilityError(
                "IQM twirler returned malformed group"
            ) from None
        if len(group) != instances:
            raise BackendCompatibilityError(
                "IQM twirler instance count does not match configuration"
            )
        if any(not isinstance(circuit, QuantumCircuit) for circuit in group):
            raise BackendCompatibilityError(
                "IQM twirler returned a non-Qiskit circuit"
            )
        flattened.extend(group)
        original_indices.extend([original_index] * instances)
        instance_indices.extend(range(instances))

    return TwirledBatch(
        circuits=tuple(flattened),
        original_indices=tuple(original_indices),
        instance_indices=tuple(instance_indices),
        metadata={
            "provider": "iqm-error-reduction-tools",
            "method": "circuit_twirling",
            "readout_strategy": "NONE",
            "instances_per_circuit": instances,
            "seed": seed,
        },
    )
