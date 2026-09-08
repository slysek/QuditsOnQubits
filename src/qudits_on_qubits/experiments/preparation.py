"""Preparation of validated artifacts for sampler-based Bell measurements."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from qudits_on_qubits.bell_measurements import build_sampler_circuits_for_candidate

from .artifacts import BasisArtifacts
from .errors import ExperimentValidationError
from .setting_schedule import SettingSchedule


@dataclass(frozen=True)
class PreparedMeasurements:
    """Sampler circuits and their in-memory postprocessing metadata."""

    circuits: tuple[Any, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "circuits", tuple(self.circuits))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))



def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(value) for key, value in values.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        frozen = value.copy()
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def prepare_measurements(artifacts: BasisArtifacts, schedule: SettingSchedule | None = None) -> PreparedMeasurements:
    """Build measured sampler circuits for a validated candidate artifact."""
    if schedule is not None:
        if not isinstance(schedule, SettingSchedule):
            raise ExperimentValidationError("schedule must be SettingSchedule or None")
        if schedule.state != artifacts.state:
            raise ExperimentValidationError("schedule state must match basis artifact state")
        if not schedule.complete:
            raise ExperimentValidationError("measurement preparation requires a complete schedule")
    qutrit_qubits = tuple(
        (index, index + 1)
        for index in range(artifacts.state_circuit.num_qubits - 2, -1, -2)
    )
    try:
        circuits, metadata = build_sampler_circuits_for_candidate(
            artifacts.state,
            artifacts.state_circuit,
            artifacts.encoding,
            qutrit_qubits=qutrit_qubits,
            **({"explicit_settings": [block.settings for block in schedule.blocks]} if schedule is not None else {}),
        )
    except Exception as error:
        raise ExperimentValidationError("could not prepare sampler measurement circuits") from error
    circuits = tuple(circuits)
    metadata = {**metadata, "state": artifacts.state}
    settings = metadata.get("setting_by_circuit_index")
    if not circuits:
        raise ExperimentValidationError("measurement preparation produced no circuits")
    if not isinstance(settings, list) or len(settings) != len(circuits):
        raise ExperimentValidationError("measurement setting count must match circuit count")
    if any(circuit.num_clbits == 0 or not any(item.operation.name == "measure" for item in circuit.data) for circuit in circuits):
        raise ExperimentValidationError("prepared measurement circuits must contain measurements")
    if schedule is not None:
        catalog_index_by_setting = {tuple(setting): index for index, setting in enumerate(settings)}
        metadata["catalog_index_by_setting"] = catalog_index_by_setting
        metadata["catalog_index_by_block_id"] = {
            block.block_id: catalog_index_by_setting[block.settings] for block in schedule.blocks
        }
    return PreparedMeasurements(circuits=circuits, metadata=metadata)


def metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only JSON-safe reproducibility metadata, omitting operator payloads."""
    settings = metadata.get("setting_by_circuit_index", ())
    pairs = metadata.get("qutrit_qubits", ())
    return {
        "state": metadata.get("state", metadata.get("candidate")),
        "candidate": metadata.get("candidate"),
        "setting_by_circuit_index": [list(setting) for setting in settings],
        "qutrit_qubits": [list(pair) for pair in pairs],
        "d": metadata.get("d"),
        "circuit_count": len(settings),
        **({
            "catalog_index_by_block_id": dict(metadata["catalog_index_by_block_id"]),
            "catalog_index_by_setting": [
                {"settings": list(setting), "catalog_index": index}
                for setting, index in metadata["catalog_index_by_setting"].items()
            ],
        } if "catalog_index_by_block_id" in metadata else {}),
    }
