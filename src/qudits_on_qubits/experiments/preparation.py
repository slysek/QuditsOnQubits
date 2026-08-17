"""Preparation of validated artifacts for sampler-based Bell measurements."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qudits_on_qubits.bell_measurements import build_sampler_circuits_for_candidate

from .artifacts import BasisArtifacts
from .errors import ExperimentValidationError


@dataclass(frozen=True)
class PreparedMeasurements:
    """Sampler circuits and their in-memory postprocessing metadata."""

    circuits: tuple[Any, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "circuits", tuple(self.circuits))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def prepare_measurements(artifacts: BasisArtifacts) -> PreparedMeasurements:
    """Build measured sampler circuits for a validated candidate artifact."""
    qutrit_qubits = tuple((index, index + 1) for index in range(0, artifacts.state_circuit.num_qubits, 2))
    try:
        circuits, metadata = build_sampler_circuits_for_candidate(
            artifacts.state,
            artifacts.state_circuit,
            artifacts.encoding,
            qutrit_qubits=qutrit_qubits,
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
    }
