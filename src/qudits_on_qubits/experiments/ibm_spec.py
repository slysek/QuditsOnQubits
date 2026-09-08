"""Credential-free IBM hardware selection for experiment specifications."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, ClassVar, Mapping

from .errors import ExperimentValidationError
from .execution import ExecutionMode
from .safety import unsafe_persisted_text, validate_persisted_strings


@dataclass(frozen=True)
class IBMHardware:
    device: str
    account_name: str | None = None
    instance: str | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.HARDWARE

    def __post_init__(self) -> None:
        for field in ("device", "account_name", "instance"):
            value = getattr(self, field)
            if value is None and field != "device":
                continue
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 512
                or unsafe_persisted_text(value)
            ):
                raise ExperimentValidationError(f"IBM {field} must be a safe non-empty string")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.device):
            raise ExperimentValidationError("IBM device must be a backend name")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "ibm_hardware",
            "device": self.device,
            "account_name": self.account_name,
            "instance": self.instance,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IBMHardware":
        if not isinstance(data, Mapping):
            raise ExperimentValidationError("IBM specification must be a mapping")
        validate_persisted_strings(
            data, description="IBM specification", error_type=ExperimentValidationError
        )
        if data.get("kind", "ibm_hardware") != "ibm_hardware":
            raise ExperimentValidationError("IBM specification has an invalid kind")
        if data.get("execution_mode", "hardware") != ExecutionMode.HARDWARE:
            raise ExperimentValidationError("execution_mode does not match backend kind")
        if set(data) - {"kind", "device", "account_name", "instance", "execution_mode"}:
            raise ExperimentValidationError("IBM specification contains unsupported fields")
        return cls(data.get("device"), data.get("account_name"), data.get("instance"))


__all__ = ["IBMHardware"]
