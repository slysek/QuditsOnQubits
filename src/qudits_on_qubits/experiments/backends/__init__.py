"""Unified backend adapter contract and built-in adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import BackendCompatibilityError
from ..models import AerIdeal, CustomBackend, IQMHardware, NoisySimulator, PiastQHardware
from .aer import AerAdapter, NoisyAerAdapter, build_noisy_adapter
from .base import (
    Availability,
    BackendAdapter,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
)
from .custom import CustomBackendAdapter
from .iqm import IQMAdapter
from .piastq import PiastQAdapter


AdapterFactory = Callable[..., BackendAdapter]


class BackendAdapterRegistry:
    """Extensible mapping from backend specification types to adapter factories."""

    def __init__(self) -> None:
        self._factories: dict[type[Any], AdapterFactory] = {}

    def register(self, specification_type: type[Any], factory: AdapterFactory) -> None:
        if not isinstance(specification_type, type) or not callable(factory):
            raise BackendCompatibilityError("backend adapter registration requires a type and callable factory")
        self._factories[specification_type] = factory

    def create(self, specification: Any, **injected: Any) -> BackendAdapter:
        for specification_type, factory in self._factories.items():
            if isinstance(specification, specification_type):
                return factory(specification, **injected)
        raise BackendCompatibilityError(
            f"unsupported backend specification type {type(specification).__name__}"
        )


backend_registry = BackendAdapterRegistry()
backend_registry.register(AerIdeal, AerAdapter)
backend_registry.register(CustomBackend, CustomBackendAdapter)
backend_registry.register(IQMHardware, IQMAdapter)
backend_registry.register(PiastQHardware, PiastQAdapter)
backend_registry.register(NoisySimulator, build_noisy_adapter)


def create_backend_adapter(specification: Any, **injected: Any) -> BackendAdapter:
    return backend_registry.create(specification, **injected)


__all__ = [
    "AerAdapter",
    "Availability",
    "BackendAdapter",
    "BackendAdapterRegistry",
    "BackendCapabilities",
    "BackendIdentity",
    "BaseBackendAdapter",
    "CompiledBatch",
    "CustomBackendAdapter",
    "ExecutionResult",
    "IQMAdapter",
    "NoisyAerAdapter",
    "PiastQAdapter",
    "SubmittedJob",
    "backend_registry",
    "build_noisy_adapter",
    "create_backend_adapter",
]
