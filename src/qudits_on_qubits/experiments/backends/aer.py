"""Ideal local Qiskit Aer adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from qiskit import transpile

from ..errors import BackendCompatibilityError, BackendUnavailableError, OptionalDependencyError
from ..models import AerIdeal, TranspilationConfig
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    SubmittedJob,
    _validated_circuit_tuple,
    _validated_run_options,
)


class AerAdapter(BaseBackendAdapter):
    def __init__(self, spec: AerIdeal, simulator: Any = None):
        if not isinstance(spec, AerIdeal):
            raise BackendCompatibilityError("AerAdapter requires an AerIdeal specification")
        self._spec = spec
        self._simulator = simulator
        self._identity: BackendIdentity | None = None

    def _load_aer_simulator(self):
        from qiskit_aer import AerSimulator

        return AerSimulator

    def _simulator_instance(self) -> Any:
        if self._simulator is None:
            try:
                simulator_class = self._load_aer_simulator()
            except (ImportError, ModuleNotFoundError) as error:
                raise OptionalDependencyError(
                    "AerIdeal requires qiskit-aer; install it with `pip install qiskit-aer`"
                ) from error
            try:
                self._simulator = simulator_class(method="statevector")
            except Exception as error:
                raise BackendUnavailableError("could not create ideal Aer simulator") from error
        return self._simulator

    def resolve(self) -> BackendIdentity:
        simulator = self._simulator_instance()
        if self._identity is None:
            name = getattr(simulator, "name", "aer_simulator_statevector")
            if callable(name):
                name = name()
            if not isinstance(name, str) or not name:
                name = "aer_simulator_statevector"
            version = getattr(simulator, "backend_version", None)
            if callable(version):
                version = version()
            if isinstance(version, (str, int, float)) and not isinstance(version, bool):
                version = str(version)
            else:
                version = None
            self._identity = BackendIdentity(
                kind="aer_ideal",
                name=name,
                provider="qiskit-aer",
                version=version,
                metadata={"method": "statevector", "noise_model": None},
            )
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(local=True, supports_resume=False)

    def availability(self) -> Availability:
        simulator = self._simulator_instance()
        if not callable(getattr(simulator, "run", None)):
            return Availability(False, "Aer simulator does not expose run")
        return Availability(True)

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        simulator = self._simulator_instance()
        options = {key: value for key, value in config.to_safe_dict().items() if value is not None}
        try:
            compiled = transpile(list(batch), backend=simulator, **options)
        except Exception as error:
            identity = self.resolve()
            raise BackendCompatibilityError(
                f"could not compile circuits for backend {identity.kind}:{identity.name}"
            ) from error
        compiled_batch = tuple(compiled) if isinstance(compiled, (list, tuple)) else (compiled,)
        return CompiledBatch(compiled_batch, self.resolve(), {"transpilation": options})

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        run_options = _validated_run_options(options)
        noise_model = run_options.get("noise_model")
        method = run_options.get("method")
        if noise_model is not None or (method is not None and method != "statevector"):
            raise BackendCompatibilityError("AerIdeal only accepts ideal statevector execution options")
        supplied_seed = run_options.get("seed_simulator")
        if supplied_seed is not None and supplied_seed != self._spec.seed_simulator:
            raise BackendCompatibilityError(
                "run option seed_simulator conflicts with AerIdeal.seed_simulator"
            )
        run_options["seed_simulator"] = self._spec.seed_simulator
        return self._submit_to_backend(self._simulator_instance(), circuits, shots, run_options)

    def restore_job(
        self,
        job_id: str,
        *,
        circuit_count: int | None = None,
        shots: int | None = None,
    ) -> SubmittedJob:
        raise BackendCompatibilityError("local Aer jobs cannot be restored")


__all__ = ["AerAdapter"]
