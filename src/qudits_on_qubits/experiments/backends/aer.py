"""Ideal local Qiskit Aer adapter."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from qiskit import transpile

from ..errors import BackendCompatibilityError, BackendUnavailableError, OptionalDependencyError
from ..models import AerIdeal, IQMHardware, NoisySimulator, TranspilationConfig
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    SubmittedJob,
    _exception_name,
    _validated_circuit_tuple,
    _validated_run_options,
)


_EXPLICIT_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+@-]{0,127}\Z")
_CREDENTIAL_MARKERS = ("token=", "api_key=", "password=", "secret=")


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
                    "AerIdeal requires qiskit-aer; install it with `pip install qiskit-aer` "
                    f"({_exception_name(error)})"
                ) from None
            try:
                self._simulator = simulator_class(method="statevector")
            except Exception as error:
                raise BackendUnavailableError(
                    f"could not create ideal Aer simulator ({_exception_name(error)})"
                ) from None
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
                f"could not compile circuits for backend {identity.kind}:{identity.name} "
                f"({_exception_name(error)})"
            ) from None
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


class NoisyAerAdapter(BaseBackendAdapter):
    """Aer execution with a distinct, fixed transpilation target."""

    def __init__(
        self,
        spec: NoisySimulator,
        *,
        simulator: Any,
        target_backend: Any,
        identity: BackendIdentity,
    ) -> None:
        if not isinstance(spec, NoisySimulator):
            raise BackendCompatibilityError(
                "NoisyAerAdapter requires a NoisySimulator specification"
            )
        if simulator is None or not callable(getattr(simulator, "run", None)):
            raise BackendUnavailableError("noisy Aer simulator does not expose run")
        if target_backend is None:
            raise BackendCompatibilityError("noisy simulator transpilation target is required")
        if not isinstance(identity, BackendIdentity) or identity.kind != "noisy":
            raise BackendCompatibilityError("noisy simulator identity is invalid")
        self._spec = spec
        self._simulator = simulator
        self._target_backend = target_backend
        self._identity = identity

    @property
    def simulator(self) -> Any:
        return self._simulator

    @property
    def target_backend(self) -> Any:
        return self._target_backend

    def resolve(self) -> BackendIdentity:
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(local=True, supports_resume=False)

    def availability(self) -> Availability:
        if not callable(getattr(self._simulator, "run", None)):
            return Availability(False, "noisy Aer simulator does not expose run")
        return Availability(True)

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = {
            key: value
            for key, value in config.to_safe_dict().items()
            if value is not None
        }
        try:
            compiled = transpile(list(batch), backend=self._target_backend, **options)
        except Exception as error:
            raise BackendCompatibilityError(
                f"could not compile circuits for backend noisy:{self._identity.name} "
                f"({_exception_name(error)})"
            ) from None
        compiled_batch = tuple(compiled) if isinstance(compiled, (list, tuple)) else (compiled,)
        return CompiledBatch(compiled_batch, self._identity, {"transpilation": options})

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        return self._submit_to_backend(self._simulator, circuits, shots, options)

    def restore_job(
        self,
        job_id: str,
        *,
        circuit_count: int | None = None,
        shots: int | None = None,
    ) -> SubmittedJob:
        raise BackendCompatibilityError("local noisy Aer jobs cannot be restored")


def build_noisy_adapter(
    spec: NoisySimulator,
    iqm_backend: Any = None,
    fake_backend_factory: Any = None,
    simulator_factory: Any = None,
) -> NoisyAerAdapter:
    """Build a noisy adapter without retaining raw provider profile objects."""

    if not isinstance(spec, NoisySimulator):
        raise BackendCompatibilityError(
            "build_noisy_adapter requires a NoisySimulator specification"
        )
    if spec.identity is not None and not _safe_explicit_identity(spec.identity):
        raise BackendCompatibilityError("noisy simulator identity must be a safe label")
    if fake_backend_factory is not None and not callable(fake_backend_factory):
        raise BackendCompatibilityError("IQM fake-backend factory must be callable")
    if simulator_factory is not None and not callable(simulator_factory):
        raise BackendCompatibilityError("noisy simulator factory must be callable")

    if spec.source is not None:
        source = spec.source
        if isinstance(source, IQMHardware):
            from .iqm import IQMAdapter

            source_adapter = IQMAdapter(source, backend=iqm_backend)
            target_backend = source_adapter.backend
            source_identity = f"iqm:{source.device}"
            emulates = source_identity
            calibration_set_id = _calibration_set_id(target_backend)
            if fake_backend_factory is not None:
                profile_provenance = "iqm-fake-backend"
                profile_backend = _convert_profile(
                    fake_backend_factory,
                    target_backend,
                    stage="IQM noise profile",
                )
            else:
                profile_provenance = "aer-from-iqm-backend"
                profile_backend = target_backend
        else:
            target_backend = source
            source_name = _backend_name(source)
            source_identity = f"custom:{source_name}"
            emulates = None
            calibration_set_id = _calibration_set_id(source)
            profile_backend = source
            profile_provenance = "aer-from-backend"
        simulator = _simulator_from_backend(profile_backend, simulator_factory)
    else:
        target_backend = spec.target_backend
        source_name = _backend_name(target_backend)
        source_identity = f"custom:{source_name}"
        emulates = None
        calibration_set_id = _calibration_set_id(target_backend)
        profile_provenance = "explicit-noise-model"
        simulator = _simulator_from_noise_model(spec.noise_model, simulator_factory)

    if simulator is None or not callable(getattr(simulator, "run", None)):
        raise BackendUnavailableError("noisy Aer simulator does not expose run")
    name = spec.identity or f"noisy-{_backend_name(target_backend)}"
    identity = BackendIdentity(
        kind="noisy",
        name=name,
        provider="qiskit-aer",
        version=_backend_version(simulator),
        emulates=emulates,
        metadata={
            "calibration_set_id": calibration_set_id,
            "source_identity": source_identity,
            "profile_provenance": profile_provenance,
        },
    )
    return NoisyAerAdapter(
        spec,
        simulator=simulator,
        target_backend=target_backend,
        identity=identity,
    )


def _load_aer_simulator_type() -> Any:
    from qiskit_aer import AerSimulator

    return AerSimulator


def _simulator_from_backend(profile_backend: Any, factory: Any) -> Any:
    try:
        if factory is not None:
            return factory(profile_backend)
        simulator_type = _load_aer_simulator_type()
        return simulator_type.from_backend(profile_backend)
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalDependencyError(
            "NoisySimulator requires qiskit-aer; install it with `pip install qiskit-aer` "
            f"({_exception_name(error)})"
        ) from None
    except Exception as error:
        raise BackendUnavailableError(
            f"could not convert backend noise profile ({_exception_name(error)})"
        ) from None


def _simulator_from_noise_model(noise_model: Any, factory: Any) -> Any:
    try:
        if factory is not None:
            return factory(noise_model=noise_model)
        simulator_type = _load_aer_simulator_type()
        return simulator_type(noise_model=noise_model)
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalDependencyError(
            "NoisySimulator requires qiskit-aer; install it with `pip install qiskit-aer` "
            f"({_exception_name(error)})"
        ) from None
    except Exception as error:
        raise BackendUnavailableError(
            f"could not create simulator from noise profile ({_exception_name(error)})"
        ) from None


def _convert_profile(factory: Any, source: Any, *, stage: str) -> Any:
    try:
        converted = factory(source)
    except Exception as error:
        raise BackendUnavailableError(
            f"could not convert {stage} ({_exception_name(error)})"
        ) from None
    if converted is None:
        raise BackendUnavailableError(f"could not convert {stage}")
    return converted


def _backend_name(backend: Any) -> str:
    try:
        value = getattr(backend, "name", None)
        value = value() if callable(value) else value
    except Exception:
        value = None
    if not isinstance(value, str) or not value or any(
        marker in value.lower()
        for marker in ("token=", "api_key=", "password=", "secret=")
    ) or _credentialed_url(value):
        return "source-backend"
    return value


def _backend_version(backend: Any) -> str | None:
    try:
        value = getattr(backend, "backend_version", None)
        value = value() if callable(value) else value
    except Exception:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _calibration_set_id(backend: Any) -> str:
    for attribute in ("calibration_set_id", "_calibration_set_id"):
        try:
            value = getattr(backend, attribute, None)
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            rendered = str(value)
            if not _credentialed_url(rendered) and not any(
                marker in rendered.lower()
                for marker in ("token=", "api_key=", "password=", "secret=")
            ):
                return rendered
    return ""


def _credentialed_url(value: str) -> bool:
    if "://" not in value:
        return False
    return "@" in value.split("://", 1)[1].split("/", 1)[0]


def _safe_explicit_identity(value: str) -> bool:
    if not _EXPLICIT_IDENTITY.fullmatch(value):
        return False
    if any(marker in value.lower() for marker in _CREDENTIAL_MARKERS):
        return False
    return not _credentialed_url(value)


__all__ = ["AerAdapter", "NoisyAerAdapter", "build_noisy_adapter"]
