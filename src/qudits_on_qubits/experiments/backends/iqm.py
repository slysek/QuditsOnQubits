"""IQM hardware adapter for the unified experiment runner."""

from __future__ import annotations

from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit

from ...benchmarks.direct_basis.iqm_backend import (
    backend_metadata,
    load_iqm_backend,
)
from ..errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    JobResultError,
    OptionalDependencyError,
)
from ..models import IQMHardware, IQMQubitSelectorConfig, TranspilationConfig
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    SubmittedJob,
    _exception_name,
    _extract_job_id,
    _positive_integer,
    _safe_identifier,
    _validated_circuit_tuple,
)


_SECRET_KEYS = ("token", "password", "secret", "api_key", "dashboard_api_key")


def _default_backend_loader(device: str, use_metrics: bool, env_path: Any) -> Any:
    return load_iqm_backend(device, use_metrics=use_metrics, env_path=env_path)


def _default_layout_selector(
    backend: Any,
    circuit: Any,
    config: IQMQubitSelectorConfig,
) -> tuple[Any, Any, str]:
    from importlib.metadata import version

    from iqm.qubit_selector.qubit_selector import (
        CostEvaluator,
        CostFunction,
        ReadoutMode,
    )

    cost_functions = {
        "cz": CostFunction.GATE_COST_CZ,
        "clifford": CostFunction.GATE_COST_CLIFFORD,
    }
    readout_modes = {
        "none": ReadoutMode.NONE,
        "fidelity": ReadoutMode.FIDELITY,
        "qndness": ReadoutMode.QNDNESS,
    }
    evaluator = CostEvaluator(
        backend=backend,
        quantum_circuit=circuit,
        cost_function=cost_functions[config.cost_function],
        readoutmode=readout_modes[config.readout_mode],
        remove_qubits=(list(config.remove_qubits) if config.remove_qubits else None),
        num_trials=config.num_trials,
    )
    layouts, costs = evaluator.get_top_layouts(num_layouts=config.top_k)
    return layouts, costs, version("iqm-qubit-selector")


def _default_transpiler(circuit: Any, backend: Any, **options: Any) -> Any:
    from iqm.qiskit_iqm import transpile_to_IQM

    return transpile_to_IQM(circuit, backend=backend, **options)


def _effective_options(config: TranspilationConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.to_safe_dict().items()
        if value is not None
    }


def _default_backend_metadata(backend: Any, spec: IQMHardware) -> Mapping[str, Any]:
    return backend_metadata(
        backend,
        iqm_backend_name=spec.device,
        iqm_use_metrics=spec.use_metrics,
        optimization_level=3,
        layout_method=None,
        routing_method=None,
        scheduling_method=None,
    )


class IQMAdapter(BaseBackendAdapter):
    """One lazily resolved IQM backend used for compilation and execution."""

    def __init__(
        self,
        spec: IQMHardware,
        backend: Any = None,
        backend_loader: Any = None,
        transpiler: Any = None,
        loader: Any = None,
        layout_selector: Any = None,
    ) -> None:
        if not isinstance(spec, IQMHardware):
            raise BackendCompatibilityError("IQMAdapter requires an IQMHardware specification")
        if not _safe_metadata_text(spec.device):
            raise BackendCompatibilityError("IQM device must not contain credential material")
        if backend_loader is not None and loader is not None:
            raise BackendCompatibilityError("provide only one IQM backend loader")
        selected_loader = loader if loader is not None else backend_loader
        if selected_loader is not None and not callable(selected_loader):
            raise BackendCompatibilityError("IQM backend loader must be callable")
        if transpiler is not None and not callable(transpiler):
            raise BackendCompatibilityError("IQM transpiler must be callable")
        if layout_selector is not None and not callable(layout_selector):
            raise BackendCompatibilityError("IQM layout selector must be callable")
        self._spec = spec
        self._backend = backend
        self._backend_loader = selected_loader or _default_backend_loader
        self._transpiler = transpiler or _default_transpiler
        self._layout_selector = (
            layout_selector
            if layout_selector is not None
            else _default_layout_selector
        )
        self._identity: BackendIdentity | None = None

    @property
    def backend(self) -> Any:
        return self._backend_instance()

    def _backend_instance(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._backend_loader(
                self._spec.device,
                use_metrics=self._spec.use_metrics,
                env_path=self._spec.env_path,
            )
        except MemoryError:
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQMHardware requires the IQM Qiskit adapter; install project IQM dependencies "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            if "Missing IQM Qiskit adapter" in str(error):
                raise OptionalDependencyError(
                    "IQMHardware requires the IQM Qiskit adapter; install project IQM dependencies "
                    f"({_exception_name(error)})"
                ) from None
            raise BackendUnavailableError(
                f"could not load IQM backend iqm:{self._spec.device} ({_exception_name(error)})"
            ) from None
        if backend is None:
            raise BackendUnavailableError(f"IQM backend iqm:{self._spec.device} is unavailable")
        self._backend = backend
        return backend

    def resolve(self) -> BackendIdentity:
        backend = self._backend_instance()
        if self._identity is None:
            raw_metadata: Mapping[str, Any]
            try:
                raw_metadata = _default_backend_metadata(backend, self._spec)
            except Exception:
                raw_metadata = {}
            calibration_set_id = _calibration_set_id(backend)
            metadata = {
                **_normalise_metadata(raw_metadata),
                "calibration_set_id": calibration_set_id,
                "target": f"iqm:{self._spec.device}",
                "provider": "iqm",
            }
            self._identity = BackendIdentity(
                kind="iqm",
                name=self._spec.device,
                provider="iqm",
                version=_backend_version(backend),
                metadata=metadata,
            )
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        backend = self._backend_instance()
        return BackendCapabilities(
            local=False,
            supports_resume=callable(getattr(backend, "retrieve_job", None)),
            max_circuits=_max_circuits(backend),
        )

    def availability(self) -> Availability:
        try:
            backend = self._backend_instance()
        except OptionalDependencyError as error:
            return Availability(False, f"IQM dependency is unavailable ({_exception_name(error)})")
        except BackendUnavailableError as error:
            return Availability(False, f"IQM backend is unavailable ({_exception_name(error)})")
        if not callable(getattr(backend, "run", None)):
            return Availability(False, "IQM backend does not expose run")
        return _backend_availability(backend, "IQM")

    def preflight(self, circuits: Sequence[Any], shots: int) -> None:
        batch = _validated_circuit_tuple(circuits)
        super().preflight(batch, shots)
        capacity = _num_qubits(self._backend_instance())
        if capacity is None:
            return
        for circuit in batch:
            required = _num_qubits(circuit)
            if required is not None and required > capacity:
                raise BackendCompatibilityError(
                    f"circuit requires {required} qubits but IQM backend provides {capacity} qubits"
                )

    def _transpile_batch(
        self,
        circuits: tuple[Any, ...],
        options: Mapping[str, Any],
        *,
        operation: str,
    ) -> tuple[Any, ...]:
        backend = self._backend_instance()
        try:
            return tuple(
                self._transpiler(circuit, backend, **dict(options))
                for circuit in circuits
            )
        except MemoryError:
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQM transpilation requires the IQM Qiskit adapter "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            identity = self.resolve()
            raise BackendCompatibilityError(
                f"could not {operation} for backend {identity.kind}:{identity.name} "
                f"({_exception_name(error)})"
            ) from None

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = _effective_options(config)
        compiled = self._transpile_batch(batch, options, operation="compile circuits")
        return CompiledBatch(
            compiled,
            self.resolve(),
            {"transpilation": options},
        )

    def compile_restricted(
        self,
        circuits: Sequence[Any],
        config: TranspilationConfig,
        physical_qubits: Sequence[int],
    ) -> CompiledBatch:
        """Compile on an allowed IQM routing subgraph, then restore provider indices."""

        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError(
                "compile_restricted requires TranspilationConfig"
            )
        if config.initial_layout is not None:
            raise BackendCompatibilityError(
                "compile_restricted does not accept initial_layout"
            )
        if not isinstance(physical_qubits, Sequence) or isinstance(
            physical_qubits, (str, bytes)
        ):
            raise BackendCompatibilityError(
                "restricted physical qubits must be a sequence of integers"
            )
        restricted_qubits = tuple(physical_qubits)
        if any(type(index) is not int for index in restricted_qubits):
            raise BackendCompatibilityError(
                "restricted physical qubits must contain integers"
            )
        if any(index < 0 for index in restricted_qubits):
            raise BackendCompatibilityError(
                "restricted physical qubits must be non-negative"
            )
        if len(set(restricted_qubits)) != len(restricted_qubits):
            raise BackendCompatibilityError(
                "restricted physical qubits must be distinct"
            )

        backend = self._backend_instance()
        try:
            capacity = _num_qubits(backend)
            logical_widths = tuple(_num_qubits(circuit) for circuit in batch)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise BackendCompatibilityError(
                "restricted IQM compilation requires qubit widths"
            ) from None
        if capacity is None or any(width is None for width in logical_widths):
            raise BackendCompatibilityError(
                "restricted IQM compilation requires qubit widths"
            )
        if len(restricted_qubits) < max(logical_widths):
            raise BackendCompatibilityError(
                "restricted physical subgraph is smaller than the logical circuit width"
            )
        if len(restricted_qubits) > capacity or any(
            index >= capacity for index in restricted_qubits
        ):
            raise BackendCompatibilityError(
                "restricted physical qubits exceed backend capacity"
            )

        options = _effective_options(config)
        options["restrict_to_qubits"] = list(restricted_qubits)
        compiled = self._transpile_batch(
            batch,
            options,
            operation="compile restricted circuits",
        )
        try:
            inflated = tuple(
                _inflate_restricted_circuit(
                    circuit,
                    restricted_qubits=restricted_qubits,
                    backend_width=capacity,
                )
                for circuit in compiled
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as error:
            identity = self.resolve()
            raise BackendCompatibilityError(
                "could not compile restricted circuits for backend "
                f"{identity.kind}:{identity.name} ({_exception_name(error)})"
            ) from None
        return CompiledBatch(
            inflated,
            self.resolve(),
            {
                "transpilation": options,
                "restricted_physical_qubits": restricted_qubits,
            },
        )

    def compile_physical(
        self, circuits: Sequence[Any], config: TranspilationConfig
    ) -> CompiledBatch:
        """Compile single-qubit calibration circuits without changing targets."""

        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError(
                "compile_physical requires TranspilationConfig"
            )
        widths = {_num_qubits(circuit) for circuit in batch}
        if None in widths or len(widths) != 1:
            raise BackendCompatibilityError(
                "physical calibration circuits must have one common qubit width"
            )
        width = next(iter(widths))
        assert width is not None
        options = _effective_options(config)
        options["initial_layout"] = list(range(width))
        compiled = self._transpile_batch(
            batch,
            options,
            operation="compile physical calibration circuits",
        )
        return CompiledBatch(
            compiled,
            self.resolve(),
            {"transpilation": options, "physical_layout": True},
        )

    def suggest_layouts(
        self,
        circuit: QuantumCircuit,
        config: IQMQubitSelectorConfig,
    ) -> Mapping[str, Any]:
        if not isinstance(circuit, QuantumCircuit):
            raise BackendCompatibilityError(
                "IQM qubit selector requires a QuantumCircuit"
            )
        if not isinstance(config, IQMQubitSelectorConfig):
            raise BackendCompatibilityError(
                "IQM qubit selector requires IQMQubitSelectorConfig"
            )
        backend = self._backend_instance()
        try:
            capacity = _num_qubits(backend)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise BackendCompatibilityError(
                "IQM qubit selector requires backend qubit capacity"
            ) from None
        if capacity is None:
            raise BackendCompatibilityError(
                "IQM qubit selector requires backend qubit capacity"
            )
        if any(index >= capacity for index in config.remove_qubits):
            raise BackendCompatibilityError(
                "IQM qubit selector remove_qubits exceed backend capacity"
            )
        try:
            raw_result = self._layout_selector(backend, circuit, config)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQM automatic layout selection requires iqm-qubit-selector "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            raise BackendCompatibilityError(
                "IQM qubit selector failed for backend "
                f"iqm:{self._spec.device} ({_exception_name(error)})"
            ) from None

        try:
            if not isinstance(raw_result, (tuple, list)) or len(raw_result) != 3:
                raise ValueError
            raw_layouts, raw_costs, version = raw_result
            if (
                not isinstance(raw_layouts, Sequence)
                or isinstance(raw_layouts, (str, bytes))
                or not isinstance(raw_costs, Sequence)
                or isinstance(raw_costs, (str, bytes))
                or not raw_layouts
                or len(raw_layouts) != len(raw_costs)
                or len(raw_layouts) > config.top_k
                or not isinstance(version, str)
                or not version
                or not _safe_metadata_text(version)
            ):
                raise ValueError
            if any(
                not isinstance(layout, Sequence)
                or isinstance(layout, (str, bytes))
                for layout in raw_layouts
            ):
                raise ValueError
            raw_layout_tuples = tuple(tuple(layout) for layout in raw_layouts)
            if (
                any(
                    not layout
                    or len(layout) < circuit.num_qubits
                    or len(layout) > capacity
                    or any(
                        type(index) is not int
                        or index < 0
                        or index >= capacity
                        or index in config.remove_qubits
                        for index in layout
                    )
                    or len(set(layout)) != len(layout)
                    for layout in raw_layout_tuples
                )
            ):
                raise ValueError
            costs = tuple(float(cost) for cost in raw_costs)
            if (
                any(
                    isinstance(cost, bool)
                    or not isinstance(cost, Real)
                    or not math.isfinite(float(cost))
                    or float(cost) < 0
                    for cost in raw_costs
                )
                or any(left > right for left, right in zip(costs, costs[1:]))
            ):
                raise ValueError
            layouts_list: list[tuple[int, ...]] = []
            costs_list: list[float] = []
            seen_layouts: set[tuple[int, ...]] = set()
            for raw_layout, cost in zip(raw_layout_tuples, costs):
                layout = tuple(sorted(raw_layout))
                if layout in seen_layouts:
                    continue
                seen_layouts.add(layout)
                layouts_list.append(layout)
                costs_list.append(cost)
            layouts = tuple(layouts_list)
            costs = tuple(costs_list)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise BackendCompatibilityError(
                "IQM qubit selector output is invalid"
            ) from None

        return {
            "provider": "iqm-qubit-selector",
            "version": version,
            "configuration": config.to_safe_dict(),
            "layouts": layouts,
            "costs": costs,
        }

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        return self._submit_to_backend(self._backend_instance(), circuits, shots, options)

    def restore_job(
        self,
        job_id: str,
        *,
        circuit_count: int | None = None,
        shots: int | None = None,
    ) -> SubmittedJob:
        job_id = _safe_identifier(job_id, "job_id")
        _positive_integer(circuit_count, "circuit_count", optional=True)
        _positive_integer(shots, "shots", optional=True)
        backend = self._backend_instance()
        retrieve = getattr(backend, "retrieve_job", None)
        if not callable(retrieve):
            raise BackendCompatibilityError("IQM backend does not support resume")
        try:
            handle = retrieve(job_id)
        except Exception as error:
            raise JobResultError(
                f"could not restore IQM job {job_id} ({_exception_name(error)})"
            ) from None
        if handle is None:
            raise JobResultError(f"could not restore IQM job {job_id}")
        try:
            actual_job_id = _extract_job_id(handle, allow_local_fallback=False)
        except Exception as error:
            raise JobResultError(
                f"restored IQM job did not provide the requested job ID ({_exception_name(error)})"
            ) from None
        if actual_job_id != job_id:
            raise JobResultError("restored IQM job ID does not match requested job ID")
        return SubmittedJob(
            job_id,
            handle,
            self.resolve(),
            circuit_count,
            shots,
            {"restored": True},
        )


def _inflate_restricted_circuit(
    circuit: Any,
    *,
    restricted_qubits: tuple[int, ...],
    backend_width: int,
) -> QuantumCircuit:
    if (
        not isinstance(circuit, QuantumCircuit)
        or circuit.num_qubits != len(restricted_qubits)
    ):
        raise ValueError("invalid restricted IQM transpiler output")
    auxiliary = QuantumCircuit(backend_width, circuit.num_clbits)
    return auxiliary.compose(
        circuit,
        qubits=list(restricted_qubits),
        clbits=list(range(circuit.num_clbits)),
    )


def _backend_availability(backend: Any, provider: str) -> Availability:
    status_method = getattr(backend, "status", None)
    if not callable(status_method):
        return Availability(True)
    try:
        status = status_method()
    except Exception as error:
        return Availability(False, f"{provider} backend status is unavailable ({_exception_name(error)})")
    if getattr(status, "operational", None) is False:
        return Availability(False, _safe_status_reason(status, provider))
    return Availability(True)


def _safe_status_reason(status: Any, provider: str) -> str:
    try:
        value = getattr(status, "status_msg", None)
    except Exception:
        value = None
    if (
        isinstance(value, str)
        and value
        and len(value) <= 512
        and all(ord(character) >= 32 for character in value)
        and not any(marker in value.lower() for marker in _SECRET_KEYS)
    ):
        return value
    return f"{provider} backend is not operational"


def _num_qubits(value: Any) -> int | None:
    candidate = getattr(value, "num_qubits", None)
    if callable(candidate):
        try:
            candidate = candidate()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            candidate = None
    if candidate is None:
        target = getattr(value, "target", None)
        candidate = getattr(target, "num_qubits", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    return None


def _max_circuits(backend: Any) -> int | None:
    candidate = getattr(backend, "max_circuits", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    return None


def _calibration_set_id(backend: Any) -> str:
    for name in ("calibration_set_id", "_calibration_set_id"):
        try:
            value = getattr(backend, name, None)
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            rendered = str(value)
            return rendered if _safe_metadata_text(rendered) else ""
    return ""


def _backend_version(backend: Any) -> str | None:
    try:
        value = getattr(backend, "backend_version", None)
        value = value() if callable(value) else value
    except Exception:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _normalise_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or any(marker in key.lower() for marker in _SECRET_KEYS):
            continue
        safe_value = _normalise_value(value)
        if safe_value is not None or value is None:
            normalised[key] = safe_value
    return normalised


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _normalise_value(value.value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value if _safe_metadata_text(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return _normalise_metadata(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalise_value(item) for item in value]
    rendered = str(value)
    return rendered if _safe_metadata_text(rendered) else None


def _safe_metadata_text(value: str) -> bool:
    lower = value.lower()
    if any(marker in lower for marker in _SECRET_KEYS):
        return False
    if "://" in value:
        authority = value.split("://", 1)[1].split("/", 1)[0]
        if "@" in authority:
            return False
    return len(value) <= 4096 and all(ord(character) >= 32 for character in value)


__all__ = ["IQMAdapter"]
