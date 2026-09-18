"""Explicit, frozen compilation targets for the encoding benchmark.

Creating a hardware target may read current backend metadata. Compilation only
uses that captured data and never retains a provider client or submits jobs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from numbers import Integral
from typing import Any, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ParameterExpression
from qiskit.transpiler import CouplingMap, Target, generate_preset_pass_manager


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(value)


def _versions() -> dict[str, str | None]:
    values = {}
    for package in ("qiskit", "iqm-client", "qiskit-ibm-runtime"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def _parameter(value: Any) -> Any:
    if isinstance(value, ParameterExpression):
        return {"expression": str(value), "parameters": sorted(str(p) for p in value.parameters)}
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if hasattr(value, "tolist"):
        return _parameter(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_parameter(item) for item in value]
    raise ValueError(f"Cannot snapshot instruction parameter type {type(value).__name__}.")


def _target_description(target: Target) -> dict[str, Any]:
    instructions = []
    for name in sorted(target.operation_names):
        operation = target.operation_from_name(name)
        # Variable-width control flow is not used by these preparation circuits.
        if isinstance(operation, type):
            instructions.append({"name": name, "operation_class": f"{operation.__module__}.{operation.__qualname__}",
                                 "variable_width": True})
            continue
        loci = []
        for qargs, properties in sorted(target[name].items(), key=lambda item: str(item[0])):
            loci.append({
                "qubits": None if qargs is None else list(qargs),
                "duration": None if properties is None else properties.duration,
                "error": None if properties is None else properties.error,
            })
        instructions.append({
            "name": name,
            "operation_class": f"{operation.base_class.__module__}.{operation.base_class.__qualname__}",
            "num_qubits": operation.num_qubits,
            "num_clbits": operation.num_clbits,
            "parameters": [_parameter(p) for p in operation.params],
            "loci": loci,
        })
    qubit_properties = None
    if target.qubit_properties is not None:
        qubit_properties = [
            None if item is None else {key: getattr(item, key) for key in ("t1", "t2", "frequency")}
            for item in target.qubit_properties
        ]
    return {
        "num_qubits": target.num_qubits,
        "instructions": instructions,
        "qubit_properties": qubit_properties,
        "dt": target.dt,
        "granularity": target.granularity,
        "min_length": target.min_length,
        "pulse_alignment": target.pulse_alignment,
        "acquire_alignment": target.acquire_alignment,
        "concurrent_measurements": target.concurrent_measurements,
    }


def _target_hash(manifest: dict[str, Any]) -> str:
    content = {key: manifest[key] for key in ("target", "native_target", "iqm_architecture", "iqm_component_to_index") if key in manifest}
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sabre_budget(optimization_level: int, layout_method: str, routing_method: str) -> dict[str, Any]:
    # The pinned Qiskit preset can increase trials using local configuration or
    # QISKIT_SABRE_ALL_THREADS. Capture its effective budget, not only its name.
    from qiskit.transpiler.preset_passmanagers.builtin_plugins import _get_trial_count
    trials = _get_trial_count(5 if optimization_level < 2 else 20)
    return {
        "policy": "qiskit_preset_for_optimization_level",
        "layout_trials": trials if layout_method == "sabre" else None,
        "swap_trials": trials if layout_method == "sabre" or routing_method == "sabre" else None,
        "max_iterations": (1, 2, 2, 4)[optimization_level] if layout_method == "sabre" else None,
    }


def _configuration(num_qubits, initial_layout, optimization_level, layout_method,
                   routing_method, approximation_degree) -> dict[str, Any]:
    optimization_level = _integer(optimization_level, "optimization_level")
    if optimization_level > 3:
        raise ValueError("optimization_level must be between 0 and 3.")
    if approximation_degree != 1.0:
        raise ValueError("State preparation requires approximation_degree=1.0.")
    if not isinstance(layout_method, str) or not layout_method:
        raise ValueError("layout_method must be an explicit nonempty method name.")
    if not isinstance(routing_method, str) or not routing_method:
        raise ValueError("routing_method must be an explicit nonempty method name.")
    layout = None
    if initial_layout is not None:
        layout = [_integer(q, "initial_layout qubit") for q in initial_layout]
        if not layout or len(set(layout)) != len(layout) or max(layout) >= num_qubits:
            raise ValueError("initial_layout must contain distinct in-range physical qubits.")
    return {
        "optimization_level": optimization_level,
        "initial_layout": layout,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "approximation_degree": 1.0,
        "layout_search_budget": _sabre_budget(optimization_level, layout_method, routing_method),
        "allowed_physical_qubits": list(range(num_qubits)),
    }


@dataclass(frozen=True)
class BackendTarget:
    """A captured ISA and one compilation protocol shared by all candidates.

    Construct with :meth:`local`, :meth:`iqm`, or :meth:`ibm`. ``initial_layout``
    maps input circuit qubits to physical indices; output layout stays attached
    to the compiled circuit. No implicit topology or fallback backend is used.
    """

    provider: str
    name: str
    strategy: str
    _target: Any = field(repr=False, compare=False)
    _native_target: Any = field(repr=False, compare=False)
    _manifest: dict[str, Any] = field(repr=False, compare=False)
    _iqm_backend: Any = field(default=None, repr=False, compare=False)

    @property
    def num_qubits(self) -> int:
        return self._target.num_qubits

    @classmethod
    def local(cls, num_qubits: int = 4, coupling_map=None,
              basis_gates: Sequence[str] = ("u", "cz"), initial_layout=None,
              optimization_level: int = 3, *, layout_method: str = "sabre",
              routing_method: str = "sabre", approximation_degree: float = 1.0) -> BackendTarget:
        """Build an explicitly synthetic target; ``None`` means all-to-all."""
        width = _integer(num_qubits, "num_qubits", 1)
        config = _configuration(width, initial_layout, optimization_level, layout_method,
                                routing_method, approximation_degree)
        basis = list(basis_gates)
        if not basis or any(not isinstance(gate, str) or not gate for gate in basis) or len(set(basis)) != len(basis):
            raise ValueError("basis_gates must be distinct nonempty gate names.")
        edges = None
        coupling = None
        if coupling_map is not None:
            raw_edges = coupling_map.get_edges() if isinstance(coupling_map, CouplingMap) else coupling_map
            edges = []
            for edge in raw_edges:
                if len(edge) != 2:
                    raise ValueError("Each coupling_map edge must contain two qubits.")
                a, b = (_integer(q, "coupling_map qubit") for q in edge)
                if a == b or max(a, b) >= width:
                    raise ValueError("coupling_map edges must join distinct in-range qubits.")
                edges.append([a, b])
            edges = sorted(set(map(tuple, edges)))
            coupling = CouplingMap(edges)
            for qubit in range(coupling.size(), width):
                coupling.add_physical_qubit(qubit)
        target = Target.from_configuration(basis, num_qubits=width, coupling_map=coupling)
        manifest = cls._manifest_for("local", "synthetic_" + "_".join(basis), "qiskit_preset", target, config)
        manifest.update({"synthetic": True, "topology": "all_to_all" if edges is None else "explicit",
                         "local_profile": {"basis_gates": basis, "coupling_map": None if edges is None else [list(e) for e in edges]}})
        return cls("local", manifest["name"], "qiskit_preset", target, target, manifest)

    @classmethod
    def iqm(cls, name: str, backend=None, strategy: str = "transpile_to_iqm_exact",
            initial_layout=None, optimization_level: int = 3, *,
            layout_method: str = "sabre", routing_method: str = "sabre",
            approximation_degree: float = 1.0, use_metrics: bool = False,
            env_path=None) -> BackendTarget:
        """Capture a Garnet/Emerald backend and preserve relative phases."""
        from .direct_basis.iqm_transpiler_strategies import (
            get_iqm_transpiler_strategy, validate_state_preserving_iqm_strategies,
        )

        if name not in ("garnet", "emerald"):
            raise ValueError("IQM profile must be 'garnet' or 'emerald'.")
        selected = get_iqm_transpiler_strategy(strategy)
        validate_state_preserving_iqm_strategies([strategy])
        if backend is None:
            from dotenv import find_dotenv
            from .direct_basis.iqm_backend import load_iqm_backend
            if env_path is None:
                env_path = find_dotenv(usecwd=True) or ".env"
            backend = load_iqm_backend(name, use_metrics=use_metrics, env_path=env_path)
        from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
        from .direct_basis.iqm_compilation_backend import IqmCompilationBackend

        if not isinstance(backend, IQMBackendBase):
            raise ValueError("IQM profile requires an actual IQMBackendBase backend.")
        backend_name = str(backend.name)
        if any(profile in backend_name.lower() and profile != name for profile in ("garnet", "emerald")):
            raise ValueError("IQM backend identity does not match the requested profile name.")
        frozen = IqmCompilationBackend(deepcopy(backend.architecture), metrics=deepcopy(backend.metrics), name=backend_name)
        target = frozen.target
        native = frozen.get_real_target()
        config = _configuration(target.num_qubits, initial_layout, optimization_level,
                                layout_method, routing_method, approximation_degree)
        config.update({"remove_final_rzs": False,
                       "scheduling_method": selected.scheduling_method or "move_routing_exact_global_phase",
                       "perform_move_routing": True,
                       "optimize_single_qubits": True})
        manifest = cls._manifest_for("iqm", name, strategy, target, config)
        manifest.update({
            "backend_name": backend_name,
            "native_target": _target_description(native),
            "iqm_architecture": frozen.architecture.model_dump(mode="json"),
            "iqm_component_to_index": dict(target.iqm_component_to_idx),
            "calibration_set_id": str(frozen.architecture.calibration_set_id),
            "metrics_used": frozen.metrics is not None,
            "synthetic": False,
            "topology": "provider_target",
        })
        manifest["target_hash"] = _target_hash(manifest)
        return cls("iqm", name, strategy, target, native, manifest, frozen)

    @classmethod
    def ibm(cls, name: str, backend=None, initial_layout=None, optimization_level: int = 3, *,
            layout_method: str = "sabre", routing_method: str = "sabre",
            approximation_degree: float = 1.0, account_name=None, instance=None) -> BackendTarget:
        """Capture IBM ISA once, then use Qiskit's preset manager offline."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("IBM backend name must be nonempty.")
        if backend is None:
            from .direct_basis.ibm_backend import load_ibm_backend
            backend = load_ibm_backend(name, account_name=account_name, instance=instance)
        if backend.name != name:
            raise ValueError("IBM backend identity does not match the requested name.")
        if not isinstance(backend.target, Target):
            raise ValueError("IBM backend must expose a Qiskit Target.")
        target = deepcopy(backend.target)
        config = _configuration(target.num_qubits, initial_layout, optimization_level,
                                layout_method, routing_method, approximation_degree)
        manifest = cls._manifest_for("ibm", name, "qiskit_preset", target, config)
        manifest.update({"synthetic": False, "topology": "provider_target"})
        return cls("ibm", name, "qiskit_preset", target, target, manifest)

    @staticmethod
    def _manifest_for(provider, name, strategy, target, compilation):
        manifest = {
            "schema_version": 1, "provider": provider, "name": name, "strategy": strategy,
            "target": _target_description(target), "compilation": compilation,
            "calibration_set_id": None,
            "target_retrieved_at": datetime.now(timezone.utc).isoformat(),
            "versions": _versions(),
        }
        manifest["target_hash"] = _target_hash(manifest)
        return manifest

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> BackendTarget:
        """Restore a synthetic local manifest without network access.

        Hardware manifests retain compilation constraints for auditing, but are
        not advertised as replayable serialized IQM or IBM backend objects.
        """
        manifest = deepcopy(snapshot)
        if manifest.get("provider") != "local":
            raise ValueError("Only synthetic local snapshots support offline reconstruction.")
        if manifest.get("target_hash") != _target_hash(manifest):
            raise ValueError("Target snapshot fingerprint/hash mismatch.")
        settings = manifest["compilation"]
        restored = cls.local(
            num_qubits=manifest["target"]["num_qubits"],
            coupling_map=manifest["local_profile"]["coupling_map"],
            basis_gates=manifest["local_profile"]["basis_gates"],
            **{key: settings[key] for key in ("initial_layout", "optimization_level", "layout_method", "routing_method", "approximation_degree")},
        )
        if restored.to_dict()["target_hash"] != manifest["target_hash"]:
            raise ValueError("Snapshot target fingerprint is not reproducible from local profile.")
        return cls(restored.provider, restored.name, restored.strategy, restored._target,
                   restored._native_target, manifest)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible manifest; credentials are excluded."""
        return deepcopy(self._manifest)

    def compile(self, circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
        """Compile to the final native ISA, retaining initial/final layout."""
        seed = _integer(seed, "seed")
        if circuit.num_qubits > self.num_qubits:
            raise ValueError("Circuit width exceeds target num_qubits.")
        config = self._manifest["compilation"]
        if config["layout_search_budget"] != _sabre_budget(
            config["optimization_level"], config["layout_method"], config["routing_method"],
        ):
            raise ValueError("Qiskit routing budget environment changed since target capture.")
        if config["initial_layout"] is not None and len(config["initial_layout"]) != circuit.num_qubits:
            raise ValueError("initial_layout length must equal circuit num_qubits.")
        options = {key: deepcopy(config[key]) for key in ("initial_layout", "optimization_level", "layout_method", "routing_method", "approximation_degree")}
        options["seed_transpiler"] = seed
        if self.provider == "iqm":
            from .direct_basis.iqm_transpiler_strategies import get_iqm_transpiler_strategy, _load_transpile_to_iqm
            selected = get_iqm_transpiler_strategy(self.strategy)
            if selected.kind == "transpile_to_iqm":
                compiled = _load_transpile_to_iqm()(circuit, self._iqm_backend,
                                                  remove_final_rzs=False, **options)
            else:
                from .direct_basis.iqm_backend import build_iqm_pass_manager
                compiled = build_iqm_pass_manager(self._iqm_backend,
                                                 scheduling_method=selected.scheduling_method,
                                                 **options).run(circuit)
            from .direct_basis.circuit_serialization import normalize_circuit_bit_indices
            compiled = normalize_circuit_bit_indices(compiled)
        elif self.provider == "ibm":
            from .direct_basis.ibm_backend import build_ibm_pass_manager
            compiled = build_ibm_pass_manager(None, target=self._target, **options).run(circuit)
        else:
            compiled = generate_preset_pass_manager(target=self._target, **options).run(circuit)
        self.validate_native(compiled)
        return compiled

    def validate_native(self, circuit: QuantumCircuit) -> None:
        """Reject unsupported operations, parameters, or ordered physical loci."""
        if circuit.num_qubits > self._native_target.num_qubits:
            raise ValueError("Compiled circuit width exceeds native target num_qubits.")
        for instruction in circuit.data:
            operation = instruction.operation
            if isinstance(operation, Barrier):
                continue
            qargs = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
            supported = self._native_target.instruction_supported(
                operation_name=operation.name, qargs=qargs, parameters=operation.params,
            )
            if supported:
                native_operation = self._native_target.operation_from_name(operation.name)
                supported = not isinstance(native_operation, type) and operation.base_class is native_operation.base_class
            if not supported:
                raise ValueError(f"Unsupported native instruction {operation.name!r} on qubits {qargs}.")
