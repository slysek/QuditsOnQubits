from __future__ import annotations

from importlib import import_module
from typing import Any


_CALLABLE_EXPORTS = {
    "create_ame_circuit": "qudits_on_qubits.core.create_ame_circuit",
    "draw_graph": "qudits_on_qubits.core.draw_graph",
    "generate_b_ame": "qudits_on_qubits.core.generate_b_ame",
    "get_encoding": "qudits_on_qubits.reference_experiments",
    "get_reference_experiment": "qudits_on_qubits.reference_experiments",
    "list_reference_experiments": "qudits_on_qubits.reference_experiments",
    "prepare_op_to_ibm": "qudits_on_qubits.core.prepare_op_to_ibm",
}
_CLASS_EXPORTS = {
    "EncodingSpec": ("qudits_on_qubits.reference_experiments", "EncodingSpec"),
    "QuditsOnQubits": ("qudits_on_qubits.core.quditsonqubits", "QuditsOnQubits"),
    "ReferenceExperimentSpec": (
        "qudits_on_qubits.reference_experiments",
        "ReferenceExperimentSpec",
    ),
}
_EXPERIMENT_EXPORTS = {
    "AerIdeal",
    "BellEstimate",
    "BenchmarkBasis",
    "BootstrapBellResults",
    "BootstrapConfig",
    "BootstrapDiagnostics",
    "BootstrapInputs",
    "ComplexComponents",
    "ComplexConfidenceInterval",
    "ConfidenceInterval",
    "CustomBackend",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "IQMHardware",
    "MitigationConfig",
    "NoisySimulator",
    "PathBasis",
    "PiastQHardware",
    "ReadoutBootstrapStrategy",
    "RetryConfig",
    "TranspilationConfig",
    "ZNEBootstrapStrategy",
    "bootstrap_bell_results",
    "resume_experiment",
    "run_experiment",
    "run_experiments",
}
_VERTICAL_SLICE_EXPORTS = {
    "ArtifactIntegrityError",
    "ArtifactRef",
    "BackendSnapshot",
    "BellPostprocessorSpec",
    "BellReferenceCircuitSpec",
    "CircuitSpec",
    "EncodingValidationError",
    "ExecutionMode",
    "ExecutionSpec",
    "IsometricQuditEncoding",
    "LogicalOutcome",
    "ManifestValidationError",
    "PostprocessorSpec",
    "PreparedExperiment",
    "QuditEncoding",
    "QuditExperimentResult",
    "QuditExperimentSpec",
    "RunManifest",
    "SoftwareProvenance",
    "SpecValidationError",
    "canonical_qutrit_encoding",
    "load_run_manifest",
    "run_vertical_slice",
}


def _lazy_callable(name: str, module_name: str):
    def wrapper(*args, **kwargs):
        target = getattr(import_module(module_name), name)
        return target(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = __name__
    return wrapper


def __getattr__(name: str) -> Any:
    if name in _CALLABLE_EXPORTS:
        value = _lazy_callable(name, _CALLABLE_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _CLASS_EXPORTS:
        module_name, attr_name = _CLASS_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    if name in _EXPERIMENT_EXPORTS:
        value = getattr(import_module("qudits_on_qubits.experiments"), name)
        globals()[name] = value
        return value
    if name in _VERTICAL_SLICE_EXPORTS:
        value = getattr(import_module("qudits_on_qubits.vertical_slice"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EncodingSpec",
    "QuditsOnQubits",
    "ReferenceExperimentSpec",
    "AerIdeal",
    "BellEstimate",
    "BenchmarkBasis",
    "BootstrapBellResults",
    "BootstrapConfig",
    "BootstrapDiagnostics",
    "BootstrapInputs",
    "ComplexComponents",
    "ComplexConfidenceInterval",
    "ConfidenceInterval",
    "CustomBackend",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "IQMHardware",
    "MitigationConfig",
    "NoisySimulator",
    "PathBasis",
    "PiastQHardware",
    "ReadoutBootstrapStrategy",
    "RetryConfig",
    "TranspilationConfig",
    "ZNEBootstrapStrategy",
    "bootstrap_bell_results",
    "create_ame_circuit",
    "draw_graph",
    "generate_b_ame",
    "get_encoding",
    "get_reference_experiment",
    "list_reference_experiments",
    "prepare_op_to_ibm",
    "resume_experiment",
    "run_experiment",
    "run_experiments",
    "ArtifactIntegrityError",
    "ArtifactRef",
    "BackendSnapshot",
    "BellPostprocessorSpec",
    "BellReferenceCircuitSpec",
    "CircuitSpec",
    "EncodingValidationError",
    "ExecutionMode",
    "ExecutionSpec",
    "IsometricQuditEncoding",
    "LogicalOutcome",
    "ManifestValidationError",
    "PostprocessorSpec",
    "PreparedExperiment",
    "QuditEncoding",
    "QuditExperimentResult",
    "QuditExperimentSpec",
    "RunManifest",
    "SoftwareProvenance",
    "SpecValidationError",
    "canonical_qutrit_encoding",
    "load_run_manifest",
    "run_vertical_slice",
]
