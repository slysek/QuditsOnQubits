"""Unified encoding search, backend compilation and Pareto analysis.

Public imports are lazy; importing the package never loads a backend or starts
synthesis. Legacy direct_basis and theta_continuation imports remain supported.
"""
from importlib import import_module

_EXPORTS = {
    "run_benchmark": ("runner", "run_benchmark"),
    "load_benchmark": ("artifacts", "load_benchmark"),
    "BenchmarkResult": ("artifacts", "BenchmarkResult"),
    "BenchmarkConfig": ("config", "BenchmarkConfig"),
    "EncodingCandidate": ("models", "EncodingCandidate"),
    "LocalSU2": ("families", "LocalSU2"),
    "SchmidtTheta": ("families", "SchmidtTheta"),
    "canonical_candidate": ("families", "canonical_candidate"),
    "BackendTarget": ("targets", "BackendTarget"),
    "LogicalCircuit": ("workloads", "LogicalCircuit"),
    "LogicalOperation": ("workloads", "LogicalOperation"),
    "two_qutrit_graph_circuit": ("workloads", "two_qutrit_graph_circuit"),
    "OptimizedSynthesis": ("synthesis", "OptimizedSynthesis"),
    "ExactSynthesis": ("synthesis", "ExactSynthesis"),
    "ThetaContinuationSynthesis": ("synthesis", "ThetaContinuationSynthesis"),
    "SavedGateSynthesis": ("synthesis", "SavedGateSynthesis"),
    "saved_candidate": ("synthesis", "saved_candidate"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
