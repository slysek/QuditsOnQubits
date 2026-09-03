"""Direct qutrit-basis encoding benchmarks.

This package keeps the direct-basis method separate from the legacy
"append W as a physical basis-change gate" pipeline.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "benchmark_direct_basis": ("benchmark", "benchmark_direct_basis"),
    "benchmark_direct_basis_candidates": ("benchmark", "benchmark_direct_basis_candidates"),
    "DirectBasisCandidate": ("candidates", "DirectBasisCandidate"),
    "generate_all_qutrit_u3_candidates": ("candidates", "generate_all_qutrit_u3_candidates"),
    "generate_extended_legacy_candidates": ("candidates", "generate_extended_legacy_candidates"),
    "generate_sanity_basis_candidates": ("candidates", "generate_sanity_basis_candidates"),
    "generate_v2_stage1_direct_candidates": ("candidates", "generate_v2_stage1_direct_candidates"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "DirectBasisCandidate",
    "benchmark_direct_basis",
    "benchmark_direct_basis_candidates",
    "generate_all_qutrit_u3_candidates",
    "generate_extended_legacy_candidates",
    "generate_sanity_basis_candidates",
    "generate_v2_stage1_direct_candidates",
]
