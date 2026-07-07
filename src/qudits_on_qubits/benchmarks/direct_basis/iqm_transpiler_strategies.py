from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    EXACT_RZ_SCHEDULING_METHOD,
)


@dataclass(frozen=True)
class IqmTranspilerStrategy:
    name: str
    description: str
    kind: str
    scheduling_method: str | None
    remove_final_rzs: bool


@dataclass(frozen=True)
class IqmTranspilerStrategyResult:
    strategy_name: str
    seed_transpiler: int
    success: bool
    circuit: Any
    compile_time_seconds: float
    error_type: str = ""
    error_message: str = ""


_BUILTIN_IQM_TRANSPILER_STRATEGIES = {
    "preset_default": IqmTranspilerStrategy(
        name="preset_default",
        description="Qiskit preset pass manager with IQM backend defaults.",
        kind="preset",
        scheduling_method=None,
        remove_final_rzs=True,
    ),
    "preset_exact": IqmTranspilerStrategy(
        name="preset_exact",
        description="Qiskit preset pass manager with IQM exact RZ scheduling.",
        kind="preset",
        scheduling_method=EXACT_RZ_SCHEDULING_METHOD,
        remove_final_rzs=False,
    ),
    "transpile_to_iqm_default": IqmTranspilerStrategy(
        name="transpile_to_iqm_default",
        description="IQM transpile_to_IQM helper with final RZ removal.",
        kind="transpile_to_iqm",
        scheduling_method=None,
        remove_final_rzs=True,
    ),
    "transpile_to_iqm_exact": IqmTranspilerStrategy(
        name="transpile_to_iqm_exact",
        description="IQM transpile_to_IQM helper preserving final RZ gates.",
        kind="transpile_to_iqm",
        scheduling_method=None,
        remove_final_rzs=False,
    ),
}

BUILTIN_IQM_TRANSPILER_STRATEGIES: Mapping[str, IqmTranspilerStrategy] = (
    MappingProxyType(_BUILTIN_IQM_TRANSPILER_STRATEGIES)
)


def iqm_transpiler_strategy_names() -> tuple[str, ...]:
    return tuple(BUILTIN_IQM_TRANSPILER_STRATEGIES)


def get_iqm_transpiler_strategy(name: str) -> IqmTranspilerStrategy:
    try:
        return BUILTIN_IQM_TRANSPILER_STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(iqm_transpiler_strategy_names())
        raise ValueError(
            f"Unknown IQM transpiler strategy {name!r}. "
            f"Available strategies: {available}"
        ) from exc


def _load_transpile_to_iqm() -> Any:
    try:
        from iqm.qiskit_iqm import transpile_to_IQM
    except ImportError:
        from iqm.qiskit_iqm.iqm_naive_move_pass import transpile_to_IQM

    return transpile_to_IQM


def run_iqm_transpiler_strategy(
    strategy_name: str,
    circuit: Any,
    *,
    backend: Any,
    seed_transpiler: int,
    optimization_level: int = 3,
) -> IqmTranspilerStrategyResult:
    started = time.perf_counter()
    seed_value: Any = seed_transpiler

    try:
        seed_value = int(seed_transpiler)
        strategy = get_iqm_transpiler_strategy(strategy_name)
        optimization_value = int(optimization_level)

        if strategy.kind == "preset":
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=optimization_value,
                seed_transpiler=seed_value,
                scheduling_method=strategy.scheduling_method,
            )
            transpiled_circuit = pass_manager.run(circuit)
        elif strategy.kind == "transpile_to_iqm":
            transpile_to_iqm = _load_transpile_to_iqm()
            transpiled_circuit = transpile_to_iqm(
                circuit,
                backend,
                optimization_level=optimization_value,
                seed_transpiler=seed_value,
                remove_final_rzs=bool(strategy.remove_final_rzs),
            )
        else:
            raise ValueError(f"Unsupported IQM transpiler strategy kind: {strategy.kind}")

        return IqmTranspilerStrategyResult(
            strategy_name=strategy.name,
            seed_transpiler=seed_value,
            success=True,
            circuit=transpiled_circuit,
            compile_time_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return IqmTranspilerStrategyResult(
            strategy_name=strategy_name,
            seed_transpiler=seed_value,
            success=False,
            circuit=None,
            compile_time_seconds=time.perf_counter() - started,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
