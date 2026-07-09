from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qudits_on_qubits.benchmarks.direct_basis.piast_backend import (
    AQT_SCHEDULING_METHOD,
    AQT_TRANSLATION_METHOD,
)


@dataclass(frozen=True)
class PiastTranspilerStrategy:
    name: str
    description: str
    kind: str
    translation_method: str
    scheduling_method: str


@dataclass(frozen=True)
class PiastTranspilerStrategyResult:
    strategy_name: str
    seed_transpiler: int
    success: bool
    circuit: Any
    compile_time_seconds: float
    error_type: str = ""
    error_message: str = ""


_BUILTIN_PIAST_TRANSPILER_STRATEGIES = {
    "transpile_aqt_plugin": PiastTranspilerStrategy(
        name="transpile_aqt_plugin",
        description="Qiskit transpile using the PCSS AQT translation and scheduling plugins.",
        kind="transpile",
        translation_method=AQT_TRANSLATION_METHOD,
        scheduling_method=AQT_SCHEDULING_METHOD,
    ),
    "preset_aqt_plugin": PiastTranspilerStrategy(
        name="preset_aqt_plugin",
        description="Qiskit preset pass manager using the PCSS AQT translation and scheduling plugins.",
        kind="preset",
        translation_method=AQT_TRANSLATION_METHOD,
        scheduling_method=AQT_SCHEDULING_METHOD,
    ),
}

BUILTIN_PIAST_TRANSPILER_STRATEGIES: Mapping[str, PiastTranspilerStrategy] = (
    MappingProxyType(_BUILTIN_PIAST_TRANSPILER_STRATEGIES)
)


def piast_transpiler_strategy_names() -> tuple[str, ...]:
    return tuple(BUILTIN_PIAST_TRANSPILER_STRATEGIES)


def get_piast_transpiler_strategy(name: str) -> PiastTranspilerStrategy:
    try:
        return BUILTIN_PIAST_TRANSPILER_STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(piast_transpiler_strategy_names())
        raise ValueError(
            f"Unknown Piast transpiler strategy {name!r}. "
            f"Available strategies: {available}"
        ) from exc


def _should_capture_transpiler_failure(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError)):
        return False
    if isinstance(exc, Exception):
        return True

    exc_type = type(exc)
    return (
        exc_type.__name__ == "PanicException"
        and "pyo3_runtime" in str(getattr(exc_type, "__module__", ""))
    )


def _failure_result(
    *,
    strategy_name: str,
    seed_transpiler: Any,
    started: float,
    exc: BaseException,
) -> PiastTranspilerStrategyResult:
    return PiastTranspilerStrategyResult(
        strategy_name=strategy_name,
        seed_transpiler=seed_transpiler,
        success=False,
        circuit=None,
        compile_time_seconds=time.perf_counter() - started,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def run_piast_transpiler_strategy(
    strategy_name: str,
    circuit: Any,
    *,
    backend: Any,
    seed_transpiler: int,
    optimization_level: int = 3,
) -> PiastTranspilerStrategyResult:
    started = time.perf_counter()
    seed_value: Any = seed_transpiler

    try:
        seed_value = int(seed_transpiler)
        strategy = get_piast_transpiler_strategy(strategy_name)
        optimization_value = int(optimization_level)

        if strategy.kind == "transpile":
            transpiled_circuit = transpile(
                circuit,
                backend=backend,
                optimization_level=optimization_value,
                seed_transpiler=seed_value,
                translation_method=strategy.translation_method,
                scheduling_method=strategy.scheduling_method,
            )
        elif strategy.kind == "preset":
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=optimization_value,
                seed_transpiler=seed_value,
                translation_method=strategy.translation_method,
                scheduling_method=strategy.scheduling_method,
            )
            transpiled_circuit = pass_manager.run(circuit)
        else:
            raise ValueError(
                f"Unsupported Piast transpiler strategy kind: {strategy.kind}"
            )

        return PiastTranspilerStrategyResult(
            strategy_name=strategy.name,
            seed_transpiler=seed_value,
            success=True,
            circuit=transpiled_circuit,
            compile_time_seconds=time.perf_counter() - started,
        )
    except BaseException as exc:
        if not _should_capture_transpiler_failure(exc):
            raise
        return _failure_result(
            strategy_name=strategy_name,
            seed_transpiler=seed_value,
            started=started,
            exc=exc,
        )
