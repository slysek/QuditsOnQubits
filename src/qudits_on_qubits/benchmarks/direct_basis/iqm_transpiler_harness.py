from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    backend_metadata,
    repo_path,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    IqmTranspilerStrategyResult,
    get_iqm_transpiler_strategy,
    iqm_transpiler_strategy_names,
    run_iqm_transpiler_strategy,
)
from qudits_on_qubits.core.benchmark_encoding_bases import TWO_Q_GATES


@dataclass(frozen=True)
class IqmTranspilerHarnessConfig:
    state_name: str
    n_qutrits: int | None
    backend: Any
    iqm_backend_name: str
    iqm_use_metrics: bool
    candidates: Iterable[DirectBasisCandidate]
    strategy_names: tuple[str, ...] = ()
    n_transpile_runs: int = 1
    optimization_level: int = 3
    max_depth_warning: int = 100
    max_cz_warning: int = 50


_METRIC_KEYS = (
    "num_qubits",
    "depth",
    "size",
    "cz_count",
    "r_count",
    "one_qubit_gate_count",
    "two_qubit_gate_count",
    "count_ops_json",
)


def default_iqm_transpiler_harness_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_iqm_transpiler_harness_output_dir(run_id: str | None = None) -> str:
    resolved_run_id = run_id or default_iqm_transpiler_harness_run_id()
    return str(
        repo_path(
            "artifacts",
            "iqm_runs",
            "processed",
            "transpiler_harness",
            resolved_run_id,
        )
    )


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return ""


def _iqm_adapter_path() -> str:
    try:
        module = importlib.import_module("iqm.qiskit_iqm")
    except ImportError:
        return ""
    return str(getattr(module, "__file__", "") or "")


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "qiskit_version": _package_version("qiskit"),
        "iqm_client_version": _package_version("iqm-client"),
        "iqm_qiskit_adapter_path": _iqm_adapter_path(),
    }


def _backend_metadata(
    backend: Any,
    *,
    iqm_backend_name: str,
    iqm_use_metrics: bool,
    optimization_level: int,
) -> dict[str, Any]:
    return backend_metadata(
        backend,
        iqm_backend_name=iqm_backend_name,
        iqm_use_metrics=iqm_use_metrics,
        optimization_level=optimization_level,
        layout_method=None,
        routing_method=None,
    )


def _metric_row(circuit: Any) -> dict[str, Any]:
    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    two_qubit_gate_count = int(
        sum(count for name, count in ops.items() if name in TWO_Q_GATES)
    )
    one_qubit_gate_count = int(
        sum(
            1
            for instruction in getattr(circuit, "data", ())
            if _instruction_arity(instruction) == 1
        )
    )
    size = int(circuit.size())
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": size,
        "cz_count": int(ops.get("cz", 0)),
        "r_count": int(ops.get("r", 0)),
        "one_qubit_gate_count": one_qubit_gate_count,
        "two_qubit_gate_count": two_qubit_gate_count,
        "count_ops_json": json.dumps(ops, sort_keys=True),
    }


def _warning_flags(
    row: dict[str, Any],
    *,
    max_depth_warning: int,
    max_cz_warning: int,
) -> str:
    flags: list[str] = []
    depth = row.get("depth")
    cz_count = row.get("cz_count")
    if _value_gt(depth, max_depth_warning):
        flags.append(f"depth_gt_{max_depth_warning}")
    if _value_gt(cz_count, max_cz_warning):
        flags.append(f"cz_gt_{max_cz_warning}")
    return ";".join(flags)


def _candidate_identity(candidate: DirectBasisCandidate) -> dict[str, Any]:
    return {
        "class_name": candidate.class_name,
        "candidate_name": candidate.candidate_name,
        "basis_candidate_name": candidate.name,
        "basis_candidate_type": candidate.candidate_type,
    }


def _unsupported_candidate_row(
    candidate: DirectBasisCandidate,
    *,
    config: IqmTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = _base_row(candidate, config=config, metadata=metadata)
    row.update(
        {
            "strategy_name": "",
            "seed_transpiler": None,
            "success": False,
            "status": "unsupported_candidate",
            "error_type": "UnsupportedCandidate",
            "error_message": candidate.error_message,
            "compile_time_seconds": None,
            "warning_flags": "",
        }
    )
    row.update(_empty_strategy_metadata())
    row.update(_null_metrics())
    return row


def _trial_row(
    candidate: DirectBasisCandidate,
    *,
    result: Any,
    config: IqmTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = _base_row(candidate, config=config, metadata=metadata)
    strategy_name = getattr(result, "strategy_name", "")
    success = bool(getattr(result, "success", False))
    row.update(
        {
            "strategy_name": strategy_name,
            "seed_transpiler": getattr(result, "seed_transpiler", None),
            "success": success,
            "status": "ok" if success else "failed",
            "error_type": "" if success else getattr(result, "error_type", ""),
            "error_message": "" if success else getattr(result, "error_message", ""),
            "compile_time_seconds": getattr(result, "compile_time_seconds", None),
        }
    )
    row.update(_strategy_metadata(strategy_name))
    if success:
        metrics = _metric_row(getattr(result, "circuit"))
        row.update(metrics)
        row["warning_flags"] = _warning_flags(
            metrics,
            max_depth_warning=config.max_depth_warning,
            max_cz_warning=config.max_cz_warning,
        )
    else:
        row.update(_null_metrics())
        row["warning_flags"] = ""
    return row


def _best_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("class_name"), row.get("candidate_name"))
        grouped.setdefault(key, []).append(row)

    best_rows: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        unsupported = next(
            (
                row
                for row in group_rows
                if row.get("status") == "unsupported_candidate"
            ),
            None,
        )
        if unsupported is not None:
            best_rows.append(dict(unsupported))
            continue

        successful = [row for row in group_rows if bool(row.get("success"))]
        if not successful:
            failed_row = dict(group_rows[0])
            failed_row["status"] = "failed_all_strategies"
            failed_row["warning_flags"] = "failed_all_strategies"
            best_rows.append(failed_row)
            continue

        best_rows.append(
            dict(
                sorted(
                    successful,
                    key=lambda row: (
                        _sort_metric(row.get("depth")),
                        _sort_metric(row.get("cz_count")),
                        _sort_metric(row.get("r_count")),
                        _sort_metric(row.get("size")),
                    ),
                )[0]
            )
        )
    return best_rows


def _summary(
    rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_count": len(best_rows),
        "trial_count": len(rows),
        "successful_trial_count": sum(1 for row in rows if bool(row.get("success"))),
        "failed_trial_count": sum(1 for row in rows if row.get("status") == "failed"),
        "unsupported_candidate_count": sum(
            1 for row in rows if row.get("status") == "unsupported_candidate"
        ),
        "failed_all_strategy_count": sum(
            1 for row in best_rows if row.get("status") == "failed_all_strategies"
        ),
    }


def run_iqm_transpiler_harness(
    config: IqmTranspilerHarnessConfig,
    *,
    strategy_runner: Any = run_iqm_transpiler_strategy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    strategy_names = config.strategy_names or iqm_transpiler_strategy_names()
    n_transpile_runs = _validated_n_transpile_runs(config.n_transpile_runs)
    metadata = {
        **_runtime_metadata(),
        **_backend_metadata(
            config.backend,
            iqm_backend_name=config.iqm_backend_name,
            iqm_use_metrics=config.iqm_use_metrics,
            optimization_level=config.optimization_level,
        ),
    }

    rows: list[dict[str, Any]] = []
    for candidate in config.candidates:
        if not candidate.is_supported:
            rows.append(
                _unsupported_candidate_row(
                    candidate,
                    config=config,
                    metadata=metadata,
                )
            )
            continue

        circuit = build_direct_basis_graph_state_circuit(
            config.state_name,
            candidate.matrix,
            n_qutrits=config.n_qutrits,
        )
        for seed in range(n_transpile_runs):
            for strategy_name in strategy_names:
                result = _run_strategy_trial(
                    strategy_runner,
                    strategy_name,
                    circuit,
                    backend=config.backend,
                    seed_transpiler=seed,
                    optimization_level=config.optimization_level,
                )
                rows.append(
                    _trial_row(
                        candidate,
                        result=result,
                        config=config,
                        metadata=metadata,
                    )
                )

    best_rows = _best_trial_rows(rows)
    summary = _summary(rows, best_rows)
    return pd.DataFrame(rows), pd.DataFrame(best_rows), summary


def write_iqm_transpiler_harness_outputs(
    output_dir: str | Path,
    *,
    all_trials: pd.DataFrame,
    best_by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_trials_csv = output_path / "all_trials.csv"
    best_by_candidate_csv = output_path / "best_by_candidate.csv"
    summary_json = output_path / "summary.json"

    all_trials.to_csv(all_trials_csv, index=False)
    best_by_candidate.to_csv(best_by_candidate_csv, index=False)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "all_trials_csv": str(all_trials_csv),
        "best_by_candidate_csv": str(best_by_candidate_csv),
        "summary_json": str(summary_json),
    }


def _base_row(
    candidate: DirectBasisCandidate,
    *,
    config: IqmTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **metadata,
        "state_name": config.state_name,
        "n_qutrits": config.n_qutrits,
        **_candidate_identity(candidate),
    }


def _null_metrics() -> dict[str, Any]:
    return {key: None for key in _METRIC_KEYS}


def _instruction_arity(instruction: Any) -> int:
    qubits = getattr(instruction, "qubits", None)
    if qubits is None:
        try:
            qubits = instruction[1]
        except (IndexError, TypeError):
            return 0
    return len(qubits)


def _strategy_metadata(strategy_name: str) -> dict[str, Any]:
    try:
        strategy = get_iqm_transpiler_strategy(strategy_name)
    except ValueError:
        return _empty_strategy_metadata()
    return {
        "strategy_kind": strategy.kind,
        "strategy_scheduling_method": strategy.scheduling_method or "",
        "strategy_remove_final_rzs": bool(strategy.remove_final_rzs),
    }


def _empty_strategy_metadata() -> dict[str, Any]:
    return {
        "strategy_kind": "",
        "strategy_scheduling_method": "",
        "strategy_remove_final_rzs": None,
    }


def _validated_n_transpile_runs(n_transpile_runs: Any) -> int:
    try:
        value = int(n_transpile_runs)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_transpile_runs must be an integer >= 1") from exc
    if value < 1:
        raise ValueError("n_transpile_runs must be >= 1")
    return value


def _run_strategy_trial(
    strategy_runner: Any,
    strategy_name: str,
    circuit: Any,
    *,
    backend: Any,
    seed_transpiler: int,
    optimization_level: int,
) -> Any:
    started = time.perf_counter()
    try:
        return strategy_runner(
            strategy_name,
            circuit.copy(),
            backend=backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return IqmTranspilerStrategyResult(
            strategy_name=strategy_name,
            seed_transpiler=seed_transpiler,
            success=False,
            circuit=None,
            compile_time_seconds=time.perf_counter() - started,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _value_gt(value: Any, threshold: int) -> bool:
    try:
        return value is not None and float(value) > float(threshold)
    except (TypeError, ValueError):
        return False


def _sort_metric(value: Any) -> float:
    try:
        if value is None:
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
