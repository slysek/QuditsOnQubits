from __future__ import annotations

import importlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from qiskit import qpy

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    export_direct_basis_candidate_circuits,
)
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding
from qudits_on_qubits.benchmarks.direct_basis.piast_backend import (
    backend_metadata,
)
from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies import (
    PiastTranspilerStrategyResult,
    _should_capture_transpiler_failure,
    get_piast_transpiler_strategy,
    piast_transpiler_strategy_names,
    run_piast_transpiler_strategy,
)
from qudits_on_qubits.core.project_paths import repo_path


@dataclass(frozen=True)
class PiastTranspilerHarnessConfig:
    state_name: str
    n_qutrits: int | None
    backend: Any
    candidates: Iterable[DirectBasisCandidate]
    quantum_circuits_dir: str | Path | None = None
    strategy_names: tuple[str, ...] = ()
    n_transpile_runs: int = 1
    optimization_level: int = 3
    max_depth_warning: int = 100
    max_rxx_warning: int = 50


_METRIC_KEYS = (
    "num_qubits",
    "depth",
    "size",
    "rxx_count",
    "r_count",
    "rz_count",
    "one_qubit_gate_count",
    "two_qubit_gate_count",
    "non_native_gate_count",
    "non_native_ops_json",
    "native_compliant",
    "count_ops_json",
)

_ARTIFACT_KEYS = (
    "quantum_circuit_dir",
    "f3_w_qpy",
    "cz3_w_qpy",
    "graph_state_qpy",
    "graph_state_transpiled_qpy",
    "basis_change_qpy",
    "basis_change_matrix_npy",
    "E_npy",
    "W_npy",
)

_SELECTION_METRIC_KEYS = (
    "best_depth",
    "best_size",
    "best_two_qubit_gate_count",
    "best_one_qubit_gate_count",
)

_IGNORED_ARITY_OPS = {"barrier", "delay", "measure"}
_TRIAL_QPY_STRATEGY_STEMS = {
    "transpile_aqt_plugin": "tr_aqt",
    "preset_aqt_plugin": "pm_aqt",
}


def default_piast_transpiler_harness_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_piast_transpiler_harness_output_dir(run_id: str | None = None) -> str:
    resolved_run_id = run_id or default_piast_transpiler_harness_run_id()
    return str(
        repo_path(
            "artifacts",
            "piast_runs",
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


def _module_path(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return ""
    return str(getattr(module, "__file__", "") or "")


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "qiskit_version": _package_version("qiskit"),
        "qiskit_aqt_provider_version": _package_version("qiskit-aqt-provider"),
        "pcss_qapi_version": _package_version("pcss_qapi"),
        "pcss_qapi_path": _module_path("pcss_qapi"),
    }


def _backend_metadata(
    backend: Any,
    *,
    optimization_level: int,
) -> dict[str, Any]:
    return backend_metadata(
        backend,
        optimization_level=optimization_level,
    )


def _instruction_name(instruction: Any) -> str:
    operation = getattr(instruction, "operation", None)
    name = getattr(operation, "name", None)
    if name is None:
        try:
            name = instruction[0].name
        except (AttributeError, IndexError, TypeError):
            return ""
    return str(name)


def _instruction_arity(instruction: Any) -> int:
    qubits = getattr(instruction, "qubits", None)
    if qubits is None:
        try:
            qubits = instruction[1]
        except (IndexError, TypeError):
            return 0
    return len(qubits)


def _metric_row(
    circuit: Any,
    *,
    native_operation_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    native_names = {str(name) for name in native_operation_names or ()}
    non_native_ops = _non_native_ops(ops, native_names)
    non_native_gate_count = int(sum(non_native_ops.values()))
    one_qubit_gate_count = int(
        sum(
            1
            for instruction in getattr(circuit, "data", ())
            if _instruction_arity(instruction) == 1
            and _instruction_name(instruction) not in _IGNORED_ARITY_OPS
        )
    )
    two_qubit_gate_count = int(
        sum(
            1
            for instruction in getattr(circuit, "data", ())
            if _instruction_arity(instruction) == 2
            and _instruction_name(instruction) not in _IGNORED_ARITY_OPS
        )
    )
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "rxx_count": int(ops.get("rxx", 0)),
        "r_count": int(ops.get("r", 0)),
        "rz_count": int(ops.get("rz", 0)),
        "one_qubit_gate_count": one_qubit_gate_count,
        "two_qubit_gate_count": two_qubit_gate_count,
        "non_native_gate_count": non_native_gate_count,
        "non_native_ops_json": json.dumps(non_native_ops, sort_keys=True),
        "native_compliant": non_native_gate_count == 0,
        "count_ops_json": json.dumps(ops, sort_keys=True),
    }


def _non_native_ops(ops: dict[str, int], native_names: set[str]) -> dict[str, int]:
    if not native_names:
        return {}
    return {
        name: int(count)
        for name, count in ops.items()
        if name not in native_names and name not in _IGNORED_ARITY_OPS
    }


def _operation_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("backend_operation_names")
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    try:
        return [str(name) for name in value]
    except TypeError:
        return []


def _warning_flags(
    row: dict[str, Any],
    *,
    max_depth_warning: int,
    max_rxx_warning: int,
) -> str:
    flags: list[str] = []
    if _value_gt(row.get("depth"), max_depth_warning):
        flags.append(f"depth_gt_{max_depth_warning}")
    if _value_gt(row.get("rxx_count"), max_rxx_warning):
        flags.append(f"rxx_gt_{max_rxx_warning}")
    if _value_gt(row.get("non_native_gate_count"), 0):
        flags.append("non_native_gate_count_gt_0")
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
    config: PiastTranspilerHarnessConfig,
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
    config: PiastTranspilerHarnessConfig,
    metadata: dict[str, Any],
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    row = _base_row(candidate, config=config, metadata=metadata)
    row.update(artifact_paths or _empty_artifact_paths())
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
        metrics = _metric_row(
            getattr(result, "circuit"),
            native_operation_names=_operation_names_from_metadata(metadata),
        )
        row.update(metrics)
        row.update(_selection_metric_aliases(metrics))
        row["warning_flags"] = _warning_flags(
            metrics,
            max_depth_warning=config.max_depth_warning,
            max_rxx_warning=config.max_rxx_warning,
        )
        transpiled_qpy = _export_trial_transpiled_circuit(
            artifact_paths or {},
            getattr(result, "circuit"),
            strategy_name=strategy_name,
            seed_transpiler=getattr(result, "seed_transpiler", None),
        )
        if transpiled_qpy:
            row["graph_state_transpiled_qpy"] = transpiled_qpy
    else:
        row.update(_null_metrics())
        row.update(_null_selection_metric_aliases())
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
            (row for row in group_rows if row.get("status") == "unsupported_candidate"),
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
                        _native_compliance_rank(row.get("native_compliant")),
                        _sort_metric(row.get("non_native_gate_count")),
                        _sort_metric(row.get("depth")),
                        _sort_metric(row.get("rxx_count")),
                        _sort_metric(row.get("two_qubit_gate_count")),
                        _sort_metric(row.get("r_count")),
                        _sort_metric(row.get("rz_count")),
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


def run_piast_transpiler_harness(
    config: PiastTranspilerHarnessConfig,
    *,
    strategy_runner: Any = run_piast_transpiler_strategy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    strategy_names = config.strategy_names or piast_transpiler_strategy_names()
    n_transpile_runs = _validated_n_transpile_runs(config.n_transpile_runs)
    metadata = {
        **_runtime_metadata(),
        **_backend_metadata(
            config.backend,
            optimization_level=config.optimization_level,
        ),
    }

    candidates = list(config.candidates)
    candidate_count = len(candidates)

    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            "[piast_transpiler_harness] "
            f"{candidate_index}/{candidate_count} "
            f"{candidate.class_name}/{candidate.candidate_name}",
            flush=True,
        )
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
        artifact_paths = _export_candidate_artifacts(
            config,
            candidate,
            graph_state_circuit=circuit,
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
                        artifact_paths=artifact_paths,
                    )
                )

    best_rows = _best_trial_rows(rows)
    summary = _summary(rows, best_rows)
    return pd.DataFrame(rows), pd.DataFrame(best_rows), summary


def write_piast_transpiler_harness_outputs(
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
    config: PiastTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **metadata,
        "state_name": config.state_name,
        "n_qutrits": config.n_qutrits,
        **_candidate_identity(candidate),
        **_empty_artifact_paths(),
        **_null_selection_metric_aliases(),
    }


def _null_metrics() -> dict[str, Any]:
    return {key: None for key in _METRIC_KEYS}


def _empty_artifact_paths() -> dict[str, str]:
    return {key: "" for key in _ARTIFACT_KEYS}


def _null_selection_metric_aliases() -> dict[str, Any]:
    return {key: None for key in _SELECTION_METRIC_KEYS}


def _selection_metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_depth": metrics.get("depth"),
        "best_size": metrics.get("size"),
        "best_two_qubit_gate_count": metrics.get("two_qubit_gate_count"),
        "best_one_qubit_gate_count": metrics.get("one_qubit_gate_count"),
    }


def _strategy_metadata(strategy_name: str) -> dict[str, Any]:
    try:
        strategy = get_piast_transpiler_strategy(strategy_name)
    except ValueError:
        return _empty_strategy_metadata()
    return {
        "strategy_kind": strategy.kind,
        "strategy_translation_method": strategy.translation_method,
        "strategy_scheduling_method": strategy.scheduling_method,
    }


def _empty_strategy_metadata() -> dict[str, Any]:
    return {
        "strategy_kind": "",
        "strategy_translation_method": "",
        "strategy_scheduling_method": "",
    }


def _validated_n_transpile_runs(n_transpile_runs: Any) -> int:
    try:
        value = int(n_transpile_runs)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_transpile_runs must be an integer >= 1") from exc
    if value < 1:
        raise ValueError("n_transpile_runs must be >= 1")
    return value


def _export_candidate_artifacts(
    config: PiastTranspilerHarnessConfig,
    candidate: DirectBasisCandidate,
    *,
    graph_state_circuit: Any,
) -> dict[str, str]:
    if config.quantum_circuits_dir is None:
        return _empty_artifact_paths()

    paths = export_direct_basis_candidate_circuits(
        quantum_circuits_dir=str(config.quantum_circuits_dir),
        state_name=config.state_name,
        n_qutrits=config.n_qutrits,
        class_name=candidate.class_name,
        candidate_name=candidate.candidate_name,
        basis_matrix=candidate.matrix,
        graph_state_circuit=graph_state_circuit,
        selection_label="exact",
        legacy_exact_transpiled_filename=False,
    )
    paths["graph_state_transpiled_qpy"] = ""
    paths.update(_encoding_artifact_aliases(paths.get("basis_change_matrix_npy", "")))
    return {key: str(paths.get(key, "")) for key in _ARTIFACT_KEYS}


def _encoding_artifact_aliases(matrix_path: str) -> dict[str, str]:
    if not matrix_path:
        return {"E_npy": "", "W_npy": ""}

    path = Path(matrix_path)
    if not path.is_file():
        return {"E_npy": "", "W_npy": ""}

    matrix = np.load(path)
    if matrix.shape == (3, 3):
        e_path = path.with_name("E.npy")
        if not e_path.is_file():
            np.save(e_path, encoding_embedding(matrix))
        return {"E_npy": str(e_path), "W_npy": str(path)}
    if matrix.shape == (4, 3):
        return {"E_npy": str(path), "W_npy": ""}
    return {"E_npy": "", "W_npy": ""}


def _export_trial_transpiled_circuit(
    artifact_paths: dict[str, str],
    circuit: Any,
    *,
    strategy_name: str,
    seed_transpiler: Any,
) -> str:
    output_dir = artifact_paths.get("quantum_circuit_dir")
    if not output_dir or circuit is None:
        return ""
    path = Path(output_dir) / (
        f"t_{_trial_qpy_strategy_stem(strategy_name)}_s{_safe_path_part(seed_transpiler)}.qpy"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        qpy.dump(circuit, handle)
    return str(path)


def _trial_qpy_strategy_stem(strategy_name: str) -> str:
    return _TRIAL_QPY_STRATEGY_STEMS.get(
        str(strategy_name),
        _safe_path_part(strategy_name)[:32],
    )


def _safe_path_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


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
        if not _should_capture_transpiler_failure(exc):
            raise
        return PiastTranspilerStrategyResult(
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


def _native_compliance_rank(value: Any) -> int:
    return 0 if bool(value) else 1
