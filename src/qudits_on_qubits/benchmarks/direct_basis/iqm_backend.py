from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

REQUIRED_IQM_ENV_KEYS = ("IQM_SERVER_URL", "IQM_TOKEN")
EXACT_RZ_SCHEDULING_METHOD = "move_routing_exact_global_phase"


@dataclass(frozen=True)
class IqmEnvironment:
    server_url: str
    token: str


def repo_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[4].joinpath(*parts)


def default_env_path() -> Path:
    return repo_path(".env")


def safe_backend_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "iqm_backend"


def load_iqm_environment(env_path: str | Path | None = None) -> IqmEnvironment:
    path = Path(env_path) if env_path is not None else default_env_path()
    if not path.is_file():
        raise RuntimeError(f"Missing required IQM env file: {path}")

    values = dotenv_values(path)
    missing_keys = [
        key for key in REQUIRED_IQM_ENV_KEYS if not str(values.get(key) or "").strip()
    ]
    if missing_keys:
        raise RuntimeError(
            "Missing required IQM env value(s): " + ", ".join(missing_keys)
        )

    load_dotenv(path, override=True)
    return IqmEnvironment(
        server_url=str(values["IQM_SERVER_URL"]).strip(),
        token=str(values["IQM_TOKEN"]).strip(),
    )


def load_iqm_backend(
    quantum_computer: str,
    use_metrics: bool = False,
    env_path: str | Path | None = None,
) -> Any:
    try:
        from iqm.qiskit_iqm import IQMProvider
    except ModuleNotFoundError as exc:
        if exc.name == "iqm.qiskit_iqm" or "iqm.qiskit_iqm" in str(exc):
            raise RuntimeError(
                "Missing IQM Qiskit adapter. Install the pinned project "
                "dependencies with `python -m pip install -e .`, or reinstall "
                "IQM with `python -m pip install --force-reinstall "
                "\"iqm-client[qiskit]>=34,<35\"`."
            ) from exc
        raise

    env = load_iqm_environment(env_path)
    provider = IQMProvider(
        env.server_url,
        quantum_computer=quantum_computer,
    )
    return provider.get_backend(use_metrics=bool(use_metrics))


def build_iqm_pass_manager(
    backend: Any,
    optimization_level: int = 3,
    seed_transpiler: int | None = None,
    layout_method: str | None = None,
    routing_method: str | None = None,
    scheduling_method: str | None = EXACT_RZ_SCHEDULING_METHOD,
    approximation_degree: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "backend": backend,
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
    }
    if layout_method is not None:
        kwargs["layout_method"] = layout_method
    if routing_method is not None:
        kwargs["routing_method"] = routing_method
    if scheduling_method is not None:
        kwargs["scheduling_method"] = scheduling_method
    if approximation_degree is not None:
        kwargs["approximation_degree"] = float(approximation_degree)
    return generate_preset_pass_manager(**kwargs)


def _value_or_call(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _backend_operation_names(backend: Any) -> list[str]:
    operation_names = _value_or_call(getattr(backend, "operation_names", None))
    if operation_names is None:
        target = getattr(backend, "target", None)
        operation_names = getattr(target, "operation_names", None)
    if operation_names is None:
        return []
    return sorted(str(name) for name in operation_names)


def _backend_num_qubits(backend: Any) -> int:
    num_qubits = _value_or_call(getattr(backend, "num_qubits", None))
    if num_qubits is None:
        target = getattr(backend, "target", None)
        num_qubits = getattr(target, "num_qubits", None)
    if num_qubits is None:
        configuration = _value_or_call(getattr(backend, "configuration", None))
        num_qubits = getattr(configuration, "num_qubits", 0)
    try:
        return int(num_qubits or 0)
    except (TypeError, ValueError):
        return 0


def _backend_coupling_map_size(backend: Any) -> int:
    coupling_map = _value_or_call(getattr(backend, "coupling_map", None))
    if coupling_map is None:
        return 0
    get_edges = getattr(coupling_map, "get_edges", None)
    if callable(get_edges):
        return len(get_edges())
    try:
        return len(coupling_map)
    except TypeError:
        return 0


def _backend_has_resonators(backend: Any) -> bool:
    has_resonators = _value_or_call(getattr(backend, "has_resonators", None))
    return bool(has_resonators)


def _backend_calibration_set_id(backend: Any) -> str:
    for attribute in ("calibration_set_id", "_calibration_set_id"):
        value = _value_or_call(getattr(backend, attribute, None))
        if value is not None:
            return str(value)
    return ""


def backend_metadata(
    backend: Any,
    *,
    iqm_backend_name: str,
    iqm_use_metrics: bool,
    optimization_level: int,
    layout_method: str | None,
    routing_method: str | None,
    scheduling_method: str | None = EXACT_RZ_SCHEDULING_METHOD,
) -> dict[str, Any]:
    return {
        "transpiler_backend": "iqm",
        "iqm_backend_name": iqm_backend_name,
        "iqm_use_metrics": bool(iqm_use_metrics),
        "optimization_level": optimization_level,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "scheduling_method": scheduling_method,
        "backend_num_qubits": _backend_num_qubits(backend),
        "backend_operation_names": json.dumps(_backend_operation_names(backend)),
        "backend_coupling_map_size": _backend_coupling_map_size(backend),
        "backend_has_resonators": _backend_has_resonators(backend),
        "backend_calibration_set_id": _backend_calibration_set_id(backend),
    }
