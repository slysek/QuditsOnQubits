from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

from qudits_on_qubits.core.project_paths import repo_path


PIAST_BACKEND_NAME = "piast"
PCSS_PROVIDER_NAME = "PCSS_AQTProvider"
REQUIRED_PIAST_ENV_KEYS = ("PCSS_QAPI_TOKEN",)
AQT_TRANSLATION_METHOD = "aqt"
AQT_SCHEDULING_METHOD = "aqt"


@dataclass(frozen=True)
class PiastEnvironment:
    token: str


def default_env_path() -> Path:
    return Path(repo_path(".env"))


def safe_backend_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return slug or "piast_backend"


def load_piast_environment(env_path: str | Path | None = None) -> PiastEnvironment:
    path = Path(env_path) if env_path is not None else default_env_path()
    if not path.is_file():
        raise RuntimeError(f"Missing required Piast env file: {path}")

    values = dotenv_values(path)
    missing_keys = [
        key for key in REQUIRED_PIAST_ENV_KEYS if not str(values.get(key) or "").strip()
    ]
    if missing_keys:
        raise RuntimeError(
            "Missing required Piast env value(s): " + ", ".join(missing_keys)
        )

    load_dotenv(path, override=True)
    return PiastEnvironment(token=str(values["PCSS_QAPI_TOKEN"]).strip())


def load_piast_backend(env_path: str | Path | None = None) -> Any:
    try:
        from pcss_qapi import AuthorizationService
        from pcss_qapi.aqt.provider import PCSS_AQTProvider
    except ModuleNotFoundError as exc:
        if exc.name == "pcss_qapi" or "pcss_qapi" in str(exc):
            raise RuntimeError(
                "Missing PCSS QAPI package. Activate the PiastQEnv conda "
                "environment or install `pcss_qapi` in the active environment."
            ) from exc
        raise

    env = load_piast_environment(env_path)
    AuthorizationService.login(env.token)
    provider = PCSS_AQTProvider()
    return provider.get_direct_access_backend()


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


def _backend_name(backend: Any) -> str:
    name = _value_or_call(getattr(backend, "name", None))
    return str(name or "direct-access")


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return ""


def backend_metadata(
    backend: Any,
    *,
    optimization_level: int,
    translation_method: str = AQT_TRANSLATION_METHOD,
    scheduling_method: str = AQT_SCHEDULING_METHOD,
) -> dict[str, Any]:
    return {
        "transpiler_backend": PIAST_BACKEND_NAME,
        "piast_backend_name": PIAST_BACKEND_NAME,
        "pcss_provider": PCSS_PROVIDER_NAME,
        "backend_name": _backend_name(backend),
        "optimization_level": int(optimization_level),
        "translation_method": translation_method,
        "scheduling_method": scheduling_method,
        "backend_num_qubits": _backend_num_qubits(backend),
        "backend_operation_names": json.dumps(_backend_operation_names(backend)),
        "backend_coupling_map_size": _backend_coupling_map_size(backend),
        "qiskit_aqt_provider_version": _package_version("qiskit-aqt-provider"),
        "pcss_qapi_version": _package_version("pcss_qapi"),
    }
