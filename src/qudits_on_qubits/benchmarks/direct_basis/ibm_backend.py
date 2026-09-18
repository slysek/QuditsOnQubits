"""Read-only IBM target loading and ISA compilation for encoding benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from qiskit.transpiler import generate_preset_pass_manager

from ..._ibm_runtime import runtime_account_options

def load_ibm_backend(backend_name: str, *, account_name: str | None = None, instance: str | None = None) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        options = runtime_account_options(account_name=account_name, instance=instance)
        backend = QiskitRuntimeService(**options).backend(backend_name)
        if backend.name != backend_name:
            raise ValueError("backend identity mismatch")
        return backend
    except Exception as error:
        raise RuntimeError(f"Could not load requested IBM backend ({type(error).__name__})") from None


def ibm_backend_metadata(backend: Any) -> dict[str, Any]:
    return {
        "transpiler_backend": "ibm",
        "ibm_backend_name": backend.name,
        "backend_num_qubits": backend.num_qubits,
        "backend_operation_names": json.dumps(sorted(backend.operation_names)),
        "target_retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def build_ibm_pass_manager(backend: Any, **options: Any) -> Any:
    return generate_preset_pass_manager(
        backend=backend,
        **{key: (list(value) if key == "initial_layout" else value)
           for key, value in options.items() if value is not None},
    )
