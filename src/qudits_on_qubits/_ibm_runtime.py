"""Local IBM credentials, shared by compilation and hardware execution."""
from __future__ import annotations

import os

from dotenv import dotenv_values, find_dotenv


def runtime_account_options(*, account_name: str | None = None, instance: str | None = None) -> dict[str, str]:
    """Read the nearest .env from cwd upwards without changing process state.

    Explicit named accounts bypass .env entirely. Otherwise process variables
    override file values, and an explicit instance overrides both. Never persist
    or log the returned dictionary: it can contain an API key.
    """
    options = {"instance": instance} if instance is not None else {}
    if account_name is not None:
        return {"name": account_name, **options}

    path = find_dotenv(usecwd=True)
    values = dotenv_values(path, encoding="utf-8-sig", interpolate=False) if path else {}

    def setting(name: str) -> str | None:
        value = os.environ.get(name, values.get(name))
        return value.strip() or None if isinstance(value, str) else None

    if instance is None:
        configured_instance = setting("QISKIT_IBM_INSTANCE")
        if configured_instance:
            options["instance"] = configured_instance
    token = setting("QISKIT_IBM_TOKEN")
    if token:
        channel = setting("QISKIT_IBM_CHANNEL") or "ibm_quantum_platform"
        if channel not in {"ibm_quantum_platform", "ibm_cloud"}:
            raise ValueError("QISKIT_IBM_CHANNEL must be ibm_quantum_platform or ibm_cloud")
        options.update(token=token, channel=channel)
    return options
