"""Shared validation for text crossing experiment artifact boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_CREDENTIAL_MARKER = re.compile(
    r"(?i)(?:authorization|bearer|token|api[-_ ]?key|password|secret)\b"
    r"(?:\s*[:=]\s*|\s+)\S+"
)
_URL = re.compile(r"(?i)https?://[^\s<>\"']+")
_CREDENTIAL_QUERY_KEYS = {
    "authorization",
    "bearer",
    "token",
    "api_key",
    "apikey",
    "password",
    "secret",
}


def credential_field_name(value: str) -> bool:
    normalized = re.sub(r"[-_\s]+", "_", value.strip().lower())
    return normalized in _CREDENTIAL_QUERY_KEYS or normalized.endswith(
        ("_authorization", "_token", "_api_key", "_apikey", "_password", "_secret")
    )


def unsafe_persisted_text(value: str) -> bool:
    if _CONTROL_CHARACTERS.search(value) or _CREDENTIAL_MARKER.search(value):
        return True
    for candidate in _URL.findall(value):
        try:
            parsed = urlsplit(candidate)
            if parsed.username is not None or parsed.password is not None:
                return True
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                if credential_field_name(key):
                    return True
        except ValueError:
            return True
    return False


def validate_persisted_strings(
    value: Any,
    *,
    description: str,
    error_type: type[Exception],
    active: set[int] | None = None,
) -> None:
    if isinstance(value, str):
        if unsafe_persisted_text(value):
            raise error_type(f"{description} contains unsafe text") from None
        return
    if isinstance(value, (bytes, bytearray)):
        return
    if not isinstance(value, (Mapping, Sequence, Set)):
        return
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise error_type(f"{description} contains a recursive value") from None
    active.add(identity)
    try:
        items = value.items() if isinstance(value, Mapping) else enumerate(value)
        for key, item in items:
            if isinstance(value, Mapping) and isinstance(key, str) and credential_field_name(key):
                raise error_type(f"{description} contains unsafe text") from None
            validate_persisted_strings(
                key,
                description=description,
                error_type=error_type,
                active=active,
            )
            validate_persisted_strings(
                item,
                description=description,
                error_type=error_type,
                active=active,
            )
    finally:
        active.remove(identity)


__all__ = [
    "credential_field_name",
    "unsafe_persisted_text",
    "validate_persisted_strings",
]
