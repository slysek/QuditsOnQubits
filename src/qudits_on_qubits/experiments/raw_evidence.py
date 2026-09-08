"""Bounded JSON evidence attached to strict provider-result validation errors.

This payload is diagnostic evidence, never a validated ExecutionResult. It may
contain incomplete histograms, null PUB positions or invalid numeric counts.
The block runner persists it as invalid while legacy result callers still get
the original exception.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from .safety import credential_field_name, unsafe_persisted_text


MAX_EVIDENCE_ITEMS = 50_000
MAX_EVIDENCE_CONTAINER_ITEMS = 4096
MAX_EVIDENCE_CHARACTERS = 2_000_000
MAX_EVIDENCE_STRING = 4096


class _EvidenceCopy:
    def __init__(self) -> None:
        self.remaining = MAX_EVIDENCE_ITEMS
        self.characters = MAX_EVIDENCE_CHARACTERS
        self.incomplete = False
        self.active: set[int] = set()

    def omitted(self, reason: str) -> dict[str, str]:
        self.incomplete = True
        return {"unavailable": reason}

    def text(self, value: str) -> str | None:
        if len(value) > MAX_EVIDENCE_STRING or unsafe_persisted_text(value):
            self.incomplete = True
            return None
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if size > self.characters:
            self.incomplete = True
            return None
        self.characters -= size
        return value

    def copy(self, value: Any, depth: int = 0) -> Any:
        if self.remaining <= 0 or depth > 8:
            return self.omitted("evidence_limit")
        self.remaining -= 1
        if value is None or type(value) is bool:
            return value
        if type(value) is int:
            if value.bit_length() > 840:
                return self.omitted("integer_limit")
            size = len(str(value))
            if size > self.characters:
                return self.omitted("evidence_limit")
            self.characters -= size
            return value
        if type(value) is float:
            if not math.isfinite(value):
                return self.omitted("nonfinite_number")
            size = len(repr(value))
            if size > self.characters:
                return self.omitted("evidence_limit")
            self.characters -= size
            return value
        if isinstance(value, str):
            text = self.text(value)
            return text if text is not None else self.omitted("unsafe_or_large_text")
        if not isinstance(value, (Mapping, list, tuple)):
            return self.omitted("unsupported_value")
        if id(value) in self.active:
            return self.omitted("recursive_value")
        self.active.add(id(value))
        try:
            if isinstance(value, Mapping):
                result = {}
                for key, item in value.items():
                    if self.remaining <= 0 or len(result) >= MAX_EVIDENCE_CONTAINER_ITEMS:
                        self.incomplete = True
                        break
                    self.remaining -= 1
                    if not isinstance(key, str) or credential_field_name(key):
                        self.incomplete = True
                        continue
                    safe_key = self.text(key)
                    if safe_key is not None:
                        result[safe_key] = self.copy(item, depth + 1)
                return result
            result = []
            for item in value:
                if self.remaining <= 0 or len(result) >= MAX_EVIDENCE_CONTAINER_ITEMS:
                    self.incomplete = True
                    break
                result.append(self.copy(item, depth + 1))
            return result
        finally:
            self.active.remove(id(value))

    def histograms(self, counts: Any) -> list[Any]:
        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
            return [self.omitted("unsupported_counts_collection")]
        result = []
        for histogram in counts:
            if self.remaining <= 0 or len(result) >= MAX_EVIDENCE_CONTAINER_ITEMS:
                self.incomplete = True
                break
            self.remaining -= 1
            if histogram is None:
                self.incomplete = True
                result.append(None)
                continue
            if not isinstance(histogram, Mapping):
                result.append(self.omitted("unsupported_histogram"))
                continue
            copied = {}
            for index, (key, count) in enumerate(histogram.items()):
                if self.remaining <= 0 or len(copied) >= MAX_EVIDENCE_CONTAINER_ITEMS:
                    self.incomplete = True
                    break
                self.remaining -= 1
                # Raw keys are measurement words, not arbitrary provider text.
                if (not isinstance(key, str) or not key or len(key) > MAX_EVIDENCE_STRING
                        or set(key) - {"0", "1", " "}
                        or self.text(key) is None):
                    self.incomplete = True
                    copied[f"invalid_count_key_{index}"] = self.omitted("invalid_bitstring_key")
                    continue
                # Never persist a string/object supplied in place of a count.
                copied[key] = self.copy(count) if type(count) in (int, float, bool) else self.omitted("invalid_count_type")
            result.append(copied)
        return result


def attach_raw_evidence(error: Exception, *, job_id: str, target_identity: Any,
                        counts: Any, diagnostics: Any = None) -> Exception:
    """Attach safe retrieved counts without weakening the adapter's exception.

    None histogram entries preserve PUB positions when only register-level
    evidence is available. Truncation, nonfinite data and unsafe fields are
    explicitly marked; untrusted provider objects are never serialized.
    """
    # Origin has its own small allowance, so a large histogram cannot consume
    # the identity needed to associate even invalid evidence with its job.
    origin = _EvidenceCopy()
    origin.remaining = min(MAX_EVIDENCE_ITEMS, 256)
    origin.characters = min(MAX_EVIDENCE_CHARACTERS, 32_768)
    identity = target_identity.to_safe_dict() if hasattr(target_identity, "to_safe_dict") else target_identity
    safe_job = origin.text(job_id) if isinstance(job_id, str) else None
    safe_identity = origin.copy(identity)
    copier = _EvidenceCopy()
    histograms = copier.histograms(counts)
    metadata = {"validation": "invalid_provider_raw"}
    if diagnostics is not None:
        metadata["diagnostics"] = copier.copy(diagnostics)
    metadata["evidence_incomplete"] = copier.incomplete or origin.incomplete
    error.raw_evidence = {  # type: ignore[attr-defined]
        "counts": histograms, "job_id": safe_job, "target_identity": safe_identity,
        "status": None, "timing": {}, "metadata": metadata,
    }
    return error
