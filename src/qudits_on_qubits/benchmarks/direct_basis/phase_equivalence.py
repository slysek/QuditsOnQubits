from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .candidates import DirectBasisCandidate

DEFAULT_PHASE_ATOL = 1e-9
DEFAULT_PHASE_RTOL = 1e-7
DEFAULT_BUCKET_DECIMALS = 9
PHASE_DUPLICATE_COLUMNS = (
    "global_phase_group_id", "representative_class_name", "representative_candidate_name",
    "duplicate_class_name", "duplicate_candidate_name", "detected_phase_real",
    "detected_phase_imag", "reason",
)


@dataclass(frozen=True)
class CandidatePhaseDeduplication:
    representatives: tuple[DirectBasisCandidate, ...]
    duplicate_rows: tuple[dict[str, Any], ...]

    @property
    def removed_count(self) -> int:
        return len(self.duplicate_rows)


def global_phase_between(reference, candidate, *, atol=DEFAULT_PHASE_ATOL, rtol=DEFAULT_PHASE_RTOL):
    ref = np.asarray(reference, dtype=complex)
    cand = np.asarray(candidate, dtype=complex)
    if ref.shape != cand.shape:
        return None
    ref_zero = np.all(ref == 0)
    cand_zero = np.all(cand == 0)
    if ref_zero or cand_zero:
        return 1 + 0j if ref_zero and cand_zero else None
    pivot = int(np.argmax(np.abs(ref)))
    if cand.flat[pivot] == 0:
        return None
    phase = cand.flat[pivot] / ref.flat[pivot]
    magnitude = abs(phase)
    if not np.isclose(magnitude, 1.0, atol=atol, rtol=rtol) or magnitude == 0:
        return None
    phase = phase / magnitude
    if np.allclose(cand, phase * ref, atol=atol, rtol=rtol):
        return complex(phase)
    return None


def _bucket_key(matrix, decimals, atol=DEFAULT_PHASE_ATOL):
    arr = np.asarray(matrix, dtype=complex)
    if np.all(arr == 0):
        return (arr.shape, (0.0, 0.0))
    nonzero = np.flatnonzero(np.abs(arr) > atol)
    pivot = int(nonzero[0] if nonzero.size else np.argmax(np.abs(arr)))
    phase = arr.flat[pivot] / abs(arr.flat[pivot])
    normalized = arr / phase
    real = np.where(np.isclose(normalized.real, 0.0, atol=atol), 0.0, normalized.real)
    imag = np.where(np.isclose(normalized.imag, 0.0, atol=atol), 0.0, normalized.imag)
    values = tuple(np.round(np.column_stack((real.ravel(), imag.ravel())), decimals).ravel())
    return (arr.shape, values)


# Descriptive alias retained for callers that need to inspect the initial key.
_canonical_bucket_key = _bucket_key


def _sort_key(candidate):
    return (0 if candidate.class_name == "baseline" else 1,
            candidate.class_name, candidate.candidate_name)


def deduplicate_candidates_up_to_global_phase(
    candidates: Iterable[DirectBasisCandidate], *, atol=DEFAULT_PHASE_ATOL,
    rtol=DEFAULT_PHASE_RTOL, bucket_decimals=DEFAULT_BUCKET_DECIMALS,
) -> CandidatePhaseDeduplication:
    ordered = sorted(list(candidates), key=_sort_key)
    groups: list[list[DirectBasisCandidate]] = []
    bucket_groups: dict[tuple, list[int]] = {}
    scalar_groups: dict[tuple, list[int]] = {}
    shape_groups: dict[tuple, list[int]] = {}
    phase_group_numbers: dict[int, int] = {}
    next_phase_group = 1
    supported = [c for c in ordered if c.is_supported and c.matrix is not None]
    max_size = max((np.asarray(c.matrix, dtype=complex).size for c in supported), default=0)
    weights = np.arange(1, max_size + 1, dtype=float)
    rmax = max((float(np.sum(weights[:np.asarray(c.matrix).size] *
                              (atol + rtol * np.abs(np.asarray(c.matrix, dtype=complex).flat))))
                for c in supported), default=0.0)
    duplicates: list[tuple[DirectBasisCandidate, DirectBasisCandidate, complex, int]] = []
    for item in ordered:
        if not item.is_supported or item.matrix is None:
            groups.append([item])
            continue
        key = _bucket_key(item.matrix, bucket_decimals, atol)
        found = None
        checked = set()
        for index in bucket_groups.get(key, []):
            checked.add(index)
            group = groups[index]
            if not group[0].is_supported:
                continue
            # The rounded key is only an initial bucket.  Numerical ties in
            # the pivot can put projectively equivalent matrices in buckets
            # that differ, so equivalence is always decided by verification.
            phase = global_phase_between(group[0].matrix, item.matrix, atol=atol, rtol=rtol)
            if phase is not None:
                found = (index, phase)
                break
        if found is None and rmax > 0:
            arr = np.asarray(item.matrix, dtype=complex).ravel()
            fingerprint = float(np.sum(weights[:arr.size] * np.abs(arr)))
            scalar_bin = int(np.floor(fingerprint / rmax))
            scalar_shape = np.asarray(item.matrix).shape
            for probe in (scalar_bin - 1, scalar_bin, scalar_bin + 1):
                for index in scalar_groups.get((scalar_shape, probe), []):
                    if index in checked:
                        continue
                    checked.add(index)
                    phase = global_phase_between(groups[index][0].matrix, item.matrix, atol=atol, rtol=rtol)
                    if phase is not None:
                        found = (index, phase)
                        break
                if found is not None:
                    break
        if found is None and rmax == 0 and atol == 0 and rtol == 0:
            shape = np.asarray(item.matrix).shape
            for index in shape_groups.get(shape, []):
                if index in checked:
                    continue
                checked.add(index)
                phase = global_phase_between(groups[index][0].matrix, item.matrix, atol=atol, rtol=rtol)
                if phase is not None:
                    found = (index, phase)
                    break
        if found is None:
            index = len(groups)
            groups.append([item])
            phase_group_numbers[index] = next_phase_group
            next_phase_group += 1
            bucket_groups.setdefault(key, []).append(index)
            arr = np.asarray(item.matrix, dtype=complex).ravel()
            if rmax > 0:
                scalar_groups.setdefault((np.asarray(item.matrix).shape,
                                          int(np.floor(float(np.sum(weights[:arr.size] * np.abs(arr))) / rmax))), []).append(index)
            shape_groups.setdefault(np.asarray(item.matrix).shape, []).append(index)
        else:
            index, phase = found
            duplicates.append((groups[index][0], item, phase, index))
    representatives = tuple(group[0] for group in groups)
    rows = []
    for rep, dup, phase, index in sorted(duplicates, key=lambda x: (x[3], x[1].class_name, x[1].candidate_name)):
        rows.append({
            "global_phase_group_id": f"global_phase_{phase_group_numbers[index]:04d}",
            "representative_class_name": rep.class_name,
            "representative_candidate_name": rep.candidate_name,
            "duplicate_class_name": dup.class_name,
            "duplicate_candidate_name": dup.candidate_name,
            "detected_phase_real": float(np.real(phase)),
            "detected_phase_imag": float(np.imag(phase)),
            "reason": "global_phase_equivalent_matrix",
        })
    return CandidatePhaseDeduplication(representatives, tuple(rows))
