from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

from qudits_on_qubits.encoding_search.candidates import Candidate, E_OLD


DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-10


@dataclass(frozen=True)
class BaselineEquivalence:
    is_baseline_reference: bool
    is_trivial_identity: bool
    is_baseline_equivalent: bool
    skip_reason: str = ""

    def as_row_fields(self):
        return asdict(self)


def _as_embedding(e_new):
    if e_new is None:
        return E_OLD.copy()
    return np.asarray(e_new, dtype=complex)


def _is_matrix_equal_up_to_global_phase(
    candidate,
    reference,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    candidate = np.asarray(candidate, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    if candidate.shape != reference.shape:
        return False
    if np.allclose(candidate, reference, atol=atol, rtol=rtol):
        return True

    pivot = np.unravel_index(np.argmax(np.abs(reference)), reference.shape)
    if abs(reference[pivot]) <= atol:
        return np.linalg.norm(candidate - reference) <= atol

    phase = candidate[pivot] / reference[pivot]
    if abs(phase) <= atol:
        return False
    phase = phase / abs(phase)
    return bool(np.allclose(candidate, phase * reference, atol=atol, rtol=rtol))


def _is_embedding_equal_up_to_global_phase(
    e_new,
    e_base=None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    base = E_OLD if e_base is None else np.asarray(e_base, dtype=complex)
    candidate = _as_embedding(e_new)
    return _is_matrix_equal_up_to_global_phase(candidate, base, atol=atol, rtol=rtol)


def _is_w_identity(
    e_new,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    if e_new is None:
        return True

    try:
        from encoding_change_unitary import build_encoding_change_unitary

        w = build_encoding_change_unitary(np.asarray(e_new, dtype=complex), tol=atol)
    except Exception:
        return False

    identity = np.eye(4, dtype=complex)
    return bool(
        np.allclose(w, identity, atol=atol, rtol=rtol)
        or _is_matrix_equal_up_to_global_phase(w, identity, atol=atol, rtol=rtol)
    )


def _is_baseline_equivalent_candidate(
    class_name: str,
    candidate_name: str,
    e_new,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> BaselineEquivalence:
    is_reference = class_name == "baseline" and candidate_name == "E_old"
    if is_reference:
        return BaselineEquivalence(
            is_baseline_reference=True,
            is_trivial_identity=True,
            is_baseline_equivalent=True,
            skip_reason="",
        )

    if e_new is None:
        return BaselineEquivalence(
            is_baseline_reference=False,
            is_trivial_identity=True,
            is_baseline_equivalent=True,
            skip_reason="candidate uses E_new=None, which is the baseline embedding",
        )

    embedding = _as_embedding(e_new)
    exact_embedding = np.allclose(embedding, E_OLD, atol=atol, rtol=rtol)
    phase_embedding = _is_embedding_equal_up_to_global_phase(
        embedding,
        E_OLD,
        atol=atol,
        rtol=rtol,
    )
    w_identity = _is_w_identity(embedding, atol=atol, rtol=rtol)

    if exact_embedding:
        return BaselineEquivalence(
            is_baseline_reference=False,
            is_trivial_identity=True,
            is_baseline_equivalent=True,
            skip_reason="same embedding as baseline within tolerance",
        )
    if w_identity:
        return BaselineEquivalence(
            is_baseline_reference=False,
            is_trivial_identity=True,
            is_baseline_equivalent=True,
            skip_reason="W is identity up to tolerance",
        )
    if phase_embedding:
        return BaselineEquivalence(
            is_baseline_reference=False,
            is_trivial_identity=False,
            is_baseline_equivalent=True,
            skip_reason="embedding equals baseline up to one global phase",
        )

    return BaselineEquivalence(
        is_baseline_reference=False,
        is_trivial_identity=False,
        is_baseline_equivalent=False,
        skip_reason="",
    )


def _skipped_candidate_row(
    state_name: str,
    stage: int,
    class_name: str,
    candidate_name: str,
    metadata: BaselineEquivalence,
):
    row = {
        "state_name": state_name,
        "pipeline_stage": int(stage),
        "class_name": class_name,
        "candidate_name": candidate_name,
        "is_valid": True,
        "status": "skipped_baseline_equivalent",
        "error_message": "",
        "best_depth": None,
        "mean_depth": None,
        "std_depth": None,
        "best_size": None,
        "mean_size": None,
        "best_two_qubit_gate_count": None,
        "mean_two_qubit_gate_count": None,
        "num_qubits": None,
        "best_count_ops": None,
        "n_transpile_runs": 0,
        "successful_trials": 0,
        "failed_trials": 0,
        "approx_status": "skipped",
        "approx_error_message": metadata.skip_reason,
    }
    row.update(metadata.as_row_fields())
    return row


def _filter_trivial_candidates(
    candidates: Iterable[Candidate],
    state_name: str,
    stage: int,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[list[Candidate], list[dict]]:
    kept: list[Candidate] = []
    skipped_rows: list[dict] = []

    for class_name, candidate_name, e_new in candidates:
        metadata = _is_baseline_equivalent_candidate(
            class_name,
            candidate_name,
            e_new,
            atol=atol,
            rtol=rtol,
        )
        if metadata.is_baseline_reference:
            kept.append((class_name, candidate_name, e_new))
            continue
        if metadata.is_baseline_equivalent:
            skipped_rows.append(
                _skipped_candidate_row(
                    state_name=state_name,
                    stage=stage,
                    class_name=class_name,
                    candidate_name=candidate_name,
                    metadata=metadata,
                )
            )
            continue
        kept.append((class_name, candidate_name, e_new))

    return kept, skipped_rows


def candidate_metadata_fields(
    class_name: str,
    candidate_name: str,
    e_new,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict:
    return _is_baseline_equivalent_candidate(
        class_name,
        candidate_name,
        e_new,
        atol=atol,
        rtol=rtol,
    ).as_row_fields()
