from __future__ import annotations

from typing import Any

from .postprocessing import compute_bell_value_from_counts
from .sampler_circuits import decoding_kwargs_from_metadata


def compute_bell_value_from_counts_aqt(
    sampler_circuits: Any,
    metadata: Any,
    *,
    backend: Any,
    shots: int = 1024,
    sampler_options: Any = None,
    run_options: Any = None,
    timeout: Any = None,
    poll_interval: float = 5.0,
) -> tuple[complex, dict[str, Any]]:
    circuits = list(sampler_circuits)
    settings = [
        tuple(setting)
        for setting in metadata["setting_by_circuit_index"]
    ]
    sampler_type = _load_piastq_sampler()
    sampler = sampler_type(backend, options=dict(sampler_options or {}))
    job = sampler.run(circuits, shots=shots, **dict(run_options or {}))
    result = job.result(timeout=timeout, poll_interval=poll_interval)
    counts = job.counts()
    counts_by_setting = dict(zip(settings, counts, strict=True))
    bell_value = compute_bell_value_from_counts(
        counts_by_setting,
        metadata["terms"],
        metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )
    return complex(bell_value), {
        "backend": backend,
        "sampler": sampler,
        "job": job,
        "result": result,
        "counts_by_setting": counts_by_setting,
        "circuits": circuits,
        "shots": shots,
    }


def _load_piastq_sampler() -> Any:
    try:
        from cft_piastq import PiastQSampler
    except ImportError as exc:
        raise ImportError(
            "compute_bell_value_from_counts_aqt requires the optional "
            "cft-piastq integration. Install it with: pip install -e .[piastq]"
        ) from exc
    return PiastQSampler
