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
    options = dict(run_options or {})
    settings = _validate_local_inputs(
        circuits,
        metadata,
        shots=shots,
        poll_interval=poll_interval,
        run_options=options,
    )
    sampler_type = _load_piastq_sampler()
    sampler = sampler_type(backend, options=dict(sampler_options or {}))
    job = sampler.run(circuits, shots=shots, **options)
    result = job.result(timeout=timeout, poll_interval=poll_interval)
    counts = list(job.counts())
    if len(counts) != len(circuits):
        raise ValueError(
            f"expected {len(circuits)} count dictionaries, received {len(counts)}"
        )
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


def _validate_local_inputs(
    circuits: Any,
    metadata: Any,
    *,
    shots: object,
    poll_interval: object,
    run_options: Any,
) -> list[tuple[object, ...]]:
    if not circuits:
        raise ValueError("sampler_circuits must contain at least one circuit")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, (int, float))
        or poll_interval <= 0
    ):
        raise ValueError("poll_interval must be a positive number")
    if "shots" in run_options:
        raise ValueError("pass shots via the shots argument, not run_options")
    if "setting_by_circuit_index" not in metadata:
        raise ValueError("metadata must include setting_by_circuit_index")

    settings = [
        tuple(setting) for setting in metadata["setting_by_circuit_index"]
    ]
    if len(circuits) != len(settings):
        raise ValueError(
            "number of sampler_circuits must match metadata settings"
        )
    if len(set(settings)) != len(settings):
        raise ValueError("metadata settings must be unique")
    return settings


def _load_piastq_sampler() -> Any:
    try:
        from cft_piastq import PiastQSampler
    except ImportError as exc:
        raise ImportError(
            "compute_bell_value_from_counts_aqt requires the optional "
            "cft-piastq integration. Install it with: pip install -e .[piastq]"
        ) from exc
    return PiastQSampler
