# AQT Bell Pipeline via cft-piastq Design

**Date:** 2026-07-10

**Status:** Approved

## Context

The Bell-measurement pipeline already builds one measured circuit per Bell
setting and evaluates a Bell expression from `counts_by_setting`. The existing
postprocessing accepts integer counts, while `cft-piastq` exposes Qiskit's
`SamplerResult` through `PiastQJob.result()` and an estimated integer count view
through `PiastQJob.counts()`.

The integration must allow callers to select `auto`, `managed`, or `direct`
execution when they construct `PiastQClient`. The Bell pipeline receives the
resulting `client.backend` and must not duplicate or override that mode choice.

## Goals

- Add an AQT/PiastQ Bell helper that accepts a backend returned by
  `PiastQClient`.
- Submit all Bell-setting circuits in one `PiastQSampler.run(...)` call.
- Use the public `PiastQJob.counts()` contract to convert quasi distributions
  into estimated integer counts.
- Preserve circuit/result order when mapping counts to Bell settings.
- Reuse the existing `compute_bell_value_from_counts` implementation for all
  Bell mathematics and qutrit decoding.
- Return both the Bell value and inspectable execution objects.
- Keep `cft-piastq` optional for users who do not run the PiastQ path.

## Non-goals

- Do not modify `cft-piastq` or reimplement its quasi-distribution conversion.
- Do not select between `auto`, `managed`, and `direct` inside this project.
- Do not change IQM, Aer, or generic backend execution behavior.
- Do not add an automatic real-hardware test to the normal test suite.
- Do not expose dashboard keys or PCSS tokens in function arguments, execution
  metadata, exceptions, documentation examples, or logs.

## Architecture

Create a focused PiastQ adapter module:

`src/qudits_on_qubits/bell_measurements/piastq_runner.py`

It exposes one public convenience function:

```python
def compute_bell_value_from_counts_aqt(
    sampler_circuits: Sequence[QuantumCircuit],
    metadata: Mapping[str, Any],
    *,
    backend: Any,
    shots: int = 1024,
    sampler_options: Mapping[str, Any] | None = None,
    run_options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> tuple[complex, dict[str, Any]]:
    ...
```

The execution path is:

```text
PiastQClient(mode=...) -> client.backend
    -> PiastQSampler(backend, options=sampler_options)
    -> sampler.run(all_circuits, shots=shots, **run_options)
    -> one PiastQJob
    -> job.result(timeout=timeout, poll_interval=poll_interval)
    -> job.counts()
    -> counts_by_setting
    -> compute_bell_value_from_counts(...)
```

The backend is opaque to the helper and is passed unchanged to
`PiastQSampler`. Consequently, `auto`, `managed`, and `direct` use the same Bell
pipeline code. The batching contract mirrors the IQM pipeline: accept the full
ordered circuit list produced upstream, submit it once, and preserve that order
when associating one result with each metadata setting.

The module imports `PiastQSampler` lazily inside the public function. Importing
`qudits_on_qubits.bell_measurements` therefore remains possible without
installing `cft-piastq`.

## Data Contract and Ordering

For a Bell measurement with `N` setting circuits:

1. `sampler.run(...)` receives the ordered list of `N` circuits once.
2. The call returns one `PiastQJob`, not `N` jobs.
3. `job.counts()` returns a list of `N` dictionaries in submitted circuit
   order.
4. `metadata["setting_by_circuit_index"]` contains `N` settings in the same
   order.
5. The adapter constructs the mapping with strict positional pairing:

```python
counts_by_setting = {
    tuple(setting): counts
    for setting, counts in zip(
        metadata["setting_by_circuit_index"],
        job.counts(),
        strict=True,
    )
}
```

`N` is determined entirely by the circuits produced by the existing Bell
pipeline. The adapter must not encode a candidate-specific circuit count. It
submits the complete incoming list in one job and requires `job.counts()` to
return exactly `N` dictionaries, each mapped to the setting at the same index.

The adapter does not multiply quasi probabilities itself. `cft-piastq`
already calls `binary_probabilities(num_bits=...)`, multiplies each probability
by the configured shot count, rounds to an integer, and floors negative
estimates at zero. That library remains the single owner of count-estimation
semantics.

## Bell Evaluation

After building `counts_by_setting`, the adapter delegates to the existing
function:

```python
bell_value = compute_bell_value_from_counts(
    counts_by_setting,
    metadata["terms"],
    metadata["qutrit_bit_indices_by_setting"],
    **decoding_kwargs_from_metadata(metadata),
)
```

This preserves the current qutrit outcome map, bit order, leakage handling, and
Bell-term coefficients. There is no AQT-specific Bell formula.

The helper returns:

```python
return bell_value, {
    "backend": backend,
    "sampler": sampler,
    "job": job,
    "result": sampler_result,
    "counts_by_setting": counts_by_setting,
    "circuits": circuits,
    "shots": shots,
}
```

The result object is retained for quasi-distribution and metadata inspection;
the Bell calculation uses only the estimated count view.

## Validation and Errors

The adapter validates local input before submission:

- `sampler_circuits` must contain at least one circuit.
- `shots` must be a positive integer and booleans are rejected.
- `poll_interval` must be positive.
- `run_options` must not contain `shots`; the explicit `shots` argument is the
  only shot source.
- The number of circuits must equal the number of
  `metadata["setting_by_circuit_index"]` entries.
- Settings must be unique after conversion to tuples.

After execution, the adapter validates that `job.counts()` returned one count
dictionary per submitted circuit. A mismatch raises `ValueError` containing
the expected and received lengths before any Bell computation occurs.

Authentication, dashboard, timeout, cancellation, and provider exceptions from
`cft-piastq` propagate unchanged. Preserving the library's exception types lets
callers distinguish configuration failures from remote execution failures.

If the optional package is missing, the lazy import raises an `ImportError`
that instructs the caller to install the project's `piastq` extra.

## Optional Dependency

Add a project extra rather than a mandatory dependency:

```toml
[project.optional-dependencies]
piastq = [
    "cft-piastq[direct]>=0.1,<0.2",
]
```

The `direct` extra covers the PCSS/AQT packages needed when `PiastQClient`
selects direct execution, including an `auto` fallback. Managed-only execution
still uses the same installed package and public API.

For development against the local library checkout, install
`C:\Users\szymo\cft-piastq` as an editable package in the same environment
before installing or running this project's PiastQ integration tests. The
integration contract is `cft-piastq` 0.1 exposing `PiastQClient`,
`PiastQSampler`, `PiastQJob.result()`, and `PiastQJob.counts()`.

## Public Export and Documentation

Export `compute_bell_value_from_counts_aqt` from
`qudits_on_qubits.bell_measurements`.

Add a README example that:

- constructs `PiastQClient(mode="auto" | "managed" | "direct")` outside the
  helper;
- passes `client.backend` into the helper;
- loads all credentials from environment variables;
- supplies `cft_job_name` through `sampler_options`;
- shows the returned Bell value and job identifier without printing secrets.

## Testing

Add `tests/test_bell_measurements_piastq_runner.py`.

Unit tests use recording fake sampler and job objects and make no network calls.
They verify:

- the caller's backend object is passed unchanged to `PiastQSampler`;
- sampler options, run options, shots, timeout, and poll interval are forwarded;
- all circuits are submitted in one call and one job is produced;
- multiple circuit-list lengths map to the same number of settings in index
  order, without a candidate-specific count assumption;
- the implementation calls `job.counts()` and does not convert
  `SamplerResult.quasi_dists` itself;
- the existing `compute_bell_value_from_counts` function receives the mapped
  counts and produces the final value;
- the returned execution dictionary contains the agreed objects and values;
- empty circuits, invalid shots, duplicate settings, a `shots` run option, and
  mismatched circuit/setting/result lengths fail with precise messages;
- a missing optional dependency produces the documented installation error;
- opaque fake backends representing `auto`, `managed`, and `direct` follow the
  same adapter path.

An optional local integration test uses a real `PiastQClient(mode="fake")` and
`PiastQSampler` to verify the `PiastQJob.counts()` contract without a dashboard
or AQT hardware. It is skipped when the `piastq` extra is unavailable.

The normal automated suite never submits a remote job. The README provides an
explicit manual smoke procedure using `mode="auto"` and credentials from the
environment. Running that smoke procedure is a deliberate user action because
it can contact the dashboard or direct provider and consume hardware shots.

## Baseline Note

Before this design document was added, the isolated worktree ran 158 tests:
157 passed and the unrelated IQM fidelity-preservation test failed because the
active circuit had five qubits while the reference had four. PiastQ/Bell work
must use targeted tests so this existing IQM failure is not confused with an
AQT regression.
