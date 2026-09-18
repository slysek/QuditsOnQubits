# PiastQ managed Bell execution

### Managed compilation contract and offline validation

`run_experiment(spec)` uses the managed adapter when configured with
`ExperimentSpec(backend=PiastQHardware(mode="managed", owner=...))`.
The adapter validates bound logical `QuantumCircuit` objects and hands them to
`PiastQSampler` in their original order. The client sends one QPY payload per
circuit to `POST /api/runner/jobs`; the dashboard runner loads QPY and invokes
`AQTSampler`, whose default behavior performs hardware transpilation.
`ManagedPiastQBackend` is a transport handle, not a Qiskit compilation target.

Use the default `TranspilationConfig`. Its default optimization level is a
framework placeholder on this path, not a requested runner optimization level.
The dashboard API does not carry local transpiler settings: non-default
optimization levels, seeds, layouts, routing and scheduling options are rejected.
Compilation metadata records `compilation_owner: managed_runner` and
`circuit_representation: logical`. Physical layouts and hardware gate metrics are
not available locally. Local readout mitigation, ZNE and workload optimization
are rejected because they require control over physical compilation.

The managed client supports Qiskit `>=1.4,<2.2`; this project selects `>=2,<2.2`.
Use the updated `cft-piastq` source that emits QPY version 13, readable by the
Qiskit 1.4 runner as well as the 2.1 client. Keep the direct PCSS/AQT stack in its
separate environment. A clean installation must resolve both local projects
together; do not bypass dependency resolution with `--no-deps`:

```bash
python -m venv artifacts/piastq-validation-env
# Activate the new environment using the command appropriate for your shell.
python -m pip install ".[dev]" "/path/to/cft-piastq[dev,fake]"
python -m pip check
python -m pytest -q tests/test_experiment_piastq_adapter.py tests/test_experiment_piastq_managed_integration.py
```

The managed integration tests use the real optional client and an HTTP
`MockTransport`. They verify QPY circuit order, polling, counts, Bell decoding
and the completed artifact without contacting a dashboard or consuming shots.
They skip when `cft-piastq` is absent; a skipped run does not validate the managed
integration. This validates the local contract, not the deployed runner version
or hardware availability.

Install `cft-piastq` separately in the environment used by this project. The
QuditsOnQubits package metadata intentionally contains no private repository URL
and does not install `cft-piastq`.

The QuditsOnQubits integration is managed-only. `PiastQHardware` accepts
`mode="managed"`; `auto` and `direct` are rejected before any provider import or
network action. Configure `CFT_PIASTQ_DASHBOARD_API_URL`,
`CFT_PIASTQ_DASHBOARD_API_KEY`, and optionally `CFT_PIASTQ_OWNER` through the
environment. Do not place dashboard credentials in notebooks or source files.

Direct PCSS/AQT experiments require a separate environment and are not installed
by QuditsOnQubits.

This explicit smoke example prepares the zero state in the two-qutrit encoding,
builds every Bell-setting circuit required by the existing pipeline, and
submits one PiastQ job containing every generated circuit:

```python
import os

from qiskit import QuantumCircuit

from cft_piastq import PiastQClient
from qudits_on_qubits.bell_measurements import (
    build_sampler_circuits_for_candidate,
    canonical_Ez,
    compute_bell_value_from_counts_aqt,
)

state_circuit = QuantumCircuit(4)
sampler_circuits, metadata = build_sampler_circuits_for_candidate(
    candidate="two_qutrit",
    state_circuit=state_circuit,
    E=canonical_Ez(),
    qutrit_qubits=((0, 1), (2, 3)),
)

client = PiastQClient(
    mode="managed",
    owner=os.environ["CFT_PIASTQ_OWNER"],
    dashboard_api_url=os.environ["CFT_PIASTQ_DASHBOARD_API_URL"],
    dashboard_api_key=os.environ["CFT_PIASTQ_DASHBOARD_API_KEY"],
)

bell_value, execution = compute_bell_value_from_counts_aqt(
    sampler_circuits,
    metadata,
    backend=client.backend,
    shots=20_480,
    sampler_options={"cft_job_name": "two-qutrit-bell-smoke"},
    timeout=900.0,
    poll_interval=5.0,
)

print("Bell value:", bell_value)
print("PiastQ job:", execution["job"].job_id())
```

`job.result()` remains available in `execution["result"]` as a Qiskit
`SamplerResult`. Bell postprocessing uses the estimated integer dictionaries
returned by `PiastQJob.counts()`; this project does not independently multiply
or round the quasi probabilities.

The example contacts the managed dashboard and can consume real hardware shots.
Run it only as an intentional manual smoke test.
