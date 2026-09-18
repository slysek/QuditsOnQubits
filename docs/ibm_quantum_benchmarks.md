# IBM compilation benchmarks

Use the same encoding benchmark with a named IBM backend:

```python
from qudits_on_qubits.benchmarks import (
    BackendTarget, BenchmarkConfig, ExactSynthesis, SchmidtTheta,
    run_benchmark, two_qutrit_graph_circuit,
)

backend = BackendTarget.ibm("YOUR_BACKEND_NAME")
result = run_benchmark(
    two_qutrit_graph_circuit(),
    families=[SchmidtTheta(points=5)],
    backend=backend,
    config=BenchmarkConfig(transpiler_seeds=(0, 1, 2)),
    synthesis=ExactSynthesis(),
)
result.pareto_front
```

Creating the target reads backend metadata through Qiskit Runtime. The benchmark
then compiles against that frozen target, including native gates and routing.
It does not submit a QPU job. `ExactSynthesis` is a quick control; use
`OptimizedSynthesis` for the optimized gate search shown in the
[benchmark guide](encoding_benchmark.md).

Use a saved Qiskit Runtime account, or configure `QISKIT_IBM_TOKEN`,
`QISKIT_IBM_CHANNEL=ibm_quantum_platform` and optionally `QISKIT_IBM_INSTANCE`
in a local `.env`. Process variables override file values. A named saved account
can be selected with `account_name=...`; it bypasses `.env` credentials.
Never put tokens in notebooks or saved benchmark artifacts.

For offline integration tests, pass a Qiskit backend test double as
`BackendTarget.ibm(name, backend=backend)`. Each target has its own comparison
cohort; Pareto fronts from different devices or synthesis policies are separate.

Bell measurement execution uses the separate
[experiment runner](experiment_runner.md).
