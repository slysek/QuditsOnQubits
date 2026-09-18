# Contributing

Reproducibility reports, bug fixes, encoding families, backend tests and clearer
examples are welcome. For a substantial change, first describe the problem and
intended behavior in a [GitHub issue](https://github.com/slysek/QuditsOnQubits/issues).

## Local development

Use Python 3.11–3.13 and a virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Tests use local simulation and provider test doubles. Optional private PiastQ
client tests and explicitly opt-in synthesis checks may be skipped. Do not add
tests that contact hardware or need credentials during normal collection or CI.
Generated runs belong in `artifacts/`; deterministic inputs belong in
`experiment_inputs/` or `examples/data/`, with provenance and hashes.

For the unified benchmark, run tests matching `tests/test_benchmark_*.py`
and `tests/test_encoding_benchmark_notebook.py`. The full suite also covers older
Bell execution and direct-basis tools.

## Extend the benchmark

Use the [public contracts](docs/encoding_benchmark.md#where-to-extend):

- A family yields valid `EncodingCandidate` objects and serializable parameters.
- A synthesis policy returns gates with independently checked logical action.
- A backend adapter preserves final layout and reports native compiled costs.
- A workload defines a logical reference independent of candidate synthesis.

Keep the canonical baseline, compiler budget and acceptance thresholds comparable.
Report failures explicitly. Never pool different backend or synthesis protocols
into one Pareto front. Circuit costs do not establish hardware performance.

## Validate a distribution

```bash
python -m build
```

In a fresh virtual environment, install the resulting wheel:

```bash
python -m pip install dist/qudits_on_qubits-0.1.1-py3-none-any.whl
python -m pip check
python scripts/verify_distribution.py --wheel dist/qudits_on_qubits-0.1.1-py3-none-any.whl --sdist dist/qudits_on_qubits-0.1.1.tar.gz
```

The verifier checks package contents and runs both installed entry points outside
the checkout, then verifies their saved evidence. It performs no hardware work.
CI also rebuilds the wheel from the source distribution to catch omitted inputs.

## Pull requests

Describe the behavior changed, relevant tests and scientific limitations. Keep
examples short and in English. Do not commit credentials, personal environment
files or large generated campaigns. Include a small reproducible input for bugs;
for benchmark reports, include versions, target, seeds, tolerances and manifest.
Contributions use the project's [Apache-2.0 license](LICENSE).
