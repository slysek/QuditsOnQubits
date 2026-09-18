# Installation

Use Python 3.11–3.13 in a virtual environment. From the checkout:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

`pyproject.toml` defines package dependencies. `requirements.txt` mirrors the base
dependencies for existing research environments. The supported Qiskit range is
`>=2,<2.2`; the IQM adapter comes from `iqm-client[qiskit]>=35,<36`.
The current installation includes notebook and provider SDK dependencies even for
local use. Accounts are only required when accessing provider services.

Run the [encoding notebook](../notebooks/encoding_benchmark.ipynb) using this
environment's kernel, or `python examples/two_qutrit_encoding_benchmark.py`.
`python -m qudits_on_qubits.benchmarks.cli --help` exposes the same installed
benchmark CLI as `qoq-benchmark`.

## Optional integrations

```bash
python -m pip install -e ".[mitigation]"
```

This adds M3 and IQM error-reduction tools. See [IBM setup](ibm_quantum_benchmarks.md)
and [PiastQ setup](piastq.md) for execution. PiastQ requires a separately obtained
private `cft-piastq` client; public examples and the default test suite do not require it.
Copy `.env.example` to `.env` only when configuring provider access.
`BackendTarget.iqm(...)` finds the nearest `.env` from the working directory
upwards; pass `env_path=...` to select another file.
`OptimizedSynthesis()` stores its validated gate cache under
`~/.cache/qudits_on_qubits/optimized_gates`; use `cache_dir=...` to override it.
The default theta baseline is bundled in both source and wheel installations. Never place
credentials in notebooks, source files or saved results.

## Development and packages

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

The wheel contains the library, command-line entry points and runtime QPY gates.
The source distribution additionally includes examples, portable replay data,
notebooks, documentation and offline tests. A Git checkout is the simplest way to
run the notebook. Distribution validation is described in
[CONTRIBUTING.md](../CONTRIBUTING.md).

If an old environment contains both `qiskit-iqm` and `iqm-client`, create a new
virtual environment and install the project there to avoid overlapping adapters.
