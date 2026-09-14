# QuditsOnQubits

**Hardware-aware compilation and benchmarking of qudit circuits encoded on qubit quantum processors.**

[![CI](https://github.com/slysek/QuditsOnQubits/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/slysek/QuditsOnQubits/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.13-blue)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

QuditsOnQubits is a Python research library for studying how higher-dimensional quantum systems can be implemented on qubit hardware. It combines qudit-to-qubit encoding search, code-space-aware gate synthesis, hardware-aware compilation, and experimental analysis.

The central question is not only **how to encode a qudit**, but **which encoding and circuit implementation work best for a given workload and device**.

The current end-to-end reference experiments focus on qutrit graph states and Bell correlations. They provide concrete workloads for developing and evaluating a broader qudit-on-qubit toolchain.

[Quick start](#quick-start) · [Encoding and compilation](#encoding-and-compilation) · [Hardware benchmarks](#hardware-benchmarks-and-reproducibility) · [Documentation](#documentation)

## Why QuditsOnQubits?

An encoding specifies how logical qudit states occupy a qubit register. Different encodings of the same logical experiment can lead to different gate decompositions, routing requirements, and physical circuit costs.

QuditsOnQubits makes these choices part of the experiment rather than fixing a single representation throughout the workflow:

```text
Logical experiment
    → qudit-to-qubit encoding
    → code-space-aware gate synthesis
    → target-aware compilation and selection
    → simulator or QPU execution
    → logical outcomes, leakage diagnostics, and analysis
```

The library supports comparisons between canonical and alternative encodings, records compilation metrics, and keeps experimental results available for offline analysis. Its purpose is to investigate these trade-offs—not to assume that a smaller circuit always produces a better hardware result.

## Main capabilities

- **Encoding search:** generate and compare qutrit encoding candidates, including monomial encodings and dense local basis changes.
- **Code-space-aware synthesis:** optimize the encoded qutrit Fourier gate, $F_3$, and two-qutrit controlled-phase gate, $CZ_3$, without unnecessarily fixing their action outside the logical subspace.
- **Hardware-aware compilation:** compare transpiler strategies, seeds, routing choices, and complete measurement workloads; use calibration-aware layout candidates on supported IQM paths.
- **Reference experiments:** construct and analyze two-qutrit, three-qutrit GHZ-type, and four-qutrit AME Bell workloads.
- **Experimental analysis:** report conditional and unconditional Bell estimates, invalid-codeword statistics, bootstrap uncertainty, and optional readout mitigation and zero-noise extrapolation on supported execution paths.
- **Reproducible research:** inspect saved QPU counts, circuit files, metadata, and archived benchmark results without submitting new hardware jobs.

## Quick start

### Install from source

Use **Python 3.11, 3.12, or 3.13**. From a terminal:

```bash
git clone https://github.com/slysek/QuditsOnQubits.git
cd QuditsOnQubits
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For optional readout-mitigation dependencies, use `python -m pip install ".[mitigation]"`. See [Development](#development) for an editable installation.

### Run a two-qutrit Bell experiment

```bash
qoq-two-qutrit-bell --shots 2048 --seed 42
```

This command runs locally on **ideal Qiskit Aer**. It requires no provider credentials and submits no QPU jobs.

It prepares the logical two-qutrit reference in the `canonical_ez` encoding, generates nine measured four-qubit circuits, executes them, and decodes the counts into logical outcomes. The output includes conditional and unconditional Bell estimates, the leakage rate, and the path to the run manifest.

For this reference and its normalization, the ideal Bell value is **6**. Finite-shot estimates fluctuate around that value; the ideal canonical run has zero code-space leakage. Results are written under `artifacts/vertical_slice_runs/` by default. Change the destination with `--output-root`.

See the [step-by-step guide](docs/two_qutrit_bell_vertical_slice.md) or open the [annotated notebook](notebooks/two_qutrit_bell_vertical_slice.ipynb).

### Inspect a reference from Python

```python
from qudits_on_qubits import get_encoding, get_reference_experiment

reference = get_reference_experiment("two_qutrit")
encoding = get_encoding(reference.default_encoding_id)

print("Logical state shape:", reference.state.statevector().shape)
print("Encoding matrix shape:", encoding.as_array().shape)
print("Ideal Bell value:", reference.expected.ideal_bell_value)
print("Classical bound:", reference.bell_functional.classical_bound)
print("Reference hash:", reference.stable_hash())
```

This inspects the logical specification without executing an experiment. The state vector has shape `(9,)`, and the single-qutrit encoding matrix has shape `(4, 3)`.

## Encoding and compilation

### The encoding is an optimization variable

A qutrit occupies a three-dimensional subspace of a two-qubit register. The canonical encoding uses three computational basis states; alternative encodings change the representation of the logical states and gates.

The direct-basis benchmarks compare candidate encodings through their compiled circuits. They support repeated transpilation, fidelity-filtered approximate variants, Top-K selection, and downstream analysis of circuit cost and state equivalence.

Explore the available benchmark options with:

```bash
python scripts/run_direct_basis_benchmarks.py --help
```

The [IQM transpiler harness](scripts/run_iqm_transpiler_harness.py) compares compilation strategies and produces per-trial metrics, Pareto rankings, and circuit recommendations. The harness compiles circuits; it does not submit QPU jobs. Loading a hardware target or calibration data may still require provider access.

### Optimize the logical action, not an arbitrary full-space extension

Encoding a logical gate does not uniquely determine its action on the unused part of the qubit Hilbert space. Fixing that action to the identity can impose an unnecessary synthesis constraint.

The [optimized gate library](src/qudits_on_qubits/benchmarks/direct_basis/optimized_gates.py) uses this freedom explicitly:

**$F_3$:** the optimizer chooses the phase on the unused one-dimensional subspace. It uses an analytic phase for monomial encodings and a numerical invariant-based solve for other supported bases. Accepted isolated $F_3$ blocks contain at most **two CNOTs**, before hardware routing or whole-circuit optimization.

**$CZ_3$:** BQSKit synthesis constrains the action on the nine-dimensional two-qutrit code space while leaving the seven-dimensional orthogonal complement unconstrained.

Synthesized gates are checked for logical-action error and leakage out of the code space. Validation uses a single global phase across all logical basis states, so it does not discard relative-phase errors. Accepted gates are cached for reuse across benchmark runs.

These checks are numerical acceptance criteria, not a guarantee of globally optimal gate counts. In particular, the $CZ_3$ synthesis is a numerical search, and synthesizing a new encoding may be expensive.

## Reference workloads

| Reference ID | Logical system | Qubits for the encoding | Purpose |
| --- | --- | ---: | --- |
| `two_qutrit` | Two qutrits | 4 | Minimal end-to-end Bell reference |
| `ghz3` | Three-qutrit GHZ-type graph state | 6 | Multipartite Bell workload |
| `ame43` | Four-qutrit AME graph state | 8 | Higher-order entanglement and Bell workload |

The register sizes above describe the logical encoding, not the width of a circuit represented on a full hardware backend. Reference specifications define the state, measurement settings, Bell functional, expected ideal value, and leakage policy.

The [randomized-settings notebook](notebooks/bell_randomized_raw.ipynb) also demonstrates independently sampled local measurement settings and block-based acquisition. Its hardware examples are disabled by default.

## Execution backends

| Backend or path | Scope |
| --- | --- |
| **Qiskit Aer** | Local ideal execution and supported noise-model simulations |
| **IQM** | Hardware execution, target-aware compilation, and calibration-aware layout selection on supported paths |
| **IBM Quantum** | Raw `SamplerV2` execution through the IBM adapter, randomized-block workflows, and archived hardware benchmarks |
| **PiastQ** | Managed execution through the separately installed `cft-piastq` client |
| **Custom backends** | User-provided backend objects with an explicit execution mode |

Backend capabilities are not interchangeable. For example, IQM automatic layout selection is IQM-specific, and PiastQ managed execution delegates compilation to its runner rather than exposing local physical-layout control. Mitigation and resumption support also depend on the execution path.

Configure credentials through environment variables or provider tooling, never in committed source files or notebook outputs. Real hardware examples require an intentional opt-in and an appropriate shot budget.

## Hardware benchmarks and reproducibility

The repository includes an [archived IBM/IQM Bell benchmark](benchmarks/bell_20260907/README.md), including a Fez/Garnet repeat with **5,000 shots per setting**. The archive contains measured counts, submitted QPY circuits, calibration information, reports, and a SHA-256 integrity manifest.

Check archive integrity from the repository root:

```bash
python benchmarks/bell_20260907/reproduce.py --verify-only
```

This check requires only the Python standard library. For numerical replay and bootstrap recomputation, follow the [benchmark's installation and reproduction instructions](benchmarks/bell_20260907/README.md), including its dedicated dependency environment. After installation, replay uses saved data and requires no IBM/IQM credentials or QPU credits.

Additional IQM Bell and ZNE experiments are documented in the [IQM experiment guide](notebooks/working/iqm/README_bell_zne.md).

Reproducing an archived analysis is different from reproducing a hardware outcome: a new experiment uses new samples and device calibrations. The archive retains weak results as well as stronger ones; it is not a claim of uniform improvement across devices.

### Interpreting results

In the encoding and synthesis workflow, *code-space leakage* means population outside the chosen logical subspace within the qubit register. It should not be confused with physical device leakage into levels outside a qubit's computational Hilbert space.

Conditional estimates use postselected data and must be interpreted alongside unconditional estimates and rejected-shot statistics. Bootstrap intervals quantify sampling uncertainty under their stated assumptions; they do not account for all hardware drift or ZNE model bias. The randomized-block path uses its own block-based concentration bounds rather than the same bootstrap procedure.

These experiments are research benchmarks, **not loophole-free Bell tests**. Neither a postselected violation nor a mitigated estimate alone establishes device-independent nonlocality.

## Documentation

| Start here | Contents |
| --- | --- |
| [Usage guide](docs/usage_guide.md) | Experiment runner, artifacts and resume, IQM layouts, Top-K benchmarks, optimized gates, PiastQ, randomized blocks, and mitigation |
| [Two-qutrit guide](docs/two_qutrit_bell_vertical_slice.md) | Expected results, artifacts, clean-install verification, and troubleshooting |
| [Annotated example](notebooks/two_qutrit_bell_vertical_slice.ipynb) | Logical specification → encoding → execution → analysis |
| [Randomized local settings](notebooks/bell_randomized_raw.ipynb) | Block-based acquisition and reference comparisons |
| [Hardware benchmark](benchmarks/bell_20260907/README.md) | Archived data, methods, limitations, and offline replay |
| [IQM Bell/ZNE experiments](notebooks/working/iqm/README_bell_zne.md) | Additional hardware campaigns and their analysis |

Deterministic circuit inputs live in [`experiment_inputs/`](experiment_inputs/README.md). The [`artifacts/` guide](artifacts/README.md) describes result storage. Notebooks under `notebooks/working/` are research workflows rather than a uniform beginner tutorial collection.

## Project status

QuditsOnQubits is **early-stage research software**. The current end-to-end reference workloads and optimized gate pipeline center on qutrits. Generalizing the toolchain to broader qudit circuits and dimensions is a development direction, not a claim that every dimension or circuit is already supported.

Public APIs and artifact formats may evolve. Record the exact release or commit, dependency versions, encoding, compilation settings, and backend configuration used in a study.

## Development

From the repository root, install an editable development environment and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
git diff --check
```

Some optional integration tests require additional dependencies. A skipped integration test does not validate that integration. Use the [clean-install guide](docs/two_qutrit_bell_vertical_slice.md) when checking distribution builds rather than relying only on an editable installation.

Bug reports, reproducible examples, documentation improvements, and new benchmark workloads are welcome through [GitHub issues](https://github.com/slysek/QuditsOnQubits/issues) and pull requests. For numerical or hardware issues, include the software versions, encoding, experiment configuration, and sanitized diagnostics.

## Citing the project

When using QuditsOnQubits in research, cite the [repository](https://github.com/slysek/QuditsOnQubits) and the exact release or commit used. For archived experiments, also identify the benchmark and its documented methodology.

## License

QuditsOnQubits is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).
