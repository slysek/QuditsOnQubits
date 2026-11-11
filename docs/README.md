# Documentation

Start with the [state preparation harness](harness.md), its
[published results](../benchmarks/state_preparation/README.md), and
[saved circuits](../quantum_circuits/README.md).

## User guides

- [State preparation harness](harness.md): searches, encodings, agents, AME(4,6), routing, and resume.
- [Installation](installation.md): environments, dependency bounds and optional integrations.
- [Encoding benchmark](encoding_benchmark.md): families, backend compilation, validation and Pareto results.
- [Experiment runner](experiment_runner.md): Bell measurements, execution, uncertainty and saved results.
- [Two-qutrit Bell example](two_qutrit_bell_vertical_slice.md): a complete local Aer run.
- [IBM Quantum](ibm_quantum_benchmarks.md): account setup, compilation and explicit hardware execution.
- [PiastQ](piastq.md): optional managed execution.
- [IQM state preservation](iqm_state_preservation.md): compilation requirements.

## Research and extension

- [Current state preparation results](../benchmarks/state_preparation/README.md): checked comparisons and source files.
- [Harness research log in Polish](pipeline-ewolucji-stanow.md): development history and earlier campaigns.
- [Roadmap](roadmap.md): current scope and proposed next milestones.
- [Contributing](../CONTRIBUTING.md): tests and extension points.
- [Benchmark specification](unified-benchmark-spec.md): comparison and artifact contracts.
- [Theta continuation](theta_benchmark.md) and [hardware layer](theta-hardware-layer.md).
- [Legacy direct-basis tools](legacy_benchmarks.md): previous candidate scans and transpiler harnesses.
- [Frozen Bell hardware results](../benchmarks/bell_20260907/README.md): separate historical measurements and offline replay.

The unified benchmark ranks compiled preparation circuits. Bell execution has a
separate runner and artifact schema. The research notebooks under
[`notebooks/working/`](../notebooks/working/) may require provider access or saved inputs.
