# Documentation

Start with the [encoding benchmark](encoding_benchmark.md) and its
[short notebook](../notebooks/encoding_benchmark.ipynb).

## User guides

- [Installation](installation.md): environments, dependency bounds and optional integrations.
- [Encoding benchmark](encoding_benchmark.md): families, backend compilation, validation and Pareto results.
- [Experiment runner](experiment_runner.md): Bell measurements, execution, uncertainty and saved results.
- [Two-qutrit Bell example](two_qutrit_bell_vertical_slice.md): a complete local Aer run.
- [IBM Quantum](ibm_quantum_benchmarks.md): account setup, compilation and explicit hardware execution.
- [PiastQ](piastq.md): optional managed execution.
- [IQM state preservation](iqm_state_preservation.md): compilation requirements.

## Research and extension

- [Roadmap](roadmap.md): current scope and proposed next milestones.
- [Contributing](../CONTRIBUTING.md): tests and extension points.
- [Benchmark specification](unified-benchmark-spec.md): comparison and artifact contracts.
- [Theta continuation](theta_benchmark.md) and [hardware layer](theta-hardware-layer.md).
- [Legacy direct-basis tools](legacy_benchmarks.md): previous candidate scans and transpiler harnesses.
- [Frozen Bell hardware results](../benchmarks/bell_20260907/README.md): separate historical measurements and offline replay.

The unified benchmark ranks compiled preparation circuits. Bell execution has a
separate runner and artifact schema. The research notebooks under
[`notebooks/working/`](../notebooks/working/) may require provider access or saved inputs.
