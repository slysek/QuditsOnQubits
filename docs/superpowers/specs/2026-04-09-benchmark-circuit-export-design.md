# Benchmark Circuit Export Design

**Date:** 2026-04-09

**Scope:** Extend `QuditsOnQubits/benchmark_encoding_bases.py` so each benchmark candidate saves its pre-transpile circuit as a `.qpy` file under `benchmark_circuits/<class_name>/`.

## Context

`benchmark_encoding_bases.py` currently:

- builds a circuit for each encoding candidate,
- transpiles it multiple times,
- writes only aggregated CSV statistics.

The requested change is to also persist the original circuit before transpilation, grouped by the existing `class_name` values.

## Goals

- Save one `.qpy` file per benchmark candidate before transpilation.
- Use `benchmark_circuits/` as the output root.
- Use the exact current `class_name` values as subfolder names.
- Keep the existing CSV/statistics flow unchanged.

## Design

- Add a helper `_save_benchmark_circuit(qc, class_name, candidate_name, output_root="benchmark_circuits")`.
- The helper creates `output_root/<class_name>/` if needed and writes `<candidate_name>.qpy` with `qpy.dump(...)`.
- Extend `benchmark_basis(...)` with an optional `circuits_output_dir` argument.
- After successful `create_ame_circuit(...)` and before the transpilation loop, call the helper when `circuits_output_dir` is not `None`.
- Keep `run_benchmark(...)` defaulting to `benchmark_circuits` so export happens automatically unless explicitly disabled.

## Verification

- Unit test for the helper writing a `.qpy` file into the correct class-specific directory.
- Integration test for `benchmark_basis(...)` saving the pre-transpile circuit when export is enabled.
- Re-run the benchmark-related test file and the existing test suite.
