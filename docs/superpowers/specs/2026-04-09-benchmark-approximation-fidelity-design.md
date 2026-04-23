# Benchmark Approximation-Fidelity Sweep Design

**Date:** 2026-04-09

**Scope:** Extend `QuditsOnQubits/benchmark_encoding_bases.py` with a helper-driven approximation sweep based on `generate_preset_pass_manager(..., approximation_degree=...)`, storing the best result for each fidelity threshold in the same CSV row as the existing benchmark statistics.

## Context

The benchmark currently records classical transpilation statistics such as best depth, mean depth, size, and two-qubit gate counts. The new requirement adds an approximation sweep:

- reference circuit `qcmax` is obtained with `approximation_degree=1.0`,
- test circuits are generated for `approximation_degree` values from `0.90` to `0.99`,
- fidelity is computed as `state_fidelity(DensityMatrix(qctest), DensityMatrix(qcmax))`,
- for fidelity thresholds `0.85`, `0.90`, `0.95`, the benchmark should keep the smallest circuit according to:
  1. minimum `depth()`
  2. tie-break by minimum two-qubit gate count.

Because the transpiler can expand the circuit width onto a large coupling map, idle qubits must be removed before constructing density matrices.

## Goals

- Use `generate_preset_pass_manager(..., basis_gates=BASIS_GATES, optimization_level=3, approximation_degree=...)` for the new sweep.
- Keep the current benchmark row shape: one candidate per CSV row.
- Add fidelity-threshold results into the same row.
- Compare all approximation candidates to a single `qcmax` reference built with `approximation_degree=1.0`.
- Remove idle qubits before building density matrices.

## Design

- Add helper `_strip_idle_qubits(qc)` that removes idle wires from a circuit via DAG conversion.
- Add helper `_benchmark_approximation_sweep(...)` that:
  - builds `qcmax`,
  - computes `DensityMatrix` on the reduced active-qubit circuit,
  - sweeps approximation values `0.90..0.99`,
  - computes fidelity against the reference,
  - selects the best candidate per threshold using `(depth, two_qubit_gate_count)` ordering.
- Extend `benchmark_basis(...)` with optional sweep configuration for testability.
- Merge the helper output into the existing `row` dict.

## Output Fields

Reference fields:

- `approx_ref_depth`
- `approx_ref_two_qubit_gate_count`
- `approx_status`
- `approx_error_message`

Per-threshold fields:

- `fid085_best_approx_degree`
- `fid085_best_fidelity`
- `fid085_best_depth`
- `fid085_best_two_qubit_gate_count`

- `fid090_best_approx_degree`
- `fid090_best_fidelity`
- `fid090_best_depth`
- `fid090_best_two_qubit_gate_count`

- `fid095_best_approx_degree`
- `fid095_best_fidelity`
- `fid095_best_depth`
- `fid095_best_two_qubit_gate_count`

## Verification

- Unit test for the approximation helper with a small circuit and controlled sweep values.
- Integration test for `benchmark_basis(...)` confirming the new fields are present in the returned row.
- Full `python -m unittest discover -s tests -v`.
