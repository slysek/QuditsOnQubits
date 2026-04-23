# Benchmark Approximation-Fidelity Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a helper-based approximation-degree sweep to the benchmark and record best fidelity-threshold matches in the same CSV row as each benchmark candidate.

**Architecture:** Keep the existing benchmark statistics intact, but attach a second helper-driven analysis stage that uses `generate_preset_pass_manager(...)`, strips idle qubits for state construction, and stores the best result per fidelity threshold according to `(depth, two_qubit_gate_count)`.

**Tech Stack:** Python, Qiskit preset pass managers, DensityMatrix, state_fidelity, unittest

---

## File Structure

- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Modify: `tests/test_benchmark_encoding_bases.py`

### Task 1: Add Failing Tests For Approximation Sweep

**Files:**
- Modify: `tests/test_benchmark_encoding_bases.py`

- [ ] Add a focused helper test for the approximation sweep.
- [ ] Add an integration-style benchmark row test covering the new CSV fields.
- [ ] Run the targeted benchmark test file and confirm failure before implementation.

### Task 2: Implement Helper-Based Sweep

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`

- [ ] Add helper(s) for idle-qubit stripping and approximation result defaults.
- [ ] Add `_benchmark_approximation_sweep(...)`.
- [ ] Extend `benchmark_basis(...)` with sweep configuration and row merging.
- [ ] Preserve existing benchmark statistics and circuit export behavior.

### Task 3: Verify Green State

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Modify: `tests/test_benchmark_encoding_bases.py`

- [ ] Run `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Confirm the new approximation fields are present and populated as expected.
