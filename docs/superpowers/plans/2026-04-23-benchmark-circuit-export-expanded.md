# Benchmark Circuit Export Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save each benchmark candidate's pre-transpile circuit, up to three best transpiled circuits, and the encoding-change `W` circuit in the same class output directory.

**Architecture:** Keep the current state-aware output root resolution, but expand circuit export into a small helper layer that can write named `.qpy` files for one candidate. `benchmark_basis(...)` will still build the raw circuit once, run the transpilation trials, rank successful trials by the same metrics already used for `best_*`, and then export the raw circuit plus the top three transpiled circuits. When `E_new` is present, build a standalone 2-qubit `W` circuit and export it next to the benchmark circuits.

**Tech Stack:** Python, Qiskit `qpy`, unittest, tempfile

---

## File Structure

- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Modify: `tests/test_benchmark_encoding_bases.py`

### Task 1: Add Failing Export Layout Tests

**Files:**
- Modify: `tests/test_benchmark_encoding_bases.py`

- [ ] Add a failing integration test asserting that `benchmark_basis(..., circuits_output_dir=...)` writes the raw pre-transpile circuit plus the best transpiled outputs under the state/class directory.
- [ ] Add a failing integration test asserting that a non-baseline encoding also writes a standalone `W` circuit in that same directory.
- [ ] Run `python -m unittest tests.test_benchmark_encoding_bases.TestBenchmarkBasisStateAware -v` and confirm failure from the missing expanded export layout.

### Task 2: Implement Expanded Circuit Export

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`

- [ ] Add helper(s) that write named `.qpy` files for a candidate without changing the resolved output-root behavior.
- [ ] Keep exporting the raw circuit before transpilation, but use an explicit filename that distinguishes it from the transpiled outputs.
- [ ] Collect every successful transpiled trial together with its ranking tuple and circuit object, sort them by `(depth, two_qubit_gate_count, size)`, and export the best three.
- [ ] Build and export a standalone 2-qubit `W` circuit when `E_new` is provided.
- [ ] Preserve the existing benchmark result fields and ranking semantics.

### Task 3: Verify Green State

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Modify: `tests/test_benchmark_encoding_bases.py`

- [ ] Run `python -m unittest tests.test_benchmark_encoding_bases.TestBenchmarkBasisStateAware -v`.
- [ ] Run `python -m unittest tests.test_benchmark_encoding_bases -v`.
- [ ] Review the working diff to confirm only the plan, benchmark export logic, and targeted tests changed.
