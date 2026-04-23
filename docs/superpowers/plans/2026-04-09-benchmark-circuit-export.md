# Benchmark Circuit Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save each benchmark candidate's pre-transpile circuit as a `.qpy` file under `benchmark_circuits/<class_name>/`.

**Architecture:** Add a small helper responsible for directory creation and `qpy` export, then call it from `benchmark_basis(...)` immediately after circuit construction and before any transpilation trials. Cover the helper and the integration path with focused unittest cases.

**Tech Stack:** Python, Qiskit `qpy`, unittest, tempfile

---

## File Structure

- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Create: `tests/test_benchmark_encoding_bases.py`

### Task 1: Add Failing Export Tests

**Files:**
- Create: `tests/test_benchmark_encoding_bases.py`

- [ ] Write a failing helper test for `_save_benchmark_circuit(...)`.
- [ ] Write a failing integration test for `benchmark_basis(..., circuits_output_dir=...)`.
- [ ] Run `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v` and confirm failure due to missing export functionality.

### Task 2: Implement Helper-Based Circuit Export

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`

- [ ] Add imports needed for filesystem work and `qpy`.
- [ ] Implement `_save_benchmark_circuit(...)`.
- [ ] Extend `benchmark_basis(...)` with optional export configuration.
- [ ] Save the pre-transpile circuit right after `create_ame_circuit(...)` succeeds.
- [ ] Thread the output directory through `run_benchmark(...)`.

### Task 3: Verify Green State

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Test: `tests/test_benchmark_encoding_bases.py`

- [ ] Run `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Confirm the exported `.qpy` files appear under the expected class-name folders during the integration test.
