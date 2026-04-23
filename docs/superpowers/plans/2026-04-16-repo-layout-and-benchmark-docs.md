# Repo Layout And Benchmark Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize benchmark artifacts into clearer folders and add a readable Markdown summary for the full benchmark results without breaking existing code paths.

**Architecture:** Introduce package-level path helpers, point benchmark outputs to repository-relative data directories, and relocate documentation/results into `docs/benchmarks` and `data/benchmarks`. Keep runtime-sensitive assets in place unless they can be moved safely.

**Tech Stack:** Python, unittest, Markdown, PowerShell file moves

---

### Task 1: Add regression tests for repository-relative path handling

**Files:**
- Create: `tests/test_repo_layout.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import tempfile
import unittest

import QuditsOnQubits.benchmark_encoding_bases as benchmark_encoding_bases
from QuditsOnQubits.create_ame_circuit import create_ame_circuit


class RepoLayoutTests(unittest.TestCase):
    def test_create_ame_circuit_works_outside_repo_root(self):
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                try:
                    qc, graph = create_ame_circuit(n=2, dim=3, graph_type="star")
                except FileNotFoundError as exc:
                    self.fail(f"create_ame_circuit should not depend on cwd: {exc}")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(graph.vcount(), 2)

    def test_benchmark_default_paths_live_under_data_benchmarks(self):
        results_path_fn = getattr(benchmark_encoding_bases, "benchmark_results_path", None)
        circuits_dir_fn = getattr(benchmark_encoding_bases, "benchmark_circuits_dir", None)

        self.assertTrue(callable(results_path_fn))
        self.assertTrue(callable(circuits_dir_fn))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repo_layout.py -v`
Expected: FAIL because `create_ame_circuit` still depends on cwd and the benchmark path helper functions do not exist yet.

### Task 2: Implement path helpers and update runtime code

**Files:**
- Create: `QuditsOnQubits/project_paths.py`
- Modify: `QuditsOnQubits/create_ame_circuit.py`
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`

- [ ] **Step 1: Add repository-relative path helpers**

```python
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parent.parent
```

- [ ] **Step 2: Use those helpers in circuit-loading code**

```python
with open(quantum_circuits_path("Fgate3.qpy"), "rb") as fd:
    Fgate3 = qpy.load(fd)[0]
```

- [ ] **Step 3: Point benchmark defaults to `data/benchmarks`**

```python
csv_path = benchmark_results_path(mode)
circuits_output_dir = benchmark_circuits_dir()
```

- [ ] **Step 4: Run the repo-layout tests**

Run: `python -m pytest tests/test_repo_layout.py -v`
Expected: PASS

### Task 3: Move benchmark files and write the overview doc

**Files:**
- Create: `docs/benchmarks/benchmark_encoding_bases_full_results_overview.md`
- Move: `benchmark_encoding_bases_results.csv` -> `data/benchmarks/benchmark_encoding_bases_results.csv`
- Move: `benchmark_encoding_bases_extended_results.csv` -> `data/benchmarks/benchmark_encoding_bases_extended_results.csv`
- Move: `benchmark_encoding_bases_full_results.csv` -> `data/benchmarks/benchmark_encoding_bases_full_results.csv`
- Move: `benchmark_-full_results.csv` -> `data/benchmarks/benchmark_-full_results.csv`
- Move: `benchmark_encoding_bases_extended_results.md` -> `docs/benchmarks/benchmark_encoding_bases_extended_results.md`
- Move: `benchmark_encoding_bases_full_results_analysis.md` -> `docs/benchmarks/benchmark_encoding_bases_full_results_analysis.md`
- Move: `benchmark_circuits/` -> `data/benchmarks/circuits/`

- [ ] **Step 1: Move benchmark artifacts into their new folders**

Run:

```powershell
New-Item -ItemType Directory -Force -Path data\benchmarks, docs\benchmarks | Out-Null
```

- [ ] **Step 2: Write the benchmark overview Markdown file**

```md
# Benchmark Encoding Bases Full Results
```

- [ ] **Step 3: Re-run focused tests after the moves**

Run: `python -m pytest tests/test_repo_layout.py tests/test_create_ame_circuit.py tests/test_benchmark_encoding_bases.py -v`
Expected: PASS or explicit report of any pre-existing failures unrelated to the move.
