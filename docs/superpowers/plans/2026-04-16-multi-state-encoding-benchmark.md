# Multi-State Encoding Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the encoding benchmark so it can benchmark `two_qutrit`, reuse existing `ghz3`, benchmark `ame43`, and generate one combined markdown report with fidelity-dependent depth tables for all three states.

**Architecture:** Keep the existing candidate-generation and per-candidate transpilation logic, but add a small state-selection layer that knows how to build the right circuit and where to save outputs. Extract the current `run_benchmark(...)` body into a single-state helper, then let `run_benchmark(...)` either run one state or orchestrate the three-state workflow and emit a combined markdown report.

**Tech Stack:** Python, pandas, Qiskit `qpy`, igraph `Graph`, unittest, unittest.mock

---

## File Structure

- Modify: `QuditsOnQubits/project_paths.py`
  Responsibility: centralize state-aware result/report/circuit output paths while preserving the existing GHZ CSV location.
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
  Responsibility: build state-specific circuits, store state-specific outputs, orchestrate one or three experiments, and generate the combined markdown report.
- Modify: `tests/test_benchmark_encoding_bases.py`
  Responsibility: cover state-specific circuit export, state-specific path helpers, combined-report markdown generation, and the `run_benchmark(..., state_name="all")` orchestration path.

### Task 1: Add Failing Tests For State-Specific Benchmarks

**Files:**
- Modify: `tests/test_benchmark_encoding_bases.py:1-115`
- Modify: `QuditsOnQubits/project_paths.py:27-40`

- [ ] **Step 1: Add imports and a sample report row helper to the benchmark test file**

```python
import pandas as pd
from igraph import Graph
from unittest.mock import patch

from QuditsOnQubits.project_paths import (
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
)


def _sample_report_frame(state_name):
    return pd.DataFrame(
        [
            {
                "state_name": state_name,
                "class_name": "baseline",
                "candidate_name": "E_old",
                "status": "ok",
                "best_depth": 47,
                "mean_depth": 53.9,
                "best_size": 141,
                "best_two_qubit_gate_count": 32,
                "mean_two_qubit_gate_count": 33.05,
                "fid085_best_approx_degree": 0.91,
                "fid085_best_fidelity": 0.8786,
                "fid085_best_depth": 41,
                "fid085_best_two_qubit_gate_count": 28,
                "fid090_best_approx_degree": 0.95,
                "fid090_best_fidelity": 0.9098,
                "fid090_best_depth": 43,
                "fid090_best_two_qubit_gate_count": 30,
                "fid095_best_approx_degree": 0.99,
                "fid095_best_fidelity": 0.9909,
                "fid095_best_depth": 46,
                "fid095_best_two_qubit_gate_count": 31,
            }
        ]
    )
```

- [ ] **Step 2: Add failing tests for state-aware path helpers and `two_qutrit` / `ame43` circuit export**

```python
    def test_state_specific_results_paths_preserve_ghz3_and_add_new_states(self):
        self.assertTrue(
            benchmark_state_results_path("ghz3", "full").endswith(
                os.path.join("data", "benchmarks", "benchmark_encoding_bases_full_results.csv")
            )
        )
        self.assertTrue(
            benchmark_state_results_path("two_qutrit", "full").endswith(
                os.path.join(
                    "data",
                    "benchmarks",
                    "benchmark_encoding_bases_two_qutrit_full_results.csv",
                )
            )
        )
        self.assertTrue(
            benchmark_state_results_path("ame43", "original").endswith(
                os.path.join(
                    "data",
                    "benchmarks",
                    "benchmark_encoding_bases_ame43_original_results.csv",
                )
            )
        )
        self.assertTrue(
            multi_state_benchmark_report_path().endswith(
                os.path.join(
                    "docs",
                    "benchmarks",
                    "benchmark_encoding_bases_multi_state_analysis.md",
                )
            )
        )

    def test_benchmark_basis_saves_two_qutrit_circuit_under_state_folder(self):
        expected_qc, _ = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=None)

        tmpdir = _workspace_tempdir()
        try:
            row = benchmark_basis(
                E_new=None,
                class_name="baseline",
                candidate_name="E_old",
                state_name="two_qutrit",
                coupling_map=[[0, 1], [1, 2], [2, 3]],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=1,
                circuits_output_dir=tmpdir,
            )

            saved_path = os.path.join(tmpdir, "two_qutrit", "baseline", "E_old.qpy")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "two_qutrit")
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "rb") as fd:
                saved_qc = qpy.load(fd)[0]

            self.assertEqual(saved_qc.num_qubits, expected_qc.num_qubits)
            self.assertEqual(saved_qc.count_ops(), expected_qc.count_ops())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_benchmark_basis_saves_ame43_circuit_under_state_folder(self):
        game43 = Graph(n=4, edges=[[0, 1], [0, 1], [1, 2], [2, 3], [3, 0]])
        expected_qc, _ = create_ame_circuit(dim=3, graph=game43, E_new=None)

        tmpdir = _workspace_tempdir()
        try:
            row = benchmark_basis(
                E_new=None,
                class_name="baseline",
                candidate_name="E_old",
                state_name="ame43",
                coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=1,
                circuits_output_dir=tmpdir,
            )

            saved_path = os.path.join(tmpdir, "ame43", "baseline", "E_old.qpy")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "ame43")
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "rb") as fd:
                saved_qc = qpy.load(fd)[0]

            self.assertEqual(saved_qc.num_qubits, expected_qc.num_qubits)
            self.assertEqual(saved_qc.count_ops(), expected_qc.count_ops())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 3: Run the targeted benchmark test file and confirm it fails before implementation**

Run: `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`

Expected: FAIL with at least one of:
- `ImportError: cannot import name 'benchmark_state_results_path'`
- `TypeError: benchmark_basis() got an unexpected keyword argument 'state_name'`

- [ ] **Step 4: Commit the red-state tests**

```bash
git add tests/test_benchmark_encoding_bases.py
git commit -m "test: cover multi-state benchmark paths and circuit export"
```

### Task 2: Implement State-Aware Paths And Circuit Construction

**Files:**
- Modify: `QuditsOnQubits/project_paths.py:27-40`
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py:213-803`

- [ ] **Step 1: Add state-aware output path helpers while keeping the existing GHZ CSV path unchanged**

```python
def benchmark_state_slug(state_name):
    valid = {"two_qutrit", "ghz3", "ame43"}
    if state_name not in valid:
        raise ValueError(f"Unknown benchmark state: {state_name}")
    return state_name


def benchmark_state_results_path(state_name, mode):
    if benchmark_state_slug(state_name) == "ghz3":
        return benchmark_results_path(mode)
    filename = f"benchmark_encoding_bases_{state_name}_{mode}_results.csv"
    return benchmark_data_dir(filename)


def benchmark_state_circuits_dir(state_name, *parts):
    return benchmark_circuits_dir(benchmark_state_slug(state_name), *parts)


def multi_state_benchmark_report_path():
    return benchmark_docs_dir("benchmark_encoding_bases_multi_state_analysis.md")
```

- [ ] **Step 2: Add a helper in `benchmark_encoding_bases.py` that builds the right base circuit for each state**

```python
from igraph import Graph

from QuditsOnQubits.project_paths import (
    benchmark_circuits_dir,
    benchmark_docs_dir,
    benchmark_results_path,
    benchmark_state_circuits_dir,
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
)


def _build_state_circuit(state_name, E_new):
    if state_name == "two_qutrit":
        return create_ame_circuit(n=2, dim=3, graph_type="star", E_new=E_new)
    if state_name == "ghz3":
        return create_ame_circuit(n=3, dim=3, graph_type="star", E_new=E_new)
    if state_name == "ame43":
        game43 = Graph(n=4, edges=[[0, 1], [0, 1], [1, 2], [2, 3], [3, 0]])
        return create_ame_circuit(dim=3, graph=game43, E_new=E_new)
    raise ValueError(f"Unknown benchmark state: {state_name}")


def _resolve_circuits_output_dir(state_name, circuits_output_dir):
    if circuits_output_dir is _DEFAULT_CIRCUITS_OUTPUT_DIR:
        return benchmark_state_circuits_dir(state_name)
    if circuits_output_dir is None:
        return None
    return os.path.join(circuits_output_dir, state_name)
```

- [ ] **Step 3: Thread `state_name` through `benchmark_basis(...)` and save outputs under the state subdirectory**

```python
def benchmark_basis(
    E_new,
    class_name,
    candidate_name,
    state_name="ghz3",
    n_qutrits=3,
    coupling_map=None,
    basis_gates=None,
    n_transpile_runs=20,
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
):
    if coupling_map is None:
        coupling_map = COUPLING_MAP
    if basis_gates is None:
        basis_gates = BASIS_GATES
    if fidelity_thresholds is None:
        fidelity_thresholds = DEFAULT_FIDELITY_THRESHOLDS

    circuits_output_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)
    meta = compute_encoding_metadata(E_new)

    row = {
        "state_name": state_name,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "is_valid": True,
        "uses_old_codespace_only": meta["uses_old_codespace_only"],
        "avg_codeword_entanglement": meta["avg_codeword_entanglement"],
        "overlap_with_old_codespace": meta["overlap_with_old_codespace"],
        "best_depth": None,
        "mean_depth": None,
        "std_depth": None,
        "best_size": None,
        "mean_size": None,
        "best_two_qubit_gate_count": None,
        "mean_two_qubit_gate_count": None,
        "num_qubits": None,
        "best_count_ops": None,
        "n_transpile_runs": n_transpile_runs,
        "successful_trials": 0,
        "failed_trials": 0,
        "status": "ok",
        "error_message": "",
    }
    row.update(_make_approximation_result_fields(fidelity_thresholds))

    try:
        qc, _ = _build_state_circuit(state_name, E_new=E_new)
        if circuits_output_dir is not None:
            _save_benchmark_circuit(
                qc,
                class_name=class_name,
                candidate_name=candidate_name,
                output_root=circuits_output_dir,
            )
    except Exception:
        row["status"] = "build_error"
        row["error_message"] = traceback.format_exc()
        return row
```

- [ ] **Step 4: Re-run the targeted benchmark test file and confirm all current tests pass**

Run: `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`

Expected: PASS for:
- `test_state_specific_results_paths_preserve_ghz3_and_add_new_states`
- `test_benchmark_basis_saves_two_qutrit_circuit_under_state_folder`
- `test_benchmark_basis_saves_ame43_circuit_under_state_folder`
- all previously existing tests in the same file

- [ ] **Step 5: Commit the green state for state-aware circuit handling**

```bash
git add QuditsOnQubits/project_paths.py QuditsOnQubits/benchmark_encoding_bases.py tests/test_benchmark_encoding_bases.py
git commit -m "feat: add state-aware encoding benchmark circuits"
```

### Task 3: Add Failing Tests For Combined Markdown Reporting And All-State Orchestration

**Files:**
- Modify: `tests/test_benchmark_encoding_bases.py:1-220`

- [ ] **Step 1: Add a failing markdown report test that checks for sections and fidelity-depth columns**

```python
    def test_write_multi_state_benchmark_report_includes_fidelity_depth_columns(self):
        frames = {
            "two_qutrit": _sample_report_frame("two_qutrit"),
            "ghz3": _sample_report_frame("ghz3"),
            "ame43": _sample_report_frame("ame43"),
        }

        tmpdir = _workspace_tempdir()
        try:
            report_path = os.path.join(tmpdir, "benchmark_encoding_bases_multi_state_analysis.md")
            write_multi_state_benchmark_report(frames, report_path)

            with open(report_path, "r", encoding="utf-8") as fd:
                content = fd.read()

            self.assertIn("# Multi-State Encoding Benchmark Analysis", content)
            self.assertIn("## two_qutrit", content)
            self.assertIn("## ghz3", content)
            self.assertIn("## ame43", content)
            self.assertIn("## Cross-state comparison", content)
            self.assertIn("fid085 depth", content)
            self.assertIn("fid090 depth", content)
            self.assertIn("fid095 depth", content)
            self.assertIn("fid085 2Q", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Add a failing orchestration test for `run_benchmark(..., state_name=\"all\")`**

```python
    @patch("QuditsOnQubits.benchmark_encoding_bases.write_multi_state_benchmark_report")
    @patch("QuditsOnQubits.benchmark_encoding_bases._run_single_state_benchmark")
    @patch("QuditsOnQubits.benchmark_encoding_bases.pd.read_csv")
    def test_run_benchmark_all_reuses_existing_ghz3_results(
        self,
        read_csv_mock,
        run_single_mock,
        write_report_mock,
    ):
        two_df = _sample_report_frame("two_qutrit")
        ghz_df = _sample_report_frame("ghz3")
        ame_df = _sample_report_frame("ame43")

        run_single_mock.side_effect = [
            (two_df, "two.csv"),
            (ame_df, "ame.csv"),
        ]
        read_csv_mock.return_value = ghz_df

        result = run_benchmark(
            state_name="all",
            mode="original",
            reuse_existing_ghz3=True,
            n_transpile_runs=1,
            approximation_values=[1.0],
            fidelity_thresholds=(0.95,),
            circuits_output_dir=None,
        )

        self.assertEqual(set(result.keys()), {"two_qutrit", "ghz3", "ame43"})
        self.assertEqual(run_single_mock.call_args_list[0].kwargs["state_name"], "two_qutrit")
        self.assertEqual(run_single_mock.call_args_list[1].kwargs["state_name"], "ame43")
        read_csv_mock.assert_called_once()
        write_report_mock.assert_called_once()
```

- [ ] **Step 3: Run the targeted benchmark test file and confirm it fails before report/orchestration implementation**

Run: `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`

Expected: FAIL with at least one of:
- `NameError` or `ImportError` for `write_multi_state_benchmark_report`
- `AttributeError` for `_run_single_state_benchmark`
- `TypeError: run_benchmark() got an unexpected keyword argument 'state_name'`

- [ ] **Step 4: Commit the red-state report/orchestration tests**

```bash
git add tests/test_benchmark_encoding_bases.py
git commit -m "test: cover combined multi-state benchmark reporting"
```

### Task 4: Implement Combined Report Generation And `run_benchmark(...)` Orchestration

**Files:**
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py:243-979`

- [ ] **Step 1: Extract the current single-state benchmark body into `_run_single_state_benchmark(...)`**

```python
def _run_single_state_benchmark(
    state_name="ghz3",
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
):
    if csv_path is None:
        csv_path = benchmark_state_results_path(state_name, mode)

    all_candidates = []
    if mode in ("full", "original"):
        all_candidates += generate_baseline()
        all_candidates += generate_monomial_bases(max_candidates=120)
        all_candidates += generate_fourier_like_bases(max_candidates=80)
        all_candidates += generate_householder_bases(n_samples=20, seed=42)
        all_candidates += generate_clifford_wh_bases()
        all_candidates += generate_haar_random_isometries(n_samples=20, seed=100)
        all_candidates += generate_perturbed_isometries(n_samples_per_eps=8, seed=200)
        all_candidates += generate_entangling_isometries(n_samples=20, seed=300)
        all_candidates += generate_structured_entangling_isometries()

    n_orig = len(all_candidates)

    if mode in ("full", "extended"):
        all_candidates += generate_local_ry_only(n_grid=10)
        all_candidates += generate_local_general_su2(n_samples=30, seed=600)
        all_candidates += generate_real_orthogonal_isometries(n_samples=20, seed=400)
        all_candidates += generate_near_identity_isometries(n_samples_per_eps=10, seed=500)
        all_candidates += generate_finer_structured_grid()
        all_candidates += generate_two_cz_ansatz(n_samples=50, seed=700)

    results = []
    for cls, name, E_new in all_candidates:
        row = benchmark_basis(
            E_new,
            cls,
            name,
            state_name=state_name,
            coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES,
            n_transpile_runs=n_transpile_runs,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
        )
        results.append(row)

    df = pd.DataFrame(results)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df, csv_path
```

- [ ] **Step 2: Add small markdown helpers and the combined report writer**

```python
def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_fidelity_cell(row, prefix, suffix):
    value = row.get(f"{prefix}_{suffix}")
    return "brak" if pd.isna(value) else value


def write_multi_state_benchmark_report(state_frames, output_path):
    lines = ["# Multi-State Encoding Benchmark Analysis", ""]
    comparison_rows = []

    for state_name in ("two_qutrit", "ghz3", "ame43"):
        df = state_frames[state_name]
        df_ok = df[df["status"] == "ok"].copy()
        top = df_ok.sort_values(
            by=["best_depth", "best_two_qubit_gate_count", "best_size"],
            ascending=True,
        ).head(10)
        per_class = (
            df_ok.sort_values(
                by=["best_depth", "best_two_qubit_gate_count", "best_size"],
                ascending=True,
            )
            .groupby("class_name", as_index=False)
            .first()
        )

        lines.extend(
            [
                f"## {state_name}",
                "",
                _markdown_table(
                    ["Metric", "Value"],
                    [
                        ["Rows", len(df)],
                        ["Successful rows", int((df["status"] == "ok").sum())],
                        ["Failed rows", int((df["status"] != "ok").sum())],
                    ],
                ),
                "",
                _markdown_table(
                    ["Class", "Candidate", "best_depth", "best_2q", "mean_depth"],
                    [
                        [
                            row["class_name"],
                            row["candidate_name"],
                            row["best_depth"],
                            row["best_two_qubit_gate_count"],
                            row["mean_depth"],
                        ]
                        for _, row in top.iterrows()
                    ],
                ),
                "",
                _markdown_table(
                    ["Class", "Candidate", "best_depth", "best_2q", "mean_depth"],
                    [
                        [
                            row["class_name"],
                            row["candidate_name"],
                            row["best_depth"],
                            row["best_two_qubit_gate_count"],
                            row["mean_depth"],
                        ]
                        for _, row in per_class.iterrows()
                    ],
                ),
                "",
                _markdown_table(
                    [
                        "Class",
                        "Candidate",
                        "fid085 approx",
                        "fid085 fidelity",
                        "fid085 depth",
                        "fid085 2Q",
                        "fid090 approx",
                        "fid090 fidelity",
                        "fid090 depth",
                        "fid090 2Q",
                        "fid095 approx",
                        "fid095 fidelity",
                        "fid095 depth",
                        "fid095 2Q",
                    ],
                    [
                        [
                            row["class_name"],
                            row["candidate_name"],
                            _format_fidelity_cell(row, "fid085", "best_approx_degree"),
                            _format_fidelity_cell(row, "fid085", "best_fidelity"),
                            _format_fidelity_cell(row, "fid085", "best_depth"),
                            _format_fidelity_cell(row, "fid085", "best_two_qubit_gate_count"),
                            _format_fidelity_cell(row, "fid090", "best_approx_degree"),
                            _format_fidelity_cell(row, "fid090", "best_fidelity"),
                            _format_fidelity_cell(row, "fid090", "best_depth"),
                            _format_fidelity_cell(row, "fid090", "best_two_qubit_gate_count"),
                            _format_fidelity_cell(row, "fid095", "best_approx_degree"),
                            _format_fidelity_cell(row, "fid095", "best_fidelity"),
                            _format_fidelity_cell(row, "fid095", "best_depth"),
                            _format_fidelity_cell(row, "fid095", "best_two_qubit_gate_count"),
                        ]
                        for _, row in per_class.iterrows()
                    ],
                ),
                "",
            ]
        )

        comparison_rows.extend(
            [
                [
                    state_name,
                    row["class_name"],
                    row["candidate_name"],
                    row["best_depth"],
                    row["best_two_qubit_gate_count"],
                    row["fid095_best_depth"],
                ]
                for _, row in per_class.iterrows()
            ]
        )

    lines.extend(
        [
            "## Cross-state comparison",
            "",
            _markdown_table(
                ["State", "Class", "Candidate", "best_depth", "best_2q", "fid095 depth"],
                comparison_rows,
            ),
        ]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fd:
        fd.write("\n".join(lines).strip() + "\n")
```

- [ ] **Step 3: Make `run_benchmark(...)` dispatch either one state or the three-state workflow**

```python
def run_benchmark(
    n_qutrits=3,
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    state_name="ghz3",
    reuse_existing_ghz3=True,
    combined_report_path=None,
):
    if state_name != "all":
        df, _ = _run_single_state_benchmark(
            state_name=state_name,
            n_transpile_runs=n_transpile_runs,
            csv_path=csv_path,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
        )
        return df

    state_frames = {}
    state_frames["two_qutrit"], _ = _run_single_state_benchmark(
        state_name="two_qutrit",
        n_transpile_runs=n_transpile_runs,
        mode=mode,
        circuits_output_dir=circuits_output_dir,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=approximation_seed,
    )

    ghz_csv = benchmark_state_results_path("ghz3", mode)
    if reuse_existing_ghz3 and os.path.exists(ghz_csv):
        state_frames["ghz3"] = pd.read_csv(ghz_csv)
    else:
        state_frames["ghz3"], _ = _run_single_state_benchmark(
            state_name="ghz3",
            n_transpile_runs=n_transpile_runs,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
        )

    state_frames["ame43"], _ = _run_single_state_benchmark(
        state_name="ame43",
        n_transpile_runs=n_transpile_runs,
        mode=mode,
        circuits_output_dir=circuits_output_dir,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=approximation_seed,
    )

    report_path = combined_report_path or multi_state_benchmark_report_path()
    write_multi_state_benchmark_report(state_frames, report_path)
    return state_frames
```

- [ ] **Step 4: Update the CLI entry point so the script can run one state or all three**

```python
if __name__ == "__main__":
    import sys

    _mode = sys.argv[1] if len(sys.argv) > 1 else "extended"
    _state = sys.argv[2] if len(sys.argv) > 2 else "ghz3"

    if _state == "all":
        run_benchmark(mode=_mode, state_name="all")
    else:
        _csv = benchmark_state_results_path(_state, _mode)
        run_benchmark(mode=_mode, csv_path=_csv, state_name=_state)
```

- [ ] **Step 5: Re-run the targeted benchmark test file and confirm all tests pass**

Run: `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`

Expected: PASS for:
- all previously existing benchmark tests
- `test_write_multi_state_benchmark_report_includes_fidelity_depth_columns`
- `test_run_benchmark_all_reuses_existing_ghz3_results`

- [ ] **Step 6: Commit the report/orchestration implementation**

```bash
git add QuditsOnQubits/benchmark_encoding_bases.py tests/test_benchmark_encoding_bases.py
git commit -m "feat: add multi-state benchmark orchestration and report"
```

### Task 5: Verify The Whole Test Suite And Smoke-Check Output Paths

**Files:**
- Modify: `QuditsOnQubits/project_paths.py`
- Modify: `QuditsOnQubits/benchmark_encoding_bases.py`
- Modify: `tests/test_benchmark_encoding_bases.py`

- [ ] **Step 1: Run the focused benchmark test file**

Run: `python -m unittest discover -s tests -p "test_benchmark_encoding_bases.py" -v`

Expected: PASS with all benchmark-related tests green.

- [ ] **Step 2: Run the full unittest suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with `OK` at the end and no unexpected failures.

- [ ] **Step 3: Smoke-check the state-aware path helpers from the interpreter**

Run: `python -c "from QuditsOnQubits.project_paths import benchmark_state_results_path, multi_state_benchmark_report_path; print(benchmark_state_results_path('two_qutrit','full')); print(benchmark_state_results_path('ghz3','full')); print(benchmark_state_results_path('ame43','full')); print(multi_state_benchmark_report_path())"`

Expected: prints four paths ending with:
- `data\\benchmarks\\benchmark_encoding_bases_two_qutrit_full_results.csv`
- `data\\benchmarks\\benchmark_encoding_bases_full_results.csv`
- `data\\benchmarks\\benchmark_encoding_bases_ame43_full_results.csv`
- `docs\\benchmarks\\benchmark_encoding_bases_multi_state_analysis.md`

- [ ] **Step 4: Commit the final verified state**

```bash
git add QuditsOnQubits/project_paths.py QuditsOnQubits/benchmark_encoding_bases.py tests/test_benchmark_encoding_bases.py
git commit -m "test: verify multi-state encoding benchmark workflow"
```
