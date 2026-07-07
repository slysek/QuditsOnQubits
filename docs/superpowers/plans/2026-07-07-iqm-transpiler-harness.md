# IQM Transpiler Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an IQM transpiler harness that compares IQM-aware transpilation strategies for direct-basis candidates loaded from previous benchmark CSVs.

**Architecture:** Add a separate strategy layer and harness module under `src/qudits_on_qubits/benchmarks/direct_basis/`, plus a CLI script under `scripts/`. The harness reuses existing direct-basis candidate loading and circuit construction, records all trial metrics, writes `all_trials.csv` and `best_by_candidate.csv`, and leaves the main benchmark default unchanged.

**Tech Stack:** Python 3.12 in `C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe`, Qiskit, `iqm-client[qiskit]`, pandas, unittest.

---

## File Structure

- Create: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_strategies.py`
  - Own the built-in IQM transpilation strategy registry and strategy execution.
- Create: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py`
  - Own trial metric extraction, warning flags, best-trial selection, package/backend metadata, and output writing.
- Create: `scripts/run_iqm_transpiler_harness.py`
  - CLI entry point for candidate loading, backend loading, harness execution, and output path selection.
- Create: `tests/test_direct_basis_iqm_transpiler_strategies.py`
  - Unit tests for strategy registry and fake-backend strategy execution.
- Create: `tests/test_direct_basis_iqm_transpiler_harness.py`
  - Unit tests for metrics, ranking, warnings, unsupported candidates, and output writing.
- Create: `tests/test_direct_basis_iqm_transpiler_harness_cli.py`
  - Unit tests for parser defaults and CLI wiring without real network access.
- Modify: `README.md`
  - Add a short usage section for the IQM transpiler harness.

Use the explicit environment Python for verification commands:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest discover -s tests -v
```

`conda run -n quditsD3_laptop ...` is not used in this workspace because it attempted to write under `AppData\Local\conda` and failed under sandbox permissions.

---

### Task 1: IQM Transpiler Strategy Layer

**Files:**
- Create: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_strategies.py`
- Create: `tests/test_direct_basis_iqm_transpiler_strategies.py`

- [ ] **Step 1: Write failing strategy tests**

Create `tests/test_direct_basis_iqm_transpiler_strategies.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    EXACT_RZ_SCHEDULING_METHOD,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    BUILTIN_IQM_TRANSPILER_STRATEGIES,
    get_iqm_transpiler_strategy,
    iqm_transpiler_strategy_names,
    run_iqm_transpiler_strategy,
)


def _fake_garnet():
    try:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    except ImportError as exc:
        raise unittest.SkipTest(f"IQM fake backend is unavailable: {exc}") from exc
    return IQMFakeGarnet()


class IqmTranspilerStrategyTests(unittest.TestCase):
    def test_strategy_registry_contains_expected_names(self):
        self.assertEqual(
            set(iqm_transpiler_strategy_names()),
            {
                "preset_default",
                "preset_exact",
                "transpile_to_iqm_default",
                "transpile_to_iqm_exact",
            },
        )

    def test_preset_exact_records_exact_scheduling_method(self):
        strategy = get_iqm_transpiler_strategy("preset_exact")

        self.assertEqual(strategy.scheduling_method, EXACT_RZ_SCHEDULING_METHOD)
        self.assertIs(strategy.remove_final_rzs, False)

    def test_unknown_strategy_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown IQM transpiler strategy"):
            get_iqm_transpiler_strategy("missing")

    def test_builtin_strategy_mapping_is_immutable_to_callers(self):
        names = iqm_transpiler_strategy_names()

        self.assertIsInstance(names, tuple)
        self.assertIn("preset_default", BUILTIN_IQM_TRANSPILER_STRATEGIES)

    def test_run_preset_default_strategy_with_fake_backend(self):
        backend = _fake_garnet()
        circuit = build_direct_basis_graph_state_circuit(
            "two_qutrit",
            np.eye(3, dtype=complex),
            n_qutrits=2,
        )

        result = run_iqm_transpiler_strategy(
            "preset_default",
            circuit,
            backend=backend,
            seed_transpiler=0,
        )

        self.assertTrue(result.success, result.error_message)
        self.assertIsNotNone(result.circuit)
        ops = result.circuit.count_ops()
        self.assertIn("r", ops)
        self.assertIn("cz", ops)
        self.assertGreater(result.compile_time_seconds, 0.0)

    def test_run_strategy_captures_exception(self):
        circuit = build_direct_basis_graph_state_circuit(
            "two_qutrit",
            np.eye(3, dtype=complex),
            n_qutrits=2,
        )

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies.generate_preset_pass_manager",
            side_effect=RuntimeError("boom"),
        ):
            result = run_iqm_transpiler_strategy(
                "preset_default",
                circuit,
                backend=object(),
                seed_transpiler=0,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("boom", result.error_message)
        self.assertIsNone(result.circuit)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run strategy tests to verify failure**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_strategies -v
```

Expected: FAIL with `ModuleNotFoundError` for `iqm_transpiler_strategies`.

- [ ] **Step 3: Implement strategy module**

Create `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_strategies.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    EXACT_RZ_SCHEDULING_METHOD,
)


@dataclass(frozen=True)
class IqmTranspilerStrategy:
    name: str
    description: str
    kind: str
    scheduling_method: str | None
    remove_final_rzs: bool


@dataclass(frozen=True)
class IqmTranspilerStrategyResult:
    strategy_name: str
    seed_transpiler: int
    success: bool
    circuit: QuantumCircuit | None
    compile_time_seconds: float
    error_type: str = ""
    error_message: str = ""


_BUILTINS: dict[str, IqmTranspilerStrategy] = {
    "preset_default": IqmTranspilerStrategy(
        name="preset_default",
        description="Qiskit preset pass manager with IQM backend default scheduling.",
        kind="preset",
        scheduling_method=None,
        remove_final_rzs=True,
    ),
    "preset_exact": IqmTranspilerStrategy(
        name="preset_exact",
        description="Qiskit preset pass manager with IQM exact global phase scheduling.",
        kind="preset",
        scheduling_method=EXACT_RZ_SCHEDULING_METHOD,
        remove_final_rzs=False,
    ),
    "transpile_to_iqm_default": IqmTranspilerStrategy(
        name="transpile_to_iqm_default",
        description="IQM transpile_to_IQM helper with final RZ removal enabled.",
        kind="transpile_to_iqm",
        scheduling_method=None,
        remove_final_rzs=True,
    ),
    "transpile_to_iqm_exact": IqmTranspilerStrategy(
        name="transpile_to_iqm_exact",
        description="IQM transpile_to_IQM helper preserving final RZ gates.",
        kind="transpile_to_iqm",
        scheduling_method=None,
        remove_final_rzs=False,
    ),
}

BUILTIN_IQM_TRANSPILER_STRATEGIES = MappingProxyType(_BUILTINS)


def iqm_transpiler_strategy_names() -> tuple[str, ...]:
    return tuple(_BUILTINS)


def get_iqm_transpiler_strategy(name: str) -> IqmTranspilerStrategy:
    try:
        return _BUILTINS[str(name)]
    except KeyError as exc:
        available = ", ".join(iqm_transpiler_strategy_names())
        raise ValueError(
            f"Unknown IQM transpiler strategy: {name}. Available: {available}"
        ) from exc


def _load_transpile_to_iqm():
    try:
        from iqm.qiskit_iqm import transpile_to_IQM
    except ImportError:
        from iqm.qiskit_iqm.iqm_naive_move_pass import transpile_to_IQM
    return transpile_to_IQM


def _run_strategy(
    strategy: IqmTranspilerStrategy,
    circuit: QuantumCircuit,
    *,
    backend: Any,
    seed_transpiler: int,
    optimization_level: int,
) -> QuantumCircuit:
    if strategy.kind == "preset":
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed_transpiler),
            scheduling_method=strategy.scheduling_method,
        )
        return pass_manager.run(circuit)

    if strategy.kind == "transpile_to_iqm":
        transpile_to_iqm = _load_transpile_to_iqm()
        return transpile_to_iqm(
            circuit,
            backend,
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed_transpiler),
            remove_final_rzs=bool(strategy.remove_final_rzs),
        )

    raise ValueError(f"Unsupported IQM transpiler strategy kind: {strategy.kind}")


def run_iqm_transpiler_strategy(
    strategy_name: str,
    circuit: QuantumCircuit,
    *,
    backend: Any,
    seed_transpiler: int,
    optimization_level: int = 3,
) -> IqmTranspilerStrategyResult:
    started = time.time()
    try:
        strategy = get_iqm_transpiler_strategy(strategy_name)
        transpiled = _run_strategy(
            strategy,
            circuit,
            backend=backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
        return IqmTranspilerStrategyResult(
            strategy_name=strategy.name,
            seed_transpiler=int(seed_transpiler),
            success=True,
            circuit=transpiled,
            compile_time_seconds=round(time.time() - started, 6),
        )
    except Exception as exc:
        return IqmTranspilerStrategyResult(
            strategy_name=str(strategy_name),
            seed_transpiler=int(seed_transpiler),
            success=False,
            circuit=None,
            compile_time_seconds=round(time.time() - started, 6),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
```

- [ ] **Step 4: Run strategy tests**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_strategies -v
```

Expected: PASS.

- [ ] **Step 5: Commit strategy layer**

Run:

```powershell
git add -- src\qudits_on_qubits\benchmarks\direct_basis\iqm_transpiler_strategies.py tests\test_direct_basis_iqm_transpiler_strategies.py
git commit -m "feat: add iqm transpiler strategy registry"
```

---

### Task 2: IQM Transpiler Harness Core

**Files:**
- Create: `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py`
- Create: `tests/test_direct_basis_iqm_transpiler_harness.py`

- [ ] **Step 1: Write failing harness tests**

Create `tests/test_direct_basis_iqm_transpiler_harness.py`:

```python
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_harness import (
    IqmTranspilerHarnessConfig,
    _best_trial_rows,
    _metric_row,
    _warning_flags,
    default_iqm_transpiler_harness_output_dir,
    run_iqm_transpiler_harness,
    write_iqm_transpiler_harness_outputs,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    IqmTranspilerStrategyResult,
)


class IqmTranspilerHarnessTests(unittest.TestCase):
    def test_metric_row_counts_native_ops(self):
        circuit = QuantumCircuit(2)
        circuit.r(0.1, 0.2, 0)
        circuit.r(0.3, 0.4, 1)
        circuit.cz(0, 1)

        row = _metric_row(circuit)

        self.assertEqual(row["depth"], 2)
        self.assertEqual(row["size"], 3)
        self.assertEqual(row["cz_count"], 1)
        self.assertEqual(row["r_count"], 2)
        self.assertEqual(row["one_qubit_gate_count"], 2)
        self.assertEqual(row["two_qubit_gate_count"], 1)
        self.assertEqual(json.loads(row["count_ops_json"]), {"cz": 1, "r": 2})

    def test_warning_flags_include_depth_and_cz(self):
        flags = _warning_flags(
            {"depth": 101, "cz_count": 51},
            max_depth_warning=100,
            max_cz_warning=50,
        )

        self.assertEqual(flags, "depth_gt_100;cz_gt_50")

    def test_warning_flags_empty_for_acceptable_metrics(self):
        flags = _warning_flags(
            {"depth": 35, "cz_count": 18},
            max_depth_warning=100,
            max_cz_warning=50,
        )

        self.assertEqual(flags, "")

    def test_best_trial_rows_rank_success_by_depth_cz_r_size(self):
        rows = [
            {
                "candidate_name": "A",
                "class_name": "baseline",
                "success": True,
                "depth": 40,
                "cz_count": 10,
                "r_count": 20,
                "size": 30,
                "status": "ok",
            },
            {
                "candidate_name": "A",
                "class_name": "baseline",
                "success": True,
                "depth": 35,
                "cz_count": 18,
                "r_count": 30,
                "size": 48,
                "status": "ok",
            },
            {
                "candidate_name": "B",
                "class_name": "monomial_full",
                "success": False,
                "depth": None,
                "cz_count": None,
                "r_count": None,
                "size": None,
                "status": "failed",
            },
        ]

        best = _best_trial_rows(rows)

        self.assertEqual(len(best), 2)
        self.assertEqual(best[0]["candidate_name"], "A")
        self.assertEqual(best[0]["depth"], 35)
        self.assertEqual(best[1]["candidate_name"], "B")
        self.assertEqual(best[1]["status"], "failed_all_strategies")

    def test_run_harness_records_success_and_failure_rows(self):
        backend = object()
        candidates = [
            DirectBasisCandidate(
                name="I",
                candidate_type="identity",
                matrix=np.eye(3, dtype=complex),
                source_class_name="baseline",
                source_candidate_name="E_old",
            )
        ]

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            if strategy_name == "bad":
                return IqmTranspilerStrategyResult(
                    strategy_name=strategy_name,
                    seed_transpiler=seed_transpiler,
                    success=False,
                    circuit=None,
                    compile_time_seconds=0.01,
                    error_type="RuntimeError",
                    error_message="boom",
                )
            output = QuantumCircuit(2)
            output.r(0.1, 0.2, 0)
            output.cz(0, 1)
            return IqmTranspilerStrategyResult(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=output,
                compile_time_seconds=0.02,
            )

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=backend,
            iqm_backend_name="garnet",
            iqm_use_metrics=False,
            candidates=candidates,
            strategy_names=("good", "bad"),
            n_transpile_runs=1,
        )

        all_trials, best, summary = run_iqm_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        self.assertEqual(len(all_trials), 2)
        self.assertEqual(set(all_trials["strategy_name"]), {"good", "bad"})
        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["strategy_name"], "good")
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["trial_count"], 2)

    def test_run_harness_records_unsupported_candidate(self):
        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="garnet",
            iqm_use_metrics=False,
            candidates=[
                DirectBasisCandidate(
                    name="bad",
                    candidate_type="unsupported",
                    matrix=None,
                    source_class_name="missing",
                    source_candidate_name="bad",
                    error_message="not found",
                )
            ],
            strategy_names=("preset_default",),
            n_transpile_runs=1,
        )

        all_trials, best, summary = run_iqm_transpiler_harness(config)

        self.assertEqual(list(all_trials["status"]), ["unsupported_candidate"])
        self.assertEqual(list(best["status"]), ["unsupported_candidate"])
        self.assertEqual(summary["unsupported_candidate_count"], 1)

    def test_write_outputs_creates_csv_and_summary(self):
        backend = object()
        candidates = [
            DirectBasisCandidate(
                name="I",
                candidate_type="identity",
                matrix=np.eye(3, dtype=complex),
            )
        ]

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            output = QuantumCircuit(1)
            output.r(0.1, 0.2, 0)
            return IqmTranspilerStrategyResult(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=output,
                compile_time_seconds=0.01,
            )

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=backend,
            iqm_backend_name="garnet",
            iqm_use_metrics=False,
            candidates=candidates,
            strategy_names=("preset_default",),
            n_transpile_runs=1,
        )
        all_trials, best, summary = run_iqm_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_iqm_transpiler_harness_outputs(
                tmp,
                all_trials=all_trials,
                best_by_candidate=best,
                summary=summary,
            )

            self.assertTrue(os.path.isfile(paths["all_trials_csv"]))
            self.assertTrue(os.path.isfile(paths["best_by_candidate_csv"]))
            self.assertTrue(os.path.isfile(paths["summary_json"]))

    def test_default_output_dir_contains_iqm_processed_harness_path(self):
        path = default_iqm_transpiler_harness_output_dir("run123")

        self.assertTrue(
            path.endswith(
                os.path.join(
                    "artifacts",
                    "iqm_runs",
                    "processed",
                    "transpiler_harness",
                    "run123",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run harness tests to verify failure**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_harness -v
```

Expected: FAIL with `ModuleNotFoundError` for `iqm_transpiler_harness`.

- [ ] **Step 3: Implement harness module**

Create `src/qudits_on_qubits/benchmarks/direct_basis/iqm_transpiler_harness.py`:

```python
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import backend_metadata
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    iqm_transpiler_strategy_names,
    run_iqm_transpiler_strategy,
)
from qudits_on_qubits.core.benchmark_encoding_bases import TWO_Q_GATES
from qudits_on_qubits.core.project_paths import repo_path


StrategyRunner = Callable[..., Any]


@dataclass(frozen=True)
class IqmTranspilerHarnessConfig:
    state_name: str
    n_qutrits: int | None
    backend: Any
    iqm_backend_name: str
    iqm_use_metrics: bool
    candidates: Iterable[DirectBasisCandidate]
    strategy_names: tuple[str, ...] = ()
    n_transpile_runs: int = 1
    optimization_level: int = 3
    max_depth_warning: int = 100
    max_cz_warning: int = 50


def default_iqm_transpiler_harness_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_iqm_transpiler_harness_output_dir(run_id: str | None = None) -> str:
    value = run_id or default_iqm_transpiler_harness_run_id()
    return str(
        Path(
            repo_path(
                "artifacts",
                "iqm_runs",
                "processed",
                "transpiler_harness",
                value,
            )
        )
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _iqm_adapter_path() -> str:
    try:
        import iqm.qiskit_iqm as qiskit_iqm
    except ImportError:
        return ""
    return str(getattr(qiskit_iqm, "__file__", "") or "")


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "qiskit_version": _package_version("qiskit"),
        "iqm_client_version": _package_version("iqm-client"),
        "iqm_qiskit_adapter_path": _iqm_adapter_path(),
    }


def _backend_metadata(
    backend: Any,
    *,
    iqm_backend_name: str,
    iqm_use_metrics: bool,
    optimization_level: int,
) -> dict[str, Any]:
    return backend_metadata(
        backend,
        iqm_backend_name=iqm_backend_name,
        iqm_use_metrics=iqm_use_metrics,
        optimization_level=optimization_level,
        layout_method=None,
        routing_method=None,
    )


def _metric_row(circuit) -> dict[str, Any]:
    ops = dict(circuit.count_ops())
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "cz_count": int(ops.get("cz", 0)),
        "r_count": int(ops.get("r", 0)),
        "one_qubit_gate_count": int(
            sum(1 for instruction in circuit.data if len(instruction.qubits) == 1)
        ),
        "two_qubit_gate_count": int(
            sum(value for name, value in ops.items() if name in TWO_Q_GATES)
        ),
        "count_ops_json": json.dumps(ops, sort_keys=True),
    }


def _warning_flags(
    row: dict[str, Any],
    *,
    max_depth_warning: int,
    max_cz_warning: int,
) -> str:
    flags: list[str] = []
    depth = row.get("depth")
    cz_count = row.get("cz_count")
    if depth is not None and int(depth) > int(max_depth_warning):
        flags.append(f"depth_gt_{int(max_depth_warning)}")
    if cz_count is not None and int(cz_count) > int(max_cz_warning):
        flags.append(f"cz_gt_{int(max_cz_warning)}")
    return ";".join(flags)


def _candidate_identity(candidate: DirectBasisCandidate) -> dict[str, Any]:
    return {
        "class_name": candidate.class_name,
        "candidate_name": candidate.candidate_name,
        "basis_candidate_name": candidate.name,
        "basis_candidate_type": candidate.candidate_type,
    }


def _unsupported_candidate_row(
    candidate: DirectBasisCandidate,
    *,
    config: IqmTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        **metadata,
        **_candidate_identity(candidate),
        "strategy_name": "",
        "seed_transpiler": "",
        "success": False,
        "status": "unsupported_candidate",
        "error_type": "UnsupportedCandidate",
        "error_message": candidate.error_message,
        "compile_time_seconds": 0.0,
        "warning_flags": "failed_all_strategies",
    }
    row.update(
        {
            "num_qubits": None,
            "depth": None,
            "size": None,
            "cz_count": None,
            "r_count": None,
            "one_qubit_gate_count": None,
            "two_qubit_gate_count": None,
            "count_ops_json": "",
        }
    )
    return row


def _trial_row(
    candidate: DirectBasisCandidate,
    *,
    result,
    config: IqmTranspilerHarnessConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        **metadata,
        **_candidate_identity(candidate),
        "strategy_name": result.strategy_name,
        "seed_transpiler": result.seed_transpiler,
        "success": bool(result.success),
        "status": "ok" if result.success else "failed",
        "error_type": result.error_type,
        "error_message": result.error_message,
        "compile_time_seconds": result.compile_time_seconds,
    }
    if result.success and result.circuit is not None:
        row.update(_metric_row(result.circuit))
    else:
        row.update(
            {
                "num_qubits": None,
                "depth": None,
                "size": None,
                "cz_count": None,
                "r_count": None,
                "one_qubit_gate_count": None,
                "two_qubit_gate_count": None,
                "count_ops_json": "",
            }
        )
    row["warning_flags"] = _warning_flags(
        row,
        max_depth_warning=config.max_depth_warning,
        max_cz_warning=config.max_cz_warning,
    )
    return row


def _best_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("class_name", "")), str(row.get("candidate_name", "")))
        grouped.setdefault(key, []).append(row)

    best_rows: list[dict[str, Any]] = []
    for _, values in grouped.items():
        unsupported = [row for row in values if row.get("status") == "unsupported_candidate"]
        if unsupported:
            best_rows.append(dict(unsupported[0]))
            continue

        successful = [row for row in values if bool(row.get("success"))]
        if not successful:
            row = dict(values[0])
            row["status"] = "failed_all_strategies"
            row["warning_flags"] = "failed_all_strategies"
            best_rows.append(row)
            continue

        best_rows.append(
            dict(
                sorted(
                    successful,
                    key=lambda row: (
                        int(row["depth"]),
                        int(row["cz_count"]),
                        int(row["r_count"]),
                        int(row["size"]),
                    ),
                )[0]
            )
        )
    return best_rows


def _summary(rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(
            {
                (str(row.get("class_name", "")), str(row.get("candidate_name", "")))
                for row in rows
            }
        ),
        "trial_count": len(rows),
        "successful_trial_count": sum(1 for row in rows if bool(row.get("success"))),
        "failed_trial_count": sum(1 for row in rows if row.get("status") == "failed"),
        "unsupported_candidate_count": sum(
            1 for row in best_rows if row.get("status") == "unsupported_candidate"
        ),
        "failed_all_strategy_count": sum(
            1 for row in best_rows if row.get("status") == "failed_all_strategies"
        ),
    }


def run_iqm_transpiler_harness(
    config: IqmTranspilerHarnessConfig,
    *,
    strategy_runner: StrategyRunner = run_iqm_transpiler_strategy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    strategy_names = config.strategy_names or iqm_transpiler_strategy_names()
    candidates = list(config.candidates)
    metadata = {
        **_runtime_metadata(),
        **_backend_metadata(
            config.backend,
            iqm_backend_name=config.iqm_backend_name,
            iqm_use_metrics=config.iqm_use_metrics,
            optimization_level=config.optimization_level,
        ),
    }
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        if not candidate.is_supported:
            rows.append(_unsupported_candidate_row(candidate, config=config, metadata=metadata))
            continue

        circuit = build_direct_basis_graph_state_circuit(
            config.state_name,
            candidate.matrix,
            n_qutrits=config.n_qutrits,
        )
        for seed in range(int(config.n_transpile_runs)):
            for strategy_name in strategy_names:
                result = strategy_runner(
                    strategy_name,
                    circuit,
                    backend=config.backend,
                    seed_transpiler=seed,
                    optimization_level=config.optimization_level,
                )
                rows.append(
                    _trial_row(
                        candidate,
                        result=result,
                        config=config,
                        metadata=metadata,
                    )
                )

    best_rows = _best_trial_rows(rows)
    all_trials = pd.DataFrame(rows)
    best_by_candidate = pd.DataFrame(best_rows)
    summary = _summary(rows, best_rows)
    return all_trials, best_by_candidate, summary


def write_iqm_transpiler_harness_outputs(
    output_dir: str | Path,
    *,
    all_trials: pd.DataFrame,
    best_by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_trials_csv": str(root / "all_trials.csv"),
        "best_by_candidate_csv": str(root / "best_by_candidate.csv"),
        "summary_json": str(root / "summary.json"),
    }
    all_trials.to_csv(paths["all_trials_csv"], index=False)
    best_by_candidate.to_csv(paths["best_by_candidate_csv"], index=False)
    with open(paths["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return paths
```

- [ ] **Step 4: Run harness tests**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_harness -v
```

Expected: PASS.

- [ ] **Step 5: Run strategy and harness tests together**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_strategies tests.test_direct_basis_iqm_transpiler_harness -v
```

Expected: PASS.

- [ ] **Step 6: Commit harness core**

Run:

```powershell
git add -- src\qudits_on_qubits\benchmarks\direct_basis\iqm_transpiler_harness.py tests\test_direct_basis_iqm_transpiler_harness.py
git commit -m "feat: add iqm transpiler harness core"
```

---

### Task 3: IQM Transpiler Harness CLI

**Files:**
- Create: `scripts/run_iqm_transpiler_harness.py`
- Create: `tests/test_direct_basis_iqm_transpiler_harness_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_direct_basis_iqm_transpiler_harness_cli.py`:

```python
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.run_iqm_transpiler_harness import (
    _default_results_prefix,
    _load_candidates,
    build_parser,
    main,
)


class IqmTranspilerHarnessCliTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "sanity", "--iqm-backend", "garnet"]
        )

        self.assertEqual(args.state, "two_qutrit")
        self.assertEqual(args.candidate_set, "sanity")
        self.assertEqual(args.iqm_backend, "garnet")
        self.assertEqual(args.n_transpile_runs, 1)
        self.assertEqual(args.max_depth_warning, 100)
        self.assertEqual(args.max_cz_warning, 50)
        self.assertEqual(args.strategy, [])

    def test_from_old_csv_requires_old_csv(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "from-old-csv", "--iqm-backend", "garnet"]
        )

        with self.assertRaisesRegex(ValueError, "--old-csv is required"):
            _load_candidates(args)

    def test_default_results_prefix_contains_backend_state_and_candidate_set(self):
        args = build_parser().parse_args(
            [
                "--state",
                "two_qutrit",
                "--candidate-set",
                "from-old-csv",
                "--old-csv",
                "old.csv",
                "--iqm-backend",
                "garnet",
                "--n-transpile-runs",
                "3",
            ]
        )

        prefix = _default_results_prefix(args)

        self.assertEqual(prefix, "iqm_transpiler_harness_garnet_two_qutrit_from_old_csv_runs3")

    def test_main_wires_backend_and_harness_without_network(self):
        backend = object()
        candidates = [object()]
        output_paths = {
            "all_trials_csv": os.path.join("out", "all_trials.csv"),
            "best_by_candidate_csv": os.path.join("out", "best_by_candidate.csv"),
            "summary_json": os.path.join("out", "summary.json"),
        }

        with (
            patch("scripts.run_iqm_transpiler_harness._load_candidates", return_value=candidates),
            patch("scripts.run_iqm_transpiler_harness.load_iqm_backend", return_value=backend) as load_backend,
            patch(
                "scripts.run_iqm_transpiler_harness.default_iqm_transpiler_harness_output_dir",
                return_value="out",
            ),
            patch(
                "scripts.run_iqm_transpiler_harness.run_iqm_transpiler_harness",
                return_value=("all", "best", {"trial_count": 1}),
            ) as run_harness,
            patch(
                "scripts.run_iqm_transpiler_harness.write_iqm_transpiler_harness_outputs",
                return_value=output_paths,
            ) as write_outputs,
        ):
            return_code = main(
                [
                    "--state",
                    "two_qutrit",
                    "--candidate-set",
                    "sanity",
                    "--iqm-backend",
                    "garnet",
                    "--iqm-use-metrics",
                    "--strategy",
                    "preset_default",
                    "--strategy",
                    "transpile_to_iqm_default",
                    "--n-transpile-runs",
                    "2",
                    "--max-depth-warning",
                    "80",
                    "--max-cz-warning",
                    "40",
                ]
            )

        self.assertEqual(return_code, 0)
        load_backend.assert_called_once_with("garnet", use_metrics=True)
        config = run_harness.call_args.args[0]
        self.assertIs(config.backend, backend)
        self.assertEqual(config.iqm_backend_name, "garnet")
        self.assertEqual(config.strategy_names, ("preset_default", "transpile_to_iqm_default"))
        self.assertEqual(config.n_transpile_runs, 2)
        self.assertEqual(config.max_depth_warning, 80)
        self.assertEqual(config.max_cz_warning, 40)
        write_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_harness_cli -v
```

Expected: FAIL with `ModuleNotFoundError` for `scripts.run_iqm_transpiler_harness`.

- [ ] **Step 3: Implement CLI script**

Create `scripts/run_iqm_transpiler_harness.py`:

```python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    candidates_from_old_csv,
    generate_all_qutrit_u3_candidates,
    generate_legacy_qutrit_u3_candidates,
    generate_sanity_basis_candidates,
    generate_v2_stage1_direct_candidates,
    limit_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    load_iqm_backend,
    safe_backend_slug,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_harness import (
    IqmTranspilerHarnessConfig,
    default_iqm_transpiler_harness_output_dir,
    run_iqm_transpiler_harness,
    write_iqm_transpiler_harness_outputs,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    iqm_transpiler_strategy_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare IQM transpilation strategies for direct-basis candidates.",
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--n-qutrits", type=int, default=None)
    parser.add_argument(
        "--candidate-set",
        choices=("sanity", "all-qutrit-u3", "old_qutrit", "v2-stage1", "from-old-csv"),
        default="from-old-csv",
    )
    parser.add_argument("--old-csv", default=None)
    parser.add_argument("--random-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--iqm-backend", required=True)
    parser.add_argument("--iqm-use-metrics", action="store_true")
    parser.add_argument("--n-transpile-runs", type=int, default=1)
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=iqm_transpiler_strategy_names(),
        help="IQM transpiler strategy to run. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-depth-warning", type=int, default=100)
    parser.add_argument("--max-cz-warning", type=int, default=50)
    return parser


def _safe_filename_part(value) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "value"


def _default_results_prefix(args) -> str:
    return "_".join(
        [
            "iqm_transpiler_harness",
            safe_backend_slug(args.iqm_backend),
            _safe_filename_part(args.state),
            _safe_filename_part(args.candidate_set),
            f"runs{int(args.n_transpile_runs)}",
        ]
    )


def _load_candidates(args):
    if args.candidate_set == "sanity":
        candidates = generate_sanity_basis_candidates(
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.candidate_set == "all-qutrit-u3":
        candidates = generate_all_qutrit_u3_candidates()
    elif args.candidate_set == "old_qutrit":
        candidates = generate_legacy_qutrit_u3_candidates("old_qutrit")
    elif args.candidate_set == "v2-stage1":
        candidates = generate_v2_stage1_direct_candidates(include_unsupported=True)
    else:
        if not args.old_csv:
            raise ValueError("--old-csv is required for --candidate-set from-old-csv.")
        candidates = candidates_from_old_csv(args.old_csv, include_unsupported=True)
    return limit_candidates(candidates, args.limit_candidates)


def _output_dir_from_args(args) -> str:
    if args.output_dir:
        return args.output_dir
    run_id = args.run_id or _default_results_prefix(args)
    return default_iqm_transpiler_harness_output_dir(run_id)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        candidates = _load_candidates(args)
    except ValueError as exc:
        parser.error(str(exc))

    backend = load_iqm_backend(args.iqm_backend, use_metrics=args.iqm_use_metrics)
    config = IqmTranspilerHarnessConfig(
        state_name=args.state,
        n_qutrits=args.n_qutrits,
        backend=backend,
        iqm_backend_name=args.iqm_backend,
        iqm_use_metrics=args.iqm_use_metrics,
        candidates=candidates,
        strategy_names=tuple(args.strategy),
        n_transpile_runs=int(args.n_transpile_runs),
        max_depth_warning=int(args.max_depth_warning),
        max_cz_warning=int(args.max_cz_warning),
    )

    output_dir = _output_dir_from_args(args)
    print(
        f"Running IQM transpiler harness: state={args.state}, "
        f"candidates={len(candidates)}, backend={args.iqm_backend}, output={output_dir}"
    )
    all_trials, best_by_candidate, summary = run_iqm_transpiler_harness(config)
    paths = write_iqm_transpiler_harness_outputs(
        output_dir,
        all_trials=all_trials,
        best_by_candidate=best_by_candidate,
        summary=summary,
    )
    print(f"All trials: {paths['all_trials_csv']}")
    print(f"Best by candidate: {paths['best_by_candidate_csv']}")
    print(f"Summary: {paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_harness_cli -v
```

Expected: PASS.

- [ ] **Step 5: Run all new harness tests**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest tests.test_direct_basis_iqm_transpiler_strategies tests.test_direct_basis_iqm_transpiler_harness tests.test_direct_basis_iqm_transpiler_harness_cli -v
```

Expected: PASS.

- [ ] **Step 6: Commit CLI**

Run:

```powershell
git add -- scripts\run_iqm_transpiler_harness.py tests\test_direct_basis_iqm_transpiler_harness_cli.py
git commit -m "feat: add iqm transpiler harness cli"
```

---

### Task 4: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README section**

Append this section after the existing "IQM Direct-Basis Transpilation" section:

````markdown
## IQM Transpiler Harness

Use the harness to compare IQM-aware transpilation strategies for candidates
selected by earlier benchmark CSVs:

```powershell
python scripts/run_iqm_transpiler_harness.py `
  --state two_qutrit `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_from_old_csv_runs20_20260706_204350.csv `
  --iqm-backend garnet `
  --n-transpile-runs 3
```

The harness only transpiles circuits. It does not submit jobs to IQM hardware.
It writes:

```text
artifacts/iqm_runs/processed/transpiler_harness/<run_id>/
  all_trials.csv
  best_by_candidate.csv
  summary.json
```

Built-in strategies:

```text
preset_default
preset_exact
transpile_to_iqm_default
transpile_to_iqm_exact
```

`best_by_candidate.csv` chooses the best successful trial by
`(depth, cz_count, r_count, size)` and flags warning thresholds such as
`depth_gt_100` and `cz_gt_50`.
````

- [ ] **Step 2: Check CLI help**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe scripts\run_iqm_transpiler_harness.py --help
```

Expected: output includes `--strategy`, `--old-csv`, `--iqm-backend`, `--max-depth-warning`, and `--max-cz-warning`.

- [ ] **Step 3: Commit docs**

Run:

```powershell
git add -- README.md
git commit -m "docs: document iqm transpiler harness"
```

---

### Task 5: Verification And Manual Real-Backend Smoke

**Files:**
- All changed files

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe -m unittest discover -s tests -v
```

Expected: PASS. Tests that rely on unavailable optional IQM fake backends may skip by raising `unittest.SkipTest`.

- [ ] **Step 2: Run CLI help**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe scripts\run_iqm_transpiler_harness.py --help
```

Expected: exit code 0 and help text lists all harness options.

- [ ] **Step 3: Run real-backend harness smoke when `.env` is present**

Run:

```powershell
C:\Users\szymo\anaconda3\envs\qudityD3_laptop\python.exe scripts\run_iqm_transpiler_harness.py `
  --state two_qutrit `
  --candidate-set from-old-csv `
  --old-csv artifacts\iqm_runs\raw\direct_basis_iqm_garnet_two_qutrit_from_old_csv_runs20_20260706_204350.csv `
  --iqm-backend garnet `
  --n-transpile-runs 1 `
  --run-id smoke_two_qutrit_garnet
```

Expected:

```text
Running IQM transpiler harness: state=two_qutrit, candidates=11, backend=garnet, output=...
All trials: ...\all_trials.csv
Best by candidate: ...\best_by_candidate.csv
Summary: ...\summary.json
```

Open `artifacts\iqm_runs\processed\transpiler_harness\smoke_two_qutrit_garnet\best_by_candidate.csv` and verify:

```text
candidate_name includes E_old
strategy_name is one of the built-in strategy names
depth for the 11 candidates is around 22-35 in the current environment
cz_count for the 11 candidates is around 13-18 in the current environment
```

If `.env` is missing or the token is expired, record that the real-backend smoke was skipped and keep the unit-test verification as the required completion gate.

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only unrelated pre-existing user changes remain:

```text
 M notebooks/meas_settings_2qutryt.ipynb
 M notebooks/working/meas_settings_2qutryt.ipynb
 M src/qudits_on_qubits/bell_measurements/__init__.py
 M src/qudits_on_qubits/bell_measurements/sampler_circuits.py
?? notebooks/working/iqm_meas_test.ipynb
?? tests/test_bell_measurements_iqm_runner.py
```

- [ ] **Step 5: Commit any final verification-only doc tweak**

If README or plan text needed a correction during verification, commit only that correction:

```powershell
git add -- README.md docs\superpowers\plans\2026-07-07-iqm-transpiler-harness.md
git commit -m "docs: refine iqm transpiler harness notes"
```

If no files changed during verification, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers a separate harness, IQM strategy comparison, `from-old-csv` candidate reuse, all-trials and best-by-candidate CSVs, metadata capture, warning thresholds, no hardware execution, fake-backend tests, CLI, README, and manual real-backend smoke.
- Placeholder scan: The plan contains no TBD markers, no incomplete paths, and no undefined future functions that are not introduced in earlier tasks.
- Type consistency: The strategy layer defines `IqmTranspilerStrategyResult`; the harness tests and implementation use that exact type. The harness defines `IqmTranspilerHarnessConfig`; the CLI imports and constructs that exact type.
