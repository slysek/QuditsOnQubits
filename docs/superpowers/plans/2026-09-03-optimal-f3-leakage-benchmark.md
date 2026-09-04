# Optimal F3 Leakage-Phase Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in, paired comparison of full qutrit graph-state circuits using the historical `F3 ⊕ 1` embedding and the analytically optimal monomial-basis leakage phase.

**Architecture:** Put the basis-dependent phase calculation in Qiskit-free numerical helpers, add a Fourier-explicit graph-state circuit builder beside the unchanged historical builder, and compile baseline/optimal circuit pairs through the existing direct-basis transpiler path. Expose the comparison through one CLI flag and additive CSV/QPY fields so existing benchmark runs remain compatible.

**Tech Stack:** Python 3.11+, NumPy, Qiskit 2.x, pandas, pytest/unittest, QPY.

**Implementation status (2026-09-04):** Tasks 1–4 are implemented and tested.
Comparison tests live in the focused `tests/test_direct_basis_f3_comparison.py`
module. Tasks 3 and 4 share a final integration commit. Direct-basis verification
passes 346 tests and 86 subtests. Final repository-wide verification: 1581
passed, 7 skipped, 383 passing subtests, and the same 17 pre-existing failures
as the baseline (Qiskit/QPY/IQM dependency incompatibilities). No new failures.
A real local CLI smoke run also wrote three successful comparison rows and QPY
artifacts. Canonical full-graph metrics: depth 48→45, size 96→89, 2Q 24→22,
1Q 72→67. No QPU jobs were submitted. Branch is preserved for review; automatic
merge/PR is not performed with the repository-wide environment checks failing.

**Verification correction:** Real exact CZ synthesis showed that qutrit-labelled
`C12*C21` must be `C[1,2]*C[2,1]`, after local-X normalization of the unused
state to `|11>`. The snippets below have been corrected accordingly. Canonical
E_Z has phase `11π/6`. All PowerShell test commands also require
`$env:PYTHONPATH=(Join-Path (Get-Location) 'src')` in the relocated worktree.

---

### Task 1: Analytic monomial F3 leakage phase

**Files:**
- Create: `tests/test_direct_basis_f3_leakage_phase.py`
- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/math_utils.py`

- [x] **Step 1: Write failing numerical tests**

Create tests that construct `E = B_s D P` from the production monomial
generator and assert the desired public API:

```python
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    F3LeakagePhaseAnalysis,
    encoding_embedding,
    optimal_f3_leakage_phase,
    physical_single_qutrit_gate_in_encoding,
    qutrit_fourier,
)
from qudits_on_qubits.core.benchmark_encoding_bases import (
    generate_monomial_full_bases,
)


def _permutation_matrix(permutation: tuple[int, int, int]) -> np.ndarray:
    return np.eye(3, dtype=complex)[list(permutation)]


@pytest.mark.parametrize("permutation", list(permutations(range(3))))
def test_optimal_phase_for_effective_f3_permutations(permutation):
    encoding = np.eye(4, 3, dtype=complex) @ _permutation_matrix(permutation)

    analysis = optimal_f3_leakage_phase(encoding)

    expected = 11 * np.pi / 6 if permutation in {(0, 1, 2), (0, 2, 1)} else np.pi / 2
    assert isinstance(analysis, F3LeakagePhaseAnalysis)
    assert analysis.phase == pytest.approx(expected)
    assert abs(abs(analysis.phase_factor) - 1.0) < 1e-12
    assert analysis.support == (0, 1, 2)


def test_all_generated_monomial_full_phases_are_analytic_two_value_family():
    phases = {
        round(optimal_f3_leakage_phase(encoding).phase / np.pi, 12)
        for _, _, encoding in generate_monomial_full_bases(max_candidates=None)
    }

    assert phases == {0.5, round(11 / 6, 12)}


def test_diagonal_monomial_phases_do_not_change_optimal_phase():
    permutation = _permutation_matrix((1, 2, 0))
    first = np.eye(4, 3, dtype=complex) @ permutation
    diagonal = np.diag(np.exp(1j * np.array([0.2, -0.7, 1.3])))
    second = np.eye(4, 3, dtype=complex) @ diagonal @ permutation

    assert optimal_f3_leakage_phase(first).phase == pytest.approx(
        optimal_f3_leakage_phase(second).phase
    )


def test_support_is_ordered_by_local_x_mapping_unused_state_to_11():
    support_embedding = np.eye(4, dtype=complex)[:, [0, 2, 3]]
    permutation = _permutation_matrix((2, 0, 1))
    encoding = support_embedding @ permutation

    analysis = optimal_f3_leakage_phase(encoding)

    assert analysis.support == (0, 2, 3)
    np.testing.assert_allclose(
        analysis.effective_fourier,
        encoding[[2, 3, 0], :] @ qutrit_fourier() @ encoding[[2, 3, 0], :].conj().T,
    )


def test_dense_encoding_is_not_accepted_as_monomial():
    with pytest.raises(ValueError, match="monomial"):
        optimal_f3_leakage_phase(qutrit_fourier())


def test_phase_embedding_changes_only_the_leakage_complement():
    encoding = np.eye(4, 3, dtype=complex)
    analysis = optimal_f3_leakage_phase(encoding)
    baseline = physical_single_qutrit_gate_in_encoding(qutrit_fourier(), encoding)
    optimal = physical_single_qutrit_gate_in_encoding(
        qutrit_fourier(), encoding, leakage_phase=analysis.phase
    )
    projector = encoding_embedding(encoding) @ encoding_embedding(encoding).conj().T

    np.testing.assert_allclose(projector @ baseline @ projector, projector @ optimal @ projector)
    np.testing.assert_allclose(optimal[3, 3], analysis.phase_factor)


@pytest.mark.parametrize("phase", [True, np.inf, np.nan, "11pi/6"])
def test_phase_embedding_rejects_invalid_phase(phase):
    with pytest.raises(ValueError, match="leakage_phase"):
        physical_single_qutrit_gate_in_encoding(
            qutrit_fourier(), np.eye(3), leakage_phase=phase
        )
```

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q -p no:cacheprovider tests/test_direct_basis_f3_leakage_phase.py
```

Expected: collection fails because `F3LeakagePhaseAnalysis` and
`optimal_f3_leakage_phase` do not exist.

- [x] **Step 3: Implement the pure phase analysis and phased embedding**

In `math_utils.py`, import `dataclass` and `Real`, define the immutable result,
and add the helper below. Extend `physical_single_qutrit_gate_in_encoding` with
the shown keyword argument while retaining `0.0` as the compatibility default.

```python
@dataclass(frozen=True)
class F3LeakagePhaseAnalysis:
    phase: float
    phase_factor: complex
    support: tuple[int, int, int]
    effective_fourier: np.ndarray


def optimal_f3_leakage_phase(
    encoding: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> F3LeakagePhaseAnalysis:
    e_new = encoding_embedding(encoding)
    logical_to_physical: list[int] = []
    for logical_level in range(3):
        occupied = np.flatnonzero(np.abs(e_new[:, logical_level]) > tolerance)
        if len(occupied) != 1:
            raise ValueError("encoding must be monomial")
        logical_to_physical.append(int(occupied[0]))
    if len(set(logical_to_physical)) != 3:
        raise ValueError("encoding must be monomial with distinct support rows")

    support = tuple(sorted(logical_to_physical))
    unused = next(index for index in range(4) if index not in support)
    mask = unused ^ 3
    effective_basis = e_new[[index ^ mask for index in range(3)], :]
    if not is_unitary(effective_basis, tol=tolerance):
        raise ValueError("effective monomial basis must be unitary")
    effective_fourier = effective_basis @ qutrit_fourier() @ effective_basis.conj().T
    product_12_21 = effective_fourier[1, 2] * effective_fourier[2, 1]
    if abs(product_12_21) <= tolerance:
        raise ValueError("optimal F3 leakage phase is undefined when C12*C21 is zero")
    phase_factor = product_12_21 / (
        np.linalg.det(effective_fourier) * np.conj(product_12_21)
    )
    phase_factor /= abs(phase_factor)
    phase = float(np.mod(np.angle(phase_factor), 2 * np.pi))
    if np.isclose(phase, 2 * np.pi, atol=tolerance, rtol=0.0):
        phase = 0.0
    return F3LeakagePhaseAnalysis(
        phase=phase,
        phase_factor=complex(phase_factor),
        support=support,
        effective_fourier=effective_fourier,
    )
```

```python
def physical_single_qutrit_gate_in_encoding(
    qutrit_gate: np.ndarray,
    encoding: np.ndarray,
    *,
    leakage_phase: float = 0.0,
) -> np.ndarray:
    if (
        isinstance(leakage_phase, bool)
        or not isinstance(leakage_phase, Real)
        or not np.isfinite(leakage_phase)
    ):
        raise ValueError("leakage_phase must be a finite real number")
    gate = validate_unitary(qutrit_gate, 3, name="qutrit_gate")
    e_new = encoding_embedding(encoding)
    projector = e_new @ e_new.conj().T
    embedded = (
        e_new @ gate @ e_new.conj().T
        + np.exp(1j * float(leakage_phase))
        * (np.eye(4, dtype=complex) - projector)
    )
    if not is_unitary(embedded):
        raise ValueError("Encoded single-qutrit physical gate is not unitary.")
    return embedded
```

- [x] **Step 4: Run tests and verify GREEN**

Run the Task 1 command. Expected: all tests pass.

- [x] **Step 5: Commit the numerical unit**

```powershell
git add -- tests/test_direct_basis_f3_leakage_phase.py src/qudits_on_qubits/benchmarks/direct_basis/math_utils.py
git commit -m "feat: calculate optimal monomial F3 leakage phase"
```

### Task 2: Fourier-explicit graph-state circuits

**Files:**
- Modify: `tests/test_direct_basis_f3_leakage_phase.py`
- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/circuits.py`

- [x] **Step 1: Add failing graph-circuit tests**

Append tests which import `build_direct_basis_fourier_gate` and
`build_direct_basis_fourier_graph_state_circuit`:

```python
from qiskit.quantum_info import Operator, Statevector

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_fourier_gate,
    build_direct_basis_fourier_graph_state_circuit,
)


def test_fourier_gate_carries_requested_leakage_phase():
    encoding = np.eye(4, 3, dtype=complex)
    phase = optimal_f3_leakage_phase(encoding).phase
    gate = build_direct_basis_fourier_gate(encoding, leakage_phase=phase)

    np.testing.assert_allclose(Operator(gate).data[3, 3], np.exp(1j * phase))


def test_full_graph_circuits_are_equivalent_and_contain_explicit_f3_per_qutrit():
    encoding = np.eye(4, 3, dtype=complex)
    phase = optimal_f3_leakage_phase(encoding).phase
    baseline = build_direct_basis_fourier_graph_state_circuit(
        "two_qutrit", encoding, leakage_phase=0.0
    )
    optimal = build_direct_basis_fourier_graph_state_circuit(
        "two_qutrit", encoding, leakage_phase=phase
    )

    assert baseline.count_ops()["F3_W"] == 2
    assert optimal.count_ops()["F3_W"] == 2
    assert Statevector.from_instruction(baseline).equiv(Statevector.from_instruction(optimal))
```

- [x] **Step 2: Run the focused test and verify RED**

Run the Task 1 pytest command. Expected: import failure for the new graph-state
builder or a signature failure for `leakage_phase`.

- [x] **Step 3: Implement the phase-aware gate and full-circuit builder**

Change `build_direct_basis_fourier_gate` to pass `leakage_phase` into the
physical embedding. Add a new builder which prepares `E|0>`, appends one F3 per
qutrit, and reuses the existing edge ordering:

```python
def build_direct_basis_fourier_gate(
    encoding: np.ndarray,
    *,
    leakage_phase: float = 0.0,
) -> UnitaryGate:
    embedded = physical_single_qutrit_gate_in_encoding(
        qutrit_fourier(), encoding, leakage_phase=leakage_phase
    )
    gate = UnitaryGate(embedded, label="F3_W")
    gate.name = "F3_W"
    return gate


def build_direct_basis_fourier_graph_state_circuit(
    state_name: str,
    basis_matrix: np.ndarray,
    *,
    leakage_phase: float,
    n_qutrits: int | None = None,
) -> QuantumCircuit:
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    qubit_pairs = [[2 * index, 2 * index + 1] for index in range(state.num_qutrits)]
    qc = QuantumCircuit(2 * state.num_qutrits, name=f"{state.state_id}_direct_basis_f3")
    encoded_zero = encoding_embedding(basis_matrix)[:, 0]
    zero_preparation = StatePreparation(encoded_zero, label="zero_W")
    fourier = build_direct_basis_fourier_gate(
        basis_matrix, leakage_phase=leakage_phase
    )
    edge_gate = build_direct_basis_edge_gate(basis_matrix)
    for pair in qubit_pairs:
        qc.append(zero_preparation, pair)
        qc.append(fourier, pair)
    for left, right in state.edges:
        if left == right:
            continue
        left_pair = qubit_pairs[state.num_qutrits - 1 - left]
        right_pair = qubit_pairs[state.num_qutrits - 1 - right]
        qc.append(
            edge_gate,
            [left_pair[0], left_pair[1], right_pair[0], right_pair[1]],
        )
    return qc
```

Import `encoding_embedding` into `circuits.py`.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 pytest command. Expected: all phase and circuit tests pass.

- [x] **Step 5: Commit the circuit unit**

```powershell
git add -- tests/test_direct_basis_f3_leakage_phase.py src/qudits_on_qubits/benchmarks/direct_basis/circuits.py
git commit -m "feat: build graph states with explicit phased F3 gates"
```

### Task 3: Paired full-circuit benchmark metrics

**Files:**
- Modify: `tests/test_direct_basis_f3_leakage_phase.py`
- Modify: `tests/test_direct_basis_iqm_benchmark.py`
- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py`

- [x] **Step 1: Add failing paired-comparison tests**

Import `QuantumCircuit`, `patch`, and
`benchmark_optimal_f3_graph_comparison`, then add these tests:

```python
def test_full_graph_comparison_pairs_seeds_and_reports_positive_improvements():
    calls = []

    def fake_transpile(circuit, *, trial, **kwargs):
        variant = "optimal" if circuit.name.endswith("_optimal") else "baseline"
        calls.append((variant, trial))
        output = QuantumCircuit(circuit.num_qubits)
        output.cz(0, 1)
        if variant == "baseline":
            output.cz(2, 3)
        return output

    with patch(
        "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
        side_effect=fake_transpile,
    ):
        result = benchmark_optimal_f3_graph_comparison(
            state_name="two_qutrit",
            basis_matrix=np.eye(4, 3, dtype=complex),
            n_qutrits=2,
            coupling_map=[[0, 1], [1, 2], [2, 3]],
            basis_gates=("cz", "rz", "sx", "x"),
            n_transpile_runs=2,
            transpiler_backend=None,
            optimization_level=3,
            layout_method=None,
            routing_method=None,
            approximation_degree=None,
            iqm_strategy_names=None,
        )

    assert calls == [
        ("baseline", 0),
        ("optimal", 0),
        ("baseline", 1),
        ("optimal", 1),
    ]
    assert result["f3_graph_comparison_status"] == "ok"
    assert result["f3_graph_successful_pairs"] == 2
    assert result["f3_optimal_leakage_phase_over_pi"] == pytest.approx(11 / 6)
    assert result["f3_graph_two_qubit_gate_count_improvement"] == 1
    assert result["f3_graph_optimal_is_better"] is True


def test_full_graph_comparison_skips_dense_non_monomial_encoding():
    with patch(
        "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
        side_effect=AssertionError("transpiler must not run"),
    ):
        result = benchmark_optimal_f3_graph_comparison(
            state_name="two_qutrit",
            basis_matrix=qutrit_fourier(),
            n_qutrits=2,
            coupling_map=[[0, 1]],
            basis_gates=("cz", "rz", "sx", "x"),
            n_transpile_runs=1,
            transpiler_backend=None,
            optimization_level=3,
            layout_method=None,
            routing_method=None,
            approximation_degree=None,
            iqm_strategy_names=None,
        )

    assert result["f3_graph_comparison_status"] == "not_monomial"


def test_full_graph_comparison_reports_failed_pairs():
    with patch(
        "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
        side_effect=RuntimeError("compile failed"),
    ):
        result = benchmark_optimal_f3_graph_comparison(
            state_name="two_qutrit",
            basis_matrix=np.eye(4, 3, dtype=complex),
            n_qutrits=2,
            coupling_map=[[0, 1]],
            basis_gates=("cz", "rz", "sx", "x"),
            n_transpile_runs=2,
            transpiler_backend=None,
            optimization_level=3,
            layout_method=None,
            routing_method=None,
            approximation_degree=None,
            iqm_strategy_names=None,
        )

    assert result["f3_graph_comparison_status"] == "all_transpile_failed"
    assert result["f3_graph_successful_pairs"] == 0
    assert result["f3_graph_failed_pairs"] == 2
```

In `test_direct_basis_iqm_benchmark.py`, assert
`benchmark_direct_basis_candidates(..., compare_optimal_f3_leakage=True)`
forwards the flag into each `benchmark_direct_basis` call.

- [x] **Step 2: Run focused tests and verify RED**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q -p no:cacheprovider tests/test_direct_basis_f3_leakage_phase.py tests/test_direct_basis_iqm_benchmark.py
```

Expected: failures for the missing comparison API and keyword forwarding.

- [x] **Step 3: Implement result defaults and paired compilation**

Add imports for the new circuit builder and phase helper. Extend `_base_row`
with empty comparison/artifact fields and `not_requested` status. Add these
metric helpers and implement the comparison routine:

```python
def _f3_graph_metrics(circuit) -> dict:
    operations = circuit.count_ops()
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "two_qubit_gate_count": _count_two_qubit_gates_from_ops(operations),
        "one_qubit_gate_count": _count_one_qubit_gates(circuit),
        "count_ops": dict(operations),
    }


def _f3_graph_rank(metrics: dict, *, transpiler_backend) -> tuple[int, ...]:
    if transpiler_backend is not None:
        return (
            metrics["depth"],
            metrics["two_qubit_gate_count"],
            metrics["one_qubit_gate_count"],
            metrics["size"],
        )
    return (
        metrics["depth"],
        metrics["two_qubit_gate_count"],
        metrics["size"],
    )


def benchmark_optimal_f3_graph_comparison(
    *,
    state_name: str,
    basis_matrix: np.ndarray,
    n_qutrits: int | None,
    coupling_map,
    basis_gates,
    n_transpile_runs: int,
    transpiler_backend,
    optimization_level: int,
    layout_method: str | None,
    routing_method: str | None,
    approximation_degree: float | None,
    iqm_strategy_names: Iterable[str] | None,
    quantum_circuits_dir: str | None = None,
    class_name: str = "",
    candidate_name: str = "",
) -> dict:
    result = {
        "f3_graph_comparison_status": "analysis_error",
        "f3_graph_comparison_error": "",
        "f3_graph_successful_pairs": 0,
        "f3_graph_failed_pairs": 0,
    }
    try:
        analysis = optimal_f3_leakage_phase(basis_matrix)
    except ValueError as exc:
        message = str(exc)
        result["f3_graph_comparison_status"] = (
            "not_monomial" if "monomial" in message else "analysis_error"
        )
        result["f3_graph_comparison_error"] = message
        return result

    result.update(
        {
            "f3_optimal_leakage_phase": analysis.phase,
            "f3_optimal_leakage_phase_over_pi": analysis.phase / np.pi,
            "f3_optimal_leakage_phase_real": analysis.phase_factor.real,
            "f3_optimal_leakage_phase_imag": analysis.phase_factor.imag,
        }
    )
    baseline = build_direct_basis_fourier_graph_state_circuit(
        state_name,
        basis_matrix,
        leakage_phase=0.0,
        n_qutrits=n_qutrits,
    )
    baseline.name = f"{baseline.name}_baseline"
    optimal = build_direct_basis_fourier_graph_state_circuit(
        state_name,
        basis_matrix,
        leakage_phase=analysis.phase,
        n_qutrits=n_qutrits,
    )
    optimal.name = f"{optimal.name}_optimal"

    baseline_runs = []
    optimal_runs = []
    strategy_names = tuple(iqm_strategy_names or ())
    last_error = ""
    for trial in range(int(n_transpile_runs)):
        active_strategies = (
            strategy_names
            if transpiler_backend is not None and strategy_names
            else (None,)
        )
        for strategy_index, strategy_name in enumerate(active_strategies):
            common = {
                "trial": trial,
                "transpiler_backend": transpiler_backend,
                "basis_gates": basis_gates,
                "coupling_map": coupling_map,
                "optimization_level": int(optimization_level),
                "layout_method": layout_method,
                "routing_method": routing_method,
                "approximation_degree": approximation_degree,
                "iqm_strategy_name": strategy_name,
            }
            try:
                compiled_baseline = _transpile_one_trial(baseline, **common)
                compiled_optimal = _transpile_one_trial(optimal, **common)
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except BaseException as exc:
                result["f3_graph_failed_pairs"] += 1
                last_error = f"{type(exc).__name__}: {exc}".splitlines()[0]
                continue
            for runs, compiled in (
                (baseline_runs, compiled_baseline),
                (optimal_runs, compiled_optimal),
            ):
                runs.append(
                    {
                        **_f3_graph_metrics(compiled),
                        "seed_transpiler": trial,
                        "strategy_name": strategy_name or "",
                        "strategy_index": strategy_index,
                    }
                )
            result["f3_graph_successful_pairs"] += 1

    if not baseline_runs:
        result["f3_graph_comparison_status"] = "all_transpile_failed"
        result["f3_graph_comparison_error"] = last_error
        return result

    best_by_variant = {}
    for prefix, runs in (
        ("f3_graph_baseline", baseline_runs),
        ("f3_graph_optimal", optimal_runs),
    ):
        best = min(
            runs,
            key=lambda item: _f3_graph_rank(
                item, transpiler_backend=transpiler_backend
            ),
        )
        best_by_variant[prefix] = best
        for metric in (
            "depth",
            "size",
            "two_qubit_gate_count",
            "one_qubit_gate_count",
        ):
            result[f"{prefix}_best_{metric}"] = best[metric]
            result[f"{prefix}_mean_{metric}"] = round(
                float(np.mean([item[metric] for item in runs])), 6
            )
        result[f"{prefix}_best_count_ops"] = json.dumps(
            best["count_ops"], sort_keys=True
        )
        result[f"{prefix}_best_seed_transpiler"] = best["seed_transpiler"]
        result[f"{prefix}_best_strategy"] = best["strategy_name"]

    baseline_best = best_by_variant["f3_graph_baseline"]
    optimal_best = best_by_variant["f3_graph_optimal"]
    for metric in (
        "depth",
        "size",
        "two_qubit_gate_count",
        "one_qubit_gate_count",
    ):
        result[f"f3_graph_{metric}_improvement"] = (
            baseline_best[metric] - optimal_best[metric]
        )
    result["f3_graph_optimal_is_better"] = _f3_graph_rank(
        optimal_best, transpiler_backend=transpiler_backend
    ) < _f3_graph_rank(
        baseline_best, transpiler_backend=transpiler_backend
    )
    result["f3_graph_comparison_status"] = "ok"
    return result
```

- [x] **Step 4: Wire the comparison through candidate execution**

Add `compare_optimal_f3_leakage: bool = False` to `benchmark_direct_basis`,
`_benchmark_direct_basis_candidate_group`, and
`benchmark_direct_basis_candidates`. Place it in the shared group keyword map.
After backend/basis-gate resolution and before the historical transpilation
loop, merge the comparison result only when the flag is true. Leave the existing
primary circuit, rank keys, and metrics untouched.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Task 3 pytest command. Expected: all tests pass.

- [x] **Step 6: Commit the benchmark unit**

```powershell
git add -- tests/test_direct_basis_f3_leakage_phase.py tests/test_direct_basis_iqm_benchmark.py src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py
git commit -m "feat: compare optimal F3 phase in full graph benchmarks"
```

### Task 4: CLI, QPY artifacts, and user documentation

**Files:**
- Modify: `tests/test_direct_basis_f3_leakage_phase.py`
- Modify: `tests/test_direct_basis_iqm_cli.py`
- Modify: `scripts/run_direct_basis_benchmarks.py`
- Modify: `src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py`
- Modify: `README.md`

- [x] **Step 1: Add failing CLI and artifact tests**

Add parser assertions:

```python
args = build_parser().parse_args(["--state", "two_qutrit"])
self.assertFalse(args.compare_optimal_f3_leakage)

args = build_parser().parse_args(
    ["--state", "two_qutrit", "--compare-optimal-f3-leakage"]
)
self.assertTrue(args.compare_optimal_f3_leakage)
```

Extend the mocked `main` test to assert
`benchmark_kwargs["compare_optimal_f3_leakage"] is True` when the flag is
present. In the phase test module, run one comparison in a temporary directory
with a fake compiler and assert the four non-empty QPY paths exist and load one
circuit each:

```text
F3_W_phi0.qpy
F3_W_phi_optimal.qpy
graph_state_f3_baseline.qpy
graph_state_f3_optimal.qpy
```

- [x] **Step 2: Run focused tests and verify RED**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q -p no:cacheprovider tests/test_direct_basis_f3_leakage_phase.py tests/test_direct_basis_iqm_cli.py
```

Expected: parser/forwarding and artifact-path assertions fail.

- [x] **Step 3: Add CLI forwarding**

Add this parser option and pass it to `benchmark_direct_basis_candidates`:

```python
parser.add_argument(
    "--compare-optimal-f3-leakage",
    action="store_true",
    help=(
        "For monomial encodings, compare full graph-state circuits using "
        "F3 leakage phases 0 and the analytic per-basis optimum."
    ),
)
```

- [x] **Step 4: Export comparison source artifacts**

When the comparison analysis succeeds and `quantum_circuits_dir` is provided,
write the two full source circuits and two local F3 circuits into the existing
candidate output directory with `_save_qpy`. Return their paths in the row using
the four artifact fields specified by the design. Do not alter historical
`F3_W.qpy` or `graph_state_direct_basis.qpy`.

- [x] **Step 5: Document invocation and result semantics**

Add a concise README example:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set v2-stage1 --compare-optimal-f3-leakage
```

Document that the analytic comparison applies only to monomial rows and that
positive `f3_graph_*_improvement` values mean fewer gates/layers in the optimal
circuit.

- [x] **Step 6: Run focused tests and verify GREEN**

Run the Task 4 pytest command. Expected: all tests pass.

- [x] **Step 7: Commit the interface unit**

```powershell
git add -- README.md scripts/run_direct_basis_benchmarks.py src/qudits_on_qubits/benchmarks/direct_basis/benchmark.py tests/test_direct_basis_f3_leakage_phase.py tests/test_direct_basis_iqm_cli.py
git commit -m "feat: expose optimal F3 graph comparison"
```

### Task 5: Regression and acceptance verification

**Files:**
- Modify only if a test reveals a feature regression.

- [x] **Step 1: Run the new feature tests**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q -p no:cacheprovider tests/test_direct_basis_f3_leakage_phase.py tests/test_direct_basis_iqm_benchmark.py tests/test_direct_basis_iqm_cli.py
```

Expected: all pass.

- [x] **Step 2: Run the complete direct-basis regression selection**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$tests = Get-ChildItem -LiteralPath tests -Filter 'test_direct_basis*.py' | ForEach-Object { $_.FullName }
python -m pytest -q -p no:cacheprovider @tests
```

Expected baseline: at least the previously observed 280 tests and 86 subtests
pass, with no failures.

- [x] **Step 3: Run the repository-wide suite**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q -p no:cacheprovider
```

Expected: no new failures relative to the recorded baseline. Existing baseline
has 17 environment/artifact failures, 1515 passes, 7 skips, and 383 passing
subtests due to incompatible global IQM/Qiskit distributions and historical QPY
artifacts.

- [x] **Step 4: Verify diff quality and acceptance criteria**

```powershell
git diff --check main...HEAD
git status --short
git log --oneline main..HEAD
```

Expected: no whitespace errors; only intended source, test, README, spec, and
plan changes; every acceptance criterion from the design has a corresponding
test.

- [x] **Step 5: Record final verification commit if needed**

If verification required code changes, stage only those intended files and use:

```powershell
git commit -m "fix: complete optimal F3 benchmark verification"
```

If no changes were required, do not create an empty commit.
