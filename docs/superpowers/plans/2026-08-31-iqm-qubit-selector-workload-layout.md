# IQM Qubit Selector Workload Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Implementation correction from the live compile smoke:** The detailed steps
> below are a historical execution plan. `iqm-qubit-selector` 1.1.2 returns
> unordered physical routing subgraphs, not ordered logical-to-physical layouts,
> and a subgraph may be wider than the logical circuit. The implemented selector
> path canonical-sorts and deduplicates candidates as sets. Both generated
> candidates and selector-enabled explicit `initial_layouts` are routing
> restrictions compiled with
> `transpile_to_IQM(..., restrict_to_qubits=list(subgraph))`, never by passing a
> selector candidate as `initial_layout`. Restricted local-index circuits are
> then inflated to full
> backend width at their real provider indices. Outside selector mode,
> `TranspilationConfig(initial_layout=...)` retains ordered logical-to-physical
> semantics. Any later selector-specific snippet that assumes ordered values,
> exact logical width, or direct `initial_layout` compilation is superseded by
> this correction.

**Goal:** Integrate IQM's calibration-aware qubit selector into the shared experiment pipeline, then validate and rank its Top-K layouts over the complete GHZ Bell workload before any hardware submission.

**Architecture:** Add a JSON-safe selector configuration model, keep IQM package imports and provider error handling inside `IQMAdapter.suggest_layouts`, and let `_compile_measurement_workload` merge generated layouts with explicit baselines before its existing layout-by-seed full-workload search. The notebook becomes only the first configuration consumer; Aer and PIAST-Q retain their current behavior.

**Tech Stack:** Python 3.11-3.13, Qiskit 2.x, `iqm-client[qiskit]` 35.x, `iqm-qubit-selector` 1.1.x, frozen dataclasses, pytest, Jupyter notebook JSON.

**Git authorization:** This continues the user-requested PR on
`codex/ghz-bell-optimization-1-3-current`. Task commits and the final push update
that existing PR; never merge it from this plan.

**Test interpreter:** The checkout `.venv` is stale and points at a removed
Microsoft Store interpreter. Resolve the notebook's existing
`QuditsOnQubitsEnv` Conda environment once per PowerShell session:

```powershell
$testPython = (Resolve-Path (
    Join-Path $env:USERPROFILE '.conda\envs\QuditsOnQubitsEnv\python.exe'
)).Path
$qoqWorktreeSource = (Resolve-Path 'src').Path
$env:PYTHONPATH = $qoqWorktreeSource
```

This prevents the environment's editable install from importing the dirty owning
checkout instead of this PR worktree. Run each command block in that initialized
PowerShell session.

**Official IQM references:**

- `CostEvaluator` 1.1 API: <https://docs.iqm.tech/iqm-qubit-selector/api/iqm.qubit_selector.qubit_selector.CostEvaluator.html>
- Qubit Selector workflow: <https://docs.iqm.tech/iqm-qubit-selector/qubit_selector_quick_start.html>

---

## File Map

- Modify `src/qudits_on_qubits/experiments/models.py`: selector configuration and workload-search compatibility.
- Modify `src/qudits_on_qubits/experiments/__init__.py`: public experiment export.
- Modify `src/qudits_on_qubits/__init__.py`: lazy top-level export.
- Modify `src/qudits_on_qubits/experiments/backends/iqm.py`: IQM selector invocation, normalization, and safe errors.
- Modify `src/qudits_on_qubits/experiments/runner.py`: representative selection, generated/explicit merge, full-workload metadata.
- Modify `notebooks/ghz3_bell_canonical_baseline.ipynb`: enable the shared selector configuration.
- Modify `pyproject.toml`, `requirements.txt`, `src/qudits_on_qubits.egg-info/PKG-INFO`, and `src/qudits_on_qubits.egg-info/requires.txt`: require selector 1.1.x API.
- Modify `README.md`: document pipeline-level automatic layout search.
- Modify `tests/test_experiment_models.py`: model validation and round trips.
- Modify `tests/test_public_api.py`: public export contract.
- Modify `tests/test_experiment_iqm_adapter.py`: selector adapter contract and error handling.
- Modify `tests/test_experiment_runner.py`: candidate generation, merge, ranking, scoping, and safety.
- Modify `tests/test_experiment_runner.py`: persisted selector metadata and no reselection during direct-run resume.
- Modify `tests/test_ghz3_canonical_baseline_notebook.py`: notebook configuration and offline pipeline contract.
- Modify `tests/test_iqm_dependency_compatibility.py`: package window and public-symbol compatibility.
- Create `tests/test_iqm_layout_selector_live.py`: opt-in, zero-submit live selector/compile smoke test.

### Task 1: Add the selector configuration model and public exports

**Files:**
- Modify: `src/qudits_on_qubits/experiments/models.py:374-462`
- Modify: `src/qudits_on_qubits/experiments/__init__.py:25-49,81-147`
- Modify: `src/qudits_on_qubits/__init__.py:35-56,115-147`
- Test: `tests/test_experiment_models.py:342-468`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: Write failing selector-model round-trip and validation tests**

Add `IQMQubitSelectorConfig` to the existing model import tuple in the test file,
then add these tests:

```python
def test_iqm_qubit_selector_config_round_trips_with_safe_shape():
    config = IQMQubitSelectorConfig(
        top_k=8,
        num_trials=4000,
        cost_function="clifford",
        readout_mode="fidelity",
        remove_qubits=(5, 11),
    )
    payload = {
        "top_k": 8,
        "num_trials": 4000,
        "cost_function": "clifford",
        "readout_mode": "fidelity",
        "remove_qubits": [5, 11],
    }

    assert config.to_safe_dict() == payload
    assert IQMQubitSelectorConfig.from_safe_dict(payload) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", True),
        ("num_trials", 0),
        ("num_trials", 1.5),
        ("cost_function", "iswap"),
        ("readout_mode", "raw"),
        ("remove_qubits", (1, 1)),
        ("remove_qubits", (-1,)),
    ],
)
def test_iqm_qubit_selector_config_rejects_invalid_values(field, value):
    kwargs = {field: value}
    with pytest.raises(ExperimentValidationError, match=field):
        IQMQubitSelectorConfig(**kwargs)


def test_workload_optimization_allows_selector_without_explicit_layouts():
    selector = IQMQubitSelectorConfig()
    config = WorkloadOptimizationConfig(
        initial_layouts=(),
        seed_transpilers=(3, 7),
        iqm_qubit_selector=selector,
    )

    assert config.initial_layouts == ()
    assert config.iqm_qubit_selector is selector
    assert config.to_safe_dict()["iqm_qubit_selector"] == selector.to_safe_dict()
    assert WorkloadOptimizationConfig.from_safe_dict(config.to_safe_dict()) == config


def test_workload_optimization_requires_layout_source():
    with pytest.raises(ExperimentValidationError, match="layout source"):
        WorkloadOptimizationConfig(initial_layouts=())


def test_workload_optimization_rejects_invalid_selector_type():
    with pytest.raises(ExperimentValidationError, match="iqm_qubit_selector"):
        WorkloadOptimizationConfig(
            initial_layouts=((0, 1),),
            iqm_qubit_selector={},
        )


def test_workload_optimization_legacy_payload_keeps_legacy_safe_shape():
    payload = {
        "initial_layouts": [[0, 1]],
        "seed_transpilers": [3],
        "require_exact_physical_qubit_set": True,
        "prefer_calibration_metrics": True,
    }

    restored = WorkloadOptimizationConfig.from_safe_dict(payload)

    assert restored.iqm_qubit_selector is None
    assert restored.to_safe_dict() == payload
```

Add this public-API assertion:

```python
from qudits_on_qubits.experiments import IQMQubitSelectorConfig


def test_top_level_iqm_qubit_selector_config_alias_matches_experiments_export():
    assert qudits_on_qubits.IQMQubitSelectorConfig is IQMQubitSelectorConfig
```

- [ ] **Step 2: Run the model/API tests and verify RED**

Run:

```powershell
& $testPython -m pytest tests/test_experiment_models.py tests/test_public_api.py -q
```

Expected: collection or import failures because `IQMQubitSelectorConfig` does not exist, plus the empty-layout behavior still rejects selector-driven search.

- [ ] **Step 3: Implement the immutable configuration and workload integration**

Add before `WorkloadOptimizationConfig`:

```python
@dataclass(frozen=True)
class IQMQubitSelectorConfig:
    top_k: int = 10
    num_trials: int = 2000
    cost_function: str = "cz"
    readout_mode: str = "none"
    remove_qubits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ExperimentValidationError("top_k must be a positive integer")
        if type(self.num_trials) is not int or self.num_trials <= 0:
            raise ExperimentValidationError("num_trials must be a positive integer")
        if self.cost_function not in {"cz", "clifford"}:
            raise ExperimentValidationError(
                "cost_function must be 'cz' or 'clifford'"
            )
        if self.readout_mode not in {"none", "fidelity", "qndness"}:
            raise ExperimentValidationError(
                "readout_mode must be 'none', 'fidelity', or 'qndness'"
            )
        if not isinstance(self.remove_qubits, Sequence) or isinstance(
            self.remove_qubits, (str, bytes)
        ):
            raise ExperimentValidationError(
                "remove_qubits must be a sequence of distinct non-negative integers"
            )
        removed = tuple(self.remove_qubits)
        if (
            any(type(index) is not int or index < 0 for index in removed)
            or len(set(removed)) != len(removed)
        ):
            raise ExperimentValidationError(
                "remove_qubits must contain distinct non-negative integers"
            )
        object.__setattr__(self, "remove_qubits", removed)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "num_trials": self.num_trials,
            "cost_function": self.cost_function,
            "readout_mode": self.readout_mode,
            "remove_qubits": list(self.remove_qubits),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IQMQubitSelectorConfig":
        return cls(**dict(data))
```

Keep `initial_layouts` required for backward-compatible construction, and change the
workload model fields to:

```python
class WorkloadOptimizationConfig:
    initial_layouts: tuple[tuple[int, ...], ...]
    seed_transpilers: tuple[int, ...] = (0,)
    require_exact_physical_qubit_set: bool = True
    prefer_calibration_metrics: bool = True
    iqm_qubit_selector: IQMQubitSelectorConfig | None = None
```

Replace the current unconditional non-empty-layout check with the following exact
logic after `normalized_layouts = tuple(layouts)`:

Also change the outer-sequence validation text from
`initial_layouts must be a non-empty sequence of equal-width layouts` to
`initial_layouts must be a sequence of equal-width layouts`.

```python
        if any(
            len(layout) != len(normalized_layouts[0])
            for layout in normalized_layouts[1:]
        ) or len(set(normalized_layouts)) != len(normalized_layouts):
            raise ExperimentValidationError(
                "initial_layouts must be equal-width and unique"
            )
        if self.iqm_qubit_selector is not None and not isinstance(
            self.iqm_qubit_selector, IQMQubitSelectorConfig
        ):
            raise ExperimentValidationError(
                "iqm_qubit_selector must be IQMQubitSelectorConfig or None"
            )
        if not normalized_layouts and self.iqm_qubit_selector is None:
            raise ExperimentValidationError(
                "initial_layouts require at least one layout source"
            )
```

Build the safe payload without changing the legacy shape:

```python
    def to_safe_dict(self) -> dict[str, Any]:
        payload = {
            "initial_layouts": [list(layout) for layout in self.initial_layouts],
            "seed_transpilers": list(self.seed_transpilers),
            "require_exact_physical_qubit_set": self.require_exact_physical_qubit_set,
            "prefer_calibration_metrics": self.prefer_calibration_metrics,
        }
        if self.iqm_qubit_selector is not None:
            payload["iqm_qubit_selector"] = self.iqm_qubit_selector.to_safe_dict()
        return payload

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "WorkloadOptimizationConfig":
        selector_payload = data.get("iqm_qubit_selector")
        if selector_payload is not None and not isinstance(selector_payload, Mapping):
            raise ExperimentValidationError(
                "iqm_qubit_selector must be a safe mapping"
            )
        return cls(
            initial_layouts=data["initial_layouts"],
            seed_transpilers=data.get("seed_transpilers", (0,)),
            require_exact_physical_qubit_set=data.get(
                "require_exact_physical_qubit_set", True
            ),
            prefer_calibration_metrics=data.get("prefer_calibration_metrics", True),
            iqm_qubit_selector=(
                None
                if selector_payload is None
                else IQMQubitSelectorConfig.from_safe_dict(selector_payload)
            ),
        )
```

Add `IQMQubitSelectorConfig` immediately after `IQMHardware` in the model import
tuple and in `__all__` in `experiments/__init__.py`:

```python
    IQMHardware,
    IQMQubitSelectorConfig,
    MitigationConfig,
```

```python
    "IQMHardware",
    "IQMQubitSelectorConfig",
    "JobResultError",
```

Add the exact string immediately after `"IQMHardware"` in both
`_EXPERIMENT_EXPORTS` and `__all__` in the top-level `__init__.py`:

```python
    "IQMHardware",
    "IQMQubitSelectorConfig",
    "MitigationConfig",
```

- [ ] **Step 4: Run the model/API tests and verify GREEN**

Run:

```powershell
& $testPython -m pytest tests/test_experiment_models.py tests/test_public_api.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the model contract**

```powershell
git add src/qudits_on_qubits/experiments/models.py src/qudits_on_qubits/experiments/__init__.py src/qudits_on_qubits/__init__.py tests/test_experiment_models.py tests/test_public_api.py
git commit -m "feat: configure IQM automatic layout selection"
```

### Task 2: Add `IQMAdapter.suggest_layouts`

**Files:**
- Modify: `src/qudits_on_qubits/experiments/backends/iqm.py:1-95,184-253`
- Test: `tests/test_experiment_iqm_adapter.py`

- [ ] **Step 1: Write failing adapter tests for arguments and normalized output**

Add `IQMQubitSelectorConfig` to the test imports and add:

```python
def test_iqm_suggest_layouts_uses_resolved_backend_and_safe_config():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    circuit = QuantumCircuit(2)
    calls = []

    def selector(actual_backend, actual_circuit, config):
        calls.append((actual_backend, actual_circuit, config))
        return ([[4, 7], [8, 9]], [0.02, 0.03], "1.1.0")

    config = IQMQubitSelectorConfig(
        top_k=2,
        num_trials=500,
        cost_function="cz",
        readout_mode="none",
        remove_qubits=(5,),
    )
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        layout_selector=selector,
    )

    result = adapter.suggest_layouts(circuit, config)

    assert calls == [(backend, circuit, config)]
    assert result == {
        "provider": "iqm-qubit-selector",
        "version": "1.1.0",
        "configuration": config.to_safe_dict(),
        "layouts": ((4, 7), (8, 9)),
        "costs": (0.02, 0.03),
    }


def test_iqm_rejects_non_callable_layout_selector():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    with pytest.raises(BackendCompatibilityError, match="layout selector"):
        IQMAdapter(
            IQMHardware("garnet"),
            backend=_Backend(),
            layout_selector=object(),
        )
```

Add this complete malformed-output matrix:

```python
@pytest.mark.parametrize(
    ("layouts", "costs", "version"),
    [
        ((), (), "1.1.0"),
        (((4, 7), (4, 7)), (0.02, 0.03), "1.1.0"),
        (((4, 4),), (0.02,), "1.1.0"),
        (((4,),), (0.02,), "1.1.0"),
        (((4, 20),), (0.02,), "1.1.0"),
        (((True, 7),), (0.02,), "1.1.0"),
        (((4, 7),), (), "1.1.0"),
        (((4, 7),), (-0.01,), "1.1.0"),
        (((4, 7),), (float("nan"),), "1.1.0"),
        (((4, 7), (8, 9)), (0.03, 0.02), "1.1.0"),
        (((4, 7),), (0.02,), "token" + "=selector-test-value"),
        (
            tuple((index, index + 1) for index in range(11)),
            tuple(index / 100 for index in range(11)),
            "1.1.0",
        ),
    ],
)
def test_iqm_suggest_layouts_rejects_malformed_provider_output(
    layouts, costs, version
):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: (layouts, costs, version),
    )

    with pytest.raises(
        BackendCompatibilityError,
        match=r"^IQM qubit selector output is invalid$",
    ):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


def test_iqm_suggest_layouts_rejects_malformed_return_shape():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: object(),
    )

    with pytest.raises(BackendCompatibilityError, match="output is invalid"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


def test_iqm_suggest_layouts_redacts_provider_failure():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    sensitive_message = "Authorization: " + "Bearer selector-test-value"

    def fail(*_args):
        raise RuntimeError(sensitive_message)

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=fail,
    )
    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    _assert_sanitized(caught, sensitive_message)
    assert str(caught.value) == (
        "IQM qubit selector failed for backend iqm:garnet (RuntimeError)"
    )


def test_iqm_suggest_layouts_maps_missing_package_to_optional_dependency():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    def missing(*_args):
        raise ModuleNotFoundError("iqm.qubit_selector")

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=missing,
    )

    with pytest.raises(OptionalDependencyError, match="iqm-qubit-selector"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


def test_iqm_suggest_layouts_propagates_memory_error():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    def exhausted(*_args):
        raise MemoryError("selector exhausted")

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=exhausted,
    )

    with pytest.raises(MemoryError, match="selector exhausted"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())
```

Add the enum/argument translation test before implementation so it also starts RED:

```python
def test_default_layout_selector_translates_safe_config_to_iqm_api(
    monkeypatch,
):
    import importlib.metadata
    import sys
    from types import ModuleType, SimpleNamespace

    from qudits_on_qubits.experiments.backends.iqm import (
        _default_layout_selector,
    )

    calls = {}
    cz = object()
    clifford = object()
    none = object()
    fidelity = object()
    qndness = object()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def get_top_layouts(self, *, num_layouts):
            calls["num_layouts"] = num_layouts
            return [[4, 7], [8, 9]], [0.02, 0.03]

    module = ModuleType("iqm.qubit_selector.qubit_selector")
    module.CostEvaluator = FakeEvaluator
    module.CostFunction = SimpleNamespace(
        GATE_COST_CZ=cz,
        GATE_COST_CLIFFORD=clifford,
    )
    module.ReadoutMode = SimpleNamespace(
        NONE=none,
        FIDELITY=fidelity,
        QNDNESS=qndness,
    )
    monkeypatch.setitem(
        sys.modules,
        "iqm.qubit_selector.qubit_selector",
        module,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "1.1.0" if package == "iqm-qubit-selector" else None,
    )
    backend = _Backend()
    circuit = QuantumCircuit(2)
    config = IQMQubitSelectorConfig(
        top_k=2,
        num_trials=500,
        cost_function="clifford",
        readout_mode="qndness",
        remove_qubits=(5,),
    )

    result = _default_layout_selector(backend, circuit, config)

    assert result == ([[4, 7], [8, 9]], [0.02, 0.03], "1.1.0")
    assert calls == {
        "kwargs": {
            "backend": backend,
            "quantum_circuit": circuit,
            "cost_function": clifford,
            "readoutmode": qndness,
            "remove_qubits": [5],
            "num_trials": 500,
        },
        "num_layouts": 2,
    }
```

- [ ] **Step 2: Run the adapter tests and verify RED**

Run:

```powershell
& $testPython -m pytest tests/test_experiment_iqm_adapter.py -q
```

Expected: failures because the constructor has no `layout_selector` seam and the adapter has no `suggest_layouts` method.

- [ ] **Step 3: Implement the default IQM selector bridge**

Add this provider bridge:

```python
def _default_layout_selector(
    backend: Any,
    circuit: Any,
    config: IQMQubitSelectorConfig,
) -> tuple[Any, Any, str]:
    from importlib.metadata import version

    from iqm.qubit_selector.qubit_selector import (
        CostEvaluator,
        CostFunction,
        ReadoutMode,
    )

    cost_functions = {
        "cz": CostFunction.GATE_COST_CZ,
        "clifford": CostFunction.GATE_COST_CLIFFORD,
    }
    readout_modes = {
        "none": ReadoutMode.NONE,
        "fidelity": ReadoutMode.FIDELITY,
        "qndness": ReadoutMode.QNDNESS,
    }
    evaluator = CostEvaluator(
        backend=backend,
        quantum_circuit=circuit,
        cost_function=cost_functions[config.cost_function],
        readoutmode=readout_modes[config.readout_mode],
        remove_qubits=(list(config.remove_qubits) if config.remove_qubits else None),
        num_trials=config.num_trials,
    )
    layouts, costs = evaluator.get_top_layouts(num_layouts=config.top_k)
    return layouts, costs, version("iqm-qubit-selector")
```

Add `layout_selector` to `IQMAdapter.__init__`, validate it as callable, and store the injected function. Implement `suggest_layouts` with strict normalization to the exact mapping asserted above. Catch import failures as `OptionalDependencyError`, ordinary provider failures as a redacted `BackendCompatibilityError` containing only `_exception_name(error)`, and propagate `KeyboardInterrupt`, `SystemExit`, and `MemoryError`.

Add `layout_selector: Any = None` after the existing `loader` parameter. Insert
this validation after the transpiler validation:

```python
        if layout_selector is not None and not callable(layout_selector):
            raise BackendCompatibilityError(
                "IQM layout selector must be callable"
            )
```

Insert this assignment after `self._transpiler = ...`:

```python
        self._layout_selector = layout_selector or _default_layout_selector
```

Then add the complete method:

```python
    def suggest_layouts(
        self,
        circuit: QuantumCircuit,
        config: IQMQubitSelectorConfig,
    ) -> Mapping[str, Any]:
        if not isinstance(circuit, QuantumCircuit):
            raise BackendCompatibilityError(
                "IQM qubit selector requires a QuantumCircuit"
            )
        if not isinstance(config, IQMQubitSelectorConfig):
            raise BackendCompatibilityError(
                "IQM qubit selector requires IQMQubitSelectorConfig"
            )
        backend = self._backend_instance()
        capacity = _num_qubits(backend)
        if capacity is None:
            raise BackendCompatibilityError(
                "IQM qubit selector requires backend qubit capacity"
            )
        if any(index >= capacity for index in config.remove_qubits):
            raise BackendCompatibilityError(
                "IQM qubit selector remove_qubits exceed backend capacity"
            )
        try:
            raw_result = self._layout_selector(
                backend,
                circuit,
                config,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQM automatic layout selection requires iqm-qubit-selector "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            raise BackendCompatibilityError(
                "IQM qubit selector failed for backend "
                f"iqm:{self._spec.device} ({_exception_name(error)})"
            ) from None

        try:
            if (
                not isinstance(raw_result, (tuple, list))
                or len(raw_result) != 3
            ):
                raise ValueError
            raw_layouts, raw_costs, version = raw_result
            if (
                not isinstance(raw_layouts, Sequence)
                or isinstance(raw_layouts, (str, bytes))
                or not isinstance(raw_costs, Sequence)
                or isinstance(raw_costs, (str, bytes))
                or not raw_layouts
                or len(raw_layouts) != len(raw_costs)
                or len(raw_layouts) > config.top_k
                or not isinstance(version, str)
                or not version
                or not _safe_metadata_text(version)
            ):
                raise ValueError
            if any(
                not isinstance(layout, Sequence)
                or isinstance(layout, (str, bytes))
                for layout in raw_layouts
            ):
                raise ValueError
            layouts = tuple(tuple(layout) for layout in raw_layouts)
            if (
                any(
                    not layout
                    or len(layout) != circuit.num_qubits
                    or any(
                        type(index) is not int
                        or index < 0
                        or index >= capacity
                        or index in config.remove_qubits
                        for index in layout
                    )
                    or len(set(layout)) != len(layout)
                    for layout in layouts
                )
                or len(set(layouts)) != len(layouts)
            ):
                raise ValueError
            costs = tuple(float(cost) for cost in raw_costs)
            if (
                any(
                    isinstance(cost, bool)
                    or not isinstance(cost, Real)
                    or not math.isfinite(float(cost))
                    or float(cost) < 0
                    for cost in raw_costs
                )
                or any(left > right for left, right in zip(costs, costs[1:]))
            ):
                raise ValueError
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise BackendCompatibilityError(
                "IQM qubit selector output is invalid"
            ) from None

        return {
            "provider": "iqm-qubit-selector",
            "version": version,
            "configuration": config.to_safe_dict(),
            "layouts": layouts,
            "costs": costs,
        }
```

Add these imports:

```python
from numbers import Real
from qiskit import QuantumCircuit

from ..models import IQMHardware, IQMQubitSelectorConfig, TranspilationConfig
```

- [ ] **Step 4: Run the enum translation test directly**

```powershell
& $testPython -m pytest tests/test_experiment_iqm_adapter.py::test_default_layout_selector_translates_safe_config_to_iqm_api -q
```

Expected: PASS; the test does not import or contact a live IQM backend.

- [ ] **Step 5: Run the adapter tests and verify GREEN**

```powershell
& $testPython -m pytest tests/test_experiment_iqm_adapter.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the adapter bridge**

```powershell
git add src/qudits_on_qubits/experiments/backends/iqm.py tests/test_experiment_iqm_adapter.py
git commit -m "feat: expose IQM calibration-aware layout suggestions"
```

### Task 3: Merge selector layouts into the full-workload optimizer

**Files:**
- Modify: `src/qudits_on_qubits/experiments/runner.py:88-101,1122-1362`
- Test: `tests/test_experiment_runner.py:2425-2909`

- [ ] **Step 1: Write failing representative-circuit and candidate-merge tests**

Add a focused representative test:

```python
def test_selector_representative_prefers_two_qubit_count_then_depth_then_index():
    from qudits_on_qubits.experiments.runner import _representative_circuit_index

    first = QuantumCircuit(3)
    first.cz(0, 1)
    second = QuantumCircuit(3)
    second.cz(0, 1)
    second.barrier()
    second.x(2)
    third = QuantumCircuit(3)
    third.cz(0, 1)
    third.cz(1, 2)

    assert _representative_circuit_index((first, second, third)) == 2
    assert _representative_circuit_index((first, first.copy())) == 0
```

Add this reusable test adapter beside `_CandidateAdapter`:

```python
class _SelectorCandidateAdapter(_CandidateAdapter):
    def __init__(self, compiler, selector_result=None, selector_error=None):
        super().__init__(compiler)
        self.identity = BackendIdentity(
            "iqm",
            "garnet",
            provider="iqm",
            version="35.0.0",
            metadata={"calibration_set_id": "cal-17"},
        )
        self.selector_result = selector_result
        self.selector_error = selector_error
        self.selector_calls = []

    def suggest_layouts(self, circuit, config):
        self.selector_calls.append((circuit, config))
        if self.selector_error is not None:
            raise self.selector_error
        if self.selector_result is not None:
            return self.selector_result
        return {
            "provider": "iqm-qubit-selector",
            "version": "1.1.0",
            "configuration": config.to_safe_dict(),
            "layouts": ((2, 3), (4, 5)),
            "costs": (0.01, 0.02),
        }
```

Add the generated-first merge and complete-workload ranking test:

```python
def test_iqm_selector_candidates_merge_generated_first_and_rank_full_workload(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    logical, _settings = _install_two_setting_workload(monkeypatch)
    cz_counts = {(2, 3): 4, (4, 5): 0, (0, 1): 2}

    def compiler(config, identity):
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"selector-{config.initial_layout}-{index}",
                cz_count=cz_counts[config.initial_layout],
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    selector = IQMQubitSelectorConfig(top_k=2)
    adapter = _SelectorCandidateAdapter(compiler)
    result = run_experiment(
        make_spec(
            tmp_path,
            backend=IQMHardware("garnet"),
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((4, 5), (0, 1)),
                seed_transpilers=(3, 7),
                prefer_calibration_metrics=False,
                iqm_qubit_selector=selector,
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )

    assert len(adapter.selector_calls) == 1
    assert adapter.selector_calls[0][0] is logical[0]
    assert [call[1].initial_layout for call in adapter.compile_calls] == [
        (2, 3),
        (2, 3),
        (4, 5),
        (4, 5),
        (0, 1),
        (0, 1),
    ]
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    metadata = document["workload_optimization"]
    assert metadata["selected_layout"] == [4, 5]
    assert metadata["candidates"][0]["layout_source"] == "iqm_qubit_selector"
    assert metadata["candidates"][0]["selector_cost"] == 0.01
    assert metadata["candidates"][4]["layout_source"] == "explicit"
    assert metadata["candidates"][4]["selector_cost"] is None
    assert metadata["selector"] == {
        "provider": "iqm-qubit-selector",
        "version": "1.1.0",
        "calibration_set_id": "cal-17",
        "configuration": selector.to_safe_dict(),
        "representative_circuit_index": 0,
        "representative_circuit_name": "logical-0",
        "generated_layouts": [[2, 3], [4, 5]],
        "generated_costs": [0.01, 0.02],
        "explicit_layouts": [[4, 5], [0, 1]],
        "merged_layouts": [[2, 3], [4, 5], [0, 1]],
    }
```

This deliberately makes layout `(2, 3)` have the lowest selector cost but a worse
compiled workload than `(4, 5)`, proving the selector is only the preselection
stage.

- [ ] **Step 2: Run the focused runner tests and verify RED**

```powershell
& $testPython -m pytest tests/test_experiment_runner.py -k "selector or workload_candidate_search" -q
```

Expected: missing helper/method integration and unchanged compile matrix.

- [ ] **Step 3: Add internal layout candidate records and representative selection**

Add:

```python
@dataclass(frozen=True)
class _WorkloadLayoutCandidate:
    layout: tuple[int, ...]
    source: str
    selector_cost: float | None


def _representative_circuit_index(circuits: Sequence[QuantumCircuit]) -> int:
    if not circuits:
        raise ExperimentValidationError(
            "workload optimization requires logical measurement circuits"
        )

    def rank(item: tuple[int, QuantumCircuit]) -> tuple[int, int, int]:
        index, circuit = item
        two_qubit_count = sum(
            len(instruction.qubits) == 2
            and not getattr(instruction.operation, "_directive", False)
            for instruction in circuit.data
        )
        return two_qubit_count, circuit.depth(), -index

    return max(enumerate(circuits), key=rank)[0]
```

Add the complete provider-result validator and merge helper:

```python
def _validated_iqm_selector_result(
    value: Any,
    *,
    config: IQMQubitSelectorConfig,
    logical_width: int,
) -> tuple[str, tuple[tuple[int, ...], ...], tuple[float, ...]]:
    try:
        if not isinstance(value, Mapping) or set(value) != {
            "provider",
            "version",
            "configuration",
            "layouts",
            "costs",
        }:
            raise ValueError
        version = value["version"]
        if (
            value["provider"] != "iqm-qubit-selector"
            or not isinstance(version, str)
            or not version
            or _unsafe_persisted_text(version)
            or value["configuration"] != config.to_safe_dict()
        ):
            raise ValueError
        raw_layouts = value["layouts"]
        raw_costs = value["costs"]
        if (
            not isinstance(raw_layouts, Sequence)
            or isinstance(raw_layouts, (str, bytes))
            or not isinstance(raw_costs, Sequence)
            or isinstance(raw_costs, (str, bytes))
            or not raw_layouts
            or len(raw_layouts) != len(raw_costs)
            or len(raw_layouts) > config.top_k
        ):
            raise ValueError
        if any(
            not isinstance(layout, Sequence)
            or isinstance(layout, (str, bytes))
            for layout in raw_layouts
        ):
            raise ValueError
        layouts = tuple(tuple(layout) for layout in raw_layouts)
        if (
            any(
                len(layout) != logical_width
                or any(
                    type(index) is not int
                    or index < 0
                    or index in config.remove_qubits
                    for index in layout
                )
                or len(set(layout)) != len(layout)
                for layout in layouts
            )
            or len(set(layouts)) != len(layouts)
        ):
            raise ValueError
        if any(
            isinstance(cost, bool)
            or not isinstance(cost, Real)
            or not math.isfinite(float(cost))
            or float(cost) < 0
            for cost in raw_costs
        ):
            raise ValueError
        costs = tuple(float(cost) for cost in raw_costs)
        if any(left > right for left, right in zip(costs, costs[1:])):
            raise ValueError
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise BackendCompatibilityError(
            "IQM qubit selector output is invalid"
        ) from None
    return version, layouts, costs


def _workload_layout_candidates(
    adapter: Any,
    circuits: tuple[QuantumCircuit, ...],
    spec: ExperimentSpec,
    *,
    logical_width: int,
    expected_identity: BackendIdentity | None,
) -> tuple[
    tuple[_WorkloadLayoutCandidate, ...],
    Mapping[str, object] | None,
]:
    search = spec.workload_optimization
    if search is None:
        raise ExperimentValidationError(
            "workload layout candidates require workload optimization"
        )
    explicit = tuple(
        _WorkloadLayoutCandidate(layout, "explicit", None)
        for layout in search.initial_layouts
    )
    config = search.iqm_qubit_selector
    if config is None:
        return explicit, None
    if not isinstance(spec.backend, IQMHardware):
        raise ExperimentValidationError(
            "iqm_qubit_selector requires an IQMHardware backend"
        )

    identity = expected_identity
    if identity is None:
        try:
            identity = adapter.resolve()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as error:
            raise BackendCompatibilityError(
                "IQM adapter resolution failed before layout selection "
                f"({_workload_rejection_category(error)})"
            ) from None
        if not isinstance(identity, BackendIdentity):
            raise BackendCompatibilityError(
                "adapter resolve must return BackendIdentity"
            )
        _validate_adapter_target(spec, identity)

    calibration_set_id = identity.metadata.get("calibration_set_id")
    if (
        not isinstance(calibration_set_id, str)
        or not calibration_set_id
        or _unsafe_persisted_text(calibration_set_id)
    ):
        raise BackendCompatibilityError(
            "IQM selector requires a safe calibration_set_id"
        )
    try:
        suggest_layouts = getattr(adapter, "suggest_layouts")
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise BackendCompatibilityError(
            "IQM adapter does not expose suggest_layouts"
        ) from None
    if not callable(suggest_layouts):
        raise BackendCompatibilityError(
            "IQM adapter does not expose callable suggest_layouts"
        )

    representative_index = _representative_circuit_index(circuits)
    representative = circuits[representative_index]
    if (
        not isinstance(representative.name, str)
        or _unsafe_persisted_text(representative.name)
    ):
        raise ExperimentValidationError(
            "selector representative circuit name is unsafe"
        )
    try:
        raw_result = suggest_layouts(representative, config)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except OptionalDependencyError:
        raise
    except Exception as error:
        raise BackendCompatibilityError(
            "IQM qubit selector failed for backend "
            f"{identity.kind}:{identity.name} "
            f"({_workload_rejection_category(error)})"
        ) from None

    version, generated_layouts, generated_costs = (
        _validated_iqm_selector_result(
            raw_result,
            config=config,
            logical_width=logical_width,
        )
    )
    candidates = [
        _WorkloadLayoutCandidate(layout, "iqm_qubit_selector", cost)
        for layout, cost in zip(
            generated_layouts,
            generated_costs,
            strict=True,
        )
    ]
    seen = set(generated_layouts)
    for candidate in explicit:
        if candidate.layout not in seen:
            seen.add(candidate.layout)
            candidates.append(candidate)

    metadata = {
        "provider": "iqm-qubit-selector",
        "version": version,
        "calibration_set_id": calibration_set_id,
        "configuration": config.to_safe_dict(),
        "representative_circuit_index": representative_index,
        "representative_circuit_name": representative.name,
        "generated_layouts": generated_layouts,
        "generated_costs": generated_costs,
        "explicit_layouts": search.initial_layouts,
        "merged_layouts": tuple(candidate.layout for candidate in candidates),
    }
    return tuple(candidates), metadata
```

Add `from numbers import Real` and import `IQMQubitSelectorConfig` from
`.models` at the top of `runner.py`.

- [ ] **Step 4: Feed merged records into the existing layout-by-seed loop**

After the explicit-width validation and before target lookup, resolve candidates:

```python
    layout_candidates, selector_metadata = _workload_layout_candidates(
        adapter,
        workload_circuits,
        spec,
        logical_width=logical_width,
        expected_identity=expected_identity,
    )
```

Replace `for layout in search.initial_layouts:` with:

```python
    for layout_candidate in layout_candidates:
        layout = layout_candidate.layout
```

Keep existing compilation, mapping extraction, exact-set rejection, target-metric
ranking, and winner reuse. Add selector-only fields without changing legacy manual
artifact rows:

```python
row = {
    "status": "rejected",
    "candidate_index": candidate_index,
    "layout": list(layout),
    "seed_transpiler": seed,
}
if selector_metadata is not None:
    row["layout_source"] = layout_candidate.source
    row["selector_cost"] = layout_candidate.selector_cost
```

Add selector metadata to the final mapping only when enabled:

```python
    if selector_metadata is not None:
        metadata["selector"] = selector_metadata
```

- [ ] **Step 5: Add safety/scoping tests**

Extend the model imports in `tests/test_experiment_runner.py` with
`AerIdeal`, `IQMQubitSelectorConfig`, and `PiastQHardware`, then add:

```python
def test_iqm_selector_failure_is_redacted_before_compile_or_submit(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    sensitive_message = "Authorization: " + "Bearer runner-selector-test-value"
    adapter = _SelectorCandidateAdapter(
        lambda *_args: pytest.fail("compile called"),
        selector_error=RuntimeError(sensitive_message),
    )
    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=IQMQubitSelectorConfig(),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert sensitive_message not in str(caught.value)
    assert caught.value.__cause__ is None
    assert not (tmp_path / "runs").exists()


def test_iqm_selector_memory_error_propagates_before_compile_or_submit(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    adapter = _SelectorCandidateAdapter(
        lambda *_args: pytest.fail("compile called"),
        selector_error=MemoryError("selector exhausted"),
    )
    with pytest.raises(MemoryError, match="selector exhausted"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    iqm_qubit_selector=IQMQubitSelectorConfig(),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


def test_iqm_selector_empty_output_does_not_fall_back_to_explicit_layout(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    config = IQMQubitSelectorConfig()
    adapter = _SelectorCandidateAdapter(
        lambda *_args: pytest.fail("compile called"),
        selector_result={
            "provider": "iqm-qubit-selector",
            "version": "1.1.0",
            "configuration": config.to_safe_dict(),
            "layouts": (),
            "costs": (),
        },
    )
    with pytest.raises(
        BackendCompatibilityError,
        match="selector output is invalid",
    ):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=config,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


@pytest.mark.parametrize("backend", [AerIdeal(), PiastQHardware()])
def test_iqm_selector_configuration_is_rejected_for_non_iqm_backend(
    tmp_path, backend
):
    from qudits_on_qubits.experiments.runner import (
        _compile_measurement_workload,
    )

    logical = QuantumCircuit(2, 2, name="logical")
    logical.measure((0, 1), (0, 1))
    adapter = RecordingAdapter()
    adapter.compile = Mock(side_effect=AssertionError("compile called"))
    spec = make_spec(
        tmp_path,
        backend=backend,
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=(),
            iqm_qubit_selector=IQMQubitSelectorConfig(),
        ),
    )

    with pytest.raises(ExperimentValidationError, match="IQMHardware"):
        _compile_measurement_workload(
            adapter,
            (logical,),
            (("A0",),),
            spec,
        )

    adapter.compile.assert_not_called()
    assert adapter.submit_calls == 0


def test_iqm_selector_requires_callable_adapter_method_before_compile(tmp_path):
    from qudits_on_qubits.experiments.runner import (
        _compile_measurement_workload,
    )

    logical = QuantumCircuit(2, 2, name="logical")
    logical.measure((0, 1), (0, 1))
    adapter = _CandidateAdapter(lambda *_args: pytest.fail("compile called"))
    adapter.identity = BackendIdentity(
        "iqm",
        "garnet",
        metadata={"calibration_set_id": "cal-17"},
    )
    spec = make_spec(
        tmp_path,
        backend=IQMHardware("garnet"),
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=(),
            iqm_qubit_selector=IQMQubitSelectorConfig(),
        ),
    )

    with pytest.raises(
        BackendCompatibilityError,
        match="suggest_layouts",
    ):
        _compile_measurement_workload(
            adapter,
            (logical,),
            (("A0",),),
            spec,
            expected_identity=adapter.identity,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


def test_unsafe_injected_selector_metadata_never_reaches_artifact(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    config = IQMQubitSelectorConfig(top_k=1)
    adapter = _SelectorCandidateAdapter(
        lambda *_args: pytest.fail("compile called"),
        selector_result={
            "provider": "iqm-qubit-selector",
            "version": "token" + "=metadata-test-value",
            "configuration": config.to_safe_dict(),
            "layouts": ((0, 1),),
            "costs": (0.01,),
        },
    )

    with pytest.raises(BackendCompatibilityError, match="output is invalid"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    iqm_qubit_selector=config,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert not list((tmp_path / "runs").glob("**/experiment.json"))


def test_selector_candidate_layout_escape_fails_before_submit(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)

    def escaping_compiler(_config, identity):
        circuits = tuple(
            _physical_measurement_circuit(
                (0, 2),
                name=f"escaped-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    adapter = _SelectorCandidateAdapter(
        escaping_compiler,
        selector_result={
            "provider": "iqm-qubit-selector",
            "version": "1.1.0",
            "configuration": selector.to_safe_dict(),
            "layouts": ((0, 1),),
            "costs": (0.01,),
        },
    )
    with pytest.raises(BackendCompatibilityError, match="no workload candidate"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    seed_transpilers=(3,),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert len(adapter.compile_calls) == 1
    assert adapter.submit_calls == 0
```

Retain and rerun the existing
`test_workload_candidate_failure_does_not_discard_accepted_candidate_or_leak`.
Its exact candidate-row equality proves manual `initial_layouts` keep their prior
artifact shape without selector-only fields.

- [ ] **Step 6: Run all runner workload tests and verify GREEN**

```powershell
& $testPython -m pytest tests/test_experiment_runner.py -k "selector or workload" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit pipeline integration**

```powershell
git add src/qudits_on_qubits/experiments/runner.py tests/test_experiment_runner.py
git commit -m "feat: rank IQM selector layouts over Bell workloads"
```

### Task 4: Preserve selector metadata through checkpoint resume

**Files:**
- Modify: `tests/test_experiment_runner.py`

- [ ] **Step 1: Write the checkpoint characterization/regression test**

This behavior is supplied by the existing generic checkpoint flow, so the test may
already pass after Task 3. Add the complete regression beside
`test_interrupted_postprocessing_resumes_exact_direct_result_from_saved_counts`:

```python
def test_selector_metadata_survives_checkpoint_resume_without_reselection(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    setting = ("A0",)
    logical = QuantumCircuit(2, 2, name="selector-resume-logical")
    logical.measure((0, 1), (0, 1))
    metadata = {
        "setting_by_circuit_index": (setting,),
        "terms": (
            {
                "coeff": 1.0 + 0.0j,
                "settings": setting,
                "powers": (0,),
                "source": "selector-resume",
            },
        ),
        "qutrit_bit_indices_by_setting": {setting: ((0, 1),)},
        "physical_to_logical_outcome_map": {
            0: 0,
            1: 1,
            2: 2,
            3: None,
        },
        "d": 3,
    }
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(
            circuits=(logical,),
            metadata=metadata,
        ),
    )

    def compiler(config, identity):
        circuit = _physical_measurement_circuit(
            config.initial_layout,
            name="selector-resume-compiled",
        )
        return CompiledBatch((circuit,), identity)

    class SelectorResumeAdapter(_SelectorCandidateAdapter):
        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            return ExecutionResult(
                ({"00": 8, "11": 2},),
                submitted.job_id,
                self.identity,
                status="done",
            )

    selector = IQMQubitSelectorConfig(top_k=1)
    selector_result = {
        "provider": "iqm-qubit-selector",
        "version": "1.1.0",
        "configuration": selector.to_safe_dict(),
        "layouts": ((0, 1),),
        "costs": (0.02,),
    }
    adapter = SelectorResumeAdapter(
        compiler,
        selector_result=selector_result,
    )
    spec = make_spec(
        tmp_path,
        backend=IQMHardware("garnet"),
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=((0, 1),),
            seed_transpilers=(3,),
            iqm_qubit_selector=selector,
        ),
    )
    real_bootstrap = runner.bootstrap_bell_results

    def interrupt_after_checkpoint(*_args, **_kwargs):
        checkpoint_paths = list(spec.output_root.glob("**/experiment.json"))
        assert len(checkpoint_paths) == 1
        checkpoint = json.loads(
            checkpoint_paths[0].read_text(encoding="utf-8")
        )
        assert checkpoint["status"] == "postprocessing"
        assert checkpoint["workload_optimization"]["selector"]["provider"] == (
            "iqm-qubit-selector"
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(
        runner,
        "bootstrap_bell_results",
        interrupt_after_checkpoint,
    )
    with pytest.raises(KeyboardInterrupt) as interrupted:
        runner.run_experiment(spec, adapter=adapter)

    artifact_dir = interrupted.value.__qoq_artifact_dir__
    checkpoint = json.loads(
        (artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    selector_metadata = checkpoint["workload_optimization"]["selector"]
    assert selector_metadata["calibration_set_id"] == "cal-17"
    assert adapter.selector_calls and len(adapter.selector_calls) == 1

    monkeypatch.setattr(runner, "bootstrap_bell_results", real_bootstrap)
    resumed = runner.resume_experiment(artifact_dir, spec=spec)

    assert resumed.status is ExperimentStatus.COMPLETED
    assert len(adapter.selector_calls) == 1
    final_document = json.loads(
        (artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert final_document["workload_optimization"]["selector"] == (
        selector_metadata
    )
```

- [ ] **Step 2: Run the resume regression**

```powershell
& $testPython -m pytest tests/test_experiment_runner.py -k "selector_metadata_survives" -q
```

Expected: PASS. A failure means Task 3 did not keep selector metadata inside
`schema_fragments["workload_optimization"]`; fix Task 3 before proceeding. Do not
add a second artifact lifecycle or a selector call to `resume_experiment`.

- [ ] **Step 3: Run resume and workload regression suites**

```powershell
& $testPython -m pytest tests/test_experiment_resume.py tests/test_experiment_runner.py -k "selector or workload_optimization or postprocessing" -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit the resume contract**

```powershell
git add tests/test_experiment_runner.py
git commit -m "test: preserve IQM selector choice across resume"
```

### Task 5: Enable the shared selector from the GHZ3 notebook

**Files:**
- Modify: `notebooks/ghz3_bell_canonical_baseline.ipynb`
- Modify: `tests/test_ghz3_canonical_baseline_notebook.py:150-310,614-816`

- [ ] **Step 1: Write failing notebook contract assertions**

In `test_notebook_has_aer_and_opt_in_iqm_runs_and_clean_cells`, add:

```python
assert "IQMQubitSelectorConfig" in imported
```

In `test_configuration_and_empty_summary_are_semantically_complete`, assert the
configuration cell evaluates to:

```python
assert namespace["IQM_LAYOUT_SELECTOR"] == namespace[
    "IQMQubitSelectorConfig"
](
    top_k=10,
    num_trials=2000,
    cost_function="cz",
    readout_mode="none",
)
assert namespace["workload_optimization"] == namespace[
    "WorkloadOptimizationConfig"
](
    initial_layouts=((0, 1, 2, 7, 3, 4),),
    seed_transpilers=(3, 7, 13),
    iqm_qubit_selector=namespace["IQMQubitSelectorConfig"](
        top_k=10,
        num_trials=2000,
        cost_function="cz",
        readout_mode="none",
    ),
)
```

In `OfflineIQMAdapter.__init__`, add this exact counter assignment:

```python
            self.selector_calls = 0
```

Add this method to `OfflineIQMAdapter`:

```python
def suggest_layouts(self, circuit, config):
    self.selector_calls += 1
    assert circuit.num_qubits == 6
    return {
        "provider": "iqm-qubit-selector",
        "version": "1.1.0",
        "configuration": config.to_safe_dict(),
        "layouts": ((0, 1, 2, 7, 3, 4),),
        "costs": (0.02,),
    }
```

After the run, add these exact assertions:

```python
assert adapter.selector_calls == 1
assert adapter.compile_calls == 3
document = json.loads(
    (Path(result.artifact_dir) / "experiment.json").read_text(encoding="utf-8")
)
selector_metadata = document["workload_optimization"]["selector"]
assert selector_metadata["provider"] == "iqm-qubit-selector"
assert selector_metadata["version"] == "1.1.0"
assert selector_metadata["calibration_set_id"] == "offline-cal"
assert selector_metadata["configuration"] == namespace[
    "IQM_LAYOUT_SELECTOR"
].to_safe_dict()
assert selector_metadata["representative_circuit_index"] == 0
assert selector_metadata["representative_circuit_name"] == "ghz3_direct_basis"
assert selector_metadata["generated_layouts"] == [[0, 1, 2, 7, 3, 4]]
assert selector_metadata["generated_costs"] == [0.02]
assert selector_metadata["explicit_layouts"] == [[0, 1, 2, 7, 3, 4]]
assert selector_metadata["merged_layouts"] == [[0, 1, 2, 7, 3, 4]]
```

- [ ] **Step 2: Run notebook contract tests and verify RED**

```powershell
& $testPython -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
```

Expected: missing import/configuration and missing offline selector method.

- [ ] **Step 3: Edit the existing notebook import and configuration cells**

Add this name to the existing `qudits_on_qubits.experiments` import tuple:

```python
    IQMHardware,
    IQMQubitSelectorConfig,
    MitigationConfig,
```

Replace the start of the configuration cell with this exact source:

```python
SHOTS = 100
IQM_LAYOUT_CANDIDATES = ((0, 1, 2, 7, 3, 4),)
IQM_SEED_CANDIDATES = (3, 7, 13)
IQM_LAYOUT_SELECTOR = IQMQubitSelectorConfig(
    top_k=10,
    num_trials=2000,
    cost_function="cz",
    readout_mode="none",
)
workload_optimization = WorkloadOptimizationConfig(
    initial_layouts=IQM_LAYOUT_CANDIDATES,
    seed_transpilers=IQM_SEED_CANDIDATES,
    iqm_qubit_selector=IQM_LAYOUT_SELECTOR,
)
```

Keep the remainder of the cell, `RUN_IQM = False`, `SHOTS = 100`, mitigation,
uncertainty, cell IDs, notebook metadata, execution counts, and outputs unchanged.

- [ ] **Step 4: Run notebook contract tests and JSON validation**

```powershell
& $testPython -m pytest tests/test_ghz3_canonical_baseline_notebook.py -q
& $testPython -m json.tool notebooks/ghz3_bell_canonical_baseline.ipynb NUL
```

Expected: all notebook tests pass; JSON command exits 0.

- [ ] **Step 5: Commit notebook integration**

```powershell
git add notebooks/ghz3_bell_canonical_baseline.ipynb tests/test_ghz3_canonical_baseline_notebook.py
git commit -m "feat: enable automatic IQM layout search in GHZ3 notebook"
```

### Task 6: Align IQM dependencies and documentation

**Files:**
- Modify: `pyproject.toml:25-26`
- Modify: `requirements.txt:15-16`
- Modify: `src/qudits_on_qubits.egg-info/PKG-INFO:20-21`
- Modify: `src/qudits_on_qubits.egg-info/requires.txt:15-16`
- Modify: `tests/test_iqm_dependency_compatibility.py`
- Modify: `README.md:130-160`

- [ ] **Step 1: Write failing dependency-window and API tests**

Extend the required dependency tuple with `"iqm-qubit-selector>=1.1,<2"`. Add:

```python
def test_qubit_selector_public_api_matches_pipeline_contract(self):
    import inspect

    from iqm.qubit_selector.qubit_selector import (
        CostEvaluator,
        CostFunction,
        ReadoutMode,
    )

    parameters = inspect.signature(CostEvaluator).parameters
    for name in (
        "backend",
        "quantum_circuit",
        "cost_function",
        "readoutmode",
        "remove_qubits",
        "num_trials",
    ):
        self.assertIn(name, parameters)
    self.assertIn("num_layouts", inspect.signature(CostEvaluator.get_top_layouts).parameters)
    self.assertTrue(hasattr(CostFunction, "GATE_COST_CZ"))
    self.assertTrue(hasattr(CostFunction, "GATE_COST_CLIFFORD"))
    self.assertTrue(hasattr(ReadoutMode, "NONE"))
    self.assertTrue(hasattr(ReadoutMode, "FIDELITY"))
    self.assertTrue(hasattr(ReadoutMode, "QNDNESS"))


def test_active_iqm_modules_match_the_declared_distributions(self):
    from importlib.metadata import distribution, version
    from importlib.util import find_spec

    from packaging.version import Version

    selector_version = Version(version("iqm-qubit-selector"))
    client_version = Version(version("iqm-client"))
    self.assertGreaterEqual(selector_version, Version("1.1"))
    self.assertLess(selector_version, Version("2"))
    self.assertGreaterEqual(client_version, Version("35"))
    self.assertLess(client_version, Version("36"))

    for module_name, distribution_name in (
        ("iqm.qubit_selector.qubit_selector", "iqm-qubit-selector"),
        ("iqm.qiskit_iqm", "iqm-client"),
    ):
        with self.subTest(module=module_name):
            spec = find_spec(module_name)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.origin)
            module_path = Path(spec.origin).resolve()
            distribution_root = Path(
                distribution(distribution_name).locate_file("")
            ).resolve()
            self.assertTrue(module_path.is_relative_to(distribution_root))
```

- [ ] **Step 2: Run compatibility tests and verify RED**

```powershell
& $testPython -m pytest tests/test_iqm_dependency_compatibility.py -q
```

Expected: dependency-window assertion fails while files still declare `>=1,<2`; an environment shadowed by selector 1.0/client 34 may also expose the mismatch.

- [ ] **Step 3: Update every tracked dependency declaration**

Apply these exact replacements; do not change the existing
`iqm-client[qiskit]>=35,<36` window:

```text
pyproject.toml:
    "iqm-qubit-selector>=1.1,<2",

requirements.txt:
iqm-qubit-selector>=1.1,<2

src/qudits_on_qubits.egg-info/PKG-INFO:
Requires-Dist: iqm-qubit-selector<2,>=1.1

src/qudits_on_qubits.egg-info/requires.txt:
iqm-qubit-selector<2,>=1.1
```

- [ ] **Step 4: Document pipeline behavior**

Insert this block before `### Direct pipeline and final artifact`:

````markdown
### IQM automatic layout selection

`iqm-qubit-selector` is integrated at pipeline level. The notebook only enables
its safe configuration:

```python
from qudits_on_qubits import (
    IQMQubitSelectorConfig,
    WorkloadOptimizationConfig,
)

workload_optimization = WorkloadOptimizationConfig(
    initial_layouts=((0, 1, 2, 7, 3, 4),),
    seed_transpilers=(3, 7, 13),
    iqm_qubit_selector=IQMQubitSelectorConfig(
        top_k=10,
        num_trials=2000,
        cost_function="cz",
        readout_mode="none",
    ),
)
```

For IQM Crystal devices, the selector searches the full device calibration and
preselects Top-K layouts. The explicit layout above remains a comparison baseline.
The runner then compiles every merged layout with every seed across the complete
Bell measurement workload and chooses the final batch. Selector failure aborts
before submission; Aer and PIAST-Q do not accept this IQM-only configuration.
````

- [ ] **Step 5: Install only the declared compatible dependency set if local tests need it**

```powershell
& $testPython -m pip install "iqm-client[qiskit]>=35,<36" "iqm-qubit-selector>=1.1,<2"
```

Expected: the active project test interpreter reports IQM client 35.x and selector 1.1.x. Do not install into a different global/user-site interpreter.

- [ ] **Step 6: Run compatibility tests and verify GREEN**

```powershell
& $testPython -m pytest tests/test_iqm_dependency_compatibility.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit dependency and documentation alignment**

```powershell
git add pyproject.toml requirements.txt src/qudits_on_qubits.egg-info/PKG-INFO src/qudits_on_qubits.egg-info/requires.txt tests/test_iqm_dependency_compatibility.py README.md
git commit -m "docs: require IQM selector 1.1 pipeline API"
```

### Task 7: Add an opt-in zero-submit live selector/compile smoke test

**Files:**
- Create: `tests/test_iqm_layout_selector_live.py`

- [ ] **Step 1: Write the gated live test**

Create `tests/test_iqm_layout_selector_live.py` with this complete content:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ghz3_bell_canonical_baseline.ipynb"


def _resolve_iqm_env_with_notebook_contract(monkeypatch) -> Path:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    setup_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def resolve_iqm_env_path" in "".join(cell["source"])
    )
    namespace = {
        "__name__": "__iqm_selector_smoke__",
        "__file__": str(NOTEBOOK_PATH),
    }
    monkeypatch.chdir(REPO_ROOT)
    exec(
        compile(
            "".join(setup_cell["source"]),
            str(NOTEBOOK_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["resolve_iqm_env_path"](REPO_ROOT)


@pytest.mark.skipif(
    os.environ.get("QOQ_RUN_IQM_SELECTOR_SMOKE") != "1",
    reason="set QOQ_RUN_IQM_SELECTOR_SMOKE=1 for live compile-only IQM smoke",
)
def test_iqm_selector_and_full_workload_compile_never_submit(
    tmp_path,
    monkeypatch,
):
    from qudits_on_qubits.experiments.artifacts import load_basis_artifacts
    from qudits_on_qubits.experiments.backends import IQMAdapter
    from qudits_on_qubits.experiments.models import (
        ExperimentSpec,
        IQMHardware,
        IQMQubitSelectorConfig,
        PathBasis,
        WorkloadOptimizationConfig,
    )
    from qudits_on_qubits.experiments.preparation import prepare_measurements
    from qudits_on_qubits.experiments.runner import (
        _compile_measurement_workload,
    )

    env_path = _resolve_iqm_env_with_notebook_contract(monkeypatch)
    basis = PathBasis(
        REPO_ROOT
        / "experiment_inputs"
        / "reference_bases"
        / "ghz3"
        / "canonical_ez"
    )
    artifacts = load_basis_artifacts(basis, "ghz3", REPO_ROOT)
    prepared = prepare_measurements(artifacts)
    settings = tuple(
        tuple(setting)
        for setting in prepared.metadata["setting_by_circuit_index"]
    )
    selector = IQMQubitSelectorConfig(top_k=2, num_trials=200)
    spec = ExperimentSpec(
        state="ghz3",
        basis=basis,
        backend=IQMHardware(
            "garnet",
            use_metrics=True,
            env_path=env_path,
        ),
        shots=1,
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=((0, 1, 2, 7, 3, 4),),
            seed_transpilers=(3,),
            iqm_qubit_selector=selector,
        ),
        output_root=tmp_path / "forbidden-runs",
    )
    adapter = IQMAdapter(spec.backend)
    identity = adapter.resolve()
    backend = adapter.backend
    submit_guard = Mock(side_effect=AssertionError("adapter.submit called"))
    backend_run_guard = Mock(side_effect=AssertionError("backend.run called"))
    monkeypatch.setattr(adapter, "submit", submit_guard)
    monkeypatch.setattr(backend, "run", backend_run_guard)

    selection = _compile_measurement_workload(
        adapter,
        prepared.circuits,
        settings,
        spec,
        expected_identity=identity,
    )

    assert selection.metadata["selector"]["provider"] == (
        "iqm-qubit-selector"
    )
    assert len(selection.metadata["selector"]["generated_layouts"]) >= 1
    assert len(selection.batch.circuits) == len(settings) == 12
    assert selection.metadata["selected_layout"]
    assert selection.metadata["selected_seed_transpiler"] == 3
    submit_guard.assert_not_called()
    backend_run_guard.assert_not_called()
    assert not spec.output_root.exists()
```

- [ ] **Step 2: Verify default CI behavior skips the live test**

```powershell
& $testPython -m pytest tests/test_iqm_layout_selector_live.py -q
```

Expected: one skipped test, zero network calls.

- [ ] **Step 3: Run the opt-in live selector/compile smoke when credentials and compatible packages are available**

```powershell
$env:QOQ_RUN_IQM_SELECTOR_SMOKE = "1"
& $testPython -m pytest tests/test_iqm_layout_selector_live.py -q -s
Remove-Item Env:QOQ_RUN_IQM_SELECTOR_SMOKE
```

Expected: PASS with generated and selected layout metadata; IQM dashboard/job history remains unchanged because no job is submitted.

- [ ] **Step 4: Commit the smoke contract**

```powershell
git add tests/test_iqm_layout_selector_live.py
git commit -m "test: add zero-submit IQM layout selector smoke"
```

### Task 8: Full verification, simplification, and independent review

**Files:**
- Review all files changed in Tasks 1-7.

- [ ] **Step 1: Run focused tests**

```powershell
& $testPython -m pytest tests/test_experiment_models.py tests/test_public_api.py tests/test_experiment_iqm_adapter.py tests/test_experiment_runner.py tests/test_experiment_resume.py tests/test_ghz3_canonical_baseline_notebook.py tests/test_iqm_dependency_compatibility.py tests/test_iqm_layout_selector_live.py -q
```

Expected: all non-live tests pass and the live test is skipped by default.

Measure the changed experiment modules and enforce the repository target:

```powershell
& $testPython -m pytest tests/test_experiment_models.py tests/test_experiment_iqm_adapter.py tests/test_experiment_runner.py --cov=qudits_on_qubits.experiments.models --cov=qudits_on_qubits.experiments.backends.iqm --cov=qudits_on_qubits.experiments.runner --cov-report=term-missing --cov-fail-under=80 -q
```

Expected: tests pass and combined coverage is at least 80%.

- [ ] **Step 2: Run the complete project suite**

```powershell
& $testPython -m pytest -q
```

Expected: zero failures; only documented opt-in/environment skips.

- [ ] **Step 3: Run syntax, notebook, and diff checks**

```powershell
& $testPython -m py_compile src/qudits_on_qubits/experiments/models.py src/qudits_on_qubits/experiments/backends/iqm.py src/qudits_on_qubits/experiments/runner.py
& $testPython -m json.tool notebooks/ghz3_bell_canonical_baseline.ipynb NUL
git diff --check origin/main...HEAD
git status --short
```

Expected: commands exit 0; status contains only intentional files or is clean after commits.

- [ ] **Step 4: Apply the code-simplifier skill without changing behavior**

Review new helpers for duplication, names, unnecessary dynamic typing, and overly broad exception handling. Rerun the focused suite after any simplification.

- [ ] **Step 5: Run the mandated independent reviewer loop**

Spawn a fresh `reviewer` subagent over the complete relevant diff. Fix every valid P0-P2 finding, rerun relevant tests, and request a fresh complete review. Stop at `CLEAN` or after three rounds according to the repository quality gate.

- [ ] **Step 6: Push the updated branch only after verification**

```powershell
git push origin codex/ghz-bell-optimization-1-3-current
```

Expected: the existing PR updates with the selector integration commits. Do not merge.
