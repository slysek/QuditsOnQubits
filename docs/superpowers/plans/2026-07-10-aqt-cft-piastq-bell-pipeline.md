# AQT Bell Pipeline via cft-piastq Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional PiastQ/AQT execution path that accepts a backend chosen by `PiastQClient`, submits every Bell-setting circuit in one job, obtains estimated integer counts through `PiastQJob.counts()`, and delegates the Bell calculation to the existing postprocessor.

**Architecture:** Add a focused `piastq_runner.py` adapter under `bell_measurements`; it lazily imports `PiastQSampler`, treats the supplied backend as opaque, and maps the ordered list returned by `job.counts()` to `metadata["setting_by_circuit_index"]`. The adapter returns the existing Bell value plus execution objects for inspection, while `cft-piastq` remains the sole owner of quasi-distribution-to-count conversion.

**Tech Stack:** Python 3.10+, Qiskit 1.4–2.1, `cft-piastq` 0.1 with its `direct` extra, NumPy, `unittest`, `unittest.mock`, setuptools/PEP 621.

---

## File Structure

- Create `src/qudits_on_qubits/bell_measurements/piastq_runner.py`
  - Owns PiastQ job submission, ordered count mapping, local validation, and delegation to the existing Bell postprocessor.
  - Lazily imports `cft_piastq` so the normal package import does not require PiastQ dependencies.
- Create `tests/test_piastq_optional_dependency.py`
  - Locks the `pyproject.toml` optional-extra contract.
- Create `tests/test_bell_measurements_piastq_runner.py`
  - Covers the PiastQ happy path, ordering, forwarding, validation, dependency errors, opaque backend modes, and the optional real fake-backend contract.
- Modify `src/qudits_on_qubits/bell_measurements/__init__.py`
  - Exports `compute_bell_value_from_counts_aqt` as public API.
- Modify `pyproject.toml`
  - Adds the optional `piastq` dependency extra.
- Modify `tests/test_clean_repo_smoke.py`
  - Locks the README integration documentation.
- Modify `README.md`
  - Documents installation, mode selection, one-job ordering, environment-only credentials, and an explicit smoke example.

### Task 1: Declare the Optional PiastQ Dependency

**Files:**
- Create: `tests/test_piastq_optional_dependency.py`
- Modify: `pyproject.toml:27`

- [ ] **Step 1: Write the failing optional-extra contract test**

Create `tests/test_piastq_optional_dependency.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PiastQOptionalDependencyTests(unittest.TestCase):
    def test_piastq_extra_installs_cft_direct_contract(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            '[project.optional-dependencies]\n'
            'piastq = [\n'
            '    "cft-piastq[direct]>=0.1,<0.2",\n'
            ']\n',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the dependency test to verify it fails**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_piastq_optional_dependency.py" -v
```

Expected: FAIL because the expected `[project.optional-dependencies]` PiastQ block is not present.

- [ ] **Step 3: Add the optional dependency extra**

Add this block after the main `dependencies` list in `pyproject.toml`:

```toml
[project.optional-dependencies]
piastq = [
    "cft-piastq[direct]>=0.1,<0.2",
]
```

Do not add `cft-piastq` to `requirements.txt`; the base installation must remain independent of PiastQ and PCSS credentials.

- [ ] **Step 4: Run the dependency test to verify it passes**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_piastq_optional_dependency.py" -v
```

Expected: PASS, one test.

- [ ] **Step 5: Commit the optional-extra contract**

```powershell
git add pyproject.toml tests/test_piastq_optional_dependency.py
git commit -m "build: add optional cft-piastq integration"
```

### Task 2: Implement the Ordered PiastQ Happy Path

**Files:**
- Create: `src/qudits_on_qubits/bell_measurements/piastq_runner.py`
- Create: `tests/test_bell_measurements_piastq_runner.py`
- Modify: `src/qudits_on_qubits/bell_measurements/__init__.py:18-61`

- [ ] **Step 1: Write failing public-export and nine-circuit happy-path tests**

Create `tests/test_bell_measurements_piastq_runner.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.bell_measurements import piastq_runner


def _settings_3_by_3() -> list[tuple[str, str]]:
    return [(f"A{left}", f"B{right}") for left in range(3) for right in range(3)]


def _metadata_for(
    settings: list[tuple[str, ...]],
) -> dict[str, object]:
    return {
        "setting_by_circuit_index": settings,
        "terms": (
            []
            if not settings
            else [
                {
                    "settings": settings[0],
                    "powers": (1,) * len(settings[0]),
                    "coeff": 1.0,
                }
            ]
        ),
        "qutrit_bit_indices_by_setting": {
            setting: [(2 * index, 2 * index + 1) for index in range(len(setting))]
            for setting in settings
        },
        "physical_to_logical_outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
        "d": 3,
    }


class BellMeasurementPiastQRunnerTests(unittest.TestCase):
    def test_public_package_exports_aqt_helper(self):
        from qudits_on_qubits.bell_measurements import (
            compute_bell_value_from_counts_aqt,
        )

        self.assertIs(
            compute_bell_value_from_counts_aqt,
            piastq_runner.compute_bell_value_from_counts_aqt,
        )

    def test_runs_nine_circuits_in_one_job_and_maps_counts_in_order(self):
        circuits = [object() for _ in range(9)]
        settings = _settings_3_by_3()
        metadata = _metadata_for(settings)
        counts = [{format(index, "04b"): 20_480} for index in range(9)]
        expected_counts_by_setting = dict(zip(settings, counts, strict=True))
        backend = object()
        sampler_result = object()

        job = MagicMock()
        job.result.return_value = sampler_result
        job.counts.return_value = counts
        sampler = MagicMock()
        sampler.run.return_value = job
        sampler_type = MagicMock(return_value=sampler)

        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                return_value=sampler_type,
            ),
            patch.object(
                piastq_runner,
                "compute_bell_value_from_counts",
                return_value=3.0 + 4.0j,
            ) as bell_compute,
        ):
            value, execution = piastq_runner.compute_bell_value_from_counts_aqt(
                circuits,
                metadata,
                backend=backend,
                shots=20_480,
                sampler_options={"cft_job_name": "two-qutrit-bell"},
                run_options={"memory": False},
                timeout=30.0,
                poll_interval=0.5,
            )

        sampler_type.assert_called_once_with(
            backend,
            options={"cft_job_name": "two-qutrit-bell"},
        )
        sampler.run.assert_called_once_with(
            circuits,
            shots=20_480,
            memory=False,
        )
        job.result.assert_called_once_with(timeout=30.0, poll_interval=0.5)
        job.counts.assert_called_once_with()
        bell_compute.assert_called_once_with(
            expected_counts_by_setting,
            metadata["terms"],
            metadata["qutrit_bit_indices_by_setting"],
            outcome_map=metadata["physical_to_logical_outcome_map"],
            d=3,
        )
        self.assertEqual(value, 3.0 + 4.0j)
        self.assertIs(execution["backend"], backend)
        self.assertIs(execution["sampler"], sampler)
        self.assertIs(execution["job"], job)
        self.assertIs(execution["result"], sampler_result)
        self.assertEqual(execution["counts_by_setting"], expected_counts_by_setting)
        self.assertEqual(execution["circuits"], circuits)
        self.assertEqual(execution["shots"], 20_480)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new runner tests to verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected: FAIL while importing `piastq_runner` because the module and public function do not exist.

- [ ] **Step 3: Implement the minimal happy-path adapter**

Create `src/qudits_on_qubits/bell_measurements/piastq_runner.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .postprocessing import compute_bell_value_from_counts
from .sampler_circuits import decoding_kwargs_from_metadata

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def compute_bell_value_from_counts_aqt(
    sampler_circuits: Sequence["QuantumCircuit"],
    metadata: Mapping[str, Any],
    *,
    backend: Any,
    shots: int = 1024,
    sampler_options: Mapping[str, Any] | None = None,
    run_options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> tuple[complex, dict[str, Any]]:
    """Run Bell-setting circuits through cft-piastq and evaluate their counts."""

    circuits = list(sampler_circuits)
    settings = [
        tuple(setting) for setting in metadata["setting_by_circuit_index"]
    ]
    options = dict(run_options or {})

    piastq_sampler_type = _load_piastq_sampler()
    sampler = piastq_sampler_type(
        backend,
        options=dict(sampler_options or {}),
    )
    job = sampler.run(circuits, shots=shots, **options)
    sampler_result = job.result(
        timeout=timeout,
        poll_interval=poll_interval,
    )
    counts = list(job.counts())
    counts_by_setting = dict(zip(settings, counts, strict=True))

    bell_value = compute_bell_value_from_counts(
        counts_by_setting,
        metadata["terms"],
        metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )
    return complex(bell_value), {
        "backend": backend,
        "sampler": sampler,
        "job": job,
        "result": sampler_result,
        "counts_by_setting": counts_by_setting,
        "circuits": circuits,
        "shots": shots,
    }


def _load_piastq_sampler() -> Any:
    try:
        from cft_piastq import PiastQSampler
    except ImportError as exc:
        raise ImportError(
            "compute_bell_value_from_counts_aqt requires the optional "
            "cft-piastq integration. Install it with: pip install -e .[piastq]"
        ) from exc
    return PiastQSampler
```

- [ ] **Step 4: Export the new helper**

In `src/qudits_on_qubits/bell_measurements/__init__.py`, add:

```python
from .piastq_runner import compute_bell_value_from_counts_aqt
```

Add this entry to `__all__` directly after `compute_bell_value_from_counts`:

```python
"compute_bell_value_from_counts_aqt",
```

- [ ] **Step 5: Run the happy-path tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected: PASS, two tests. The test must show one `sampler.run` call containing all nine circuit objects.

- [ ] **Step 6: Commit the ordered happy path**

```powershell
git add src/qudits_on_qubits/bell_measurements/piastq_runner.py src/qudits_on_qubits/bell_measurements/__init__.py tests/test_bell_measurements_piastq_runner.py
git commit -m "feat: add cft-piastq Bell runner"
```

### Task 3: Harden Local Validation and Dependency Errors

**Files:**
- Modify: `tests/test_bell_measurements_piastq_runner.py`
- Modify: `src/qudits_on_qubits/bell_measurements/piastq_runner.py`

- [ ] **Step 1: Add failing validation tests**

Add `SimpleNamespace` to the imports in `tests/test_bell_measurements_piastq_runner.py`:

```python
from types import SimpleNamespace
```

Add these methods to `BellMeasurementPiastQRunnerTests`:

```python
    def test_rejects_empty_circuit_list_before_loading_piastq(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(ValueError, "at least one circuit"),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [],
                _metadata_for([]),
                backend=object(),
            )

    def test_rejects_invalid_shots_before_loading_piastq(self):
        circuits = [object()]
        metadata = _metadata_for([("A0",)])
        for shots in (True, 0, -1, 1.5, "100"):
            with self.subTest(shots=shots):
                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        side_effect=AssertionError("PiastQ must not load"),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "shots must be a positive integer",
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        circuits,
                        metadata,
                        backend=object(),
                        shots=shots,  # type: ignore[arg-type]
                    )

    def test_rejects_nonpositive_poll_interval_before_loading_piastq(self):
        circuits = [object()]
        metadata = _metadata_for([("A0",)])
        for poll_interval in (True, 0, -0.5):
            with self.subTest(poll_interval=poll_interval):
                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        side_effect=AssertionError("PiastQ must not load"),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "poll_interval must be a positive number",
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        circuits,
                        metadata,
                        backend=object(),
                        poll_interval=poll_interval,
                    )

    def test_rejects_shots_inside_run_options(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "pass shots via the shots argument",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object()],
                _metadata_for([("A0",)]),
                backend=object(),
                run_options={"shots": 200},
            )

    def test_rejects_circuit_setting_length_mismatch(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "number of sampler_circuits must match metadata settings",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object()],
                _metadata_for([("A0",), ("A1",)]),
                backend=object(),
            )

    def test_rejects_duplicate_settings(self):
        duplicate = ("A0", "B0")
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(ValueError, "metadata settings must be unique"),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object(), object()],
                _metadata_for([duplicate, duplicate]),
                backend=object(),
            )

    def test_rejects_result_count_length_mismatch_with_lengths(self):
        settings = [("A0",), ("A1",)]
        job = MagicMock()
        job.result.return_value = object()
        job.counts.return_value = [{"00": 100}]
        sampler = MagicMock()
        sampler.run.return_value = job
        sampler_type = MagicMock(return_value=sampler)

        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                return_value=sampler_type,
            ),
            self.assertRaisesRegex(
                ValueError,
                "expected 2 count dictionaries, received 1",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object(), object()],
                _metadata_for(settings),
                backend=object(),
            )

    def test_missing_cft_piastq_reports_install_command(self):
        with patch.dict(sys.modules, {"cft_piastq": None}):
            with self.assertRaisesRegex(
                ImportError,
                r"pip install -e \.\[piastq\]",
            ):
                piastq_runner._load_piastq_sampler()

    def test_backend_mode_is_opaque_to_the_runner(self):
        metadata = _metadata_for([("A0",)])
        for mode in ("auto", "managed", "direct"):
            with self.subTest(mode=mode):
                backend = SimpleNamespace(mode=mode)
                job = MagicMock()
                job.result.return_value = object()
                job.counts.return_value = [{"00": 32}]
                sampler = MagicMock()
                sampler.run.return_value = job
                sampler_type = MagicMock(return_value=sampler)

                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        return_value=sampler_type,
                    ),
                    patch.object(
                        piastq_runner,
                        "compute_bell_value_from_counts",
                        return_value=1.0 + 0.0j,
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        [object()],
                        metadata,
                        backend=backend,
                        shots=32,
                    )

                sampler_type.assert_called_once_with(backend, options={})
```

- [ ] **Step 2: Run the validation tests to verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected: the original two tests pass; validation tests fail because invalid inputs currently reach the loader or produce generic `zip(strict=True)` errors.

- [ ] **Step 3: Implement precise validation and count-length checks**

Replace `compute_bell_value_from_counts_aqt` in `piastq_runner.py` with this complete implementation and add the two helpers below it:

```python
def compute_bell_value_from_counts_aqt(
    sampler_circuits: Sequence["QuantumCircuit"],
    metadata: Mapping[str, Any],
    *,
    backend: Any,
    shots: int = 1024,
    sampler_options: Mapping[str, Any] | None = None,
    run_options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> tuple[complex, dict[str, Any]]:
    """Run Bell-setting circuits through cft-piastq and evaluate their counts."""

    circuits = list(sampler_circuits)
    options = dict(run_options or {})
    settings = _validate_local_inputs(
        circuits,
        metadata,
        shots=shots,
        poll_interval=poll_interval,
        run_options=options,
    )

    piastq_sampler_type = _load_piastq_sampler()
    sampler = piastq_sampler_type(
        backend,
        options=dict(sampler_options or {}),
    )
    job = sampler.run(circuits, shots=shots, **options)
    sampler_result = job.result(
        timeout=timeout,
        poll_interval=poll_interval,
    )
    counts = list(job.counts())
    if len(counts) != len(circuits):
        raise ValueError(
            f"expected {len(circuits)} count dictionaries, received {len(counts)}"
        )
    counts_by_setting = dict(zip(settings, counts, strict=True))

    bell_value = compute_bell_value_from_counts(
        counts_by_setting,
        metadata["terms"],
        metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )
    return complex(bell_value), {
        "backend": backend,
        "sampler": sampler,
        "job": job,
        "result": sampler_result,
        "counts_by_setting": counts_by_setting,
        "circuits": circuits,
        "shots": shots,
    }


def _validate_local_inputs(
    circuits: Sequence["QuantumCircuit"],
    metadata: Mapping[str, Any],
    *,
    shots: object,
    poll_interval: object,
    run_options: Mapping[str, Any],
) -> list[tuple[object, ...]]:
    if not circuits:
        raise ValueError("sampler_circuits must contain at least one circuit")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, int | float)
        or poll_interval <= 0
    ):
        raise ValueError("poll_interval must be a positive number")
    if "shots" in run_options:
        raise ValueError("pass shots via the shots argument, not run_options")
    if "setting_by_circuit_index" not in metadata:
        raise ValueError("metadata must include setting_by_circuit_index")

    settings = [
        tuple(setting) for setting in metadata["setting_by_circuit_index"]
    ]
    if len(circuits) != len(settings):
        raise ValueError("number of sampler_circuits must match metadata settings")
    if len(set(settings)) != len(settings):
        raise ValueError("metadata settings must be unique")
    return settings
```

Keep `_load_piastq_sampler` from Task 2 unchanged. It already owns the precise optional-install error required by the new test.

- [ ] **Step 4: Run the hardened runner tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected: PASS, eleven tests including subtests for each invalid value and each opaque backend mode.

- [ ] **Step 5: Commit validation behavior**

```powershell
git add src/qudits_on_qubits/bell_measurements/piastq_runner.py tests/test_bell_measurements_piastq_runner.py
git commit -m "test: harden PiastQ Bell result mapping"
```

### Task 4: Verify the Real cft-piastq Fake-Backend Contract

**Files:**
- Modify: `tests/test_bell_measurements_piastq_runner.py`

- [ ] **Step 1: Add availability checks and the local integration test**

Add these imports near the top of `tests/test_bell_measurements_piastq_runner.py`:

```python
import importlib.util

from qiskit import QuantumCircuit
```

Add these constants after the `piastq_runner` import:

```python
CFT_PIASTQ_AVAILABLE = importlib.util.find_spec("cft_piastq") is not None
QISKIT_AER_AVAILABLE = importlib.util.find_spec("qiskit_aer") is not None
```

Add this method to `BellMeasurementPiastQRunnerTests`:

```python
    @unittest.skipUnless(
        CFT_PIASTQ_AVAILABLE and QISKIT_AER_AVAILABLE,
        "requires cft-piastq and qiskit-aer",
    )
    def test_real_cft_piastq_fake_job_counts_feed_bell_postprocessing(self):
        from cft_piastq import PiastQClient

        circuit = QuantumCircuit(2, 2, name="piastq-fake-zero")
        circuit.measure([0, 1], [0, 1])
        setting = ("A0",)
        metadata = {
            "setting_by_circuit_index": [setting],
            "terms": [
                {
                    "settings": setting,
                    "powers": (1,),
                    "coeff": 1.0,
                }
            ],
            "qutrit_bit_indices_by_setting": {setting: [(0, 1)]},
            "physical_to_logical_outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
            "d": 3,
        }
        client = PiastQClient(mode="fake", owner="bell-pipeline-test")
        backend = client.fake_backend(use_backend_noise=False)

        value, execution = piastq_runner.compute_bell_value_from_counts_aqt(
            [circuit],
            metadata,
            backend=backend,
            shots=32,
            sampler_options={"cft_job_name": "fake-bell-contract"},
        )

        counts = execution["counts_by_setting"][setting]
        self.assertEqual(sum(counts.values()), 32)
        self.assertAlmostEqual(value.real, 1.0)
        self.assertAlmostEqual(value.imag, 0.0)
```

- [ ] **Step 2: Run the local contract test**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected with the local editable `cft-piastq` and `qiskit-aer` installed: PASS, including the real fake-backend contract test. Expected without either optional package: all unit tests PASS and this single integration test reports `skipped`.

- [ ] **Step 3: Commit the local integration contract**

```powershell
git add tests/test_bell_measurements_piastq_runner.py
git commit -m "test: cover cft-piastq fake Bell contract"
```

### Task 5: Document Backend Selection and the Explicit AQT Smoke Path

**Files:**
- Modify: `tests/test_clean_repo_smoke.py:15-56`
- Modify: `README.md:195`

- [ ] **Step 1: Add a failing README contract test**

Add this method to `CleanRepoSmokeTests` in `tests/test_clean_repo_smoke.py`:

```python
    def test_readme_documents_piastq_aqt_bell_execution(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## PiastQ AQT Bell Execution", readme)
        self.assertIn('mode=os.environ.get("CFT_PIASTQ_MODE", "auto")', readme)
        self.assertIn("backend=client.backend", readme)
        self.assertIn("compute_bell_value_from_counts_aqt", readme)
        self.assertIn("one PiastQ job containing nine circuits", readme)
```

- [ ] **Step 2: Run the README contract test to verify it fails**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_clean_repo_smoke.py" -v
```

Expected: FAIL because the PiastQ AQT Bell section is absent.

- [ ] **Step 3: Add the installation and smoke documentation**

Append this exact section to `README.md`:

````markdown
## PiastQ AQT Bell Execution

Install the optional PiastQ integration in the environment used by this
project:

```powershell
python -m pip install -e ".[piastq]"
```

Choose `auto`, `managed`, or `direct` when constructing `PiastQClient`. The Bell
helper receives `client.backend` unchanged and does not choose or override the
execution mode. Credentials must come from environment variables; do not place
PCSS tokens or dashboard API keys in notebooks or source files.

This explicit smoke example prepares the zero state in the two-qutrit encoding,
builds all nine Bell-setting circuits, and submits one PiastQ job containing
nine circuits:

```python
import os

from qiskit import QuantumCircuit

from cft_piastq import PiastQClient
from qudits_on_qubits.bell_measurements import (
    build_sampler_circuits_for_candidate,
    canonical_Ez,
    compute_bell_value_from_counts_aqt,
)

state_circuit = QuantumCircuit(4)
sampler_circuits, metadata = build_sampler_circuits_for_candidate(
    candidate="two_qutrit",
    state_circuit=state_circuit,
    E=canonical_Ez(),
    qutrit_qubits=((0, 1), (2, 3)),
)

client = PiastQClient(
    mode=os.environ.get("CFT_PIASTQ_MODE", "auto"),
    owner=os.environ["CFT_PIASTQ_OWNER"],
    token=os.environ.get("PCSS_TOKEN") or os.environ.get("PCSS_QAPI_TOKEN"),
    dashboard_api_url=os.environ.get("CFT_PIASTQ_DASHBOARD_API_URL"),
    dashboard_api_key=os.environ.get("CFT_PIASTQ_DASHBOARD_API_KEY"),
)

bell_value, execution = compute_bell_value_from_counts_aqt(
    sampler_circuits,
    metadata,
    backend=client.backend,
    shots=20_480,
    sampler_options={"cft_job_name": "two-qutrit-bell-smoke"},
    timeout=900.0,
    poll_interval=5.0,
)

print("Bell value:", bell_value)
print("PiastQ job:", execution["job"].job_id())
```

`job.result()` remains available in `execution["result"]` as a Qiskit
`SamplerResult`. Bell postprocessing uses the estimated integer dictionaries
returned by `PiastQJob.counts()`; this project does not independently multiply
or round the quasi probabilities.

The example can contact the managed dashboard or direct AQT provider and can
consume real hardware shots. Run it only as an intentional manual smoke test.
````

- [ ] **Step 4: Run the README and focused PiastQ tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_clean_repo_smoke.py" -v
python -m unittest discover -s tests -p "test_piastq_optional_dependency.py" -v
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
```

Expected: all three commands PASS. The real fake-backend contract may report one intentional skip when the optional local packages are not installed.

- [ ] **Step 5: Commit the PiastQ documentation**

```powershell
git add README.md tests/test_clean_repo_smoke.py
git commit -m "docs: explain PiastQ AQT Bell execution"
```

### Task 6: Final Focused and Repository Verification

**Files:**
- Verify only; no source changes expected.

- [ ] **Step 1: Run all new PiastQ and Bell-focused tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p "test_piastq_optional_dependency.py" -v
python -m unittest discover -s tests -p "test_bell_measurements_piastq_runner.py" -v
python -m unittest discover -s tests -p "test_clean_repo_smoke.py" -v
```

Expected: all unit and documentation tests PASS; the optional real fake-backend test either PASSes or reports one documented skip.

- [ ] **Step 2: Run the existing generic Bell smoke import**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Resolve-Path src)
python -c "from qudits_on_qubits.bell_measurements import compute_bell_value_from_counts, compute_bell_value_from_counts_aqt; print(compute_bell_value_from_counts.__name__, compute_bell_value_from_counts_aqt.__name__)"
```

Expected output:

```text
compute_bell_value_from_counts compute_bell_value_from_counts_aqt
```

This command must succeed even when `cft-piastq` is not installed, proving that the dependency is imported lazily.

- [ ] **Step 3: Run the full repository suite and compare with the baseline**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

Expected: every new PiastQ/AQT test passes. On the isolated baseline used to create this plan, the only existing failure was `test_build_iqm_pass_manager_preserves_state_fidelity`, reporting that the active circuit had five qubits while its reference had four. Do not attribute that pre-existing IQM failure to this AQT change; no additional failures are acceptable.

- [ ] **Step 4: Check formatting and final branch state**

Run:

```powershell
git diff --check
git status --short
git log -5 --oneline
```

Expected: `git diff --check` prints nothing, `git status --short` is empty, and the log contains the five implementation commits from Tasks 1–5 in addition to the design/plan documentation commits.

- [ ] **Step 5: Leave real AQT execution manual**

Do not execute the README hardware smoke automatically. Its use of `mode="auto"` may select the managed dashboard or direct AQT provider and consume 20,480 shots per Bell-setting circuit. Run it only after the user explicitly authorizes a remote submission and confirms the intended credentials, mode, and shot budget.

## Plan Self-Review

- **Spec coverage:** Tasks 1–5 cover the optional dependency, opaque backend selection, one-job/N-circuit ordering, `PiastQJob.counts()`, delegation to existing Bell math, agreed execution metadata, validation, security, fake integration, and manual remote smoke path.
- **Scope:** The plan does not modify `cft-piastq`, IQM execution, Aer's generic runner, or existing Bell formulas.
- **Type consistency:** The public function name, arguments, return type, execution keys, setting tuples, and count mappings match the approved design throughout every task.
- **Baseline:** The known IQM fidelity failure is recorded only as comparison evidence and is not part of the AQT implementation.
