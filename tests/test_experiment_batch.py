from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)

from qudits_on_qubits.experiments.backends import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
)
from qudits_on_qubits.experiments.errors import BackendCompatibilityError, ExperimentValidationError
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import (
    BootstrapConfig,
    CustomBackend,
    ExperimentSpec,
    ExperimentStatus,
    PathBasis,
)


class BatchAdapter:
    def __init__(self):
        self.identity = BackendIdentity("custom", "batch-target")
        self.compile_calls = 0
        self.submit_calls = 0

    def resolve(self):
        return self.identity

    def capabilities(self):
        return BackendCapabilities(False, True)

    def metadata(self):
        return {}

    def availability(self):
        return Availability(True)

    def preflight(self, circuits, shots):
        return None

    def compile(self, circuits, config):
        self.compile_calls += 1
        if self.compile_calls == 2:
            raise BackendCompatibilityError("second spec rejected")
        return CompiledBatch(tuple(circuits), self.identity)

    def submit(self, circuits, shots, options=None):
        self.submit_calls += 1
        return SubmittedJob(
            f"job-{self.submit_calls}", object(), self.identity, len(circuits), shots
        )

    def result(self, submitted, timeout=None):
        return ExecutionResult(
            tuple({"0": submitted.shots} for _ in range(submitted.circuit_count)),
            submitted.job_id,
            self.identity,
        )


def _spec(root, name):
    return ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(root / "unused"),
        backend=CustomBackend(
            object(),
            identity="batch-target",
            supports_resume=True,
            execution_mode=ExecutionMode.HARDWARE,
        ),
        shots=4,
        bootstrap=BootstrapConfig(samples=2),
        output_root=root / name,
    )


def _patch_preparation(monkeypatch):
    source = QuantumCircuit(1)
    logical = QuantumCircuit(1, 1)
    logical.measure(0, 0)
    setting = ("A0",)
    artifacts = SimpleNamespace(
        state_circuit=source,
        encoding=[[1.0]],
        source_hashes={},
        provenance={},
    )
    prepared = SimpleNamespace(
        circuits=(logical,),
        metadata={
            "setting_by_circuit_index": [setting],
            "terms": [],
            "qutrit_bit_indices_by_setting": {setting: [(0, 0)]},
            "d": 3,
        },
    )
    import qudits_on_qubits.experiments.runner as runner

    monkeypatch.setattr(runner, "load_basis_artifacts", lambda *_args, **_kwargs: artifacts)
    monkeypatch.setattr(runner, "prepare_measurements", lambda _artifacts: prepared)


def test_batch_keeps_success_and_failure_in_independent_output_roots(tmp_path, monkeypatch):
    from qudits_on_qubits.experiments.runner import run_experiments

    _patch_preparation(monkeypatch)
    first = _spec(tmp_path, "first-root")
    second = _spec(tmp_path, "second-root")
    results = run_experiments(
        (item for item in (first, second)),
        adapter=BatchAdapter(),
        _clock=lambda: datetime(2026, 8, 17, tzinfo=timezone.utc),
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 2 + 0j,
    )

    assert [result.status for result in results] == [
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
    ]
    assert results[0].artifact_dir.is_relative_to((tmp_path / "first-root").resolve())
    assert results[1].artifact_dir.is_relative_to((tmp_path / "second-root").resolve())
    assert results[0].artifact_dir.exists()
    assert results[1].artifact_dir.exists()
    assert results[1].values["failure"]["exception_type"] == "BackendCompatibilityError"


@pytest.mark.parametrize("bad", [None, "specs", {"spec": 1}, 17])
def test_batch_rejects_non_iterators_and_string_or_mapping_iterators(bad):
    from qudits_on_qubits.experiments.runner import run_experiments

    with pytest.raises(ExperimentValidationError):
        run_experiments(bad)


def test_batch_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path):
    import qudits_on_qubits.experiments.runner as runner

    spec = _spec(tmp_path, "root")
    monkeypatch.setattr(runner, "run_experiment", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        runner.run_experiments([spec])
