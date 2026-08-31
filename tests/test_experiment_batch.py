from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)

from qudits_on_qubits.experiments.errors import (
    ExperimentValidationError,
    JobSubmissionError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import (
    BootstrapConfig,
    CustomBackend,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    PathBasis,
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


def test_run_experiments_propagates_failure_without_partial_result(
    monkeypatch, tmp_path
):
    import qudits_on_qubits.experiments.runner as runner

    calls = 0
    completed = ExperimentResult(
        experiment_id="first",
        status=ExperimentStatus.COMPLETED,
        artifact_dir=Path("first"),
        values={"raw": {}},
        backend={"kind": "custom", "name": "target"},
        job_ids=("job-1",),
    )
    artifact_dir = tmp_path / "second" / "2026-08-27" / "failed-run"
    failure = JobSubmissionError("submission failed")
    setattr(failure, "__qoq_artifact_dir__", artifact_dir)

    def failing_on_second_call(_spec, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return completed

    monkeypatch.setattr(runner, "run_experiment", failing_on_second_call)
    experiment_store = Mock(side_effect=AssertionError("ExperimentStore called"))
    monkeypatch.setattr(runner, "ExperimentStore", experiment_store)

    with pytest.raises(JobSubmissionError, match="submission failed") as caught:
        runner.run_experiments(
            (_spec(tmp_path, "first"), _spec(tmp_path, "second"))
        )

    assert caught.value is failure
    assert calls == 2
    experiment_store.assert_not_called()
    assert not (tmp_path / "second").exists()
    assert not artifact_dir.exists()


def test_run_experiments_returns_distinct_results_in_sequential_spec_order(
    monkeypatch, tmp_path
):
    import qudits_on_qubits.experiments.runner as runner

    first_spec = _spec(tmp_path, "first")
    second_spec = _spec(tmp_path, "second")
    first_result = ExperimentResult(
        experiment_id="first-result",
        status=ExperimentStatus.COMPLETED,
        artifact_dir=Path("first-result"),
        values={"raw": {"estimate": 1}},
        backend={"kind": "custom", "name": "target"},
        job_ids=("job-1",),
    )
    second_result = ExperimentResult(
        experiment_id="second-result",
        status=ExperimentStatus.COMPLETED,
        artifact_dir=Path("second-result"),
        values={"raw": {"estimate": 2}},
        backend={"kind": "custom", "name": "target"},
        job_ids=("job-2",),
    )
    calls = []

    def record_run(spec, **_kwargs):
        calls.append(spec)
        return first_result if spec is first_spec else second_result

    monkeypatch.setattr(runner, "run_experiment", record_run)

    results = runner.run_experiments(spec for spec in (first_spec, second_spec))

    assert calls == [first_spec, second_spec]
    assert results == (first_result, second_result)
    assert results[0] is first_result
    assert results[1] is second_result


@pytest.mark.parametrize("bad", [None, "specs", {"spec": 1}, 17])
def test_batch_rejects_non_iterators_and_string_or_mapping_iterators(bad):
    from qudits_on_qubits.experiments.runner import run_experiments

    with pytest.raises(ExperimentValidationError):
        run_experiments(bad)


def test_batch_validates_every_item_before_running_any_spec(monkeypatch, tmp_path):
    import qudits_on_qubits.experiments.runner as runner

    run_calls = []
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda spec, **_kwargs: run_calls.append(spec),
    )

    with pytest.raises(ExperimentValidationError, match="every batch item"):
        runner.run_experiments((_spec(tmp_path, "first"), object()))

    assert run_calls == []


def test_batch_propagates_type_error_raised_during_iterator_consumption(
    monkeypatch, tmp_path
):
    import qudits_on_qubits.experiments.runner as runner

    failure = TypeError("iterator failed")
    run_experiment = Mock()

    def failing_specs():
        yield _spec(tmp_path, "first")
        raise failure

    monkeypatch.setattr(runner, "run_experiment", run_experiment)

    with pytest.raises(TypeError, match="iterator failed") as caught:
        runner.run_experiments(failing_specs())

    assert caught.value is failure
    run_experiment.assert_not_called()


def test_batch_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path):
    import qudits_on_qubits.experiments.runner as runner

    spec = _spec(tmp_path, "root")
    monkeypatch.setattr(runner, "run_experiment", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        runner.run_experiments([spec])
