import io
import os
import queue
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.direct_basis import iqm_transpiler_harness as harness
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate


def test_progress_reports_unknown_synthesis_eta_and_trial_bar():
    from qudits_on_qubits.benchmarks.direct_basis.harness_progress import CandidateProgress

    progress = CandidateProgress("baseline/I", os.getpid(), started=100, total=40)
    progress.update("CZ3 synthesis (BQSKit)", now=110)
    text = progress.render(now=190, activity="CPU +12.0s")
    assert "baseline/I" in text and "CZ3 synthesis" in text
    assert "00:01:30" in text and "ETA unknown" in text
    assert "CPU +12.0s" in text
    assert "100%" not in text
    progress.update("transpiling preset_exact seed=10", now=200, completed=20)
    text = progress.render(now=205, activity="CPU +1.0s")
    assert "20/40" in text and "20 remaining" in text and "50%" in text


@pytest.mark.parametrize("stage", ["preparing optimized graph circuit", "exporting candidate circuits", "recording preset_exact seed=0"])
def test_non_synthesis_stages_do_not_claim_to_be_synthesis(stage):
    from qudits_on_qubits.benchmarks.direct_basis.harness_progress import CandidateProgress

    progress = CandidateProgress("baseline/I", os.getpid(), started=0, total=40)
    progress.update(stage, now=1)
    text = progress.render(now=2, activity="CPU sampling")
    assert stage in text and "ETA unknown" in text
    assert "synthesis" not in text


def test_monitor_keeps_reporting_without_new_worker_events():
    from qudits_on_qubits.benchmarks.direct_basis.harness_progress import ProgressMonitor

    events = queue.Queue()
    output = io.StringIO()
    with ProgressMonitor(events, interval=0.03, total_candidates=1, stream=output):
        events.put(dict(kind="start", key="I", label="baseline/I", pid=os.getpid(), total=40, at=time.monotonic()))
        events.put(dict(kind="stage", key="I", stage="CZ3 synthesis (BQSKit)", completed=None, at=time.monotonic()))
        deadline = time.monotonic() + 3
        while output.getvalue().count("CZ3 synthesis") < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        events.put(dict(kind="finish", key="I", status="gate_validation_failed", at=time.monotonic()))
    text = output.getvalue()
    assert text.count("CZ3 synthesis") >= 2
    assert "gate_validation_failed" in text
    assert "100%" not in text


def test_stage_scope_is_reset_after_errors():
    from qudits_on_qubits.benchmarks.direct_basis.harness_progress import report_stage, stage_scope

    events = []
    with pytest.raises(ValueError), stage_scope(lambda stage, completed: events.append(stage)):
        report_stage("F3 synthesis")
        raise ValueError("test")
    report_stage("outside scope")
    assert events == ["F3 synthesis"]


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf"), True, "10"])
def test_invalid_progress_interval_rejected_before_work(interval):
    config = harness.IqmTranspilerHarnessConfig(
        "two_qutrit", 2, object(), "offline", False, [], progress_interval=interval,
    )
    with patch.object(harness, "_backend_metadata") as metadata:
        with pytest.raises(ValueError, match="progress_interval"):
            harness.run_iqm_transpiler_harness(config)
        metadata.assert_not_called()


def test_candidate_stages_include_synthesis_and_trial_counts(exact_cz3_synthesis):
    from qudits_on_qubits.benchmarks.direct_basis.harness_progress import stage_scope
    from test_iqm_harness_parallel import _test_runner

    candidate = DirectBasisCandidate("I", "test", np.eye(3))
    config = harness.IqmTranspilerHarnessConfig(
        "two_qutrit", 2, object(), "offline", False, [candidate],
        strategy_names=("ok", "failed"), n_transpile_runs=2,
    )
    events = []
    with stage_scope(lambda stage, completed: events.append((stage, completed))):
        rows = harness._run_candidate(candidate, config, {}, _test_runner)
    assert len(rows) == 4
    assert any("F3 synthesis" in stage for stage, _ in events)
    assert any("CZ3" in stage for stage, _ in events)
    assert [completed for _, completed in events if completed is not None][-1] == 4


def test_cpu_activity_includes_children_and_does_not_label_idle_as_hung(monkeypatch):
    from qudits_on_qubits.benchmarks.direct_basis import harness_progress as progress

    counters = {10: 1.0, 20: 10.0}

    def process(pid):
        return SimpleNamespace(
            pid=pid, create_time=lambda: float(pid), is_running=lambda: True,
            status=lambda: "running", children=lambda recursive: [process(20)],
            cpu_times=lambda: SimpleNamespace(user=counters[pid], system=0),
        )

    monkeypatch.setattr(progress.psutil, "Process", process)
    sampler = progress.ProcessActivity(10)
    assert "2 processes" in sampler.sample()
    counters[20] += 5
    assert "CPU +5.0s" in sampler.sample()
    idle = sampler.sample()
    assert "no CPU activity observed" in idle and "may be waiting" in idle
    assert "hung" not in idle


def test_process_exit_and_permission_failure_are_visible(monkeypatch):
    from qudits_on_qubits.benchmarks.direct_basis import harness_progress as progress

    for error, expected in (
        (progress.psutil.NoSuchProcess(123), "process exited"),
        (progress.psutil.AccessDenied(123), "CPU activity unavailable"),
    ):
        with patch.object(progress.psutil, "Process", side_effect=error):
            assert progress.ProcessActivity(123).sample() == expected


def test_disabled_progress_does_not_start_a_monitor():
    from qudits_on_qubits.benchmarks.direct_basis import harness_progress as progress

    candidate = DirectBasisCandidate("unsupported", "test", None)
    config = harness.IqmTranspilerHarnessConfig(
        "two_qutrit", 2, object(), "offline", False, [candidate], progress_interval=0,
    )
    with patch.object(progress, "ProgressMonitor") as monitor:
        trials, _, _ = harness.run_iqm_transpiler_harness(config)
    monitor.assert_not_called()
    assert trials.iloc[0].status == "unsupported_candidate"
