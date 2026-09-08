import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qudits_on_qubits.experiments import AerIdeal, ExperimentSpec, PathBasis
from qudits_on_qubits.experiments.backends.base import (
    BackendIdentity, BackendCapabilities, Availability, CompiledBatch, SubmittedJob, ExecutionResult,
)
from qudits_on_qubits.experiments.measurement import RandomizedBlocks


class RecordingAdapter:
    def __init__(self, fail_submit=None, fail_result=None, leakage=False):
        self.identity = BackendIdentity("aer_ideal", "test")
        self.submissions = []
        self.preflights = []
        self.restores = []
        self.fail_submit = fail_submit
        self.fail_result = fail_result
        self.leakage = leakage
        self.jobs = {}

    def resolve(self): return self.identity
    def capabilities(self): return BackendCapabilities(True, True, 4)
    def availability(self): return Availability(True)
    def compile(self, circuits, config): return CompiledBatch(tuple(c.copy() for c in circuits), self.identity)
    def preflight(self, circuits, shots): self.preflights.append((len(circuits), shots))
    def submit(self, circuits, shots, options=None):
        index = len(self.submissions)
        self.submissions.append((tuple(circuits), shots))
        if index == self.fail_submit: raise RuntimeError("unknown remote submission")
        job = SubmittedJob(f"job-{index}", object(), self.identity, len(circuits), shots)
        self.jobs[job.job_id] = (job, circuits)
        return job
    def result(self, job, timeout=None):
        if int(job.job_id.split('-')[1]) == self.fail_result: raise TimeoutError()
        circuits = self.jobs[job.job_id][1]
        digit = "1" if self.leakage else "0"
        return ExecutionResult(tuple({digit*c.num_clbits: job.shots} for c in circuits), job.job_id, self.identity)
    def restore_job(self, job_id, *, circuit_count=None, shots=None):
        self.restores.append(job_id)
        return self.jobs[job_id][0]
    def metadata(self): return {}


def make_spec(tmp_path, state="two_qutrit", draws=12, cap=12, shots=8, limit=3):
    parties = {"two_qutrit": 2, "ghz3": 3, "ame43": 4}[state]
    directory = tmp_path / "basis"
    directory.mkdir(exist_ok=True)
    with (directory / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump(QuantumCircuit(parties*2), handle)
    np.save(directory / "E.npy", np.eye(4, 3))
    return ExperimentSpec(state=state, basis=PathBasis(directory), backend=AerIdeal(17),
                          measurement=RandomizedBlocks(draws, shots, cap, limit), output_root=tmp_path / "runs")


def all_settings_draw(state="two_qutrit"):
    counts = {"two_qutrit": (3,3), "ghz3": (3,3,2), "ame43": (3,3,2,2)}[state]
    choices = itertools.cycle(itertools.chain.from_iterable(itertools.product(*(range(n) for n in counts))))
    return lambda n: next(choices)


def test_blocks_are_executed_separately_with_k_shots_and_preflight_all(tmp_path):
    from qudits_on_qubits.experiments.block_runner import run_randomized_experiment
    adapter = RecordingAdapter()
    result = run_randomized_experiment(make_spec(tmp_path), adapter=adapter, _randbelow=all_settings_draw())
    assert result.status.value == "completed"
    assert [len(batch) for batch, _ in adapter.submissions] == [3,3,3,3]
    assert all(shots == 8 for _, shots in adapter.submissions)
    assert [c.name for batch, _ in adapter.submissions for c in batch] == [f"block_{i:08d}" for i in range(12)]
    assert result.values["budget"]["completed_shots"] == 96
    assert len(adapter.preflights) >= 4
    assert (result.artifact_dir / "batches/0003/counts.json").is_file()


def test_coverage_cap_saves_schedule_without_touching_backend(tmp_path):
    from qudits_on_qubits.experiments.block_runner import run_randomized_experiment
    result = run_randomized_experiment(make_spec(tmp_path), adapter=object(), _randbelow=lambda n: 0)
    assert result.status.value == "failed"
    assert result.values["preparation"]["status"] == "coverage_limit_reached"
    assert (result.artifact_dir / "schedule.json").is_file()
    assert not (result.artifact_dir / "batches").exists()


def test_full_leakage_keeps_raw_and_finishes_with_conditional_unavailable(tmp_path):
    from qudits_on_qubits.experiments.block_runner import run_randomized_experiment
    result = run_randomized_experiment(make_spec(tmp_path), adapter=RecordingAdapter(leakage=True), _randbelow=all_settings_draw())
    assert result.status.value == "completed"
    assert result.values["raw"] == {"real": 0.0, "imag": 0.0}
    assert result.values["conditional"] is None
    assert result.values["conditional_reason"] == "no_accepted_shots"
