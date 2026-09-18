"""Recovery durability and independent validation, without real BQSKit runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import qpy

from qudits_on_qubits.benchmarks.theta_continuation import reassessment as module
from qudits_on_qubits.benchmarks.theta_continuation import runner
from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint
from qudits_on_qubits.benchmarks.theta_continuation.cz3_continuation import (
    ContinuationResult, Cz3FitResult,
)
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig


@pytest.fixture
def source(tmp_path, monkeypatch):
    config = ThetaBenchmarkConfig(theta_points=100_001, limit_points=2,
                                  transpiler_seeds=(0,), states=("two_qutrit",))
    template = runner.load_cz3_template(config.baseline_qpy)

    def continuation(template, state, theta, *, target_id, **kwargs):
        circuit = template.bind(state.parameters)
        metrics = module._unrestricted_metrics(circuit, theta)
        fit = Cz3FitResult(state.parameters.copy(), circuit, 0.0, False,
                          metrics["E_norm"], metrics["L_norm"], 1, 1, 1, "controlled", 0.01)
        attempt = {"id": target_id + "__attempt_001", "theta": theta, "requested_grid_point": True,
                   "warm_start_parent_id": state.parent_id, "p_start": state.parameters.tolist(),
                   "fit": fit.to_dict()}
        return ContinuationResult(fit, state, [attempt])

    def rejected(encoding, **kwargs):
        from dataclasses import asdict
        from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
        validation = runner.optimized_gates.validate_code_space_gate(
            template.original_circuit, np.kron(encoding, encoding), qutrit_cz(),
        )
        metrics = asdict(validation)
        raise runner.optimized_gates.GateSynthesisError(f"CZ3 failed gate validation: {metrics}", metrics)

    monkeypatch.setattr(runner, "continue_cz3", continuation)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", rejected)
    root = tmp_path / "source"
    runner.run_theta_benchmark(config, root, mode="gates")
    return root, config, template


@pytest.fixture(autouse=True)
def prevent_real_recovery(monkeypatch):
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("real BQSKit called"))
    # Rendering has its own suite; keep these tests about physical results.
    monkeypatch.setitem(sys.modules,
                        "qudits_on_qubits.benchmarks.theta_continuation.reassessment_report",
                        SimpleNamespace(generate_reassessment_report=lambda root: Path(root) / "report.md"))


def _hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _run(source, tmp_path, monkeypatch, **kwargs):
    root, config, template = source
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda _: template.original_circuit.copy())
    return module.run_reassessment(root, tmp_path / "new", workers=1, **kwargs)


def test_recovery_reuses_f3_preserves_source_and_historic_branch(source, tmp_path, monkeypatch):
    root, config, template = source
    before = _hashes(root)
    result = _run(source, tmp_path, monkeypatch)
    assert result["counts"] == {"total": 2, "correct": 2, "failed": 0, "reconstructed": 1,
                                "full_circuits": 2, "full_circuits_correct": 2}
    assert _hashes(root) == before
    store = RunStore.open(result["output_dir"])
    original = RunStore.open(root).read_bundle("points/00001")
    point = store.read_bundle("points/00001")
    assert point["documents"]["attempts.json"] == original["documents"]["attempts.json"]
    assert point["metadata"]["branch_state"] == original["metadata"]["branch_state"]
    assert point["metadata"]["continuation_valid"] is False
    details = point["metadata"]["reassessment"]
    assert details["origin"] == "reconstructed"
    assert details["fallback_comparison"]["matches_original"] is True
    assert details["attempts"][0]["valid_at_new_tolerance"] is True
    assert store.read_bundle("recovery/00001")["metadata"]["status"] == "raw_saved"
    for index in range(2):
        for name in ("f3_optimal.qpy", "f3_alpha_zero.qpy"):
            assert (root / f"points/{index:05d}" / name).read_bytes() == (store.root / f"points/{index:05d}" / name).read_bytes()
    assert all(row["fidelity"] > 1 - 1e-8 for row in result["full_circuits"])


def test_resume_reuses_recovered_raw_and_full_circuits(source, tmp_path, monkeypatch):
    result = _run(source, tmp_path, monkeypatch)
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("raw recovered twice"))
    monkeypatch.setattr(runner, "benchmark_direct_basis", lambda **_: pytest.fail("full circuit rebuilt"))
    resumed = module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)
    assert resumed == result


def test_raw_checkpoint_survives_interruption_before_point_publication(source, tmp_path, monkeypatch):
    original_build = module._build_point
    monkeypatch.setattr(module, "_build_point", lambda *_: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(source, tmp_path, monkeypatch)
    assert (tmp_path / "new/recovery/00001/raw_cz3.qpy").is_file()
    monkeypatch.setattr(module, "_build_point", original_build)
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("completed recovery repeated"))
    assert module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)["counts"]["correct"] == 2


def test_raw_rejected_at_new_limit_is_still_checkpointed(source, tmp_path, monkeypatch):
    bad = source[2].original_circuit.copy()
    bad.u(0.02, 0.0, 0.0, 0)
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda _: bad)
    result = module.run_reassessment(source[0], tmp_path / "new", workers=1)
    details = result["points"][1]["reassessment"]
    assert details["fallback_comparison"]["matches_original"] is False
    assert details["fallback_comparison"]["actual_metrics"]["E_norm"] > 5e-4
    assert any("bqskit:" in error for error in details["candidate_errors"])
    assert (tmp_path / "new/recovery/00001/raw_cz3.qpy").is_file()
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("rejected recovery repeated"))
    assert module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)["counts"] == result["counts"]


def test_failed_synthesis_is_persisted_and_not_retried(source, tmp_path, monkeypatch):
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda _: (_ for _ in ()).throw(ValueError("controlled failure")))
    result = module.run_reassessment(source[0], tmp_path / "new", workers=1)
    store = RunStore.open(tmp_path / "new")
    assert store.read_bundle("recovery/00001")["metadata"]["status"] == "synthesis_failed"
    assert "controlled failure" in result["points"][1]["reassessment"]["candidate_errors"][0]
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("failed synthesis repeated"))
    module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)


@pytest.mark.parametrize("path", ["points/00000/E.npy", "baseline/cz3_original.qpy"])
def test_source_corruption_is_rejected_before_creating_output(source, tmp_path, path):
    (source[0] / path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("path", ["recovery/00001/raw_cz3.qpy", "points/00001/cz3_selected.qpy",
                                  "circuits/two_qutrit/00001/graph_state_transpiled.qpy"])
def test_resume_rejects_corrupt_new_artifacts(source, tmp_path, monkeypatch, path):
    _run(source, tmp_path, monkeypatch)
    (tmp_path / "new" / path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)


def test_resume_rejects_threshold_change(source, tmp_path, monkeypatch):
    _run(source, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="fingerprint"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True, cz3_tolerance=6e-4)


def test_source_manifest_content_is_verified(source, tmp_path):
    path = source[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["max_nfev"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1)


@pytest.mark.parametrize("workers", [True, 0, 4, 1.5])
def test_invalid_workers_fail_before_source_read(tmp_path, workers):
    with pytest.raises((ValueError, TypeError), match="workers"):
        module.run_reassessment(tmp_path / "missing", tmp_path / "new", workers=workers)


@pytest.mark.parametrize("nesting", ["same", "child", "parent"])
def test_output_must_not_overlap_source(tmp_path, nesting):
    source = tmp_path / "source"
    output = {"same": source, "child": source / "new", "parent": tmp_path}[nesting]
    with pytest.raises(ValueError, match="separate"):
        module.run_reassessment(source, output)


def test_tighter_reassessment_not_permitted(source, tmp_path):
    with pytest.raises(ValueError, match="at least"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1, cz3_tolerance=1e-6)


def test_parallel_dispatch_uses_spawn_and_separate_recovery_bundles(source, tmp_path, monkeypatch):
    calls = []

    class Executor:
        def __init__(self, **kwargs):
            calls.append((kwargs["max_workers"], kwargs["mp_context"].get_start_method()))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, fn, *args):
            value = fn(*args)
            return SimpleNamespace(result=lambda: value)

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda _: source[2].original_circuit.copy())
    assert module.run_reassessment(source[0], tmp_path / "new", workers=3)["counts"]["correct"] == 2
    assert calls == [(3, "spawn")]


def test_recovery_requires_original_package_versions(source, tmp_path, monkeypatch):
    real = module.runtime_provenance
    monkeypatch.setattr(module, "runtime_provenance", lambda: {**real(), "versions": {"python": "changed"}})
    with pytest.raises(ValueError, match="package versions"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1)


def _cli():
    path = Path(__file__).parents[1] / "scripts/reassess_theta_benchmark.py"
    spec = importlib.util.spec_from_file_location("reassessment_cli", path)
    module_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_cli)
    return module_cli


def test_cli_defaults_and_dispatch(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(module, "run_reassessment", lambda **kwargs: calls.append(kwargs) or
                        {"counts": {"correct": 41}, "report": "report.md"})
    cli = _cli()
    assert cli.main(["--source-run", str(tmp_path / "source"), "--output-dir", str(tmp_path / "new"), "--resume"]) == 0
    assert calls[0]["cz3_tolerance"] == 5e-4
    assert calls[0]["workers"] == 3
    assert calls[0]["resume"] is True
    assert '"correct": 41' in capsys.readouterr().out


@pytest.mark.parametrize("error,code", [(KeyboardInterrupt(), 130), (ValueError("bad"), 2), (RuntimeError("bad"), 1)])
def test_cli_error_statuses(monkeypatch, error, code):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(module, "run_reassessment", fail)
    assert _cli().main(["--source-run", "source", "--output-dir", "new"]) == code


def test_cli_help_and_invalid_arguments():
    assert _cli().main(["--help"]) == 0
    assert _cli().main([]) == 2


def _rehash_file(path):
    complete_path = path.with_name("complete.json")
    complete = json.loads(complete_path.read_text())
    complete["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete))


def test_saved_raw_fallback_avoids_recovery(source, tmp_path):
    raw = source[0] / "points/00001/cz3_fallback.qpy"
    with raw.open("wb") as stream:
        qpy.dump(source[2].original_circuit, stream)
    _rehash_file(raw)
    result = module.run_reassessment(source[0], tmp_path / "new", workers=1)
    assert result["counts"]["reconstructed"] == 0
    assert result["points"][1]["reassessment"]["origin"] == "source_saved"
    assert result["counts"]["correct"] == 2


def test_rehashed_raw_operator_cannot_break_recovery_binding(source, tmp_path, monkeypatch):
    _run(source, tmp_path, monkeypatch)
    raw = tmp_path / "new/recovery/00001/raw_cz3.qpy"
    with raw.open("rb") as stream:
        circuit = qpy.load(stream)[0]
    circuit.global_phase += 0.1
    with raw.open("wb") as stream:
        qpy.dump(circuit, stream)
    _rehash_file(raw)
    with pytest.raises(ValueError, match="raw recovery"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True)


def test_rehashed_attempt_norm_is_physically_rejected(source, tmp_path):
    path = source[0] / "points/00001/attempts.json"
    data = json.loads(path.read_text())
    data["attempts"][0]["fit"]["E_norm"] = 0.1
    path.write_text(json.dumps(data))
    _rehash_file(path)
    with pytest.raises(ValueError, match="historical attempt"):
        module.run_reassessment(source[0], tmp_path / "new", workers=1)


def test_both_candidates_rejected_produce_failed_point_and_no_full_circuit(source, tmp_path, monkeypatch):
    bad = source[2].original_circuit.copy()
    bad.u(0.02, 0.0, 0.0, 0)
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda _: bad)
    result = module.run_reassessment(source[0], tmp_path / "new", workers=1, cz3_tolerance=1e-5)
    assert result["counts"]["failed"] == 1
    assert result["counts"]["full_circuits"] == 1
    assert result["points"][1]["cz3_metrics"] == {}
    assert result["points"][1]["errors"]
    store = RunStore.open(tmp_path / "new")
    point = RunStore.open(source[0]).read_bundle("points/00001")
    assert module._recover_raw(str(store.root), store.manifest["fingerprint"], 1,
                               point["metadata"]["theta"], module._original_fallback_metrics(point)) == 1
    monkeypatch.setattr(module.optimized_gates, "_compile_cz3", lambda *_: pytest.fail("failed point retried"))
    assert module.run_reassessment(source[0], tmp_path / "new", workers=1, resume=True,
                                   cz3_tolerance=1e-5)["counts"] == result["counts"]
