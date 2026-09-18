from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig


@pytest.fixture
def cli():
    script = Path(__file__).resolve().parents[1] / "scripts/run_theta_benchmark.py"
    spec = importlib.util.spec_from_file_location("theta_benchmark_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calls(cli, monkeypatch):
    seen = []

    def run(config, output_dir, **kwargs):
        seen.append((config, output_dir, kwargs))
        kwargs["progress"]({"stage": "point", "index": 0, "theta": 0.0})
        return {"output_dir": str(output_dir), "points": [{"status": "failed"}],
                "counts": {"failed": 1}, "full_circuits": []}

    monkeypatch.setattr(cli, "run_theta_benchmark", run)
    monkeypatch.setattr(cli, "generate_report", lambda output_dir, **kwargs: Path(output_dir) / "report.md")
    return seen


def _existing(tmp_path, **kwargs):
    config = ThetaBenchmarkConfig(**kwargs)
    store = RunStore.create(tmp_path / "existing", {"fingerprint": "cli-fixture", "config": config.to_dict()})
    return store, config


def test_default_protocol_and_scientific_failures_return_zero(cli, calls, tmp_path, capsys):
    output = tmp_path / "new"
    assert cli.main(["--output-dir", str(output), "--mode", "gates"]) == 0
    config, directory, kwargs = calls[0]
    assert config.to_dict() == ThetaBenchmarkConfig().to_dict()
    assert Path(directory) == output
    assert kwargs["mode"] == "gates" and kwargs["resume"] is False
    printed = capsys.readouterr().out
    assert "point" in printed and "report.md" in printed and "failed" in printed


def test_custom_grid_prefix_and_all_flags(cli, calls, tmp_path):
    args = ["--output-dir", str(tmp_path / "new"), "--theta-points", "41", "--limit-points", "3",
            "--max-nfev", "5", "--max-subdivisions", "0", "--f3-tolerance", "1e-11",
            "--cz3-tolerance", "1e-6", "--transpiler-seeds", "7", "19", "--optimization-level", "2",
            "--baseline-qpy", str(tmp_path / "CZ3.qpy"), "--baseline-sha256", "a" * 64,
            "--states", "two_qutrit", "ghz3"]
    assert cli.main(args) == 0
    config = calls[0][0]
    assert len(config.grid()) == 3 and config.grid()[-1] < 0.1
    assert config.max_nfev == 5 and config.max_subdivisions == 0
    assert config.f3_tolerance == 1e-11 and config.cz3_tolerance == 1e-6
    assert config.transpiler_seeds == (7, 19) and config.optimization_level == 2
    assert config.states == ("two_qutrit", "ghz3")
    assert config.baseline_sha256 == "a" * 64


def test_resume_uses_stored_config_and_allows_identical_explicit_arguments(cli, calls, tmp_path):
    store, config = _existing(tmp_path, theta_points=9, limit_points=3, max_nfev=17,
                              transpiler_seeds=(9, 2), states=("ghz3",))
    assert cli.main(["--resume", str(store.root), "--mode", "all", "--max-nfev", "17",
                     "--transpiler-seeds", "9", "2", "--states", "ghz3"]) == 0
    assert calls[0][0] == config and calls[0][2]["resume"] is True


@pytest.mark.parametrize("extra", [["--max-nfev", "19"], ["--limit-points", "2"],
    ["--f3-tolerance", "1e-8"], ["--transpiler-seeds", "2", "1"], ["--states", "ghz3"],
    ["--baseline-sha256", "b" * 64], ["--output-dir", "different-run"]])
def test_resume_rejects_conflicting_arguments(cli, calls, tmp_path, extra, capsys):
    store, _ = _existing(tmp_path)
    assert cli.main(["--resume", str(store.root), *extra]) == 2
    assert calls == []
    assert "error" in capsys.readouterr().err.lower()


def test_circuits_mode_requires_existing_run_and_loads_it(cli, calls, tmp_path):
    assert cli.main(["--mode", "circuits", "--output-dir", str(tmp_path / "missing")]) == 2
    assert calls == []
    store, config = _existing(tmp_path, theta_points=7)
    assert cli.main(["--mode", "circuits", "--output-dir", str(store.root)]) == 0
    assert calls[0][0] == config and calls[0][2]["resume"] is True


@pytest.mark.parametrize("args", [["--theta-points", "1"], ["--max-nfev", "0"],
    ["--max-subdivisions", "-1"], ["--f3-tolerance", "nan"], ["--cz3-tolerance", "inf"],
    ["--f3-tolerance", "2e-10"], ["--cz3-tolerance", "0"],
    ["--limit-points", "42"], ["--transpiler-seeds", "1", "1"],
    ["--baseline-sha256", "bad"]])
def test_invalid_configuration_never_runs(cli, calls, args, tmp_path):
    assert cli.main(["--output-dir", str(tmp_path / "run"), *args]) == 2
    assert calls == []


@pytest.mark.parametrize("error, expected", [(ValueError("checksum mismatch"), 2),
    (FileNotFoundError("manifest missing"), 2), (RuntimeError("runtime crashed"), 1),
    (KeyboardInterrupt(), 130)])
def test_process_errors_have_distinct_exit_codes(cli, calls, monkeypatch, tmp_path, error, expected):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(cli, "run_theta_benchmark", fail)
    assert cli.main(["--output-dir", str(tmp_path / "run")]) == expected


def test_argument_parser_errors_return_two(cli, calls):
    assert cli.main(["--theta-points", "not-an-integer"]) == 2
    assert calls == []


def test_explicit_relaxed_cz3_threshold_reaches_runner(cli, calls, tmp_path):
    assert cli.main(["--output-dir", str(tmp_path / "relaxed"), "--cz3-tolerance", "5e-4"]) == 0
    assert calls[0][0].cz3_tolerance == 5e-4
    assert calls[0][0].f3_tolerance == 1e-10
