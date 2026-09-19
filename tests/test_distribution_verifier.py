"""Regression checks for the distribution acceptance gate."""
import io
import tarfile
import zipfile

import pytest
from scripts import verify_distribution as check


def packages(tmp_path, *, omit_gate=False, omit_baseline=False, license=True, entry=True, omit_source=False,
             omitted_sources=()):
    wheel = tmp_path / "demo.whl"
    sdist = tmp_path / "demo.tar.gz"
    gates = sorted(check.GATES)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in gates[int(omit_gate):]:
            archive.writestr(f"qudits_on_qubits/quantum_circuits/{name}", b"gate")
        if not omit_baseline:
            for name in check.THETA_DATA:
                archive.writestr(name, b"baseline")
        if license:
            archive.writestr("demo.dist-info/licenses/LICENSE", "Apache-2.0")
        if entry:
            archive.writestr("demo.dist-info/entry_points.txt",
                             "[console_scripts]\nqoq-benchmark = demo:main\nqoq-two-qutrit-bell = demo:main\n")
    with tarfile.open(sdist, "w:gz") as archive:
        excluded = set(omitted_sources) | ({"examples/data/theta_demo.zip"} if omit_source else set())
        for name in check.SOURCE_FILES - excluded:
            info = tarfile.TarInfo("demo/" + name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    return wheel, sdist


def test_complete_distributions_are_accepted(tmp_path):
    check.verify_archives(*packages(tmp_path))


@pytest.mark.parametrize("option,message", [
    ("omit_gate", "wheel is missing"),
    ("omit_baseline", "theta_continuation/data"),
    ("license", "LICENSE"),
    ("entry", "entry-point"),
    ("omit_source", "theta_demo.zip"),
])
def test_incomplete_distributions_are_rejected(tmp_path, option, message):
    value = option not in {"license", "entry"}
    with pytest.raises(ValueError, match=message):
        check.verify_archives(*packages(tmp_path, **{option: value}))


@pytest.mark.parametrize("missing_source", [
    "CHANGELOG.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/ISSUE_TEMPLATE/config.yml",
])
def test_missing_community_file_is_rejected(tmp_path, missing_source):
    with pytest.raises(ValueError) as error:
        check.verify_archives(*packages(tmp_path, omitted_sources=(missing_source,)))
    assert str(error.value) == f"sdist is missing: {missing_source}"


def test_missing_console_command_is_rejected(tmp_path):
    wheel, sdist = packages(tmp_path)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in check.GATES:
            archive.writestr(f"qudits_on_qubits/quantum_circuits/{name}", b"gate")
        for name in check.THETA_DATA:
            archive.writestr(name, b"baseline")
        archive.writestr("demo.dist-info/LICENSE", "Apache-2.0")
        archive.writestr("demo.dist-info/entry_points.txt", "[console_scripts]\nqoq-benchmark=demo:main\n")
    with pytest.raises(ValueError, match="qoq-two-qutrit-bell"):
        check.verify_archives(wheel, sdist)


def test_runtime_smoke_is_isolated_and_failures_propagate(tmp_path, monkeypatch):
    wheel, sdist = packages(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[1:3] == ["-I", "-c"]
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 300
        assert __import__("pathlib").Path(kwargs["cwd"]).is_dir()
        raise check.subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(check.subprocess, "run", run)
    with pytest.raises(check.subprocess.CalledProcessError):
        check.main(["--wheel", str(wheel), "--sdist", str(sdist)])
    assert len(calls) == 1
    assert not __import__("pathlib").Path(calls[0][1]["cwd"]).exists()


def test_successful_smoke_returns_zero(tmp_path, monkeypatch):
    wheel, sdist = packages(tmp_path)
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **kw: None)
    assert check.main(["--wheel", str(wheel), "--sdist", str(sdist)]) == 0
