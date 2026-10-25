"""Check Git metadata in saved experiment evidence."""

from types import SimpleNamespace

import pytest

from qudits_on_qubits.vertical_slice import runner


@pytest.mark.parametrize(
    "status,returncode,expected",
    [("", 0, False), (" M README.md\n", 0, True), ("", 128, None)],
)
def test_provenance_distinguishes_clean_dirty_and_unavailable_git(
    monkeypatch, status, returncode, expected
):
    def git(command, **kwargs):
        if command[1] == "rev-parse":
            return SimpleNamespace(stdout="a" * 40 + "\n", returncode=0)
        assert command[1:] == ("status", "--porcelain")
        return SimpleNamespace(stdout=status, returncode=returncode)

    monkeypatch.setattr(runner.subprocess, "run", git)

    assert runner._software_provenance().dirty_worktree is expected
