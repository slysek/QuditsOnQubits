"""Tests for installed two-qutrit Bell vertical-slice command."""

from __future__ import annotations

from pathlib import Path

from qudits_on_qubits.vertical_slice.cli import main


def test_cli_prints_result_and_manifest(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--shots",
            "256",
            "--seed",
            "42",
            "--output-root",
            str(tmp_path),
        ]
    )

    lines = capsys.readouterr().out.splitlines()
    output = dict(line.split("=", 1) for line in lines)
    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["benchmark"] == "two_qutrit"
    assert output["encoding"] == "canonical_ez"
    assert output["circuit_count"] == "9"
    assert output["shots_per_circuit"] == "256"
    assert abs(float(output["bell_unconditional"]) - 6.0) < 0.3
    assert float(output["leakage_rate"]) == 0.0
    assert Path(output["manifest"]).is_file()
