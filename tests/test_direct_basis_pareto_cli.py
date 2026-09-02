from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.pareto_selection import (
    ParetoAnalysisResult,
)
from qudits_on_qubits.benchmarks.direct_basis.phase_equivalence import (
    PHASE_DUPLICATE_COLUMNS,
)
from scripts.analyze_iqm_transpiler_harness import build_parser, main


def _trial_rows() -> list[dict[str, object]]:
    return [
        {
            "state_name": "ghz3",
            "class_name": "test",
            "candidate_name": candidate,
            "strategy_name": candidate,
            "seed_transpiler": seed,
            "success": True,
            "status": "ok",
            "depth": depth,
            "size": depth + 1,
            "one_qubit_gate_count": 2,
            "two_qubit_gate_count": two_qubit,
            "graph_state_transpiled_qpy": "missing.qpy",
        }
        for candidate, depth, two_qubit in (("low_2q", 10, 2), ("low_depth", 5, 4))
        for seed in (1, 2)
    ]


def _analysis() -> ParetoAnalysisResult:
    return ParetoAnalysisResult(
        strategy_statistics=pd.DataFrame({"strategy": ["a"]}),
        pareto_ranked=pd.DataFrame({"candidate": ["a"]}),
        state_equivalence_groups=pd.DataFrame({"group": ["one"]}),
        recommended_circuits=pd.DataFrame({"candidate": ["a"]}),
        summary_counts={"recommended_circuit_count": 1},
    )


class ParetoPostProcessingCliTests(unittest.TestCase):
    def _write_trials(self, directory: Path) -> Path:
        path = directory / "all_trials.csv"
        pd.DataFrame(_trial_rows()).to_csv(path, index=False)
        return path

    def test_parser_defaults_are_exact(self) -> None:
        args = build_parser().parse_args(["--all-trials", "all_trials.csv"])

        self.assertEqual(args.all_trials, "all_trials.csv")
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.two_qubit_weight, 0.50)
        self.assertEqual(args.depth_weight, 0.30)
        self.assertEqual(args.std_depth_weight, 0.20)
        self.assertEqual(args.max_state_qubits, 12)

    def test_main_defaults_output_to_all_trials_parent_and_writes_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            (directory / "summary.json").write_text('{"candidate_count": 2}', encoding="utf-8")

            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()) as analyze:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = main(["--all-trials", str(all_trials)])

            self.assertEqual(result, 0)
            analyze.assert_called_once_with(
                unittest.mock.ANY,
                objective_weights={
                    "mean_two_qubit_gate_count": 0.50,
                    "mean_depth": 0.30,
                    "std_depth": 0.20,
                },
                max_state_qubits=12,
            )
            for name in (
                "candidate_global_phase_duplicates.csv",
                "strategy_statistics.csv",
                "pareto_ranked.csv",
                "state_equivalence_groups.csv",
                "recommended_circuits.csv",
            ):
                self.assertTrue((directory / name).is_file())
            self.assertEqual(
                json.loads((directory / "summary.json").read_text(encoding="utf-8")),
                {"candidate_count": 2, "recommended_circuit_count": 1},
            )
            self.assertIn(
                f"Candidate global phase duplicates CSV: {directory / 'candidate_global_phase_duplicates.csv'}",
                stdout.getvalue(),
            )

    def test_main_honors_output_dir_and_preserves_existing_phase_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "output"
            source.mkdir()
            output.mkdir()
            all_trials = self._write_trials(source)
            audit = output / "candidate_global_phase_duplicates.csv"
            expected = "known,audit\\nrow,value\\n"
            audit.write_text(expected, encoding="utf-8")

            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()):
                result = main(["--all-trials", str(all_trials), "--output-dir", str(output)])

            self.assertEqual(result, 0)
            self.assertEqual(audit.read_text(encoding="utf-8"), expected)
            self.assertFalse((source / "strategy_statistics.csv").exists())
            self.assertTrue((output / "strategy_statistics.csv").exists())

    def test_absent_phase_audit_is_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()):
                main(["--all-trials", str(all_trials)])

            phase = pd.read_csv(directory / "candidate_global_phase_duplicates.csv")
            self.assertEqual(phase.columns.tolist(), list(PHASE_DUPLICATE_COLUMNS))
            self.assertTrue(phase.empty)

    def test_custom_weights_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()) as analyze:
                main([
                    "--all-trials", str(all_trials), "--two-qubit-weight", "0.1",
                    "--depth-weight", "0.9", "--std-depth-weight", "0.0",
                ])

            self.assertEqual(
                analyze.call_args.kwargs["objective_weights"],
                {"mean_two_qubit_gate_count": 0.1, "mean_depth": 0.9, "std_depth": 0.0},
            )

    def test_custom_weights_change_written_pareto_recommendation_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            original = all_trials.read_bytes()

            main(["--all-trials", str(all_trials), "--output-dir", str(directory / "default")])
            main([
                "--all-trials", str(all_trials), "--output-dir", str(directory / "depth"),
                "--two-qubit-weight", "0.1", "--depth-weight", "0.9", "--std-depth-weight", "0.0",
            ])

            default = pd.read_csv(directory / "default" / "pareto_ranked.csv")
            depth = pd.read_csv(directory / "depth" / "pareto_ranked.csv")
            self.assertEqual(default.iloc[0]["candidate_name"], "low_2q")
            self.assertEqual(depth.iloc[0]["candidate_name"], "low_depth")
            self.assertEqual(all_trials.read_bytes(), original)

    def test_rejects_invalid_input_and_summary_without_running_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            invalid_cases = [
                (["--all-trials", str(directory / "missing.csv")], "does not exist"),
                (["--all-trials", str(all_trials), "--max-state-qubits", "0"], "max-state-qubits"),
            ]
            for argv, message in invalid_cases:
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        main(argv)

            (directory / "summary.json").write_text("[]", encoding="utf-8")
            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()):
                with self.assertRaisesRegex(ValueError, "top-level object"):
                    main(["--all-trials", str(all_trials)])

            malformed = "{not json"
            (directory / "summary.json").write_text(malformed, encoding="utf-8")
            with patch("scripts.analyze_iqm_transpiler_harness.analyze_iqm_trials", return_value=_analysis()):
                with self.assertRaisesRegex(ValueError, "summary JSON is invalid"):
                    main(["--all-trials", str(all_trials)])
            self.assertEqual((directory / "summary.json").read_text(encoding="utf-8"), malformed)

            with self.assertRaisesRegex(ValueError, "objective_weights"):
                main(["--all-trials", str(all_trials), "--two-qubit-weight", "-1"])

    def test_has_no_harness_or_backend_imports_or_calls(self) -> None:
        source = (REPO_ROOT / "scripts" / "analyze_iqm_transpiler_harness.py").read_text(encoding="utf-8")
        self.assertNotIn("iqm_transpiler_harness", source)
        self.assertNotIn("iqm_backend", source)


if __name__ == "__main__":
    unittest.main()
