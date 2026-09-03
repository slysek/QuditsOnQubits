from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.pareto_selection import (
    ParetoAnalysisResult,
    write_pareto_analysis_outputs,
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

            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(
                    lambda *_args, **_kwargs: _analysis(),
                    write_pareto_analysis_outputs,
                    PHASE_DUPLICATE_COLUMNS,
                ),
            ) as dependencies:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = main(["--all-trials", str(all_trials)])

            self.assertEqual(result, 0)
            dependencies.assert_called_once_with()
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

            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(
                    lambda *_args, **_kwargs: _analysis(),
                    write_pareto_analysis_outputs,
                    PHASE_DUPLICATE_COLUMNS,
                ),
            ):
                result = main(["--all-trials", str(all_trials), "--output-dir", str(output)])

            self.assertEqual(result, 0)
            self.assertEqual(audit.read_text(encoding="utf-8"), expected)
            self.assertFalse((source / "strategy_statistics.csv").exists())
            self.assertTrue((output / "strategy_statistics.csv").exists())

    def test_absent_phase_audit_is_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(
                    lambda *_args, **_kwargs: _analysis(),
                    write_pareto_analysis_outputs,
                    PHASE_DUPLICATE_COLUMNS,
                ),
            ):
                main(["--all-trials", str(all_trials)])

            phase = pd.read_csv(directory / "candidate_global_phase_duplicates.csv")
            self.assertEqual(phase.columns.tolist(), list(PHASE_DUPLICATE_COLUMNS))
            self.assertTrue(phase.empty)

    def test_custom_weights_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            captured: dict[str, object] = {}

            def analyze(_trials, **kwargs):
                captured.update(kwargs)
                return _analysis()

            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(analyze, write_pareto_analysis_outputs, PHASE_DUPLICATE_COLUMNS),
            ):
                main([
                    "--all-trials", str(all_trials), "--two-qubit-weight", "0.1",
                    "--depth-weight", "0.9", "--std-depth-weight", "0.0",
                ])

            self.assertEqual(
                captured["objective_weights"],
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
            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(
                    lambda *_args, **_kwargs: _analysis(),
                    write_pareto_analysis_outputs,
                    PHASE_DUPLICATE_COLUMNS,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "top-level object"):
                    main(["--all-trials", str(all_trials)])

            malformed = "{not json"
            (directory / "summary.json").write_text(malformed, encoding="utf-8")
            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(
                    lambda *_args, **_kwargs: _analysis(),
                    write_pareto_analysis_outputs,
                    PHASE_DUPLICATE_COLUMNS,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "summary JSON is invalid"):
                    main(["--all-trials", str(all_trials)])
            self.assertEqual((directory / "summary.json").read_text(encoding="utf-8"), malformed)

            (directory / "summary.json").unlink()
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main(["--all-trials", str(all_trials), "--two-qubit-weight", "-1"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("objective", stderr.getvalue())

    def test_invalid_summary_preserves_sentinel_outputs_and_skips_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_directory = directory / "input"
            input_directory.mkdir()
            all_trials = self._write_trials(input_directory)
            output = directory / "output"
            output.mkdir()
            summary = output / "summary.json"
            summary.write_text("[]", encoding="utf-8")
            sentinel_paths = {
                name: output / name
                for name in (
                    "strategy_statistics.csv",
                    "pareto_ranked.csv",
                    "state_equivalence_groups.csv",
                    "recommended_circuits.csv",
                    "candidate_global_phase_duplicates.csv",
                )
            }
            sentinel_bytes = {}
            for index, path in enumerate(sentinel_paths.values()):
                value = f"sentinel-{index}\n".encode()
                path.write_bytes(value)
                sentinel_bytes[path] = value
            analysis = Mock()

            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(analysis, write_pareto_analysis_outputs, PHASE_DUPLICATE_COLUMNS),
            ):
                with self.assertRaisesRegex(ValueError, "top-level object"):
                    main(["--all-trials", str(all_trials), "--output-dir", str(output)])

            analysis.assert_not_called()
            for path, value in sentinel_bytes.items():
                self.assertEqual(path.read_bytes(), value)

    def test_invalid_summary_json_preserves_sentinel_outputs_and_skips_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_directory = directory / "input"
            input_directory.mkdir()
            all_trials = self._write_trials(input_directory)
            output = directory / "output"
            output.mkdir()
            summary = output / "summary.json"
            summary.write_text("{not json", encoding="utf-8")
            phase = output / "candidate_global_phase_duplicates.csv"
            phase_bytes = b"existing\nphase\n"
            phase.write_bytes(phase_bytes)
            analysis = Mock()

            with patch(
                "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                return_value=(analysis, write_pareto_analysis_outputs, PHASE_DUPLICATE_COLUMNS),
            ):
                with self.assertRaisesRegex(ValueError, "summary JSON is invalid"):
                    main(["--all-trials", str(all_trials), "--output-dir", str(output)])

            analysis.assert_not_called()
            self.assertEqual(phase.read_bytes(), phase_bytes)

    def test_input_collisions_are_rejected_without_modifying_source(self) -> None:
        target_names = (
            "strategy_statistics.csv",
            "pareto_ranked.csv",
            "state_equivalence_groups.csv",
            "recommended_circuits.csv",
            "candidate_global_phase_duplicates.csv",
            "summary.json",
        )
        for target_name in target_names:
            with self.subTest(target_name=target_name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                output = directory / "output"
                output.mkdir()
                source = output / target_name
                original = f"source-{target_name}\n".encode()
                source.write_bytes(original)
                analysis = Mock()
                with patch(
                    "scripts.analyze_iqm_transpiler_harness._analysis_dependencies",
                    return_value=(analysis, write_pareto_analysis_outputs, PHASE_DUPLICATE_COLUMNS),
                ):
                    with contextlib.redirect_stderr(io.StringIO()) as stderr:
                        with self.assertRaises(SystemExit) as raised:
                            main(["--all-trials", str(source), "--output-dir", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(target_name, stderr.getvalue())
                self.assertEqual(source.read_bytes(), original)
                analysis.assert_not_called()

    def test_cli_converts_weight_and_csv_schema_errors_to_argparse_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            all_trials = self._write_trials(directory)
            for option, value, message in (
                ("--two-qubit-weight", "-1", "finite nonnegative"),
                ("--depth-weight", "nan", "finite nonnegative"),
                ("--std-depth-weight", "inf", "finite nonnegative"),
            ):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        main(["--all-trials", str(all_trials), option, value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main([
                        "--all-trials", str(all_trials),
                        "--two-qubit-weight", "0", "--depth-weight", "0", "--std-depth-weight", "0",
                    ])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("positive total", stderr.getvalue())

            missing = directory / "missing-columns.csv"
            missing.write_text("state_name\nonly-state\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main(["--all-trials", str(missing)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("missing required columns", stderr.getvalue())

            malformed = directory / "malformed.csv"
            malformed.write_text('"unterminated\n', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main(["--all-trials", str(malformed)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("CSV", stderr.getvalue())

    @staticmethod
    def _isolated_script_modules(*arguments: str) -> set[str]:
        script = REPO_ROOT / "scripts" / "analyze_iqm_transpiler_harness.py"
        probe = """
import json
import runpy
import sys

sys.argv = {argv!r}
try:
    runpy.run_path({script!r}, run_name="__main__")
except SystemExit as error:
    if error.code not in (None, 0):
        raise
print("LOADED_MODULES=" + json.dumps(sorted(sys.modules)))
""".format(argv=[str(script), *arguments], script=str(script))
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        marker = "LOADED_MODULES="
        output = completed.stdout.split(marker, 1)[1]
        return set(json.loads(output))

    def test_help_isolated_process_loads_no_qiskit_or_iqm_modules(self) -> None:
        modules = self._isolated_script_modules("--help")

        self.assertFalse(any(name == "qiskit" or name.startswith("qiskit.") for name in modules))
        for suffix in ("iqm_backend", "iqm_transpiler_harness", "iqm_transpiler_strategies"):
            self.assertNotIn(f"qudits_on_qubits.benchmarks.direct_basis.{suffix}", modules)

    def test_analysis_isolated_process_loads_no_project_iqm_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            all_trials = self._write_trials(Path(temp))
            modules = self._isolated_script_modules("--all-trials", str(all_trials))

        for suffix in ("iqm_backend", "iqm_transpiler_harness", "iqm_transpiler_strategies"):
            self.assertNotIn(f"qudits_on_qubits.benchmarks.direct_basis.{suffix}", modules)

    def test_lazy_direct_basis_reflection_lists_exports_before_access(self) -> None:
        import importlib

        direct_basis = importlib.import_module("qudits_on_qubits.benchmarks.direct_basis")
        for name in direct_basis._EXPORTS:
            direct_basis.__dict__.pop(name, None)

        before_dir = set(sys.modules)
        listed = set(dir(direct_basis))
        after_dir = set(sys.modules)
        self.assertTrue(set(direct_basis.__all__).issubset(listed))
        self.assertTrue(
            set(direct_basis._EXPORTS).issubset(listed)
        )
        self.assertEqual(after_dir, before_dir)


if __name__ == "__main__":
    unittest.main()
