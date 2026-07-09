from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.piast_backend import (
    AQT_SCHEDULING_METHOD,
    AQT_TRANSLATION_METHOD,
)
from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies import (
    BUILTIN_PIAST_TRANSPILER_STRATEGIES,
    get_piast_transpiler_strategy,
    piast_transpiler_strategy_names,
    run_piast_transpiler_strategy,
)


class PanicException(BaseException):
    pass


PanicException.__module__ = "pyo3_runtime"


class PiastTranspilerStrategyTests(unittest.TestCase):
    def test_strategy_registry_contains_expected_names(self):
        self.assertEqual(
            set(piast_transpiler_strategy_names()),
            {"transpile_aqt_plugin", "preset_aqt_plugin"},
        )

    def test_strategy_registry_is_immutable_to_callers(self):
        names = piast_transpiler_strategy_names()

        self.assertIsInstance(names, tuple)
        self.assertIn("transpile_aqt_plugin", BUILTIN_PIAST_TRANSPILER_STRATEGIES)
        with self.assertRaises(TypeError):
            BUILTIN_PIAST_TRANSPILER_STRATEGIES["new"] = get_piast_transpiler_strategy(
                "transpile_aqt_plugin"
            )

    def test_unknown_strategy_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown Piast transpiler strategy"):
            get_piast_transpiler_strategy("missing")

    def test_run_transpile_aqt_plugin_passes_aqt_methods(self):
        circuit = object()
        backend = object()
        transpiled = object()

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies.transpile",
            return_value=transpiled,
        ) as transpile_mock:
            result = run_piast_transpiler_strategy(
                "transpile_aqt_plugin",
                circuit,
                backend=backend,
                seed_transpiler=11,
                optimization_level=2,
            )

        self.assertTrue(result.success, result.error_message)
        self.assertIs(result.circuit, transpiled)
        transpile_mock.assert_called_once_with(
            circuit,
            backend=backend,
            optimization_level=2,
            seed_transpiler=11,
            translation_method=AQT_TRANSLATION_METHOD,
            scheduling_method=AQT_SCHEDULING_METHOD,
        )

    def test_run_preset_aqt_plugin_passes_aqt_methods(self):
        circuit = object()
        backend = object()
        transpiled = object()
        fake_pass_manager = Mock()
        fake_pass_manager.run.return_value = transpiled

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies.generate_preset_pass_manager",
            return_value=fake_pass_manager,
        ) as generate_preset_pass_manager:
            result = run_piast_transpiler_strategy(
                "preset_aqt_plugin",
                circuit,
                backend=backend,
                seed_transpiler=13,
                optimization_level=3,
            )

        self.assertTrue(result.success, result.error_message)
        self.assertIs(result.circuit, transpiled)
        generate_preset_pass_manager.assert_called_once_with(
            backend=backend,
            optimization_level=3,
            seed_transpiler=13,
            translation_method=AQT_TRANSLATION_METHOD,
            scheduling_method=AQT_SCHEDULING_METHOD,
        )
        fake_pass_manager.run.assert_called_once_with(circuit)

    def test_run_strategy_captures_invalid_seed_input(self):
        result = run_piast_transpiler_strategy(
            "transpile_aqt_plugin",
            object(),
            backend=object(),
            seed_transpiler="not-an-int",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ValueError")
        self.assertIn("invalid literal", result.error_message)
        self.assertIsNone(result.circuit)

    def test_run_strategy_captures_exception(self):
        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies.transpile",
            side_effect=RuntimeError("boom"),
        ):
            result = run_piast_transpiler_strategy(
                "transpile_aqt_plugin",
                object(),
                backend=object(),
                seed_transpiler=0,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("boom", result.error_message)
        self.assertIsNone(result.circuit)

    def test_run_strategy_captures_native_panic_exception(self):
        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies.transpile",
            side_effect=PanicException("native failure"),
        ):
            result = run_piast_transpiler_strategy(
                "transpile_aqt_plugin",
                object(),
                backend=object(),
                seed_transpiler=0,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "PanicException")
        self.assertIn("native failure", result.error_message)
        self.assertIsNone(result.circuit)

    def test_run_strategy_re_raises_process_level_failures(self):
        for failure in (
            KeyboardInterrupt("stop"),
            SystemExit("stop"),
            GeneratorExit("stop"),
            MemoryError("stop"),
        ):
            with self.subTest(failure_type=type(failure).__name__):
                with patch(
                    "qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_strategies.transpile",
                    side_effect=failure,
                ):
                    with self.assertRaises(type(failure)):
                        run_piast_transpiler_strategy(
                            "transpile_aqt_plugin",
                            object(),
                            backend=object(),
                            seed_transpiler=0,
                        )


if __name__ == "__main__":
    unittest.main()
