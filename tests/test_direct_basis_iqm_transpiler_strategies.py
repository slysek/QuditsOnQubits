from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    EXACT_RZ_SCHEDULING_METHOD,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    BUILTIN_IQM_TRANSPILER_STRATEGIES,
    get_iqm_transpiler_strategy,
    iqm_transpiler_strategy_names,
    run_iqm_transpiler_strategy,
)


def _fake_garnet():
    try:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    except ImportError as exc:
        raise unittest.SkipTest(f"IQM fake backend is unavailable: {exc}") from exc
    return IQMFakeGarnet()


class IqmTranspilerStrategyTests(unittest.TestCase):
    def test_strategy_registry_contains_expected_names(self):
        self.assertEqual(
            set(iqm_transpiler_strategy_names()),
            {
                "preset_default",
                "preset_exact",
                "transpile_to_iqm_default",
                "transpile_to_iqm_exact",
            },
        )

    def test_preset_exact_records_exact_scheduling_method(self):
        strategy = get_iqm_transpiler_strategy("preset_exact")

        self.assertEqual(strategy.scheduling_method, EXACT_RZ_SCHEDULING_METHOD)
        self.assertIs(strategy.remove_final_rzs, False)

    def test_unknown_strategy_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown IQM transpiler strategy"):
            get_iqm_transpiler_strategy("missing")

    def test_builtin_strategy_mapping_is_immutable_to_callers(self):
        names = iqm_transpiler_strategy_names()

        self.assertIsInstance(names, tuple)
        self.assertIn("preset_default", BUILTIN_IQM_TRANSPILER_STRATEGIES)

    def test_run_preset_default_strategy_with_fake_backend(self):
        backend = _fake_garnet()
        circuit = build_direct_basis_graph_state_circuit(
            "two_qutrit",
            np.eye(3, dtype=complex),
            n_qutrits=2,
        )

        result = run_iqm_transpiler_strategy(
            "preset_default",
            circuit,
            backend=backend,
            seed_transpiler=0,
        )

        self.assertTrue(result.success, result.error_message)
        self.assertIsNotNone(result.circuit)
        ops = result.circuit.count_ops()
        self.assertIn("r", ops)
        self.assertIn("cz", ops)
        self.assertGreater(result.compile_time_seconds, 0.0)

    def test_run_strategy_captures_exception(self):
        circuit = build_direct_basis_graph_state_circuit(
            "two_qutrit",
            np.eye(3, dtype=complex),
            n_qutrits=2,
        )

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies.generate_preset_pass_manager",
            side_effect=RuntimeError("boom"),
        ):
            result = run_iqm_transpiler_strategy(
                "preset_default",
                circuit,
                backend=object(),
                seed_transpiler=0,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("boom", result.error_message)
        self.assertIsNone(result.circuit)


if __name__ == "__main__":
    unittest.main()
