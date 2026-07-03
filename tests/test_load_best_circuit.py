from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.load_best_circuit import load_selected_circuit, resolve_selected_candidate_dir


class LoadBestCircuitTests(unittest.TestCase):
    def test_load_selected_circuit_defaults_to_transpiled_and_requires_e(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            selected = (
                repo
                / "artifacts"
                / "direct_basis_runs"
                / "selected_best"
                / "ghz3"
                / "20260703_153000"
                / "fid099"
                / "rank01_product__p001"
            )
            selected.mkdir(parents=True)
            raw = QuantumCircuit(2, name="raw")
            transpiled = QuantumCircuit(2, name="transpiled")
            with (selected / "graph_state_direct_basis.qpy").open("wb") as handle:
                qpy.dump(raw, handle)
            with (selected / "graph_state_direct_basis_transpiled.qpy").open("wb") as handle:
                qpy.dump(transpiled, handle)
            np.save(selected / "E.npy", np.eye(4, 3, dtype=complex))

            directory = resolve_selected_candidate_dir(
                "direct_basis_runs",
                "ghz3",
                "20260703_153000",
                "fid099",
                rank=1,
                repo_root=repo,
            )
            self.assertEqual(directory, selected)

            circuit, E = load_selected_circuit(
                "direct_basis_runs",
                "ghz3",
                "20260703_153000",
                "fid099",
                rank=1,
                repo_root=repo,
            )
            self.assertEqual(circuit.name, "transpiled")
            self.assertEqual(E.shape, (4, 3))

            raw_loaded, _ = load_selected_circuit(
                "direct_basis_runs",
                "ghz3",
                "20260703_153000",
                "fid099",
                rank=1,
                circuit_kind="raw",
                repo_root=repo,
            )
            self.assertEqual(raw_loaded.name, "raw")

    def test_missing_e_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            selected = (
                repo
                / "artifacts"
                / "direct_basis_runs"
                / "selected_best"
                / "ghz3"
                / "20260703_153000"
                / "exact"
                / "rank01_product__p001"
            )
            selected.mkdir(parents=True)
            circuit = QuantumCircuit(2)
            with (selected / "graph_state_direct_basis_transpiled.qpy").open("wb") as handle:
                qpy.dump(circuit, handle)

            with self.assertRaisesRegex(FileNotFoundError, "E.npy"):
                load_selected_circuit(
                    "direct_basis_runs",
                    "ghz3",
                    "20260703_153000",
                    "exact",
                    rank=1,
                    repo_root=repo,
                )


if __name__ == "__main__":
    unittest.main()
