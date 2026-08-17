import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO_ROOT / "notebooks" / "working" / "iqm" / "best_garnet_ghz.ipynb"
)


class NotebookPipelineContractTests(unittest.TestCase):
    def test_notebook_uses_durable_public_experiment_batch(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("from qudits_on_qubits.experiments import (", source)
        self.assertIn('state="ghz3"', source)
        self.assertIn('IQMHardware(device="garnet", use_metrics=True)', source)
        self.assertIn("BenchmarkBasis(", source)
        self.assertIn("run_experiments(SPECS)", source)
        self.assertIn("result.values", source)
        self.assertIn("result.artifact_dir", source)
        self.assertNotIn("def full_pipeline", source)
        self.assertNotIn("build_readout_calibration_matrices", source)
        self.assertNotIn("select_and_transpile_candidate", source)


class DependencyContractTests(unittest.TestCase):
    def test_iqm_qubit_selector_is_declared(self):
        for relative_path in ("pyproject.toml", "requirements.txt"):
            with self.subTest(path=relative_path):
                content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("iqm-qubit-selector>=1,<2", content)


if __name__ == "__main__":
    unittest.main()
