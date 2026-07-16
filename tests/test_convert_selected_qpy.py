import importlib.util
import tempfile
import unittest
from pathlib import Path

from qiskit import QuantumCircuit, qpy


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_selected_qpy.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("convert_selected_qpy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConvertSelectedQpyTests(unittest.TestCase):
    def test_converts_only_selected_candidate_directories(self):
        module = load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "output"
            selected = source_root / "monomial_full__selected"
            unselected = source_root / "monomial_full__unselected"
            selected.mkdir(parents=True)
            unselected.mkdir(parents=True)

            circuit = QuantumCircuit(1)
            circuit.x(0)
            with (selected / "circuit.qpy").open("wb") as handle:
                qpy.dump(circuit, handle)
            (selected / "E.npy").write_bytes(b"matrix-data")
            (unselected / "do_not_touch.qpy").write_bytes(b"not-qpy")

            converted = module.convert_candidates(
                source_root,
                output_root,
                ["selected"],
                target_version=13,
            )

            converted_qpy = output_root / "monomial_full__selected" / "circuit.qpy"
            self.assertEqual(converted, 1)
            self.assertTrue(converted_qpy.is_file())
            with converted_qpy.open("rb") as handle:
                self.assertEqual(qpy.get_qpy_version(handle), 13)
            self.assertEqual(
                (output_root / "monomial_full__selected" / "E.npy").read_bytes(),
                b"matrix-data",
            )
            self.assertFalse((output_root / "monomial_full__unselected").exists())


if __name__ == "__main__":
    unittest.main()
