import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

class DependencyContractTests(unittest.TestCase):
    def test_supported_python_range_starts_at_3_11(self):
        content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.11,<3.14"', content)

    def test_iqm_qubit_selector_is_declared(self):
        dependency = "iqm-qubit-selector>=1.1,<2"
        for relative_path in ("pyproject.toml", "requirements.txt"):
            with self.subTest(path=relative_path):
                content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                declarations = [
                    line.strip().strip('",')
                    for line in content.splitlines()
                    if "iqm-qubit-selector" in line
                ]
                self.assertEqual(declarations, [dependency])


if __name__ == "__main__":
    unittest.main()
