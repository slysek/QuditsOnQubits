import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
class DependencyContractTests(unittest.TestCase):
    def test_iqm_qubit_selector_is_declared_only_for_supported_python(self):
        dependency = "iqm-qubit-selector>=1,<2; python_version >= '3.11'"
        for relative_path in ("pyproject.toml", "requirements.txt"):
            with self.subTest(path=relative_path):
                content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(dependency, content)


if __name__ == "__main__":
    unittest.main()
