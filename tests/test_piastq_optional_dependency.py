import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PiastQOptionalDependencyTests(unittest.TestCase):
    def test_piastq_optional_dependency_is_declared(self):
        pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        expected_block = """[project.optional-dependencies]
piastq = [
    "cft-piastq[direct]>=0.1,<0.2",
]
"""

        self.assertIn(expected_block, pyproject_text)


if __name__ == "__main__":
    unittest.main()
