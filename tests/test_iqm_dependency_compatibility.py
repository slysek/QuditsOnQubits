import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class IQMDependencyCompatibilityTests(unittest.TestCase):
    def test_base_dependencies_match_iqm_os_4_6_qiskit_window(self):
        pyproject = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        project_dependencies = pyproject["project"]["dependencies"]
        requirements = {
            line.strip()
            for line in (REPO_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        for dependency in ("qiskit>=2,<2.2", "iqm-client[qiskit]>=35,<36"):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, project_dependencies)
                self.assertIn(dependency, requirements)

    def test_mitigation_extra_uses_rem_compatible_versions(self):
        pyproject = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        mitigation_dependencies = pyproject["project"]["optional-dependencies"][
            "mitigation"
        ]

        self.assertIn("mthree>=3,<4", mitigation_dependencies)
        self.assertIn(
            "iqm-error-reduction-tools>=0.2,<0.3", mitigation_dependencies
        )


if __name__ == "__main__":
    unittest.main()
