import unittest
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


class PiastQOptionalDependencyTests(unittest.TestCase):
    def test_piastq_extra_does_not_reference_private_managed_client(self):
        pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        pyproject = tomllib.loads(pyproject_text)
        requirements_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertNotIn("piastq", pyproject["project"]["optional-dependencies"])
        self.assertNotIn("git+https://", pyproject_text.casefold())
        self.assertNotIn("cft-piastq", requirements_text.casefold())

    def test_piastq_optional_dependency_excludes_direct_provider_stack(self):
        pyproject = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = "\n".join(
            dependency
            for group in pyproject["project"]["optional-dependencies"].values()
            for dependency in group
        ).casefold()

        self.assertNotIn("[direct]", dependencies)
        self.assertNotIn("pcss-qapi", dependencies)
        self.assertNotIn("qiskit-aqt-provider", dependencies)

    def test_tracked_metadata_does_not_reference_private_managed_client(self):
        metadata_paths = (
            REPO_ROOT / "src" / "qudits_on_qubits.egg-info" / "PKG-INFO",
            REPO_ROOT / "src" / "qudits_on_qubits.egg-info" / "requires.txt",
        )

        for metadata_path in metadata_paths:
            metadata = metadata_path.read_text(encoding="utf-8").casefold()
            self.assertNotIn("cft-piastq", metadata)
            self.assertNotIn("git+https://", metadata)

if __name__ == "__main__":    unittest.main()
