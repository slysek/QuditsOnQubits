from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CleanRepoSmokeTests(unittest.TestCase):
    def test_core_imports(self):
        from qudits_on_qubits import create_ame_circuit, generate_b_ame

        self.assertTrue(callable(create_ame_circuit))
        self.assertTrue(callable(generate_b_ame))

    def test_bell_measurement_imports(self):
        from qudits_on_qubits.bell_measurements import (
            build_sampler_circuits_from_graph,
            canonical_Ez,
        )

        self.assertTrue(callable(build_sampler_circuits_from_graph))
        self.assertEqual(canonical_Ez().shape, (4, 3))

    def test_artifact_layout_exists(self):
        expected_dirs = [
            "artifacts/iqm_runs/raw",
            "artifacts/iqm_runs/processed",
            "artifacts/iqm_runs/selected_best",
            "artifacts/direct_basis_runs/raw",
            "artifacts/direct_basis_runs/processed",
            "artifacts/direct_basis_runs/selected_best",
        ]
        for rel in expected_dirs:
            self.assertTrue((REPO_ROOT / rel).is_dir(), rel)

    def test_working_notebook_has_no_absolute_user_artifact_paths(self):
        nb_path = REPO_ROOT / "notebooks" / "working" / "iqm" / "meas_settings_2qutryt.ipynb"
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in data.get("cells", [])
        )

        self.assertNotIn("C:\\\\Users\\\\", source)
        self.assertIn('repo_root / "artifacts"', source)
        self.assertIn("qudits_on_qubits", source)

    def test_readme_links_to_detailed_usage_guide(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertRegex(readme, r"\[[^\]]+\]\(docs/usage_guide\.md\)")
        self.assertTrue((REPO_ROOT / "docs" / "usage_guide.md").is_file())

    def test_usage_guide_documents_piastq_managed_bell_execution(self):
        guide = (REPO_ROOT / "docs" / "usage_guide.md").read_text(encoding="utf-8")

        self.assertIn("## PiastQ managed Bell execution", guide)
        self.assertIn('mode="managed"', guide)
        self.assertIn("CFT_PIASTQ_DASHBOARD_API_URL", guide)
        self.assertIn("CFT_PIASTQ_DASHBOARD_API_KEY", guide)
        self.assertIn("Install `cft-piastq` separately", guide)
        self.assertIn("backend=client.backend", guide)
        self.assertIn("compute_bell_value_from_counts_aqt", guide)
        self.assertIn("one PiastQ job containing every generated circuit", guide)

    def test_public_usage_docs_exclude_private_urls_and_legacy_credentials(self):
        for relative_path in ("README.md", "docs/usage_guide.md"):
            with self.subTest(path=relative_path):
                source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("CFT_PIASTQ_MODE", source)
                self.assertNotIn("PCSS_TOKEN", source)
                self.assertNotIn("PCSS_QAPI_TOKEN", source)
                self.assertNotIn("git+https://", source.casefold())


if __name__ == "__main__":
    unittest.main()
