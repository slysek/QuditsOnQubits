from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IQM_NOTEBOOK_ROOT = REPO_ROOT / "notebooks" / "working" / "iqm"
PROVIDER_PATH = IQM_NOTEBOOK_ROOT / "provider.py"
HARDCODED_TOKEN = re.compile(
    r"(?:(?:\\?[\"']token\\?[\"']|token)\s*[:=]|token\s*:\s*str\s*=)"
    r"\s*\(?\s*\\?[\"'][^\"'\r\n]{32,}\\?[\"']\s*\)?",
    re.IGNORECASE,
)
ENVIRONMENT_TOKEN = re.compile(
    r"os\.environ(?:\.get)?\s*(?:\(|\[)\s*[\"']IQM_TOKEN[\"']",
)


class IqmNotebookSecretTests(unittest.TestCase):
    def test_hardcoded_token_detector_covers_supported_forms(self):
        value = "x" * 32
        samples = (
            f'token = "{value}"',
            f'token: str = "{value}"',
            f'token = ("{value}")',
            f'{{"token": "{value}"}}',
            json.dumps(f'config = {{"token": "{value}"}}'),
            f'token=\\"{value}\\"',
        )

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertIsNotNone(HARDCODED_TOKEN.search(sample))

    def test_iqm_notebooks_are_valid_json(self):
        for path in sorted(IQM_NOTEBOOK_ROOT.glob("*.ipynb")):
            with self.subTest(path=path.name):
                json.loads(path.read_text(encoding="utf-8"))

    def test_iqm_notebooks_contain_no_hardcoded_tokens(self):
        offenders: list[str] = []

        for path in sorted(IQM_NOTEBOOK_ROOT.rglob("*")):
            if path.suffix not in {".py", ".ipynb"}:
                continue
            source = path.read_text(encoding="utf-8")
            if HARDCODED_TOKEN.search(source):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

        self.assertEqual([], offenders, f"hardcoded tokens found in: {offenders}")

    def test_provider_uses_iqm_token_environment_variable(self):
        source = PROVIDER_PATH.read_text(encoding="utf-8")

        self.assertIsNotNone(
            ENVIRONMENT_TOKEN.search(source),
            "provider.py must read credentials from IQM_TOKEN",
        )


if __name__ == "__main__":
    unittest.main()
