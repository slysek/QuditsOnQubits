from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.piast_backend import (
    AQT_SCHEDULING_METHOD,
    AQT_TRANSLATION_METHOD,
    PIAST_BACKEND_NAME,
    PiastEnvironment,
    backend_metadata,
    default_env_path,
    load_piast_backend,
    load_piast_environment,
    safe_backend_slug,
)


class PiastBackendAdapterTests(unittest.TestCase):
    def test_safe_backend_slug(self):
        self.assertEqual(safe_backend_slug("piast"), "piast")
        self.assertEqual(safe_backend_slug("AQT Piast 20"), "AQT_Piast_20")
        self.assertEqual(safe_backend_slug("../bad name"), "bad_name")

    def test_default_env_path_points_to_repo_env(self):
        self.assertEqual(default_env_path(), REPO_ROOT / ".env")

    def test_load_piast_environment_requires_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            with self.assertRaisesRegex(RuntimeError, "Missing required Piast env file"):
                load_piast_environment(env_path)

    def test_load_piast_environment_requires_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("IQM_TOKEN=not-used-here\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "PCSS_QAPI_TOKEN"):
                load_piast_environment(env_path)

    def test_load_piast_environment_reads_token_and_loads_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PCSS_QAPI_TOKEN=pcss-secret\n", encoding="utf-8")
            old_token = os.environ.get("PCSS_QAPI_TOKEN")
            try:
                loaded = load_piast_environment(env_path)
                self.assertEqual(loaded, PiastEnvironment(token="pcss-secret"))
                self.assertEqual(os.environ["PCSS_QAPI_TOKEN"], "pcss-secret")
            finally:
                if old_token is None:
                    os.environ.pop("PCSS_QAPI_TOKEN", None)
                else:
                    os.environ["PCSS_QAPI_TOKEN"] = old_token

    def test_load_piast_backend_logs_in_and_returns_direct_access_backend(self):
        class FakeProvider:
            def __init__(self):
                self.created = True

            def get_direct_access_backend(self):
                return {"backend": "piast-direct-access"}

        fake_auth = types.SimpleNamespace(login=Mock())
        fake_provider_module = types.SimpleNamespace(PCSS_AQTProvider=FakeProvider)

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PCSS_QAPI_TOKEN=secret-token\n", encoding="utf-8")
            with patch.dict(
                sys.modules,
                {
                    "pcss_qapi": types.SimpleNamespace(AuthorizationService=fake_auth),
                    "pcss_qapi.aqt.provider": fake_provider_module,
                },
            ):
                backend = load_piast_backend(env_path)

        self.assertEqual(backend, {"backend": "piast-direct-access"})
        fake_auth.login.assert_called_once_with("secret-token")

    def test_load_piast_backend_reports_missing_pcss_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PCSS_QAPI_TOKEN=secret-token\n", encoding="utf-8")
            with patch.dict(sys.modules, {"pcss_qapi": None}):
                with self.assertRaisesRegex(RuntimeError, "Missing PCSS QAPI package"):
                    load_piast_backend(env_path)

    def test_backend_metadata_for_aqt_like_backend(self):
        target = types.SimpleNamespace(operation_names={"r", "rz", "rxx", "measure"}, num_qubits=20)
        backend = types.SimpleNamespace(
            name="direct-access",
            num_qubits=20,
            target=target,
            coupling_map=None,
        )

        metadata = backend_metadata(backend, optimization_level=3)

        self.assertEqual(metadata["transpiler_backend"], "piast")
        self.assertEqual(metadata["piast_backend_name"], PIAST_BACKEND_NAME)
        self.assertEqual(metadata["translation_method"], AQT_TRANSLATION_METHOD)
        self.assertEqual(metadata["scheduling_method"], AQT_SCHEDULING_METHOD)
        self.assertEqual(metadata["optimization_level"], 3)
        self.assertEqual(metadata["backend_num_qubits"], 20)
        self.assertEqual(json.loads(metadata["backend_operation_names"]), ["measure", "r", "rxx", "rz"])
        self.assertEqual(metadata["backend_coupling_map_size"], 0)
        self.assertEqual(metadata["pcss_provider"], "PCSS_AQTProvider")


if __name__ == "__main__":
    unittest.main()
