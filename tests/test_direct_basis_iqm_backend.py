from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    IqmEnvironment,
    backend_metadata,
    build_iqm_pass_manager,
    load_iqm_environment,
    safe_backend_slug,
)


def _fake_garnet():
    try:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    except ImportError as exc:
        raise unittest.SkipTest(f"IQM fake backend is unavailable: {exc}") from exc
    return IQMFakeGarnet()


class IqmBackendAdapterTests(unittest.TestCase):
    def test_safe_backend_slug(self):
        self.assertEqual(safe_backend_slug("garnet"), "garnet")
        self.assertEqual(safe_backend_slug("IQM Garnet 20"), "IQM_Garnet_20")
        self.assertEqual(safe_backend_slug("../bad name"), "bad_name")

    def test_load_iqm_environment_requires_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            with self.assertRaisesRegex(RuntimeError, "Missing required IQM env file"):
                load_iqm_environment(env_path)

    def test_load_iqm_environment_requires_both_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("IQM_SERVER_URL=https://example.invalid/\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "IQM_TOKEN"):
                load_iqm_environment(env_path)

    def test_load_iqm_environment_reads_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "IQM_SERVER_URL=https://example.invalid/\n"
                "IQM_TOKEN=secret-token\n",
                encoding="utf-8",
            )
            old_url = os.environ.get("IQM_SERVER_URL")
            old_token = os.environ.get("IQM_TOKEN")
            try:
                loaded = load_iqm_environment(env_path)
                self.assertEqual(
                    loaded,
                    IqmEnvironment(
                        server_url="https://example.invalid/",
                        token="secret-token",
                    ),
                )
                self.assertEqual(os.environ["IQM_SERVER_URL"], "https://example.invalid/")
                self.assertEqual(os.environ["IQM_TOKEN"], "secret-token")
            finally:
                if old_url is None:
                    os.environ.pop("IQM_SERVER_URL", None)
                else:
                    os.environ["IQM_SERVER_URL"] = old_url
                if old_token is None:
                    os.environ.pop("IQM_TOKEN", None)
                else:
                    os.environ["IQM_TOKEN"] = old_token

    def test_backend_metadata_for_fake_iqm_backend(self):
        backend = _fake_garnet()
        metadata = backend_metadata(
            backend,
            iqm_backend_name="garnet",
            iqm_use_metrics=True,
            optimization_level=3,
            layout_method=None,
            routing_method="sabre",
        )
        self.assertEqual(metadata["transpiler_backend"], "iqm")
        self.assertEqual(metadata["iqm_backend_name"], "garnet")
        self.assertEqual(metadata["iqm_use_metrics"], True)
        self.assertEqual(metadata["optimization_level"], 3)
        self.assertIsNone(metadata["layout_method"])
        self.assertEqual(metadata["routing_method"], "sabre")
        self.assertGreaterEqual(metadata["backend_num_qubits"], 1)
        self.assertIn("r", json.loads(metadata["backend_operation_names"]))
        self.assertIn("cz", json.loads(metadata["backend_operation_names"]))
        self.assertGreater(metadata["backend_coupling_map_size"], 0)

    def test_build_iqm_pass_manager_uses_backend_plugins(self):
        backend = _fake_garnet()
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        pass_manager = build_iqm_pass_manager(
            backend,
            optimization_level=3,
            seed_transpiler=0,
            layout_method=None,
            routing_method=None,
        )
        transpiled = pass_manager.run(circuit)
        ops = transpiled.count_ops()
        self.assertIn("r", ops)
        self.assertIn("cz", ops)
        self.assertNotIn("cx", ops)


if __name__ == "__main__":
    unittest.main()
