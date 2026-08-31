from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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


def _load_provider_module():
    spec = importlib.util.spec_from_file_location("iqm_notebook_provider", PROVIDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load IQM notebook provider")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_iqm_notebooks_do_not_use_random_hardware_error_profiles(self):
        offenders: list[str] = []

        for path in sorted(IQM_NOTEBOOK_ROOT.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", ()))
                for cell in notebook.get("cells", ())
                if cell.get("cell_type") == "code"
            )
            if "generate_random_error_profile" in code:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

        self.assertEqual(
            [],
            offenders,
            "IQM hardware notebooks must use backend calibration metrics, not "
            f"random error profiles: {offenders}",
        )

    def test_live_iqm_profile_cells_have_no_stale_random_outputs(self):
        offenders: list[str] = []

        for path in sorted(IQM_NOTEBOOK_ROOT.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            profile_loaded = False
            for index, cell in enumerate(notebook.get("cells", ())):
                source = "".join(cell.get("source", ()))
                if "error_profile = get_backend_error_profile" in source:
                    profile_loaded = True
                if not profile_loaded or cell.get("cell_type") != "code":
                    continue
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}#cell-{index}"
                    )

        self.assertEqual(
            [],
            offenders,
            "live IQM profile cells must be rerun before displaying calibration "
            f"data: {offenders}",
        )

    def test_iqm_token_cells_never_persist_outputs(self):
        offenders: list[str] = []

        for path in sorted(IQM_NOTEBOOK_ROOT.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", ())):
                source = "".join(cell.get("source", ()))
                if "IQM_TOKEN" in source and cell.get("outputs"):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}#cell-{index}"
                    )

        self.assertEqual([], offenders, f"IQM token outputs found in: {offenders}")

    def test_provider_uses_iqm_token_environment_variable(self):
        source = PROVIDER_PATH.read_text(encoding="utf-8")

        self.assertIsNotNone(
            ENVIRONMENT_TOKEN.search(source),
            "provider.py must read credentials from IQM_TOKEN",
        )

    def test_provider_does_not_silently_replace_failed_hardware_connection(self):
        provider = _load_provider_module()

        with patch.object(
            provider,
            "IQMClient",
            side_effect=RuntimeError("connection failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to connect to IQM backend"):
                provider.get_backend(quantum_computer="garnet", token="test-token")

    def test_backend_error_profile_name_records_calibration_set(self):
        provider = _load_provider_module()
        backend = SimpleNamespace(name="garnet", calibration_set_id="cal-17")

        self.assertEqual(
            "garnet@cal-17",
            provider._error_profile_name(backend, requested_name=None),
        )

    def test_depolarizing_parameter_matches_composed_gate_fidelity(self):
        from qiskit.quantum_info import average_gate_fidelity
        from qiskit_aer.noise.errors import (
            depolarizing_error,
            thermal_relaxation_error,
        )

        provider = _load_provider_module()
        target_fidelity = 0.99
        one_qubit_thermal = thermal_relaxation_error(50_000.0, 40_000.0, 40.0)
        two_qubit_thermal = one_qubit_thermal.tensor(
            thermal_relaxation_error(60_000.0, 45_000.0, 80.0)
        )

        for num_qubits, thermal_channel in (
            (1, one_qubit_thermal),
            (2, two_qubit_thermal),
        ):
            with self.subTest(num_qubits=num_qubits):
                parameter = provider._depolarizing_parameter(
                    target_fidelity,
                    thermal_channel,
                    num_qubits=num_qubits,
                )
                composed = thermal_channel.compose(
                    depolarizing_error(parameter, num_qubits)
                )
                self.assertAlmostEqual(
                    target_fidelity,
                    average_gate_fidelity(composed),
                    places=10,
                )


if __name__ == "__main__":
    unittest.main()
