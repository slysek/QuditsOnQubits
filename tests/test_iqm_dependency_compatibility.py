import importlib.metadata
import importlib.util
import inspect
import tomllib
import unittest
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


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

        for dependency in (
            "qiskit>=2,<2.2",
            "iqm-client[qiskit]>=35,<36",
            "iqm-qubit-selector>=1.1,<2",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, project_dependencies)
                self.assertIn(dependency, requirements)

        package_metadata = (
            REPO_ROOT / "src" / "qudits_on_qubits.egg-info" / "PKG-INFO"
        ).read_text(encoding="utf-8")
        installed_requirements = (
            REPO_ROOT / "src" / "qudits_on_qubits.egg-info" / "requires.txt"
        ).read_text(encoding="utf-8").splitlines()
        canonical_selector_requirement = "iqm-qubit-selector<2,>=1.1"

        self.assertIn(
            f"Requires-Dist: {canonical_selector_requirement}",
            package_metadata.splitlines(),
        )
        self.assertIn(canonical_selector_requirement, installed_requirements)

    def test_iqm_qubit_selector_public_api_matches_adapter_usage(self):
        from iqm.qubit_selector.qubit_selector import (
            CostEvaluator,
            CostFunction,
            ReadoutMode,
        )

        evaluator_parameters = inspect.signature(CostEvaluator).parameters
        for parameter in (
            "backend",
            "quantum_circuit",
            "cost_function",
            "readoutmode",
            "remove_qubits",
            "num_trials",
        ):
            with self.subTest(api="CostEvaluator", parameter=parameter):
                self.assertIn(parameter, evaluator_parameters)

        top_layout_parameters = inspect.signature(
            CostEvaluator.get_top_layouts
        ).parameters
        self.assertIn("num_layouts", top_layout_parameters)

        for member in ("GATE_COST_CZ", "GATE_COST_CLIFFORD"):
            with self.subTest(api="CostFunction", member=member):
                self.assertTrue(hasattr(CostFunction, member))
        for member in ("NONE", "FIDELITY", "QNDNESS"):
            with self.subTest(api="ReadoutMode", member=member):
                self.assertTrue(hasattr(ReadoutMode, member))

    def test_active_iqm_distributions_match_supported_windows_and_modules(self):
        expectations = (
            (
                "iqm-qubit-selector",
                SpecifierSet(">=1.1,<2"),
                "iqm.qubit_selector.qubit_selector",
            ),
            ("iqm-client", SpecifierSet(">=35,<36"), "iqm.qiskit_iqm"),
        )

        for distribution_name, supported_versions, module_name in expectations:
            with self.subTest(distribution=distribution_name, module=module_name):
                distribution = importlib.metadata.distribution(distribution_name)
                self.assertIn(Version(distribution.version), supported_versions)

                distribution_root = Path(distribution.locate_file("")).resolve()
                module_spec = importlib.util.find_spec(module_name)
                self.assertIsNotNone(module_spec)
                self.assertIsNotNone(module_spec.origin)
                module_origin = Path(module_spec.origin).resolve()
                self.assertTrue(
                    module_origin.is_relative_to(distribution_root),
                    f"{module_name} resolved outside {distribution_name}: "
                    f"{module_origin} is not inside {distribution_root}",
                )

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
