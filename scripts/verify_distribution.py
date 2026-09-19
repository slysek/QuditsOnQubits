"""Verify distributable contents and installed offline examples without a checkout import."""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

GATES = {f"{gate}gate{dimension}{suffix}.qpy"
         for dimension in (3, 4) for gate in "XZ" for suffix in ("", "dag")}
GATES |= {"Fgate3.qpy", "Fgate4.qpy", "CZgate3.qpy", "CZgate4cor.qpy"}
THETA_DATA = {f"qudits_on_qubits/benchmarks/theta_continuation/data/{name}"
              for name in ("CZ3_W.qpy", "E.npy")}
SOURCE_FILES = {
    "LICENSE", "CITATION.cff", "CONTRIBUTING.md", "README.md", "pyproject.toml",
    "CHANGELOG.md", "SECURITY.md", "CODE_OF_CONDUCT.md",
    "examples/data/theta_demo.zip", "examples/encoding_benchmark_demo.py",
    "examples/two_qutrit_encoding_benchmark.py", "notebooks/encoding_benchmark.ipynb",
    "docs/README.md", "docs/encoding_benchmark.md", "docs/assets/encoding_benchmark.png",
    ".github/workflows/ci.yml",
    ".github/PULL_REQUEST_TEMPLATE.md", ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/bug_report.md", ".github/ISSUE_TEMPLATE/feature_request.md",
    "tests/test_encoding_benchmark_notebook.py", "scripts/verify_distribution.py",
    "scripts/iqm_bell_short.py", "scripts/iqm_randomized_bell_campaign.py",
    "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy",
}


def require_members(names, expected, label):
    missing = expected - set(names)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}")


def verify_archives(wheel, sdist):
    """Fail before execution when runtime gates or reproducibility inputs are absent."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require_members(names, {f"qudits_on_qubits/quantum_circuits/{n}" for n in GATES} | THETA_DATA, "wheel")
        if not any(".dist-info/" in n and n.endswith("/LICENSE") for n in names):
            raise ValueError("wheel is missing LICENSE metadata")
        entry_files = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        if len(entry_files) != 1:
            raise ValueError("wheel must contain one entry-point table")
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry_files[0]).decode("utf-8"))
        require_members(parser["console_scripts"], {"qoq-benchmark", "qoq-two-qutrit-bell"}, "entry points")
    with tarfile.open(sdist, "r:gz") as archive:
        names = {n.partition("/")[2] for n in archive.getnames()}
        require_members(names, SOURCE_FILES, "sdist")


SMOKE = r"""
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path
import sys
import qudits_on_qubits
from qiskit import qpy
from qudits_on_qubits import load_run_manifest
from qudits_on_qubits.benchmarks import load_benchmark
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template

theta_config = ThetaBenchmarkConfig()
theta_template = load_cz3_template(theta_config.baseline_qpy)
assert theta_template.baseline_sha256 == theta_config.baseline_sha256
assert theta_template.original_circuit.num_qubits == 4

source = Path(sys.argv[1]).resolve() / "src"
module = Path(qudits_on_qubits.__file__).resolve()
assert not module.is_relative_to(source), f"Imported checkout instead of installation: {module}"
print(f"installed_package={module}")
# Resolve runtime defaults from the installed package, outside any checkout.
from unittest.mock import patch
from types import SimpleNamespace
import os
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qudits_on_qubits.benchmarks import BackendTarget, OptimizedSynthesis
from qudits_on_qubits.benchmarks.direct_basis import iqm_backend

with patch.object(Path, "home", return_value=Path.cwd() / "user"):
    assert OptimizedSynthesis().cache_dir.is_relative_to(Path.cwd() / "user")
settings = Path(".env")
settings.write_text("IQM_SERVER_URL=https://example.invalid\nIQM_TOKEN=offline-test\n", encoding="utf-8")
provider = SimpleNamespace(get_backend=lambda **options: IQMFakeGarnet())
with patch.dict(os.environ), patch("iqm.qiskit_iqm.IQMProvider", return_value=provider), patch.object(
    iqm_backend, "default_env_path", side_effect=AssertionError("installation-relative credentials")
):
    assert BackendTarget.iqm("garnet").provider == "iqm"
settings.unlink()
gates = list(files("qudits_on_qubits").joinpath("quantum_circuits").iterdir())
gates = [path for path in gates if path.name.endswith(".qpy")]
assert len(gates) == 12
for path in gates:
    with path.open("rb") as stream:
        assert len(qpy.load(stream)) == 1

commands = [
    ("qoq-benchmark", ["--backend", "local", "--family", "local-su2", "--family",
     "schmidt-theta", "--local-samples", "1", "--theta-points", "2", "--synthesis",
     "exact", "--seeds", "0", "1", "--output-dir", "encoding"]),
    ("qoq-two-qutrit-bell", ["--shots", "2048", "--seed", "42", "--output-root", "bell"]),
]
installed = entry_points(group="console_scripts")
for name, arguments in commands:
    sys.argv = [name, *arguments]
    try:
        installed[name].load()()
    except SystemExit as error:
        assert error.code in (0, None), (name, error.code)
result = load_benchmark("encoding")
assert len(result.trials) == 8 and result.trials.success.all()
assert result.trials.fidelity.min() >= 1 - 1e-6
assert result.trials.leakage.max() <= 1e-6
assert not result.pareto_front.empty
assert result.statistics.is_baseline.sum() == 1
manifests = list(Path("bell").rglob("run-manifest.json"))
assert len(manifests) == 1
manifest = load_run_manifest(manifests[0].parent)
assert manifest.status == "completed"
assert manifest.result["circuit_count"] == 9
assert manifest.result["leakage_rate"] == 0
assert abs(manifest.result["bell_unconditional"]["real"] - 6) < .15
print("distribution_smoke=passed")
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    args = parser.parse_args(argv)
    verify_archives(args.wheel, args.sdist)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qoq-distribution-") as directory:
        subprocess.run([sys.executable, "-I", "-c", SMOKE, str(root)],
                       cwd=directory, check=True, timeout=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
