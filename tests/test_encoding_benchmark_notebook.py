import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/encoding_benchmark.ipynb"
ARCHIVE = ROOT / "examples/data/theta_demo.zip"
HELPER = ROOT / "examples/encoding_benchmark_demo.py"


@pytest.fixture
def helper(monkeypatch):
    assert HELPER.is_file(), "Missing extracted notebook helpers"
    monkeypatch.syspath_prepend(str(HELPER.parent))
    return importlib.import_module("encoding_benchmark_demo")


def test_bundle_contains_complete_original_grid():
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["config"]["theta_points"] == 41
        assert manifest["config"]["cz3_tolerance"] == 5e-4
        for index in range(41):
            prefix = f"points/{index:05d}/"
            complete = json.loads(archive.read(prefix + "complete.json"))
            for name, digest in complete["files"].items():
                assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest


def test_archive_tampering_rejected(helper, tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(ARCHIVE.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        helper.load_theta_demo(archive)


def test_loader_failure_cleans_temporary_files(helper, monkeypatch):
    directories = []
    original = helper.TemporaryDirectory
    def temporary():
        value = original()
        directories.append(Path(value.name))
        return value
    def fail(*args, **kwargs):
        raise ValueError("controlled validation failure")
    monkeypatch.setattr(helper, "TemporaryDirectory", temporary)
    monkeypatch.setattr(helper, "ThetaContinuationSynthesis", fail)
    with pytest.raises(ValueError, match="controlled"):
        helper.load_theta_demo(ARCHIVE)
    assert directories and all(not path.exists() for path in directories)


def test_archive_paths_cannot_escape(helper, tmp_path, monkeypatch):
    archive = tmp_path / "escape.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escaped.txt", "bad")
    monkeypatch.setattr(helper, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="escapes"):
        helper.load_theta_demo(archive)


class ThetaSubset:
    def __init__(self, family):
        self.family = family

    def generate(self):
        candidates = list(self.family.generate())
        return [candidates[0], candidates[40]]

    def to_dict(self):
        return {**self.family.to_dict(), "selected_indices": [0, 40]}


@pytest.mark.parametrize("fresh", [False, True])
def test_short_notebook_runs_from_clean_inputs(helper, tmp_path, monkeypatch, fresh):
    from qudits_on_qubits.benchmarks import ExactSynthesis, LocalSU2, SchmidtTheta, load_benchmark
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore

    project = tmp_path / "checkout"
    for directory in ("notebooks", "examples/data", "src/qudits_on_qubits"):
        (project / directory).mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='notebook-smoke'\n")
    shutil.copy2(ARCHIVE, project / "examples/data" / ARCHIVE.name)
    shutil.copy2(HELPER, project / "examples" / HELPER.name)
    monkeypatch.chdir(project / "notebooks")
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"__name__": "__notebook_test__"}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
            if cell["id"] == "configuration":
                namespace["output_dir"] = tmp_path / "external-results"
                if fresh:
                    namespace["families"] = [LocalSU2(samples=1), SchmidtTheta(points=2)]
                    namespace["synthesis"] = ExactSynthesis()
                else:
                    namespace["families"] = [ThetaSubset(namespace["families"][0])]
        result = namespace["result"]
        assert len(result.trials) == (12 if fresh else 9)
        assert result.trials.success.all()
        assert result.trials.fidelity.min() > 1 - 1e-6
        assert result.trials.leakage.max() < 1e-6
        assert result.trials.comparison_id.nunique() == 1
        assert namespace["summary"].candidate_name.eq("canonical").any()
        assert namespace["summary"].pareto_rank.eq(1).any()
        restored = load_benchmark(result.output_dir)
        assert restored.statistics.to_json() == result.statistics.to_json()
        if not fresh:
            store = RunStore.open(result.output_dir)
            source = RunStore.open(namespace["demo"].synthesis.source_run)
            for trial in result.trials.drop_duplicates("candidate_bundle").itertuples():
                bundle = store.read_bundle(trial.candidate_bundle)
                index = 0 if trial.candidate_name == "canonical" else bundle["metadata"]["parameters"]["grid_index"]
                original = source.read_bundle(f"points/{index:05d}")
                assert bundle["circuits"]["F3.qpy"] == original["circuits"]["f3_optimal.qpy"]
                assert bundle["circuits"]["CZ3.qpy"] == original["circuits"]["cz3_selected.qpy"]
    finally:
        if "demo" in namespace:
            directory = Path(namespace["demo"].synthesis.source_run)
            namespace["demo"].close()
            assert not directory.exists()
