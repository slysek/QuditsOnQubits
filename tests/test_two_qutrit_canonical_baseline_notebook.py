import ast
import json
import os
import re
import stat
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from qiskit.quantum_info import Operator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "two_qutrit_bell_canonical_baseline.ipynb"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def named_calls(cell_source, name):
    tree = ast.parse(cell_source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def runner_backend(cell_source, call):
    tree = ast.parse(cell_source)
    specs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "ExperimentSpec":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    specs[target.id] = node.value
    value = next((keyword.value for keyword in call.keywords if keyword.arg == "spec"), call.args[0] if call.args else None)
    if isinstance(value, ast.Name):
        value = specs.get(value.id)
    assert isinstance(value, ast.Call)
    backend = next(keyword.value for keyword in value.keywords if keyword.arg == "backend")
    assert isinstance(backend, ast.Call) and isinstance(backend.func, ast.Name)
    return backend.func.id


def test_notebook_has_three_separate_high_level_backend_runs():
    notebook = load_notebook()
    cells = code_cells(notebook)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 3
    assert all(len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells)
    tree = ast.parse("\n".join(source(cell) for cell in cells))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "qudits_on_qubits.experiments" for alias in node.names}
    assert {"AerIdeal", "BootstrapConfig", "ExperimentSpec", "IQMHardware", "MitigationConfig", "PathBasis", "PiastQHardware", "run_experiment"} <= imported
    expected = ["AerIdeal", "IQMHardware", "PiastQHardware"]
    assert [runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0]) for cell in run_cells] == expected
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)
    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo) == 1 and isinstance(repo[0].value, ast.Name) and repo[0].value.id == "REPO_ROOT"


def test_hardware_cells_are_false_by_default_and_separately_guarded():
    notebook = load_notebook()
    cells = code_cells(notebook)
    full_source = "\n".join(source(cell) for cell in cells)
    assert re.search(r"^RUN_IQM\s*=\s*False\s*$", full_source, re.MULTILINE)
    assert re.search(r"^RUN_PIASTQ\s*=\s*False\s*$", full_source, re.MULTILINE)
    runner_cells = [cell for cell in cells if len(named_calls(source(cell), "run_experiment")) == 1]
    for backend, flag in (("IQMHardware", "RUN_IQM"), ("PiastQHardware", "RUN_PIASTQ")):
        cell = next(cell for cell in runner_cells if runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0]) == backend)
        tree = ast.parse(source(cell))
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == flag]
        runner = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_experiment"]
        assert guards and len(runner) == 1
        assert any(runner[0] in ast.walk(statement) for guard in guards for statement in guard.body)
        assert runner_backend(source(cell), runner[0]) == backend


def test_canonical_baseline_configuration_and_empty_summary_are_semantically_complete():
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["SHOTS"] == 100
    assert namespace["UNCERTAINTY"].samples == 2_000
    assert namespace["UNCERTAINTY"].seed == 7
    assert namespace["HARDWARE_MITIGATION"].readout is True
    assert namespace["HARDWARE_MITIGATION"].zne is True
    assert namespace["HARDWARE_MITIGATION"].zne_factors == (1, 3, 5)

    cells = [cell for cell in code_cells(load_notebook()) if named_calls(source(cell), "run_experiment")]
    specs = {}
    for cell in cells:
        tree = ast.parse(source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "ExperimentSpec":
                    target = next(target for target in node.targets if isinstance(target, ast.Name))
                    specs[target.id] = node.value

    expected = {
        "AER_SPEC": ("AerIdeal", {"seed_simulator": 11}, "aer_ideal", False),
        "IQM_SPEC": ("IQMHardware", {"device": "garnet", "use_metrics": True}, "iqm_garnet", True),
        "PIASTQ_SPEC": ("PiastQHardware", {"mode": "managed", "owner": "notebook"}, "piastq", True),
    }
    assert set(specs) == set(expected)
    for name, (backend_name, backend_kwargs, backend_tag, needs_mitigation) in expected.items():
        spec = specs[name]
        keywords = {keyword.arg: keyword.value for keyword in spec.keywords}
        assert ast.literal_eval(keywords["state"]) == "two_qutrit"
        assert isinstance(keywords["basis"], ast.Call)
        assert isinstance(keywords["basis"].func, ast.Name) and keywords["basis"].func.id == "PathBasis"
        assert isinstance(keywords["basis"].args[0], ast.Name)
        assert keywords["basis"].args[0].id == "CANONICAL_BASIS_DIRECTORY"
        assert isinstance(keywords["shots"], ast.Name) and keywords["shots"].id == "SHOTS"
        assert isinstance(keywords["uncertainty"], ast.Name) and keywords["uncertainty"].id == "UNCERTAINTY"
        assert isinstance(keywords["backend"], ast.Call)
        assert isinstance(keywords["backend"].func, ast.Name) and keywords["backend"].func.id == backend_name
        assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in keywords["backend"].keywords} == backend_kwargs
        assert ast.literal_eval(keywords["tags"]) == {"baseline": "canonical_ez", "backend": backend_tag}
        if needs_mitigation:
            assert isinstance(keywords["mitigation"], ast.Name)
            assert keywords["mitigation"].id == "HARDWARE_MITIGATION"
        else:
            assert "mitigation" not in keywords

    summary = namespace["summarize_results"]({}, namespace["REFERENCE"])
    assert [row["backend"] for row in summary] == ["aer_ideal", "iqm_garnet", "piastq"]
    assert [row["status"] for row in summary] == ["not_run", "skipped", "skipped"]
    for row in summary:
        assert row["classical_bound"] == namespace["REFERENCE"].bell_functional.classical_bound
        assert row["ideal_bell_value"] == namespace["REFERENCE"].expected.ideal_bell_value


def test_summary_preserves_present_result_values_and_explicit_skips():
    namespace = setup_namespace(REPO_ROOT)
    result_values = {
        "raw": {"estimate": 1.23},
        "readout_mitigated": {"estimate": 1.24},
        "zne": {"estimate": 1.25},
        "zne_readout_mitigated": {"estimate": 1.26},
        "diagnostics": {"resamples": 2_000},
        "leakage_rate": 0.01,
    }
    artifact_dir = Path("artifacts") / "present-aer-result"
    present_aer = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=artifact_dir,
        values=result_values,
    )

    summary = namespace["summarize_results"](
        {"aer_ideal": present_aer}, namespace["REFERENCE"]
    )
    json.dumps(summary)

    aer_row, iqm_row, piastq_row = summary
    assert aer_row["backend"] == "aer_ideal"
    assert aer_row["status"] == "completed"
    for field, value in result_values.items():
        assert aer_row[field] == value
    assert aer_row["classical_bound"] == namespace["REFERENCE"].bell_functional.classical_bound
    assert aer_row["ideal_bell_value"] == namespace["REFERENCE"].expected.ideal_bell_value
    assert aer_row["artifact_dir"] == str(artifact_dir)
    assert iqm_row["status"] == "skipped"
    assert piastq_row["status"] == "skipped"


def test_notebook_has_no_secrets_user_paths_or_low_level_execution():
    notebook = load_notebook()
    full_source = "\n".join(source(cell) for cell in code_cells(notebook))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(rf"(?im)os\.environ(?:\.get)?\([^)]*{credential}", full_source)
    assert not re.search(rf"(?im)os\.environ\[[^\]]*{credential}[^\]]*\]\s*=", full_source)
    assert not re.search(rf"(?im)[\"']\w*{credential}\w*[\"']\s*:", full_source)
    assert "dashboard_api" not in full_source
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in ("PiastQClient", "IQMProvider(", "compute_bell_value_from_counts", "build_sampler_circuits", ".run("):
        assert forbidden not in full_source
def setup_namespace(cwd):
    """Execute notebook setup without materializing inputs or running experiments."""
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for cell in code_cells(load_notebook()):
            cell_source = source(cell)
            if "canonical-input-materialization" in cell.get("metadata", {}).get("tags", []):
                continue
            if named_calls(cell_source, "run_experiment") or "display(" in cell_source:
                continue
            exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def sha256_file(path):
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def replace_edge_with_legacy_unitary(namespace, target):
    circuit_path = target / "graph_state_direct_basis.qpy"
    circuit = namespace["load_single_circuit"](circuit_path)
    edge = circuit.data[-1]
    assert isinstance(edge.operation, DiagonalGate)
    circuit.data[-1] = CircuitInstruction(
        UnitaryGate(Operator(edge.operation).data, label="CZ_W"),
        edge.qubits,
        edge.clbits,
    )
    with circuit_path.open("wb") as handle:
        namespace["qpy"].dump(circuit, handle)

    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["graph_state_direct_basis.qpy"]["sha256"] = sha256_file(
        circuit_path
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize("cwd", [REPO_ROOT, NOTEBOOK_PATH.parent])
def test_setup_discovers_repo_root_from_root_and_notebook_directory(monkeypatch, tmp_path, cwd):
    monkeypatch.chdir(tmp_path)
    namespace = setup_namespace(cwd)
    assert namespace["REPO_ROOT"] == REPO_ROOT
    assert Path.cwd() == tmp_path


def test_prepare_canonical_basis_creates_exact_idempotent_bundle(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    assert target == tmp_path / "experiment_inputs" / "reference_bases" / "two_qutrit" / "canonical_ez"
    assert {path.name for path in target.iterdir()} == {
        "graph_state_direct_basis.qpy",
        "E.npy",
        "metadata.json",
    }
    first_hashes = {path.name: sha256_file(path) for path in target.iterdir()}
    assert namespace["prepare_canonical_basis"](tmp_path) == target
    assert {path.name: sha256_file(path) for path in target.iterdir()} == first_hashes


def test_prepare_canonical_basis_rebuilds_legacy_unitary_gate_bundle(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    replace_edge_with_legacy_unitary(namespace, target)

    returned = namespace["prepare_canonical_basis"](tmp_path)

    rebuilt = namespace["load_single_circuit"](
        returned / "graph_state_direct_basis.qpy"
    )
    assert isinstance(rebuilt.data[-1].operation, DiagonalGate)


@pytest.mark.parametrize(
    ("mode", "file_attributes"),
    [(stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)],
)
def test_prepare_canonical_basis_rejects_legacy_symlink_or_reparse_cache(
    tmp_path, monkeypatch, mode, file_attributes
):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    replace_edge_with_legacy_unitary(namespace, target)
    original_files = {path.name: path.read_bytes() for path in target.iterdir()}
    real_lstat = Path.lstat

    def simulated_lstat(path):
        if path == target:
            return SimpleNamespace(
                st_mode=mode,
                st_file_attributes=file_attributes,
            )
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        namespace["prepare_canonical_basis"](tmp_path)

    assert {path.name: path.read_bytes() for path in target.iterdir()} == original_files


def test_prepare_canonical_basis_rejects_corrupt_metadata(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    (target / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata"):
        namespace["prepare_canonical_basis"](tmp_path)


def test_validate_canonical_basis_enforces_isometry_and_unmeasured_qpy(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    expected_encoding = namespace["get_encoding"]("canonical_ez").as_array()
    expected_circuit = namespace["build_direct_basis_graph_state_circuit"]("two_qutrit", expected_encoding)
    namespace["validate_canonical_basis"](target, expected_encoding, expected_circuit)
    assert expected_encoding.shape == (4, 3)
    numpy = __import__("numpy")
    assert numpy.allclose(expected_encoding.conj().T @ expected_encoding, numpy.eye(3))
    circuit = namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy")
    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 0
    assert not any(instruction.operation.name == "measure" for instruction in circuit.data)

def test_prepare_canonical_basis_cleans_staging_after_write_failure(tmp_path, monkeypatch):
    namespace = setup_namespace(REPO_ROOT)

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated NPY failure")

    monkeypatch.setattr(namespace["np"], "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated NPY failure"):
        namespace["prepare_canonical_basis"](tmp_path)

    final_directory = tmp_path / "experiment_inputs" / "reference_bases" / "two_qutrit" / "canonical_ez"
    assert not final_directory.exists()
    parent = final_directory.parent
    assert not parent.exists() or not any(path.name.startswith(".canonical_ez.tmp-") for path in parent.iterdir())


def test_canonical_notebook_setup_runs_two_qutrit_bell_on_aer(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="two_qutrit",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=64,
        bootstrap=namespace["BootstrapConfig"](samples=20, seed=7),
        output_root=tmp_path / "runs",
    )

    result = namespace["run_experiment"](spec, repo_root=tmp_path)

    assert result.status.value == "completed"
    assert set(result.values) == {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "config",
        "diagnostics",
    }
    assert result.values["raw"] == result.values["raw_conditional"]
    raw_real = result.values["raw"]["estimate"]["real"]
    # Seed 11 and 64 shots produce 6.004560301527426; 0.01 covers only this
    # deterministic finite-shot deviation from the frozen ideal value 6.0.
    assert raw_real == pytest.approx(
        namespace["REFERENCE"].expected.ideal_bell_value,
        abs=0.01,
    )

    artifact_dir = Path(result.artifact_dir)
    assert artifact_dir.is_relative_to(tmp_path / "runs")
    assert (artifact_dir / "experiment.json").is_file()
    assert {path.name for path in artifact_dir.iterdir()} == {"experiment.json"}

    numpy = namespace["np"]
    expected_encoding = namespace["get_encoding"]("canonical_ez").as_array()
    numpy.testing.assert_array_equal(
        numpy.load(basis / "E.npy", allow_pickle=False), expected_encoding
    )
    circuit = namespace["load_single_circuit"](basis / "graph_state_direct_basis.qpy")
    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 0
    assert circuit.data
    expected_circuit = namespace["build_direct_basis_graph_state_circuit"](
        "two_qutrit", expected_encoding
    )
    assert namespace["Statevector"].from_instruction(circuit).equiv(
        namespace["Statevector"].from_instruction(expected_circuit)
    )
