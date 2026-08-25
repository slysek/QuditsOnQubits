import ast
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ghz3_bell_canonical_baseline.ipynb"
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
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def runner_backend(cell_source, call):
    tree = ast.parse(cell_source)
    specs = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ExperimentSpec"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    specs[target.id] = node.value
    value = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "spec"),
        call.args[0] if call.args else None,
    )
    if isinstance(value, ast.Name):
        value = specs.get(value.id)
    assert isinstance(value, ast.Call)
    backend = next(keyword.value for keyword in value.keywords if keyword.arg == "backend")
    assert isinstance(backend, ast.Call) and isinstance(backend.func, ast.Name)
    return backend.func.id


def run_cell_for_backend(backend_name):
    for cell in code_cells(load_notebook()):
        if any(
            runner_backend(source(cell), call) == backend_name
            for call in named_calls(source(cell), "run_experiment")
        ):
            return cell
    raise AssertionError(f"missing run cell for backend {backend_name}")


def setup_namespace(cwd):
    """Execute setup cells without materializing inputs or running experiments."""
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for cell in code_cells(load_notebook()):
            cell_source = source(cell)
            if "canonical-input-materialization" in cell.get("metadata", {}).get(
                "tags", []
            ):
                continue
            if named_calls(cell_source, "run_experiment") or "SUMMARY =" in cell_source:
                continue
            exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def sha256_file(path):
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def test_notebook_has_aer_and_opt_in_iqm_runs_and_clean_cells():
    notebook = load_notebook()
    cells = code_cells(notebook)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 2
    assert {
        runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0])
        for cell in run_cells
    } == {"AerIdeal", "IQMHardware"}
    assert all(
        len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells
    )
    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo) == 1
        assert isinstance(repo[0].value, ast.Name)
        assert repo[0].value.id == "REPO_ROOT"

    tree = ast.parse("\n".join(source(cell) for cell in cells))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "qudits_on_qubits.experiments"
        for alias in node.names
    }
    assert {
        "AerIdeal",
        "BootstrapConfig",
        "ExperimentSpec",
        "IQMHardware",
        "MitigationConfig",
        "PathBasis",
        "run_experiment",
    } <= imported
    assert "PiastQHardware" not in imported
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)


def test_iqm_cell_is_opt_in_and_does_not_submit_by_default():
    namespace = setup_namespace(REPO_ROOT)
    iqm_cell = run_cell_for_backend("IQMHardware")
    submissions = []
    namespace["run_experiment"] = lambda *args, **kwargs: submissions.append(
        (args, kwargs)
    )

    exec(compile(source(iqm_cell), str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["RUN_IQM"] is False
    assert submissions == []


def test_configuration_and_empty_summary_are_semantically_complete():
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["SHOTS"] == 100
    assert namespace["UNCERTAINTY"].samples == 2_000
    assert namespace["UNCERTAINTY"].seed == 7
    assert namespace["HARDWARE_MITIGATION"].readout is True
    assert namespace["HARDWARE_MITIGATION"].zne is True
    assert namespace["HARDWARE_MITIGATION"].zne_factors == (1, 3, 5)

    run_cell = next(
        cell
        for cell in code_cells(load_notebook())
        if named_calls(source(cell), "run_experiment")
    )
    tree = ast.parse(source(run_cell))
    spec = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "AER_SPEC" for target in node.targets)
        and isinstance(node.value, ast.Call)
    )
    keywords = {keyword.arg: keyword.value for keyword in spec.keywords}
    assert ast.literal_eval(keywords["state"]) == "ghz3"
    assert isinstance(keywords["basis"], ast.Call)
    assert isinstance(keywords["basis"].func, ast.Name)
    assert keywords["basis"].func.id == "PathBasis"
    assert isinstance(keywords["basis"].args[0], ast.Name)
    assert keywords["basis"].args[0].id == "CANONICAL_BASIS_DIRECTORY"
    assert isinstance(keywords["backend"], ast.Call)
    assert isinstance(keywords["backend"].func, ast.Name)
    assert keywords["backend"].func.id == "AerIdeal"
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in keywords["backend"].keywords
    } == {"seed_simulator": 11}
    assert isinstance(keywords["shots"], ast.Name) and keywords["shots"].id == "SHOTS"
    assert (
        isinstance(keywords["uncertainty"], ast.Name)
        and keywords["uncertainty"].id == "UNCERTAINTY"
    )
    assert ast.literal_eval(keywords["tags"]) == {
        "baseline": "canonical_ez",
        "backend": "aer_ideal",
    }

    iqm_run_cell = run_cell_for_backend("IQMHardware")
    iqm_tree = ast.parse(source(iqm_run_cell))
    iqm_spec = next(
        node.value
        for node in ast.walk(iqm_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "IQM_SPEC"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    )
    iqm_keywords = {keyword.arg: keyword.value for keyword in iqm_spec.keywords}
    assert ast.literal_eval(iqm_keywords["state"]) == "ghz3"
    assert isinstance(iqm_keywords["basis"], ast.Call)
    assert isinstance(iqm_keywords["basis"].func, ast.Name)
    assert iqm_keywords["basis"].func.id == "PathBasis"
    assert isinstance(iqm_keywords["basis"].args[0], ast.Name)
    assert iqm_keywords["basis"].args[0].id == "CANONICAL_BASIS_DIRECTORY"
    assert isinstance(iqm_keywords["backend"], ast.Call)
    assert isinstance(iqm_keywords["backend"].func, ast.Name)
    assert iqm_keywords["backend"].func.id == "IQMHardware"
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in iqm_keywords["backend"].keywords
    } == {"device": "garnet", "use_metrics": True}
    assert (
        isinstance(iqm_keywords["shots"], ast.Name)
        and iqm_keywords["shots"].id == "SHOTS"
    )
    assert (
        isinstance(iqm_keywords["mitigation"], ast.Name)
        and iqm_keywords["mitigation"].id == "HARDWARE_MITIGATION"
    )
    assert (
        isinstance(iqm_keywords["uncertainty"], ast.Name)
        and iqm_keywords["uncertainty"].id == "UNCERTAINTY"
    )
    assert ast.literal_eval(iqm_keywords["tags"]) == {
        "baseline": "canonical_ez",
        "backend": "iqm_garnet",
    }

    summary = namespace["summarize_results"]({}, namespace["REFERENCE"])
    assert [row["backend"] for row in summary] == ["aer_ideal", "iqm_garnet"]
    assert [row["status"] for row in summary] == ["not_run", "skipped"]
    assert all(
        row["classical_bound"]
        == namespace["REFERENCE"].bell_functional.classical_bound
        for row in summary
    )
    assert all(
        row["ideal_bell_value"]
        == namespace["REFERENCE"].expected.ideal_bell_value
        for row in summary
    )
    json.dumps(summary)


def test_summary_preserves_runner_values():
    namespace = setup_namespace(REPO_ROOT)
    values = {
        "raw": {"estimate": 6.0},
        "readout_mitigated": {"estimate": 6.1},
        "zne": {"estimate": 6.2},
        "zne_readout_mitigated": {"estimate": 6.3},
        "diagnostics": {"factors": [1, 3, 5]},
        "leakage_rate": 0.01,
    }
    aer_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ghz3-aer",
        values=values,
    )
    iqm_values = {
        **values,
        "raw": {"estimate": 4.3},
        "zne_readout_mitigated": {"estimate": 5.1},
    }
    iqm_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ghz3-iqm",
        values=iqm_values,
    )
    rows = namespace["summarize_results"](
        {"aer_ideal": aer_result, "iqm_garnet": iqm_result},
        namespace["REFERENCE"],
    )
    assert [row["backend"] for row in rows] == ["aer_ideal", "iqm_garnet"]
    assert rows[0]["status"] == "completed"
    for field, value in values.items():
        assert rows[0][field] == value
    assert rows[0]["artifact_dir"] == str(aer_result.artifact_dir)
    assert rows[1]["status"] == "completed"
    for field, value in iqm_values.items():
        assert rows[1][field] == value
    assert rows[1]["artifact_dir"] == str(iqm_result.artifact_dir)


def test_notebook_has_no_secrets_paths_provider_or_low_level_execution():
    full_source = "\n".join(source(cell) for cell in code_cells(load_notebook()))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(rf"(?im)os\.environ(?:\.get)?\([^)]*{credential}", full_source)
    assert not re.search(rf"(?im)os\.environ\[[^\]]*{credential}[^\]]*\]\s*=", full_source)
    assert not re.search(rf"(?im)[\"']\w*{credential}\w*[\"']\s*:", full_source)
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in (
        "PiastQClient",
        "PiastQHardware",
        "IQMProvider(",
        "compute_bell_value_from_counts",
        "build_sampler_circuits",
        ".run(",
    ):
        assert forbidden not in full_source


@pytest.mark.parametrize("cwd", [REPO_ROOT, NOTEBOOK_PATH.parent])
def test_setup_discovers_repo_root(monkeypatch, tmp_path, cwd):
    monkeypatch.chdir(tmp_path)
    namespace = setup_namespace(cwd)
    assert namespace["REPO_ROOT"] == REPO_ROOT
    assert Path.cwd() == tmp_path


def test_prepare_canonical_basis_creates_exact_idempotent_bundle(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    assert target == (
        tmp_path
        / "experiment_inputs"
        / "reference_bases"
        / "ghz3"
        / "canonical_ez"
    )
    assert {path.name for path in target.iterdir()} == {
        "graph_state_direct_basis.qpy",
        "E.npy",
        "metadata.json",
    }
    first_hashes = {path.name: sha256_file(path) for path in target.iterdir()}
    assert namespace["prepare_canonical_basis"](tmp_path) == target
    assert {path.name: sha256_file(path) for path in target.iterdir()} == first_hashes


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
    expected_circuit = namespace["build_direct_basis_graph_state_circuit"](
        "ghz3", expected_encoding
    )
    namespace["validate_canonical_basis"](
        target, expected_encoding, expected_circuit
    )
    assert expected_encoding.shape == (4, 3)
    numpy = namespace["np"]
    assert numpy.allclose(
        expected_encoding.conj().T @ expected_encoding, numpy.eye(3)
    )
    circuit = namespace["load_single_circuit"](
        target / "graph_state_direct_basis.qpy"
    )
    assert circuit.num_qubits == 6
    assert circuit.num_clbits == 0
    assert not any(
        instruction.operation.name == "measure" for instruction in circuit.data
    )


def test_prepare_canonical_basis_cleans_staging_after_write_failure(
    tmp_path, monkeypatch
):
    namespace = setup_namespace(REPO_ROOT)

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated NPY failure")

    monkeypatch.setattr(namespace["np"], "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated NPY failure"):
        namespace["prepare_canonical_basis"](tmp_path)

    final_directory = (
        tmp_path
        / "experiment_inputs"
        / "reference_bases"
        / "ghz3"
        / "canonical_ez"
    )
    assert not final_directory.exists()
    parent = final_directory.parent
    assert not parent.exists() or not any(
        path.name.startswith(".canonical_ez.tmp-") for path in parent.iterdir()
    )


def test_canonical_notebook_runs_ghz3_on_aer(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="ghz3",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=64,
        uncertainty=namespace["BootstrapConfig"](samples=20, seed=7),
        output_root=tmp_path / "runs",
    )

    result = namespace["run_experiment"](spec, repo_root=tmp_path)

    assert result.status.value == "completed"
    assert set(result.values) == {"raw", "config", "diagnostics"}
    # Seed 11 and 64 shots produce 6.071911234791627. The tolerance covers only
    # this deterministic finite-shot deviation from the frozen ideal value 6.0.
    assert result.values["raw"]["estimate"]["real"] == pytest.approx(
        namespace["REFERENCE"].expected.ideal_bell_value,
        abs=0.08,
    )
    artifact_dir = Path(result.artifact_dir)
    assert artifact_dir.is_relative_to(tmp_path / "runs")
    assert (artifact_dir / "experiment.json").is_file()
    assert (artifact_dir / "result.json").is_file()

    expected_encoding = namespace["get_encoding"]("canonical_ez").as_array()
    numpy = namespace["np"]
    numpy.testing.assert_array_equal(
        numpy.load(basis / "E.npy", allow_pickle=False), expected_encoding
    )
    circuit = namespace["load_single_circuit"](
        basis / "graph_state_direct_basis.qpy"
    )
    expected_circuit = namespace["build_direct_basis_graph_state_circuit"](
        "ghz3", expected_encoding
    )
    assert namespace["Statevector"].from_instruction(circuit).equiv(
        namespace["Statevector"].from_instruction(expected_circuit)
    )


@pytest.mark.skipif(
    os.environ.get("QOQ_RUN_IQM_HARDWARE") != "1",
    reason="set QOQ_RUN_IQM_HARDWARE=1 to submit the IQM Garnet smoke run",
)
def test_ghz3_canonical_full_pipeline_on_iqm_garnet():
    from qudits_on_qubits.experiments import (
        IQMHardware,
        MitigationConfig,
        TranspilationConfig,
    )

    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](REPO_ROOT)
    configured_env_path = os.environ.get("QOQ_IQM_ENV_PATH")
    spec = namespace["ExperimentSpec"](
        state="ghz3",
        basis=namespace["PathBasis"](basis),
        backend=IQMHardware(
            device="garnet",
            use_metrics=True,
            env_path=(
                None
                if configured_env_path is None
                else Path(configured_env_path)
            ),
        ),
        shots=50,
        mitigation=MitigationConfig(
            readout=True,
            zne=True,
            zne_factors=(1, 3, 5),
        ),
        uncertainty=namespace["BootstrapConfig"](samples=20, seed=7),
        transpilation=TranspilationConfig(seed_transpiler=11),
        tags={"baseline": "canonical_ez", "backend": "iqm_garnet_smoke"},
    )

    result = namespace["run_experiment"](spec, repo_root=REPO_ROOT)

    assert result.status.value == "completed"
    assert {
        "raw",
        "readout_mitigated",
        "zne",
        "zne_readout_mitigated",
        "diagnostics",
    } <= set(result.values)
    assert result.values["diagnostics"]["factors"] == [1, 3, 5]
