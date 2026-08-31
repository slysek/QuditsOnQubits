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
            and node.value.func.id in {"ExperimentSpec", "build_iqm_spec"}
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
    if (
        isinstance(value.func, ast.Name)
        and value.func.id == "build_iqm_spec"
    ):
        return "IQMHardware"
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


def assert_full_twirling_artifact(result):
    artifact_dir = Path(result.artifact_dir)
    document = json.loads(
        (artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert document["twirling"] == {
        "provider": "iqm-error-reduction-tools",
        "method": "circuit_twirling",
        "readout_strategy": "NONE",
        "instances_per_circuit": 5,
        "seed": 7,
        "shots_per_instance": 20,
        "total_shots_per_circuit": 100,
    }
    assert set(document["counts_by_factor"]) == {"1", "3", "5"}
    assert all(
        len(entries) == 12
        and all(sum(entry["counts"].values()) == 100 for entry in entries)
        for entries in document["counts_by_factor"].values()
    )
    assert {path.name for path in artifact_dir.iterdir()} == {"experiment.json"}


def test_notebook_has_aer_and_opt_in_iqm_runs_and_clean_cells():
    notebook = load_notebook()
    cells = code_cells(notebook)
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "QuditsOnQubitsEnv",
        "language": "python",
        "name": "quditsonqubitsenv",
    }
    assert notebook["metadata"]["language_info"] == {"name": "python"}
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
        "WorkloadOptimizationConfig",
        "run_experiment",
    } <= imported
    assert "PiastQHardware" not in imported
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)


def test_iqm_cell_is_opt_in_and_does_not_submit_by_default():
    namespace = setup_namespace(REPO_ROOT)
    iqm_cell = run_cell_for_backend("IQMHardware")
    submissions = []
    namespace["CANONICAL_BASIS_DIRECTORY"] = REPO_ROOT / "unused-basis"
    namespace["run_experiment"] = lambda *args, **kwargs: submissions.append(
        (args, kwargs)
    )

    exec(compile(source(iqm_cell), str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["RUN_IQM"] is False
    assert submissions == []


def test_iqm_spec_receives_securely_resolved_env_path(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    iqm_cell_source = source(run_cell_for_backend("IQMHardware"))
    env_path = tmp_path / ".env"

    assert callable(namespace["resolve_iqm_env_path"])
    assert "IQM_ENV_PATH = resolve_iqm_env_path(REPO_ROOT)" in iqm_cell_source
    assert "env_path=IQM_ENV_PATH" in iqm_cell_source

    spec = namespace["build_iqm_spec"](
        tmp_path / "basis",
        output_root=tmp_path / "runs",
        env_path=env_path,
    )

    assert spec.backend.env_path == env_path


def test_configuration_and_empty_summary_are_semantically_complete(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["SHOTS"] == 100
    assert namespace["IQM_LAYOUT_CANDIDATES"] == ((0, 1, 2, 7, 3, 4),)
    assert tuple(
        zip(
            namespace["IQM_LAYOUT_CANDIDATES"][0][::2],
            namespace["IQM_LAYOUT_CANDIDATES"][0][1::2],
            strict=True,
        )
    ) == ((0, 1), (2, 7), (3, 4))
    assert namespace["IQM_SEED_CANDIDATES"] == (3, 7, 13)
    assert namespace["workload_optimization"] == namespace[
        "WorkloadOptimizationConfig"
    ](
        initial_layouts=((0, 1, 2, 7, 3, 4),),
        seed_transpilers=(3, 7, 13),
    )
    assert namespace["UNCERTAINTY"].samples == 2_000
    assert namespace["UNCERTAINTY"].seed == 7
    assert namespace["HARDWARE_MITIGATION"].readout is True
    assert namespace["HARDWARE_MITIGATION"].zne is True
    assert namespace["HARDWARE_MITIGATION"].zne_factors == (1, 3, 5)
    assert namespace["HARDWARE_MITIGATION"].circuit_twirling is True
    assert namespace["HARDWARE_MITIGATION"].twirling_instances == 5
    assert namespace["HARDWARE_MITIGATION"].twirling_seed == 7

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
    assert "workload_optimization" not in keywords
    assert (
        isinstance(keywords["uncertainty"], ast.Name)
        and keywords["uncertainty"].id == "UNCERTAINTY"
    )
    assert ast.literal_eval(keywords["tags"]) == {
        "baseline": "canonical_ez",
        "backend": "aer_ideal",
    }

    iqm_spec = namespace["build_iqm_spec"](
        tmp_path / "basis",
        output_root=tmp_path / "runs",
    )
    assert iqm_spec.state == "ghz3"
    assert iqm_spec.basis.directory == tmp_path / "basis"
    assert type(iqm_spec.backend).__name__ == "IQMHardware"
    assert iqm_spec.backend.device == "garnet"
    assert iqm_spec.backend.use_metrics is True
    assert iqm_spec.shots == 100
    assert iqm_spec.mitigation == namespace["HARDWARE_MITIGATION"]
    assert iqm_spec.uncertainty == namespace["UNCERTAINTY"]
    assert iqm_spec.workload_optimization == namespace["workload_optimization"]
    assert dict(iqm_spec.tags) == {
        "baseline": "canonical_ez",
        "backend": "iqm_garnet",
    }
    assert iqm_spec.output_root == tmp_path / "runs"

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
    assert all(row["raw"] is None for row in summary)
    assert all(row["raw_conditional"] is None for row in summary)
    assert all(row["raw_unconditional"] is None for row in summary)
    assert all(row["raw_invalid_codeword_rate"] is None for row in summary)
    assert all(row["selected_layout"] is None for row in summary)
    assert all(row["selected_seed_transpiler"] is None for row in summary)
    assert all(row["selected_workload_aggregate"] is None for row in summary)
    assert all(row["selected_unconditional"] is None for row in summary)
    assert all(row["selected_unconditional_source"] is None for row in summary)
    assert all(
        row["unconditional_exceeds_classical_bound"] is None for row in summary
    )
    json.dumps(summary)


def test_summary_preserves_runner_values_and_reads_workload_artifact(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    values = {
        "raw": {"estimate": 6.0},
        "raw_conditional": {"estimate": 6.0},
        "raw_unconditional": {"estimate": {"real": 5.9, "imag": 0.0}},
        "raw_invalid_codeword_rate": {"estimate": 0.01},
        "readout_mitigated": {"estimate": 6.1},
        "zne": {"estimate": 6.2},
        "zne_readout_mitigated": {"estimate": 6.3},
        "diagnostics": {"factors": [1, 3, 5]},
        "leakage_rate": 0.01,
    }
    aer_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=tmp_path / "missing-ghz3-aer",
        values=values,
    )
    iqm_values = {
        **values,
        "raw": {"estimate": 4.3},
        "raw_conditional": {"estimate": 4.3},
        "raw_unconditional": {"estimate": {"real": 3.9, "imag": 0.0}},
        "zne_readout_mitigated": {"estimate": 5.1},
        "zne_readout_mitigated_unconditional": {
            "estimate": {"real": 4.2, "imag": 0.0}
        },
    }
    iqm_artifact_dir = tmp_path / "ghz3-iqm"
    iqm_artifact_dir.mkdir()
    workload_optimization = {
        "selected_layout": [0, 1, 2, 3, 4, 7],
        "selected_seed_transpiler": 7,
        "selected_workload": {
            "aggregate": {
                "circuit_count": 12,
                "maximum_depth": 20,
                "total_two_qubit_gate_count": 48,
            }
        },
    }
    (iqm_artifact_dir / "experiment.json").write_text(
        json.dumps({"workload_optimization": workload_optimization}),
        encoding="utf-8",
    )
    iqm_result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=iqm_artifact_dir,
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
    assert rows[0]["selected_layout"] is None
    assert rows[0]["selected_seed_transpiler"] is None
    assert rows[0]["selected_workload_aggregate"] is None
    assert rows[1]["status"] == "completed"
    for field, value in iqm_values.items():
        assert rows[1][field] == value
    assert rows[1]["artifact_dir"] == str(iqm_result.artifact_dir)
    assert rows[1]["workload_optimization"] == workload_optimization
    assert rows[1]["selected_layout"] == [0, 1, 2, 3, 4, 7]
    assert rows[1]["selected_seed_transpiler"] == 7
    assert rows[1]["selected_workload_aggregate"] == {
        "circuit_count": 12,
        "maximum_depth": 20,
        "total_two_qubit_gate_count": 48,
    }
    assert rows[1]["selected_unconditional_source"] == (
        "zne_readout_mitigated_unconditional"
    )
    assert rows[1]["selected_unconditional"] == iqm_values[
        "zne_readout_mitigated_unconditional"
    ]


def test_bound_comparison_uses_only_explicit_unconditional_estimates():
    namespace = setup_namespace(REPO_ROOT)
    conditional_only = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("missing-conditional-only-artifact"),
        values={"raw": {"estimate": {"real": 100.0, "imag": 0.0}}},
    )
    structured = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("missing-structured-artifact"),
        values={
            "raw": {"estimate": {"real": 100.0, "imag": 0.0}},
            "raw_conditional": {"estimate": {"real": 100.0, "imag": 0.0}},
            "raw_unconditional": {"estimate": {"real": 3.0, "imag": 0.0}},
        },
    )

    rows = namespace["summarize_results"](
        {"aer_ideal": conditional_only, "iqm_garnet": structured},
        namespace["REFERENCE"],
    )

    assert rows[0]["selected_unconditional"] is None
    assert rows[0]["selected_unconditional_source"] is None
    assert rows[0]["unconditional_exceeds_classical_bound"] is None
    assert rows[1]["selected_unconditional"] == structured.values[
        "raw_unconditional"
    ]
    assert rows[1]["selected_unconditional_source"] == "raw_unconditional"
    assert rows[1]["unconditional_exceeds_classical_bound"] is False


def test_notebook_source_names_explicit_semantics_and_unconditional_bound_evidence():
    cells = code_cells(load_notebook())
    full_source = "\n".join(source(cell) for cell in cells)
    summary_source = next(
        source(cell) for cell in cells if "def summarize_results" in source(cell)
    )

    for required in (
        "WorkloadOptimizationConfig",
        "IQM_LAYOUT_CANDIDATES",
        "IQM_SEED_CANDIDATES",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
    ):
        assert required in full_source
    assert "selected_unconditional" in summary_source
    assert "unconditional_exceeds_classical_bound" in summary_source


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
    assert {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "config",
        "diagnostics",
    } <= set(result.values)
    assert result.values["raw"] == result.values["raw_conditional"]
    # Seed 11 and 64 shots produce 6.071911234791627. The tolerance covers only
    # this deterministic finite-shot deviation from the frozen ideal value 6.0.
    assert result.values["raw"]["estimate"]["real"] == pytest.approx(
        namespace["REFERENCE"].expected.ideal_bell_value,
        abs=0.08,
    )
    artifact_dir = Path(result.artifact_dir)
    assert artifact_dir.is_relative_to(tmp_path / "runs")
    assert (artifact_dir / "experiment.json").is_file()
    assert {path.name for path in artifact_dir.iterdir()} == {"experiment.json"}

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


def test_ghz3_notebook_full_mitigation_pipeline_offline(tmp_path):
    from qiskit import QuantumCircuit

    from qudits_on_qubits.experiments.backends import (
        BackendIdentity,
        CompiledBatch,
        ExecutionResult,
        SubmittedJob,
    )
    from qudits_on_qubits.experiments.mitigation import TwirledBatch

    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["build_iqm_spec"](
        basis,
        output_root=tmp_path / "runs",
    )

    class OfflineIQMAdapter:
        def __init__(self):
            self.identity = BackendIdentity(
                "iqm",
                "garnet",
                metadata={"calibration_set_id": "offline-cal"},
            )
            self.compile_calls = 0
            self.compile_physical_calls = 0
            self.submissions = []

        def resolve(self):
            return self.identity

        def compile(self, circuits, config):
            self.compile_calls += 1
            layout = config.initial_layout
            compiled = []
            for source_circuit in circuits:
                circuit = QuantumCircuit(
                    max(layout) + 1,
                    source_circuit.num_clbits,
                    name=source_circuit.name,
                )
                circuit.metadata = dict(source_circuit.metadata or {})
                circuit.measure(layout, range(source_circuit.num_clbits))
                compiled.append(circuit)
            return CompiledBatch(tuple(compiled), self.identity)

        def compile_physical(self, circuits, config):
            self.compile_physical_calls += 1
            return CompiledBatch(tuple(circuits), self.identity)

        def submit(self, circuits, shots, options=None):
            batch = tuple(circuits)
            self.submissions.append((batch, shots))
            return SubmittedJob(
                f"offline-{len(self.submissions)}",
                batch,
                self.identity,
                len(batch),
                shots,
            )

        def result(self, submitted, timeout=None):
            counts = []
            for circuit in submitted.handle:
                prepared_state = (circuit.metadata or {}).get("prepared_state")
                if prepared_state is not None:
                    counts.append({str(prepared_state): submitted.shots})
                else:
                    counts.append({"0" * circuit.num_clbits: submitted.shots})
            return ExecutionResult(
                tuple(counts),
                submitted.job_id,
                self.identity,
                status="done",
            )

    class PureReadout:
        def build_context(self, calibration):
            return calibration.assignment_matrices

        def resample_calibration(self, calibration, rng):
            return calibration.assignment_matrices

        def apply(self, counts_by_setting, _context):
            return {
                setting: {
                    outcome: count / sum(counts.values())
                    for outcome, count in counts.items()
                }
                for setting, counts in counts_by_setting.items()
            }

    class PureZNE:
        def extrapolate(self, _factors, values):
            return values[0]

    transform_calls = []

    def offline_twirl(circuits, *, instances, seed):
        compiled = tuple(circuits)
        transform_calls.append((compiled, instances, seed))
        variants = tuple(
            circuit.copy(name=f"{circuit.name}-twirl-{instance}")
            for circuit in compiled
            for instance in range(instances)
        )
        return TwirledBatch(
            circuits=variants,
            original_indices=tuple(
                original
                for original in range(len(compiled))
                for _ in range(instances)
            ),
            instance_indices=tuple(
                instance
                for _ in compiled
                for instance in range(instances)
            ),
            metadata={
                "provider": "iqm-error-reduction-tools",
                "method": "circuit_twirling",
                "readout_strategy": "NONE",
                "instances_per_circuit": instances,
                "seed": seed,
            },
        )

    adapter = OfflineIQMAdapter()
    result = namespace["run_experiment"](
        spec,
        adapter=adapter,
        repo_root=tmp_path,
        _twirling_transform=offline_twirl,
        _readout_strategy=PureReadout(),
        _zne_strategy=PureZNE(),
    )

    assert result.status.value == "completed"
    assert adapter.compile_calls == 3
    assert adapter.compile_physical_calls == 1
    assert len(transform_calls) == 1
    assert len(transform_calls[0][0]) == 12
    assert transform_calls[0][1:] == (5, 7)
    assert [shots for _circuits, shots in adapter.submissions] == [100, 20, 20, 20]
    assert [len(circuits) for circuits, _shots in adapter.submissions] == [12, 60, 60, 60]
    assert {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "readout_mitigated",
        "readout_mitigated_conditional",
        "readout_mitigated_unconditional",
        "readout_effective_invalid_codeword_weight",
        "zne",
        "zne_conditional",
        "zne_unconditional",
        "zne_readout_mitigated",
        "zne_readout_mitigated_conditional",
        "zne_readout_mitigated_unconditional",
        "diagnostics",
    } <= set(result.values)
    assert result.values["raw"] == result.values["raw_conditional"]

    assert_full_twirling_artifact(result)


@pytest.mark.skipif(
    os.environ.get("QOQ_RUN_IQM_HARDWARE") != "1",
    reason="set QOQ_RUN_IQM_HARDWARE=1 to submit the IQM Garnet smoke run",
)
def test_ghz3_canonical_full_pipeline_on_iqm_garnet():
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](REPO_ROOT)
    env_path = namespace["resolve_iqm_env_path"](REPO_ROOT)
    spec = namespace["build_iqm_spec"](basis, env_path=env_path)

    result = namespace["run_experiment"](spec, repo_root=REPO_ROOT)

    assert result.status.value == "completed"
    assert {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "readout_mitigated",
        "readout_mitigated_conditional",
        "readout_mitigated_unconditional",
        "readout_effective_invalid_codeword_weight",
        "zne",
        "zne_conditional",
        "zne_unconditional",
        "zne_readout_mitigated",
        "zne_readout_mitigated_conditional",
        "zne_readout_mitigated_unconditional",
        "diagnostics",
    } <= set(result.values)
    assert result.values["diagnostics"]["factors"] == [1, 3, 5]

    assert_full_twirling_artifact(result)
