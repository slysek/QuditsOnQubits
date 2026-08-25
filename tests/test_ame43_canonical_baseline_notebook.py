import ast
import json
import os
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from qiskit.quantum_info import Operator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ame43_canonical_baseline.ipynb"
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


def setup_namespace(cwd):
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_notebook_exists():
    assert NOTEBOOK_PATH.is_file()


def test_notebook_has_three_separate_high_level_backend_runs():
    cells = code_cells(load_notebook())
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    assert len(run_cells) == 3
    assert all(len(named_calls(source(cell), "run_experiment")) == 1 for cell in run_cells)
    assert [
        runner_backend(source(cell), named_calls(source(cell), "run_experiment")[0])
        for cell in run_cells
    ] == ["AerIdeal", "IQMHardware", "PiastQHardware"]
    assert all(cell["execution_count"] is None for cell in cells)
    assert all(cell["outputs"] == [] for cell in cells)
    for cell in run_cells:
        call = named_calls(source(cell), "run_experiment")[0]
        repo = [keyword for keyword in call.keywords if keyword.arg == "repo_root"]
        assert len(repo) == 1
        assert isinstance(repo[0].value, ast.Name)
        assert repo[0].value.id == "REPO_ROOT"


def test_hardware_cells_are_false_by_default_and_separately_guarded():
    cells = code_cells(load_notebook())
    full_source = "\n".join(source(cell) for cell in cells)
    assert re.search(r"^RUN_IQM\s*=\s*False\s*$", full_source, re.MULTILINE)
    assert re.search(r"^RUN_PIASTQ\s*=\s*False\s*$", full_source, re.MULTILINE)
    run_cells = [cell for cell in cells if named_calls(source(cell), "run_experiment")]
    for backend, flag in (("IQMHardware", "RUN_IQM"), ("PiastQHardware", "RUN_PIASTQ")):
        cell = next(
            item
            for item in run_cells
            if runner_backend(source(item), named_calls(source(item), "run_experiment")[0])
            == backend
        )
        tree = ast.parse(source(cell))
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == flag
        ]
        assert guards


def test_iqm_env_path_resolver_prefers_repo_and_worktree_fallback(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    resolver = namespace["resolve_iqm_env_path"]

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    checkout_env = checkout / ".env"
    checkout_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert resolver(checkout) == checkout_env

    for index, absolute in enumerate((False, True)):
        owner = tmp_path / f"owner-repository-{index}"
        worktree = tmp_path / f"external-checkout-{index}"
        owner_git = owner / ".git"
        worktree_git = owner_git / "worktrees" / "external"
        worktree.mkdir()
        worktree_git.mkdir(parents=True)
        common_value = str(owner_git) if absolute else "../.."
        (worktree_git / "commondir").write_text(f"{common_value}\n", encoding="utf-8")
        gitdir_value = str(worktree_git) if absolute else os.path.relpath(worktree_git, worktree)
        (worktree / ".git").write_text(f"gitdir: {gitdir_value}\n", encoding="utf-8")
        fallback_env = owner / ".env"
        fallback_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
        assert resolver(worktree) == fallback_env


@pytest.mark.parametrize("metadata", ["missing", "malformed"])
def test_iqm_env_path_resolver_reports_missing_or_malformed_git_metadata(tmp_path, metadata):
    namespace = setup_namespace(REPO_ROOT)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if metadata == "malformed":
        (checkout / ".git").write_text("not git metadata\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"IQM .*\.env|Git metadata"):
        namespace["resolve_iqm_env_path"](checkout)


@pytest.mark.parametrize(("mode", "file_attributes"), [(stat.S_IFLNK, 0), (stat.S_IFREG, 0x400)])
def test_iqm_env_path_resolver_rejects_symlink_or_reparse_env(tmp_path, monkeypatch, mode, file_attributes):
    namespace = setup_namespace(REPO_ROOT)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    env_path = checkout / ".env"
    env_path.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    real_lstat = Path.lstat

    def simulated_lstat(path):
        if path == env_path:
            return SimpleNamespace(st_mode=mode, st_file_attributes=file_attributes)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        namespace["resolve_iqm_env_path"](checkout)


def test_iqm_cell_resolves_env_only_inside_guard_and_passes_explicit_path():
    cells = code_cells(load_notebook())
    cell = next(
        cell
        for cell in cells
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_experiment"
            for node in ast.walk(ast.parse(source(cell)))
        )
        and "IQMHardware" in source(cell)
    )
    tree = ast.parse(source(cell))
    iqm_guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "RUN_IQM"
    )
    guard_nodes = set(ast.walk(iqm_guard))
    resolver_calls = [
        node
        for node in ast.walk(iqm_guard)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_iqm_env_path"
    ]
    assert len(resolver_calls) == 1
    iqm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "IQMHardware"
    ]
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_experiment"
    ]
    run_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_experiment"
    ]
    assert len(iqm_calls) == len(run_calls) == len(run_assignments) == 1
    assert all(node in guard_nodes for node in (*resolver_calls, *iqm_calls, *run_calls, *run_assignments))
    env_path = next(keyword for keyword in iqm_calls[0].keywords if keyword.arg == "env_path")
    assert isinstance(env_path.value, ast.Name)
    assert env_path.value.id == "IQM_ENV_PATH"


def test_notebook_configuration_is_canonical_ame43():
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["SHOTS"] == 100
    assert namespace["UNCERTAINTY"].samples == 2_000
    assert namespace["UNCERTAINTY"].seed == 7
    assert namespace["HARDWARE_MITIGATION"].readout is True
    assert namespace["HARDWARE_MITIGATION"].zne is True
    assert namespace["HARDWARE_MITIGATION"].zne_factors == (1, 3, 5)
    assert namespace["REFERENCE"].experiment_id == "ame43"

    cells = [
        cell
        for cell in code_cells(load_notebook())
        if named_calls(source(cell), "run_experiment")
    ]
    specs = {}
    for cell in cells:
        for node in ast.walk(ast.parse(source(cell))):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "ExperimentSpec"
            ):
                target = next(target for target in node.targets if isinstance(target, ast.Name))
                specs[target.id] = node.value

    expected = {
        "AER_SPEC": ("AerIdeal", "aer_ideal", False),
        "IQM_SPEC": ("IQMHardware", "iqm_garnet", True),
        "PIASTQ_SPEC": ("PiastQHardware", "piastq", True),
    }
    assert set(specs) == set(expected)
    for name, (backend_name, backend_tag, needs_mitigation) in expected.items():
        keywords = {keyword.arg: keyword.value for keyword in specs[name].keywords}
        assert ast.literal_eval(keywords["state"]) == "ame43"
        assert isinstance(keywords["basis"], ast.Call)
        assert keywords["basis"].func.id == "PathBasis"
        assert keywords["basis"].args[0].id == "CANONICAL_BASIS_DIRECTORY"
        assert keywords["shots"].id == "SHOTS"
        assert keywords["uncertainty"].id == "UNCERTAINTY"
        assert keywords["backend"].func.id == backend_name
        assert ast.literal_eval(keywords["tags"]) == {
            "baseline": "canonical_ez",
            "backend": backend_tag,
        }
        if needs_mitigation:
            assert keywords["mitigation"].id == "HARDWARE_MITIGATION"
        else:
            assert "mitigation" not in keywords


def test_summary_preserves_values_and_explicit_skips():
    namespace = setup_namespace(REPO_ROOT)
    values = {
        "raw": {"estimate": {"real": 8.0, "imag": 0.0}},
        "readout_mitigated": {"estimate": {"real": 7.9, "imag": 0.0}},
        "zne": {"estimate": {"real": 8.1, "imag": 0.0}},
        "zne_readout_mitigated": {"estimate": {"real": 8.0, "imag": 0.0}},
        "diagnostics": {"resamples": 2_000},
        "leakage_rate": 0.01,
    }
    result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        artifact_dir=Path("artifacts") / "ame43-aer",
        values=values,
    )

    summary = namespace["summarize_results"](
        {"aer_ideal": result}, namespace["REFERENCE"]
    )

    assert [row["backend"] for row in summary] == [
        "aer_ideal",
        "iqm_garnet",
        "piastq",
    ]
    assert [row["status"] for row in summary] == [
        "completed",
        "skipped",
        "skipped",
    ]
    for key, value in values.items():
        assert summary[0][key] == value
    assert summary[0]["ideal_bell_value"] == 8.0
    json.dumps(summary)


def test_notebook_has_no_secrets_user_paths_or_low_level_execution():
    full_source = "\n".join(source(cell) for cell in code_cells(load_notebook()))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(rf"(?im)os\.environ(?:\.get)?\([^)]*{credential}", full_source)
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in (
        "PiastQClient",
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


def test_prepare_canonical_basis_creates_exact_idempotent_ame43_bundle(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    assert target == (
        tmp_path
        / "experiment_inputs"
        / "reference_bases"
        / "ame43"
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


def test_validate_canonical_basis_enforces_ame43_state(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    circuit = namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy")
    assert circuit.num_qubits == 8
    assert circuit.num_clbits == 0
    assert not any(item.operation.name == "measure" for item in circuit.data)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "ame43"
    assert metadata["num_qubits"] == 8


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


def test_prepare_canonical_basis_rejects_noncanonical_encoding(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_canonical_basis"](tmp_path)
    numpy = namespace["np"]
    with (target / "E.npy").open("wb") as handle:
        numpy.save(handle, numpy.zeros((4, 3)), allow_pickle=False)
    with pytest.raises(RuntimeError, match="isometry|canonical_ez"):
        namespace["prepare_canonical_basis"](tmp_path)


def test_prepare_canonical_basis_cleans_staging_after_write_failure(
    tmp_path, monkeypatch
):
    namespace = setup_namespace(REPO_ROOT)

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated NPY failure")

    monkeypatch.setattr(namespace["np"], "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated NPY failure"):
        namespace["prepare_canonical_basis"](tmp_path)

    parent = tmp_path / "experiment_inputs" / "reference_bases" / "ame43"
    assert not parent.exists() or not any(
        path.name.startswith(".canonical_ez.tmp-") for path in parent.iterdir()
    )


def test_canonical_notebook_setup_runs_ame43_on_aer(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_canonical_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="ame43",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=4096,
        bootstrap=namespace["BootstrapConfig"](samples=20, seed=7),
        output_root=tmp_path / "runs",
    )

    result = namespace["run_experiment"](spec, repo_root=tmp_path)

    assert result.status.value == "completed"
    raw_real = result.values["raw"]["estimate"]["real"]
    assert raw_real == pytest.approx(8.0, abs=0.05)
    assert raw_real > namespace["REFERENCE"].bell_functional.classical_bound
    artifact_dir = Path(result.artifact_dir)
    assert artifact_dir.is_relative_to(tmp_path / "runs")
    assert (artifact_dir / "experiment.json").is_file()
    assert (artifact_dir / "result.json").is_file()
