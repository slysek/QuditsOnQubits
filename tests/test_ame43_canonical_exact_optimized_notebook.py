import ast
import json
import multiprocessing
import os
import stat
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import qiskit.qpy as qpy
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ame43_canonical_exact_optimized.ipynb"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def calls(cell_source, name):
    return [
        node
        for node in ast.walk(ast.parse(cell_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def setup_namespace(cwd):
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for cell in code_cells(load_notebook()):
            cell_source = source(cell)
            if "canonical-input-materialization" in cell.get("metadata", {}).get("tags", []):
                continue
            if calls(cell_source, "run_experiment") or "display(" in cell_source:
                continue
            exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def sha256_file(path):
    return __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()


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
        qpy.dump(circuit, handle)
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["graph_state_direct_basis.qpy"]["sha256"] = sha256_file(circuit_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_qpy_hash(target):
    circuit_path = target / "graph_state_direct_basis.qpy"
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["graph_state_direct_basis.qpy"]["sha256"] = sha256_file(circuit_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_in_independent_notebook_namespace(notebook_cwd, repo_root, start, outcomes):
    try:
        namespace = setup_namespace(Path(notebook_cwd))
        outcomes.put(("ready", None))
        if not start.wait(timeout=20):
            raise RuntimeError("concurrent preparation start timed out")
        result = namespace["prepare_exact_optimized_basis"](Path(repo_root))
    except BaseException:
        outcomes.put(("error", traceback.format_exc()))
    else:
        outcomes.put(("ok", str(result)))


def test_notebook_exists():
    assert NOTEBOOK_PATH.is_file()


def test_notebook_uses_high_level_runner_with_optimized_configuration():
    notebook = load_notebook()
    cells = code_cells(notebook)
    full_source = "\n".join(source(cell) for cell in cells)
    assert "build_exact_optimized_direct_basis_graph_state_circuit" in full_source
    assert "canonical_ez_exact_optimized" in full_source
    assert "TranspilationConfig(optimization_level=3, seed_transpiler=13)" in full_source
    assert "# Canonical AME(4,3) exact-optimized Bell baseline" in "\n".join(
        source(cell) for cell in notebook["cells"]
    )
    assert len([cell for cell in cells if calls(source(cell), "run_experiment")]) == 3
    assert "RUN_IQM = False" in full_source
    assert "RUN_PIASTQ = False" in full_source
    assert "IQMHardware(device='garnet', use_metrics=True, env_path=IQM_ENV_PATH)" in full_source
    assert "PiastQHardware" in full_source
    for forbidden in ("PiastQClient", "IQMProvider(", ".run(", "token", "api_key"):
        assert forbidden not in full_source
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)


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
        backlink_value = (
            str(worktree / ".git")
            if absolute
            else os.path.relpath(worktree / ".git", worktree_git)
        )
        (worktree_git / "gitdir").write_text(f"{backlink_value}\n", encoding="utf-8")
        fallback_env = owner / ".env"
        fallback_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
        assert resolver(worktree) == fallback_env


def test_iqm_env_path_resolver_rejects_spoofed_worktree_admin_directory(tmp_path):
    resolver = setup_namespace(REPO_ROOT)["resolve_iqm_env_path"]
    owner = tmp_path / "owner-repository"
    original = tmp_path / "original-checkout"
    spoof = tmp_path / "spoof-checkout"
    worktree_git = owner / ".git" / "worktrees" / "external"
    original.mkdir()
    spoof.mkdir()
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    (worktree_git / "gitdir").write_text(
        f"{os.path.relpath(original / '.git', worktree_git)}\n", encoding="utf-8"
    )
    (original / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
    (spoof / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
    (owner / ".env").write_text("IQM_TOKEN=top-secret\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        resolver(spoof)

    assert str(raised.value) == (
        "Cannot validate worktree ownership for IQM .env fallback; "
        "provide checkout-local .env or explicit env_path."
    )
    assert "spoof-checkout" not in str(raised.value)
    assert "top-secret" not in str(raised.value)


def test_iqm_env_path_resolver_rejects_external_fake_admin_directory(tmp_path):
    resolver = setup_namespace(REPO_ROOT)["resolve_iqm_env_path"]
    checkout = tmp_path / "external-checkout"
    fake_admin = tmp_path / "fake-admin-secret"
    other_repo = tmp_path / "other-repository"
    other_git = other_repo / ".git"
    checkout.mkdir()
    fake_admin.mkdir()
    other_git.mkdir(parents=True)
    (checkout / ".git").write_text(f"gitdir: {fake_admin}\n", encoding="utf-8")
    (fake_admin / "gitdir").write_text(f"{checkout / '.git'}\n", encoding="utf-8")
    (fake_admin / "commondir").write_text(f"{other_git}\n", encoding="utf-8")
    (other_repo / ".env").write_text("IQM_TOKEN=top-secret\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        resolver(checkout)

    assert str(raised.value) == (
        "Cannot validate non-bare owning repository for IQM .env fallback; "
        "provide checkout-local .env or explicit env_path."
    )
    assert "fake-admin-secret" not in str(raised.value)
    assert "top-secret" not in str(raised.value)


@pytest.mark.parametrize("backlink_kind", ["missing", "empty", "directory", "symlink", "reparse"])
def test_iqm_env_path_resolver_rejects_invalid_worktree_backlink(
    tmp_path, monkeypatch, backlink_kind
):
    resolver = setup_namespace(REPO_ROOT)["resolve_iqm_env_path"]
    owner = tmp_path / "owner-repository"
    worktree = tmp_path / "external-checkout"
    worktree_git = owner / ".git" / "worktrees" / "external"
    worktree.mkdir()
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
    backlink = worktree_git / "gitdir"
    if backlink_kind == "directory":
        backlink.mkdir()
    elif backlink_kind != "missing":
        backlink.write_text(
            "\n" if backlink_kind == "empty" else f"{worktree / '.git'}\n",
            encoding="utf-8",
        )
    if backlink_kind in {"symlink", "reparse"}:
        real_lstat = Path.lstat

        def simulated_lstat(path):
            if path == backlink:
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK if backlink_kind == "symlink" else stat.S_IFREG,
                    st_file_attributes=0 if backlink_kind == "symlink" else 0x400,
                )
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", simulated_lstat)

    with pytest.raises(RuntimeError) as raised:
        resolver(worktree)

    assert str(raised.value) == (
        "Cannot validate worktree ownership for IQM .env fallback; "
        "provide checkout-local .env or explicit env_path."
    )


def test_iqm_env_path_resolver_allows_local_env_without_git_metadata(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    checkout = tmp_path / "source-archive"
    checkout.mkdir()
    env_path = checkout / ".env"
    env_path.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert namespace["resolve_iqm_env_path"](checkout) == env_path


def test_iqm_env_path_resolver_rejects_non_git_common_directory(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    owner = tmp_path / "owner-repository"
    worktree = tmp_path / "external-checkout"
    worktree_git = owner / ".git" / "worktrees" / "external"
    bare_git = owner / "bare.git"
    worktree.mkdir()
    worktree_git.mkdir(parents=True)
    bare_git.mkdir()
    (worktree_git / "commondir").write_text(f"{bare_git}\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
    (owner / ".env").write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"non-\.git|checkout-local|explicit"):
        namespace["resolve_iqm_env_path"](worktree)


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


def assert_iqm_execution_is_guarded(cell_source):
    tree = ast.parse(cell_source)
    iqm_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "RUN_IQM"
    ]
    assert len(iqm_guards) == 1
    iqm_guard = iqm_guards[0]
    body_nodes = {
        child
        for statement in iqm_guard.body
        for child in ast.walk(statement)
    }
    orelse_nodes = {
        child
        for statement in iqm_guard.orelse
        for child in ast.walk(statement)
    }
    all_nodes = set(ast.walk(tree))

    def calls_in(nodes, name):
        return [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    resolver_calls = calls_in(body_nodes, "resolve_iqm_env_path")
    iqm_calls = calls_in(body_nodes, "IQMHardware")
    run_calls = calls_in(body_nodes, "run_experiment")
    run_assignments = [
        node
        for node in body_nodes
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_experiment"
    ]
    assert len(resolver_calls) == len(iqm_calls) == len(run_calls) == len(run_assignments) == 1
    assert not any(calls_in(orelse_nodes, name) for name in ("resolve_iqm_env_path", "IQMHardware", "run_experiment"))
    outside_nodes = all_nodes - body_nodes
    assert not any(calls_in(outside_nodes, name) for name in ("resolve_iqm_env_path", "IQMHardware", "run_experiment"))
    return iqm_calls


def test_iqm_guard_assertion_rejects_execution_in_else():
    with pytest.raises(AssertionError):
        assert_iqm_execution_is_guarded(
            "if RUN_IQM:\n"
            "    resolve_iqm_env_path(REPO_ROOT)\n"
            "else:\n"
            "    IQMHardware(env_path=IQM_ENV_PATH)\n"
            "    result = run_experiment(spec)\n"
        )


def test_iqm_cell_resolves_env_only_inside_guard_and_passes_explicit_path():
    cells = code_cells(load_notebook())
    cell = next(
        cell
        for cell in cells
        if "IQMHardware" in source(cell) and "RUN_IQM" in source(cell)
    )
    iqm_calls = assert_iqm_execution_is_guarded(source(cell))
    env_path = next(keyword for keyword in iqm_calls[0].keywords if keyword.arg == "env_path")
    assert isinstance(env_path.value, ast.Name)
    assert env_path.value.id == "IQM_ENV_PATH"


def test_exact_optimized_bundle_is_idempotent_and_state_equivalent(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    assert target == tmp_path / "experiment_inputs" / "reference_bases" / "ame43" / "canonical_ez_exact_optimized"
    first_hashes = {path.name: sha256_file(path) for path in target.iterdir()}
    assert namespace["prepare_exact_optimized_basis"](tmp_path) == target
    assert {path.name: sha256_file(path) for path in target.iterdir()} == first_hashes

    circuit = namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy")
    encoding = namespace["get_encoding"]("canonical_ez").as_array()
    expected = build_direct_basis_graph_state_circuit("ame43", encoding)
    assert circuit.num_qubits == 8
    assert circuit.num_clbits == 0
    assert circuit.count_ops().get("diagonal") == 4
    assert circuit.depth() == 3
    assert Statevector.from_instruction(circuit).equiv(Statevector.from_instruction(expected))
    assert json.loads((target / "metadata.json").read_text(encoding="utf-8"))["encoding_id"] == "canonical_ez"


def test_exact_optimized_bundle_rebuilds_state_equivalent_reordered_edges(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    circuit_path = target / "graph_state_direct_basis.qpy"
    circuit = namespace["load_single_circuit"](circuit_path)
    data = list(circuit.data)
    circuit.data = [*data[:4], data[4], data[6], data[5], data[7]]
    assert circuit.depth() == 5
    with circuit_path.open("wb") as handle:
        qpy.dump(circuit, handle)
    update_qpy_hash(target)

    rebuilt = namespace["prepare_exact_optimized_basis"](tmp_path)
    assert namespace["load_single_circuit"](rebuilt / "graph_state_direct_basis.qpy").depth() == 3


@pytest.mark.parametrize(
    ("mode", "file_attributes"),
    [(stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)],
)
def test_exact_optimized_bundle_rebuilds_legacy_format_and_rejects_reparse_cache(
    tmp_path, monkeypatch, mode, file_attributes
):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    replace_edge_with_legacy_unitary(namespace, target)
    rebuilt = namespace["prepare_exact_optimized_basis"](tmp_path)
    assert isinstance(namespace["load_single_circuit"](rebuilt / "graph_state_direct_basis.qpy").data[-1].operation, DiagonalGate)

    original_files = {path.name: path.read_bytes() for path in target.iterdir()}
    real_lstat = Path.lstat

    def simulated_lstat(path):
        if path == target:
            return SimpleNamespace(st_mode=mode, st_file_attributes=file_attributes)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        namespace["prepare_exact_optimized_basis"](tmp_path)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original_files


def test_exact_optimized_bundle_normalizes_nonnumeric_encoding_error(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    encoding_path = target / "E.npy"
    with encoding_path.open("wb") as handle:
        namespace["np"].save(handle, namespace["np"].full((4, 3), "x"), allow_pickle=False)

    with pytest.raises(RuntimeError, match="encoding"):
        namespace["prepare_exact_optimized_basis"](tmp_path)


def test_exact_optimized_bundle_cleans_staging_after_write_failure(tmp_path, monkeypatch):
    namespace = setup_namespace(REPO_ROOT)

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated NPY failure")

    monkeypatch.setattr(namespace["np"], "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated NPY failure"):
        namespace["prepare_exact_optimized_basis"](tmp_path)

    parent = tmp_path / "experiment_inputs" / "reference_bases" / "ame43"
    assert not parent.exists() or not any(
        path.name.startswith(".canonical_ez_exact_optimized.tmp-") for path in parent.iterdir()
    )


@pytest.mark.parametrize(
    ("mode", "file_attributes"),
    [(stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)],
)
def test_exact_optimized_bundle_rejects_unsafe_existing_ancestor(
    tmp_path, monkeypatch, mode, file_attributes
):
    namespace = setup_namespace(REPO_ROOT)
    ancestor = tmp_path / "experiment_inputs"
    ancestor.mkdir()
    real_lstat = Path.lstat

    def simulated_lstat(path):
        if path == ancestor:
            return SimpleNamespace(st_mode=mode, st_file_attributes=file_attributes)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        namespace["prepare_exact_optimized_basis"](tmp_path)


def test_exact_optimized_bundle_serializes_concurrent_stale_rebuilds(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    replace_edge_with_legacy_unitary(namespace, target)

    with ThreadPoolExecutor(max_workers=2) as executor:
        returned = list(executor.map(lambda _: namespace["prepare_exact_optimized_basis"](tmp_path), range(2)))

    assert returned == [target, target]
    assert namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy").depth() == 3
    assert not any(
        path.name.startswith((".canonical_ez_exact_optimized.tmp-", ".canonical_ez_exact_optimized.legacy-"))
        for path in target.parent.iterdir()
    )


def test_exact_optimized_bundle_serializes_independent_process_stale_rebuilds(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    target = namespace["prepare_exact_optimized_basis"](tmp_path)
    replace_edge_with_legacy_unitary(namespace, target)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=prepare_in_independent_notebook_namespace,
            args=(str(REPO_ROOT), str(tmp_path), start, outcomes),
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        ready = [outcomes.get(timeout=20) for _ in workers]
        assert ready == [("ready", None), ("ready", None)]
        start.set()
        for worker in workers:
            worker.join(timeout=30)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        results = [outcomes.get(timeout=10) for _ in workers]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)

    assert results == [("ok", str(target)), ("ok", str(target))]
    assert namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy").depth() == 3
    assert (target.parent / ".canonical_ez_exact_optimized.lock").is_file()
    assert not any(
        path.name.startswith((".canonical_ez_exact_optimized.tmp-", ".canonical_ez_exact_optimized.legacy-"))
        for path in target.parent.iterdir()
    )


def test_exact_optimized_bundle_serializes_independent_process_first_materialization(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=prepare_in_independent_notebook_namespace,
            args=(str(REPO_ROOT), str(tmp_path), start, outcomes),
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        ready = [outcomes.get(timeout=20) for _ in workers]
        assert ready == [("ready", None), ("ready", None)]
        start.set()
        for worker in workers:
            worker.join(timeout=30)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        results = [outcomes.get(timeout=10) for _ in workers]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)

    target = tmp_path / "experiment_inputs" / "reference_bases" / "ame43" / "canonical_ez_exact_optimized"
    assert results == [("ok", str(target)), ("ok", str(target))]
    namespace = setup_namespace(REPO_ROOT)
    assert namespace["load_single_circuit"](target / "graph_state_direct_basis.qpy").depth() == 3
    assert not any(
        path.name.startswith((".canonical_ez_exact_optimized.tmp-", ".canonical_ez_exact_optimized.legacy-"))
        for path in target.parent.iterdir()
    )


def test_exact_optimized_notebook_aer_smoke(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    basis = namespace["prepare_exact_optimized_basis"](tmp_path)
    spec = namespace["ExperimentSpec"](
        state="ame43",
        basis=namespace["PathBasis"](basis),
        backend=namespace["AerIdeal"](seed_simulator=11),
        shots=512,
        bootstrap=namespace["BootstrapConfig"](samples=20, seed=7),
        transpilation=namespace["TranspilationConfig"](optimization_level=3, seed_transpiler=0),
        output_root=tmp_path / "runs",
    )
    result = namespace["run_experiment"](spec, repo_root=tmp_path)
    assert result.status.value == "completed"
    assert result.values["raw"]["estimate"]["real"] == pytest.approx(8.0, abs=0.2)
