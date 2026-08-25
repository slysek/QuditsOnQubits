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
    checkout_env = checkout / ".env"
    checkout_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert resolver(checkout) == checkout_env

    worktree_parent = tmp_path / ".worktrees"
    worktree = worktree_parent / "feature"
    worktree.mkdir(parents=True)
    fallback_env = tmp_path / ".env"
    fallback_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert resolver(worktree) == fallback_env


def test_iqm_env_path_resolver_reports_missing_iqm_env(tmp_path):
    namespace = setup_namespace(REPO_ROOT)
    with pytest.raises(RuntimeError, match=r"IQM .*\.env"):
        namespace["resolve_iqm_env_path"](tmp_path / "checkout")


def test_iqm_cell_resolves_env_only_inside_guard_and_passes_explicit_path():
    cells = code_cells(load_notebook())
    cell = next(
        cell
        for cell in cells
        if "IQMHardware" in source(cell) and "RUN_IQM" in source(cell)
    )
    tree = ast.parse(source(cell))
    iqm_guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "RUN_IQM"
    )
    resolver_calls = [
        node
        for node in ast.walk(iqm_guard)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_iqm_env_path"
    ]
    assert len(resolver_calls) == 1
    iqm_calls = calls(source(cell), "IQMHardware")
    assert len(iqm_calls) == 1
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
