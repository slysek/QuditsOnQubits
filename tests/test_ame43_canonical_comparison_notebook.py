import ast
import json
import os
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit.quantum_info import Statevector


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ame43_canonical_comparison.ipynb"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def setup_namespace(cwd=REPO_ROOT):
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        for cell in code_cells(load_notebook()):
            tags = cell.get("metadata", {}).get("tags", [])
            if set(tags) & {
                "comparison-materialization",
                "offline-execution",
                "comparison-display",
                "iqm-compile",
                "iqm-hardware",
            }:
                continue
            exec(compile(source(cell), str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def execute_default_notebook(tmp_path):
    namespace = {"__name__": "__notebook_test__", "__file__": str(NOTEBOOK_PATH)}
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        for cell in code_cells(load_notebook()):
            exec(compile(source(cell), str(NOTEBOOK_PATH), "exec"), namespace)
            if "repo-setup" in cell.get("metadata", {}).get("tags", []):
                namespace["REPO_ROOT"] = tmp_path
    finally:
        os.chdir(previous_cwd)
    return namespace


def test_notebook_exists():
    assert NOTEBOOK_PATH.is_file()


def test_notebook_uses_project_kernelspec():
    kernelspec = load_notebook()["metadata"]["kernelspec"]
    assert kernelspec == {
        "display_name": "QuditsOnQubitsEnv",
        "language": "python",
        "name": "quditsonqubitsenv",
    }


def test_notebook_is_unexecuted_offline_first_and_uses_existing_builders():
    notebook = load_notebook()
    cells = code_cells(notebook)
    full_source = "\n".join(source(cell) for cell in cells)
    all_source = "\n".join(source(cell) for cell in notebook["cells"])

    assert "# Canonical AME(4,3): baseline vs exact-optimized" in all_source
    assert "build_direct_basis_graph_state_circuit" in full_source
    assert "build_exact_optimized_direct_basis_graph_state_circuit" in full_source
    assert "RUN_IQM_COMPILE = False" in full_source
    assert "RUN_IQM_HARDWARE = False" in full_source
    assert "HARDWARE_SHOTS = 50" in full_source
    assert "IQM_SEED = 13" in full_source
    assert "OFFLINE_SHOTS = 512" in full_source
    assert "basis_gates=['u', 'cz']" in full_source
    assert "seed_transpiler=0" in full_source
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)


def test_notebook_has_no_secrets_absolute_user_paths_or_low_level_execution():
    full_source = "\n".join(source(cell) for cell in code_cells(load_notebook()))
    credential = r"(?:token|api[_ -]?key|password|credentials?|client_secret|secret)"
    assert not re.search(rf"(?im)^\s*\w*{credential}\w*\s*=", full_source)
    assert not re.search(rf"(?im)os\.environ(?:\.get)?\([^)]*{credential}", full_source)
    assert not re.search(r"(?i)\\Users\\|[A-Za-z]:[\\/]", full_source)
    for forbidden in ("IQMProvider(", "IQMClient(", "backend.run(", ".submit("):
        assert forbidden not in full_source


def test_iqm_optional_work_is_separately_and_strictly_guarded():
    cells = code_cells(load_notebook())
    compile_cell = next(cell for cell in cells if "iqm-compile" in cell.get("metadata", {}).get("tags", []))
    hardware_cell = next(cell for cell in cells if "iqm-hardware" in cell.get("metadata", {}).get("tags", []))

    def guarded_body(cell, flag):
        tree = ast.parse(source(cell))
        guard = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == flag
        )
        body = {child for statement in guard.body for child in ast.walk(statement)}
        outside = set(ast.walk(tree)) - body
        return body, outside

    compile_body, compile_outside = guarded_body(compile_cell, "RUN_IQM_COMPILE")
    hardware_body, hardware_outside = guarded_body(hardware_cell, "RUN_IQM_HARDWARE")

    def call_names(nodes):
        return {
            node.func.id
            for node in nodes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    assert {"resolve_iqm_env_path", "IQMHardware", "create_backend_adapter"} <= call_names(compile_body)
    assert "resolve_iqm_env_path" not in call_names(compile_outside)
    assert "create_backend_adapter" not in call_names(compile_outside)
    assert {"resolve_iqm_env_path", "IQMHardware", "run_experiment"} <= call_names(hardware_body)
    assert "resolve_iqm_env_path" not in call_names(hardware_outside)
    assert "run_experiment" not in call_names(hardware_outside)
    assert source(hardware_cell).count("run_experiment(") == 2
    assert "mitigation=RAW_HARDWARE_MITIGATION" in source(hardware_cell)
    assert "env_path=IQM_ENV_PATH" in source(hardware_cell)


def test_iqm_env_path_resolver_prefers_checkout_and_supports_linked_worktrees(tmp_path):
    resolver = setup_namespace()["resolve_iqm_env_path"]

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout_env = checkout / ".env"
    checkout_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert resolver(checkout) == checkout_env.resolve()

    owner = tmp_path / "owner"
    worktree = tmp_path / "worktree"
    gitdir = owner / ".git" / "worktrees" / "comparison"
    gitdir.mkdir(parents=True)
    worktree.mkdir()
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (worktree / ".git").write_text(
        f"gitdir: {os.path.relpath(gitdir, worktree)}\n", encoding="utf-8"
    )
    owner_env = owner / ".env"
    owner_env.write_text("IQM_TOKEN=not-read\n", encoding="utf-8")
    assert resolver(worktree) == owner_env.resolve()


def test_iqm_env_path_resolver_rejects_unrelated_directory_named_git(tmp_path):
    resolver = setup_namespace()["resolve_iqm_env_path"]

    owner = tmp_path / "owner"
    worktree = tmp_path / "worktree"
    gitdir = owner / ".git" / "worktrees" / "comparison"
    unrelated_git = tmp_path / "unrelated" / ".git"
    gitdir.mkdir(parents=True)
    unrelated_git.mkdir(parents=True)
    worktree.mkdir()
    (gitdir / "commondir").write_text(str(unrelated_git), encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    (unrelated_git.parent / ".env").write_text("IQM_TOKEN=not-read\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="owning repository|Git common directory"):
        resolver(worktree)


def test_materializes_distinct_valid_state_equivalent_bundles(tmp_path):
    namespace = setup_namespace()
    bundles = namespace["materialize_comparison_bundles"](tmp_path)

    assert set(bundles) == {"baseline", "exact_optimized"}
    assert bundles["baseline"] != bundles["exact_optimized"]
    assert all(path.is_relative_to(tmp_path) for path in bundles.values())
    for path in bundles.values():
        assert {item.name for item in path.iterdir()} == {
            "graph_state_direct_basis.qpy",
            "E.npy",
            "metadata.json",
        }

    baseline = namespace["load_single_circuit"](
        bundles["baseline"] / "graph_state_direct_basis.qpy"
    )
    optimized = namespace["load_single_circuit"](
        bundles["exact_optimized"] / "graph_state_direct_basis.qpy"
    )
    assert Statevector.from_instruction(baseline).equiv(Statevector.from_instruction(optimized))
    assert namespace["state_fidelity"](baseline, optimized) == pytest.approx(1.0, abs=1e-12)


def test_bundle_validation_keeps_safety_boundary_at_supplied_repository_root(
    tmp_path, monkeypatch
):
    namespace = setup_namespace()
    bundles = namespace["materialize_comparison_bundles"](tmp_path)
    encoding = namespace["get_encoding"]("canonical_ez").as_array()
    circuit = namespace["build_comparison_circuits"]()["baseline"]
    real_assertion = namespace["assert_safe_path_components"]
    checked_roots = []

    def recording_assertion(root, path):
        checked_roots.append(Path(root))
        return real_assertion(root, path)

    monkeypatch.setitem(namespace, "assert_safe_path_components", recording_assertion)
    namespace["validate_comparison_bundle"](
        bundles["baseline"], "baseline", encoding, circuit
    )

    assert checked_roots == [tmp_path]


def test_materialization_rejects_unsafe_existing_ancestor(tmp_path, monkeypatch):
    namespace = setup_namespace()
    ancestor = tmp_path / "experiment_inputs"
    ancestor.mkdir()
    real_lstat = Path.lstat

    def simulated_lstat(path):
        if path == ancestor:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        namespace["materialize_comparison_bundles"](tmp_path)


def test_optimized_preparation_and_measurement_metrics_are_comparable(tmp_path):
    namespace = setup_namespace()
    bundles = namespace["materialize_comparison_bundles"](tmp_path)
    circuits = namespace["build_comparison_circuits"]()
    metrics = namespace["collect_circuit_metrics"](bundles, circuits)

    assert set(metrics) == {"baseline", "exact_optimized"}
    assert all(row["measurement_circuit_count"] == 13 for row in metrics.values())
    assert metrics["exact_optimized"]["preparation_cz"] < metrics["baseline"]["preparation_cz"]
    assert metrics["exact_optimized"]["preparation_depth"] < metrics["baseline"]["preparation_depth"]
    for row in metrics.values():
        assert row["measurement_cz_min"] <= row["measurement_cz_max"]
        assert row["measurement_depth_min"] <= row["measurement_depth_max"]


def test_default_run_all_is_offline_and_both_aer_values_are_ideal(tmp_path):
    namespace = execute_default_notebook(tmp_path)

    assert namespace["RUN_IQM_COMPILE"] is False
    assert namespace["RUN_IQM_HARDWARE"] is False
    assert namespace["STATE_FIDELITY"] == pytest.approx(1.0, abs=1e-12)
    assert len(namespace["COMPARISON_TABLE"]) == 2
    assert set(namespace["AER_RESULTS"]) == {"baseline", "exact_optimized"}
    for result in namespace["AER_RESULTS"].values():
        assert result.status.value == "completed"
        assert result.values["raw"]["estimate"]["real"] == pytest.approx(8.0, abs=0.3)
