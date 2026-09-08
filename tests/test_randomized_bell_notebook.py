"""The raw Bell example is portable, guarded, and executable with local Aer."""

import ast
import json
from pathlib import Path
import random

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "bell_randomized_raw.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def code(tag):
    return "".join(next(cell["source"] for cell in cells() if tag in cell.get("metadata", {}).get("tags", [])))


def namespace(tmp_path):
    values = {"__name__": "__notebook_test__"}
    exec(compile(code("configuration"), str(NOTEBOOK), "exec"), values)
    values["WORK_ROOT"] = tmp_path
    exec(compile(code("basis-helper"), str(NOTEBOOK), "exec"), values)
    return values


def test_notebook_is_output_free_and_hardware_runs_are_guarded():
    notebook_cells = cells()
    source = "\n".join("".join(cell["source"]) for cell in notebook_cells)
    assert "C:/Users/" not in source and "C:\\Users\\" not in source
    for cell in notebook_cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    setup = ast.parse(code("configuration"))
    assignments = {node.targets[0].id: node.value for node in setup.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    for flag, tag in [("RUN_IBM", "ibm"), ("RUN_IQM", "iqm"), ("RUN_PIASTQ", "piastq")]:
        assert ast.literal_eval(assignments[flag]) is False
        tree = ast.parse(code(tag))
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
        assert isinstance(tree.body[0].test, ast.Name) and tree.body[0].test.id == flag
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert sum(node.func.id == "run_experiment" for node in calls) == 1
    assert "conditional_schedule_block_hoeffding_v1" in source
    assert "no_accepted_shots" in source
    assert "resume_experiment" in code("resume")
    assert "_randbelow" not in source


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
@pytest.mark.parametrize("encoding", ["canonical", "permuted"])
@pytest.mark.parametrize("preparation", ["reference", "product"])
def test_portable_basis_helper_reconstructs_encoded_state(tmp_path, state, encoding, preparation):
    values = namespace(tmp_path)
    values["STATE_PREPARATION"] = preparation
    directory = values["prepare_basis"](state, encoding)
    with (directory / "graph_state_direct_basis.qpy").open("rb") as handle:
        circuit = qpy.load(handle)[0]
    reference = values["get_reference_experiment"](state)
    e = np.load(directory / "E.npy")
    transform = np.ones((1, 1), complex)
    for _ in range(reference.state.num_parties):
        transform = np.kron(transform, e)
    logical = reference.state.statevector() if preparation == "reference" else np.eye(3 ** reference.state.num_parties)[:, 0]
    np.testing.assert_allclose(Statevector.from_instruction(circuit).data, transform @ logical, atol=1e-10, rtol=0)


def test_notebook_executes_aer_comparisons_and_resumes_offline(tmp_path, monkeypatch):
    pytest.importorskip("qiskit_aer")
    from qudits_on_qubits.experiments import block_runner
    from qudits_on_qubits.experiments.measurement import RandomizedBlocks
    original_generate = block_runner.generate_schedule
    generator = random.Random(294)
    monkeypatch.setattr(block_runner, "generate_schedule", lambda reference, config, **kwargs: original_generate(reference, config, _randbelow=generator.randrange, _source="test_seeded", _seed=294))
    values = namespace(tmp_path)
    values["MEASUREMENT"] = RandomizedBlocks(12, 4, 512, 100)
    exec(compile(code("local-aer"), str(NOTEBOOK), "exec"), values)
    assert len(values["RESULTS"]) == 6
    for result in values["RESULTS"].values():
        assert result.status.value == "completed", result.values
        assert result.values["raw"] is not None
        assert result.values["budget"]["completed_shots"] == result.values["budget"]["scheduled_blocks"] * 4
    schedules = [json.loads((result.artifact_dir / "schedule.json").read_text(encoding="utf-8")) for result in values["RESULTS"].values()]
    assert schedules[0]["blocks"] != schedules[1]["blocks"]
    # Hardware guards must continue to protect execution even when rerun.
    def forbidden(*args, **kwargs):
        pytest.fail("hardware cell or offline resume attempted a new run")
    values["run_experiment"] = forbidden
    for tag in ("ibm", "iqm", "piastq"):
        exec(compile(code(tag), str(NOTEBOOK), "exec"), values)
    monkeypatch.setattr(block_runner, "create_backend_adapter", forbidden)
    exec(compile(code("report"), str(NOTEBOOK), "exec"), values)
    exec(compile(code("resume"), str(NOTEBOOK), "exec"), values)
    assert values["RESTORED"].status.value == "completed"
    assert len(values["SUMMARY"]) == 6
    assert all(row["zero_contribution_blocks"] >= 0 for row in values["SUMMARY"])
