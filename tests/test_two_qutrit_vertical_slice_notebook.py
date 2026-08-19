"""Executable contract for the public two-qutrit Bell example notebook."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "two_qutrit_bell_vertical_slice.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_is_clean_public_example() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
    assert len(set(cell_ids)) == len(cell_ids)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) >= 4
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    source = "\n".join(
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else cell["source"]
        for cell in code_cells
    )
    for public_name in (
        "BellReferenceCircuitSpec",
        "canonical_qutrit_encoding",
        "QuditExperimentSpec",
        "run_vertical_slice",
        "load_run_manifest",
    ):
        assert public_name in source


def test_notebook_code_cells_run_end_to_end(tmp_path, monkeypatch) -> None:
    notebook = _load_notebook()
    monkeypatch.setenv("QOQ_NOTEBOOK_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("QOQ_NOTEBOOK_SHOTS", "256")

    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = (
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)

    summary = namespace["summary"]
    manifest_summary = namespace["manifest_summary"]
    assert summary["status"] == "completed"
    assert summary["benchmark"] == "two_qutrit"
    assert summary["encoded_circuit_count"] == 9
    assert abs(summary["bell_unconditional"] - 6.0) < 0.3
    assert summary["leakage_rate"] == 0.0
    assert manifest_summary["schema_version"] == "run-manifest-v1"
    assert manifest_summary["artifact_count"] == 7
    assert Path(manifest_summary["path"]).is_file()
