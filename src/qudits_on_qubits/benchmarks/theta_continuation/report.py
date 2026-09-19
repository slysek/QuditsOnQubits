"""Rebuild reproducible tables and scientific figures from validated run bundles."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from qiskit.circuit.library import UGate

from .artifacts import RunStore, fingerprint
from .models import ThetaBenchmarkConfig


_POINT_COLUMNS = (
    "index", "theta", "theta_over_pi", "template_id", "warm_start_parent_id", "status", "correct",
    "continuation_valid", "fallback_used", "fallback_reason", "selected_method", "within_baseline_budget",
    "baseline_n_cz", "cz3_n_cz", "cz3_depth", "cz3_n_1q", "cz3_E_norm", "cz3_L_norm",
    "f3_alpha", "f3_n_cz", "f3_alpha_zero_n_cz", "f3_E_norm", "f3_L_norm",
    "continuation_seconds", "fallback_seconds", "errors",
)
_PARAMETER_COLUMNS = (
    "index", "theta", "theta_over_pi", "template_id", "parameter_index", "instruction_index", "qubit", "angle",
    "continuation_valid", "raw_radians", "unwrapped_radians",
)
_DISTANCE_COLUMNS = ("index", "theta", "theta_over_pi", "template_id", "instruction_index", "qubit", "distance")


def local_u3_distance(left, right) -> float:
    """Return min_gamma ||U(left) - exp(i gamma) U(right)||_F.

    Alignment uses one phase for the whole local operator. It respects U3
    periodicities and Euler-angle gauge equivalences; no angle unwrapping is
    involved. At zero overlap any phase is a minimizer.
    """
    matrices = []
    for values in (left, right):
        angles = np.asarray(values)
        if angles.shape != (3,) or angles.dtype.kind not in "iuf" or not np.isfinite(angles).all():
            raise ValueError("U3 angles must contain three finite real numbers")
        matrices.append(UGate(*angles.astype(float)).to_matrix())
    overlap = np.vdot(matrices[1], matrices[0])
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
    return float(np.linalg.norm(matrices[0] - phase * matrices[1], ord="fro"))


def _output_path(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError) as exc:
        raise ValueError(f"Report output escapes run directory: {relative}") from exc
    return path


def _write_text(root: Path, relative: str, content: str) -> None:
    path = _output_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _flatten(row: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, name))
        else:
            flattened[name] = value
    return flattened


def _write_csv(root: Path, name: str, rows: list[dict], first_columns=()) -> None:
    columns = list(dict.fromkeys([*first_columns, *(key for row in rows for key in row)]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (list, dict, tuple)):
                value = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
            elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
                value = None
            clean[key] = value
        writer.writerow(clean)
    _write_text(root, name, buffer.getvalue())


def _template_layout(document: dict) -> list[dict]:
    metadata = document.get("parameter_metadata")
    if not isinstance(metadata, list) or len(metadata) % 3:
        raise ValueError("Template parameter_metadata must contain complete U3 triples")
    for index, item in enumerate(metadata):
        if (not isinstance(item, dict) or item.get("index") != index
                or item.get("angle") != ("theta", "phi", "lambda")[index % 3]
                or not isinstance(item.get("instruction_index"), int)
                or not isinstance(item.get("qubit"), int)):
            raise ValueError("Invalid template parameter_metadata")
        first = metadata[index - index % 3]
        if item["instruction_index"] != first["instruction_index"] or item["qubit"] != first["qubit"]:
            raise ValueError("U3 parameter triples must belong to one gate")
    return metadata


def _unwrap_segments(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for column in range(values.shape[1]):
        valid = np.flatnonzero(np.isfinite(values[:, column]))
        for segment in np.split(valid, np.flatnonzero(np.diff(valid) != 1) + 1):
            if segment.size:
                result[segment, column] = np.unwrap(values[segment, column])
    return result


def _parameter_data(grid: np.ndarray, bundles: list[dict], baseline: dict) -> tuple[list[dict], list[dict], dict]:
    documents = [baseline["documents"]["template.json"]]
    documents.extend(bundle["documents"]["template.json"] for bundle in bundles if "template.json" in bundle["documents"])
    templates = {}
    for document in documents:
        template_id = document.get("template_id")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("Template must have a nonempty template_id")
        layout = _template_layout(document)
        if template_id in templates and templates[template_id]["layout"] != layout:
            raise ValueError("Conflicting parameter layouts for one template_id")
        templates[template_id] = {"layout": layout, "raw": np.full((len(grid), len(layout)), np.nan)}
    for bundle in bundles:
        metadata = bundle["metadata"]
        if not metadata.get("continuation_valid", False):
            continue
        template_id = metadata.get("template_id")
        if template_id not in templates:
            raise ValueError("Successful continuation has no matching template description")
        parameters = bundle["arrays"].get("cz3_parameters.npy")
        count = len(templates[template_id]["layout"])
        if (parameters is None or parameters.shape != (count,) or parameters.dtype.kind not in "iuf"
                or not np.isfinite(parameters).all()):
            raise ValueError("Successful continuation requires valid cz3_parameters.npy")
        templates[template_id]["raw"][metadata["index"]] = parameters
    parameter_rows, distance_rows = [], []
    for template_id, data in templates.items():
        raw = data["raw"]
        data["unwrapped"] = _unwrap_segments(raw)
        distances = np.full((len(grid), len(data["layout"]) // 3), np.nan)
        for index, theta in enumerate(grid):
            common = {"index": index, "theta": float(theta), "theta_over_pi": float(theta / np.pi), "template_id": template_id}
            for column, item in enumerate(data["layout"]):
                parameter_rows.append({**common, "parameter_index": column,
                    **{key: item[key] for key in ("instruction_index", "qubit", "angle")},
                    "continuation_valid": bool(np.isfinite(raw[index, column])),
                    "raw_radians": raw[index, column], "unwrapped_radians": data["unwrapped"][index, column]})
            for gate, item in enumerate(data["layout"][::3]):
                local = raw[:, gate*3:gate*3+3]
                if index > 0 and np.isfinite(local[index-1:index+1]).all():
                    distances[index, gate] = local_u3_distance(local[index-1], local[index])
                distance_rows.append({**common, "instruction_index": item["instruction_index"],
                                      "qubit": item["qubit"], "distance": distances[index, gate]})
        data["distances"] = distances
    return parameter_rows, distance_rows, templates


def _number(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else np.nan


def _figures(root: Path, grid: np.ndarray, rows: list[dict], templates: dict,
             config: ThetaBenchmarkConfig) -> list[tuple[str, str]]:
    # Object-oriented Agg avoids changing an interactive user's global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figures = []

    def new(rows=1, columns=1, width=10, height=4):
        figure = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(figure)
        return figure, figure.subplots(rows, columns, squeeze=False)

    def save(figure, name, caption):
        for suffix in ("png", "pdf"):
            path = _output_path(root, f"figures/{name}.{suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=180, metadata={"Creator": "QuditsOnQubits theta benchmark"} if suffix == "pdf" else None)
        figures.append((name, caption))
        figure.clear()

    x = grid / np.pi
    for template_index, (template_id, data) in enumerate(templates.items()):
        gate_count = len(data["layout"]) // 3
        if not gate_count:
            continue
        columns = min(3, gate_count)
        nrows = math.ceil(gate_count / columns)
        for representation, label in (("raw", "Raw angles"), ("unwrapped", "Angles unwrapped in separate segments")):
            figure, axes = new(nrows, columns, width=12, height=max(3.2, 2.7*nrows))
            for gate, axis in enumerate(axes.flat):
                if gate >= gate_count:
                    axis.set_visible(False)
                    continue
                item = data["layout"][gate * 3]
                for angle, name in enumerate(("theta", "phi", "lambda")):
                    axis.plot(x, data[representation][:, gate*3+angle], marker=".", label=name, linewidth=1)
                axis.set(title=f"Instruction {item['instruction_index']}, qubit {item['qubit']}", xlabel=r"$\theta/\pi$", ylabel="angle [rad]")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=7)
            figure.suptitle(f"{label}; template_id={template_id[:16]}")
            save(figure, f"parameters_{representation}_template_{template_index:02d}",
                 f"{label}; template `{template_id}`. Gaps indicate missing valid parameters for the current point.")
        figure, axes = new(width=11, height=4.5)
        axis = axes[0, 0]
        for gate, item in enumerate(data["layout"][::3]):
            axis.plot(x, data["distances"][:, gate], marker=".", linewidth=1,
                      label=f"instr. {item['instruction_index']}, q{item['qubit']}")
        axis.set(xlabel=r"$\theta/\pi$", ylabel="min. Frobenius distance", title="Local U3 operators: distance from the adjacent valid point")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=min(4, gate_count))
        save(figure, f"operator_distances_template_{template_index:02d}",
             "Local U3 distances after aligning a single global phase; comparisons only between adjacent valid points of the same template.")

    by_index = {row["index"]: row for row in rows}

    def series(key, *, correct_only=False):
        return np.array([_number(by_index.get(index, {}), key)
                         if not correct_only or by_index.get(index, {}).get("correct") is True else np.nan
                         for index in range(len(grid))])

    fallback = [index for index, row in by_index.items() if row.get("fallback_used")]
    failed = [index for index, row in by_index.items() if row.get("correct") is not True]
    figure, axes = new(2, 2, width=12, height=8)
    axes[0, 0].plot(x, series("f3_n_cz"), ".-", label="F3, selected alpha")
    axes[0, 0].plot(x, series("f3_alpha_zero_n_cz"), ".--", label="F3, alpha = 0")
    axes[0, 0].plot(x, series("cz3_n_cz", correct_only=True), ".-", label="CZ3, selected valid circuit")
    axes[0, 0].plot(x, series("baseline_n_cz"), "--", color="gray", label="baseline after transpilation")
    if fallback:
        axes[0, 0].scatter(x[fallback], series("cz3_n_cz", correct_only=True)[fallback], marker="D", facecolors="none", edgecolors="black", label="fallback")
    axes[0, 0].set(ylabel="CZ count", title="Costs under common transpilation settings")
    axes[0, 1].plot(x, series("f3_alpha"), ".-")
    axes[0, 1].set(ylabel="alpha [rad, modulo 2pi]", title="Selected F3 leakage phase")
    for key in ("f3_E_norm", "f3_L_norm", "cz3_E_norm", "cz3_L_norm"):
        axes[1, 0].plot(x, series(key), ".-", label=key)
    axes[1, 0].axhline(config.f3_tolerance, color="gray", linestyle=":", label="F3 tolerance")
    axes[1, 0].axhline(config.cz3_tolerance, color="gray", linestyle="--", label="CZ3 tolerance")
    axes[1, 0].set_yscale("symlog", linthresh=min(config.f3_tolerance, config.cz3_tolerance)*0.01)
    axes[1, 0].set(ylabel="Frobenius norm", title="Action error and leakage")
    axes[1, 1].plot(x, series("continuation_seconds"), ".-", label="continuation")
    axes[1, 1].plot(x, series("fallback_seconds"), ".-", label="fallback")
    axes[1, 1].set(ylabel="time [s]", title="CZ3 solution time")
    for axis in axes.flat:
        for index in failed:
            axis.axvline(x[index], color="firebrick", alpha=0.3, linewidth=1)
        axis.set_xlabel(r"$\theta/\pi$")
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    save(figure, "metrics", "Costs, leakage phase, error norms and time. Red lines indicate failed points; diamonds indicate fallback use.")
    return figures


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    return "\n".join([
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(_format(row.get(key)) for key, _ in columns) + " |" for row in rows),
    ])


def _markdown(store: RunStore, config: ThetaBenchmarkConfig, baseline: dict, rows: list[dict],
              circuit_rows: list[dict], figures: list[tuple[str, str]]) -> str:
    manifest = store.manifest
    correct = sum(row.get("correct") is True for row in rows)
    failed = len(rows) - correct
    above = sum(row.get("correct") is True and row.get("within_baseline_budget") is False for row in rows)
    fallback = sum(bool(row.get("fallback_used")) for row in rows)
    continuation = sum(row.get("continuation_valid") is True for row in rows)
    pending = len(config.grid()) - len(rows)
    template = baseline["documents"]["template.json"]
    original_cz = sum(operation.get("name") == "cz" for operation in template.get("operations", []))
    original_u3 = len(template["parameter_metadata"]) // 3
    baseline_cost = next((row.get("baseline_n_cz") for row in rows if row.get("baseline_n_cz") is not None),
                         baseline["metadata"].get("baseline_n_cz"))
    provenance = manifest.get("provenance", {})
    lines = [
        "# E_theta benchmark: F3 and CZ3 continuation", "",
        f"Run: `{store.root.name}`. Fingerprint: `{manifest['fingerprint']}`.", "",
        f"Saved points: {len(rows)} / {len(config.grid())}. Valid points: {correct}. Failed points: {failed}. "
        f"Valid above baseline: {above}. Points with fallback: {fallback}. "
        f"Valid continuation template: {continuation}. Points without completed artifacts: {pending}.", "",
        "A point is valid only when F3 and the selected CZ3 satisfy the protocol criteria. "
        "An `above_baseline` result preserves correct action but does not indicate a cost improvement. "
        "A failed point does not prove that realization within this template is impossible: the result applies to the specified optimization budget.", "",
        "## Protocol and provenance", "",
        f"The grid contains {config.theta_points} evenly spaced theta values from 0 to pi/4; "
        f"the evaluated prefix has {len(config.grid())} points. Theta is given in radians. "
        "All qutrits at a given point share the same E_theta.", "",
        "`E_theta = [[cos(theta),0,0], [0,1,0], [0,0,1], [-sin(theta),0,0]]`; "
        "`l_theta = [sin(theta),0,0,cos(theta)]`. "
        "F3 has the extension `E_theta F3 E_theta† + exp(i alpha)|l_theta><l_theta|`.", "",
        f"Baseline QPY: `{config.baseline_qpy}`. SHA-256: `{manifest.get('baseline_sha256', config.baseline_sha256)}`. "
        f"Original template: {original_cz} CZ, {original_u3} U3, {len(template['parameter_metadata'])} angles. "
        f"Comparable baseline cost after transpilation: {_format(baseline_cost)} CZ. "
        "This is a pinned known realization; the experiment does not prove a global minimum.", "",
        f"Transpilation: `basis_gates=['u','cz']`, all-to-all connectivity, `approximation_degree=1.0`, "
        f"`optimization_level={config.optimization_level}`, seeds `{list(config.transpiler_seeds)}`. "
        "Valid results are ranked by CZ count, depth, 1q gate count and seed. "
        "This is not a routing cost on a physical processor.", "",
        f"F3: `E_norm <= {config.f3_tolerance}`, `L_norm <= {config.f3_tolerance}`, at most 2 CZ. "
        f"CZ3: `E_norm <= {config.cz3_tolerance}`, `L_norm <= {config.cz3_tolerance}`. "
        "The norms compare action on the code space using a single shared global phase. "
        "For CZ3, all nine operator columns are checked, rather than separate fidelities with independent phases.", "",
        f"Continuation uses the last valid parameters of the same template; `max_nfev={config.max_nfev}`, "
        f"maximum step-halving depth {config.max_subdivisions}. "
        "The StateSystem fallback runs when the template or its transpilation fails, or when the baseline cost is exceeded. "
        "BQSKit does not replace the parameters of the continued branch. The last valid template remains the parent for subsequent attempts.", "",
        "### Saved baseline metrics", "", "```json", json.dumps(baseline["metadata"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "### Configuration", "", "```json", json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "### Versions and source hashes", "", "```json", json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## F3 results", "",
        _table(rows, [("index", "index"), ("theta_over_pi", "theta/pi"), ("f3_alpha", "alpha [rad]"),
                      ("f3_n_cz", "CZ selected alpha"), ("f3_alpha_zero_n_cz", "CZ alpha=0"),
                      ("f3_E_norm", "E_norm"), ("f3_L_norm", "L_norm")]), "",
        "## CZ3 results", "",
        _table(rows, [("index", "index"), ("theta_over_pi", "theta/pi"), ("status", "status"),
                      ("selected_method", "selected method"), ("cz3_n_cz", "CZ"), ("baseline_n_cz", "baseline CZ"),
                      ("cz3_depth", "depth"), ("cz3_n_1q", "1q"), ("cz3_E_norm", "E_norm"), ("cz3_L_norm", "L_norm"),
                      ("continuation_seconds", "continuation [s]"), ("fallback_seconds", "fallback [s]")]), "",
        f"Total continuation time: {sum(_number(row, 'continuation_seconds') for row in rows if math.isfinite(_number(row, 'continuation_seconds'))):.6g} s. "
        f"Total fallback time: {sum(_number(row, 'fallback_seconds') for row in rows if math.isfinite(_number(row, 'fallback_seconds'))):.6g} s.", "",
        "## Full circuits", "",
        f"Saved circuit results: {len(circuit_rows)}. "
        f"Valid: {sum(row.get('success') is True for row in circuit_rows)}. "
        f"Other: {sum(row.get('success') is not True for row in circuit_rows)}. "
        "Circuits are built from saved, verified F3/CZ3 gates. Failed gate points are skipped.", "",
        _table(circuit_rows, [("index", "index"), ("theta_over_pi", "theta/pi"), ("state_name", "state"),
                              ("status", "status"), ("two_qubit_gate_count", "CZ"), ("circuit_depth", "depth"),
                              ("fidelity", "fidelity"), ("compile_time_seconds", "time [s]")]), "",
        "### Component costs before block fusion", "",
        "The component columns give the total costs of preparing the encoded zero state and of all F3 and CZ3 blocks "
        "in a circuit, with blocks compiled separately to the same U/CZ basis and with the same settings. "
        "The total before block fusion may differ from the result of whole-circuit optimization, "
        "which can also simplify gates at block boundaries.", "",
        _table(circuit_rows, [("index", "index"), ("theta_over_pi", "theta/pi"), ("state_name", "state"),
                              ("preparation_n_cz", "preparation CZ"), ("f3_blocks_n_cz", "F3 blocks CZ"),
                              ("cz3_blocks_n_cz", "CZ3 blocks CZ"), ("unfused_n_cz", "total before fusion CZ"),
                              ("two_qubit_gate_count", "full circuit CZ")]), "",
        "The corresponding 1q costs are saved in [full_circuits.csv](full_circuits.csv): "
        "`preparation_n_1q`, `f3_blocks_n_1q`, `cz3_blocks_n_1q`, `unfused_n_1q` "
        "and `one_qubit_gate_count` for the full circuit. "
        "The — symbol indicates missing saved data, including in older artifacts; it does not mean zero cost. "
        "The report does not reconstruct a missing breakdown from the final cost.", "",
        "## Parameters and interpretation limits", "",
        "Raw angles are saved unchanged. `unwrap` is applied separately to each contiguous segment of valid "
        "points with one `template_id`; it does not cross gaps or template changes. "
        "Missing valid continuation produces an empty CSV cell and a plot gap, even if fallback found a valid circuit. "
        "Repeated parent parameters are not a result for the current theta.", "",
        "Euler angles do not have a unique representation and may jump even when the operator is continuous. "
        "A plot after `unwrap` does not prove smoothness or the existence of an analytical solution. "
        "Also shown is `min_gamma ||U_j(theta_i) - exp(i gamma) U_j(theta_(i-1))||_F`, "
        "computed from Qiskit UGate for adjacent valid points of the same template. "
        "This measure removes the global phase of an individual U3 and respects its periodicity; it still does not prove smoothness "
        "of the full circuit or remove arbitrary gauge redistribution between different gates.", "",
        "## Files", "",
        "[points.csv](points.csv) contains all completed points, including failed ones. "
        "[attempts.csv](attempts.csv) preserves intermediate attempts and their parents. "
        "[parameters.csv](parameters.csv) contains raw and separately unwrapped angles, "
        "[operator_distances.csv](operator_distances.csv) contains local operator distances, "
        "and [full_circuits.csv](full_circuits.csv) contains full-circuit results. "
        "JSON, NPY and QPY remain the data source; the report reads them after checking the hashes of complete bundles.", "",
    ]
    for name, caption in figures:
        lines.extend([f"![{name}](figures/{name}.png)", "", f"{caption} [PDF](figures/{name}.pdf).", ""])
    errors = [row for row in rows if row.get("errors") or row.get("fallback_reason")]
    if errors:
        lines.extend(["## Fallback reasons and errors", "", _table(errors, [
            ("index", "index"), ("fallback_reason", "fallback reason"), ("errors", "error log")]), ""])
    return "\n".join(lines)


def generate_report(output_dir: str | Path, *, full_circuits: list[dict] | None = None) -> Path:
    """Validate persisted inputs, then regenerate Markdown, CSV and PNG/PDF files.

    Persisted circuit rows are authoritative. ``full_circuits`` is an optional
    runner summary; it cannot replace or hide corrupt persisted bundles.
    """
    store = RunStore.open(output_dir)
    manifest = store.manifest
    if manifest.get("benchmark") == "theta_continuation_v1":
        expected = fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"})
        if manifest["fingerprint"] != expected:
            raise ValueError("Run manifest fingerprint does not match its contents")
    config = ThetaBenchmarkConfig.from_dict(manifest["config"])
    grid = np.asarray(config.grid(), dtype=float)
    baseline = store.read_bundle("baseline")
    bundles = []
    for index in store.completed_point_indices():
        bundle = store.read_bundle(f"points/{index:05d}")
        row = bundle["metadata"]
        if (row.get("index") != index or not 0 <= index < len(grid)
                or not math.isclose(_number(row, "theta"), grid[index], rel_tol=0.0, abs_tol=1e-14)):
            raise ValueError("Point index/theta does not match the configured grid")
        bundles.append(bundle)
    rows = [bundle["metadata"] for bundle in bundles]
    attempts = []
    for bundle in bundles:
        values = bundle["documents"].get("attempts.json", {}).get("attempts", [])
        if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
            raise ValueError("Point attempts must be a list of objects")
        attempts.extend(_flatten({"point_index": bundle["metadata"]["index"], **value}) for value in values)
    circuit_rows = []
    for marker in sorted((store.root / "circuits").glob("*/*/complete.json")):
        bundle = store.read_bundle(marker.parent.relative_to(store.root).as_posix())
        circuit_rows.append(bundle["metadata"])
    if full_circuits and not circuit_rows:
        raise ValueError("Full-circuit summary has no corresponding persisted bundles")
    parameter_rows, distance_rows, templates = _parameter_data(grid, bundles, baseline)
    # All source bundles and numerical layouts are validated before any derived
    # file is published, including when a runner supplies in-memory summaries.
    _write_csv(store.root, "points.csv", rows, _POINT_COLUMNS)
    _write_csv(store.root, "attempts.csv", attempts, ("point_index", "id", "theta", "requested_grid_point", "warm_start_parent_id"))
    _write_csv(store.root, "parameters.csv", parameter_rows, _PARAMETER_COLUMNS)
    _write_csv(store.root, "operator_distances.csv", distance_rows, _DISTANCE_COLUMNS)
    _write_csv(store.root, "full_circuits.csv", circuit_rows, ("index", "theta", "theta_over_pi", "state_name", "success", "status"))
    figures = _figures(store.root, grid, rows, templates, config)
    _write_text(store.root, "report.md", _markdown(store, config, baseline, rows, circuit_rows, figures))
    return store.root / "report.md"
