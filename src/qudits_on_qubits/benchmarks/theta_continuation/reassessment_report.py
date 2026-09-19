"""Publish threshold reassessment costs and diagrams from hash-checked QPY files."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit, resolve_direct_state,
)
from .artifacts import RunStore, fingerprint
from .encoding import theta_embedding
from .models import ThetaBenchmarkConfig
from .report import _format, _output_path, _table, _write_csv, _write_text


@dataclass
class _Artifact:
    name: str
    label: str
    index: int
    circuit: QuantumCircuit
    qpy_path: str
    draw: bool = True


def circuit_costs(circuit: QuantumCircuit) -> dict[str, int]:
    """Count elementary gates and dependency depths directly from the circuit.

    CZ depth filters out one-qubit gates; disjoint CZ gates occupy one layer.
    No routing, duration model or hardware scheduling is implied.
    """
    if circuit.num_clbits or circuit.num_parameters or not math.isfinite(float(circuit.global_phase)):
        raise ValueError("Report circuit must be bound, finite and have no classical bits")
    for item in circuit.data:
        operation = item.operation
        if (operation.name not in {"u", "u3", "cz"}
                or operation.num_qubits != (2 if operation.name == "cz" else 1)
                or any(not math.isfinite(float(value)) for value in operation.params)):
            raise ValueError("Report circuit must contain finite elementary U/U3/CZ gates")
    return {"n_cz": int(circuit.count_ops().get("cz", 0)),
            "n_1q": sum(item.operation.num_qubits == 1 for item in circuit.data),
            "depth": int(circuit.depth() or 0),
            "cz_depth": int(circuit.depth(lambda item: item.operation.name == "cz") or 0),
            "size": len(circuit.data)}


def _norm(metadata: dict, prefix: str, key: str) -> float | None:
    value = metadata.get(f"{prefix}_metrics", {}).get(key, metadata.get(f"{prefix}_{key}"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid saved {prefix} {key}")
    return float(value)


def _collect(store: RunStore):
    manifest = store.manifest
    if manifest.get("benchmark") != "theta_threshold_reassessment_v1":
        raise ValueError("This report only writes to a threshold reassessment run")
    expected = fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"})
    if manifest["fingerprint"] != expected:
        raise ValueError("Run manifest fingerprint does not match its contents")
    config = ThetaBenchmarkConfig.from_dict(manifest["config"])
    grid = config.grid()
    representatives = {0, (len(grid) - 1) // 2, len(grid) - 1}
    store.read_bundle("baseline")
    points, bundles, artifacts = [], {}, []
    for index in store.completed_point_indices():
        bundle = store.read_bundle(f"points/{index:05d}")
        metadata = bundle["metadata"]
        theta = metadata.get("theta")
        if (metadata.get("index") != index or not 0 <= index < len(grid)
                or not isinstance(theta, (float, int)) or isinstance(theta, bool)
                or not math.isclose(theta, grid[index], rel_tol=0, abs_tol=1e-14)):
            raise ValueError("Point index/theta does not match configured grid")
        row = dict(metadata)
        row["theta_over_pi"] = theta / np.pi
        reassessment = metadata.get("reassessment", bundle["documents"].get("reassessment.json", {}))
        row["cz3_origin"] = reassessment.get("origin", "unknown")
        row["source_status"] = reassessment.get("source_status")
        row["cz3_compilation_method"] = metadata.get("cz3_metrics", {}).get("compilation_method", "transpiler")
        row["reassessment"] = reassessment
        for prefix, filename, label in (("f3", "f3_optimal.qpy", "F3"),
                                       ("f3_alpha_zero", "f3_alpha_zero.qpy", "F3, alpha = 0"),
                                       ("cz3", "cz3_selected.qpy", "CZ3")):
            circuit = bundle["circuits"].get(filename)
            if circuit is None:
                if prefix != "cz3" or metadata.get("correct") is True:
                    raise ValueError(f"Missing selected circuit: points/{index:05d}/{filename}")
                row.update({f"{prefix}_{key}": None for key in ("n_cz", "n_1q", "depth", "cz_depth", "size", "E_norm", "L_norm")})
                continue
            costs = circuit_costs(circuit)
            if circuit.num_qubits != (4 if prefix == "cz3" else 2):
                raise ValueError(f"Invalid {prefix} qubit count")
            row.update({f"{prefix}_{key}": value for key, value in costs.items()})
            for key in ("E_norm", "L_norm"):
                row[f"{prefix}_{key}"] = _norm(metadata, prefix, key)
                if row[f"{prefix}_{key}"] is None:
                    raise ValueError(f"Missing saved {prefix} {key}")
            tolerance = config.cz3_tolerance if prefix == "cz3" else config.f3_tolerance
            if metadata.get("correct") is True and max(row[f"{prefix}_E_norm"], row[f"{prefix}_L_norm"]) > tolerance:
                raise ValueError(f"Accepted point exceeds {prefix} tolerance")
            artifacts.append(_Artifact(f"{prefix}_{index:05d}", label, index, circuit,
                                       f"points/{index:05d}/{filename}",
                                       prefix != "f3_alpha_zero" or index in representatives))
        points.append(row)
        bundles[index] = bundle
    full_rows = []
    by_index = {row["index"]: row for row in points}
    for marker in sorted((store.root / "circuits").glob("*/*/complete.json")):
        relative = marker.parent.relative_to(store.root).as_posix()
        _, state_name, index_string = relative.split("/")
        if (state_name not in config.states or not index_string.isdecimal()
                or index_string != f"{int(index_string):05d}"):
            raise ValueError("Unexpected full-circuit bundle path")
        index = int(index_string)
        bundle = store.read_bundle(relative)
        row = dict(bundle["metadata"])
        if (index not in bundles or row.get("index") != index or row.get("state_name") != state_name
                or row.get("theta") != by_index[index]["theta"]
                or row.get("gate_point_fingerprint") != fingerprint(bundles[index]["complete"])
                or ("row.json" in bundle["documents"] and row != bundle["documents"]["row.json"])):
            raise ValueError("Full-circuit bundle identity mismatch")
        row["theta_over_pi"] = row["theta"] / np.pi
        if row.get("success") is not True:
            full_rows.append(row)
            continue
        if by_index[index].get("correct") is not True:
            raise ValueError("Successful full circuit references a rejected gate point")
        circuit = bundle["circuits"].get("graph_state_transpiled.qpy")
        preparation = bundle["circuits"].get("zero_preparation.qpy")
        if circuit is None or preparation is None:
            raise ValueError("Successful full circuit requires circuit and preparation QPY")
        state = resolve_direct_state(state_name)
        if circuit.num_qubits != 2 * state.num_qutrits or preparation.num_qubits != 2:
            raise ValueError("Full circuit or preparation qubit count mismatch")
        row.update(circuit_costs(circuit))
        fidelity = row.get("fidelity")
        if (isinstance(fidelity, bool) or not isinstance(fidelity, (float, int))
                or not math.isfinite(fidelity) or not 0 <= fidelity <= 1 + 1e-12):
            raise ValueError("Full circuit requires finite saved fidelity")
        reference = build_direct_basis_graph_state_circuit(state_name, theta_embedding(row["theta"]))
        actual_fidelity = float(state_fidelity(Statevector.from_instruction(reference), Statevector.from_instruction(circuit)))
        if (not math.isfinite(actual_fidelity) or abs(actual_fidelity - fidelity) > 1e-12
                or actual_fidelity < 1 - max(1e-8, 100 * config.cz3_tolerance**2)):
            raise ValueError("Full-circuit fidelity failed independent reference validation")
        row["saved_fidelity"] = fidelity
        row["fidelity"] = actual_fidelity
        row.update(two_qubit_gate_count=row["n_cz"], one_qubit_gate_count=row["n_1q"],
                   best_depth=row["depth"], best_two_qubit_depth=row["cz_depth"])
        multiplicities = {"preparation": state.num_qutrits, "f3_blocks": state.num_qutrits,
                          "cz3_blocks": sum(left != right for left, right in state.edges)}
        component_circuits = {"preparation": preparation,
                              "f3_blocks": bundles[index]["circuits"]["f3_optimal.qpy"],
                              "cz3_blocks": bundles[index]["circuits"]["cz3_selected.qpy"]}
        row["component_num_qutrits"] = state.num_qutrits
        row["component_num_cz3_blocks"] = multiplicities["cz3_blocks"]
        for prefix, component in component_circuits.items():
            costs = circuit_costs(component)
            for key in ("n_cz", "n_1q"):
                row[f"{prefix}_{key}"] = multiplicities[prefix] * costs[key]
        for key in ("n_cz", "n_1q"):
            row[f"unfused_{key}"] = sum(row[f"{prefix}_{key}"] for prefix in multiplicities)
        for key in ("f3_E_norm", "f3_L_norm", "cz3_E_norm", "cz3_L_norm"):
            row[key] = by_index[index][key]
        row["qpy_path"] = f"{relative}/graph_state_transpiled.qpy"
        artifacts.append(_Artifact(f"{state_name}_{index:05d}", state_name, index, circuit,
                                   row["qpy_path"], index in representatives))
        full_rows.append(row)
    return config, points, full_rows, artifacts


def _parameter_rows(artifacts: list[_Artifact]) -> list[dict]:
    rows = []
    for artifact in artifacts:
        common = {"artifact_id": artifact.name, "point_index": artifact.index,
                  "qpy_path": artifact.qpy_path, "global_phase": float(artifact.circuit.global_phase)}
        if not artifact.circuit.data:
            rows.append({**common, "operation": "identity"})
        for index, item in enumerate(artifact.circuit.data):
            row = {**common, "instruction_index": index, "operation": item.operation.name,
                   "qubits": [artifact.circuit.find_bit(bit).index for bit in item.qubits]}
            if item.operation.name in {"u", "u3"}:
                row.update(zip(("theta", "phi", "lambda"), map(float, item.operation.params)))
                row["diagram_label"] = f"U{index + 1}"
            rows.append(row)
    return rows


def _save_figure(root: Path, figure, relative: str) -> None:
    for suffix in ("png", "pdf"):
        path = _output_path(root, f"{relative}.{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140, bbox_inches="tight")
    figure.clear()


def _draw_circuit(root: Path, artifact: _Artifact) -> None:
    """Draw the actual ordered instruction stream, without visual gate fusion."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    circuit = artifact.circuit
    columns = 16
    panel_count = max(1, math.ceil(len(circuit.data) / columns))
    figure = Figure(figsize=(min(18, max(7, min(columns, len(circuit.data)) * 0.75)),
                              panel_count * (0.4 * circuit.num_qubits + 1.2)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(panel_count, 1, squeeze=False).flat
    width = min(columns, max(1, len(circuit.data)))
    for panel, axis in enumerate(axes):
        start = panel * columns
        items = circuit.data[start:start + columns]
        for qubit in range(circuit.num_qubits):
            axis.plot([0.4, width + 0.6], [qubit, qubit], color="#778596", linewidth=0.8, zorder=0)
            axis.text(0.15, qubit, f"q{qubit}", ha="right", va="center", fontsize=9)
        for offset, item in enumerate(items):
            x = offset + 1
            qubits = [circuit.find_bit(bit).index for bit in item.qubits]
            if item.operation.name == "cz":
                axis.plot([x, x], qubits, color="#c34b32", linewidth=1.6)
                axis.scatter([x, x], qubits, color="#c34b32", s=35, zorder=3)
            else:
                axis.add_patch(Rectangle((x - 0.4, qubits[0] - 0.31), 0.8, 0.62,
                                         facecolor="#e4eefb", edgecolor="#235b9e", zorder=2))
                axis.text(x, qubits[0], f"U{start + offset + 1}", ha="center", va="center", fontsize=8, zorder=3)
        axis.set(xlim=(-0.3, width + 0.8), ylim=(circuit.num_qubits - 0.45, -0.55))
        axis.axis("off")
        if panel_count > 1:
            axis.set_title(f"Instructions {start + 1}–{start + len(items)}", loc="left", fontsize=9)
    costs = circuit_costs(circuit)
    figure.suptitle(f"{artifact.label}; point {artifact.index}; CZ={costs['n_cz']}, 1q={costs['n_1q']}, "
                   f"depth={costs['depth']}, CZ depth={costs['cz_depth']}\n"
                   f"Instruction order; U angles in gate_parameters.csv; global phase={float(circuit.global_phase):.8g}",
                   fontsize=10)
    _save_figure(root, figure, f"diagrams/{artifact.name}")


def _figures(root: Path, points: list[dict], full_rows: list[dict], config: ThetaBenchmarkConfig) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    x = [row["theta_over_pi"] for row in points]
    for prefix, label in (("f3", "F3"), ("f3_alpha_zero", "F3 alpha=0"), ("cz3", "CZ3")):
        for key, axis in (("n_cz", axes[0, 0]), ("depth", axes[0, 1])):
            axis.plot(x, [row.get(f"{prefix}_{key}", np.nan) for row in points], ".-", label=label)
    for key in ("cz3_E_norm", "cz3_L_norm"):
        axes[1, 0].plot(x, [max(row[key], 1e-18) if row.get(key) is not None else np.nan for row in points], ".-", label=key)
    axes[1, 0].axhline(config.cz3_tolerance, color="firebrick", linestyle="--", label="CZ3 threshold")
    axes[1, 0].set_yscale("log")
    for state in config.states:
        rows = [row for row in full_rows if row["state_name"] == state and row.get("success")]
        axes[1, 1].plot([row["theta_over_pi"] for row in rows], [row["n_cz"] for row in rows], ".-", label=state)
    for axis, ylabel, title in zip(axes.flat, ("CZ", "depth", "Frobenius norm", "CZ"),
                                  ("Gate cost", "Total gate depth", "CZ3: error and leakage", "Full circuits")):
        axis.set(xlabel="theta / pi", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    _save_figure(root, figure, "figures/reassessment_metrics")


def _links(artifact: _Artifact) -> str:
    paths = [(artifact.qpy_path, "QPY"), (f"diagrams/{artifact.name}.txt", "TXT")]
    if artifact.draw:
        paths.extend((f"diagrams/{artifact.name}.{suffix}", suffix.upper()) for suffix in ("png", "pdf"))
    return " · ".join(f'<a href="{escape(path, quote=True)}">{label}</a>' for path, label in paths)


def _gallery(artifacts: list[_Artifact], points: list[dict], full_rows: list[dict]) -> str:
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Theta circuits — reassessment</title>',
             '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#182434}',
             'img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd5df;padding:8px;text-align:left}',
             'details{border:1px solid #ccd5df;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}a{color:#195ba6}</style>',
             '<body><h1>Theta circuits — reassessment</h1>',
             '<p><a href="report.md">Report</a> · <a href="points.csv">Gates CSV</a> · '
             '<a href="full_circuits.csv">Full circuits CSV</a> · <a href="gate_parameters.csv">Exact U angles and global phases</a></p>',
             '<p>Diagrams come from saved QPY files. Columns show instruction order; independent operations may run in parallel. '
             'U&lt;n&gt; denotes the instruction index + 1 in the CSV. Depths are computed from qubit dependencies. '
             'Read successive panels from left to right, then from top to bottom.</p>']
    by_index = {row["index"]: row for row in points}
    for artifact in artifacts:
        if artifact.label in {"F3", "CZ3", "F3, alpha = 0"}:
            row = by_index[artifact.index]
            lines.append(f'<details><summary>{escape(artifact.label)} — point {artifact.index}, theta/pi={row["theta_over_pi"]:.8g}</summary>')
            lines.append(f'<p>{_links(artifact)}</p>')
            if artifact.draw:
                lines.append(f'<img loading="lazy" src="diagrams/{artifact.name}.png" alt="{escape(artifact.label)}; point {artifact.index}">')
            lines.append('</details>')
    lines.append('<h2>Full circuits</h2><p>Diagrams: start, middle and end of the grid; QPY and TXT: every saved circuit.</p>')
    lines.append('<table><tr><th>State / index</th><th>CZ / 1q</th><th>depth / CZ depth</th><th>Fidelity</th><th>Files</th></tr>')
    full = {(row["state_name"], row["index"]): row for row in full_rows}
    for artifact in artifacts:
        if artifact.label not in {"F3", "CZ3", "F3, alpha = 0"}:
            row = full[artifact.label, artifact.index]
            lines.append(f'<tr><td>{escape(artifact.label)} / {artifact.index}</td><td>{row["n_cz"]} / {row["n_1q"]}</td>'
                         f'<td>{row["depth"]} / {row["cz_depth"]}</td><td>{row["fidelity"]:.16g}</td><td>{_links(artifact)}</td></tr>')
    lines.append('</table></body></html>')
    return "\n".join(lines)


def _markdown(store, config, points, full_rows):
    manifest = store.manifest
    correct = sum(row.get("correct") is True for row in points)
    successful_full = sum(row.get("success") is True for row in full_rows)
    source_known = [row["reassessment"]["source_correct"] for row in points if isinstance(row["reassessment"].get("source_correct"), bool)]
    source_correct = str(sum(source_known)) if len(source_known) == len(points) else "incomplete data"
    attempts = [attempt for row in points for attempt in row["reassessment"].get("attempts", [])]
    historical_failed = sum(attempt.get("historical_valid") is False for attempt in attempts)
    new_valid = sum(attempt.get("valid_at_new_tolerance") is True for attempt in attempts)
    min_error = min((attempt["E_norm"] for attempt in attempts), default=None)
    display_full = [{**row, "fidelity_display": f"{row['fidelity']:.17g}" if row.get("fidelity") is not None else None} for row in full_rows]
    old_tolerance = manifest.get("source_cz3_tolerance", manifest.get("source_config", {}).get("cz3_tolerance"))
    lines = ["# Theta benchmark: CZ3 reassessment", "",
             f"CZ3 threshold: **E_norm and L_norm ≤ {config.cz3_tolerance}**; previously {_format(old_tolerance)}. "
             f"F3: E_norm and L_norm ≤ {config.f3_tolerance}. Full-operator preservation during compilation: ≤ 1e-10.", "",
             f"Valid points: **{correct}/{len(config.grid())}**; saved points: {len(points)}. "
             f"Valid full circuits: **{successful_full}/{len(config.grid()) * len(config.states)}**. "
             f"At the previous threshold: **{source_correct}/{len(config.grid())}** valid points.", "",
             "Validity at the threshold does not imply an exact ideal gate. This is a reassessment of saved attempts and recovered "
             "BQSKit fallbacks; no new angle continuation was performed. Historical attempts and their parents remain saved.", "",
             f"Historical continuation attempts: {len(attempts)}; failed at the previous threshold: {historical_failed}; "
             f"meeting the new threshold: {new_valid}. Minimum E_norm of these attempts: {_format(min_error)}. "
             "Changing the threshold alone does not prove successful continuation of the fixed template. "
             "Details: [attempts.csv](attempts.csv).", "",
             "[Gallery of all circuits](gallery.html) · [Gates CSV](points.csv) · [Full circuits CSV](full_circuits.csv) · "
             "[Gate angles and global phases CSV](gate_parameters.csv) · [Manifest and provenance](reassessment_provenance.json)", "",
             "CZ counts, 1q gate counts and depths are computed from QPY files after checking hashes. "
             "CZ3 is a two-qutrit block on four qubits; the CZ column counts physical qubit gates. "
             "Depth is the depth of all gates; CZ depth is the depth after omitting 1q gates. "
             "All-to-all connectivity is assumed, without hardware routing. CSV files preserve the full saved numerical precision.", "",
             "## F3", "",
             _table(points, [("index", "index"), ("theta_over_pi", "theta/pi"), ("f3_alpha", "alpha [rad]"),
                             ("f3_n_cz", "CZ"), ("f3_n_1q", "1q"), ("f3_depth", "depth"),
                             ("f3_cz_depth", "CZ depth"), ("f3_E_norm", "E_norm"), ("f3_L_norm", "L_norm"),
                             ("f3_alpha_zero_n_cz", "CZ at alpha=0"), ("f3_alpha_zero_depth", "depth at alpha=0"),
                             ("f3_alpha_zero_cz_depth", "CZ depth at alpha=0")]), "", "## CZ3", "",
             _table(points, [("index", "index"), ("theta_over_pi", "theta/pi"), ("correct", "valid"),
                             ("cz3_n_cz", "CZ"), ("cz3_n_1q", "1q"), ("cz3_depth", "depth"),
                             ("cz3_cz_depth", "CZ depth"), ("cz3_E_norm", "E_norm"), ("cz3_L_norm", "L_norm"),
                             ("cz3_origin", "origin"), ("cz3_compilation_method", "compilation")]), "",
             "`baseline`: pinned circuit for theta=0; `source_saved`: raw circuit saved in the previous campaign; "
             "`reconstructed`: missing QPY recovered with unchanged synthesis configuration. "
             "Comparisons of reconstruction norms and costs against historical results are saved in the CSV `reassessment` column "
             "and in point bundles. `exact_u3_to_u` means an exact rewrite of U3 as U, preserving parameters "
             "and global phase after ordinary transpilation attempts failed validation.", "", "## Full graph circuits", "",
             _table(display_full, [("state_name", "state"), ("index", "index"), ("theta_over_pi", "theta/pi"),
                               ("n_cz", "CZ"), ("n_1q", "1q"), ("depth", "depth"), ("cz_depth", "CZ depth"),
                               ("fidelity_display", "fidelity")]), "",
             "Fidelity is recomputed from QPY against an independent ideal reference state, without rounding in the CSV. "
             "The saved_fidelity column preserves the previous rounded value; agreement is checked to 1e-12. "
             "F3/CZ3 norms in the CSV describe the blocks used, not the operator norm of the entire state-preparation circuit.", "",
             "### Components before whole-circuit optimization", "",
             "Costs include preparing the encoded zero state for each qutrit. The component sum may differ "
             "from the final cost after simplifications at block boundaries. Component columns show CZ / 1q.", ""]
    components = []
    for row in full_rows:
        data = dict(row)
        for prefix in ("preparation", "f3_blocks", "cz3_blocks", "unfused"):
            data[prefix] = f"{_format(row.get(prefix + '_n_cz'))} / {_format(row.get(prefix + '_n_1q'))}"
        components.append(data)
    lines.extend([_table(components, [("state_name", "state"), ("index", "index"), ("preparation", "preparation"),
                                     ("f3_blocks", "F3"), ("cz3_blocks", "CZ3"), ("unfused", "sum")]), "",
                  "![Costs and norms](figures/reassessment_metrics.png)", "", "[Plot PDF](figures/reassessment_metrics.pdf)", "",
                  "## Provenance", "", f"New campaign fingerprint: `{manifest['fingerprint']}`.", "",
                  f"Source fingerprint: `{manifest.get('source_fingerprint', 'missing')}`.", "",
                  f"Versions: `{json.dumps(manifest.get('provenance', {}).get('versions', {}), ensure_ascii=False, sort_keys=True)}`.", "",
                  "QPY and TXT are available for every saved selected circuit. PNG/PDF diagrams cover F3 and CZ3 "
                  "at all points, plus full circuits at the start, middle and end of the grid. "
                  "Diagrams show the actual instruction order; exact parameters and global phases are in the CSV.", ""])
    reconstructed = [row for row in points if row["cz3_origin"] == "reconstructed"]
    different = [row["index"] for row in reconstructed
                 if row["reassessment"].get("fallback_comparison", {}).get("matches_original") is False]
    exact = sum(row["cz3_compilation_method"] == "exact_u3_to_u" for row in points)
    lines.extend([f"Recovered missing QPY files: {len(reconstructed)}. "
                  f"Selected circuits with an exact U3-to-U rewrite: {exact}. "
                  f"Reconstructions differing from historical norms/counts: {different or 'no differences reported'}.", ""])
    if correct != len(config.grid()) or successful_full != len(config.grid()) * len(config.states):
        failed = [row["index"] for row in points if row.get("correct") is not True]
        missing = sorted(set(range(len(config.grid()))) - {row["index"] for row in points})
        lines[2:2] = ["**Not all points or full circuits passed validation or have complete results.** "
                      f"Rejected points: {failed}; missing points: {missing}.", ""]
    source_run = manifest.get("source_run")
    if source_run:
        source_report = Path(source_run).resolve() / "report.md"
        if source_report.is_file():
            lines.extend([f"[Source report, previous threshold](<{source_report.as_posix()}>).", ""])
    return "\n".join(lines)


def generate_reassessment_report(output_dir: str | Path) -> Path:
    """Rebuild a report inside a new reassessment campaign, leaving its source intact."""
    store = RunStore.open(output_dir)
    config, points, full_rows, artifacts = _collect(store)
    parameters = _parameter_rows(artifacts)
    attempts = [{"point_index": row["index"], **attempt} for row in points for attempt in row["reassessment"].get("attempts", [])]
    # Read and validate every persisted bundle before publishing derived outputs.
    _write_csv(store.root, "points.csv", points, ("index", "theta", "theta_over_pi", "correct"))
    _write_csv(store.root, "full_circuits.csv", full_rows, ("state_name", "index", "theta", "theta_over_pi", "success"))
    _write_csv(store.root, "attempts.csv", attempts, ("point_index", "id", "theta", "historical_valid", "valid_at_new_tolerance", "E_norm", "L_norm"))
    _write_csv(store.root, "gate_parameters.csv", parameters,
               ("artifact_id", "point_index", "instruction_index", "operation", "qubits", "theta", "phi", "lambda", "global_phase"))
    for artifact in artifacts:
        _write_text(store.root, f"diagrams/{artifact.name}.txt",
                    f"{artifact.label}; point {artifact.index}; source {artifact.qpy_path}\n"
                    f"global_phase = {float(artifact.circuit.global_phase)!r}\n"
                    + str(artifact.circuit.draw(output="text", fold=140)))
        if artifact.draw:
            _draw_circuit(store.root, artifact)
    _figures(store.root, points, full_rows, config)
    _write_text(store.root, "reassessment_provenance.json", json.dumps(store.manifest, ensure_ascii=False, indent=2, allow_nan=False))
    _write_text(store.root, "gallery.html", _gallery(artifacts, points, full_rows))
    _write_text(store.root, "report.md", _markdown(store, config, points, full_rows))
    return store.root / "report.md"
