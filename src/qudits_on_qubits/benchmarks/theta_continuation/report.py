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
        for representation, label in (("raw", "Surowe kąty"), ("unwrapped", "Kąty unwrap w oddzielnych segmentach")):
            figure, axes = new(nrows, columns, width=12, height=max(3.2, 2.7*nrows))
            for gate, axis in enumerate(axes.flat):
                if gate >= gate_count:
                    axis.set_visible(False)
                    continue
                item = data["layout"][gate * 3]
                for angle, name in enumerate(("theta", "phi", "lambda")):
                    axis.plot(x, data[representation][:, gate*3+angle], marker=".", label=name, linewidth=1)
                axis.set(title=f"Instrukcja {item['instruction_index']}, kubit {item['qubit']}", xlabel=r"$\theta/\pi$", ylabel="kąt [rad]")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=7)
            figure.suptitle(f"{label}; template_id={template_id[:16]}")
            save(figure, f"parameters_{representation}_template_{template_index:02d}",
                 f"{label}; schemat `{template_id}`. Przerwy oznaczają brak poprawnych parametrów bieżącego punktu.")
        figure, axes = new(width=11, height=4.5)
        axis = axes[0, 0]
        for gate, item in enumerate(data["layout"][::3]):
            axis.plot(x, data["distances"][:, gate], marker=".", linewidth=1,
                      label=f"instr. {item['instruction_index']}, q{item['qubit']}")
        axis.set(xlabel=r"$\theta/\pi$", ylabel="min. odległość Frobeniusa", title="Lokalne operatory U3: odległość od sąsiedniego poprawnego punktu")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=min(4, gate_count))
        save(figure, f"operator_distances_template_{template_index:02d}",
             "Odległości lokalnych U3 po dopasowaniu jednej fazy globalnej; porównania wyłącznie przez sąsiednie poprawne punkty tego samego schematu.")

    by_index = {row["index"]: row for row in rows}

    def series(key, *, correct_only=False):
        return np.array([_number(by_index.get(index, {}), key)
                         if not correct_only or by_index.get(index, {}).get("correct") is True else np.nan
                         for index in range(len(grid))])

    fallback = [index for index, row in by_index.items() if row.get("fallback_used")]
    failed = [index for index, row in by_index.items() if row.get("correct") is not True]
    figure, axes = new(2, 2, width=12, height=8)
    axes[0, 0].plot(x, series("f3_n_cz"), ".-", label="F3, dobrana alpha")
    axes[0, 0].plot(x, series("f3_alpha_zero_n_cz"), ".--", label="F3, alpha = 0")
    axes[0, 0].plot(x, series("cz3_n_cz", correct_only=True), ".-", label="CZ3, wybrany poprawny obwód")
    axes[0, 0].plot(x, series("baseline_n_cz"), "--", color="gray", label="baseline po transpilacji")
    if fallback:
        axes[0, 0].scatter(x[fallback], series("cz3_n_cz", correct_only=True)[fallback], marker="D", facecolors="none", edgecolors="black", label="fallback")
    axes[0, 0].set(ylabel="liczba CZ", title="Koszty przy wspólnej transpilacji")
    axes[0, 1].plot(x, series("f3_alpha"), ".-")
    axes[0, 1].set(ylabel="alpha [rad, modulo 2pi]", title="Dobrana faza leakage F3")
    for key in ("f3_E_norm", "f3_L_norm", "cz3_E_norm", "cz3_L_norm"):
        axes[1, 0].plot(x, series(key), ".-", label=key)
    axes[1, 0].axhline(config.f3_tolerance, color="gray", linestyle=":", label="tolerancja F3")
    axes[1, 0].axhline(config.cz3_tolerance, color="gray", linestyle="--", label="tolerancja CZ3")
    axes[1, 0].set_yscale("symlog", linthresh=min(config.f3_tolerance, config.cz3_tolerance)*0.01)
    axes[1, 0].set(ylabel="norma Frobeniusa", title="Błąd działania i leakage")
    axes[1, 1].plot(x, series("continuation_seconds"), ".-", label="kontynuacja")
    axes[1, 1].plot(x, series("fallback_seconds"), ".-", label="fallback")
    axes[1, 1].set(ylabel="czas [s]", title="Czas wyznaczania CZ3")
    for axis in axes.flat:
        for index in failed:
            axis.axvline(x[index], color="firebrick", alpha=0.3, linewidth=1)
        axis.set_xlabel(r"$\theta/\pi$")
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    save(figure, "metrics", "Koszty, faza leakage, normy błędów i czas. Czerwone linie oznaczają nieudane punkty; romby oznaczają użycie fallbacku.")
    return figures


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, bool):
        return "tak" if value else "nie"
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
        "# Benchmark E_theta: F3 i kontynuacja CZ3", "",
        f"Przebieg: `{store.root.name}`. Fingerprint: `{manifest['fingerprint']}`.", "",
        f"Zapisane punkty: {len(rows)} / {len(config.grid())}. Poprawne punkty: {correct}. Nieudane punkty: {failed}. "
        f"Poprawne powyżej baseline: {above}. Punkty z fallbackiem: {fallback}. "
        f"Poprawny schemat kontynuacji: {continuation}. Punkty bez ukończonych artefaktów: {pending}.", "",
        "Punkt jest poprawny tylko wtedy, gdy F3 i wybrany CZ3 spełniają kryteria protokołu. "
        "Wynik `above_baseline` zachowuje poprawność działania, ale nie oznacza poprawy kosztu. "
        "Nieudany punkt nie dowodzi niemożliwości realizacji w tym schemacie: wynik dotyczy podanego budżetu optymalizacji.", "",
        "## Protokół i pochodzenie", "",
        f"Siatka obejmuje {config.theta_points} równomiernych theta od 0 do pi/4; "
        f"badany prefiks ma {len(config.grid())} punktów. Theta podano w radianach. "
        "Wszystkie kutryty danego punktu mają to samo E_theta.", "",
        "`E_theta = [[cos(theta),0,0], [0,1,0], [0,0,1], [-sin(theta),0,0]]`; "
        "`l_theta = [sin(theta),0,0,cos(theta)]`. "
        "F3 ma rozszerzenie `E_theta F3 E_theta† + exp(i alpha)|l_theta><l_theta|`.", "",
        f"Baseline QPY: `{config.baseline_qpy}`. SHA-256: `{manifest.get('baseline_sha256', config.baseline_sha256)}`. "
        f"Oryginalny schemat: {original_cz} CZ, {original_u3} U3, {len(template['parameter_metadata'])} kątów. "
        f"Porównywalny koszt baseline po transpilacji: {_format(baseline_cost)} CZ. "
        "To przypięta znana realizacja; eksperyment nie dowodzi globalnego minimum.", "",
        f"Transpilacja: `basis_gates=['u','cz']`, pełna łączność, `approximation_degree=1.0`, "
        f"`optimization_level={config.optimization_level}`, seedy `{list(config.transpiler_seeds)}`. "
        "Poprawne wyniki porządkuje liczba CZ, głębokość, liczba bramek 1q i seed. "
        "Nie jest to koszt routingu na procesorze fizycznym.", "",
        f"F3: `E_norm <= {config.f3_tolerance}`, `L_norm <= {config.f3_tolerance}`, najwyżej 2 CZ. "
        f"CZ3: `E_norm <= {config.cz3_tolerance}`, `L_norm <= {config.cz3_tolerance}`. "
        "Normy porównują działanie na przestrzeni kodowej przy jednej wspólnej fazie globalnej. "
        "Dla CZ3 sprawdzane jest dziewięć kolumn operatora, a nie osobne fidelities z niezależnymi fazami.", "",
        f"Kontynuacja używa ostatnich poprawnych parametrów tego samego schematu; `max_nfev={config.max_nfev}`, "
        f"maksymalna głębokość połówkowania kroku {config.max_subdivisions}. "
        "Fallback StateSystem uruchamia się po niepowodzeniu schematu, jego transpilacji lub po przekroczeniu kosztu baseline. "
        "BQSKit nie zastępuje parametrów kontynuowanej gałęzi. Ostatni poprawny schemat pozostaje rodzicem dalszych prób.", "",
        "### Zapisane metryki baseline", "", "```json", json.dumps(baseline["metadata"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "### Konfiguracja", "", "```json", json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "### Wersje i hashe źródeł", "", "```json", json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## Wyniki F3", "",
        _table(rows, [("index", "indeks"), ("theta_over_pi", "theta/pi"), ("f3_alpha", "alpha [rad]"),
                      ("f3_n_cz", "CZ dobrana alpha"), ("f3_alpha_zero_n_cz", "CZ alpha=0"),
                      ("f3_E_norm", "E_norm"), ("f3_L_norm", "L_norm")]), "",
        "## Wyniki CZ3", "",
        _table(rows, [("index", "indeks"), ("theta_over_pi", "theta/pi"), ("status", "status"),
                      ("selected_method", "wybrana metoda"), ("cz3_n_cz", "CZ"), ("baseline_n_cz", "baseline CZ"),
                      ("cz3_depth", "głębokość"), ("cz3_n_1q", "1q"), ("cz3_E_norm", "E_norm"), ("cz3_L_norm", "L_norm"),
                      ("continuation_seconds", "kontynuacja [s]"), ("fallback_seconds", "fallback [s]")]), "",
        f"Łączny czas kontynuacji: {sum(_number(row, 'continuation_seconds') for row in rows if math.isfinite(_number(row, 'continuation_seconds'))):.6g} s. "
        f"Łączny czas fallbacku: {sum(_number(row, 'fallback_seconds') for row in rows if math.isfinite(_number(row, 'fallback_seconds'))):.6g} s.", "",
        "## Pełne obwody", "",
        f"Zapisane wyniki obwodów: {len(circuit_rows)}. "
        f"Poprawne: {sum(row.get('success') is True for row in circuit_rows)}. "
        f"Pozostałe: {sum(row.get('success') is not True for row in circuit_rows)}. "
        "Obwody powstają z zapisanych, zweryfikowanych F3/CZ3. Nieudane punkty bramek są pomijane.", "",
        _table(circuit_rows, [("index", "indeks"), ("theta_over_pi", "theta/pi"), ("state_name", "stan"),
                              ("status", "status"), ("two_qubit_gate_count", "CZ"), ("circuit_depth", "głębokość"),
                              ("fidelity", "fidelity"), ("compile_time_seconds", "czas [s]")]), "",
        "### Koszt składników przed łączeniem bloków", "",
        "Kolumny składników podają łączne koszty przygotowania zakodowanego zera oraz wszystkich bloków F3 i CZ3 "
        "w danym obwodzie, przy osobnej kompilacji bloków do tej samej bazy U/CZ i z tymi samymi ustawieniami. "
        "Suma przed łączeniem bloków może różnić się od wyniku optymalizacji całego obwodu, "
        "która może upraszczać bramki także na granicach bloków.", "",
        _table(circuit_rows, [("index", "indeks"), ("theta_over_pi", "theta/pi"), ("state_name", "stan"),
                              ("preparation_n_cz", "przygotowanie CZ"), ("f3_blocks_n_cz", "bloki F3 CZ"),
                              ("cz3_blocks_n_cz", "bloki CZ3 CZ"), ("unfused_n_cz", "suma przed łączeniem CZ"),
                              ("two_qubit_gate_count", "pełny obwód CZ")]), "",
        "Odpowiadające koszty 1q zapisano w [full_circuits.csv](full_circuits.csv): "
        "`preparation_n_1q`, `f3_blocks_n_1q`, `cz3_blocks_n_1q`, `unfused_n_1q` "
        "oraz `one_qubit_gate_count` dla pełnego obwodu. "
        "Znak — oznacza brak zapisanych danych, w tym w starszych artefaktach; nie oznacza zerowego kosztu. "
        "Raport nie odtwarza brakującego podziału na podstawie kosztu końcowego.", "",
        "## Parametry i ograniczenia interpretacji", "",
        "Surowe kąty zapisano bez zmian. `unwrap` stosuje się oddzielnie do każdego spójnego segmentu poprawnych "
        "punktów jednego `template_id`; nie przechodzi przez lukę ani zmianę schematu. "
        "Brak poprawnej kontynuacji oznacza pustą komórkę CSV i przerwę na wykresie, nawet jeśli fallback znalazł poprawny obwód. "
        "Powtórzone parametry rodzica nie są wynikiem bieżącego theta.", "",
        "Kąty Eulera nie mają jednoznacznej reprezentacji i mogą zmieniać się skokowo przy ciągłym operatorze. "
        "Wykres po `unwrap` nie dowodzi gładkości ani istnienia rozwiązania analitycznego. "
        "Dodatkowo pokazano `min_gamma ||U_j(theta_i) - exp(i gamma) U_j(theta_(i-1))||_F`, "
        "liczone z Qiskit UGate dla sąsiednich poprawnych punktów tego samego schematu. "
        "Ta miara usuwa fazę globalną pojedynczej U3 i respektuje jej okresowość; nadal nie stanowi dowodu gładkości "
        "pełnego obwodu ani nie usuwa dowolnego przenoszenia gauge między różnymi bramkami.", "",
        "## Pliki", "",
        "[points.csv](points.csv) zawiera wszystkie ukończone punkty, również nieudane. "
        "[attempts.csv](attempts.csv) zachowuje próby pośrednie i ich rodziców. "
        "[parameters.csv](parameters.csv) zawiera surowe i osobno unwrapped kąty, "
        "[operator_distances.csv](operator_distances.csv) odległości lokalnych operatorów, "
        "a [full_circuits.csv](full_circuits.csv) wyniki pełnych obwodów. "
        "JSON, NPY i QPY pozostają źródłem danych; raport odczytuje je po sprawdzeniu hashy kompletnych pakietów.", "",
    ]
    for name, caption in figures:
        lines.extend([f"![{name}](figures/{name}.png)", "", f"{caption} [PDF](figures/{name}.pdf).", ""])
    errors = [row for row in rows if row.get("errors") or row.get("fallback_reason")]
    if errors:
        lines.extend(["## Przyczyny fallbacku i błędy", "", _table(errors, [
            ("index", "indeks"), ("fallback_reason", "przyczyna fallbacku"), ("errors", "zapis błędów")]), ""])
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
