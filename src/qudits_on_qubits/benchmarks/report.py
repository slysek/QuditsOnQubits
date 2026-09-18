"""Derived tables and a Pareto figure; source evidence lives in immutable bundles."""
from __future__ import annotations

from pathlib import Path


def write_report(result) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = result.output_dir
    result.trials.to_csv(root / "trials.csv", index=False)
    result.statistics.to_csv(root / "statistics.csv", index=False)
    result.pareto_front.to_csv(root / "pareto.csv", index=False)
    figure, axes = plt.subplots(figsize=(8, 5))
    valid = result.statistics[result.statistics.pareto_eligible]
    for (comparison, family), group in valid.groupby(["comparison_id", "class_name"]):
        axes.scatter(group.mean_two_qubit_gate_count, group.mean_depth,
                     marker="*" if family == "canonical" else "o",
                     label=f"{family} [{comparison[:8]}]", s=80 if family == "canonical" else 35)
    axes.set_xlabel("Mean native two-qubit gate count after routing")
    axes.set_ylabel("Mean circuit depth after routing")
    axes.set_title("Encoding candidates (separate fronts per comparison)")
    if len(valid):
        axes.legend(fontsize=8)
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(root / "pareto.png", dpi=160)
    figure.savefig(root / "pareto.pdf")
    plt.close(figure)
    backend = result.manifest["backend"]
    front = result.pareto_front
    lines = [
        "# Encoding benchmark", "",
        f"Backend: {backend.get('provider', '')} / {backend.get('name', '')}",
        f"Target: {result.manifest['target_hash']}", "",
        "Costs include native compilation and routing. This is an offline compilation benchmark, not a QPU run.",
        "Fronts use mean two-qubit gate count, mean depth, and population depth standard deviation.",
        "Every planned seed must pass the gate and state checks. Failed trials remain in trials.csv.",
        "Each synthesis protocol has its own canonical reference and Pareto boundary.", "",
        f"Trials: {len(result.trials)}; accepted: {int(result.trials.success.sum())}; nondominated alternatives: {len(front)}.", "",
        "| Candidate | Protocol | Mean 2q | Mean depth | Depth std |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in front.itertuples():
        lines.append(f"| {row.candidate_name} | {row.comparison_id[:12]} | {row.mean_two_qubit_gate_count:.3f} | {row.mean_depth:.3f} | {row.std_depth:.3f} |")
    lines += ["", "The front does not imply a global optimum or guarantee improvement over canonical.",
              "For a selected alternative, best_graph_state_transpiled_qpy identifies one actual trial; its metrics need not equal the means.",
              "", "![Pareto candidates](pareto.png)", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
