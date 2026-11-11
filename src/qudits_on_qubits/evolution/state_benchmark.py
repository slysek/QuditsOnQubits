"""Build and check saved state circuits."""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import Isometry, StatePreparation

from .classical import _completed
from .encodings import canonical_encoding
from .evaluator import evaluate
from .models import EvaluationTask
from .state_workloads import (
    AME_GRAPH_EDGES, GENERATED_BASELINES, _gf8_times_w, _prepare, build_workload, encoded_target, evaluate_workload, layout,
)
from .targets import controlled_z_d, fourier_d

TOLERANCE = 1e-10
WORKLOADS = ("two_qutrit", "ghz3", "ame43")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _phase_graph_reference(state: str) -> QuantumCircuit:
    """Build the encoded phase graph with binary controlled-phase gates."""
    n, d, q = layout(state)
    vector = canonical_encoding(d) @ (np.ones(d) / np.sqrt(d))
    if q == 2:
        preparation = _prepare(vector)
    else:
        preparation = QuantumCircuit(q)
        preparation.append(StatePreparation(vector), range(q))
    circuit = QuantumCircuit(q * n, name=state)
    for party in range(n):
        circuit.compose(preparation, range(q * party, q * (party + 1)), inplace=True)
    edges = ((0, 1, 1), (1, 2, 1), (2, 3, 1)) if state == "cluster4" else AME_GRAPH_EDGES
    for left, right, weight in edges:
        for a in range(q):
            for b in range(q):
                circuit.cp(2 * np.pi * weight * 2**(a + b) / d, q * left + a, q * right + b)
    return circuit


def _state_reference(state: str) -> QuantumCircuit:
    """Build a canonical reference with U and CZ gates and no additional qubits."""
    n, _, q = layout(state)
    if state in ("cluster4", "ame45", "ame47"):
        circuit = _phase_graph_reference(state)
    elif state == "ame48":
        circuit = QuantumCircuit(12, name=state)
        circuit.h(range(6))
        for bit in range(3):
            circuit.cx(bit, 6 + bit)
            circuit.cx(3 + bit, 6 + bit)
            circuit.cx(bit, 9 + bit)
            image = _gf8_times_w(1 << bit)
            for target in range(3):
                if image & (1 << target):
                    circuit.cx(3 + bit, 9 + target)
    else:
        circuit = QuantumCircuit(q * n, name=state)
        target, _ = encoded_target(state, canonical_encoding(3))
        left_width = q * (n // 2)
        matrix = target.reshape(1 << (q * n - left_width), 1 << left_width).T
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        # shortcut: the reference uses four Schmidt terms, enlarge the bond for targets with a higher rank.
        if np.linalg.norm(singular[4:]) > TOLERANCE:
            raise ValueError("The reference needs more than four Schmidt terms")
        circuit.compose(_prepare(singular[:4] / np.linalg.norm(singular[:4])), [0, 1], inplace=True)
        for bit in range(2):
            circuit.cx(bit, left_width + bit)
        circuit.append(Isometry(left[:, :4], 0, 0), range(left_width))
        circuit.append(Isometry(right[:4, :].T, 0, 0), range(left_width, q * n))
    return transpile(circuit, basis_gates=["u", "cz"], optimization_level=3, seed_transpiler=0)


def ensure_state_baseline(state: str, output: Path) -> None:
    """Create the canonical reference if a supported target has no baseline directory."""
    if state not in GENERATED_BASELINES:
        raise ValueError("Select a supported state benchmark")
    output = Path(output)
    if output.exists():
        return
    circuit = _state_reference(state)
    encoding = canonical_encoding(layout(state)[1])
    metrics = evaluate_workload(state, circuit, encoding, tolerance=TOLERANCE)
    if not metrics["strict_valid"]:
        raise ValueError("The reference circuit does not pass the strict limit")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "circuit.qpy").open("xb") as stream:
        qpy.dump(circuit, stream)
    np.save(output / "encoding.npy", encoding, allow_pickle=False)
    row = {"workload": state, "method": "canonical_direct", **metrics,
           "circuit_artifact": "circuit.qpy", "encoding_artifact": "encoding.npy"}
    (output / "summary.json").write_bytes(_json({"results": [row]}))
    files = {path.name: _hash(path.read_bytes()) for path in output.iterdir() if path.is_file()}
    (output / "manifest.json").write_bytes(_json({
        "configuration": {"state": state, "method": "canonical_direct", "tolerance": TOLERANCE,
                          "reference": ("gf8_linear" if state == "ame48" else "qudit_graph"
                                        if state in ("cluster4", "ame45", "ame47") else "schmidt_isometries"),
                          "seed_transpiler": 0},
        "versions": {"numpy": np.__version__, "qiskit": qiskit.__version__}, "files": files,
    }))


# Keep the original helper name for existing imports and running campaigns.
ensure_qutrit_baseline = ensure_state_baseline


def _baseline_f3(m1: Path):
    manifest = json.loads((m1 / "manifest.json").read_bytes())
    raw = (m1 / "F3_W.qpy").read_bytes()
    if _hash(raw) != manifest["files"].get("F3_W.qpy"):
        raise ValueError("The F3 circuit hash does not match")
    circuits = qpy.load(BytesIO(raw))
    if len(circuits) != 1:
        raise ValueError("The F3 file must contain one circuit")
    task = EvaluationTask(3, 1, fourier_d(3), TOLERANCE / np.sqrt(3), TOLERANCE**2 / 3,
                          ("u", "u3", "cx"))
    if not evaluate(task, circuits[0], [canonical_encoding(3)]).is_valid:
        raise ValueError("The F3 circuit does not pass the limit")
    return circuits[0], _hash(raw)


def _baseline_estar(m1: Path):
    manifest = json.loads((m1 / "manifest.json").read_bytes())
    summary = json.loads((m1 / "baseline_summary.json").read_bytes())
    rows = [row for row in summary["results"] if row["encoding"] == "CZ3_sym_k5_Estar"
            and row["is_valid"] and row["strict_valid"]]
    if len(rows) != 1:
        raise ValueError("The M1 report must contain one valid E-star case")
    names = json.loads(rows[0]["encoding_artifacts"])
    if len(names) != 2:
        raise ValueError("The E-star case must contain two encodings")
    encodings = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("The encoding path is not valid")
        raw = (m1 / name).read_bytes()
        if _hash(raw) != manifest["files"].get(name):
            raise ValueError("The E-star encoding hash does not match")
        encodings.append(np.load(BytesIO(raw), allow_pickle=False))
    if not np.array_equal(encodings[0], encodings[1]):
        raise ValueError("The E-star case must use the same encoding for both qutrits")
    return encodings[0]


def _load_case(m2: Path, name: str):
    report = json.loads((m2 / "report_0001.json").read_bytes())
    matches = [row for row in report["best_strict"] if row["encoding"] == name]
    if len(matches) != 1:
        raise ValueError(f"The M2 report must contain one {name} case")
    path = Path(matches[0]["circuit_artifact"])
    if path.is_absolute() or ".." in path.parts or path.name != "circuit.qpy":
        raise ValueError("The circuit path is not valid")
    attempt = m2 / path.parent
    result = _completed(attempt, m2)
    if result is None or not result.get("strict_valid"):
        raise ValueError("The M2 result does not pass the strict limit")
    raw = {name: (attempt / name).read_bytes() for name in ("circuit.qpy", "E_A.npy", "E_B.npy")}
    circuits = qpy.load(BytesIO(raw["circuit.qpy"]))
    if len(circuits) != 1:
        raise ValueError("The CZ3 file must contain one circuit")
    encoding_a, encoding_b = [np.load(BytesIO(raw[f"E_{part}.npy"]), allow_pickle=False) for part in "AB"]
    if not np.array_equal(encoding_a, encoding_b):
        raise ValueError("The benchmark needs one encoding for all qutrits")
    task = EvaluationTask(3, 2, controlled_z_d(3), TOLERANCE / 3, TOLERANCE**2 / 9,
                          ("u", "u3", "cz"))
    if not evaluate(task, circuits[0], [encoding_a, encoding_b]).is_valid:
        raise ValueError("The CZ3 circuit does not pass the limit")
    return circuits[0], encoding_a, {key: _hash(value) for key, value in raw.items()}, str(attempt)


def run_state_benchmark(m1: Path, m2: Path, output: Path) -> list[dict]:
    """Build circuits for three workloads and write the results."""
    m1, m2, output = Path(m1), Path(m2), Path(output)
    f3, f3_hash = _baseline_f3(m1)
    canonical, canonical_e, canonical_hash, canonical_source = _load_case(m2, "canonical")
    if not np.array_equal(canonical_e, canonical_encoding(3)):
        raise ValueError("The canonical encoding does not match")
    estar, estar_e, estar_hash, estar_source = _load_case(m2, "CZ3_sym_k5_Estar")
    if not np.array_equal(estar_e, _baseline_estar(m1)):
        raise ValueError("The M2 E-star encoding does not match the M1 encoding")
    variants = (
        ("canonical_f3", canonical_e, canonical, f3),
        ("canonical_direct", canonical_e, canonical, None),
        ("estar_direct", estar_e, estar, None),
    )
    rows = []
    circuits = {}
    for name in WORKLOADS:
        for method, encoding, cz3, preparation in variants:
            circuit = build_workload(name, encoding, cz3, f3=preparation)
            metrics = evaluate_workload(name, circuit, encoding, tolerance=TOLERANCE)
            filename = f"{name}_{method}.qpy"
            rows.append({"workload": name, "method": method, **metrics,
                         "circuit_artifact": filename, "encoding_artifact": f"{method}_E.npy"})
            buffer = BytesIO()
            qpy.dump(circuit, buffer)
            circuits[filename] = buffer.getvalue()
    output.mkdir(parents=True, exist_ok=False)
    for filename, raw in circuits.items():
        (output / filename).write_bytes(raw)
    for method, encoding, _, _ in variants:
        np.save(output / f"{method}_E.npy", encoding, allow_pickle=False)
    for row in rows:
        row["circuit_sha256"] = _hash((output / row["circuit_artifact"]).read_bytes())
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_bytes(_json({"results": rows}))
    command = ("PYTHONPATH=src " + shlex.quote(sys.executable)
               + " -m qudits_on_qubits.evolution.state_benchmark"
               + " --m1 " + shlex.quote(str(m1.resolve()))
               + " --m2 " + shlex.quote(str(m2.resolve()))
               + " --output artifacts/evolution/state_benchmark_new")
    table = "\n".join(
        f"| {row['workload']} | {row['method']} | {row['cz_count']} | {row['cx_count']} | "
        f"{row['F']:.16g} | {row['L']:.6g} | {row['is_valid']} |" for row in rows
    )
    (output / "README.md").write_text(
        "# Qutrit state circuits\n\n"
        "Run this command from the repository root. Use a new output path.\n\n"
        f"```sh\n{command}\n```\n\n"
        "The circuit starts in physical |0...0>. The gate count includes state preparation.\n"
        "The output stays in its selected encoding. The circuit does not include decoding, measurement, or routing.\n"
        "The canonical F3 method uses the saved F3 gate. The direct methods prepare the same input state.\n"
        "The ghz3 circuit prepares a graph state. Local gates map this state to the standard GHZ3 state.\n"
        "The ame43 graph has one edge of weight two. The circuit uses CZ3 inverse for this edge.\n"
        "All rows use U and CZ gates. The limit for state error and leakage norm is 1e-10.\n\n"
        "| Workload | Method | CZ | CX | Fidelity | Leakage probability | Valid |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n" + table + "\n", encoding="utf-8")
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parents[2]
    code_files = [*sorted(code_dir.glob("*.py")),
                  root / "src/qudits_on_qubits/reference_experiments.py",
                  root / "src/qudits_on_qubits/benchmarks/validation.py",
                  root / "src/qudits_on_qubits/benchmarks/direct_basis/math_utils.py"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    config = {"tolerance": TOLERANCE, "gate_set": ["u", "u3", "cz"],
              "initial_state": "physical_zero", "output_encoding": "selected",
              "decoding": False, "measurement": False, "routing": False}
    files = {p.name: _hash(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_bytes(_json({
        "inputs": {"m1": str(m1.resolve()), "m2": str(m2.resolve()),
                   "m1_manifest_sha256": _hash((m1 / "manifest.json").read_bytes()),
                   "m2_report_sha256": _hash((m2 / "report_0001.json").read_bytes()),
                   "F3_W.qpy": f3_hash,
                   "canonical": {"path": canonical_source, "files": canonical_hash},
                   "estar": {"path": estar_source, "files": estar_hash}},
        "configuration": config, "configuration_sha256": _hash(_json(config)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "qiskit": qiskit.__version__, "git_commit": commit},
        "code_sha256": {str(p.relative_to(root)): _hash(p.read_bytes()) for p in code_files},
        "files": files,
    }))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build saved qutrit state circuits.")
    parser.add_argument("--m1", type=Path, required=True, help="Path to the M1 report.")
    parser.add_argument("--m2", type=Path, required=True, help="Path to the M2 report.")
    parser.add_argument("--output", type=Path, required=True, help="Path to a new output directory.")
    args = parser.parse_args()
    for row in run_state_benchmark(args.m1, args.m2, args.output):
        print(f"{row['workload']} {row['method']}: CZ={row['cz_count']} F={row['F']:.16g} valid={row['is_valid']}")


if __name__ == "__main__":
    main()
