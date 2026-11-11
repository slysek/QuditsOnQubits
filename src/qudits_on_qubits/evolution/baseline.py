"""Check the saved F3 and CZ3 circuits and write a baseline report."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import platform
import shlex
import subprocess

import numpy as np
import qiskit
from qiskit import qpy, transpile
import scipy

from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import validate_code_space_gate

from .artifacts import file_hash, load_bytes, load_circuit, load_encoding
from .encodings import canonical_encoding, tensor_encoding
from .evaluator import evaluate
from .models import EvaluationTask
from .targets import controlled_z_d, fourier_d

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909"
HASHES = {
    "E.npy": "4829b0594b7769b357789834467f875ad10deabd185296858671119155dc6e21",
    "F3_W.qpy": "4f337e4c1e92a48cd84be7a1d9cc42dbd2904cf65ec2b7bc513d9b75018167af",
    "CZ3_W.qpy": "a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791",
}
LEGACY = ROOT / "src/qudits_on_qubits/quantum_circuits"
LEGACY_HASHES = {
    "Fgate3.qpy": "bb8c93c83c1b183d3eebe3a9a1741c804cd90e25a4a4e6ca46cb567c2c15315c",
    "CZgate3.qpy": "05b1a62af645198aee186bd40a12fbcc67cce83bc7d182e18548881a196e3689",
}
# The protocol uses unnormalized Frobenius norms.
CASES = (("F3", 1, fourier_d, 1e-10, "F3_W.qpy", False),
         ("CZ3", 2, controlled_z_d, 1e-5, "CZ3_W.qpy", False),
         ("F3", 1, fourier_d, 1e-10, "Fgate3.qpy", True),
         ("CZ3", 2, controlled_z_d, 1e-5, "CZgate3.qpy", True))
STRICT_TOLERANCE = 1e-10


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_baseline(output: Path, *, input_dir: Path = INPUT,
                 candidate_dirs: tuple[Path, ...] = ()) -> list[dict]:
    """Check the input circuits. Write the report to a new directory."""
    output = Path(output)
    input_dir = Path(input_dir)
    encoding = load_encoding(input_dir / "E.npy", HASHES["E.npy"], 3)
    if not np.array_equal(encoding, canonical_encoding(3)):
        raise ValueError("The input encoding is not canonical")
    rows = []
    artifacts = {}
    for name, n, target_factory, tolerance, filename, legacy in CASES:
        source = (LEGACY if legacy else input_dir) / filename
        source_hash = (LEGACY_HASHES if legacy else HASHES)[filename]
        circuit = load_circuit(source, source_hash, 2 * n)
        if legacy:
            circuit = transpile(circuit, basis_gates=["u", "cx" if name == "F3" else "cz"],
                                optimization_level=0, seed_transpiler=0)
            for index in reversed(range(len(circuit.data))):
                if circuit.data[index].operation.name == "barrier":
                    del circuit.data[index]
        target = target_factory(3)
        dimension = 3**n
        task = EvaluationTask(3, n, target,
                              logical_error_tolerance=tolerance / np.sqrt(dimension),
                              leakage_tolerance=tolerance**2 / dimension,
                              allowed_gates=("u", "u3", "cx" if name == "F3" else "cz"))
        result = evaluate(task, circuit, [encoding] * n)
        embedding = tensor_encoding([encoding] * n)
        old = validate_code_space_gate(circuit, embedding, target)
        leakage_norm = np.sqrt(dimension * result.L)
        if not np.isclose(result.error_norm, old.E_norm, atol=1e-13, rtol=1e-7):
            raise ValueError(f"The {name} error norms do not agree")
        if not np.isclose(leakage_norm, old.L_norm, atol=1e-13, rtol=1e-7):
            raise ValueError(f"The {name} leakage norms do not agree")
        if result.two_qubit_gate_count != old.N_2q:
            raise ValueError(f"The {name} gate counts do not agree")
        valid = result.error_norm <= tolerance and leakage_norm <= tolerance
        if valid != result.is_valid:
            raise ValueError(f"The {name} validity results do not agree")
        row = {
            "gate": name, "method": "legacy_transpile" if legacy else "saved_reference", "encoding": "canonical",
            **asdict(result), "error_frobenius": result.error_norm,
            "leakage_frobenius": float(leakage_norm), "tolerance_frobenius": tolerance,
            "logical_error_tolerance": task.logical_error_tolerance,
            "leakage_tolerance": task.leakage_tolerance,
            "strict_tolerance_frobenius": STRICT_TOLERANCE,
            "strict_valid": bool(result.error_norm <= STRICT_TOLERANCE and leakage_norm <= STRICT_TOLERANCE),
            "previous_error_frobenius": old.E_norm,
            "previous_leakage_frobenius": old.L_norm,
            "circuit_source": str(source), "source_sha256": source_hash,
            "decomposition": "qiskit transpile level 0 seed 0, remove barriers" if legacy else "none",
            "circuit_artifact": (f"legacy_{name}.qpy" if legacy else filename) if valid else None,
            "target_artifact": f"{name}_target.npy" if valid else None,
            "encoding_artifacts": json.dumps(["E.npy"] * n),
        }
        rows.append(row)
        if valid:
            if legacy:
                buffer = BytesIO()
                qpy.dump(circuit, buffer)
                artifacts[f"legacy_{name}.qpy"] = buffer.getvalue()
            else:
                artifacts[filename] = load_bytes(source, source_hash)
            artifacts[f"{name}_target.npy"] = target
    if not all(row["is_valid"] for row in rows):
        raise ValueError("A reference circuit did not pass its tolerance")

    candidate_inputs = []
    for index, directory in enumerate(candidate_dirs, start=1):
        directory = Path(directory)
        names = ("circuit.qpy", "E_in_A.npy", "E_in_B.npy", "E_out_A.npy", "E_out_B.npy", "meta.json")
        raw = {name: (directory / name).read_bytes() for name in names}
        candidate_inputs.append({"directory": str(directory),
                                 "sha256": {name: file_hash(data) for name, data in raw.items()},
                                 "hash_check": "recorded_only"})
        metadata = json.loads(raw["meta.json"])
        if not isinstance(metadata, dict) or metadata.get("target") != "CZ3":
            raise ValueError("The candidate target must be CZ3")
        encodings = [np.load(BytesIO(raw[f"E_in_{part}.npy"]), allow_pickle=False)
                     for part in ("A", "B")]
        for part, encoded in zip(("A", "B"), encodings, strict=True):
            output_encoding = np.load(BytesIO(raw[f"E_out_{part}.npy"]), allow_pickle=False)
            if not np.array_equal(encoded, output_encoding):
                raise ValueError(f"The input and output encodings for {part} must match")
        embedding = tensor_encoding(encodings, d=3)
        circuits = qpy.load(BytesIO(raw["circuit.qpy"]))
        if len(circuits) != 1 or circuits[0].num_qubits != 4:
            raise ValueError("The candidate must contain one four-qubit circuit")
        circuit = circuits[0]
        tolerance = 1e-5
        target = controlled_z_d(3)
        task = EvaluationTask(3, 2, target, logical_error_tolerance=tolerance / 3,
                              leakage_tolerance=tolerance**2 / 9,
                              allowed_gates=("u", "u3", "cz"))
        result = evaluate(task, circuit, encodings)
        old = validate_code_space_gate(circuit, embedding, target)
        leakage_norm = float(np.sqrt(9 * result.L))
        if (not np.isclose(result.error_norm, old.E_norm, atol=1e-13, rtol=1e-7)
                or not np.isclose(leakage_norm, old.L_norm, atol=1e-13, rtol=1e-7)
                or result.two_qubit_gate_count != old.N_2q):
            raise ValueError("The candidate metrics do not match the previous validator")
        prefix = f"candidate_{index:03d}"
        circuit_artifact = f"{prefix}_CZ3.qpy" if result.is_valid else None
        encoding_artifacts = [f"{prefix}_E_A.npy", f"{prefix}_E_B.npy"] if result.is_valid else []
        row = {
            "gate": "CZ3", "method": "candidate", "encoding": directory.name,
            **asdict(result), "error_frobenius": result.error_norm,
            "leakage_frobenius": leakage_norm, "tolerance_frobenius": tolerance,
            "logical_error_tolerance": task.logical_error_tolerance,
            "leakage_tolerance": task.leakage_tolerance,
            "strict_tolerance_frobenius": STRICT_TOLERANCE,
            "strict_valid": bool(result.error_norm <= STRICT_TOLERANCE and leakage_norm <= STRICT_TOLERANCE),
            "previous_error_frobenius": old.E_norm, "previous_leakage_frobenius": old.L_norm,
            "circuit_source": str(directory / "circuit.qpy"),
            "source_sha256": candidate_inputs[-1]["sha256"]["circuit.qpy"], "decomposition": "none",
            "circuit_artifact": circuit_artifact,
            "target_artifact": "CZ3_target.npy" if result.is_valid else None,
            "encoding_artifacts": json.dumps(encoding_artifacts),
        }
        rows.append(row)
        if result.is_valid:
            artifacts[circuit_artifact] = raw["circuit.qpy"]
            for name, encoded in zip(encoding_artifacts, encodings, strict=True):
                artifacts[name] = encoded

    version = {"git_commit": _git("rev-parse", "HEAD"), "git_status": _git("status", "--short"),
               "python": platform.python_version(), "qiskit": qiskit.__version__,
               "numpy": np.__version__, "scipy": scipy.__version__}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        if isinstance(data, np.ndarray):
            np.save(output / name, data, allow_pickle=False)
        else:
            (output / name).write_bytes(data)
    np.save(output / "E.npy", encoding, allow_pickle=False)
    for row in rows:
        row["artifact_sha256"] = (file_hash((output / row["circuit_artifact"]).read_bytes())
                                  if row["circuit_artifact"] else None)
    with (output / "baseline_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    five_cz = any(row["is_valid"] and row["cz_count"] == 5 and row["cx_count"] == 0
                  for row in rows if row["gate"] == "CZ3")
    _json(output / "baseline_summary.json", {"results": rows,
          "five_cz": "verified" if five_cz else "not_in_this_baseline"})
    table = "\n".join(
        f"| {row['gate']} | {row['method']} | {row['encoding']} | {row['cz_count']} | {row['cx_count']} | {row['F']:.16g} | "
        f"{row['L']:.6g} | {row['error_frobenius']:.6g} | {row['is_valid']} | "
        f"{row['strict_valid']} |" for row in rows
    )
    command = "python -m qudits_on_qubits.evolution.baseline --output artifacts/evolution/baseline_qutrit_new"
    for directory in candidate_dirs:
        command += " --candidate-dir " + shlex.quote(str(directory))
    (output / "README.md").write_text(
        "# Qutrit baseline\n\n"
        "Run this command from the repository root. Use a new output directory.\n\n"
        "```sh\n" + command + "\n```\n\n"
        "The leakage column gives the average probability. Error is the Frobenius norm.\n"
        "The saved protocol uses 1e-10 for F3 and 1e-5 for CZ3. "
        "The strict test uses 1e-10 for both norms.\n\n"
        "| Gate | Method | Encoding | CZ | CX | Fidelity | Leakage | Error | Valid | Strict valid |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n" + table + "\n\n"
        "For legacy circuits, Qiskit transpiles to U/CX or U/CZ at level 0 with seed 0. "
        "The script removes barriers before evaluation.\n\n"
        + ("This report includes a verified 5-CZ circuit.\n" if five_cz
           else "No verified 5-CZ circuit is included in this report.\n")
        + "Candidate encodings map A to qubits 0 and 1, and B to qubits 2 and 3.\n"
        + "Candidate input hashes are recorded. No trusted input hashes were supplied.\n", encoding="utf-8",
    )
    files = {path.name: file_hash(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}
    config = {"tolerance": {name: tol for name, _, _, tol, _, _ in CASES},
              "strict_tolerance": STRICT_TOLERANCE,
              "basis": {"F3": ["u", "u3", "cx"], "CZ3": ["u", "u3", "cz"]}}
    code_files = [*sorted((ROOT / "src/qudits_on_qubits/evolution").glob("*.py")),
                  ROOT / "src/qudits_on_qubits/benchmarks/validation.py",
                  ROOT / "src/qudits_on_qubits/benchmarks/direct_basis/optimized_gates.py",
                  ROOT / "src/qudits_on_qubits/benchmarks/direct_basis/math_utils.py"]
    _json(output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "version": version,
        "inputs": {name: {"path": str(input_dir / name), "sha256": sha} for name, sha in HASHES.items()},
        "legacy_inputs": {name: {"path": str(LEGACY / name), "sha256": sha} for name, sha in LEGACY_HASHES.items()},
        "candidate_inputs": candidate_inputs,
        "files": files, "seed": {"synthesis": None, "legacy_transpiler": 0},
        "synthesis": False, "configuration": config,
        "configuration_sha256": file_hash(json.dumps(config, sort_keys=True).encode()),
        "code_sha256": {str(path.relative_to(ROOT)): file_hash(path.read_bytes()) for path in code_files},
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check saved qutrit circuits and write a baseline report.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the report.")
    parser.add_argument("--candidate-dir", type=Path, action="append", default=[],
                        help="Directory with a CZ3 candidate. Repeat for multiple candidates.")
    args = parser.parse_args()
    rows = run_baseline(args.output, candidate_dirs=tuple(args.candidate_dir))
    for row in rows:
        print(f"{row['gate']}: CZ={row['cz_count']} CX={row['cx_count']} "
              f"F={row['F']:.16g} L={row['L']:.6g} valid={row['is_valid']}")


if __name__ == "__main__":
    main()
