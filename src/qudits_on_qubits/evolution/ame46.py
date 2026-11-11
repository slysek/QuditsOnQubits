"""Import AME(4,6) source tensors and known circuits as an evolution baseline.

The input is one or more tensor files (state id -> 6 x 6 x 6 x 6 array) and a directory with one circuit file
<state id>.json per state (format ame46-c18-ops-v1 with an ame46-local-basis-v1 local basis).
The output directory gets targets.npz and baseline/ in the format of the other baselines.
Write imported data below artifacts/, which Git does not track.
Public inputs are available in experiment_inputs/ame46/.
"""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile

from .encodings import canonical_encoding
from .state_workloads import AME46_STATES, AME46_TARGETS, AME46_TARGETS_VARIABLE, evaluate_workload

METHOD = "adapted_direct"
MAX_CIRCUIT_BYTES = 1 << 20


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _unitaries(basis, source):
    """Return the four local unitaries U_p of output = (U_A x U_B x U_C x U_D) source."""
    if (basis.get("format") != "ame46-local-basis-v1" or basis.get("direction") != "source_to_output"
            or basis.get("party_order") != ["A", "B", "C", "D"]):
        raise ValueError("The local basis must be ame46-local-basis-v1, source_to_output, parties A to D")
    if basis.get("source_sha256") != _hash(np.asarray(source, dtype="<c16").tobytes(order="C")):
        raise ValueError("The local basis belongs to a different source tensor")
    raw = np.asarray(basis.get("unitaries"), dtype=float)
    if raw.shape != (4, 6, 6, 2):
        raise ValueError("The local basis must have four 6 x 6 complex matrices")
    matrices = raw[..., 0] + 1j * raw[..., 1]
    for matrix in matrices:
        if not np.allclose(matrix.conj().T @ matrix, np.eye(6), atol=1e-10, rtol=0):
            raise ValueError("Each local basis matrix must be unitary")
    return matrices


def _circuit(data):
    """Return a u/cz circuit. A CX becomes a CZ between two U gates on the target."""
    if data.get("format") != "ame46-c18-ops-v1" or data.get("num_qubits") != 12:
        raise ValueError("The circuit must be ame46-c18-ops-v1 on 12 qubits")
    circuit = QuantumCircuit(12, global_phase=float(data.get("global_phase", 0.0)))
    for op in data.get("ops", []):
        if op[0] == "u" and len(op) == 5:
            circuit.u(*map(float, op[2:]), int(op[1]))
        elif op[0] == "cx" and len(op) == 3:
            circuit.cx(int(op[1]), int(op[2]))
        else:
            raise ValueError(f"The operation {op[0]!r} is not supported")
    return transpile(circuit, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)


def _check(rows, files, targets):
    """Check each circuit against its tensor. The check reads the new tensor file through the variable."""
    old = os.environ.get(AME46_TARGETS_VARIABLE)
    os.environ[AME46_TARGETS_VARIABLE] = str(targets)
    try:
        for row in rows:
            row.update(evaluate_workload(row["workload"], *files[row["workload"]], tolerance=1e-10))
            if not row["is_valid"]:
                raise ValueError(f"The circuit of {row['workload']} does not prepare its state (F={row['F']:.6f})")
    finally:
        if old is None:
            os.environ.pop(AME46_TARGETS_VARIABLE, None)
        else:
            os.environ[AME46_TARGETS_VARIABLE] = old


def _tensors(targets):
    """Read the harness states from one or more tensor files. A state must be in one file only."""
    tensors = {}
    for path in [targets] if isinstance(targets, (str, os.PathLike)) else targets:
        with np.load(path, allow_pickle=False) as data:
            for name in AME46_STATES:
                if name not in data:
                    continue
                if name in tensors:
                    raise ValueError(f"The state {name} is in more than one tensor file")
                tensors[name] = np.asarray(data[name], dtype=complex)
    return tensors


def import_ame46(targets, circuits, output):
    """Check each known circuit against its source tensor and write targets.npz and baseline/."""
    output = Path(output)
    baseline = output / "baseline"
    if baseline.exists() or (output / "targets.npz").exists():
        raise ValueError(f"{output} already has AME(4,6) data. Remove it first")
    tensors = _tensors(targets)
    rows, files, artifacts = [], {}, {}
    for name, source in tensors.items():
        path = Path(circuits) / f"{name}.json"
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_CIRCUIT_BYTES:
            raise ValueError(f"{path} is too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("state_id") != name:
            raise ValueError(f"{path} is for a different state")
        # E_p = B U_p: the harness target is (E_A x E_B x E_C x E_D) source.
        encodings = np.asarray([canonical_encoding(6) @ u for u in _unitaries(data["local_basis"], source)])
        circuit = _circuit(data["circuit"])
        circuit_bytes, encoding_bytes = BytesIO(), BytesIO()
        qpy.dump(circuit, circuit_bytes)
        np.save(encoding_bytes, encodings, allow_pickle=False)
        artifacts[f"{name}.qpy"] = circuit_bytes.getvalue()
        artifacts[f"{name}_encoding.npy"] = encoding_bytes.getvalue()
        rows.append({"workload": name, "method": METHOD, "circuit_artifact": f"{name}.qpy",
                     "encoding_artifact": f"{name}_encoding.npy", "source": str(path.resolve()),
                     "source_file_sha256": _hash(path.read_bytes())})
        files[name] = (circuit, encodings)
    if not rows:
        raise ValueError(f"No circuit in {circuits} matches a state of {', '.join(AME46_STATES)}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "targets.npz", **tensors)
    try:
        _check(rows, files, output / "targets.npz")
    except ValueError:
        (output / "targets.npz").unlink()
        raise
    baseline.mkdir()
    for name, raw in artifacts.items():
        (baseline / name).write_bytes(raw)
    (baseline / "summary.json").write_text(json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8")
    manifest = {"files": {name: _hash(raw) for name, raw in artifacts.items()},
                "targets_sha256": _hash((output / "targets.npz").read_bytes())}
    (baseline / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import AME(4,6) tensors and known circuits as a baseline.")
    parser.add_argument("--targets", type=Path, nargs="+", required=True,
                        help="One or more tensor files: state id -> 6 x 6 x 6 x 6 array.")
    parser.add_argument("--circuits", type=Path, required=True, help="Directory with one <state id>.json per state.")
    parser.add_argument("--output", type=Path, default=AME46_TARGETS.parent,
                        help="Output directory. The default is the location that the harness reads.")
    args = parser.parse_args(argv)
    try:
        rows = import_ame46(args.targets, args.circuits, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row['workload']}: {row['cz_count']} CZ, two-qubit depth {row['two_qubit_depth']}, valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
