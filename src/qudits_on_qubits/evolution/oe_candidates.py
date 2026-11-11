"""Read literal circuit candidates without executing their code."""

import ast
import math
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, U3Gate, UGate

from .skeleton_fit import MAX_SKELETON_CZ

MAX_BYTES = 131_072
MAX_AST_NODES = 8192
MAX_AST_DEPTH = 16
MAX_OPERATIONS = 512
MAX_ANGLE = 1_000_000.0
# A free-encoding AME(4,6) skeleton with 128 CZs has 3 * 12 + 6 * (128 + 52) = 1116 angles.
MAX_ANGLES = 4096
# Qutrit states use 4, 6 or 8 qubits; an AME(4,6) state uses 12.
WIDTHS = (4, 6, 8, 12)


def _angle(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("The angle must be a finite real number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("The angle must be a finite real number") from exc
    if not math.isfinite(value) or abs(value) > MAX_ANGLE:
        raise ValueError("The angle is outside the permitted range")
    return value


def _wire(value, width):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < width:
        raise ValueError("The qubit index is outside the circuit")
    return value


def candidate_from_circuit(circuit):
    """Return the literal data for a U/U3/CZ circuit."""
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits not in WIDTHS:
        raise ValueError("The circuit must have 4, 6, 8 or 12 qubits")
    if circuit.num_ancillas or circuit.num_clbits or circuit.num_vars or circuit.num_parameters:
        raise ValueError("The circuit must have no ancillas, classical bits or parameters")
    operations = []
    for item in circuit.data:
        gate = item.operation
        if gate.base_class not in (UGate, U3Gate, CZGate):
            raise ValueError("The circuit has a gate that is not supported")
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if gate.base_class is CZGate:
            operations.append(["cz", *wires])
        else:
            operations.append([gate.name, *wires, *[_angle(p) for p in gate.params]])
    result = {"num_qubits": circuit.num_qubits, "operations": operations,
              "global_phase": _angle(circuit.global_phase)}
    _circuit(result, circuit.num_qubits)
    return result


def _circuit(value, expected_qubits):
    _check_width(expected_qubits)
    if not isinstance(value, dict) or set(value) - {"num_qubits", "operations", "global_phase"}:
        raise ValueError("The candidate has invalid fields")
    width = value.get("num_qubits")
    if isinstance(width, bool) or not isinstance(width, int) or width != expected_qubits:
        raise ValueError("The candidate circuit width does not match")
    operations = value.get("operations")
    if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
        raise ValueError("The candidate operation count is invalid")
    circuit = QuantumCircuit(width, global_phase=_angle(value.get("global_phase", 0)))
    for operation in operations:
        if not isinstance(operation, list) or not operation:
            raise ValueError("Each operation must be a list")
        if operation[0] in ("u", "u3") and len(operation) == 5:
            wire = _wire(operation[1], width)
            theta, phi, lam = [_angle(value) for value in operation[2:]]
            gate = UGate(theta, phi, lam) if operation[0] == "u" else U3Gate(theta, phi, lam)
            circuit.append(gate, [wire])
        elif operation[0] == "cz" and len(operation) == 3:
            left, right = [_wire(value, width) for value in operation[1:]]
            if left == right:
                raise ValueError("CZ needs two different qubits")
            circuit.cz(left, right)
        else:
            raise ValueError("The candidate gate is not supported")
    return circuit


def _check_width(expected_qubits):
    if isinstance(expected_qubits, bool) or not isinstance(expected_qubits, int) or expected_qubits not in WIDTHS:
        raise ValueError("The circuit width must be 4, 6, 8 or 12 qubits")


def _read_literal(path):
    """Return the literal CANDIDATE dictionary without running the file."""
    with Path(path).open("rb") as stream:
        source = stream.read(MAX_BYTES + 1)
    if len(source) > MAX_BYTES:
        raise ValueError("The candidate file is too large")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, UnicodeError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError("The candidate syntax is invalid") from exc
    stack = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            raise ValueError("The candidate syntax is too large or too deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    statements = tree.body
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) and isinstance(statements[0].value.value, str):
        statements = statements[1:]
    if (len(statements) != 1 or not isinstance(statements[0], ast.Assign)
            or len(statements[0].targets) != 1
            or not isinstance(statements[0].targets[0], ast.Name)
            or statements[0].targets[0].id != "CANDIDATE"):
        raise ValueError("The file must contain one CANDIDATE assignment")
    data_node = statements[0].value
    if not isinstance(data_node, ast.Dict) or any(not isinstance(key, ast.Constant) or not isinstance(key.value, str) for key in data_node.keys):
        raise ValueError("The candidate must be a literal dictionary")
    keys = [key.value for key in data_node.keys]
    if len(keys) != len(set(keys)):
        raise ValueError("The candidate has duplicate fields")
    try:
        return ast.literal_eval(data_node)
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError) as exc:
        raise ValueError("The candidate must contain literal values") from exc


def load_candidate(path, *, expected_qubits):
    """Read one literal candidate and build its circuit."""
    _check_width(expected_qubits)
    return _circuit(_read_literal(path), expected_qubits)


def load_skeleton(path, *, expected_qubits):
    """Read a literal CZ skeleton {"num_qubits": W, "cz": [[a, b], ...]} with optional "angles"."""
    _check_width(expected_qubits)
    data = _read_literal(path)
    if not isinstance(data, dict) or not {"num_qubits", "cz"} <= set(data) <= {"num_qubits", "cz", "angles"}:
        raise ValueError("The skeleton must have the fields num_qubits and cz, and optionally angles")
    width = data["num_qubits"]
    if isinstance(width, bool) or not isinstance(width, int) or width != expected_qubits:
        raise ValueError("The candidate circuit width does not match")
    pairs = data["cz"]
    if not isinstance(pairs, list) or len(pairs) > MAX_SKELETON_CZ:
        raise ValueError(f"The skeleton must have at most {MAX_SKELETON_CZ} CZ pairs")
    result = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each CZ must be a pair of qubit indices")
        left, right = (_wire(value, width) for value in pair)
        if left == right:
            raise ValueError("CZ needs two different qubits")
        result.append((left, right))
    return tuple(result)


def candidate_angles(path):
    """Return the optional ansatz angles of a skeleton file (see skeleton_fit.fit_skeleton), or None."""
    angles = _read_literal(path).get("angles")
    if angles is None:
        return None
    if not isinstance(angles, list) or len(angles) > MAX_ANGLES:
        raise ValueError(f"The angles must be a list of at most {MAX_ANGLES} numbers")
    return [_angle(value) for value in angles]


def skeleton_program(width, pairs, angles=None):
    """Write one CZ per line with its index, so that diffs and edits stay local.

    angles (optional) are the fitted ansatz angles; the evaluator starts its check from them.
    """
    lines = [f"    [{a}, {b}],  # {k}" for k, (a, b) in enumerate(pairs)]
    extra = "" if angles is None else "    \"angles\": [%s],\n" % ", ".join(repr(float(v)) for v in angles)
    return "CANDIDATE = {\n    \"num_qubits\": %d,\n    \"cz\": [\n%s\n    ],\n%s}\n" % (
        width, "\n".join("    " + line for line in lines), extra)
