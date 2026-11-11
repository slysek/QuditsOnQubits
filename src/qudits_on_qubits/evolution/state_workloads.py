"""Build and check encoded graph, singlet, Dicke, AKLT and AME states.

A party with three or four levels uses two qubits. A party with five to eight levels uses three qubits.
"""

from functools import lru_cache
from itertools import permutations, product
import json
from numbers import Real
import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, CZGate, U3Gate, UGate
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.validation import circuit_metrics
from qudits_on_qubits.reference_experiments import get_reference_experiment

from .artifacts import main_checkout
from .encodings import tensor_encoding
from .evaluator import evaluate
from .models import EvaluationTask
from .targets import controlled_z_d, fourier_d

GRAPH_STATES = ("two_qutrit", "ghz3", "ame43")
QUTRIT_BENCHMARKS = ("aharonov3", "dicke4", "cluster4", "aklt4")
AME_DIMENSIONS = {"ame45": 5, "ame47": 7, "ame48": 8}
AME_GRAPH_EDGES = ((0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 2))
GENERATED_BASELINES = QUTRIT_BENCHMARKS + tuple(AME_DIMENSIONS)
# One representative of each local-unitary class: Bruzda60 is LU-equivalent to Pandya60, and
# HZ_rotated288_real to HZ96. With free encodings, the equivalent states are the same problem.
# Casas36 is the complex AME(4,6) state of Casas et al., arXiv:2504.05394, Eq. 26-27.
AME46_STATES = ("Bruzda60", "Pandya60prime", "HZ96", "Casas36")
# AME(4,4) from two orthogonal Latin squares over GF(4): sum over a, b of |a, b, a+b, a+wb>.
AME44 = "ame44"
# AME(6,4) from the hexacode over GF(4): sum over x, y, z of |x, y, z, x+wy+wz, wx+y+wz, wx+wy+z>.
AME64 = "ame64"
STATES = GRAPH_STATES + (AME44, AME64) + AME46_STATES + GENERATED_BASELINES
AME46_TARGETS_VARIABLE = "QOQ_AME46_TARGETS"
AME46_TARGETS = main_checkout(Path(__file__).resolve().parents[3]) / "artifacts/evolution/ame46/targets.npz"
_GATES = {"u": UGate, "u3": U3Gate, "cz": CZGate, "cx": CXGate}


def _spec(name):
    if name not in GRAPH_STATES:
        raise ValueError("The graph state name must be two_qutrit, ghz3 or ame43")
    return get_reference_experiment(name).state


def layout(name):
    """Return (parties, levels per party, qubits per party). Party k uses the qubits q*k to q*k+q-1."""
    if name in AME46_STATES:
        return 4, 6, 3
    if name == AME44:
        return 4, 4, 2
    if name == AME64:
        return 6, 4, 2
    if name in AME_DIMENSIONS:
        return 4, AME_DIMENSIONS[name], 3
    if name in QUTRIT_BENCHMARKS:
        return (3 if name == "aharonov3" else 4), 3, 2
    if name not in GRAPH_STATES:
        raise ValueError(f"The state name must be one of {', '.join(STATES)}")
    return _spec(name).num_parties, 3, 2


def ame46_targets_path():
    """Return the file with the AME(4,6) source tensors. QOQ_AME46_TARGETS overrides the default."""
    return Path(os.environ.get(AME46_TARGETS_VARIABLE) or AME46_TARGETS)


@lru_cache(maxsize=16)
def _ame46_vector(name, path):
    try:
        with np.load(path, allow_pickle=False) as data:
            tensor = np.asarray(data[name], dtype=complex)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"The AME(4,6) tensor {name!r} is not in {path}. Write the file with"
                         " `qoq-evolution ame46-import`") from exc
    norm = np.linalg.norm(tensor)
    if tensor.shape != (6,) * 4 or not np.isfinite(tensor).all() or norm < 1e-12:
        raise ValueError(f"The AME(4,6) tensor {name!r} must be a finite nonzero 6 x 6 x 6 x 6 array")
    # Tensor axes are the parties A, B, C, D. Party A (qubits 0-2) is the lowest digit.
    vector = (tensor / norm).transpose(3, 2, 1, 0).reshape(-1)
    vector.setflags(write=False)
    return vector


def _gf4_times_w(x):
    """Multiply x = x0 + x1 w in GF(4) (bit 0 is x0) by w: x w = x1 + (x0 + x1) w, because w^2 = w + 1."""
    return (x >> 1) | (((x & 1) ^ (x >> 1)) << 1)


@lru_cache(maxsize=1)
def _ame44_vector():
    vector = np.zeros(4**4, dtype=complex)
    for a in range(4):
        for b in range(4):
            # GF(4) addition is XOR of the bit pairs. Party 0 (a) is the lowest digit.
            vector[a + 4 * b + 16 * (a ^ b) + 64 * (a ^ _gf4_times_w(b))] = 0.25
    vector.setflags(write=False)
    return vector


@lru_cache(maxsize=1)
def _ame64_vector():
    w = _gf4_times_w
    vector = np.zeros(4**6, dtype=complex)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                word = (x, y, z, x ^ w(y) ^ w(z), w(x) ^ y ^ w(z), w(x) ^ w(y) ^ z)
                vector[sum(level * 4**party for party, level in enumerate(word))] = 1 / 8
    vector.setflags(write=False)
    return vector


def _gf8_times_w(x: int) -> int:
    """Multiply a three-bit field element by w, with w^3 = w + 1."""
    return (x << 1) ^ (0b1011 if x & 4 else 0)


@lru_cache(maxsize=3)
def _ame_benchmark_vector(name: str) -> np.ndarray:
    """Return the AME target with party 0 as the lowest digit."""
    d = AME_DIMENSIONS[name]
    if d == 8:
        vector = np.zeros(8**4, dtype=complex)
        for a in range(8):
            for b in range(8):
                vector[a + 8 * b + 64 * (a ^ b) + 512 * (a ^ _gf8_times_w(b))] = 1 / 8
    else:
        indices = np.arange(d**4)
        digits = [(indices // d**k) % d for k in range(4)]
        phase = sum(weight * digits[a] * digits[b] for a, b, weight in AME_GRAPH_EDGES)
        vector = np.exp(2j * np.pi * phase / d) / d**2
    vector.setflags(write=False)
    return vector


@lru_cache(maxsize=4)
def _qutrit_benchmark_vector(name: str) -> np.ndarray:
    """Return a normalized qutrit target with party 0 as the lowest digit."""
    n, _, _ = layout(name)
    vector = np.zeros(3**n, dtype=complex)
    if name == "aharonov3":
        for word in permutations(range(3)):
            inversions = sum(word[a] > word[b] for a in range(3) for b in range(a + 1, 3))
            vector[sum(level * 3**k for k, level in enumerate(word))] = (-1)**inversions / np.sqrt(6)
    elif name == "dicke4":
        for word in set(permutations((0, 0, 1, 2))):
            vector[sum(level * 3**k for k, level in enumerate(word))] = 1 / np.sqrt(12)
    elif name == "cluster4":
        indices = np.arange(3**4)
        digits = [(indices // 3**k) % 3 for k in range(4)]
        phase = sum(digits[k] * digits[k + 1] for k in range(3))
        vector = np.exp(2j * np.pi * phase / 3) / 9
    elif name == "aklt4":
        matrices = (
            np.sqrt(2 / 3) * np.array([[0, 1], [0, 0]]),
            -np.diag([1, -1]) / np.sqrt(3),
            -np.sqrt(2 / 3) * np.array([[0, 0], [1, 0]]),
        )
        for word in product(range(3), repeat=4):
            amplitude = np.trace(matrices[word[0]] @ matrices[word[1]] @ matrices[word[2]] @ matrices[word[3]])
            vector[sum(level * 3**k for k, level in enumerate(word))] = amplitude
        vector /= np.linalg.norm(vector)
    else:
        raise ValueError("Select a supported qutrit benchmark")
    vector.setflags(write=False)
    return vector


def logical_state(name):
    """Return the normalized logical target. Party 0 is the lowest digit."""
    if name in AME46_STATES:
        return _ame46_vector(name, str(ame46_targets_path()))
    if name == AME44:
        return _ame44_vector()
    if name == AME64:
        return _ame64_vector()
    if name in AME_DIMENSIONS:
        return _ame_benchmark_vector(name)
    if name in QUTRIT_BENCHMARKS:
        return _qutrit_benchmark_vector(name)
    spec = _spec(name)
    n = spec.num_parties
    indices = np.arange(3**n)
    digits = [(indices // 3**i) % 3 for i in range(n)]
    phases = sum(weight * digits[a] * digits[b] for a, b, weight in spec.weighted_edges)
    return np.exp(2j * np.pi * phases / 3) / np.sqrt(3**n)


def describe(name, encoding_mode):
    """Return the prompt text on the state, its qubit blocks and the encoding, and the plural party word."""
    n, d, q = layout(name)
    count = {2: "two", 3: "three"}[q]
    vector = os.environ.get("QOQ_VECTOR_ONLY")
    if vector:
        # Blind benchmark: the solver gets only the amplitudes, not the name or the structure of the state.
        text = (f"Prepare the exact {q * n}-qubit state whose amplitudes are in the file {vector} (numpy .npy,"
                f" complex; bit k of the basis index is qubit k) from physical zero, up to a global phase.\n"
                f"Qubits {q}k to {q}k+{q - 1} form block k.")
        if os.environ.get("QOQ_VECTOR_INLINE") == "1":
            path = Path(vector)
            if path.stat().st_size > 1_048_576:
                raise ValueError("The target vector file is too large")
            amplitudes = np.load(path, allow_pickle=False)
            if (amplitudes.shape != (1 << (n * q),) or amplitudes.dtype.kind not in "fc"
                    or not np.isfinite(amplitudes).all()
                    or abs(np.linalg.norm(amplitudes) - 1) > 1e-10):
                raise ValueError("The target must be a finite normalized vector with the expected dimension")
            keep = np.abs(amplitudes) > 1e-15
            entries = [[int(i), float(amplitudes[i].real), float(amplitudes[i].imag)] for i in np.flatnonzero(keep)]
            text += ("\nThe amplitudes are also supplied below as [index, real, imaginary]."
                     f"\nThe omitted entries have total norm {np.linalg.norm(amplitudes[~keep]):.3e}."
                     " The evaluator uses the full vector.\n" + json.dumps(entries))
        return text, "blocks"
    if name in AME46_STATES:
        unit, units = "party", "parties"
        amplitudes = "complex" if np.abs(logical_state(name).imag).max() > 1e-12 else "real"
        lines = [f'Prepare the AME(4,6) state "{name}" ({amplitudes} amplitudes) with {n} parties of {d} levels'
                 f" on {q * n} qubits from physical zero.",
                 "Each party and each pair of parties is maximally entangled with the rest.",
                 "Qubits 3k, 3k+1 and 3k+2 hold party k."]
    elif name == AME44:
        unit, units = "party", "parties"
        lines = [f"Prepare the AME(4,4) state with {n} parties of {d} levels on {q * n} qubits from physical zero:"
                 " (1/4) sum over a, b in GF(4) of |a, b, a+b, a+wb> (two orthogonal Latin squares of order 4).",
                 "Each party and each pair of parties is maximally entangled with the rest.",
                 "Qubits 2k and 2k+1 hold party k; level j is the two-bit word j (bit 0 on qubit 2k)."]
    elif name == AME64:
        unit, units = "party", "parties"
        lines = [f"Prepare the AME(6,4) state with {n} parties of {d} levels on {q * n} qubits from physical zero:"
                 " (1/8) sum over x, y, z in GF(4) of |x, y, z, x+wy+wz, wx+y+wz, wx+wy+z> (the hexacode).",
                 "Each set of at most three parties is maximally entangled with the rest.",
                 "Qubits 2k and 2k+1 hold party k; level j is the two-bit word j (bit 0 on qubit 2k)."]
    elif name in AME_DIMENSIONS:
        unit, units = "party", "parties"
        lines = [f"Prepare the AME(4,{d}) state with four parties of {d} levels on 12 qubits from physical zero."]
        if d == 8:
            lines += ["The target is (1/8) sum over a, b in GF(8) of |a, b, a+b, a+wb>.",
                      "Use GF(8) with w^3 = w + 1. Field addition is bitwise XOR.",
                      "The three-bit word j represents j0+j1*w+j2*w^2."]
        else:
            lines += ["Start each party in the equal superposition of its levels.",
                      f"Apply CZ_d gates on graph edges {AME_GRAPH_EDGES}.",
                      "Each tuple gives two parties and a weight modulo d.",
                      "A weight-t CZ_d gives |j,k> the phase exp(2*pi*i*t*j*k/d)."]
        lines += ["Each party and each pair of parties is maximally entangled with the rest.",
                  "Qubits 3k, 3k+1 and 3k+2 hold party k. Level j uses the binary word j."]
    elif name in QUTRIT_BENCHMARKS:
        unit, units = "qutrit", "qutrits"
        definition = {
            "aharonov3": "The target is (|012> + |120> + |201> - |021> - |102> - |210>)/sqrt(6).",
            "dicke4": "The target is the equal positive superposition of the 12 distinct permutations of |0012>.",
            "cluster4": "The target is CZ3(0,1) CZ3(1,2) CZ3(2,3) |+3>^4, with |+3> = (|0>+|1>+|2>)/sqrt(3).",
            "aklt4": ("The target is the spin-1 AKLT state on a four-site ring. "
                      "It has no total-spin-2 component on bonds (0,1), (1,2), (2,3) and (3,0). "
                      "Levels 0, 1 and 2 represent spin projections +1, 0 and -1."),
        }[name]
        lines = [f'Prepare the qutrit state "{name}" with {n} qutrits on {q * n} qubits from physical zero.',
                 definition, "Ket entries list parties 0 to n-1. Party 0 is the lowest base-3 digit.",
                 "Qubits 2k and 2k+1 hold qutrit k."]
    else:
        unit, units = "qutrit", "qutrits"
        edges = ", ".join(f"({a},{b}) weight {w}" for a, b, w in _spec(name).weighted_edges)
        lines = [f'Prepare the qutrit graph state "{name}" with {n} qutrits on {q * n} qubits from physical zero.',
                 f"Graph edges (qutrit pair, weight mod 3): {edges}.",
                 "Qubits 2k and 2k+1 hold qutrit k."]
    if encoding_mode == "free":
        lines.append(f"Each {unit} can use any {d}-dimensional subspace of its {count} qubits, with any basis in it,"
                     " as its encoding. The evaluator fits the encodings too and adds a separate decoder that is"
                     " not counted.")
    else:
        lines.append(f"Each {unit} uses one fixed encoding in its {count} qubits.")
    lines.append(f"All {1 << q} levels of each block are free during the circuit; only the final state must be"
                 " encoded.")
    return "\n".join(lines), units


def party_encodings(name, encoding):
    """Return one checked encoding per party, shape (n, 2**q, d). A single matrix applies to all parties."""
    n, d, _ = layout(name)
    encoding = np.asarray(encoding, dtype=complex)
    encodings = np.asarray([encoding] * n) if encoding.ndim == 2 else encoding
    if encodings.ndim != 3 or encodings.shape[0] != n:
        raise ValueError(f"Supply one encoding or one encoding for each of the {n} parties")
    for item in encodings:
        tensor_encoding([item], d=d)
    return encodings


def _per_party(matrices, vector):
    """Apply matrices[k] to party k of the vector. Party 0 is the lowest digit.

    This never builds the full tensor product: for AME(4,6) it would be a 4096 x 1296 matrix.
    """
    n = len(matrices)
    tensor = np.asarray(vector).reshape([m.shape[1] for m in matrices][::-1])
    for k, matrix in enumerate(matrices):
        axis = n - 1 - k
        tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def _check_circuit(circuit, width):
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != width:
        raise ValueError(f"The circuit must have {width} qubits")
    if circuit.num_ancillas or circuit.num_clbits or circuit.num_vars or circuit.num_parameters:
        raise ValueError("The circuit must have no ancillas, classical bits or parameters")
    if not np.isfinite(float(circuit.global_phase)):
        raise ValueError("The circuit phase must be finite")
    for item in circuit.data:
        if item.operation.base_class is not _GATES.get(item.operation.name):
            raise ValueError(f"The gate {item.operation.name!r} is not supported")
        matrix = np.asarray(item.operation.to_matrix(), dtype=complex)
        if not np.isfinite(matrix).all() or not np.allclose(
                matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-10, rtol=0):
            raise ValueError(f"The gate {item.operation.name!r} must be unitary")


def _prepare(vector):
    # Matrix rows are the high qubit. Columns are the low qubit.
    high, singular, low = np.linalg.svd(np.asarray(vector).reshape(2, 2))
    circuit = QuantumCircuit(2)
    circuit.ry(2 * np.arctan2(singular[1], singular[0]), 0)
    if singular[1] > 0:
        circuit.h(1)
        circuit.cz(0, 1)
        circuit.h(1)
    circuit.unitary(low.T, [0])
    circuit.unitary(high, [1])
    circuit = transpile(circuit, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    actual = Statevector.from_instruction(circuit).data
    overlap = np.vdot(vector, actual)
    if abs(overlap) < 1e-15 or np.linalg.norm(actual - overlap / abs(overlap) * vector) > 1e-10:
        raise ValueError("The state preparation failed")
    return circuit


def build_workload(name, encoding, cz3, *, f3=None):
    """Build a graph state with one fixed encoding and elementary gates.

    If f3 is supplied, prepare encoded zero and apply this gate on each qutrit.
    Otherwise, prepare the encoded F3 zero state directly.
    """
    spec = _spec(name)
    encoding = np.asarray(encoding, dtype=complex)
    tensor_encoding([encoding], d=3)
    _check_circuit(cz3, 4)
    gate_task = EvaluationTask(3, 2, controlled_z_d(3), 1e-5 / 3, 1e-10 / 9)
    if not evaluate(gate_task, cz3, [encoding, encoding]).is_valid:
        raise ValueError("The CZ3 gate does not match the encoding")
    if f3 is not None:
        _check_circuit(f3, 2)
        gate_task = EvaluationTask(3, 1, fourier_d(3), 1e-10 / np.sqrt(3), 1e-20 / 3)
        if not evaluate(gate_task, f3, [encoding]).is_valid:
            raise ValueError("The F3 gate does not match the encoding")
        f3 = transpile(f3, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    vector = encoding[:, 0] if f3 is not None else encoding @ fourier_d(3)[:, 0]
    preparation = _prepare(vector)
    circuit = QuantumCircuit(2 * spec.num_parties, name=name)
    for qudit in range(spec.num_parties):
        wires = [2 * qudit, 2 * qudit + 1]
        circuit.compose(preparation, qubits=wires, inplace=True)
        if f3 is not None:
            circuit.compose(f3, qubits=wires, inplace=True)
    for a, b, weight in spec.weighted_edges:
        gate = cz3 if weight == 1 else cz3.inverse()
        circuit.compose(gate, [2 * a, 2 * a + 1, 2 * b, 2 * b + 1], inplace=True)
    _check_circuit(circuit, 2 * spec.num_parties)
    return circuit


def encoded_target(name, encoding):
    """Return the encoded target state and the encoding of each party. Party 0 is the lowest digit.

    A 2**q x d encoding applies to every party. An n x 2**q x d array sets one encoding per party,
    with party k on qubits q*k to q*k+q-1.
    """
    encodings = party_encodings(name, encoding)
    return _per_party(encodings, logical_state(name)), encodings


def evaluate_workload(name, circuit, encoding, *, tolerance=1e-5):
    """Check the full state, the final leakage and the physical gate counts.

    The encoding is one 2**q x d isometry for all parties or an n x 2**q x d array with one per party.
    """
    n, _, q = layout(name)
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The tolerance must be a finite nonnegative number")
    _check_circuit(circuit, q * n)
    ideal, encodings = encoded_target(name, encoding)
    state = Statevector.from_instruction(circuit).data
    if not np.isfinite(state).all():
        raise ValueError("The circuit produced a nonfinite state")
    overlap = np.vdot(ideal, state)
    phase = np.exp(1j * np.angle(overlap))
    error = float(np.linalg.norm(state - phase * ideal))
    leakage = float(np.linalg.norm(state - _per_party([e @ e.conj().T for e in encodings], state)))
    counts = circuit.count_ops()
    metrics = circuit_metrics(circuit)
    return {
        "n": n, "num_qubits": circuit.num_qubits,
        "F": float(min(1.0, abs(overlap)**2)), "L": leakage**2,
        "error_norm": error, "leakage_norm": leakage,
        "is_valid": bool(error <= tolerance and leakage <= tolerance),
        "strict_valid": bool(error <= 1e-10 and leakage <= 1e-10),
        "cz_count": int(counts.get("cz", 0)), "cx_count": int(counts.get("cx", 0)),
        "two_qubit_gate_count": metrics["two_qubit_gate_count"],
        "one_qubit_gate_count": metrics["one_qubit_gate_count"],
        "depth": metrics["depth"], "two_qubit_depth": metrics["two_qubit_depth"],
    }
