"""Find short decoders from a party encoding to the canonical encoding.

A decoder of party k is a circuit D_k on its qubits with D_k E_k = B up to a phase, where B is
the canonical encoding. Only the d columns of the levels are fixed; the other columns are free.
For a qutrit (two qubits), the search tries 0, 1 and 2 CZs with a numeric fit. Three CZs always
suffice. They come from a known exact decoder (for example the fit tail of a free encoding) or
from a two-qubit synthesis of one unitary completion. A three-qubit block (AME(4,6) party) uses
no decoder for the canonical encoding, else the known decoder, else a Quantum Shannon synthesis.
Each decoder is checked before use.
"""
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer, qs_decomposition
from scipy.optimize import least_squares

from .encodings import canonical_encoding, tensor_encoding

_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_CANONICAL = canonical_encoding(3)
FIT_RESTARTS = 4
FIT_NFEV = 300
TOLERANCE = 1e-11


def _u(theta, phi, lam):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _layered(x, count):
    """U on both qubits, then count times (CZ, U on both). Qubit 0 is the low bit."""
    angles = x.reshape(-1, 3)
    unitary = np.kron(_u(*angles[1]), _u(*angles[0]))
    for j in range(count):
        unitary = np.kron(_u(*angles[2 * j + 3]), _u(*angles[2 * j + 2])) @ _CZ @ unitary
    return unitary


def _circuit(x, count):
    angles = np.mod(x, 4 * np.pi).reshape(-1, 3)
    circuit = QuantumCircuit(2)
    circuit.u(*angles[0], 0)
    circuit.u(*angles[1], 1)
    for j in range(count):
        circuit.cz(0, 1)
        circuit.u(*angles[2 * j + 2], 0)
        circuit.u(*angles[2 * j + 3], 1)
    return circuit


def _maps_to_canonical(circuit, encoding):
    canonical = canonical_encoding(encoding.shape[1])
    mapped = Operator(circuit).data @ encoding
    overlap = np.trace(canonical.conj().T @ mapped)
    return np.linalg.norm(mapped - np.exp(1j * np.angle(overlap)) * canonical) <= 1e-10


def _fit(encoding, count):
    def residual(x):
        difference = _layered(x[:-1], count) @ encoding - np.exp(1j * x[-1]) * _CANONICAL
        return np.r_[difference.real.ravel(), difference.imag.ravel()]

    for restart in range(FIT_RESTARTS):
        x0 = np.random.default_rng([count, restart]).uniform(0, 2 * np.pi, 3 * (2 + 2 * count) + 1)
        try:
            solution = least_squares(residual, x0, method="trf", max_nfev=FIT_NFEV,
                                     ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(solution.fun) <= TOLERANCE:
            circuit = _circuit(solution.x[:-1], count)
            if _maps_to_canonical(circuit, encoding):
                return circuit
    return None


def _synthesized(encoding):
    """Return a decoder from one unitary completion, or None if it is not exact.

    Qiskit snaps Weyl coordinates near special classes, so the result can be off by about
    1e-7 for an encoding near a cheaper class. The check rejects such a result.
    """
    rows, d = encoding.shape
    complement = np.linalg.svd(encoding.conj().T)[2][d:].conj().T
    inverse = np.column_stack((encoding, complement)).conj().T
    if rows == 4:
        circuit = TwoQubitBasisDecomposer(CZGate(), euler_basis="U")(inverse, approximate=False)
    else:
        circuit = qs_decomposition(inverse)
    circuit = transpile(circuit, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    return circuit if _maps_to_canonical(circuit, encoding) else None


@lru_cache(maxsize=64)
def _short_decoder(raw):
    encoding = np.frombuffer(raw, dtype=complex).reshape(4, 3)
    for count in range(3):
        circuit = _fit(encoding, count)
        if circuit is not None:
            return circuit
    return None


def block_decoder(encoding, fallback=None):
    """Return a short u/cz circuit on the block qubits that maps the encoding to the canonical one.

    fallback is an optional known decoder. It is used only if no shorter decoder is found
    (at most two CZs for a qutrit) and only if it passes the check.
    """
    encoding = np.ascontiguousarray(encoding, dtype=complex)
    tensor_encoding([encoding])
    if encoding.shape == (4, 3):
        circuit = _short_decoder(encoding.tobytes())
    else:
        # shortcut: only the empty decoder is tried before the fallback; add a fit if decoder cost matters.
        empty = QuantumCircuit(encoding.shape[0].bit_length() - 1)
        circuit = empty if _maps_to_canonical(empty, encoding) else None
    if circuit is None and fallback is not None and _maps_to_canonical(fallback, encoding):
        circuit = fallback
    if circuit is None:
        circuit = _synthesized(encoding)
    if circuit is None:
        raise ValueError("No exact decoder was found for the encoding")
    return circuit


def decoder_circuit(encodings, fallbacks=None):
    """Return the decoder of all parties. Party k uses the q qubits from q*k."""
    encodings = np.asarray(encodings, dtype=complex)
    fallbacks = fallbacks or [None] * len(encodings)
    q = encodings.shape[1].bit_length() - 1
    decoder = QuantumCircuit(q * len(encodings))
    for k, encoding in enumerate(encodings):
        decoder.compose(block_decoder(encoding, fallbacks[k]), range(q * k, q * k + q), inplace=True)
    return decoder


def with_decoder_check(name, circuit, decoder, metrics):
    """Return the metrics after the decoded state is checked against the canonical target."""
    from .state_workloads import evaluate_workload, layout

    canonical = canonical_encoding(layout(name)[1])
    decoded = evaluate_workload(name, circuit.compose(decoder), canonical, tolerance=1e-10)
    return {**metrics, "is_valid": bool(metrics["is_valid"] and decoded["is_valid"]),
            "strict_valid": bool(metrics["strict_valid"] and decoded["strict_valid"]),
            "decoder_cz_count": int(decoder.count_ops().get("cz", 0))}
