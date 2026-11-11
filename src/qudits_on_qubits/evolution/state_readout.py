"""Build a checked offline readout with an explicit coupling map."""

import numpy as np
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.theta_continuation.hardware_compilation import (
    _native, _route, _validate_edges, _validate_seeds, circuit_costs,
)
from .decoders import block_decoder, decoder_circuit
from .encodings import canonical_encoding
from .state_workloads import evaluate_workload


def compile_readout(name, circuit, encoding, edges, *, seeds=(0,)):
    """Check the state, decode it and route its terminal measurement offline.

    The encoding is one 4 x 3 isometry for all qutrits or an n x 4 x 3 array with one per qutrit.
    """
    source = evaluate_workload(name, circuit, encoding, tolerance=1e-10)
    if not source["is_valid"]:
        raise ValueError("The source state does not pass the limit")
    graph = _validate_edges(circuit.num_qubits, edges)
    seeds = _validate_seeds(seeds)
    encodings = np.asarray(encoding, dtype=complex)
    if encodings.ndim == 2:
        encodings = np.stack([encodings] * (circuit.num_qubits // 2))
    decoded = circuit.compose(decoder_circuit(encodings))
    result = evaluate_workload(name, decoded, canonical_encoding(3), tolerance=1e-10)
    if not result["is_valid"]:
        raise ValueError("The decoded state does not pass the limit")
    native, native_error, _ = _native(decoded, seeds[0])
    routed, routing = _route(native, graph, seeds)
    measured = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    measured.compose(routed, inplace=True)
    measured.measure(routing["final_permutation"], range(circuit.num_qubits))
    return {
        "metadata": {
            "preparation": circuit_costs(circuit),
            "decoder_per_party": [circuit_costs(block_decoder(e)) for e in encodings],
            "decoded": circuit_costs(decoded),
            "routed_with_measurement": circuit_costs(measured),
            "decoded_metrics": result,
            "native_error": native_error,
            "routing": routing,
            "physical_backend_validated": False,
        },
        "circuits": {"decoded": decoded, "routed_with_measurement": measured},
    }
