"""Shortened measurement circuits for the archived two-qutrit Bell experiment.

Only output phases immediately before computational readout may be discarded.
The synthesized blocks are measurement implementations, not unitary replacements
that can be used before subsequent interference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import PassManager

ATOL = 1e-10
REFERENCE_DIGEST = 'eb8196226c255cff6ebd55c52baafac1c1302dc2b4618c97b748e16ec8330f2d'
SETTINGS = tuple((f'A{a}', f'B{b}') for a in range(3) for b in range(3))


@dataclass(frozen=True)
class BellCatalog:
    baseline: tuple[QuantumCircuit, ...]
    optimized: tuple[QuantumCircuit, ...]
    weights: np.ndarray
    evidence: dict


def _readout(circuit: QuantumCircuit) -> tuple[QuantumCircuit, tuple[int, ...]]:
    if circuit.num_qubits != 4 or circuit.num_clbits != 4:
        raise ValueError('Expected four qubits and four terminal readout bits')
    unitary = QuantumCircuit(4, global_phase=circuit.global_phase)
    mapping = {}
    for item in circuit.data:
        wires = [circuit.find_bit(q).index for q in item.qubits]
        name = item.operation.name
        if name == 'barrier':
            continue
        if name == 'measure':
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in mapping:
                raise ValueError('Repeated terminal classical bit')
            mapping[bit] = wires[0]
        else:
            if mapping or item.clbits or name == 'reset':
                raise ValueError('Only unitary gates followed by terminal readout are allowed')
            unitary.append(item.operation, wires)
    if set(mapping) != set(range(4)) or set(mapping.values()) != set(range(4)):
        raise ValueError('Expected a bijective terminal readout map')
    return unitary, tuple(mapping[i] for i in range(4))


def probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Return the Bell experiment's ideal histogram in classical bit order."""
    unitary, mapping = _readout(circuit)
    result = np.zeros(16)
    for basis, probability in enumerate(Statevector(unitary).probabilities()):
        outcome = sum(((basis >> q) & 1) << bit for bit, q in enumerate(mapping))
        result[outcome] = probability
    return result


def split_terminal_blocks(circuit: QuantumCircuit):
    """Extract the backward light cone of cross-party gates as preparation.

    B occupies logical wires 0,1 and A occupies 2,3. An operation can move
    to a terminal block only when every subsequent preparation operation
    acts on disjoint wires. No algebraic gate commutation is assumed.
    """
    unitary, _ = _readout(circuit)
    needed, preparation_indices = set(), set()
    for index in reversed(range(len(unitary.data))):
        item = unitary.data[index]
        wires = {unitary.find_bit(q).index for q in item.qubits}
        crosses_parties = len({q // 2 for q in wires}) > 1
        if crosses_parties or wires & needed:
            preparation_indices.add(index)
            needed.update(wires)
    prep = QuantumCircuit(4, global_phase=unitary.global_phase)
    a, b = QuantumCircuit(2), QuantumCircuit(2)
    for index, item in enumerate(unitary.data):
        wires = [unitary.find_bit(q).index for q in item.qubits]
        if index in preparation_indices:
            prep.append(item.operation, wires)
        else:
            target = b if wires[0] < 2 else a
            target.append(item.operation, [q % 2 for q in wires])
    rebuilt = prep.compose(a, [2, 3]).compose(b, [0, 1])
    if np.max(abs(Operator(rebuilt).data - Operator(unitary).data)) > ATOL:
        raise ValueError('Terminal extraction changed the full unitary')
    return prep, a, b


def _synthesize_measurement(matrix: np.ndarray) -> tuple[QuantumCircuit, dict]:
    """Shorten a terminal two-qubit measurement with <=2 CZ, using D(phi) on the LEFT.

    Solve Im Tr(V J V^T J)/sqrt(det V)=0 with a continuous square-root
    branch, then synthesize V=D(phi)U exactly before dropping final local Z
    phases. The complete Bell histogram is checked after circuit assembly.
    """
    from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates

    u = np.asarray(matrix, dtype=complex)
    if (u.shape != (4, 4) or not np.isfinite(u).all()
            or not np.allclose(u.conj().T @ u, np.eye(4), atol=ATOL, rtol=0)):
        raise ValueError('Expected a finite 4x4 unitary')
    y = np.array([[0, -1j], [1j, 0]])
    j = np.kron(y, y)
    d0 = np.sqrt(np.linalg.det(u))

    def criterion(phi):
        v = np.diag([1, 1, 1, np.exp(1j * phi)]) @ u
        return float(np.imag(np.trace(v @ j @ v.T @ j) / (d0 * np.exp(.5j * phi))))

    a, b = criterion(0), criterion(np.pi)
    phi = 0. if np.hypot(a, b) < 1e-13 else float(2 * (np.arctan2(-a, b) % np.pi))
    target = np.diag([1, 1, 1, np.exp(1j * phi)]) @ u
    decomposed = TwoQubitBasisDecomposer(CZGate())(target, approximate=False)
    exact_error = float(np.max(abs(Operator(decomposed).data - target)))
    if exact_error > ATOL or decomposed.count_ops().get('cz', 0) > 2:
        raise ValueError('Exact two-CZ synthesis failed')
    native = transpile(decomposed, basis_gates=['r', 'cz'], optimization_level=0)
    native = PassManager([IQMOptimizeSingleQubitGates(drop_final_rz=True)]).run(native)
    if (native.count_ops().get('cz', 0) > 2
            or native.count_ops().get('r', 0) > 6
            or not set(native.count_ops()) <= {'r', 'cz'}):
        raise ValueError('Native terminal measurement synthesis failed')
    native.metadata = {'terminal_measurement_only': True}
    return native, dict(phi=phi, phase_residual=abs(criterion(phi)),
                        exact_synthesis_error=exact_error,
                        native_ops=dict(native.count_ops()))


def _signature(circuit):
    return [(item.operation.name, tuple(float(p) for p in item.operation.params),
             tuple(circuit.find_bit(q).index for q in item.qubits)) for item in circuit.data]


def build_catalog(source: str | Path) -> BellCatalog:
    """Load the frozen, QPY-verified reference and certify the new catalog."""
    document = json.loads(Path(source).read_text(encoding='utf-8'))
    keys = ('parents', 'settings', 'physical_layout', 'weights', 'classical_bound')
    payload = {key: document[key] for key in keys}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != REFERENCE_DIGEST or document.get('payload_sha256') != digest:
        raise ValueError('Reference payload digest mismatch')
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported reference schema')
    layout = payload['physical_layout']
    weights = np.asarray(payload['weights'], dtype=float)
    baseline, optimized, block_cache, block_evidence = [], [], {}, {}
    shared = None
    max_error = 0.
    for record, setting in zip(payload['parents'], SETTINGS, strict=True):
        index = record['setting_index']
        old = QuantumCircuit(4, 4, name=f'baseline_{setting[0]}_{setting[1]}')
        for instruction in record['instructions']:
            wires = [layout.index(q) for q in instruction['qubits']]
            gate = instruction['gate']
            if gate == 'r':
                old.r(*instruction['params'], wires[0])
            elif gate == 'cz':
                old.cz(*wires)
            elif gate == 'measure':
                old.measure(wires[0], instruction['clbits'][0])
            else:
                raise ValueError(f'Unexpected reference gate: {gate}')
        prep, a, b = split_terminal_blocks(old)
        if shared is None:
            shared = prep
        if _signature(prep) != _signature(shared):
            raise ValueError('Parents do not have an identical shared preparation')
        for name, block in zip(setting, (a, b), strict=True):
            if name not in block_cache:
                compressed, evidence = _synthesize_measurement(Operator(block).data)
                block_cache[name] = (block, compressed)
                block_evidence[name] = evidence
            elif _signature(block) != _signature(block_cache[name][0]):
                raise ValueError('Local measurement depends on the other setting')
        new = QuantumCircuit(4, 4, name=f'optimized_{setting[0]}_{setting[1]}')
        new.compose(prep, inplace=True)
        new.compose(block_cache[setting[0]][1], [2, 3], inplace=True)
        new.compose(block_cache[setting[1]][1], [0, 1], inplace=True)
        _, mapping = _readout(old)
        new.measure(mapping, range(4))
        old.metadata = new.metadata = {'setting_index': index, 'setting': list(setting),
                                       'classical_bit_to_logical_qubit': list(mapping)}
        error = float(np.max(abs(probabilities(old) - probabilities(new))))
        max_error = max(max_error, error)
        b0 = setting[1] == 'B0'
        expected = {'r': 12 if b0 else 18, 'cz': 5 if b0 else 7, 'measure': 4}
        if error > ATOL or dict(new.count_ops()) != expected:
            raise ValueError('Bell histogram or gate budget failed')
        baseline.append(old)
        optimized.append(new)
    ideal = {}
    for name, circuits in (('baseline', baseline), ('optimized', optimized)):
        value = float(sum(w @ probabilities(c)
                          for w, c in zip(weights, circuits, strict=True)))
        if abs(value - 6.) > ATOL:
            raise ValueError('Ideal Bell value differs from 6')
        ideal[f'ideal_bell_{name}'] = value
    weights.setflags(write=False)
    evidence = dict(reference_digest=digest, source_manifest_sha256=document['source_manifest_sha256'],
                    settings=[list(s) for s in SETTINGS], preparation_ops=dict(shared.count_ops()),
                    blocks=block_evidence, max_probability_error=max_error, **ideal,
                    classical_bound=float(payload['classical_bound']),
                    classical_bit_to_logical_qubit=[2, 3, 0, 1],
                    physical_layout_reference_only=layout,
                    costs={name: [dict(c.count_ops()) for c in circuits]
                           for name, circuits in (('baseline', baseline), ('optimized', optimized))})
    return BellCatalog(tuple(baseline), tuple(optimized), weights, evidence)
