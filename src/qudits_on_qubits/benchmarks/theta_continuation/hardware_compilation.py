"""Offline native and routed circuits for the additive theta hardware layer.

Output permutations are retained. Only terminal local measurement blocks may
discard output phases; supplied state preparations and gates remain exact.
Routing is a finite seeded search, not a claim of globally optimal circuits.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import CouplingMap, PassManager

from qudits_on_qubits.bell_measurements.postprocessing import compute_bell_value_from_counts
from qudits_on_qubits.bell_measurements.sampler_circuits import (
    build_sampler_circuits_for_candidate, decoding_kwargs_from_metadata,
)
from qudits_on_qubits.reference_experiments import get_reference_experiment


ATOL = 1e-10


def circuit_costs(circuit: QuantumCircuit) -> dict[str, int]:
    """Count the complete stored circuit, including routing and readout."""
    return {
        "n_cz": int(circuit.count_ops().get("cz", 0)),
        "n_1q": sum(len(item.qubits) == 1 and item.operation.name not in {"measure", "barrier"}
                    for item in circuit.data),
        "depth": int(circuit.depth()),
        "cz_depth": int(circuit.depth(lambda item: item.operation.name == "cz")),
        "size": int(circuit.size()),
    }


def _validate_source(circuit: QuantumCircuit) -> None:
    if (not isinstance(circuit, QuantumCircuit) or circuit.num_qubits not in (2, 4, 6, 8)
            or circuit.num_clbits or circuit.num_parameters):
        raise ValueError("Expected a bound 2/4/6/8-qubit unitary circuit without classical bits")
    if not np.isfinite(float(circuit.global_phase)):
        raise ValueError("Circuit parameters must be finite")
    for item in circuit.data:
        if (item.operation.name in {"measure", "reset", "initialize", "delay"}
                or hasattr(item.operation, "blocks") or item.clbits):
            raise ValueError("Expected a bound unitary circuit")
        for parameter in item.operation.params:
            if not np.isfinite(np.asarray(parameter, dtype=complex)).all():
                raise ValueError("Circuit parameters must be finite")


def _validate_edges(n: int, edges) -> tuple[tuple[int, int], ...]:
    normalized = set()
    for edge in edges:
        if len(edge) != 2 or any(isinstance(q, bool) or not isinstance(q, Integral) for q in edge):
            raise ValueError("Edges must contain two integer wire indices")
        a, b = map(int, edge)
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError("Edges must connect distinct in-range wires")
        normalized.add(tuple(sorted((a, b))))
    reached = {0}
    while True:
        enlarged = reached | {q for edge in normalized if reached.intersection(edge) for q in edge}
        if enlarged == reached:
            break
        reached = enlarged
    if reached != set(range(n)):
        raise ValueError("Hardware edges must connect every circuit wire")
    return tuple(sorted(normalized))


def _validate_seeds(seeds) -> tuple[int, ...]:
    if (not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                   or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Seeds must be a nonempty sequence of nonnegative 32-bit integers")
    return tuple(int(seed) for seed in seeds)


def _aligned_rows(values: np.ndarray, permutation) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.int64)
    physical = np.zeros_like(indices)
    for logical, wire in enumerate(permutation):
        physical |= ((indices >> logical) & 1) << wire
    return values[physical]


def _phase_error(expected: np.ndarray, actual: np.ndarray) -> float:
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.max(np.abs(actual - phase * expected)))


def _representation(circuit, full_operator):
    return Operator(circuit).data if full_operator else Statevector(circuit).data


def _native(circuit, seed, *, full_operator=False):
    expected = _representation(circuit, full_operator)
    failures = []
    for level in (1, 0):
        native = transpile(circuit, basis_gates=["r", "cz"], optimization_level=level,
                           seed_transpiler=seed, num_processes=1, approximation_degree=1.0)
        error = _phase_error(expected, _representation(native, full_operator))
        if set(native.count_ops()) <= {"r", "cz"} and error <= ATOL:
            return native, error, level
        failures.append({"optimization_level": level, "error": error})
    raise ValueError(f"No native candidate preserves the exact supplied circuit: {failures}")


def shortest_party_repair(permutation, edges):
    """Return a shortest edge-SWAP path making every logical party adjacent.

    Search identifies the two qubits within a party and identifies party names.
    This quotient preserves adjacency and edge-SWAP transitions, so at most
    105 distinct pair partitions are needed for eight wires. Replay retains
    the full oriented logical-to-wire permutation.
    """
    n = len(permutation)
    if (n not in (2, 4, 6, 8)
            or any(isinstance(q, bool) or not isinstance(q, Integral) for q in permutation)
            or set(permutation) != set(range(n))):
        raise ValueError("Expected an even-width logical-to-wire permutation")
    graph = _validate_edges(n, edges)
    edge_set = set(graph)
    occupancy = [0] * n
    for logical, wire in enumerate(permutation):
        occupancy[wire] = logical // 2

    def canonical(values):
        names = {}
        return tuple(names.setdefault(value, len(names)) for value in values)

    def adjacent(values):
        locations = {}
        for wire, party in enumerate(values):
            locations.setdefault(party, []).append(wire)
        return all(tuple(pair) in edge_set for pair in locations.values())

    start = canonical(occupancy)
    previous = {start: None}
    queue = deque([start])
    found = None
    while queue:
        current = queue.popleft()
        if adjacent(current):
            found = current
            break
        for a, b in graph:
            trial = list(current)
            trial[a], trial[b] = trial[b], trial[a]
            trial = canonical(trial)
            if trial not in previous:
                previous[trial] = (current, (a, b))
                queue.append(trial)
    if found is None:
        raise ValueError("Hardware graph has no arrangement of disjoint adjacent party pairs")
    swaps = []
    while previous[found] is not None:
        found, edge = previous[found]
        swaps.append(edge)
    swaps.reverse()
    final = list(permutation)
    for a, b in swaps:
        final = [b if q == a else a if q == b else q for q in final]
    return final, swaps


def _routing_permutation(circuit):
    permutation = (list(range(circuit.num_qubits)) if circuit.layout is None
                   else list(circuit.layout.final_index_layout(filter_ancillas=True)))
    if sorted(permutation) != list(range(circuit.num_qubits)):
        raise ValueError("Routing did not retain a bijective logical-to-wire permutation")
    return permutation


def _route(circuit, edges, seeds, *, full_operator=False, repair=False):
    expected = _representation(circuit, full_operator)
    coupling = CouplingMap([list(edge) for a, b in edges for edge in ((a, b), (b, a))])
    swap = QuantumCircuit(2)
    swap.swap(0, 1)
    native_swap = transpile(swap, basis_gates=["r", "cz"], optimization_level=0,
                            num_processes=1)
    candidates, evidence = [], []
    for seed in seeds:
        accepted_for_seed = False
        for level in (3, 1, 0):
            if level == 0 and accepted_for_seed:
                break
            routed = transpile(
                circuit, basis_gates=["r", "cz"], coupling_map=coupling,
                initial_layout=list(range(circuit.num_qubits)), layout_method="trivial",
                routing_method="sabre", optimization_level=level,
                seed_transpiler=seed, approximation_degree=1.0, num_processes=1,
            )
            before = _routing_permutation(routed)
            permutation, swaps = shortest_party_repair(before, edges) if repair else (before, [])
            # Strip transpiler layout metadata: these are explicit physical wires.
            physical = QuantumCircuit(circuit.num_qubits, global_phase=routed.global_phase)
            for item in routed.data:
                physical.append(item.operation, [routed.find_bit(q).index for q in item.qubits])
            for edge in swaps:
                physical.compose(native_swap, list(edge), inplace=True)
            error = _phase_error(expected, _aligned_rows(_representation(physical, full_operator), permutation))
            costs = circuit_costs(physical)
            supported = set(physical.count_ops()) <= {"r", "cz"}
            on_edges = all(tuple(sorted(physical.find_bit(q).index for q in item.qubits)) in edges
                           for item in physical.data if item.operation.num_qubits == 2)
            accepted = supported and on_edges and error <= ATOL
            record = dict(seed=seed, optimization_level=level, error=error,
                          accepted=accepted, repair_count=len(swaps), costs=costs)
            evidence.append(record)
            if accepted:
                info = dict(seed=seed, optimization_level=level, final_permutation=permutation,
                            transpiler_final_permutation=before,
                            repair_swaps=[list(edge) for edge in swaps], repair_count=len(swaps),
                            preservation_error=error)
                candidates.append((physical, info))
                accepted_for_seed = True
    if not candidates:
        raise ValueError(f"No routed candidate preserves the exact supplied circuit: {evidence}")
    def score(candidate):
        cost = circuit_costs(candidate[0])
        return tuple(cost[k] for k in ("n_cz", "cz_depth", "n_1q", "depth", "size")) + (candidate[1]["seed"],)
    best, info = min(candidates, key=score)
    info.update(candidates=evidence, initial_permutation=list(range(circuit.num_qubits)),
                search_scope="Fixed identity placement; finite seeded SABRE search at levels 3 and 1; "
                             "level 0 exactness fallback; shortest party adjacency repair")
    return best, info


def _split_terminal_blocks(circuit):
    n = circuit.num_qubits
    needed, prefix_indices = set(), set()
    for index in reversed(range(len(circuit.data))):
        item = circuit.data[index]
        wires = {circuit.find_bit(q).index for q in item.qubits}
        if len({q // 2 for q in wires}) > 1 or needed.intersection(wires):
            prefix_indices.add(index)
            needed.update(wires)
    prefix = QuantumCircuit(n, global_phase=circuit.global_phase)
    tails = [QuantumCircuit(2) for _ in range(n // 2)]
    for index, item in enumerate(circuit.data):
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if index in prefix_indices:
            prefix.append(item.operation, wires)
        else:
            tails[n // 2 - 1 - wires[0] // 2].append(item.operation, [q % 2 for q in wires])
    return prefix, tails


def _row_projector_error(expected, actual):
    return max(float(np.max(abs(np.outer(a.conj(), a) - np.outer(b.conj(), b))))
               for a, b in zip(expected, actual, strict=True))


def _synthesize_measurement(matrix):
    """Synthesize <=2 CZ up to output diagonal phases; certify all projectors."""
    from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates

    unitary = np.asarray(matrix, dtype=complex)
    if (unitary.shape != (4, 4) or not np.isfinite(unitary).all()
            or not np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=ATOL, rtol=0)):
        raise ValueError("Expected a finite 4x4 unitary measurement matrix")
    y = np.array([[0, -1j], [1j, 0]])
    j = np.kron(y, y)
    determinant_root = np.sqrt(np.linalg.det(unitary))

    def criterion(phi):
        target = np.diag([1, 1, 1, np.exp(1j * phi)]) @ unitary
        return float(np.imag(np.trace(target @ j @ target.T @ j)
                             / (determinant_root * np.exp(0.5j * phi))))

    a, b = criterion(0), criterion(np.pi)
    phi = 0.0 if np.hypot(a, b) < 1e-13 else float(2 * (np.arctan2(-a, b) % np.pi))
    target = np.diag([1, 1, 1, np.exp(1j * phi)]) @ unitary
    decomposed = TwoQubitBasisDecomposer(CZGate())(target, approximate=False)
    exact_error = _phase_error(target, Operator(decomposed).data)
    if exact_error > ATOL or decomposed.count_ops().get("cz", 0) > 2:
        raise ValueError("Exact two-CZ terminal measurement synthesis failed")
    native, _, _ = _native(decomposed, 0, full_operator=True)
    shortened = PassManager([IQMOptimizeSingleQubitGates(drop_final_rz=True)]).run(native)
    error = _row_projector_error(unitary, Operator(shortened).data)
    if error > ATOL:
        # The optimizer may discard a small rotation. The exact native block
        # already implements the required output-phase-equivalent measurement.
        shortened = native
        error = _row_projector_error(unitary, Operator(shortened).data)
    if error > ATOL or shortened.count_ops().get("cz", 0) > 2:
        raise ValueError("Terminal measurement row projectors were not preserved")
    shortened.metadata = {"terminal_measurement_only": True}
    return shortened, dict(phi=phi, phase_residual=abs(criterion(phi)),
                           exact_synthesis_error=exact_error, row_projector_error=error)


def _local_measurements(tails, measurement_metadata):
    cache, evidence = {}, []
    for setting, metadata in zip(measurement_metadata["setting_by_circuit_index"],
                                 measurement_metadata["circuit_metadata"], strict=True):
        for party, (label, tail, basis) in enumerate(zip(setting, tails, metadata["local_basis_gates"], strict=True)):
            key = (party, label)
            measurement = np.eye(4, dtype=complex) if basis is None else basis["unitary"]
            matrix = measurement @ Operator(tail).data
            if key in cache:
                if _phase_error(matrix, cache[key][0]) > ATOL:
                    raise ValueError("Local measurement depends on another party's setting")
                continue
            shortened, proof = _synthesize_measurement(matrix)
            original = tail.copy()
            if basis is not None:
                original.unitary(measurement, [0, 1])
            exact, _, _ = _native(original, 0, full_operator=True)
            retained = circuit_costs(exact)["n_cz"] < circuit_costs(shortened)["n_cz"]
            if retained:
                shortened = exact
            cache[key] = (matrix, shortened)
            evidence.append(dict(party=party, label=label, **proof,
                                 retained_cheaper_original=retained, original=circuit_costs(exact),
                                 native=circuit_costs(shortened)))
    return cache, evidence


def _probabilities(circuit):
    n = circuit.num_qubits
    unitary = QuantumCircuit(n, global_phase=circuit.global_phase)
    mapping = {}
    for item in circuit.data:
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in mapping:
                raise ValueError("Repeated terminal measurement bit")
            mapping[bit] = wires[0]
        else:
            if mapping:
                raise ValueError("Operations after terminal readout are not supported")
            unitary.append(item.operation, wires)
    if set(mapping) != set(range(n)) or set(mapping.values()) != set(range(n)):
        raise ValueError("Expected complete bijective terminal readout")
    return _aligned_rows(Statevector(unitary).probabilities(), [mapping[i] for i in range(n)])


def _weights(metadata, n):
    rows = []
    for setting in metadata["setting_by_circuit_index"]:
        terms = [term for term in metadata["terms"] if tuple(term["settings"]) == setting]
        rows.append([float(compute_bell_value_from_counts(
            {setting: {format(k, f"0{n}b"): 1}}, terms,
            metadata["qutrit_bit_indices_by_setting"], renormalize_after_discard=False,
            **decoding_kwargs_from_metadata(metadata)).real) for k in range(2**n)])
    return np.asarray(rows, dtype=float)


def _assemble(prefix, permutation, setting, cache, index):
    n = prefix.num_qubits
    circuit = QuantumCircuit(n, n, name=f"bell_{index:03d}")
    circuit.compose(prefix, inplace=True)
    readout = []
    for party, label in enumerate(setting):
        logical_pair = (n - 2 * party - 2, n - 2 * party - 1)
        wires = [permutation[q] for q in logical_pair]
        circuit.compose(cache[(party, label)][1], wires, inplace=True)
        readout.extend(wires)
    circuit.measure(readout, range(n))
    circuit.metadata = dict(setting_index=index, setting=list(setting),
                            classical_bit_to_physical_qubit=readout,
                            final_logical_to_wire_permutation=list(permutation),
                            shared_preparation=True, terminal_measurement_only=True)
    return circuit


def compile_gate(circuit, edges, *, seeds=(0, 1, 2)):
    """Compile an F3/CZ3 program and check its entire operator, not just |0>."""
    _validate_source(circuit)
    graph, seed_values = _validate_edges(circuit.num_qubits, edges), _validate_seeds(seeds)
    native, native_error, native_level = _native(circuit, seed_values[0], full_operator=True)
    routed, routing = _route(native, graph, seed_values, full_operator=True)
    total_error = _phase_error(Operator(circuit).data,
                              _aligned_rows(Operator(routed).data, routing["final_permutation"]))
    if total_error > ATOL:
        raise ValueError("Routed full operator does not preserve the supplied gate")
    return {"metadata": dict(original=circuit_costs(circuit), native=circuit_costs(native),
                             routed=circuit_costs(routed), routing=routing,
                             full_operator_error=max(native_error, total_error),
                             native_optimization_level=native_level, tolerance=ATOL),
            "circuits": {"native.qpy": native, "routed.qpy": routed}, "arrays": {}}


def compile_case(state_name, circuit, encoding, edges, *, seeds=(0, 1, 2)):
    """Preserve the actual supplied state and every complete Bell histogram."""
    _validate_source(circuit)
    e = np.asarray(encoding, dtype=complex)
    if (e.shape != (4, 3) or not np.isfinite(e).all()
            or not np.allclose(e.conj().T @ e, np.eye(3), atol=ATOL, rtol=0)):
        raise ValueError("Expected a finite 4x3 encoding isometry")
    reference = get_reference_experiment(state_name)
    n = circuit.num_qubits
    if 2 * len(reference.state.party_order) != n:
        raise ValueError("State name and preparation wire count disagree")
    graph, seed_values = _validate_edges(n, edges), _validate_seeds(seeds)
    embedding = e
    for _ in range(n // 2 - 1):
        embedding = np.kron(embedding, e)
    original_state = Statevector(circuit).data
    expected_state = embedding @ reference.state.statevector()
    source_fidelity = float(abs(np.vdot(expected_state, original_state)) ** 2)
    source_leakage = float(np.linalg.norm(original_state - embedding @ (embedding.conj().T @ original_state)) ** 2)
    native, native_error, native_level = _native(circuit, seed_values[0])
    routed, preparation_routing = _route(native, graph, seed_values)
    routed_error = _phase_error(original_state, _aligned_rows(Statevector(routed).data,
                                                              preparation_routing["final_permutation"]))
    if routed_error > ATOL:
        raise ValueError("Routed preparation does not preserve the supplied state")
    pairs = [(q, q + 1) for q in range(n - 2, -1, -2)]
    sources, measurement_metadata = build_sampler_circuits_for_candidate(
        state_name, circuit, e, qutrit_qubits=pairs)
    prefix, tails = _split_terminal_blocks(native)
    routed_prefix, bell_routing = _route(prefix, graph, seed_values, repair=True)
    cache, block_evidence = _local_measurements(tails, measurement_metadata)
    weights = _weights(measurement_metadata, n)
    settings = measurement_metadata["setting_by_circuit_index"]
    circuits = {"preparation_native.qpy": native, "preparation_routed.qpy": routed,
                "bell_prefix_native.qpy": prefix, "bell_prefix_routed.qpy": routed_prefix}
    probabilities = {"source": [], "native": [], "routed": []}
    rows = []
    for index, (setting, source) in enumerate(zip(settings, sources, strict=True)):
        probabilities["source"].append(_probabilities(source))
        row = dict(index=index, labels=list(setting))
        errors = []
        for kind, common, permutation in (
                ("native", prefix, list(range(n))),
                ("routed", routed_prefix, bell_routing["final_permutation"])):
            full = _assemble(common, permutation, setting, cache, index)
            circuits[f"bell_{kind}_{index:03d}.qpy"] = full
            histogram = _probabilities(full)
            probabilities[kind].append(histogram)
            errors.append(float(np.max(abs(probabilities["source"][-1] - histogram))))
            row[kind] = circuit_costs(full)
        row["max_probability_error"] = max(errors)
        row["ideal_contribution"] = float(weights[index] @ probabilities["native"][-1])
        if row["max_probability_error"] > ATOL:
            raise ValueError(f"Complete Bell histogram changed for setting {setting}: {errors}")
        rows.append(row)
    arrays = {f"probabilities_{name}.npy": np.asarray(values) for name, values in probabilities.items()}
    arrays["weights.npy"] = weights
    ideal = {name: float(np.sum(weights * arrays[f"probabilities_{name}.npy"]))
             for name in probabilities}
    metadata = dict(
        state_name=reference.experiment_id, preparation_original=circuit_costs(circuit),
        preparation_native=circuit_costs(native), preparation_routed=circuit_costs(routed),
        bell_prefix_native=circuit_costs(prefix), bell_prefix_routed=circuit_costs(routed_prefix),
        native_optimization_level=native_level, source_fidelity=source_fidelity,
        source_leakage=source_leakage, preparation_native_error=native_error,
        preparation_routed_error=routed_error, tolerance=ATOL,
        routing=dict(preparation=preparation_routing, bell=bell_routing),
        settings=[list(setting) for setting in settings], per_setting=rows,
        ideal_bell=ideal["native"], ideal_bell_source=ideal["source"], ideal_bell_routed=ideal["routed"],
        max_probability_error=max(row["max_probability_error"] for row in rows),
        shared_preparation_verified=True, independent_local_settings_verified=True,
        local_blocks=block_evidence, logical_party_pairs=[list(pair) for pair in pairs],
        weight_policy="Unconditional Bell weights; every leakage outcome has weight zero",
        physical_backend_validated=False,
    )
    return {"metadata": metadata, "circuits": circuits, "arrays": arrays}
