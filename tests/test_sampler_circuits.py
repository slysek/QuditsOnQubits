from __future__ import annotations

import numpy as np
import pytest
from igraph import Graph
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qutrit_bell_measurements import (
    build_sampler_circuits_for_candidate,
    build_sampler_circuits_from_graph,
    canonical_Ez,
    compute_bell_value_from_counts,
    counts_by_setting_from_sampler_result,
    decoding_kwargs_from_metadata,
    omega,
    run_sampler_circuits_to_counts_by_setting,
)


def _qutrit_xz() -> tuple[np.ndarray, np.ndarray]:
    w = omega(3)
    x = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        x[(j + 1) % 3, j] = 1.0
    z = np.diag([w**j for j in range(3)]).astype(complex)
    return x, z


def _observable_from_label(label: str) -> np.ndarray:
    X, Z = _qutrit_xz()
    idx = int(label[1:])
    if label[0] in {"A", "B", "C", "D"}:
        return Z @ np.linalg.matrix_power(X, idx)
    raise ValueError(f"unknown label {label!r}")


def _two_vertex_graph() -> Graph:
    graph = Graph()
    graph.add_vertices(2)
    graph.add_edge(0, 1, weight=1)
    return graph


def test_build_sampler_circuits_for_two_vertex_graph_returns_nine_unique_settings() -> None:
    state_circuit = QuantumCircuit(4)

    sampler_circuits, metadata = build_sampler_circuits_from_graph(
        state_circuit=state_circuit,
        graph=_two_vertex_graph(),
        E=canonical_Ez(),
        observable_from_label=_observable_from_label,
        d=3,
        qutrit_qubits=[(0, 1), (2, 3)],
    )

    assert len(sampler_circuits) == 9
    assert len(set(metadata["measurement_settings"])) == 9
    assert len(metadata["circuits_by_setting"]) == 9
    assert metadata["setting_by_circuit_index"][0] in metadata["circuits_by_setting"]
    assert metadata["qutrit_qubits"] == ((0, 1), (2, 3))

    powers = {tuple(term["powers"]) for term in metadata["terms"]}
    assert (1, 1) in powers
    assert (2, 2) in powers


def test_build_sampler_circuits_computes_default_observables_from_labels() -> None:
    state_circuit = QuantumCircuit(4)

    sampler_circuits, metadata = build_sampler_circuits_from_graph(
        state_circuit=state_circuit,
        graph=_two_vertex_graph(),
        E=canonical_Ez(),
        d=3,
        qutrit_qubits=[(0, 1), (2, 3)],
    )

    assert len(sampler_circuits) == 9
    assert metadata["measurement_settings"][0] == ("A0", "B0")


def test_drop_conjugate_half_keeps_settings_but_reduces_terms() -> None:
    state_circuit = QuantumCircuit(4)
    kwargs = dict(
        state_circuit=state_circuit,
        graph=_two_vertex_graph(),
        E=canonical_Ez(),
        observable_from_label=_observable_from_label,
        d=3,
        qutrit_qubits=[(0, 1), (2, 3)],
    )

    full_circuits, full_metadata = build_sampler_circuits_from_graph(
        **kwargs,
        drop_conjugate_half=False,
    )
    half_circuits, half_metadata = build_sampler_circuits_from_graph(
        **kwargs,
        drop_conjugate_half=True,
    )

    assert len(full_circuits) == 9
    assert len(half_circuits) == 9
    assert set(full_metadata["measurement_settings"]) == set(
        half_metadata["measurement_settings"]
    )
    assert len(half_metadata["terms"]) < len(full_metadata["terms"])


def test_default_qutrit_qubits_are_adjacent_two_qubit_blocks() -> None:
    state_circuit = QuantumCircuit(4)

    _, metadata = build_sampler_circuits_from_graph(
        state_circuit=state_circuit,
        graph=_two_vertex_graph(),
        E=canonical_Ez(),
        observable_from_label=_observable_from_label,
        d=3,
        qutrit_qubits=None,
    )

    assert metadata["qutrit_qubits"] == ((0, 1), (2, 3))
    for setting in metadata["measurement_settings"]:
        assert metadata["qutrit_bit_indices_by_setting"][setting] == [(0, 1), (2, 3)]


def test_sampler_circuits_use_local_two_qubit_unitaries_only() -> None:
    state_circuit = QuantumCircuit(4)

    sampler_circuits, _ = build_sampler_circuits_from_graph(
        state_circuit=state_circuit,
        graph=_two_vertex_graph(),
        E=canonical_Ez(),
        observable_from_label=_observable_from_label,
        d=3,
        qutrit_qubits=[(0, 1), (2, 3)],
    )

    for circuit in sampler_circuits:
        local_basis_gates = [
            instruction
            for instruction in circuit.data
            if instruction.operation.label
            and str(instruction.operation.label).startswith("meas_")
        ]
        assert len(local_basis_gates) == 2
        assert all(len(instruction.qubits) == 2 for instruction in local_basis_gates)
        assert all(len(instruction.qubits) <= 2 for instruction in circuit.data)


def test_run_sampler_circuits_to_counts_by_setting_returns_bell_ready_counts() -> None:
    circuits = [QuantumCircuit(2), QuantumCircuit(2)]
    metadata = {
        "setting_by_circuit_index": [("A0", "B0"), ("A0", "B1")],
    }
    expected_counts = [
        {"00": 7, "01": 1},
        {"10": 3, "11": 5},
    ]
    sampler = _FakeSampler(expected_counts)

    counts_by_setting, run_info = run_sampler_circuits_to_counts_by_setting(
        sampler_circuits=circuits,
        metadata=metadata,
        sampler=sampler,
        shots=8,
        transpile_circuits=False,
    )

    assert counts_by_setting == {
        ("A0", "B0"): {"00": 7, "01": 1},
        ("A0", "B1"): {"10": 3, "11": 5},
    }
    assert sampler.run_circuits == circuits
    assert sampler.run_shots == 8
    assert run_info["sampler"] is sampler
    assert run_info["result"] is sampler.fake_result
    assert run_info["transpiled_circuits"] == circuits


def test_two_qutrit_graph_sampler_path_recovers_quantum_value_six() -> None:
    graph = _two_vertex_graph()
    state_circuit = QuantumCircuit(4)
    state_circuit.initialize(_encoded_two_qutrit_graph_state(), range(4))

    sampler_circuits, metadata = build_sampler_circuits_from_graph(
        state_circuit=state_circuit,
        graph=graph,
        E=canonical_Ez(),
        d=3,
        qutrit_qubits=[(0, 1), (2, 3)],
    )
    counts_by_setting = _statevector_counts_by_setting(
        sampler_circuits,
        metadata,
        shots=3**12,
    )

    value = compute_bell_value_from_counts(
        counts_by_setting,
        metadata["terms"],
        metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )

    np.testing.assert_allclose(value, 6.0, atol=1e-3)


def test_candidate_sampler_paths_recover_reference_quantum_values() -> None:
    cases = [
        ("two_qutrit", 2, [(0, 1, 1)], 6.0),
        ("ghz3", 3, [(0, 1, 1), (0, 2, 1)], 6.0),
        ("ame43", 4, [(0, 1, 1), (0, 3, 1), (1, 2, 1), (2, 3, 2)], 8.0),
    ]

    for candidate, num_qutrits, edges, expected in cases:
        state_circuit = QuantumCircuit(2 * num_qutrits)
        state_circuit.initialize(
            _encoded_qutrit_graph_state(num_qutrits, edges),
            range(2 * num_qutrits),
        )

        sampler_circuits, metadata = build_sampler_circuits_for_candidate(
            candidate=candidate,
            state_circuit=state_circuit,
            E=canonical_Ez(),
            d=3,
        )
        counts_by_setting = _statevector_counts_by_setting(
            sampler_circuits,
            metadata,
            shots=3**14,
        )

        value = compute_bell_value_from_counts(
            counts_by_setting,
            metadata["terms"],
            metadata["qutrit_bit_indices_by_setting"],
            **decoding_kwargs_from_metadata(metadata),
        )

        np.testing.assert_allclose(
            value,
            expected,
            atol=3e-3,
            err_msg=f"candidate={candidate}",
        )


def test_noncanonical_encoding_recovers_two_qutrit_bell_value() -> None:
    from pathlib import Path

    from qiskit import QuantumCircuit

    from bell_functionals_qutrit.bell_builders import candidate_statevector

    root = Path(__file__).resolve().parents[1]
    e_path = (
        root
        / "basis_direct_encoding_benchmarks"
        / "quantum_circuits"
        / "two_qutrit"
        / "monomial_full__sup013_P102_ph112"
        / "E.npy"
    )
    if not e_path.is_file():
        pytest.skip(f"missing benchmark encoding fixture: {e_path}")

    E = np.load(e_path)
    state = candidate_statevector("two_qutrit", E)
    qc = QuantumCircuit(4)
    qc.initialize(state.data, range(4))

    sampler_circuits, metadata = build_sampler_circuits_for_candidate(
        candidate="two_qutrit",
        state_circuit=qc,
        E=E,
    )
    counts_by_setting = _statevector_counts_by_setting(
        sampler_circuits,
        metadata,
        shots=3**12,
    )

    value = compute_bell_value_from_counts(
        counts_by_setting,
        metadata["terms"],
        metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )

    np.testing.assert_allclose(value.real, 6.0, atol=1e-2)


def test_counts_by_setting_from_sampler_result_supports_named_classical_register() -> None:
    metadata = {"setting_by_circuit_index": [("A0", "B0")]}
    result = _FakeResultWithRegisterName("qutrit_meas", {"0000": 8})

    counts_by_setting = counts_by_setting_from_sampler_result(result, metadata)

    assert counts_by_setting == {("A0", "B0"): {"0000": 8}}


class _FakeSampler:
    def __init__(self, counts_by_index: list[dict[str, int]]) -> None:
        self.fake_result = _FakeResult(counts_by_index)
        self.run_circuits = None
        self.run_shots = None

    def run(self, circuits, shots: int | None = None):
        self.run_circuits = circuits
        self.run_shots = shots
        return _FakeJob(self.fake_result)


class _FakeJob:
    def __init__(self, result) -> None:
        self._result = result

    def result(self):
        return self._result


class _FakeResult:
    def __init__(self, counts_by_index: list[dict[str, int]]) -> None:
        self._entries = [_FakeResultEntry(counts) for counts in counts_by_index]

    def __getitem__(self, index: int):
        return self._entries[index]


class _FakeResultEntry:
    def __init__(self, counts: dict[str, int]) -> None:
        self.data = _FakeData(counts)


class _FakeData:
    def __init__(self, counts: dict[str, int]) -> None:
        self.meas = _FakeMeas(counts)


class _FakeMeas:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


def _encoded_two_qutrit_graph_state() -> np.ndarray:
    return _encoded_qutrit_graph_state(2, [(0, 1, 1)])


def _encoded_qutrit_graph_state(
    num_qutrits: int,
    edges: list[tuple[int, int, int]],
) -> np.ndarray:
    from bell_functionals_qutrit.bell_builders import _qutrit_graph_state
    from bell_functionals_qutrit.encoding import encode_qutrit_state

    qutrit_state = _qutrit_graph_state(num_qutrits, edges)
    return encode_qutrit_state(qutrit_state, canonical_Ez(), num_qutrits)


def _statevector_counts_by_setting(
    sampler_circuits,
    metadata,
    shots: int,
) -> dict[tuple, dict[str, int]]:
    counts_by_setting = {}
    for index, setting in enumerate(metadata["setting_by_circuit_index"]):
        circuit = sampler_circuits[index].remove_final_measurements(inplace=False)
        probabilities = Statevector.from_instruction(circuit).probabilities_dict()
        counts_by_setting[setting] = {
            bitstring: round(probability * shots)
            for bitstring, probability in probabilities.items()
            if round(probability * shots)
        }
    return counts_by_setting


class _FakeResultWithRegisterName:
    def __init__(self, register_name: str, counts: dict[str, int]) -> None:
        self._entry = _FakeResultEntryWithRegisterName(register_name, counts)

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self._entry


class _FakeResultEntryWithRegisterName:
    def __init__(self, register_name: str, counts: dict[str, int]) -> None:
        self.data = _FakeDataWithRegisterName(register_name, counts)


class _FakeDataWithRegisterName:
    def __init__(self, register_name: str, counts: dict[str, int]) -> None:
        setattr(self, register_name, _FakeMeas(counts))

    def keys(self):
        return ("qutrit_meas",)
