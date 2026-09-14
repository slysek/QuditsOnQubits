"""Regression tests for the user's optimized two-qutrit Bell experiment."""
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

SOURCE = Path(__file__).resolve().parents[1] / 'experiment_inputs/bell_optimized/raw_reference.json'


@pytest.fixture(scope='module')
def api():
    from qudits_on_qubits.bell_measurements import bell_optimized
    return bell_optimized


@pytest.fixture(scope='module')
def catalog(api):
    return api.build_catalog(SOURCE)


def test_bell_histograms_and_native_costs(api, catalog):
    assert len(catalog.baseline) == len(catalog.optimized) == 9
    assert catalog.evidence['preparation_ops'] == {'r': 6, 'cz': 3}
    for index, (old, new) in enumerate(zip(catalog.baseline, catalog.optimized, strict=True)):
        b0 = index % 3 == 0
        assert dict(old.count_ops()) == {'r': 14 if b0 else 22, 'cz': 6 if b0 else 9, 'measure': 4}
        assert dict(new.count_ops()) == {'r': 12 if b0 else 18, 'cz': 5 if b0 else 7, 'measure': 4}
        np.testing.assert_allclose(api.probabilities(old), api.probabilities(new), atol=1e-10, rtol=0)
    assert catalog.evidence['ideal_bell_baseline'] == pytest.approx(6, abs=1e-10)
    assert catalog.evidence['ideal_bell_optimized'] == pytest.approx(6, abs=1e-10)
    assert catalog.evidence['max_probability_error'] < 1e-10


def test_shared_preparation_extraction_preserves_parent(api, catalog):
    for old in catalog.baseline:
        prep, a, b = api.split_terminal_blocks(old)
        rebuilt = prep.compose(a, [2, 3]).compose(b, [0, 1])
        np.testing.assert_allclose(Operator(rebuilt).data,
            Operator(old.remove_final_measurements(inplace=False)).data, atol=1e-12, rtol=0)


def test_histograms_follow_classical_bits(api):
    circuit = QuantumCircuit(4, 4)
    circuit.x(0)
    circuit.measure([2, 3, 0, 1], range(4))
    assert np.argmax(api.probabilities(circuit)) == 4
    circuit.x(0)
    with pytest.raises(ValueError, match='terminal'):
        api.probabilities(circuit)


def test_reference_rejects_changed_data(api, tmp_path):
    import json
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    source['parents'][0]['instructions'][0]['params'][0] += .1
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(source), encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        api.build_catalog(path)


def test_frozen_weights_match_existing_bell_decoder(catalog):
    from qudits_on_qubits.bell_measurements.postprocessing import compute_bell_value_from_counts
    from qudits_on_qubits.bell_measurements.sampler_circuits import (
        build_sampler_circuits_for_candidate, decoding_kwargs_from_metadata,
    )
    _, metadata = build_sampler_circuits_for_candidate(
        'two_qutrit', QuantumCircuit(4), np.eye(4, 3), qutrit_qubits=[(2, 3), (0, 1)])
    for index, setting in enumerate(metadata['setting_by_circuit_index']):
        terms = [term for term in metadata['terms'] if tuple(term['settings']) == tuple(setting)]
        expected = [compute_bell_value_from_counts(
            {tuple(setting): {format(k, '04b'): 1}}, terms,
            metadata['qutrit_bit_indices_by_setting'], renormalize_after_discard=False,
            **decoding_kwargs_from_metadata(metadata)).real for k in range(16)]
        np.testing.assert_allclose(catalog.weights[index], expected, atol=1e-12, rtol=0)
