"""Offline campaign regression tests; no provider connection is permitted."""
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from scripts import iqm_randomized_bell_campaign as campaign
from qudits_on_qubits.experiments import RandomizedBlocks, TranspilationConfig
from qudits_on_qubits.experiments.setting_schedule import generate_schedule
from qudits_on_qubits.reference_experiments import get_reference_experiment

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/working/iqm/bell_randomized_mitigation.ipynb'
GATES = ROOT / 'experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909'


@pytest.fixture
def prepared():
    artifacts, evidence = campaign.load_state(GATES)
    rng = random.Random(2026)
    schedule = generate_schedule(get_reference_experiment('two_qutrit'),
        RandomizedBlocks(12, 16, 512), _randbelow=rng.randrange,
        _source='test_seeded', _seed=2026)
    class Compiler:
        def compile(self, circuits, config):
            return SimpleNamespace(circuits=transpile(list(circuits), basis_gates=['r', 'cz'], optimization_level=0))
    measurements, batches, calibrations = campaign.prepare_campaign(
        artifacts, schedule, Compiler(), TranspilationConfig(), 2, 42)
    return artifacts, evidence, schedule, measurements, batches, calibrations


def test_gate_validation_and_reference():
    _, evidence = campaign.load_state(GATES)
    assert evidence['F3']['N_2q'] == 2
    assert evidence['CZ3']['N_2q'] == 6
    assert evidence['state_fidelity'] > 1 - 1e-10


@pytest.mark.parametrize('seed', range(16))
def test_native_pauli_twirling_preserves_unitary(seed):
    circuit = QuantumCircuit(2)
    circuit.r(.31, .72, 0)
    circuit.cz(0, 1)
    circuit.r(.1, .4, 1)
    circuit.cz(1, 0)
    result = campaign.twirl_cz(circuit, np.random.default_rng(seed))
    assert Operator(result).equiv(Operator(circuit))
    assert result.count_ops()['cz'] == 2
    assert set(result.count_ops()) <= {'r', 'cz'}


def test_physical_isa_preserves_iqm_serializer_loci_with_layout():
    from qiskit.transpiler import Layout, TranspileLayout
    from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
    circuit = QuantumCircuit(4, 4)
    circuit.r(.3, .7, 0)
    circuit.cz(0, 1)
    circuit.measure(range(4), range(4))
    initial = Layout({circuit.qubits[i]: (i + 2) % 4 for i in range(4)})
    initial.add_register(circuit.qregs[0])
    circuit._layout = TranspileLayout(initial,
                                      {q: i for i, q in enumerate(circuit.qubits)})
    mapping = {i: f'QB{i+1}' for i in range(4)}
    materialized = campaign.physical_isa(circuit)
    original = serialize_instructions(circuit, mapping)
    actual = serialize_instructions(materialized, mapping)
    assert [(i.name, i.locus) for i in original] == [(i.name, i.locus) for i in actual]
    assert campaign.measurement_mapping(materialized) == (2, 3, 0, 1)
    assert materialized.layout is None


def test_readout_correction_recovers_correlated_distribution():
    matrix = .9 * np.eye(16) + .1 * np.roll(np.eye(16), 3, axis=0)
    counts = [{format(i, '04b'): int(round(p * 1000)) for i, p in enumerate(matrix[:, j])}
              for j in range(16)]
    actual = campaign.assignment_matrix(counts, 1000)
    ideal = np.arange(16, dtype=float)
    ideal /= ideal.sum()
    np.testing.assert_allclose(np.linalg.solve(actual, matrix @ ideal), ideal, atol=1e-14)
    with pytest.raises(ValueError, match='ill-conditioned'):
        campaign.assignment_matrix([{'0000': 100}] * 16, 100)


@pytest.mark.parametrize('counts,shots', [({'0': 10}, 10), ({'0000': -1}, 1),
    ({'0000': True}, 1), ({'0000': 10}, 11), ({'0000': 1.5}, 1)])
def test_invalid_counts_rejected(counts, shots):
    with pytest.raises(ValueError):
        campaign.count_vector(counts, shots)


def test_blocks_budget_and_folds(prepared):
    _, _, schedule, _, batches, calibrations = prepared
    total = len(schedule.blocks) * schedule.config.shots_per_draw
    assert sum(len(b['circuits']) * b['shots'] for b in batches) == 4 * total
    assert all(len(circuits) == 16 for circuits in calibrations.values())
    raw = batches[0]['circuits'][0]
    for batch in batches[1:]:
        transformed = batch['circuits'][0]
        assert transformed.count_ops()['cz'] == batch['factor'] * raw.count_ops()['cz']
        assert Operator(transformed.remove_final_measurements(inplace=False)).equiv(
            Operator(raw.remove_final_measurements(inplace=False)))


class FakeBackend:
    name = 'offline-recording-backend'
    calibration_set_id = 'test-only'

    def __init__(self):
        self.calls = []

    def run(self, circuits, **options):
        self.calls.append(options)
        counts = []
        for circuit in circuits:
            state = Statevector(circuit.remove_final_measurements(inplace=False))
            state.seed(123)
            counts.append({str(k): int(v) for k, v in state.sample_counts(options['shots']).items()})
        result = SimpleNamespace(get_counts=lambda index: counts[index])
        return SimpleNamespace(job_id=lambda: f'fake-{len(self.calls)}', result=lambda: result)


def test_offline_execution_checkpointing_analysis_and_no_resubmit(prepared, tmp_path):
    _, evidence, schedule, measurements, batches, calibrations = prepared
    backend = FakeBackend()
    target = tmp_path / 'campaign'
    with pytest.raises(PermissionError):
        campaign.execute_campaign(target, schedule, evidence, batches, calibrations, backend)
    assert not backend.calls and not target.exists()
    campaign.execute_campaign(target, schedule, evidence, batches, calibrations, backend,
        allow_hardware=True, calibration_shots=32, max_circuits=7)
    result = campaign.analyze_campaign(target, schedule, measurements.metadata)
    assert {r['variant'] for r in result['rows']} == {'RAW', 'MM', 'MM_TWIRL_DD', 'MM_TWIRL_DD_ZNE'}
    raw, mm = result['rows'][:2]
    assert raw['real'] == pytest.approx(result['raw_block_analysis']['raw']['real'])
    assert raw['imag'] == pytest.approx(result['raw_block_analysis']['raw']['imag'])
    assert raw['real'] == pytest.approx(mm['real'])
    assert raw['imag'] == pytest.approx(mm['imag'])
    assert result['mitigated_uncertainty'] is None
    assert any(o['circuit_compilation_options'].dd_mode.value == 'enabled' for o in backend.calls)
    assert backend.calls[0]['circuit_compilation_options'].dd_mode.value == 'disabled'
    before = len(backend.calls)
    with pytest.raises(FileExistsError):
        campaign.execute_campaign(target, schedule, evidence, batches, calibrations, backend, allow_hardware=True)
    assert len(backend.calls) == before
    names = json.loads((target / 'index.json').read_text())
    (target / f'{names[-1]}.counts.json').unlink()
    with pytest.raises(FileNotFoundError):
        campaign.analyze_campaign(target, schedule, measurements.metadata)


def test_notebook_offline_defaults_and_hardware_guard(monkeypatch, tmp_path):
    import nbformat
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = {c.id: c.source for c in notebook.cells if c.cell_type == 'code'}
    assert all(c.execution_count is None and c.outputs == [] for c in notebook.cells if c.cell_type == 'code')
    for key, source in cells.items():
        compile(source, key, 'exec')
    ns = {'display': lambda x: None}
    exec(cells['setup'], ns)
    exec(cells['configuration'], ns)
    ns.update(WORK=tmp_path, SCHEDULE_PATH=tmp_path / 'schedule.json', SETTING_DRAWS=12,
              SHOTS_PER_DRAW=16, TWIRLING_INSTANCES=2)
    exec(cells['state-and-schedule'], ns)
    exec(cells['ideal-check'], ns)
    iqm_backend_module = importlib.import_module('qudits_on_qubits.experiments.backends.iqm')
    def forbidden(*args, **kwargs):
        pytest.fail('Provider loader must never run offline')
    monkeypatch.setattr(iqm_backend_module, 'load_iqm_backend', forbidden)
    exec(cells['offline-plan'], ns)
    assert ns['RUN_HARDWARE'] is False and ns['RUN_RECOVERY'] is False
    exec(cells['hardware'], ns)
    exec(cells['recovery'], ns)
    exec(cells['analysis'], ns)
