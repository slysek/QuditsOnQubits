from types import SimpleNamespace

import pytest
import execute


@pytest.mark.parametrize('shots,randomizations', [(5000, 8), (1024, 16), (512, 16)])
def test_ibm_shots_are_exact(shots, randomizations):
    options = execute.ibm_runtime_options(
        {'shots': shots, 'reservation_seconds': 220}, None
    )
    twirling = options['twirling']
    assert twirling['num_randomizations'] == randomizations
    assert twirling['num_randomizations'] * twirling['shots_per_randomization'] == shots
    assert 'execution' not in options


def test_ibm_delay_preserves_reset_and_server_budget():
    backend = SimpleNamespace(configuration=lambda: SimpleNamespace(rep_delay_range=[0, .002]))
    options = execute.ibm_runtime_options(
        {'shots': 5000, 'reservation_seconds': 220, 'rep_delay': .000075}, backend
    )
    assert options['execution'] == {'init_qubits': True, 'rep_delay': .000075}
    assert options['max_execution_time'] == 220


@pytest.mark.parametrize('delay', [-1, .003, float('nan'), float('inf'), True])
def test_ibm_invalid_delay_rejected(delay):
    backend = SimpleNamespace(configuration=lambda: SimpleNamespace(rep_delay_range=[0, .002]))
    with pytest.raises(ValueError, match='rep_delay'):
        execute.ibm_runtime_options(
            {'shots': 5000, 'reservation_seconds': 220, 'rep_delay': delay}, backend
        )


def test_iqm_partition_keeps_complete_arm_and_both_calibrations():
    import rerun5000
    mapping = [
        {'state': 'ghz3', 'name': 'candidate', 'variant': 'optimal', 'setting_index': i}
        for i in range(12)
    ] + [
        {'state': 'ghz3', 'name': 'baseline', 'variant': 'standard', 'setting_index': 0},
        {'calibration': 0}, {'calibration': 1},
    ]
    circuits, selected = rerun5000.select_arm(list(range(15)), mapping, ('ghz3', 'candidate', 'optimal'))
    assert circuits == list(range(12)) + [13, 14]
    assert len(selected) == 14
    with pytest.raises(ValueError, match='complete'):
        rerun5000.select_arm(list(range(14)), mapping[:-1], ('ghz3', 'candidate', 'optimal'))


def test_stage_context_restores_global_ledger_even_on_failure(tmp_path, monkeypatch):
    import rerun5000 as r
    ledger = tmp_path / 'ledger'
    monkeypatch.setattr(r, 'LEDGER', ledger)
    monkeypatch.setattr(execute, 'ROOT', tmp_path / 'original')
    fez = SimpleNamespace(name='ibm_fez')
    service = SimpleNamespace(backend=lambda name: fez if name == 'ibm_fez' else None)
    monkeypatch.setattr(r.hardware, 'backend_for', lambda provider: (None, service))
    original_getter = execute.backend_for

    def fail_batch(*args):
        assert execute.ROOT == ledger / 'fez'
        raise RuntimeError('test loading failure')

    monkeypatch.setattr(execute, 'make_batch', fail_batch)
    with r.configured('ibm') as (backend, _):
        assert backend.name == 'ibm_fez'
        assert execute.ROOT == ledger
        with pytest.raises(RuntimeError, match='test loading failure'):
            execute.make_batch('ibm', backend, 1, 1, ['two_qutrit'])
        assert execute.ROOT == ledger
    assert execute.ROOT == tmp_path / 'original'
    assert execute.backend_for is original_getter
    assert execute.make_batch is fail_batch


def test_options_validate_with_installed_runtime():
    from qiskit_ibm_runtime.options import SamplerOptions
    backend = SimpleNamespace(configuration=lambda: SimpleNamespace(rep_delay_range=[0, .002]))
    options = execute.ibm_runtime_options(
        {'shots': 5000, 'reservation_seconds': 220, 'rep_delay': .000075}, backend
    )
    sampler_options = SamplerOptions(**options)
    assert sampler_options.twirling.num_randomizations * sampler_options.twirling.shots_per_randomization == 5000


def test_exact5000_survives_prepare_submit_and_retrieve(tmp_path, monkeypatch):
    import json
    import qiskit_ibm_runtime
    from qiskit import QuantumCircuit

    monkeypatch.setattr(execute, 'ROOT', tmp_path)
    backend = SimpleNamespace(name='ibm_fez',
        configuration=lambda: SimpleNamespace(simulator=False, rep_delay_range=[0, .002]),
        status=lambda: SimpleNamespace(operational=True))
    job = SimpleNamespace(job_id=lambda: 'test-job-id', status=lambda: 'DONE',
        result=lambda: [SimpleNamespace(join_data=lambda: SimpleNamespace(get_counts=lambda: {'0': 4900, '1': 100}))],
        metrics=lambda: {'usage': {'quantum_seconds': 1}})
    service = SimpleNamespace(usage=lambda: {'usage_remaining_seconds': 243}, job=lambda ident: job)
    monkeypatch.setattr(execute, 'backend_for', lambda provider: (backend, service))
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    monkeypatch.setattr(execute, 'make_batch', lambda *args: ([qc], [{'setting_index': 0}]))

    class Sampler:
        def __init__(self, mode, options):
            assert mode is backend
            assert options['twirling']['num_randomizations'] == 8
            assert options['twirling']['shots_per_randomization'] == 625
            assert options['execution']['rep_delay'] == .000075

        def run(self, circuits, shots):
            assert shots == 5000 and len(circuits) == 1
            return job

    monkeypatch.setattr(qiskit_ibm_runtime, 'SamplerV2', Sampler)
    directory = execute.prepare('ibm', 'exact5000', 5000, 1, 220, ['two_qutrit'], rep_delay=.000075)
    execute.submit('ibm', 'exact5000')
    execute.retrieve('ibm', 'exact5000')
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert receipt['status'] == 'completed'
    assert receipt['shots'] == 5000 and receipt['charged_seconds'] == 1
    assert sum(json.loads((directory / 'counts.json').read_text())[0].values()) == 5000
    with pytest.raises(ValueError, match='duplicate'):
        execute.submit('ibm', 'exact5000')
