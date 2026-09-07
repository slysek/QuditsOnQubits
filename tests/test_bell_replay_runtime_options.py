import os

import pytest

from qudits_on_qubits._ibm_runtime import runtime_account_options


def test_runtime_settings_use_local_configuration_without_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.startswith('QISKIT_IBM_'):
            monkeypatch.delenv(key)
    (tmp_path / '.env').write_text('QISKIT_IBM_TOKEN=test-key\nQISKIT_IBM_INSTANCE=test-instance\n')
    before = dict(os.environ)
    assert runtime_account_options() == {'token': 'test-key', 'channel': 'ibm_quantum_platform', 'instance': 'test-instance'}
    assert dict(os.environ) == before
    monkeypatch.setenv('QISKIT_IBM_TOKEN', 'process-test-key')
    assert runtime_account_options(instance='explicit')['token'] == 'process-test-key'
    assert runtime_account_options(instance='explicit')['instance'] == 'explicit'


def test_named_account_bypasses_environment_and_invalid_channel_does_not_echo_secrets(monkeypatch):
    monkeypatch.setenv('QISKIT_IBM_TOKEN', 'not-a-real-secret')
    monkeypatch.setenv('QISKIT_IBM_CHANNEL', 'invalid-secret-channel')
    assert runtime_account_options(account_name='saved') == {'name': 'saved'}
    with pytest.raises(ValueError) as error:
        runtime_account_options()
    assert 'not-a-real-secret' not in str(error.value)
    assert 'invalid-secret-channel' not in str(error.value)
