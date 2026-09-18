import json

import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.targets import BackendTarget


@pytest.fixture
def garnet():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    return IQMFakeGarnet()


@pytest.mark.parametrize("strategy", ["preset_exact", "transpile_to_iqm_exact"])
def test_iqm_compiles_against_real_fake_backend_and_keeps_layout(garnet, strategy):
    target = BackendTarget.iqm("garnet", backend=garnet, strategy=strategy,
                               initial_layout=[0, 1, 3, 4], optimization_level=1)
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.rz(0.37, 3)
    compiled = target.compile(circuit, seed=7)
    assert compiled.layout is not None
    assert compiled.layout.initial_index_layout(filter_ancillas=True) == [0, 1, 3, 4]
    target.validate_native(compiled)
    assert set(compiled.count_ops()) <= {"r", "cz", "id", "delay"}
    manifest = target.to_dict()
    assert manifest["provider"] == "iqm"
    assert manifest["strategy"] == strategy
    assert manifest["calibration_set_id"] == str(garnet.architecture.calibration_set_id)
    assert manifest["iqm_architecture"] == garnet.architecture.model_dump(mode="json")
    assert "token" not in json.dumps(manifest).lower()


@pytest.mark.parametrize("strategy", ["preset_default", "transpile_to_iqm_default"])
def test_probability_only_iqm_strategies_rejected_before_loading(strategy):
    with pytest.raises(ValueError, match="relative phases|State preparation"):
        BackendTarget.iqm("garnet", strategy=strategy)


def test_iqm_rejects_profile_mismatch_and_unknown_profile(garnet):
    with pytest.raises(ValueError, match="profile|name|identity"):
        BackendTarget.iqm("emerald", backend=garnet)
    with pytest.raises(ValueError, match="garnet|emerald"):
        BackendTarget.iqm("other", backend=garnet)


def test_iqm_snapshot_does_not_retain_mutable_provider_architecture(garnet):
    target = BackendTarget.iqm("garnet", backend=garnet)
    before = target.to_dict()
    garnet.architecture.qubits.reverse()
    assert target.to_dict() == before
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    target.validate_native(target.compile(circuit, seed=2))


def test_iqm_loads_target_once_and_never_falls_back(monkeypatch, garnet):
    from qudits_on_qubits.benchmarks.direct_basis import iqm_backend
    calls = []
    def load(name, **kwargs):
        calls.append(name)
        return garnet
    monkeypatch.setattr(iqm_backend, "load_iqm_backend", load)
    target = BackendTarget.iqm("garnet", optimization_level=0)
    for seed in (1, 2):
        target.compile(QuantumCircuit(1), seed=seed)
    assert calls == ["garnet"]
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(iqm_backend, "load_iqm_backend", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        BackendTarget.iqm("emerald")


@pytest.mark.parametrize("explicit", [False, True])
def test_iqm_credentials_resolve_outside_installation(monkeypatch, tmp_path, garnet, explicit):
    from types import SimpleNamespace
    import iqm.qiskit_iqm
    from qudits_on_qubits.benchmarks.direct_basis import iqm_backend
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    settings = config_dir / ".env"
    settings.write_text("IQM_SERVER_URL=https://example.invalid\nIQM_TOKEN=test-token\n", encoding="utf-8")
    child = config_dir / "notebooks"
    child.mkdir()
    monkeypatch.chdir(child if not explicit else tmp_path)
    monkeypatch.setenv("IQM_SERVER_URL", "previous-url")
    monkeypatch.setenv("IQM_TOKEN", "previous-token")
    monkeypatch.setattr(iqm_backend, "default_env_path", lambda: tmp_path / "installation" / ".env")
    calls = []
    def provider(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(get_backend=lambda **options: garnet)
    monkeypatch.setattr(iqm.qiskit_iqm, "IQMProvider", provider)
    target = BackendTarget.iqm("garnet", **({"env_path": settings} if explicit else {}))
    assert calls == [("https://example.invalid", {"quantum_computer": "garnet"})]
    assert target.provider == "iqm"
    assert "test-token" not in json.dumps(target.to_dict())