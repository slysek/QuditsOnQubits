from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.modules.pop("qudits_on_qubits", None)
import pytest

from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.models import (
    AerIdeal,
    BackendStatus,
    BenchmarkBasis,
    BootstrapConfig,
    CustomBackend,
    ExperimentSpec,
    IQMHardware,
    MitigationConfig,
    NoisySimulator,
    PathBasis,
    PiastQHardware,
    RetryConfig,
    TranspilationConfig,
)


def test_experiment_spec_normalizes_ghz_alias():
    spec = ExperimentSpec(state="ghz", basis=PathBasis(Path("basis")), backend=AerIdeal())

    assert spec.state == "ghz3"
    assert spec.to_safe_dict()["state"] == "ghz3"


@pytest.mark.parametrize("state", ["ghz4", "", 3])
def test_experiment_spec_rejects_unknown_states(state):
    with pytest.raises(ExperimentValidationError, match="state"):
        ExperimentSpec(state=state, basis=PathBasis(Path("basis")), backend=AerIdeal())


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_experiment_spec_rejects_invalid_shots(shots):
    with pytest.raises(ExperimentValidationError, match="shots"):
        ExperimentSpec(state="ghz3", basis=PathBasis(Path("basis")), backend=AerIdeal(), shots=shots)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": None, "candidate": None}, "exactly one"),
        ({"rank": 1, "candidate": "candidate"}, "exactly one"),
        ({"run_kind": "other", "rank": 1}, "run_kind"),
    ],
)
def test_benchmark_basis_validates_selection(kwargs, message):
    args = {"run_kind": "direct_basis_runs", "run_id": "run", "selection": "best"}
    args.update(kwargs)
    with pytest.raises(ExperimentValidationError, match=message):
        BenchmarkBasis(**args)


@pytest.mark.parametrize(
    "backend_factory",
    [lambda: IQMHardware("token=bad"), lambda: PiastQHardware(mode="bad")],
)
def test_backend_selections_are_validated(backend_factory):
    with pytest.raises(ExperimentValidationError):
        ExperimentSpec(state="ghz3", basis=PathBasis(Path("basis")), backend=backend_factory())


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: MitigationConfig(zne=True, zne_factors=(1, 2, 3)),
        lambda: MitigationConfig(zne=True, zne_factors=(1, 3, 3)),
        lambda: MitigationConfig(zne=True, zne_factors=(3, 5)),
        lambda: MitigationConfig(zne=True, zne_model="exponential"),
    ],
)
def test_mitigation_config_validates_zne(config_factory):
    with pytest.raises(ExperimentValidationError):
        config_factory()


def test_safe_serialization_excludes_backend_objects_and_credentials():
    backend = CustomBackend(instance=object(), identity="local", supports_resume=False)
    spec = ExperimentSpec(
        state="ghz3",
        basis=PathBasis(Path("basis")),
        backend=backend,
        tags={"purpose": "test"},
    )

    payload = spec.to_safe_dict()
    assert "instance" not in payload["backend"]
    assert "token=" not in repr(payload).lower()
    assert payload["backend"]["kind"] == "custom"


def test_builtin_spec_safe_round_trip():
    original = ExperimentSpec(
        state="ame43",
        basis=BenchmarkBasis("iqm_runs", "run-1", "top", candidate="candidate-1"),
        backend=PiastQHardware(mode="managed", owner="team"),
        mitigation=MitigationConfig(readout=True, zne=True),
        bootstrap=BootstrapConfig(samples=10),
        transpilation=TranspilationConfig(optimization_level=2),
        retry=RetryConfig(max_attempts=2),
        tags={"suite": "nightly"},
    )

    rebuilt = ExperimentSpec.from_safe_dict(original.to_safe_dict())
    assert rebuilt == original


def test_custom_and_noisy_backends_require_injection_to_reconstruct():
    custom = CustomBackend(instance=object(), identity="local")
    with pytest.raises(ExperimentValidationError, match="injection"):
        CustomBackend.from_safe_dict(custom.to_safe_dict())

    noisy = NoisySimulator(source=object(), identity="source")
    with pytest.raises(ExperimentValidationError, match="injection"):
        NoisySimulator.from_safe_dict(noisy.to_safe_dict())


@pytest.mark.parametrize(
    ("noise_model", "target_backend"),
    [(object(), None), (None, object())],
)
def test_noisy_simulator_source_mode_rejects_partial_model_configuration(
    noise_model, target_backend
):
    with pytest.raises(ExperimentValidationError, match="exactly"):
        NoisySimulator(source=object(), noise_model=noise_model, target_backend=target_backend)

def test_models_are_frozen():
    backend = AerIdeal()
    with pytest.raises(FrozenInstanceError):
        backend.seed_simulator = 7


def test_retry_config_validates_bounds():
    with pytest.raises(ExperimentValidationError):
        RetryConfig(max_attempts=0)


def test_status_is_a_string_enum():
    assert BackendStatus.SUBMISSION_UNKNOWN.value == "submission_unknown"
