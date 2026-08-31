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
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import (
    AerIdeal,
    BackendStatus,
    BenchmarkBasis,
    BootstrapConfig,
    CustomBackend,
    BellEstimate,
    ComplexComponents,
    ComplexConfidenceInterval,
    ConfidenceInterval,
    ExperimentSpec,
    IQMHardware,
    IQMQubitSelectorConfig,
    MitigationConfig,
    NoisySimulator,
    PathBasis,
    PiastQHardware,
    RetryConfig,
    ScalarEstimate,
    TranspilationConfig,
    WorkloadOptimizationConfig,
)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (AerIdeal(), ExecutionMode.IDEAL_SIMULATOR),
        (IQMHardware("garnet"), ExecutionMode.HARDWARE),
        (PiastQHardware(), ExecutionMode.HARDWARE),
        (
            CustomBackend(
                instance=object(),
                execution_mode=ExecutionMode.HARDWARE,
            ),
            ExecutionMode.HARDWARE,
        ),
        (NoisySimulator(source=object()), ExecutionMode.NOISY_SIMULATOR),
    ],
)
def test_backend_specs_serialize_one_explicit_execution_mode(backend, expected):
    assert backend.execution_mode is expected
    assert backend.to_safe_dict()["execution_mode"] == expected.value


def test_custom_backend_requires_typed_execution_mode():
    with pytest.raises(TypeError, match="execution_mode"):
        CustomBackend(instance=object())

    with pytest.raises(ExperimentValidationError, match="execution_mode"):
        CustomBackend(instance=object(), execution_mode="hardware")


def test_backend_deserialization_rejects_conflicting_execution_mode():
    with pytest.raises(ExperimentValidationError, match="execution_mode"):
        AerIdeal.from_safe_dict(
            {
                "kind": "aer_ideal",
                "seed_simulator": 123,
                "execution_mode": "hardware",
            }
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


def test_piastq_defaults_and_missing_mode_deserialization_to_managed():
    assert PiastQHardware().mode == "managed"
    assert PiastQHardware.from_safe_dict(
        {"kind": "piastq_hardware", "owner": "team"}
    ).mode == "managed"


@pytest.mark.parametrize("mode", ["auto", "direct"])
def test_piastq_rejects_non_managed_modes_with_migration_guidance(mode):
    with pytest.raises(
        ExperimentValidationError,
        match="managed.*separate environment",
    ):
        PiastQHardware(mode=mode)


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


def test_mitigation_config_twirling_defaults_and_safe_dict_round_trip():
    default = MitigationConfig()
    configured = MitigationConfig(
        circuit_twirling=True,
        twirling_instances=8,
        twirling_seed=12345,
    )

    assert default.circuit_twirling is False
    assert default.twirling_instances == 20
    assert default.twirling_seed is None
    assert MitigationConfig.from_safe_dict(configured.to_safe_dict()) == configured


def test_mitigation_config_accepts_historical_payload_without_twirling_fields():
    historical = {
        "readout": True,
        "zne": True,
        "zne_factors": [1, 3],
        "zne_model": "linear",
        "readout_max_age_hours": 12.0,
        "force_recalibration": False,
    }

    restored = MitigationConfig.from_safe_dict(historical)

    assert restored.circuit_twirling is False
    assert restored.twirling_instances == 20
    assert restored.twirling_seed is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MitigationConfig(circuit_twirling=1),
        lambda: MitigationConfig(twirling_instances=True),
        lambda: MitigationConfig(twirling_instances=0),
        lambda: MitigationConfig(twirling_instances=1.5),
        lambda: MitigationConfig(twirling_seed=True),
        lambda: MitigationConfig(twirling_seed=-1),
        lambda: MitigationConfig(twirling_seed=1.5),
    ],
)
def test_mitigation_config_rejects_invalid_twirling_values(factory):
    with pytest.raises(ExperimentValidationError, match="twirl"):
        factory()


def test_safe_serialization_excludes_backend_objects_and_credentials():
    backend = CustomBackend(
        instance=object(),
        identity="local",
        supports_resume=False,
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
    )
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


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "token:super-secret",
        "Authorization: Bearer super-secret",
        "https://user:super-secret@example.invalid/path",
        "https://example.invalid/path?access_token=super-secret",
        "unsafe\x07control",
    ],
)
def test_persisted_model_strings_reject_credential_material_without_echo(unsafe_text):
    with pytest.raises(ExperimentValidationError) as caught:
        CustomBackend(
            instance=object(),
            identity=unsafe_text,
            execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        )

    assert unsafe_text not in str(caught.value)
    assert caught.value.__cause__ is None


def test_tag_credential_field_name_is_rejected_without_echo():
    sensitive_value = "placeholder-value"

    with pytest.raises(ExperimentValidationError) as caught:
        ExperimentSpec(
            state="ghz3",
            basis=PathBasis(Path("basis")),
            backend=AerIdeal(),
            tags={"access_token": sensitive_value},
        )

    assert sensitive_value not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "backend",
    [
        IQMHardware("device", env_path=Path("private/iqm.env")),
        PiastQHardware(env_path=Path("private/piast.env")),
    ],
)
def test_hardware_safe_serialization_omits_process_local_env_path(backend):
    assert "env_path" not in backend.to_safe_dict()


@pytest.mark.parametrize(
    ("backend_type", "payload"),
    [
        (
            IQMHardware,
            {
                "kind": "iqm_hardware",
                "device": "device",
                "use_metrics": False,
                "env_path": "legacy/private.env",
            },
        ),
        (
            PiastQHardware,
            {
                "kind": "piastq_hardware",
                "mode": "managed",
                "owner": "team",
                "env_path": "legacy/private.env",
            },
        ),
    ],
)
def test_hardware_deserialization_does_not_restore_legacy_env_path(backend_type, payload):
    assert backend_type.from_safe_dict(payload).env_path is None


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


def test_transpilation_config_round_trips_initial_layout():
    config = TranspilationConfig(
        optimization_level=3,
        seed_transpiler=9,
        initial_layout=(16, 17, 18, 19),
    )

    assert TranspilationConfig.from_safe_dict(config.to_safe_dict()) == config
    assert config.to_safe_dict()["initial_layout"] == [16, 17, 18, 19]


@pytest.mark.parametrize(
    "layout",
    [
        (),
        (0, 0),
        (-1, 2),
        (True, 2),
        [0, "1"],
        {0, 1},
        {0: "zero", 1: "one"},
        (index for index in range(2)),
    ],
)
def test_transpilation_config_rejects_invalid_initial_layout(layout):
    with pytest.raises(ExperimentValidationError, match="initial_layout"):
        TranspilationConfig(initial_layout=layout)


def test_workload_optimization_config_round_trips_with_safe_shape():
    config = WorkloadOptimizationConfig(
        initial_layouts=((0, 1, 2, 3, 4, 7), (4, 7, 2, 3, 0, 1)),
        seed_transpilers=(3, 7),
    )

    payload = config.to_safe_dict()

    assert payload == {
        "initial_layouts": [[0, 1, 2, 3, 4, 7], [4, 7, 2, 3, 0, 1]],
        "seed_transpilers": [3, 7],
        "require_exact_physical_qubit_set": True,
        "prefer_calibration_metrics": True,
    }
    assert WorkloadOptimizationConfig.from_safe_dict(payload) == config


def test_iqm_qubit_selector_config_round_trips_with_safe_shape():
    config = IQMQubitSelectorConfig(remove_qubits=[1, 4])

    payload = config.to_safe_dict()

    assert payload == {
        "top_k": 10,
        "num_trials": 2000,
        "cost_function": "cz",
        "readout_mode": "none",
        "remove_qubits": [1, 4],
    }
    assert config.remove_qubits == (1, 4)
    assert IQMQubitSelectorConfig.from_safe_dict(payload) == config


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("top_k", 0),
        ("top_k", True),
        ("num_trials", 0),
        ("num_trials", 1.5),
        ("cost_function", "iswap"),
        ("cost_function", []),
        ("readout_mode", "raw"),
        ("readout_mode", []),
        ("remove_qubits", (1, 1)),
        ("remove_qubits", (-1,)),
    ],
)
def test_iqm_qubit_selector_config_rejects_invalid_values(field_name, value):
    kwargs = {field_name: value}

    with pytest.raises(ExperimentValidationError, match=field_name):
        IQMQubitSelectorConfig(**kwargs)


@pytest.mark.parametrize(
    "remove_qubits",
    ["1", b"1", {1}, (index for index in range(2)), {1: "one"}, (True,)],
)
def test_iqm_qubit_selector_config_rejects_non_sequence_or_invalid_qubits(remove_qubits):
    with pytest.raises(ExperimentValidationError, match="remove_qubits"):
        IQMQubitSelectorConfig(remove_qubits=remove_qubits)


def test_workload_optimization_config_allows_selector_as_only_layout_source():
    selector = IQMQubitSelectorConfig(
        top_k=4,
        num_trials=100,
        cost_function="clifford",
        readout_mode="fidelity",
        remove_qubits=(2,),
    )
    config = WorkloadOptimizationConfig(
        initial_layouts=(), iqm_qubit_selector=selector
    )

    payload = config.to_safe_dict()

    assert config.initial_layouts == ()
    assert config.iqm_qubit_selector is selector
    assert payload["iqm_qubit_selector"] == selector.to_safe_dict()
    assert WorkloadOptimizationConfig.from_safe_dict(payload) == config


def test_workload_optimization_config_requires_a_layout_source():
    with pytest.raises(
        ExperimentValidationError,
        match="initial_layouts require at least one layout source",
    ):
        WorkloadOptimizationConfig(initial_layouts=())


def test_workload_optimization_config_rejects_invalid_selector_payload():
    with pytest.raises(ExperimentValidationError, match="iqm_qubit_selector"):
        WorkloadOptimizationConfig(initial_layouts=((0, 1),), iqm_qubit_selector={})


def test_workload_optimization_config_preserves_legacy_payload_shape():
    payload = {
        "initial_layouts": [[0, 1]],
        "seed_transpilers": [3],
        "require_exact_physical_qubit_set": True,
        "prefer_calibration_metrics": False,
    }

    config = WorkloadOptimizationConfig.from_safe_dict(payload)

    assert config.iqm_qubit_selector is None
    assert config.to_safe_dict() == payload


def test_workload_optimization_config_normalizes_sequences_to_tuples():
    config = WorkloadOptimizationConfig(
        initial_layouts=[[0, 1], [2, 3]],
        seed_transpilers=[3, 7],
    )

    assert config.initial_layouts == ((0, 1), (2, 3))
    assert config.seed_transpilers == (3, 7)


@pytest.mark.parametrize(
    "initial_layouts",
    [
        ((0, 1), (2, 3, 4)),
        ((0, 0),),
        ((-1, 0),),
        ((0, 1), (0, 1)),
        ((True, 1),),
        "01",
        b"01",
        ({0, 1},),
        ({0: "zero", 1: "one"},),
        ((index for index in range(2)),),
    ],
)
def test_workload_optimization_config_rejects_invalid_layouts(initial_layouts):
    with pytest.raises(ExperimentValidationError, match="initial_layouts"):
        WorkloadOptimizationConfig(initial_layouts=initial_layouts)


@pytest.mark.parametrize(
    "seed_transpilers",
    [
        (),
        (3, 3),
        (-1,),
        (True,),
        (1.5,),
        ("3",),
        "37",
        b"37",
        {3, 7},
        (seed for seed in (3, 7)),
    ],
)
def test_workload_optimization_config_rejects_invalid_seeds(seed_transpilers):
    with pytest.raises(ExperimentValidationError, match="seed_transpilers"):
        WorkloadOptimizationConfig(
            initial_layouts=((0, 1),),
            seed_transpilers=seed_transpilers,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("require_exact_physical_qubit_set", 1),
        ("prefer_calibration_metrics", "true"),
    ],
)
def test_workload_optimization_config_requires_boolean_flags(field_name, value):
    kwargs = {
        "initial_layouts": ((0, 1),),
        field_name: value,
    }

    with pytest.raises(ExperimentValidationError, match=field_name):
        WorkloadOptimizationConfig(**kwargs)


def test_experiment_spec_round_trips_workload_optimization():
    workload_optimization = WorkloadOptimizationConfig(
        initial_layouts=((0, 1, 2), (3, 4, 5)),
        seed_transpilers=(3, 7),
    )
    original = ExperimentSpec(
        state="ghz3",
        basis=PathBasis(Path("basis")),
        backend=AerIdeal(),
        workload_optimization=workload_optimization,
    )

    payload = original.to_safe_dict()

    assert payload["workload_optimization"] == workload_optimization.to_safe_dict()
    assert ExperimentSpec.from_safe_dict(payload) == original


def test_experiment_spec_restores_none_from_legacy_payload_without_workload_optimization():
    original = ExperimentSpec(
        state="ghz3",
        basis=PathBasis(Path("basis")),
        backend=AerIdeal(),
    )
    payload = original.to_safe_dict()

    assert original.workload_optimization is None
    assert payload["workload_optimization"] is None
    payload.pop("workload_optimization")
    assert ExperimentSpec.from_safe_dict(payload) == original


def test_experiment_spec_rejects_invalid_workload_optimization_type():
    with pytest.raises(ExperimentValidationError, match="workload_optimization"):
        ExperimentSpec(
            state="ghz3",
            basis=PathBasis(Path("basis")),
            backend=AerIdeal(),
            workload_optimization={},
        )


def test_experiment_spec_preserves_legacy_structural_match_arguments():
    assert ExperimentSpec.__match_args__ == (
        "state",
        "basis",
        "backend",
        "shots",
        "mitigation",
        "uncertainty",
        "transpilation",
        "retry",
        "output_root",
        "tags",
    )


def test_experiment_spec_uses_canonical_uncertainty_key_and_attribute():
    uncertainty = BootstrapConfig(samples=10, seed=7)
    spec = ExperimentSpec(
        state="ghz3",
        basis=PathBasis(Path("basis")),
        backend=AerIdeal(),
        uncertainty=uncertainty,
    )

    payload = spec.to_safe_dict()
    assert spec.uncertainty is uncertainty
    assert spec.bootstrap is uncertainty
    assert payload["uncertainty"] == uncertainty.to_safe_dict()
    assert "bootstrap" not in payload


def test_experiment_spec_deserializes_canonical_and_legacy_uncertainty_keys():
    original = ExperimentSpec(
        state="ghz3",
        basis=PathBasis(Path("basis")),
        backend=AerIdeal(),
        uncertainty=BootstrapConfig(samples=10, seed=7),
    )
    canonical = original.to_safe_dict()
    legacy = dict(canonical)
    legacy["bootstrap"] = legacy.pop("uncertainty")
    both = dict(canonical)
    both["bootstrap"] = both["uncertainty"]

    assert ExperimentSpec.from_safe_dict(canonical) == original
    assert ExperimentSpec.from_safe_dict(legacy) == original
    assert ExperimentSpec.from_safe_dict(both) == original


def test_experiment_spec_rejects_conflicting_uncertainty_aliases():
    with pytest.raises(ExperimentValidationError, match="conflicting"):
        ExperimentSpec(
            state="ghz3",
            basis=PathBasis(Path("basis")),
            backend=AerIdeal(),
            bootstrap=BootstrapConfig(samples=10),
            uncertainty=BootstrapConfig(samples=20),
        )

    payload = ExperimentSpec(
        state="ghz3", basis=PathBasis(Path("basis")), backend=AerIdeal()
    ).to_safe_dict()
    payload["bootstrap"] = BootstrapConfig(samples=10).to_safe_dict()
    payload["uncertainty"] = BootstrapConfig(samples=20).to_safe_dict()
    with pytest.raises(ExperimentValidationError, match="conflicting"):
        ExperimentSpec.from_safe_dict(payload)


def test_custom_and_noisy_backends_require_injection_to_reconstruct():
    custom = CustomBackend(
        instance=object(),
        identity="local",
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
    )
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

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_persisted_float_configs_reject_non_finite_values(value):
    with pytest.raises(ExperimentValidationError):
        MitigationConfig(readout_max_age_hours=value)
    with pytest.raises(ExperimentValidationError):
        RetryConfig(initial_delay=value)
    with pytest.raises(ExperimentValidationError):
        RetryConfig(multiplier=value)
    with pytest.raises(ExperimentValidationError):
        RetryConfig(max_delay=value)
    with pytest.raises(ExperimentValidationError):
        BootstrapConfig(confidence_level=value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IQMHardware("device", use_metrics="false"),
        lambda: CustomBackend(
            object(),
            supports_resume="false",
            execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        ),
        lambda: MitigationConfig(readout=1),
        lambda: MitigationConfig(zne="false"),
        lambda: MitigationConfig(force_recalibration=0),
        lambda: BootstrapConfig(include_readout_calibration="false"),
    ],
)
def test_boolean_config_fields_require_actual_booleans(factory):
    with pytest.raises(ExperimentValidationError):
        factory()


def test_deserialization_does_not_coerce_boolean_strings():
    with pytest.raises(ExperimentValidationError):
        IQMHardware.from_safe_dict({"device": "device", "use_metrics": "false"})
    with pytest.raises(ExperimentValidationError):
        CustomBackend.from_safe_dict(
            {
                "identity": "custom",
                "supports_resume": "false",
                "execution_mode": "ideal_simulator",
            },
            instance=object(),
        )


def test_noisy_simulator_reconstruction_must_match_persisted_mode():
    source_payload = NoisySimulator(source=object()).to_safe_dict()
    with pytest.raises(ExperimentValidationError, match="source_mode"):
        NoisySimulator.from_safe_dict(
            source_payload, noise_model=object(), target_backend=object()
        )

    model_payload = NoisySimulator(noise_model=object(), target_backend=object()).to_safe_dict()
    with pytest.raises(ExperimentValidationError, match="source_mode"):
        NoisySimulator.from_safe_dict(model_payload, source=object())

def test_experiments_package_reexports_experiment_spec():
    from qudits_on_qubits.experiments import ExperimentSpec as ExportedExperimentSpec

    assert ExportedExperimentSpec is ExperimentSpec

def test_models_are_frozen():
    backend = AerIdeal()
    with pytest.raises(FrozenInstanceError):
        backend.seed_simulator = 7


def test_retry_config_validates_bounds():
    with pytest.raises(ExperimentValidationError):
        RetryConfig(max_attempts=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_complex_components_reject_non_finite_real_and_imaginary_parts(value):
    with pytest.raises(ExperimentValidationError):
        ComplexComponents(real=value, imag=0.0)
    with pytest.raises(ExperimentValidationError):
        ComplexComponents(real=0.0, imag=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_interval_rejects_non_finite_bounds(value):
    with pytest.raises(ExperimentValidationError):
        ConfidenceInterval(low=value, high=1.0)
    with pytest.raises(ExperimentValidationError):
        ConfidenceInterval(low=0.0, high=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_bell_estimate_rejects_non_finite_component_standard_error(value):
    with pytest.raises(ExperimentValidationError):
        BellEstimate(
            estimate=ComplexComponents(0.0, 0.0),
            standard_error=ComplexComponents(value, 0.0),
            confidence_interval=ComplexConfidenceInterval(
                real=ConfidenceInterval(0.0, 1.0),
                imag=ConfidenceInterval(0.0, 1.0),
            ),
        )


@pytest.mark.parametrize("standard_error", [(-0.1, 0.0), (0.0, -0.1)])
def test_bell_estimate_rejects_negative_component_standard_error(standard_error):
    with pytest.raises(ExperimentValidationError):
        BellEstimate(
            estimate=ComplexComponents(0.0, 0.0),
            standard_error=ComplexComponents(*standard_error),
            confidence_interval=ComplexConfidenceInterval(
                real=ConfidenceInterval(0.0, 1.0),
                imag=ConfidenceInterval(0.0, 1.0),
            ),
        )


def test_bell_estimate_serializes_component_wise_uncertainty():
    estimate = BellEstimate(
        estimate=ComplexComponents(1.0, -2.0),
        standard_error=ComplexComponents(0.1, 0.2),
        confidence_interval=ComplexConfidenceInterval(
            real=ConfidenceInterval(0.8, 1.2),
            imag=ConfidenceInterval(-2.4, -1.6),
        ),
    )

    assert estimate.to_safe_dict() == {
        "estimate": {"real": 1.0, "imag": -2.0},
        "standard_error": {"real": 0.1, "imag": 0.2},
        "confidence_interval": {
            "real": {"low": 0.8, "high": 1.2},
            "imag": {"low": -2.4, "high": -1.6},
        },
    }


def test_scalar_estimate_serializes_scalar_uncertainty():
    estimate = ScalarEstimate(
        estimate=0.25,
        standard_error=0.05,
        confidence_interval=ConfidenceInterval(0.15, 0.35),
    )

    assert estimate.to_safe_dict() == {
        "estimate": 0.25,
        "standard_error": 0.05,
        "confidence_interval": {"low": 0.15, "high": 0.35},
    }


def test_scalar_estimate_normalizes_real_values_to_float():
    estimate = ScalarEstimate(
        estimate=1,
        standard_error=0,
        confidence_interval=ConfidenceInterval(0.0, 1.0),
    )

    assert type(estimate.estimate) is float
    assert type(estimate.standard_error) is float


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_scalar_estimate_rejects_non_finite_estimate(value):
    with pytest.raises(ExperimentValidationError, match="estimate"):
        ScalarEstimate(
            estimate=value,
            standard_error=0.05,
            confidence_interval=ConfidenceInterval(0.15, 0.35),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_scalar_estimate_rejects_non_finite_standard_error(value):
    with pytest.raises(ExperimentValidationError, match="standard_error"):
        ScalarEstimate(
            estimate=0.25,
            standard_error=value,
            confidence_interval=ConfidenceInterval(0.15, 0.35),
        )


def test_scalar_estimate_rejects_negative_standard_error():
    with pytest.raises(ExperimentValidationError, match="standard_error"):
        ScalarEstimate(
            estimate=0.25,
            standard_error=-0.05,
            confidence_interval=ConfidenceInterval(0.15, 0.35),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("estimate", True),
        ("estimate", "0.25"),
        ("estimate", 0.25 + 0j),
        ("standard_error", False),
        ("standard_error", "0.05"),
        ("standard_error", 0.05 + 0j),
    ],
)
def test_scalar_estimate_rejects_boolean_and_non_real_values(field_name, value):
    kwargs = {
        "estimate": 0.25,
        "standard_error": 0.05,
        "confidence_interval": ConfidenceInterval(0.15, 0.35),
        field_name: value,
    }

    with pytest.raises(ExperimentValidationError, match=field_name):
        ScalarEstimate(**kwargs)


@pytest.mark.parametrize("field_name", ["estimate", "standard_error"])
def test_scalar_estimate_rejects_real_values_too_large_to_normalize(field_name):
    kwargs = {
        "estimate": 0.25,
        "standard_error": 0.05,
        "confidence_interval": ConfidenceInterval(0.15, 0.35),
        field_name: 10**1000,
    }

    with pytest.raises(ExperimentValidationError, match=field_name):
        ScalarEstimate(**kwargs)


def test_scalar_estimate_requires_confidence_interval():
    with pytest.raises(ExperimentValidationError, match="confidence_interval"):
        ScalarEstimate(
            estimate=0.25,
            standard_error=0.05,
            confidence_interval=(0.15, 0.35),
        )


def test_experiments_package_reexports_scalar_estimate():
    from qudits_on_qubits.experiments import ScalarEstimate as ExportedScalarEstimate

    assert ExportedScalarEstimate is ScalarEstimate


def test_complex_confidence_interval_requires_component_intervals():
    with pytest.raises(ExperimentValidationError, match="real"):
        ComplexConfidenceInterval(real=(0.0, 1.0), imag=ConfidenceInterval(0.0, 1.0))
    with pytest.raises(ExperimentValidationError, match="imag"):
        ComplexConfidenceInterval(real=ConfidenceInterval(0.0, 1.0), imag=(0.0, 1.0))


@pytest.mark.parametrize("samples", [1, 0, -1, True, 2.0])
def test_bootstrap_config_requires_at_least_two_integer_samples(samples):
    with pytest.raises(ExperimentValidationError, match="samples"):
        BootstrapConfig(samples=samples)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7", None])
def test_bootstrap_config_rejects_invalid_seed_at_construction(seed):
    with pytest.raises(ExperimentValidationError, match="seed"):
        BootstrapConfig(seed=seed)

def test_status_is_a_string_enum():
    assert BackendStatus.SUBMISSION_UNKNOWN.value == "submission_unknown"
