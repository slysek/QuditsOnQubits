from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .basis import measurement_basis_outcome_map, omega, physical_to_logical_outcome_map
from .graph_settings import build_general_graph_bell_settings
from .qiskit_measurements import append_measurement_for_global_setting

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def build_sampler_circuits_from_graph(
    state_circuit: "QuantumCircuit",
    graph: Any,
    E: np.ndarray,
    observable_from_label: Callable[[object], np.ndarray] | None = None,
    d: int = 3,
    qutrit_qubits: Sequence[Sequence[int]] | None = None,
    root_edge: tuple[int, int] | None = None,
    lam_fn: Callable[..., complex] | None = None,
    weight_attr: str = "weight",
    split_coefficients: str = "uniform",
    drop_conjugate_half: bool = False,
    inplace: bool = False,
    add_measurements: bool = True,
    classical_register_name: str | None = None,
    sort_settings: bool = True,
) -> tuple[list["QuantumCircuit"], dict[str, Any]]:
    """Build one Sampler-ready circuit for each global graph Bell setting.

    Classical bit metadata follows the convention established by
    ``append_measurement_for_global_setting``: qutrits are measured in flattened
    ``qutrit_qubits`` order, so qutrit 0 maps to classical bits ``(0, 1)``,
    qutrit 1 to ``(2, 3)``, and so on for newly added measurement registers.
    Qiskit count strings still display classical bit 0 on the right, so use
    ``bit_order="qiskit"`` in postprocessing unless you manually remap counts.
    If ``observable_from_label`` is omitted, labels such as ``"A0"`` and
    ``"B2"`` use the default qutrit Weyl family ``Z @ X^k``.
    """
    bell_settings_data = build_general_graph_bell_settings(
        n=d,
        graph=graph,
        root_edge=root_edge,
        lam_fn=lam_fn,
        weight_attr=weight_attr,
        split_coefficients=split_coefficients,
        drop_conjugate_half=drop_conjugate_half,
    )

    party_order = tuple(bell_settings_data["party_order"])
    num_qutrits = len(party_order)
    pairs = _normalize_or_default_qutrit_qubits(qutrit_qubits, num_qutrits)
    _validate_inputs(
        state_circuit=state_circuit,
        graph=graph,
        E=E,
        d=d,
        qutrit_qubits=pairs,
        num_qutrits=num_qutrits,
    )

    measurement_settings = [
        _extract_setting(item)
        for item in bell_settings_data["measurement_settings"]
    ]
    if sort_settings:
        measurement_settings = sorted(measurement_settings, key=_setting_sort_key)

    observables_by_label = bell_settings_data.get("observables_by_label", {})
    observable_lookup = observable_from_label or (
        lambda label: _observable_from_bell_settings_or_default(
            label,
            observables_by_label,
            d=d,
        )
    )

    sampler_circuits: list[QuantumCircuit] = []
    circuits_by_setting: dict[tuple[object, ...], int] = {}
    setting_by_circuit_index: list[tuple[object, ...]] = []
    circuit_metadata: list[dict[str, Any]] = []
    qutrit_bit_indices_by_setting: dict[
        tuple[object, ...],
        list[tuple[int, int]],
    ] = {}

    for setting in measurement_settings:
        if setting in circuits_by_setting:
            continue
        qc, circuit_meta = append_measurement_for_global_setting(
            state_circuit=state_circuit,
            global_setting=setting,
            qutrit_qubits=pairs,
            E=E,
            observable_from_label=observable_lookup,
            d=d,
            inplace=inplace,
            add_measurements=add_measurements,
            classical_register_name=classical_register_name,
        )
        index = len(sampler_circuits)
        sampler_circuits.append(qc)
        circuits_by_setting[setting] = index
        setting_by_circuit_index.append(setting)
        circuit_metadata.append(circuit_meta)
        qutrit_bit_indices_by_setting[setting] = list(
            circuit_meta.get("classical_bits_by_qutrit", [])
        )

    registry_backed = (
        "spec_hash" in bell_settings_data
        and "physical_to_logical_outcome_map" in bell_settings_data
    )
    if registry_backed:
        physical_outcome_map = dict(
            bell_settings_data["physical_to_logical_outcome_map"]
        )
        encoding_outcome_map = dict(physical_outcome_map)
    else:
        encoding_outcome_map = physical_to_logical_outcome_map(E, d=d)
        physical_outcome_map = measurement_basis_outcome_map(d=d)

    metadata: dict[str, Any] = {
        "bell_settings_data": bell_settings_data,
        "measurement_settings": setting_by_circuit_index.copy(),
        "circuits_by_setting": circuits_by_setting,
        "setting_by_circuit_index": setting_by_circuit_index,
        "circuit_metadata": circuit_metadata,
        "terms": list(bell_settings_data["terms"]),
        "qutrit_qubits": pairs,
        "qutrit_bit_indices_by_setting": qutrit_bit_indices_by_setting,
        "E": np.asarray(E, dtype=complex),
        "encoding_outcome_map": encoding_outcome_map,
        "physical_to_logical_outcome_map": physical_outcome_map,
        "d": d,
    }
    if registry_backed:
        metadata["spec_hash"] = bell_settings_data["spec_hash"]
    return sampler_circuits, metadata


def decoding_kwargs_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return keyword arguments for encoding-aware count postprocessing."""
    kwargs: dict[str, Any] = {}
    if "physical_to_logical_outcome_map" in metadata:
        kwargs["outcome_map"] = metadata["physical_to_logical_outcome_map"]
    elif "E" in metadata:
        kwargs["E"] = metadata["E"]
    if "d" in metadata:
        kwargs["d"] = metadata["d"]
    elif "bell_settings_data" in metadata:
        kwargs["d"] = metadata["bell_settings_data"].get("local_dimension", 3)
    return kwargs


def build_sampler_circuits_for_candidate(
    candidate: str,
    state_circuit: "QuantumCircuit",
    E: np.ndarray,
    d: int = 3,
    qutrit_qubits: Sequence[Sequence[int]] | None = None,
    drop_conjugate_half: bool = False,
    inplace: bool = False,
    add_measurements: bool = True,
    classical_register_name: str | None = None,
    sort_settings: bool = True,
) -> tuple[list["QuantumCircuit"], dict[str, Any]]:
    """Build Sampler-ready circuits for a supported audited Bell candidate.

    Supported candidates are ``"two_qutrit"``, ``"ghz3"``, and ``"ame43"``.
    The Bell terms and local observables are imported from the existing
    ``qudits_on_qubits.bell_functionals`` implementation, then converted to the counts
    postprocessing convention used by this package.

    This high-level API does not require ``observable_from_label``: the
    measurement observables are selected automatically from the candidate
    definition. ``powers`` in the returned metadata still do not create extra
    circuits; they are used later by ``compute_bell_value_from_counts``.
    """
    bell_settings_data = _candidate_bell_settings_data(
        candidate,
        d=d,
        drop_conjugate_half=drop_conjugate_half,
    )
    party_order = tuple(bell_settings_data["party_order"])
    num_qutrits = len(party_order)
    pairs = _normalize_or_default_qutrit_qubits(qutrit_qubits, num_qutrits)
    _validate_circuit_inputs(
        state_circuit=state_circuit,
        E=E,
        d=d,
        qutrit_qubits=pairs,
        num_qutrits=num_qutrits,
    )

    measurement_settings = [
        _extract_setting(item)
        for item in bell_settings_data["measurement_settings"]
    ]
    if sort_settings:
        measurement_settings = sorted(measurement_settings, key=_setting_sort_key)

    observables_by_label = bell_settings_data["observables_by_label"]
    observable_lookup = lambda label: np.asarray(
        observables_by_label[str(label)],
        dtype=complex,
    )

    sampler_circuits: list[QuantumCircuit] = []
    circuits_by_setting: dict[tuple[object, ...], int] = {}
    setting_by_circuit_index: list[tuple[object, ...]] = []
    circuit_metadata: list[dict[str, Any]] = []
    qutrit_bit_indices_by_setting: dict[
        tuple[object, ...],
        list[tuple[int, int]],
    ] = {}

    for setting in measurement_settings:
        if setting in circuits_by_setting:
            continue
        qc, circuit_meta = append_measurement_for_global_setting(
            state_circuit=state_circuit,
            global_setting=setting,
            qutrit_qubits=pairs,
            E=E,
            observable_from_label=observable_lookup,
            d=d,
            inplace=inplace,
            add_measurements=add_measurements,
            classical_register_name=classical_register_name,
        )
        index = len(sampler_circuits)
        sampler_circuits.append(qc)
        circuits_by_setting[setting] = index
        setting_by_circuit_index.append(setting)
        circuit_metadata.append(circuit_meta)
        qutrit_bit_indices_by_setting[setting] = list(
            circuit_meta.get("classical_bits_by_qutrit", [])
        )

    metadata: dict[str, Any] = {
        "candidate": bell_settings_data["candidate"],
        "spec_hash": bell_settings_data["spec_hash"],
        "bell_settings_data": bell_settings_data,
        "measurement_settings": setting_by_circuit_index.copy(),
        "circuits_by_setting": circuits_by_setting,
        "setting_by_circuit_index": setting_by_circuit_index,
        "circuit_metadata": circuit_metadata,
        "terms": list(bell_settings_data["terms"]),
        "qutrit_qubits": pairs,
        "qutrit_bit_indices_by_setting": qutrit_bit_indices_by_setting,
        "E": np.asarray(E, dtype=complex),
        "encoding_outcome_map": dict(
            bell_settings_data["physical_to_logical_outcome_map"]
        ),
        "physical_to_logical_outcome_map": dict(
            bell_settings_data["physical_to_logical_outcome_map"]
        ),
        "d": d,
    }
    return sampler_circuits, metadata


def default_observable_from_label(label: object, d: int = 3) -> np.ndarray:
    """Return the default qutrit observable for a setting label.

    The label family letter is used only as a party/family name. The trailing
    integer selects ``Z @ X^k`` with ``k`` reduced modulo ``d``.
    """
    index = _label_index(label)
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    Z = np.diag([omega(d) ** j for j in range(d)]).astype(complex)
    return Z @ np.linalg.matrix_power(X, index % d)


def _observable_from_bell_settings_or_default(
    label: object,
    observables_by_label: Mapping[str, np.ndarray],
    d: int,
) -> np.ndarray:
    key = str(label)
    if key in observables_by_label:
        return np.asarray(observables_by_label[key], dtype=complex)
    return default_observable_from_label(label, d=d)


def counts_by_setting_from_sampler_result(
    result: Any,
    metadata: Mapping[str, Any],
) -> dict[tuple[object, ...], Mapping[str, int]]:
    """Map a Qiskit Sampler/backend result to counts keyed by setting tuple.

    The Qiskit result APIs differ across versions. This helper supports common
    backend ``get_counts(i)`` results and newer primitive result entries with
    ``result[i].data.meas.get_counts()``. If neither format is present, map
    counts manually using ``metadata["setting_by_circuit_index"]``.
    """
    counts_by_setting: dict[tuple[object, ...], Mapping[str, int]] = {}
    setting_by_index = list(metadata["setting_by_circuit_index"])
    for index, setting in enumerate(setting_by_index):
        counts = _counts_for_index(result, index)
        counts_by_setting[tuple(setting)] = counts
    return counts_by_setting


def run_sampler_circuits_to_counts_by_setting(
    sampler_circuits: Sequence["QuantumCircuit"],
    metadata: Mapping[str, Any],
    shots: int = 1024,
    sampler: Any | None = None,
    backend: Any | None = None,
    transpile_circuits: bool = False,
    optimization_level: int = 3,
    run_options: Mapping[str, Any] | None = None,
) -> tuple[dict[tuple[object, ...], Mapping[str, int]], dict[str, Any]]:
    """Run Bell measurement circuits and return Bell-postprocessing counts.

    The first return value is ready for ``compute_bell_value_from_counts`` as
    ``counts_by_setting``. The second return value keeps execution objects for
    inspection: ``sampler``, ``backend``, ``job``, ``result``, ``circuits``,
    and ``shots``.

    Circuits are submitted as-is by default. Pass an explicit ``sampler`` for a
    SamplerV2-style object, or pass an explicit ``backend`` to run through
    ``backend.run(circuits, shots=shots, **run_options)``. This keeps IQM runs
    out of the local Qiskit transpiler path; if IQM compilation is needed, pass
    IQM/backend options through ``run_options`` or prepare the circuits before
    calling this helper.
    """
    circuits = list(sampler_circuits)
    if len(circuits) != len(metadata["setting_by_circuit_index"]):
        raise ValueError("number of sampler_circuits must match metadata settings")
    options = _run_options_without_shots(run_options)

    resolved_backend = backend
    if transpile_circuits:
        resolved_backend = resolved_backend or _make_default_aer_backend()
        executed_circuits = _transpile_circuits(
            circuits,
            resolved_backend,
            optimization_level=optimization_level,
        )
    else:
        executed_circuits = circuits

    resolved_sampler = sampler
    execution_target = "sampler"
    if resolved_sampler is not None:
        job = resolved_sampler.run(executed_circuits, shots=shots, **options)
    elif resolved_backend is not None:
        execution_target = "backend"
        job = resolved_backend.run(executed_circuits, shots=shots, **options)
    else:
        resolved_sampler = _make_default_sampler_v2(shots)
        job = resolved_sampler.run(executed_circuits, shots=shots, **options)

    result = job.result()
    counts_by_setting = counts_by_setting_from_sampler_result(result, metadata)
    return counts_by_setting, {
        "sampler": resolved_sampler,
        "backend": resolved_backend,
        "job": job,
        "result": result,
        "circuits": executed_circuits,
        "transpiled_circuits": executed_circuits,
        "transpile_circuits": transpile_circuits,
        "execution_target": execution_target,
        "run_options": options,
        "shots": shots,
    }


def run_iqm_sampler_circuits_to_counts_by_setting(
    sampler_circuits: Sequence["QuantumCircuit"],
    metadata: Mapping[str, Any],
    shots: int = 1024,
    backend: Any | None = None,
    quantum_computer: str | None = None,
    use_metrics: bool = False,
    env_path: str | Path | None = None,
    run_options: Mapping[str, Any] | None = None,
) -> tuple[dict[tuple[object, ...], Mapping[str, int]], dict[str, Any]]:
    """Run Bell measurement circuits on an IQM backend without local transpilation.

    Provide ``backend`` if it has already been loaded. Otherwise pass
    ``quantum_computer`` and this helper will use the project's IQM environment
    loader.
    """
    resolved_backend = backend
    if resolved_backend is None:
        if quantum_computer is None:
            raise ValueError("pass either backend or quantum_computer for an IQM run")
        from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
            load_iqm_backend,
        )

        resolved_backend = load_iqm_backend(
            quantum_computer,
            use_metrics=use_metrics,
            env_path=env_path,
        )

    return run_sampler_circuits_to_counts_by_setting(
        sampler_circuits,
        metadata,
        shots=shots,
        backend=resolved_backend,
        transpile_circuits=False,
        run_options=run_options,
    )


def _counts_for_index(result: Any, index: int) -> Mapping[str, int]:
    if hasattr(result, "get_counts"):
        try:
            return result.get_counts(index)
        except TypeError:
            if index == 0:
                return result.get_counts()

    try:
        entry = result[index]
    except Exception as exc:
        raise NotImplementedError(_UNKNOWN_RESULT_FORMAT) from exc

    data = getattr(entry, "data", None)
    meas = getattr(data, "meas", None)
    if meas is not None and hasattr(meas, "get_counts"):
        return meas.get_counts()
    counts = _counts_from_named_data_bin(data)
    if counts is not None:
        return counts

    raise NotImplementedError(_UNKNOWN_RESULT_FORMAT)


_UNKNOWN_RESULT_FORMAT = (
    "Unsupported Qiskit result format. Build counts_by_setting manually by "
    "zipping counts with metadata['setting_by_circuit_index']."
)


def _make_default_aer_backend() -> Any:
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise ImportError(
            "run_sampler_circuits_to_counts_by_setting requires qiskit-aer "
            "or an explicit backend when transpile_circuits=True"
        ) from exc
    return AerSimulator()


def _make_default_sampler_v2(shots: int) -> Any:
    try:
        from qiskit_aer.primitives import SamplerV2
    except Exception as exc:
        raise ImportError(
            "run_sampler_circuits_to_counts_by_setting requires "
            "qiskit_aer.primitives.SamplerV2 or an explicit sampler"
        ) from exc
    return SamplerV2(default_shots=shots)


def _transpile_circuits(
    circuits: Sequence["QuantumCircuit"],
    backend: Any,
    optimization_level: int,
) -> list["QuantumCircuit"]:
    try:
        from qiskit import transpile
    except Exception as exc:
        raise ImportError("qiskit.transpile is required to transpile circuits") from exc
    return [
        transpile(circuit, backend, optimization_level=optimization_level)
        for circuit in circuits
    ]


def _run_options_without_shots(
    run_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options = dict(run_options or {})
    if "shots" in options:
        raise ValueError("pass shots via the shots argument, not run_options")
    return options


def _counts_from_named_data_bin(data: Any) -> Mapping[str, int] | None:
    if data is None or not hasattr(data, "keys"):
        return None
    for name in data.keys():
        candidate = getattr(data, str(name), None)
        if candidate is None and hasattr(data, "__getitem__"):
            try:
                candidate = data[name]
            except Exception:
                candidate = None
        if candidate is not None and hasattr(candidate, "get_counts"):
            return candidate.get_counts()
    return None


def _candidate_bell_settings_data(
    candidate: str,
    d: int,
    drop_conjugate_half: bool,
) -> dict[str, Any]:
    if d != 3:
        raise ValueError("audited candidate Bell settings are implemented only for d=3")

    spec = get_reference_experiment(candidate)
    observables_by_label = {
        observable.label: observable.as_array()
        for observable in spec.observables
    }
    converted_terms: list[dict[str, object]] = []
    measurement_settings: list[tuple[str | None, ...]] = []
    seen_settings: set[tuple[str | None, ...]] = set()

    for term_index, term in enumerate(spec.bell_functional.terms):
        if not term.factors:
            continue
        graph_power = int(term.factors[0].outcome_power) % d
        if drop_conjugate_half and graph_power != 1:
            continue

        setting_tuple = spec.setting_for_term(term)
        if setting_tuple not in seen_settings:
            seen_settings.add(setting_tuple)
            measurement_settings.append(setting_tuple)

        converted_terms.append(
            {
                "coeff": term.sampling_coefficient(),
                "settings": setting_tuple,
                "powers": spec.powers_for_term(term),
                "source": f"{spec.experiment_id}:{term_index}",
                "graph_power": graph_power,
            }
        )

    return {
        "candidate": spec.experiment_id,
        "spec_hash": spec.stable_hash(),
        "party_order": spec.state.party_order,
        "measurement_settings": measurement_settings,
        "terms": converted_terms,
        "observables_by_label": observables_by_label,
        "physical_to_logical_outcome_map": dict(
            spec.outcome_convention.measurement_basis_index_map
        ),
    }


def _extract_setting(item: object) -> tuple[object, ...]:
    if isinstance(item, Mapping):
        return tuple(item["setting"])
    return tuple(item)  # type: ignore[arg-type]


def _setting_sort_key(setting: Sequence[object]) -> tuple[str, ...]:
    return tuple("" if item is None else str(item) for item in setting)


def _label_index(label: object) -> int:
    text = str(label)
    digits = []
    for character in reversed(text):
        if not character.isdigit():
            break
        digits.append(character)
    if not digits:
        raise ValueError(f"setting label {label!r} does not end with an integer")
    return int("".join(reversed(digits)))


def _normalize_or_default_qutrit_qubits(
    qutrit_qubits: Sequence[Sequence[int]] | None,
    num_qutrits: int,
) -> tuple[tuple[int, int], ...]:
    if qutrit_qubits is None:
        return tuple((2 * index, 2 * index + 1) for index in range(num_qutrits))

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for pair in qutrit_qubits:
        if len(pair) != 2:
            raise ValueError("each qutrit_qubits item must contain exactly two qubits")
        q0, q1 = int(pair[0]), int(pair[1])
        if q0 == q1:
            raise ValueError("a qutrit cannot use the same qubit twice")
        if q0 in used or q1 in used:
            raise ValueError("qutrit_qubits pairs must be disjoint")
        used.update((q0, q1))
        pairs.append((q0, q1))
    return tuple(pairs)


def _validate_inputs(
    state_circuit: "QuantumCircuit",
    graph: Any,
    E: np.ndarray,
    d: int,
    qutrit_qubits: Sequence[tuple[int, int]],
    num_qutrits: int,
) -> None:
    _validate_circuit_inputs(
        state_circuit=state_circuit,
        E=E,
        d=d,
        qutrit_qubits=qutrit_qubits,
        num_qutrits=num_qutrits,
    )
    if int(graph.vcount()) != num_qutrits:
        raise ValueError("graph vertex count must match the number of qutrits")


def _validate_circuit_inputs(
    state_circuit: "QuantumCircuit",
    E: np.ndarray,
    d: int,
    qutrit_qubits: Sequence[tuple[int, int]],
    num_qutrits: int,
) -> None:
    matrix = np.asarray(E, dtype=complex)
    if matrix.shape != (4, d):
        raise ValueError(f"E must have shape (4, {d})")
    if len(qutrit_qubits) != num_qutrits:
        raise ValueError("len(qutrit_qubits) must match the number of qutrits")
    max_qubit = max((qubit for pair in qutrit_qubits for qubit in pair), default=-1)
    if max_qubit >= state_circuit.num_qubits:
        raise ValueError("state_circuit does not contain all qutrit_qubits")
