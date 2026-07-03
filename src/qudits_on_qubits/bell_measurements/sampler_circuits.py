from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

from .basis import omega, ordered_qutrit_eigenbasis, measurement_basis_outcome_map, physical_to_logical_outcome_map
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
        "encoding_outcome_map": physical_to_logical_outcome_map(E, d=d),
        "physical_to_logical_outcome_map": measurement_basis_outcome_map(d=d),
        "d": d,
    }
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
    ``bell_functionals_qutrit`` implementation, then converted to the counts
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
        "candidate": candidate,
        "bell_settings_data": bell_settings_data,
        "measurement_settings": setting_by_circuit_index.copy(),
        "circuits_by_setting": circuits_by_setting,
        "setting_by_circuit_index": setting_by_circuit_index,
        "circuit_metadata": circuit_metadata,
        "terms": list(bell_settings_data["terms"]),
        "qutrit_qubits": pairs,
        "qutrit_bit_indices_by_setting": qutrit_bit_indices_by_setting,
        "E": np.asarray(E, dtype=complex),
        "encoding_outcome_map": physical_to_logical_outcome_map(E, d=d),
        "physical_to_logical_outcome_map": measurement_basis_outcome_map(d=d),
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
    transpile_circuits: bool = True,
    optimization_level: int = 3,
) -> tuple[dict[tuple[object, ...], Mapping[str, int]], dict[str, Any]]:
    """Run SamplerV2-style circuits and return Bell-postprocessing counts.

    The first return value is ready for ``compute_bell_value_from_counts`` as
    ``counts_by_setting``. The second return value keeps execution objects for
    inspection: ``sampler``, ``backend``, ``job``, ``result``,
    ``transpiled_circuits``, and ``shots``.
    """
    circuits = list(sampler_circuits)
    if len(circuits) != len(metadata["setting_by_circuit_index"]):
        raise ValueError("number of sampler_circuits must match metadata settings")

    resolved_backend = backend
    if transpile_circuits:
        resolved_backend = resolved_backend or _make_default_aer_backend()
        transpiled = _transpile_circuits(
            circuits,
            resolved_backend,
            optimization_level=optimization_level,
        )
    else:
        transpiled = circuits

    resolved_sampler = sampler or _make_default_sampler_v2(shots)
    job = resolved_sampler.run(transpiled, shots=shots)
    result = job.result()
    counts_by_setting = counts_by_setting_from_sampler_result(result, metadata)
    return counts_by_setting, {
        "sampler": resolved_sampler,
        "backend": resolved_backend,
        "job": job,
        "result": result,
        "transpiled_circuits": transpiled,
        "shots": shots,
    }


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

    try:
        from bell_functionals_qutrit.bell_builders import (
            bell_terms,
            num_qutrits_for_candidate,
        )
    except Exception as exc:
        raise ImportError(
            "build_sampler_circuits_for_candidate requires the local "
            "bell_functionals_qutrit package"
        ) from exc

    num_qutrits = num_qutrits_for_candidate(candidate)
    raw_terms = list(bell_terms(candidate))
    if drop_conjugate_half:
        raw_terms = [
            term
            for term in raw_terms
            if _primary_term_power_mod_d(term, d=d) == 1
        ]

    observables_by_label = _candidate_measurement_observables(candidate)
    converted_terms: list[dict[str, object]] = []
    measurement_settings: list[tuple[str | None, ...]] = []
    seen_settings: set[tuple[str | None, ...]] = set()

    for term_index, raw_term in enumerate(raw_terms):
        settings: list[str | None] = [None] * num_qutrits
        powers: list[int] = [0] * num_qutrits
        coeff = complex(raw_term.coefficient)

        for factor in raw_term.factors:
            label = str(factor.label)
            if label not in observables_by_label:
                raise ValueError(
                    f"candidate {candidate!r} has no measurement observable "
                    f"for label {label!r}"
                )
            if settings[factor.party] not in (None, label):
                raise ValueError("a Bell term assigns two settings to one party")
            settings[factor.party] = label
            powers[factor.party] = int(factor.power)
            coeff *= _root_expectation_scale_for_sampler(
                measurement_observable=observables_by_label[label],
                desired_operator=np.asarray(factor.matrix, dtype=complex),
                power=int(factor.power),
                d=d,
            )

        setting_tuple = tuple(settings)
        if setting_tuple not in seen_settings:
            seen_settings.add(setting_tuple)
            measurement_settings.append(setting_tuple)

        converted_terms.append(
            {
                "coeff": complex(coeff),
                "settings": setting_tuple,
                "powers": tuple(powers),
                "source": f"{candidate}:{term_index}",
                "graph_power": _primary_term_power_mod_d(raw_term, d=d),
            }
        )

    return {
        "candidate": candidate,
        "party_order": tuple(range(num_qutrits)),
        "measurement_settings": measurement_settings,
        "terms": converted_terms,
        "observables_by_label": observables_by_label,
    }


def _candidate_measurement_observables(candidate: str) -> dict[str, np.ndarray]:
    from bell_functionals_qutrit.operators import (
        make_XZ_qutrit,
        make_measurement_observables_qutrit_d3,
    )

    x, z, _ = make_XZ_qutrit()
    observables: dict[str, np.ndarray] = {}
    for index, observable in enumerate(make_measurement_observables_qutrit_d3(1)):
        observables[f"A{index}"] = np.asarray(observable, dtype=complex)
    for index in range(3):
        observables[f"B{index}"] = z @ np.linalg.matrix_power(x, index)

    if candidate == "ghz3":
        observables["C0"] = z
        observables["C1"] = z @ x
    elif candidate == "ame43":
        observables["C0"] = z
        observables["C1"] = x
        observables["D0"] = z
        observables["D1"] = z @ x
    elif candidate != "two_qutrit":
        raise ValueError(f"unknown candidate: {candidate!r}")
    return observables


def _root_expectation_scale_for_sampler(
    measurement_observable: np.ndarray,
    desired_operator: np.ndarray,
    power: int,
    d: int,
    tol: float = 1e-7,
) -> complex:
    V, _ = ordered_qutrit_eigenbasis(
        measurement_observable,
        d=d,
        tol=tol,
        allow_global_phase=True,
    )
    diagonalized = V.conj().T @ np.asarray(desired_operator, dtype=complex) @ V
    off_diagonal = diagonalized - np.diag(np.diag(diagonalized))
    if not np.allclose(off_diagonal, 0.0, atol=tol):
        raise ValueError(
            "desired Bell operator is not diagonal in the selected measurement basis"
        )

    roots = np.array(
        [omega(d) ** ((int(power) * outcome) % d) for outcome in range(d)],
        dtype=complex,
    )
    scales = np.diag(diagonalized) / roots
    if not np.allclose(scales, scales[0], atol=tol):
        raise ValueError(
            "desired Bell operator cannot be represented by one sampled power "
            "and a global scale"
        )
    return complex(scales[0])


def _primary_term_power_mod_d(term: Any, d: int) -> int:
    if not term.factors:
        return 0
    return int(term.factors[0].power) % d


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
