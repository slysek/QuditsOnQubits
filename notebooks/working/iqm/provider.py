from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm import IQMBackend
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile
from iqm.qiskit_iqm import IQMProvider
from iqm.station_control.interface.models import StaticQuantumArchitecture
import pandas as pd
import random
from statistics import mean

class BackendDescription(str):
    def _repr_markdown_(self):
        return self.replace("\n", "  \n")

def _get_backend_description(computer_name):
    descriptions = {
        "emerald": "Name: Emerald\nNumber of qubits: 54",
        "garnet": "Name: Garnet\nNumber of qubits: 20",
        "sirius": "Name: Sirius\nNumber of qubits: 16",
    }

    return BackendDescription(
        descriptions.get(
            computer_name,
            "Unknown quantum computer, please check the computer name",
        )
    )

def get_backend(quantum_computer="emerald", token="REMOVED_SECRET", fake=False):
    try:
        if not fake:
            client = IQMClient(
                        "https://resonance.iqm.tech/",
                        quantum_computer=quantum_computer,
                        token=token,
                    )

            backend = IQMBackend(client, name=quantum_computer, use_metrics=True)

            # provider = IQMProvider(
            #                 "https://resonance.iqm.tech/",
            #                 quantum_computer=quantum_computer,
            #                 token=token, 
            #             )

            # backend = provider.get_backend(use_metrics=True)

            print(f"Backend connected successfully, using {quantum_computer}")
            return backend, client
        else:
            backend = IQMFakeAphrodite()
            print(f"Backend connection failed, using fake backend, using {backend.name}")
            return backend
    except Exception as e:
        backend = IQMFakeAphrodite()
        print(f"Backend connection failed, using fake backend, using {backend.name}")
        return backend


def _default_gate_implementation(backend, gate_name, locus):
    gate = backend.architecture.gates[gate_name]
    return gate.get_default_implementation(tuple(locus))


def _measure_fidelity(metrics, impl_name, qubit):
    errors = metrics.get_measure_errors("measure", impl_name, (qubit,))
    if errors is None:
        return None

    error_0_to_1, error_1_to_0 = errors
    return 1 - ((error_0_to_1 + error_1_to_0) / 2)


def _backend_name(backend):
    name = getattr(backend, "name", None)
    return name() if callable(name) else name


def _unwrap_backend(backend):
    if isinstance(backend, tuple):
        if not backend:
            raise ValueError("Backend tuple is empty.")
        return backend[0]
    return backend


def _gate_loci(backend, gate_name):
    gate = backend.architecture.gates[gate_name]
    loci = getattr(gate, "loci", None)
    if loci is not None:
        return [tuple(locus) for locus in loci]

    implementation_name = _default_gate_implementation(
        backend, gate_name, next(iter(gate.implementations.values())).loci[0]
    )
    return [tuple(locus) for locus in gate.implementations[implementation_name].loci]


def _architecture_gate_loci(architecture, gate_name):
    if gate_name not in architecture.gates:
        return []

    gate = architecture.gates[gate_name]
    loci = getattr(gate, "loci", None)
    if loci is not None:
        return [tuple(locus) for locus in loci]

    all_loci = []
    for implementation in gate.implementations.values():
        all_loci.extend(tuple(locus) for locus in implementation.loci)
    return all_loci


def to_static_architecture(architecture, dut_label=None):
    """Return a StaticQuantumArchitecture for IQMFakeBackend construction.

    IQM live backends expose a DynamicQuantumArchitecture, while IQMFakeBackend
    expects a StaticQuantumArchitecture with a ``connectivity`` field.
    """
    if hasattr(architecture, "connectivity"):
        return architecture

    return StaticQuantumArchitecture(
        dut_label=dut_label,
        qubits=list(architecture.qubits),
        computational_resonators=list(architecture.computational_resonators),
        connectivity=_architecture_gate_loci(architecture, "cz"),
    )


def _filtered_qubits(backend, qubits):
    available_qubits = list(backend.architecture.qubits)
    if qubits is None:
        return available_qubits

    missing_qubits = [qubit for qubit in qubits if qubit not in available_qubits]
    if missing_qubits:
        raise ValueError(f"Unknown qubits for this backend: {missing_qubits}")
    return list(qubits)


def _ns_or_none(value_seconds):
    return None if value_seconds is None else value_seconds * 1e9


def _error_from_fidelity(fidelity):
    return None if fidelity is None else round(1 - fidelity, 12)


def _filter_error_profile(profile, qubits, name=None):
    selected = set(qubits)

    return IQMErrorProfile(
        t1s={qubit: profile.t1s[qubit] for qubit in qubits if qubit in profile.t1s},
        t2s={qubit: profile.t2s[qubit] for qubit in qubits if qubit in profile.t2s},
        single_qubit_gate_depolarizing_error_parameters={
            gate: {
                qubit: error
                for qubit, error in gate_errors.items()
                if qubit in selected
            }
            for gate, gate_errors in profile.single_qubit_gate_depolarizing_error_parameters.items()
        },
        two_qubit_gate_depolarizing_error_parameters={
            gate: {
                tuple(locus): error
                for locus, error in gate_errors.items()
                if set(locus).issubset(selected)
            }
            for gate, gate_errors in profile.two_qubit_gate_depolarizing_error_parameters.items()
        },
        single_qubit_gate_durations=dict(profile.single_qubit_gate_durations),
        two_qubit_gate_durations=dict(profile.two_qubit_gate_durations),
        readout_errors={
            qubit: dict(profile.readout_errors[qubit])
            for qubit in qubits
            if qubit in profile.readout_errors
        },
        name=name or profile.name,
    )


def _print_error_profile_report(profile):
    print(f"\nNoise profile: {profile.name or 'unnamed'}")

    qubit_rows = []
    for qubit in profile.readout_errors:
        readout = profile.readout_errors[qubit]
        readout_0 = readout.get("0")
        readout_1 = readout.get("1")
        qubit_rows.append(
            {
                "qubit": qubit,
                "T1 [us]": None if qubit not in profile.t1s else profile.t1s[qubit] / 1e3,
                "T2 [us]": None if qubit not in profile.t2s else profile.t2s[qubit] / 1e3,
                "readout 0->1": readout_0,
                "readout 1->0": readout_1,
                "readout avg error": None if None in (readout_0, readout_1) else (readout_0 + readout_1) / 2,
                "readout asymmetry": None if None in (readout_0, readout_1) else readout_1 - readout_0,
            }
        )

    print("\nT1 / T2 / readout")
    print(pd.DataFrame(qubit_rows).to_string(index=False))

    one_qubit_rows = []
    for gate, gate_errors in profile.single_qubit_gate_depolarizing_error_parameters.items():
        duration = profile.single_qubit_gate_durations.get(gate)
        for qubit, error in gate_errors.items():
            one_qubit_rows.append(
                {
                    "gate": gate,
                    "qubit": qubit,
                    "depol error": error,
                    "duration [ns]": duration,
                }
            )

    print("\n1Q gate errors and durations")
    print(pd.DataFrame(one_qubit_rows).to_string(index=False))

    two_qubit_rows = []
    for gate, gate_errors in profile.two_qubit_gate_depolarizing_error_parameters.items():
        duration = profile.two_qubit_gate_durations.get(gate)
        for locus, error in gate_errors.items():
            two_qubit_rows.append(
                {
                    "gate": gate,
                    "connection": " - ".join(locus),
                    "depol error": error,
                    "duration [ns]": duration,
                }
            )

    print("\n2Q / CZ gate errors and durations")
    print(pd.DataFrame(two_qubit_rows).to_string(index=False))


def get_backend_error_profile(backend, qubits=None, name=None, print_report=True):
    """Return an IQMErrorProfile extracted from backend calibration data.

    Args:
        backend: IQM backend created with metrics, or an IQM fake backend with
            an ``error_profile`` property.
        qubits: Optional list of physical qubit names. If provided, only these
            qubits and two-qubit gates fully inside the subset are included.
        name: Optional profile name.
        print_report: Print a readable report with T1/T2, gate errors,
            readout errors, gate durations, and readout asymmetry.

    Returns:
        IQMErrorProfile compatible with IQM fake backends.
    """
    backend = _unwrap_backend(backend)
    selected_qubits = _filtered_qubits(backend, qubits)

    if getattr(backend, "metrics", None) is None:
        if hasattr(backend, "error_profile"):
            profile = _filter_error_profile(
                backend.error_profile,
                selected_qubits,
                name=name or _backend_name(backend),
            )
            if print_report:
                _print_error_profile_report(profile)
            return profile
        raise ValueError(
            "Backend has no metrics and no error_profile. Use a backend created "
            "with use_metrics=True, or pass an IQM fake backend."
        )

    metrics = backend.metrics
    t1_times, t2_times = metrics.get_coherence_times(selected_qubits)

    t1s = {qubit: t1_times[qubit] * 1e9 for qubit in selected_qubits if qubit in t1_times}
    t2s = {qubit: t2_times[qubit] * 1e9 for qubit in selected_qubits if qubit in t2_times}
    selected = set(selected_qubits)

    single_qubit_errors = {}
    single_qubit_durations = {}
    for gate_name in ("prx", "cc_prx"):
        if gate_name not in backend.architecture.gates:
            continue

        gate_errors = {}
        durations = []
        for locus in _gate_loci(backend, gate_name):
            if len(locus) != 1 or locus[0] not in selected:
                continue

            implementation = _default_gate_implementation(backend, gate_name, locus)
            error = _error_from_fidelity(metrics.get_gate_fidelity(gate_name, implementation, locus))
            duration = _ns_or_none(metrics.get_gate_duration(gate_name, implementation, locus))
            if error is not None:
                gate_errors[locus[0]] = error
            if duration is not None:
                durations.append(duration)

        if gate_errors:
            single_qubit_errors[gate_name] = gate_errors
        if durations:
            single_qubit_durations[gate_name] = round(mean(durations), 12)

    two_qubit_errors = {}
    two_qubit_durations = {}
    for gate_name in ("cz", "move"):
        if gate_name not in backend.architecture.gates:
            continue

        gate_errors = {}
        durations = []
        for locus in _gate_loci(backend, gate_name):
            if len(locus) != 2 or not set(locus).issubset(selected):
                continue

            implementation = _default_gate_implementation(backend, gate_name, locus)
            error = _error_from_fidelity(metrics.get_gate_fidelity(gate_name, implementation, locus))
            duration = _ns_or_none(metrics.get_gate_duration(gate_name, implementation, locus))
            if error is not None:
                gate_errors[tuple(locus)] = error
            if duration is not None:
                durations.append(duration)

        if gate_errors:
            two_qubit_errors[gate_name] = gate_errors
        if durations:
            two_qubit_durations[gate_name] = round(mean(durations), 12)

    readout_errors = {}
    for qubit in selected_qubits:
        implementation = _default_gate_implementation(backend, "measure", (qubit,))
        errors = metrics.get_measure_errors("measure", implementation, (qubit,))
        if errors is None:
            continue

        error_0_to_1, error_1_to_0 = errors
        readout_errors[qubit] = {"0": error_0_to_1, "1": error_1_to_0}

    profile = IQMErrorProfile(
        t1s=t1s,
        t2s=t2s,
        single_qubit_gate_depolarizing_error_parameters=single_qubit_errors,
        two_qubit_gate_depolarizing_error_parameters=two_qubit_errors,
        single_qubit_gate_durations=single_qubit_durations,
        two_qubit_gate_durations=two_qubit_durations,
        readout_errors=readout_errors,
        name=name or _backend_name(backend),
    )

    if print_report:
        _print_error_profile_report(profile)
    return profile


def _architecture_qubits(architecture, qubits=None):
    architecture = to_static_architecture(architecture)
    available_qubits = list(architecture.qubits)
    if qubits is None:
        return available_qubits

    missing_qubits = [qubit for qubit in qubits if qubit not in available_qubits]
    if missing_qubits:
        raise ValueError(f"Unknown qubits for this architecture: {missing_qubits}")
    return list(qubits)


def _random_connection_loci(backend, qubits, connectivity, architecture=None):
    selected = set(qubits)
    if connectivity is not None:
        return [tuple(locus) for locus in connectivity if set(locus).issubset(selected)]

    if architecture is not None:
        architecture = to_static_architecture(architecture)
        return [tuple(locus) for locus in architecture.connectivity if set(locus).issubset(selected)]

    if backend is None:
        return []

    architecture = backend.architecture
    if hasattr(architecture, "connectivity"):
        return [tuple(locus) for locus in architecture.connectivity if set(locus).issubset(selected)]

    if "cz" in architecture.gates:
        return [tuple(locus) for locus in _gate_loci(backend, "cz") if set(locus).issubset(selected)]

    return []


def generate_random_error_profile(
    backend=None,
    architecture=None,
    qubits=None,
    connectivity=None,
    name="random-noise-profile",
    seed=None,
    t1_range_us=(20.0, 80.0),
    t2_range_us=(10.0, 60.0),
    t2_max_t1_ratio=0.9,
    single_qubit_error_range=(0.0001, 0.003),
    two_qubit_error_range=(0.003, 0.04),
    readout_error_range=(0.01, 0.06),
    readout_asymmetry_range=(-0.01, 0.01),
    single_qubit_gate_duration_ns=40.0,
    two_qubit_gate_duration_ns=80.0,
    print_report=True,
):
    """Generate a random IQMErrorProfile for an IQM fake backend.

    If ``architecture`` is provided, the profile matches that static
    architecture. Otherwise, if ``backend`` is provided, the function uses its
    qubit names and CZ connectivity. You can override the qubit subset with
    ``qubits``.
    """
    backend = None if backend is None else _unwrap_backend(backend)
    if backend is None and architecture is None and qubits is None:
        raise ValueError("Pass backend, architecture, or qubits to define the profile shape.")

    if architecture is not None:
        selected_qubits = _architecture_qubits(architecture, qubits)
    else:
        selected_qubits = list(qubits) if backend is None else _filtered_qubits(backend, qubits)

    connection_loci = _random_connection_loci(
        backend,
        selected_qubits,
        connectivity,
        architecture=architecture,
    )
    rng = random.Random(seed)

    t1s = {}
    t2s = {}
    for qubit in selected_qubits:
        t1_ns = rng.uniform(*t1_range_us) * 1e3
        t2_ns = min(rng.uniform(*t2_range_us) * 1e3, t2_max_t1_ratio * t1_ns)
        t1s[qubit] = round(t1_ns, 12)
        t2s[qubit] = round(t2_ns, 12)

    prx_errors = {
        qubit: round(rng.uniform(*single_qubit_error_range), 12)
        for qubit in selected_qubits
    }
    cz_errors = {
        tuple(locus): round(rng.uniform(*two_qubit_error_range), 12)
        for locus in connection_loci
    }

    readout_errors = {}
    for qubit in selected_qubits:
        avg_error = rng.uniform(*readout_error_range)
        asymmetry = rng.uniform(*readout_asymmetry_range)
        error_0 = min(max(avg_error - asymmetry / 2, 0.0), 1.0)
        error_1 = min(max(avg_error + asymmetry / 2, 0.0), 1.0)
        readout_errors[qubit] = {"0": round(error_0, 12), "1": round(error_1, 12)}

    profile = IQMErrorProfile(
        t1s=t1s,
        t2s=t2s,
        single_qubit_gate_depolarizing_error_parameters={"prx": prx_errors},
        two_qubit_gate_depolarizing_error_parameters={"cz": cz_errors},
        single_qubit_gate_durations={"prx": float(single_qubit_gate_duration_ns)},
        two_qubit_gate_durations={"cz": float(two_qubit_gate_duration_ns)},
        readout_errors=readout_errors,
        name=name,
    )

    if print_report:
        _print_error_profile_report(profile)
    return profile


def get_best_qubits_and_connections(backend, top_n=10):
    """Return ranked qubits and CZ connections using IQM calibration metrics.

    Qubits are ranked by measurement fidelity. Connections are ranked by CZ fidelity.
    The function returns two pandas DataFrames, which display nicely in notebooks.
    """
    if backend.metrics is None:
        raise ValueError(
            "Backend has no metrics. Create it with IQMBackend(..., use_metrics=True) "
            "or provider.get_backend(use_metrics=True)."
        )

    metrics = backend.metrics
    t1_times, t2_times = metrics.get_coherence_times(backend.architecture.qubits)

    qubit_rows = []
    for qubit in backend.architecture.qubits:
        impl_name = _default_gate_implementation(backend, "measure", (qubit,))
        readout_fidelity = _measure_fidelity(metrics, impl_name, qubit)
        repeatability = metrics.get_measure_repeatability("measure", impl_name, (qubit,))

        qubit_rows.append(
            {
                "qubit": qubit,
                "readout_fidelity": readout_fidelity,
                "readout_error": None if readout_fidelity is None else 1 - readout_fidelity,
                "repeatability": repeatability,
                "t1_us": None if qubit not in t1_times else t1_times[qubit] * 1e6,
                "t2_us": None if qubit not in t2_times else t2_times[qubit] * 1e6,
            }
        )

    connection_rows = []
    for locus in backend.architecture.gates["cz"].implementations[
        _default_gate_implementation(backend, "cz", backend.architecture.gates["cz"].loci[0])
    ].loci:
        locus = tuple(locus)
        impl_name = _default_gate_implementation(backend, "cz", locus)
        fidelity = metrics.get_gate_fidelity("cz", impl_name, locus)
        duration = metrics.get_gate_duration("cz", impl_name, locus)

        connection_rows.append(
            {
                "connection": " - ".join(locus),
                "qubit_1": locus[0],
                "qubit_2": locus[1],
                "cz_fidelity": fidelity,
                "cz_error": None if fidelity is None else 1 - fidelity,
                "duration_ns": None if duration is None else duration * 1e9,
            }
        )

    qubits = (
        pd.DataFrame(qubit_rows)
        .sort_values(["readout_fidelity", "repeatability", "t1_us", "t2_us"], ascending=False, na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )
    connections = (
        pd.DataFrame(connection_rows)
        .sort_values(["cz_fidelity"], ascending=False, na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )

    return qubits, connections


def show_best_qubits_and_connections(backend, top_n=10):
    """Display ranked qubits and CZ connections in a notebook."""
    from IPython.display import Markdown, display

    qubits, connections = get_best_qubits_and_connections(backend, top_n=top_n)
    display(Markdown("### Best qubits"))
    display(qubits)
    display(Markdown("### Best CZ connections"))
    display(connections)
    return qubits, connections