"""Offline QPU architectures for a score after layout and routing.

Each architecture is a fake backend: a captured coupling map, native gates and calibration.
No architecture sends a job or needs an account.
"""

from functools import cache
import os
from pathlib import Path

from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.validation import circuit_metrics

SNAPSHOTS = Path(__file__).with_name("qpu_snapshots")
ARCHITECTURES = {
    "ibm_heron_r2": ("ibm", "IBM Heron r2: 156 qubits, heavy-hex, CZ (ibm_fez calibration)"),
    "ibm_heron_r3": ("ibm", "IBM Heron r3: 156 qubits, heavy-hex, CZ (ibm_boston calibration of 2026-04-17)"),
    "ibm_nighthawk": ("ibm", "IBM Nighthawk: 120 qubits, square lattice, CZ (ibm_miami r1 calibration;"
                             " r2 has the same lattice)"),
    "iqm_garnet": ("iqm", "IQM Crystal 20 (Garnet): 20 qubits, square lattice, PRX + CZ"),
    "iqm_emerald": ("iqm", "IQM Crystal 54 (Emerald): 54 qubits, square lattice, PRX + CZ (generic Crystal 54 errors)"),
    "iqm_sirius": ("iqm", "IQM Star 24 (Sirius): 24 qubits on one resonator, PRX + CZ + MOVE (synthetic errors)"),
}
GROUPS = {"all": tuple(ARCHITECTURES),
          **{vendor: tuple(name for name, (owner, _) in ARCHITECTURES.items() if owner == vendor)
             for vendor in ("ibm", "iqm")}}
LAYOUT_SEED = 0
# The routed gate count changes much with the seed, and one routing takes about 0.05 s at 12 qubits.
ROUTING_SEEDS = 16


def select(names) -> tuple[str, ...]:
    """Expand the groups all, ibm and iqm and return each architecture once, in the given order."""
    selected = []
    for name in names:
        if name not in ARCHITECTURES and name not in GROUPS:
            raise ValueError(f"Unknown architecture {name!r}. Use {', '.join([*GROUPS, *ARCHITECTURES])}")
        selected += [item for item in GROUPS.get(name, (name,)) if item not in selected]
    return tuple(selected)


def _ibm_snapshot(device):
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

    # The files come from qiskit-ibm-runtime 0.50.1, which needs a newer Qiskit than this project.
    class Snapshot(FakeBackendV2):
        dirname = str(SNAPSHOTS / device)
        conf_filename = f"conf_{device}.json"
        props_filename = f"props_{device}.json"
        backend_name = f"fake_{device}"

    return Snapshot()


def _star(size=24):
    from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend
    from iqm.station_control.interface.models import StaticQuantumArchitecture

    qubits = [f"QB{i}" for i in range(1, size + 1)]
    architecture = StaticQuantumArchitecture(dut_label=f"star{size}", qubits=qubits,
                                             computational_resonators=["CR1"],
                                             connectivity=[("CR1", qubit) for qubit in qubits])
    # Error values of the IQM Deneb fake backend (Star with six qubits), used for each qubit.
    errors = IQMErrorProfile(
        t1s={"CR1": 5400.0, **dict.fromkeys(qubits, 35000.0)},
        t2s={"CR1": 10800.0, **dict.fromkeys(qubits, 33000.0)},
        single_qubit_gate_depolarizing_error_parameters={"prx": dict.fromkeys(qubits, 0.0002)},
        two_qubit_gate_depolarizing_error_parameters={"cz": {(q, "CR1"): 0.0128 for q in qubits},
                                                      "move": {(q, "CR1"): 0.0 for q in qubits}},
        single_qubit_gate_durations={"prx": 40.0}, two_qubit_gate_durations={"cz": 120.0, "move": 96.0},
        readout_errors={q: {"0": 0.02, "1": 0.02} for q in qubits}, name=f"Star{size}")
    return IQMFakeBackend(architecture, errors, name=f"IQMFakeStar{size}")


@cache
def backend(name):
    """Return the fake backend of one architecture. One process builds each backend once."""
    if name == "ibm_heron_r2":
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez()
    if name == "ibm_heron_r3":
        return _ibm_snapshot("boston")
    if name == "ibm_nighthawk":
        return _ibm_snapshot("miami")
    if name == "iqm_garnet":
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
        return IQMFakeGarnet()
    if name == "iqm_emerald":
        from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite
        return IQMFakeAphrodite()
    if name == "iqm_sirius":
        return _star()
    raise ValueError(f"Unknown architecture {name!r}")


def route(circuit: QuantumCircuit, name: str, seed: int = LAYOUT_SEED) -> QuantumCircuit:
    """Place, route and translate the circuit to the native gates with the given seed."""
    from qiskit.transpiler import generate_preset_pass_manager

    device = backend(name)
    # The IQM scheduling plugin adds the MOVE gates of a Star device, but it renumbers the qubits of a
    # Crystal circuit; so only the Star uses the backend, the others use the target alone.
    device_options = {"backend": device} if name == "iqm_sirius" else {"target": device.target}
    # OpenEvolve forks its evaluators after the campaign process routed the diagnostics. A forked child has
    # no copy of the Rust thread pool, so a parallel pass waits forever; Qiskit runs serial with this flag.
    os.environ["QISKIT_IN_PARALLEL"] = "TRUE"
    manager = generate_preset_pass_manager(optimization_level=3, seed_transpiler=seed, **device_options)
    routed = manager.run(circuit)
    return _resonators_last(routed) if name == "iqm_sirius" else routed


def _resonators_last(routed: QuantumCircuit) -> QuantumCircuit:
    """Return the routed Star circuit with the resonator register last, at the index of the backend.

    The IQM Star plugin builds its output registers in an order that changes between processes, so the
    resonator can be qubit 0; then the error lookup and the noisy simulation use wrong components. The
    synthetic Star errors are the same on each qubit, so only the index of the resonator matters.
    """
    registers = sorted(routed.qregs, key=lambda register: register.name == "resonators")
    if registers == list(routed.qregs):
        return routed
    fixed = QuantumCircuit(*registers, *routed.cregs, name=routed.name, global_phase=routed.global_phase,
                           metadata=routed.metadata)
    for item in routed.data:
        fixed._append(item)
    fixed._layout = routed.layout
    return fixed


def _gate_error(device, gate, qargs) -> float:
    """Return the gate error of the target, or else the depolarizing parameter of an IQM error profile."""
    properties = device.target[gate].get(qargs) if gate in device.target.operation_names else None
    if properties is not None and properties.error is not None:
        return properties.error
    profile = getattr(device, "error_profile", None)
    if profile is None:
        raise ValueError(f"No error data for {gate} on {qargs}")
    names = tuple(device.index_to_qubit_name(q) for q in qargs)
    if len(names) == 1:
        return profile.single_qubit_gate_depolarizing_error_parameters[{"r": "prx"}.get(gate, gate)][names[0]]
    table = profile.two_qubit_gate_depolarizing_error_parameters[gate]
    return table[names] if names in table else table[names[::-1]]


def _costs(routed: QuantumCircuit, device) -> dict:
    success = 1.0
    for item in routed.data:
        success *= 1.0 - _gate_error(device, item.operation.name, tuple(routed.find_bit(q).index for q in item.qubits))
    metrics = circuit_metrics(routed)
    return {"routed_two_qubit_count": metrics["two_qubit_gate_count"],
            "routed_two_qubit_depth": metrics["two_qubit_depth"],
            "routed_depth": metrics["depth"],
            "routed_one_qubit_count": metrics["one_qubit_gate_count"],
            "routed_success_estimate": float(success)}


def routed_costs(circuit: QuantumCircuit, name: str, seeds: int = ROUTING_SEEDS, *,
                 objective: str = "preparation") -> dict:
    """Route with the seeds 0 to seeds - 1 and return the costs of the best routing and its seed.

    The costs are the gate counts after routing and the product of (1 - gate error) over all gates.
    The best routing has the fewest two-qubit gates, then the smallest two-qubit depth, the largest
    success estimate and the smallest depth (the order of oe_evaluator.ROUTED_RANK). With
    objective="cost", the largest success estimate comes first (oe_evaluator.COST_RANK).
    route(circuit, name, costs["routed_seed"]) gives the same routed circuit again.
    """
    device = backend(name)
    found = [{**_costs(route(circuit, name, seed), device), "routed_seed": seed} for seed in range(seeds)]

    def key(costs):
        counts = (costs["routed_two_qubit_count"], costs["routed_two_qubit_depth"])
        success = -costs["routed_success_estimate"]
        return (success, *counts) if objective == "cost" else (*counts, success)

    return min(found, key=lambda costs: (*key(costs), costs["routed_depth"]))
