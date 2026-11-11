"""Keep the best circuit of each state and processor in quantum_circuits/<state>/<processor>/.

A discovery run starts from a compatible record or a verified result of a previous campaign.
It uses the processor record, then the logical record, when no better previous campaign is available.
Without either, it uses the baseline. At its end, the run winner replaces a record only if it is
valid and ranks higher (oe_evaluator.rank_key) when both circuits are measured again with the same code.
A run with objective="cost" keeps its records apart, in quantum_circuits/<state>/<processor>_cost/.
The replaced records and one history line for each compared winner go to the archive
artifacts/evolution/records/<state>/<processor>/, which git ignores.
"""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
from qiskit import QuantumCircuit, qpy

from .architectures import route, routed_costs
from .artifacts import file_hash, main_checkout
from .decoders import with_decoder_check
from .oe_evaluator import rank_key
from .skeleton_fit import skeleton_of
from .state_workloads import evaluate_workload, layout, party_encodings

LOGICAL = "logical"
COST_SUFFIX = "_cost"
RESULTS = "results.json"
HISTORY = "history.jsonl"
# A record must have the same settings as the run that starts from it or replaces it. A fixed
# encoding also depends on the method (E* or canonical).
MATCHED_SETTINGS = ("encoding_mode", "objective")
_READ_ERRORS = (OSError, KeyError, TypeError, ValueError, qpy.QpyError)
_ROOT = main_checkout(Path(__file__).resolve().parents[3])


@dataclass(frozen=True)
class RecordStore:
    """The record folders (root) and the archive of replaced records and history lines."""

    root: Path
    archive: Path

    def directory(self, state, processor=None, objective=None):
        return self.root / state / ((processor or LOGICAL) + (COST_SUFFIX if objective == "cost" else ""))


STORE = RecordStore(_ROOT / "quantum_circuits", _ROOT / "artifacts/evolution/records")


@dataclass(frozen=True)
class Record:
    """One checked circuit: the preparation, one encoding per party, the optional decoder and its metrics."""

    directory: Path
    circuit: QuantumCircuit
    encodings: np.ndarray
    decoder: QuantumCircuit | None
    metrics: dict
    settings: dict
    program: str | None = None


def _load_circuit(path, num_qubits):
    with path.open("rb") as stream:
        circuits = qpy.load(stream)
    if len(circuits) != 1 or circuits[0].num_qubits != num_qubits:
        raise ValueError(f"{path} must contain one circuit with {num_qubits} qubits")
    return circuits[0]


def _matched(settings):
    keys = MATCHED_SETTINGS + (("method",) if settings.get("encoding_mode") == "fixed" else ())
    return {key: settings.get(key) for key in keys}


def measure(state, circuit, encodings, decoder, processor=None, objective="preparation"):
    """Check the state (and the decoder) and return the metrics, with the routed costs for a processor."""
    metrics = {**evaluate_workload(state, circuit, encodings, tolerance=1e-10), "decoder_cz_count": 0}
    if decoder is not None:
        metrics = with_decoder_check(state, circuit, decoder, metrics)
    if metrics["is_valid"] and processor is not None:
        metrics.update(routed_costs(circuit, processor, objective=objective))
    return metrics


def read(directory, state):
    """Load and check the record in directory. Raise ValueError if it is not a valid record of state."""
    n, _, q = layout(state)
    results = json.loads((directory / RESULTS).read_text(encoding="utf-8"))
    if results.get("state") != state:
        raise ValueError(f"{directory / RESULTS} is not a record of {state}")
    encoding = results["encoding"]
    encodings = party_encodings(state, np.asarray(encoding["real"]) + 1j * np.asarray(encoding["imag"]))
    circuit = _load_circuit(directory / "circuit.qpy", n * q)
    decoder = _load_circuit(directory / "decoder.qpy", n * q) if (directory / "decoder.qpy").is_file() else None
    metrics = measure(state, circuit, encodings, decoder)
    if not metrics["is_valid"]:
        raise ValueError(f"The circuit in {directory} does not prepare {state}")
    return Record(directory, circuit, encodings, decoder, metrics, results.get("settings", {}))


def seed_records(store, state, processor, settings, *, run_roots=()):
    """Return (seed, others): the record to start from and the other usable records of the state.

    The seed is the record of the processor, else the logical record. A usable record has the same
    MATCHED_SETTINGS as settings. A record that does not load or check is skipped with a message.
    A cost run without its own records starts from the records with objective "preparation".
    run_roots also supplies compatible results from previous campaigns, including saved populations.
    """
    usable = {}
    folder = store.root / state
    for directory in sorted(folder.iterdir()) if folder.is_dir() else ():
        if not (directory / RESULTS).is_file():
            continue
        try:
            record = read(directory, state)
        except _READ_ERRORS as exc:
            print(f"Record {directory}: skipped ({type(exc).__name__}: {exc})", flush=True)
            continue
        if _matched(record.settings) == _matched(settings):
            usable[directory.name] = record
    objective = settings.get("objective")
    seed = usable.get(store.directory(state, processor, objective).name) or usable.get(LOGICAL)
    others = [record for record in usable.values() if record is not seed]
    if run_roots:
        from .continuation import best_previous

        previous = best_previous(run_roots, state, processor, settings, seed)
        if previous is not seed:
            if seed is not None:
                others.insert(0, seed)
            seed = previous
    if seed is None and objective == "cost":
        seed, more = seed_records(store, state, processor, {**settings, "objective": "preparation"},
                                  run_roots=run_roots)
        others += more
    return seed, others


def _write_npy(path, array):
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def _write_qpy(path, circuit):
    with path.open("xb") as stream:
        qpy.dump(circuit, stream)


def as_baseline(record, directory, state, method):
    """Write the record as a baseline directory and return the directory.

    Write manifest.json and summary.json. Copy the saved program when one is available.
    """
    directory.mkdir(parents=True)
    _write_qpy(directory / "circuit.qpy", record.circuit)
    _write_npy(directory / "encodings.npy", record.encodings)
    files = {name: file_hash((directory / name).read_bytes()) for name in ("circuit.qpy", "encodings.npy")}
    if record.program is not None:
        (directory / "initial_program.py").write_text(record.program, encoding="utf-8")
        files["initial_program.py"] = file_hash((directory / "initial_program.py").read_bytes())
    (directory / "manifest.json").write_text(json.dumps({"files": files}, indent=1) + "\n", encoding="utf-8")
    row = {"workload": state, "method": method, "is_valid": True, "circuit_artifact": "circuit.qpy",
           "encoding_artifact": "encodings.npy", "source": str(record.directory)}
    (directory / "summary.json").write_text(json.dumps({"results": [row]}, indent=1) + "\n", encoding="utf-8")
    return directory


def as_warm_start(record, directory):
    """Write the record as best.qpy and best_encodings.npy, the files a campaign reads as a warm start."""
    directory.mkdir(parents=True)
    _write_qpy(directory / "best.qpy", record.circuit)
    _write_npy(directory / "best_encodings.npy", record.encodings)
    return directory


def _plain(data):
    """Return data with plain JSON types (NumPy scalars become Python numbers)."""
    return json.loads(json.dumps(data, default=lambda value: value.item()))


def _summary(metrics):
    names = ("cz_count", "two_qubit_depth", "decoder_cz_count", "routed_two_qubit_count",
             "routed_two_qubit_depth", "routed_success_estimate")
    return {name: metrics[name] for name in names if name in metrics}


def _native(circuit, processor, metrics):
    """Return the routed circuit, its JSON description (native gates on physical qubits) and its summary."""
    routed = route(circuit, processor, metrics["routed_seed"])
    initial = routed.layout.initial_index_layout(filter_ancillas=True)
    final = routed.layout.final_index_layout(filter_ancillas=True)
    gates = [{"name": item.operation.name, "qubits": [routed.find_bit(q).index for q in item.qubits],
              "params": [float(p) for p in item.operation.params]} for item in routed.data]
    description = {"processor": processor, "basis_gates": sorted(routed.count_ops()),
                   "num_physical_qubits": routed.num_qubits, "global_phase": float(routed.global_phase),
                   "initial_physical_qubits": initial, "final_physical_qubits": final, "gates": gates}
    summary = {"basis_gates": description["basis_gates"],
               "two_qubit_count": metrics["routed_two_qubit_count"],
               "two_qubit_depth": metrics["routed_two_qubit_depth"], "depth": metrics["routed_depth"],
               "one_qubit_count": metrics["routed_one_qubit_count"],
               "success_estimate": metrics["routed_success_estimate"], "routing_seed": metrics["routed_seed"],
               "initial_physical_qubits": initial, "final_physical_qubits": final}
    return routed, description, summary


def _write(target, state, processor, circuit, encodings, decoder, metrics, provenance):
    """Write a new record folder in the format of quantum_circuits/ame43/<processor>/."""
    n, d, q = layout(state)
    target.mkdir(parents=True)
    _write_qpy(target / "circuit.qpy", circuit)
    files = {"circuit.qpy": f"logical circuit as found (u + cz, {n * q} qubits, no measurement)"}
    if decoder is not None:
        _write_qpy(target / "decoder.qpy", decoder)
        files["decoder.qpy"] = "maps the encoding to the canonical one (qudit level j -> basis state j)"
    results = {"state": state, "processor": processor or LOGICAL, "date": f"{datetime.now():%Y-%m-%d}",
               **provenance,
               "logical": {"num_qubits": circuit.num_qubits, "gates": sorted(circuit.count_ops()),
                           "cz_count": metrics["cz_count"], "two_qubit_depth": metrics["two_qubit_depth"],
                           "depth": metrics["depth"], "one_qubit_gate_count": metrics["one_qubit_gate_count"],
                           "decoder_cz_count": metrics["decoder_cz_count"], "F": metrics["F"],
                           "leakage": metrics["L"], "cz_pairs": [list(p) for p in skeleton_of(circuit)]}}
    if processor is not None:
        routed, description, results["native"] = _native(circuit, processor, metrics)
        _write_qpy(target / "circuit_native.qpy", routed)
        (target / "circuit_native.json").write_text(json.dumps(_plain(description), indent=1) + "\n",
                                                    encoding="utf-8")
        files["circuit_native.qpy"] = "the same circuit in the processor native gates on physical qubits"
        files["circuit_native.json"] = "gate list of circuit_native.qpy"
    results["encoding"] = {
        "description": (f"per party k: {1 << q}x{d} isometry, column j = {q}-qubit state of qudit level j"
                        f" (row index = basis state of qubits {q}k to {q}k+{q - 1}, Qiskit little-endian)"),
        "real": np.real(encodings).tolist(), "imag": np.imag(encodings).tolist()}
    results["qubit_order"] = (f"party k on logical qubits {q}k to {q}k+{q - 1}"
                              + ("; logical qubit k ends on physical qubit final_physical_qubits[k]"
                                 if processor is not None else ""))
    results["files"] = files
    (target / RESULTS).write_text(json.dumps(_plain(results), indent=1) + "\n", encoding="utf-8")


def update(store, state, processor, campaign, *, objective, provenance):
    """Compare the valid best circuit of a campaign directory with the record and keep the better one.

    provenance holds the "source" and "settings" entries of results.json; settings name the
    MATCHED_SETTINGS. A record with other settings stays (the result is "skipped"). A record that does
    not load is archived and replaced. Return a JSON summary of the comparison.
    """
    n, _, q = layout(state)
    circuit = _load_circuit(campaign / "best.qpy", n * q)
    encodings = party_encodings(state, np.load(campaign / "best_encodings.npy", allow_pickle=False))
    decoder = _load_circuit(campaign / "decoder.qpy", n * q) if (campaign / "decoder.qpy").is_file() else None
    metrics = measure(state, circuit, encodings, decoder, processor, objective)
    if not metrics["is_valid"]:
        raise ValueError(f"The best circuit of {campaign} does not prepare {state}")
    target = store.directory(state, processor, objective)
    summary = {"date": f"{datetime.now():%Y%m%d_%H%M%S}", "record": str(target), **provenance.get("source", {}),
               "new": _summary(metrics), "old": None}
    if target.exists():
        try:
            record = read(target, state)
        except _READ_ERRORS as exc:
            print(f"Record {target}: replaced, it does not load ({type(exc).__name__}: {exc})", flush=True)
        else:
            if _matched(record.settings) != _matched(provenance["settings"]):
                return _plain({**summary, "accepted": False, "skipped": f"the record has other settings"
                                                                        f" {_matched(record.settings)}"})
            summary["old"] = _summary(old := measure(state, record.circuit, record.encodings, record.decoder,
                                                     processor, objective))
            if rank_key(metrics, objective) <= rank_key(old, objective):
                summary["accepted"] = False
    summary.setdefault("accepted", True)
    archive = store.archive / state / target.name
    archive.mkdir(parents=True, exist_ok=True)
    if summary["accepted"]:
        if target.exists():
            summary["archived"] = str(shutil.copytree(target, archive / summary["date"]))
            shutil.rmtree(target)
        _write(target, state, processor, circuit, encodings, decoder, metrics, provenance)
    with (archive / HISTORY).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(summary)) + "\n")
    return _plain(summary)
