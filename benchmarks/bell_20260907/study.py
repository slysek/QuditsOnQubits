"""Reproducible local design and analysis for the 2026-09-07 hardware study."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_fourier_graph_state_circuit,
    build_exact_optimized_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import optimal_f3_leakage_phase
from qudits_on_qubits.benchmarks.direct_basis.circuit_serialization import normalize_circuit_bit_indices
from qudits_on_qubits.core.benchmark_encoding_bases import generate_monomial_full_bases
from qudits_on_qubits.bell_measurements.sampler_circuits import build_sampler_circuits_for_candidate
from qudits_on_qubits.bell_measurements.postprocessing import bitstring_to_qutrit_outcomes
from qudits_on_qubits.reference_experiments import get_reference_experiment

ROOT = Path(__file__).resolve().parent / "weighted"
STATES = ("two_qutrit", "ghz3", "ame43")
VARIANTS = ("standard", "optimal")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows scanners may briefly hold the destination. Retry only the local
    # atomic rename; this never repeats a remote submission.
    for attempt in range(8):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * 2**attempt)


def encodings():
    # Repo phases are cube roots of unity. Fixing the first support phase to 0
    # removes exactly the threefold global-phase redundancy.
    result = {name: e for _, name, e in generate_monomial_full_bases(None) if name.split("_ph")[1][0] == "0"}
    assert len(result) == 216
    result["canonical_ez"] = np.eye(4, 3, dtype=complex)
    return result


def workload(state, name, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown F3 variant")
    e = encodings()[name]
    phase = 0.0 if variant == "standard" else optimal_f3_leakage_phase(e).phase
    preparation = build_direct_basis_fourier_graph_state_circuit(state, e, leakage_phase=phase)
    n = preparation.num_qubits // 2
    # Share the repository's exact weighted-edge consolidation and disjoint
    # edge scheduling in BOTH F3 arms, retaining their explicit F3 preparation.
    edge_template = build_exact_optimized_direct_basis_graph_state_circuit(state,e)
    optimized = preparation.copy_empty_like()
    for inst in preparation.data[:2*n]:
        optimized.append(inst.operation,[preparation.find_bit(q).index for q in inst.qubits])
    for inst in edge_template.data[n:]:
        optimized.append(inst.operation,[edge_template.find_bit(q).index for q in inst.qubits])
    preparation = optimized
    # Match experiments.preparation: logical parties run from the most to the
    # least significant 2-qubit block, with little-endian bits inside a block.
    pairs = [(2*(n-1-i), 2*(n-1-i)+1) for i in range(n)]
    circuits, meta = build_sampler_circuits_for_candidate(state, preparation, e, qutrit_qubits=pairs)
    return circuits, meta


def outcome_scores(meta, width):
    """Per-setting Bell scores; invalid shots receive zero unconditionally."""
    settings = meta["setting_by_circuit_index"]
    scores, accepted = [], []
    for setting in settings:
        sc = np.zeros(2**width)
        ok = np.zeros(2**width, dtype=bool)
        for idx in range(2**width):
            outcomes = bitstring_to_qutrit_outcomes(
                format(idx, f"0{width}b"), meta["qutrit_bit_indices_by_setting"][setting],
                outcome_map=meta["physical_to_logical_outcome_map"],
            )
            if None in outcomes:
                continue
            ok[idx] = True
            value = sum(complex(term["coeff"]) * np.exp(2j*np.pi/3*sum(p*v for p,v in zip(term["powers"], outcomes)))
                        for term in meta["terms"] if tuple(term["settings"]) == setting)
            if abs(value.imag) > 1e-8:
                raise ValueError("Bell scores must be real")
            sc[idx] = value.real
        scores.append(sc)
        accepted.append(ok)
    return np.asarray(scores), np.asarray(accepted)


def compact_measured(circuit):
    """Remove idle hardware wires, preserving physical and classical ordering."""
    circuit = normalize_circuit_bit_indices(circuit)
    active = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})
    positions = {q:i for i,q in enumerate(active)}
    qc = QuantumCircuit(len(active), circuit.num_clbits, global_phase=circuit.global_phase)
    for inst in circuit.data:
        qc.append(inst.operation, [positions[circuit.find_bit(q).index] for q in inst.qubits],
                  [circuit.find_bit(c).index for c in inst.clbits])
    return qc


def exact_probabilities(circuit):
    qc = compact_measured(circuit)
    if qc.num_qubits > 16:
        raise ValueError("exact verification would exceed 16 active qubits")
    mapping = {qc.find_bit(inst.clbits[0]).index: qc.find_bit(inst.qubits[0]).index
               for inst in qc.data if inst.operation.name == "measure"}
    if sorted(mapping) != list(range(qc.num_clbits)):
        raise ValueError("every classical bit must be measured")
    probabilities = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).probabilities()
    output = np.zeros(2**qc.num_clbits)
    for idx,p in enumerate(probabilities):
        classical = sum(((idx >> q)&1) << c for c,q in mapping.items())
        output[classical] += p
    return output


def metrics(circuits, backend=None):
    two = [sum(len(i.qubits) == 2 for i in c.data) for c in circuits]
    depths = [c.depth() for c in circuits]
    cost = 0.0
    missing = 0
    for c in circuits:
        for inst in c.data:
            if inst.operation.name in ("barrier", "delay"):
                continue
            error = None
            if backend is not None:
                try:
                    prop = backend.target[inst.operation.name][tuple(c.find_bit(q).index for q in inst.qubits)]
                    error = prop.error if prop else None
                except KeyError:
                    pass
            if error is None or not np.isfinite(error) or not 0 <= error < 1:
                missing += 1
                error = 0.01 if len(inst.qubits) == 2 else (0.02 if inst.operation.name == "measure" else 0.0005)
            cost += -math.log1p(-error)
    return {"two_qubit_total":int(sum(two)), "two_qubit_max":int(max(two)),
            "depth_max":int(max(depths)), "depth_mean":float(np.mean(depths)),
            "size_total":int(sum(c.size() for c in circuits)), "error_proxy":cost/len(circuits),
            "missing_error_entries":missing}


def evaluate_counts(counts, meta, *, samples=2000, seed=907):
    width = 2 * len(meta["qutrit_qubits"])
    score, accepted = outcome_scores(meta, width)
    arrays = np.zeros_like(score)
    if len(counts) != len(score):
        raise ValueError("count cardinality mismatch")
    for row, values in zip(arrays, counts):
        for bits, count in values.items():
            if type(count) is not int or count < 0:
                raise ValueError("counts must be nonnegative integers")
            row[int(bits.replace(" ", ""), 2)] += count
    total = arrays.sum(axis=1)
    good = (arrays*accepted).sum(axis=1)
    if np.any(total <= 0) or np.any(good <= 0):
        raise ValueError("empty setting or no accepted shots")
    sums = (arrays*score).sum(axis=1)
    output = {"unconditional":float(np.sum(sums/total)), "conditional":float(np.sum(sums/good)),
              "invalid_fraction":float(1-good.sum()/total.sum()), "total_shots":int(total.sum())}
    if samples:
        rng = np.random.default_rng(seed)
        bu, bc = np.zeros(samples), np.zeros(samples)
        for row,sc,ok,n in zip(arrays,score,accepted,total):
            draws = rng.multinomial(int(n),row/n,size=samples)
            numerator = draws @ sc
            denominators = draws @ ok
            if np.any(denominators == 0):
                raise ValueError("bootstrap setting without accepted shots")
            bu += numerator/n
            bc += numerator/denominators
        for key, draws in (("unconditional",bu),("conditional",bc)):
            output[key+"_se"] = float(np.std(draws,ddof=1))
            output[key+"_ci95"] = np.quantile(draws,[.025,.975]).tolist()
    return output


def generic_item(args):
    state,name = args
    rows = []
    for variant in VARIANTS:
        circuits,meta = workload(state,name,variant)
        compiled = transpile(circuits,basis_gates=["cz","rz","sx","x"], optimization_level=3,seed_transpiler=907,num_processes=1)
        rows.append({"state":state,"name":name,"variant":variant,**metrics(compiled)})
    return rows


def screen(workers=4):
    names = list(encodings())
    todo = [(state,name) for state in STATES for name in names]
    rows = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index,result in enumerate(pool.map(generic_item,todo,chunksize=1)):
            rows.extend(result)
            if index%12 == 0:
                print(json.dumps({"done":index+1,"of":len(todo),"elapsed_s":round(time.time()-start)}),flush=True)
                save_json(ROOT/"generic_screen.json",rows)
    save_json(ROOT/"generic_screen.json",rows)
    shortlist = {}
    for state in STATES:
        ranked = sorted((r for r in rows if r["state"] == state and r["variant"] == "optimal"),
                        key=lambda r:(r["two_qubit_total"],r["depth_max"],r["size_total"],r["name"]))
        # Broad shortlist includes both F3 arms' best circuits.
        standard = sorted((r for r in rows if r["state"] == state and r["variant"] == "standard"),
                          key=lambda r:(r["two_qubit_total"],r["depth_max"],r["size_total"],r["name"]))
        chosen = list(dict.fromkeys([r["name"] for r in ranked[:12]]+[r["name"] for r in standard[:6]]+["canonical_ez"]))
        shortlist[state] = chosen
    save_json(ROOT/"shortlist.json",shortlist)
    print(json.dumps(shortlist),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["screen"])
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    screen(args.workers)
