from __future__ import annotations

import os
import re
import time
import traceback
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from qiskit import qpy, transpile
from qiskit.quantum_info import Statevector

from basis_direct_encoding_benchmarks.candidates import DirectBasisCandidate
from basis_direct_encoding_benchmarks.circuits import (
    build_direct_basis_edge_gate,
    build_direct_basis_fourier_gate,
    build_direct_basis_graph_state_circuit,
    gate_as_circuit,
    resolve_direct_state,
)
from basis_direct_encoding_benchmarks.math_utils import is_unitary
from QuditsOnQubits.benchmark_encoding_bases import BASIS_GATES, COUPLING_MAP, TWO_Q_GATES


METHOD_NAME = "direct_basis_encoding"


def default_results_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "results")


def default_quantum_circuits_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "quantum_circuits")


def timestamped_results_path(
    *,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
    prefix: str = "direct_basis_benchmarks",
) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = output_dir or default_results_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{prefix}_{stamp}.csv")


def _count_one_qubit_gates(qc) -> int:
    count = 0
    for instruction in qc.data:
        operation = instruction.operation
        if getattr(operation, "num_qubits", None) == 1:
            count += 1
    return int(count)


def _count_two_qubit_gates_from_ops(ops) -> int:
    return int(sum(value for name, value in ops.items() if name in TWO_Q_GATES))


def _base_row(
    *,
    state_name: str,
    n_qutrits: int | None,
    basis_candidate_name: str,
    basis_candidate_type: str,
    class_name: str,
    candidate_name: str,
    n_transpile_runs: int,
    notes: str,
) -> dict:
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    return {
        "method": METHOD_NAME,
        "state_name": state.state_id,
        "state_family": state.state_family,
        "graph_name": state.graph_type,
        "graph_id": state.state_id,
        "num_qutrits": state.num_qutrits,
        "n_qutrits": state.num_qutrits,
        "num_physical_qubits": 2 * state.num_qutrits,
        "basis_candidate_name": basis_candidate_name,
        "basis_candidate_type": basis_candidate_type,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "basis_matrix_unitary": False,
        "two_qubit_gate_count": None,
        "one_qubit_gate_count": None,
        "total_gate_count": None,
        "circuit_depth": None,
        "fidelity": None,
        "final_error": None,
        "compile_time_seconds": None,
        "success": False,
        "failed": True,
        "status": "not_run",
        "error_message": "",
        "notes": notes,
        "best_depth": None,
        "mean_depth": None,
        "std_depth": None,
        "best_size": None,
        "mean_size": None,
        "best_two_qubit_gate_count": None,
        "mean_two_qubit_gate_count": None,
        "num_qubits": None,
        "best_count_ops": None,
        "n_transpile_runs": n_transpile_runs,
        "successful_trials": 0,
        "failed_trials": 0,
        "quantum_circuit_dir": "",
        "f3_w_qpy": "",
        "cz3_w_qpy": "",
        "graph_state_qpy": "",
    }


def _restore_input_qubit_order(state: np.ndarray, input_to_output: list[int]) -> np.ndarray:
    """Reorder a statevector from final physical order back to input logical order."""
    n_qubits = len(input_to_output)
    restored = np.zeros_like(state)
    for input_index in range(2**n_qubits):
        input_bits = [(input_index >> bit) & 1 for bit in range(n_qubits)]
        output_bits = [0] * n_qubits
        for input_bit, output_bit in enumerate(input_to_output):
            output_bits[output_bit] = input_bits[input_bit]
        output_index = sum(bit_value << bit for bit, bit_value in enumerate(output_bits))
        restored[input_index] = state[output_index]
    return restored


def _safe_fidelity(reference_qc, candidate_qc, *, max_qubits: int) -> tuple[float | None, str]:
    if candidate_qc.num_qubits > int(max_qubits):
        return None, f"Fidelity skipped because transpiled circuit has {candidate_qc.num_qubits} qubits."
    try:
        reference = Statevector.from_instruction(reference_qc)
        candidate = Statevector.from_instruction(candidate_qc)
        candidate_data = candidate.data
        layout = getattr(candidate_qc, "layout", None)
        if layout is not None and getattr(layout, "final_layout", None) is not None:
            final_index_layout = layout.final_index_layout(filter_ancillas=True)
            if len(final_index_layout) == reference_qc.num_qubits == candidate_qc.num_qubits:
                candidate_data = _restore_input_qubit_order(candidate_data, final_index_layout)
        return float(abs(np.vdot(reference.data, candidate_data)) ** 2), ""
    except Exception as exc:
        return None, f"Fidelity skipped: {exc}"


def benchmark_direct_basis(
    *,
    state_name: str,
    basis_matrix: np.ndarray,
    basis_candidate_name: str,
    basis_candidate_type: str,
    source_class_name: str = "",
    source_candidate_name: str = "",
    n_qutrits: int | None = None,
    coupling_map=None,
    basis_gates=None,
    n_transpile_runs: int = 20,
    compute_fidelity: bool = True,
    max_fidelity_qubits: int = 10,
    notes: str = "",
    quantum_circuits_dir: str | None = None,
) -> dict:
    """Benchmark graph-state preparation using direct W-defined encoding."""
    class_name = source_class_name or basis_candidate_type
    candidate_name = source_candidate_name or basis_candidate_name
    row = _base_row(
        state_name=state_name,
        n_qutrits=n_qutrits,
        basis_candidate_name=basis_candidate_name,
        basis_candidate_type=basis_candidate_type,
        class_name=class_name,
        candidate_name=candidate_name,
        n_transpile_runs=n_transpile_runs,
        notes=notes,
    )

    started = time.time()
    try:
        basis_matrix = np.asarray(basis_matrix, dtype=complex)
        row["basis_matrix_unitary"] = is_unitary(basis_matrix)
        if not row["basis_matrix_unitary"]:
            raise ValueError("basis_matrix is not unitary.")

        qc = build_direct_basis_graph_state_circuit(
            state_name,
            basis_matrix,
            n_qutrits=n_qutrits,
        )
        if quantum_circuits_dir is not None:
            paths = export_direct_basis_candidate_circuits(
                quantum_circuits_dir=quantum_circuits_dir,
                state_name=state_name,
                n_qutrits=n_qutrits,
                class_name=class_name,
                candidate_name=candidate_name,
                basis_matrix=basis_matrix,
                graph_state_circuit=qc,
            )
            row.update(paths)
    except Exception:
        row["status"] = "build_error"
        row["error_message"] = traceback.format_exc()
        row["compile_time_seconds"] = round(time.time() - started, 6)
        return row

    basis_gates = basis_gates or BASIS_GATES
    coupling_map = coupling_map if coupling_map is not None else COUPLING_MAP
    depths: list[int] = []
    sizes: list[int] = []
    twoq_counts: list[int] = []
    oneq_counts: list[int] = []
    successful = []

    for trial in range(int(n_transpile_runs)):
        try:
            qc_t = transpile(
                qc,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=3,
                seed_transpiler=trial,
            )
            ops = qc_t.count_ops()
            depth = int(qc_t.depth())
            size = int(qc_t.size())
            twoq = _count_two_qubit_gates_from_ops(ops)
            oneq = _count_one_qubit_gates(qc_t)

            depths.append(depth)
            sizes.append(size)
            twoq_counts.append(twoq)
            oneq_counts.append(oneq)
            row["successful_trials"] += 1
            successful.append(
                {
                    "rank_key": (depth, twoq, size),
                    "depth": depth,
                    "size": size,
                    "twoq": twoq,
                    "oneq": oneq,
                    "ops": dict(ops),
                    "num_qubits": qc_t.num_qubits,
                    "qc": qc_t,
                }
            )
        except Exception:
            row["failed_trials"] += 1

    row["compile_time_seconds"] = round(time.time() - started, 6)
    if row["successful_trials"] == 0:
        row["status"] = "all_transpile_failed"
        row["error_message"] = "All transpilation trials failed."
        return row

    best = sorted(successful, key=lambda item: item["rank_key"])[0]
    row["best_depth"] = best["depth"]
    row["mean_depth"] = round(float(np.mean(depths)), 2)
    row["std_depth"] = round(float(np.std(depths)), 2)
    row["best_size"] = best["size"]
    row["mean_size"] = round(float(np.mean(sizes)), 2)
    row["best_two_qubit_gate_count"] = best["twoq"]
    row["mean_two_qubit_gate_count"] = round(float(np.mean(twoq_counts)), 2)
    row["num_qubits"] = best["num_qubits"]
    row["best_count_ops"] = best["ops"]

    row["two_qubit_gate_count"] = best["twoq"]
    row["one_qubit_gate_count"] = best["oneq"]
    row["total_gate_count"] = best["size"]
    row["circuit_depth"] = best["depth"]

    if compute_fidelity:
        fidelity, fidelity_note = _safe_fidelity(qc, best["qc"], max_qubits=max_fidelity_qubits)
        if fidelity is not None:
            row["fidelity"] = round(fidelity, 12)
            row["final_error"] = round(float(1.0 - fidelity), 12)
        elif fidelity_note:
            row["notes"] = "; ".join(part for part in (row["notes"], fidelity_note) if part)

    row["success"] = True
    row["failed"] = False
    row["status"] = "ok"
    return row


def failed_candidate_row(
    *,
    state_name: str,
    candidate: DirectBasisCandidate,
    n_qutrits: int | None,
    n_transpile_runs: int,
) -> dict:
    row = _base_row(
        state_name=state_name,
        n_qutrits=n_qutrits,
        basis_candidate_name=candidate.name,
        basis_candidate_type=candidate.candidate_type,
        class_name=candidate.class_name,
        candidate_name=candidate.candidate_name,
        n_transpile_runs=n_transpile_runs,
        notes=candidate.notes,
    )
    row["status"] = "unsupported_direct_basis_candidate"
    row["error_message"] = candidate.error_message
    row["compile_time_seconds"] = 0.0
    return row


def benchmark_direct_basis_candidates(
    *,
    state_name: str,
    candidates: Iterable[DirectBasisCandidate],
    n_qutrits: int | None = None,
    coupling_map=None,
    basis_gates=None,
    n_transpile_runs: int = 20,
    compute_fidelity: bool = True,
    max_fidelity_qubits: int = 10,
    output_csv: str | None = None,
    quantum_circuits_dir: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    rows = []
    candidates = list(candidates)
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[direct_basis_encoding] {index}/{len(candidates)} "
            f"{state_name} {candidate.class_name}/{candidate.candidate_name}",
            flush=True,
        )
        if not candidate.is_supported:
            row = failed_candidate_row(
                state_name=state_name,
                candidate=candidate,
                n_qutrits=n_qutrits,
                n_transpile_runs=n_transpile_runs,
            )
        else:
            row = benchmark_direct_basis(
                state_name=state_name,
                n_qutrits=n_qutrits,
                basis_matrix=candidate.matrix,
                basis_candidate_name=candidate.name,
                basis_candidate_type=candidate.candidate_type,
                source_class_name=candidate.class_name,
                source_candidate_name=candidate.candidate_name,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                n_transpile_runs=n_transpile_runs,
                compute_fidelity=compute_fidelity,
                max_fidelity_qubits=max_fidelity_qubits,
                notes=candidate.notes,
                quantum_circuits_dir=quantum_circuits_dir,
            )

        if row["success"]:
            print(
                "  ok "
                f"depth={row['circuit_depth']} "
                f"2q={row['two_qubit_gate_count']} "
                f"size={row['total_gate_count']} "
                f"time={row['compile_time_seconds']:.3f}s",
                flush=True,
            )
        else:
            print(f"  {row['status']}: {row['error_message'].splitlines()[0] if row['error_message'] else ''}", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[direct_basis_encoding] results CSV: {output_csv}", flush=True)
    return df, output_csv


def _safe_path_part(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def candidate_circuit_output_dir(
    quantum_circuits_dir: str,
    *,
    state_name: str,
    class_name: str,
    candidate_name: str,
) -> str:
    return os.path.join(
        quantum_circuits_dir,
        _safe_path_part(state_name),
        f"{_safe_path_part(class_name)}__{_safe_path_part(candidate_name)}",
    )


def _save_qpy(circuit, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        qpy.dump(circuit, handle)
    return path


def export_direct_basis_candidate_circuits(
    *,
    quantum_circuits_dir: str,
    state_name: str,
    n_qutrits: int | None,
    class_name: str,
    candidate_name: str,
    basis_matrix: np.ndarray,
    graph_state_circuit,
) -> dict[str, str]:
    """Save F3^(W), CZ3^(W), and the full direct-basis graph circuit as QPY."""
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    output_dir = candidate_circuit_output_dir(
        quantum_circuits_dir,
        state_name=state.state_id,
        class_name=class_name,
        candidate_name=candidate_name,
    )
    f3_circuit = gate_as_circuit(
        build_direct_basis_fourier_gate(basis_matrix),
        2,
        "F3_W",
    )
    cz_circuit = gate_as_circuit(
        build_direct_basis_edge_gate(basis_matrix),
        4,
        "CZ3_W",
    )

    paths = {
        "quantum_circuit_dir": output_dir,
        "f3_w_qpy": os.path.join(output_dir, "F3_W.qpy"),
        "cz3_w_qpy": os.path.join(output_dir, "CZ3_W.qpy"),
        "graph_state_qpy": os.path.join(output_dir, "graph_state_direct_basis.qpy"),
    }
    _save_qpy(f3_circuit, paths["f3_w_qpy"])
    _save_qpy(cz_circuit, paths["cz3_w_qpy"])
    _save_qpy(graph_state_circuit, paths["graph_state_qpy"])
    return paths
