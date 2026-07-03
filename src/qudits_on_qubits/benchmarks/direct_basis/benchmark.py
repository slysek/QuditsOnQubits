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
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_edge_gate,
    build_direct_basis_fourier_gate,
    build_direct_basis_graph_state_circuit,
    gate_as_circuit,
    resolve_direct_state,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    embed_single_qutrit_gate_identity_leakage,
    is_isometry,
    is_unitary,
)
from qudits_on_qubits.benchmarks.direct_basis.selection import (
    selection_label as format_selection_label,
    transpiled_qpy_filename,
)
from qudits_on_qubits.core.benchmark_encoding_bases import BASIS_GATES, COUPLING_MAP, TWO_Q_GATES
from qudits_on_qubits.core.project_paths import repo_path


METHOD_NAME = "direct_basis_encoding"


def default_results_dir() -> str:
    return repo_path("artifacts", "direct_basis_runs", "raw")


def default_quantum_circuits_dir() -> str:
    return repo_path("artifacts", "direct_basis_runs", "raw", "quantum_circuits")


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
    approximation_degree: float | None = None,
    selection_label: str = "exact",
) -> dict:
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    return {
        "method": METHOD_NAME,
        "selection_label": selection_label,
        "approximation_degree": "" if approximation_degree is None else float(approximation_degree),
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
        "basis_matrix_isometry": False,
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
        "graph_state_transpiled_qpy": "",
        "basis_change_qpy": "",
        "basis_change_matrix_npy": "",
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
    approximation_degree: float | None = None,
    selection_label: str = "exact",
    legacy_exact_transpiled_filename: bool = True,
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
        approximation_degree=approximation_degree,
        selection_label=selection_label,
    )

    started = time.time()
    try:
        basis_matrix = np.asarray(basis_matrix, dtype=complex)
        row["basis_matrix_unitary"] = is_unitary(basis_matrix)
        row["basis_matrix_isometry"] = is_isometry(basis_matrix)

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
                selection_label=selection_label,
                legacy_exact_transpiled_filename=legacy_exact_transpiled_filename,
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

    last_trial_error = ""
    for trial in range(int(n_transpile_runs)):
        try:
            transpile_kwargs = {
                "basis_gates": basis_gates,
                "coupling_map": coupling_map,
                "optimization_level": 3,
                "seed_transpiler": trial,
            }
            if approximation_degree is not None:
                transpile_kwargs["approximation_degree"] = float(approximation_degree)
            qc_t = transpile(qc, **transpile_kwargs)
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # Catch BaseException (not just Exception) because Qiskit's Rust
            # backend can raise pyo3_runtime.PanicException (e.g. the known
            # TwoQubitWeylDecomposition bug, qiskit-terra issue #4159) which
            # does not inherit from Exception and would otherwise crash the
            # entire benchmark run instead of being treated as a failed trial.
            row["failed_trials"] += 1
            last_trial_error = f"{type(exc).__name__}: {exc}".splitlines()[0]
            print(
                f"  transpile trial {trial} failed: {last_trial_error}",
                flush=True,
            )

    row["compile_time_seconds"] = round(time.time() - started, 6)
    if row["successful_trials"] == 0:
        row["status"] = "all_transpile_failed"
        row["error_message"] = (
            f"All {n_transpile_runs} transpilation trials failed. "
            f"Last error: {last_trial_error}"
            if last_trial_error
            else "All transpilation trials failed."
        )
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

    if row["graph_state_transpiled_qpy"]:
        _save_qpy(best["qc"], row["graph_state_transpiled_qpy"])

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
    approximation_degree: float | None = None,
    selection_label: str = "exact",
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
        approximation_degree=approximation_degree,
        selection_label=selection_label,
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
    approximation_degrees: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    rows = []
    candidates = list(candidates)
    run_specs = _direct_basis_run_specs(approximation_degrees)
    for index, candidate in enumerate(candidates, start=1):
        for approximation_degree, run_label, legacy_exact_transpiled_filename in run_specs:
            print(
                f"[direct_basis_encoding] {index}/{len(candidates)} "
                f"{state_name} {run_label} {candidate.class_name}/{candidate.candidate_name}",
                flush=True,
            )
            if not candidate.is_supported:
                row = failed_candidate_row(
                    state_name=state_name,
                    candidate=candidate,
                    n_qutrits=n_qutrits,
                    n_transpile_runs=n_transpile_runs,
                    approximation_degree=approximation_degree,
                    selection_label=run_label,
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
                    approximation_degree=approximation_degree,
                    selection_label=run_label,
                    legacy_exact_transpiled_filename=legacy_exact_transpiled_filename,
                )

            if row["success"]:
                print(
                    "  ok "
                    f"depth={row.get('circuit_depth')} "
                    f"2q={row.get('two_qubit_gate_count')} "
                    f"size={row.get('total_gate_count')} "
                    f"time={float(row.get('compile_time_seconds', 0.0)):.3f}s",
                    flush=True,
                )
            else:
                error = row["error_message"].splitlines()[0] if row["error_message"] else ""
                print(f"  {row['status']}: {error}", flush=True)
            rows.append(row)

    df = pd.DataFrame(rows)
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[direct_basis_encoding] results CSV: {output_csv}", flush=True)
    return df, output_csv


def _direct_basis_run_specs(
    approximation_degrees: Iterable[float] | None,
) -> list[tuple[float | None, str, bool]]:
    if approximation_degrees is None:
        return [(None, "exact", True)]
    specs: list[tuple[float | None, str, bool]] = [(None, "exact", False)]
    for degree in approximation_degrees:
        value = float(degree)
        specs.append((value, format_selection_label(value), False))
    return specs


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


def _save_npy(matrix: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=complex))
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
    selection_label: str = "exact",
    legacy_exact_transpiled_filename: bool = True,
) -> dict[str, str]:
    """Save direct-basis circuit artifacts before a transpiled circuit is selected."""
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    basis_matrix = np.asarray(basis_matrix, dtype=complex)
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
        "graph_state_transpiled_qpy": os.path.join(
            output_dir,
            transpiled_qpy_filename(
                selection_label,
                legacy_exact=legacy_exact_transpiled_filename,
            ),
        ),
        "basis_change_qpy": "",
        "basis_change_matrix_npy": os.path.join(output_dir, "W.npy" if basis_matrix.shape == (3, 3) else "E.npy"),
    }
    _save_qpy(f3_circuit, paths["f3_w_qpy"])
    _save_qpy(cz_circuit, paths["cz3_w_qpy"])
    _save_qpy(graph_state_circuit, paths["graph_state_qpy"])
    _save_npy(basis_matrix, paths["basis_change_matrix_npy"])
    if basis_matrix.shape == (3, 3) and is_unitary(basis_matrix):
        basis_gate = UnitaryGate(
            embed_single_qutrit_gate_identity_leakage(basis_matrix),
            label="B_W",
        )
        basis_gate.name = "B_W"
        basis_circuit = gate_as_circuit(
            basis_gate,
            2,
            "B_W",
        )
        paths["basis_change_qpy"] = os.path.join(output_dir, "basis_change_gate.qpy")
        _save_qpy(basis_circuit, paths["basis_change_qpy"])
    return paths
