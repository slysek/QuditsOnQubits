from __future__ import annotations

import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from qiskit import qpy, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from qudits_on_qubits.bell_measurements import build_sampler_circuits_for_candidate
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_edge_gate,
    build_direct_basis_fourier_gate,
    build_direct_basis_graph_state_circuit,
    gate_as_circuit,
    resolve_direct_state,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    encoding_embedding,
    embed_single_qutrit_gate_identity_leakage,
    is_isometry,
    is_unitary,
)
from qudits_on_qubits.benchmarks.direct_basis.selection import (
    selection_label as format_selection_label,
    transpiled_qpy_filename,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    run_iqm_transpiler_strategy,
)
from qudits_on_qubits.core.benchmark_encoding_bases import BASIS_GATES, COUPLING_MAP, TWO_Q_GATES
from qudits_on_qubits.core.project_paths import repo_path
from qudits_on_qubits.experiments.workload_metrics import summarize_compiled_workload


METHOD_NAME = "direct_basis_encoding"
RANKING_WORKLOADS = frozenset({"state_preparation", "bell_measurements"})


def default_results_dir() -> str:
    return repo_path("artifacts", "direct_basis_runs", "raw")


def default_quantum_circuits_dir() -> str:
    return repo_path("artifacts", "direct_basis_runs", "raw", "quantum_circuits")


def default_iqm_results_dir() -> str:
    return repo_path("artifacts", "iqm_runs", "raw")


def default_iqm_quantum_circuits_dir(iqm_backend_name: str) -> str:
    from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import safe_backend_slug

    return repo_path(
        "artifacts",
        "iqm_runs",
        "raw",
        "quantum_circuits",
        safe_backend_slug(str(iqm_backend_name)),
    )


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


def _count_native_ops(ops, operation_names) -> dict[str, int]:
    if not operation_names:
        return {}
    native_names = sorted(str(name) for name in operation_names)
    return {
        name: int(ops[name])
        for name in native_names
        if int(ops.get(name, 0)) > 0
    }


def _validate_ranking_workload(ranking_workload: str) -> str:
    if ranking_workload not in RANKING_WORKLOADS:
        supported = ", ".join(sorted(RANKING_WORKLOADS))
        raise ValueError(
            f"ranking_workload must be one of: {supported}. Got {ranking_workload!r}."
        )
    return ranking_workload


def _compiled_measurement_physical_mappings(circuits) -> tuple[tuple[int, ...], ...]:
    """Return measured physical qubits in classical-bit order."""
    mappings: list[tuple[int, ...]] = []
    for circuit in circuits:
        measured: dict[int, int] = {}
        physical_by_qubit = None
        layout = getattr(circuit, "layout", None)
        initial_layout = getattr(layout, "initial_layout", None)
        get_registers = getattr(initial_layout, "get_registers", None)
        if callable(get_registers) and get_registers() == set(circuit.qregs):
            get_virtual_bits = getattr(initial_layout, "get_virtual_bits", None)
            if not callable(get_virtual_bits):
                raise ValueError("compiled measurement physical layout is invalid")
            physical_by_qubit = get_virtual_bits()
        for instruction in circuit.data:
            if (
                instruction.operation.name != "measure"
                or len(instruction.qubits) != 1
                or len(instruction.clbits) != 1
            ):
                continue
            classical = circuit.find_bit(instruction.clbits[0]).index
            qubit = instruction.qubits[0]
            if physical_by_qubit is None:
                physical = circuit.find_bit(qubit).index
            else:
                physical = physical_by_qubit.get(qubit)
                if (
                    type(physical) is not int
                    or physical < 0
                    or physical >= circuit.num_qubits
                ):
                    raise ValueError("compiled measurement physical layout is invalid")
            measured[classical] = physical
        if not measured or tuple(sorted(measured)) != tuple(range(len(measured))):
            raise ValueError(
                "compiled measurements require contiguous classical-bit mapping"
            )
        mapping = tuple(measured[index] for index in range(len(measured)))
        if len(set(mapping)) != len(mapping):
            raise ValueError(
                "compiled measurement maps multiple bits to one physical qubit"
            )
        mappings.append(mapping)
    if not mappings:
        raise ValueError("compiled measurement circuits contain no measurements")
    return tuple(mappings)


def _operation_names_from_metadata(metadata) -> list[str]:
    if not metadata:
        return []
    value = metadata.get("backend_operation_names")
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    try:
        return [str(item) for item in value]
    except TypeError:
        return []


def _strip_idle_qubits_with_indices(qc):
    no_measurements = qc.remove_final_measurements(inplace=False)
    dag = circuit_to_dag(no_measurements)
    idle_qubits = [wire for wire in dag.idle_wires() if wire in dag.qubits]
    active_original_indices = [
        no_measurements.find_bit(qubit).index
        for qubit in dag.qubits
        if qubit not in idle_qubits
    ]
    if idle_qubits:
        dag.remove_qubits(*idle_qubits)
    return dag_to_circuit(dag), active_original_indices


def _restore_stripped_input_order(
    state: np.ndarray,
    candidate_qc,
    active_indices: list[int],
    reference_qubits: int,
) -> np.ndarray:
    input_to_output = _input_to_active_output_order(
        candidate_qc,
        active_indices,
        reference_qubits,
    )
    if input_to_output is None:
        return state
    return _restore_input_qubit_order(state, input_to_output)


def _input_to_active_output_order(
    candidate_qc,
    active_indices: list[int],
    reference_qubits: int,
) -> list[int] | None:
    layout = getattr(candidate_qc, "layout", None)
    if layout is None or getattr(layout, "final_layout", None) is None:
        return None
    try:
        final_index_layout = layout.final_index_layout(filter_ancillas=True)
    except Exception:
        return None
    if len(final_index_layout) != reference_qubits:
        return None
    active_positions = {
        physical_index: idx for idx, physical_index in enumerate(active_indices)
    }
    if not all(physical_index in active_positions for physical_index in final_index_layout):
        return None
    return [
        active_positions[physical_index]
        for physical_index in final_index_layout
    ]


def _density_matrix_in_input_order(
    state: np.ndarray,
    input_to_output: list[int],
) -> np.ndarray:
    total_qubits = int(np.log2(len(state)))
    if 2**total_qubits != len(state):
        raise ValueError("state length must be a power of two")
    kept_positions = tuple(int(position) for position in input_to_output)
    if len(set(kept_positions)) != len(kept_positions):
        raise ValueError("input_to_output contains duplicate output positions")
    traced_positions = tuple(
        position for position in range(total_qubits) if position not in set(kept_positions)
    )
    dim = 2 ** len(kept_positions)
    groups: dict[int, list[tuple[int, complex]]] = {}
    for basis_index, amplitude in enumerate(np.asarray(state, dtype=complex)):
        traced_index = _bits_to_index(basis_index, traced_positions)
        kept_index = _bits_to_index(basis_index, kept_positions)
        groups.setdefault(traced_index, []).append((kept_index, amplitude))

    density = np.zeros((dim, dim), dtype=complex)
    for amplitudes in groups.values():
        vector = np.zeros(dim, dtype=complex)
        for kept_index, amplitude in amplitudes:
            vector[kept_index] = amplitude
        density += np.outer(vector, vector.conj())
    return density


def _bits_to_index(source_index: int, positions: tuple[int, ...]) -> int:
    target_index = 0
    for target_bit, source_bit in enumerate(positions):
        target_index |= ((source_index >> source_bit) & 1) << target_bit
    return target_index


def _transpile_one_trial(
    qc,
    *,
    trial: int,
    transpiler_backend=None,
    basis_gates=None,
    coupling_map=None,
    optimization_level: int = 3,
    layout_method: str | None = None,
    routing_method: str | None = None,
    approximation_degree: float | None = None,
    iqm_strategy_name: str | None = None,
):
    if transpiler_backend is None:
        transpile_kwargs = {
            "basis_gates": basis_gates,
            "coupling_map": coupling_map,
            "optimization_level": optimization_level,
            "seed_transpiler": trial,
        }
        if approximation_degree is not None:
            transpile_kwargs["approximation_degree"] = float(approximation_degree)
        return transpile(qc, **transpile_kwargs)

    if iqm_strategy_name:
        result = run_iqm_transpiler_strategy(
            iqm_strategy_name,
            qc,
            backend=transpiler_backend,
            seed_transpiler=trial,
            optimization_level=optimization_level,
        )
        if not result.success:
            error = f"{result.error_type}: {result.error_message}".strip(": ")
            raise RuntimeError(error or f"IQM strategy failed: {iqm_strategy_name}")
        return result.circuit

    from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import build_iqm_pass_manager

    pass_manager = build_iqm_pass_manager(
        backend=transpiler_backend,
        optimization_level=optimization_level,
        seed_transpiler=trial,
        layout_method=layout_method,
        routing_method=routing_method,
        approximation_degree=approximation_degree,
    )
    return pass_manager.run(qc)


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
        "best_one_qubit_gate_count": None,
        "mean_one_qubit_gate_count": None,
        "best_native_count_ops": "",
        "transpiler_backend": "manual",
        "iqm_backend_name": "",
        "iqm_use_metrics": False,
        "optimization_level": 3,
        "layout_method": None,
        "routing_method": None,
        "scheduling_method": None,
        "backend_num_qubits": None,
        "backend_operation_names": "",
        "backend_coupling_map_size": None,
        "backend_has_resonators": None,
        "backend_calibration_set_id": "",
        "iqm_transpiler_strategy": "",
        "iqm_transpiler_seed": None,
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


def logical_output_density_matrix(
    circuit,
    *,
    logical_qubit_count: int,
    max_qubits: int,
) -> tuple[DensityMatrix | None, str]:
    """Reconstruct a compiled circuit's logical output state without execution."""
    notes: list[str] = []
    if type(logical_qubit_count) is not int or logical_qubit_count < 0:
        return None, "Logical state reconstruction requires a non-negative logical_qubit_count."
    if type(max_qubits) is not int or max_qubits < 0:
        return None, "Logical state reconstruction requires a non-negative max_qubits."
    if any(instruction.operation.name == "measure" for instruction in circuit.data):
        return None, "Fidelity skipped because circuit contains measurement operations."

    fidelity_qc = circuit
    active_indices = list(range(circuit.num_qubits))
    if fidelity_qc.num_qubits > int(max_qubits):
        stripped_qc, stripped_active_indices = _strip_idle_qubits_with_indices(fidelity_qc)
        if stripped_qc.num_qubits < fidelity_qc.num_qubits:
            notes.append(
                f"Idle qubits stripped for fidelity: {fidelity_qc.num_qubits}->{stripped_qc.num_qubits}."
            )
            fidelity_qc = stripped_qc
            active_indices = stripped_active_indices
    if fidelity_qc.num_qubits > int(max_qubits):
        notes.append(f"Fidelity skipped because transpiled circuit has {fidelity_qc.num_qubits} qubits.")
        return None, " ".join(notes)
    if fidelity_qc.num_qubits < logical_qubit_count:
        notes.append(
            f"Fidelity skipped because active qubit count {fidelity_qc.num_qubits} "
            f"differs from reference {logical_qubit_count}."
        )
        return None, " ".join(notes)

    try:
        candidate = Statevector.from_instruction(fidelity_qc)
        logical_dims = (2,) * logical_qubit_count
        if fidelity_qc.num_qubits > logical_qubit_count:
            input_to_output = _input_to_active_output_order(
                circuit,
                active_indices,
                logical_qubit_count,
            )
            if input_to_output is None:
                notes.append(
                    f"Fidelity skipped because active qubit count {fidelity_qc.num_qubits} "
                    f"differs from reference {logical_qubit_count}."
                )
                return None, " ".join(notes)
            candidate_density = DensityMatrix(
                _density_matrix_in_input_order(candidate.data, input_to_output),
                dims=logical_dims,
            )
            notes.append(
                "Extra active qubits traced for fidelity: "
                f"{fidelity_qc.num_qubits}->{logical_qubit_count}."
            )
            return candidate_density, " ".join(notes)

        candidate_data = candidate.data
        if fidelity_qc is circuit:
            layout = getattr(circuit, "layout", None)
            if layout is not None:
                final_index_layout = layout.final_index_layout(filter_ancillas=True)
                valid_permutation = (
                    len(final_index_layout) == logical_qubit_count == circuit.num_qubits
                    and all(
                        isinstance(index, (int, np.integer))
                        and not isinstance(index, (bool, np.bool_))
                        for index in final_index_layout
                    )
                    and set(final_index_layout) == set(range(logical_qubit_count))
                )
                if valid_permutation:
                    candidate_data = _restore_input_qubit_order(candidate_data, final_index_layout)
        else:
            candidate_data = _restore_stripped_input_order(
                candidate_data,
                circuit,
                active_indices,
                logical_qubit_count,
            )
        candidate_state = Statevector(candidate_data, dims=logical_dims)
        return DensityMatrix(candidate_state), " ".join(notes)
    except Exception as exc:
        notes.append(f"Fidelity skipped: {exc}")
        return None, " ".join(notes)


def _safe_fidelity(reference_qc, candidate_qc, *, max_qubits: int) -> tuple[float | None, str]:
    candidate_state, notes = logical_output_density_matrix(
        candidate_qc,
        logical_qubit_count=reference_qc.num_qubits,
        max_qubits=max_qubits,
    )
    if candidate_state is None:
        return None, notes
    try:
        reference = Statevector.from_instruction(reference_qc)
        return float(state_fidelity(reference, candidate_state)), notes
    except Exception as exc:
        return None, " ".join(part for part in (notes, f"Fidelity skipped: {exc}") if part)


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
    transpiler_backend=None,
    transpiler_metadata: dict | None = None,
    optimization_level: int = 3,
    layout_method: str | None = None,
    routing_method: str | None = None,
    iqm_strategy_names: Iterable[str] | None = None,
    ranking_workload: str = "state_preparation",
) -> dict:
    """Benchmark graph-state preparation using direct W-defined encoding."""
    ranking_workload = _validate_ranking_workload(ranking_workload)
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
    if transpiler_metadata:
        row.update(transpiler_metadata)
    elif transpiler_backend is not None:
        row["transpiler_backend"] = "iqm"
    row["optimization_level"] = int(optimization_level)
    row["layout_method"] = layout_method
    row["routing_method"] = routing_method
    row["ranking_workload"] = ranking_workload

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
        measured_circuits = None
        measured_settings = None
        if ranking_workload == "bell_measurements":
            measured_circuits, measured_metadata = build_sampler_circuits_for_candidate(
                state_name,
                qc,
                encoding_embedding(basis_matrix),
            )
            measured_circuits = tuple(measured_circuits)
            measured_settings = tuple(
                measured_metadata["setting_by_circuit_index"]
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

    if transpiler_backend is None:
        basis_gates = basis_gates or BASIS_GATES
        coupling_map = coupling_map if coupling_map is not None else COUPLING_MAP
        native_operation_names = []
    else:
        basis_gates = None
        coupling_map = None
        native_operation_names = _operation_names_from_metadata(row)
    depths: list[int] = []
    sizes: list[int] = []
    twoq_counts: list[int] = []
    oneq_counts: list[int] = []
    successful = []
    iqm_strategy_names = tuple(iqm_strategy_names or ())

    last_trial_error = ""
    for trial in range(int(n_transpile_runs)):
        strategy_names = (
            iqm_strategy_names
            if transpiler_backend is not None and iqm_strategy_names
            else (None,)
        )
        for strategy_index, iqm_strategy_name in enumerate(strategy_names):
            try:
                def transpile_trial(circuit):
                    return _transpile_one_trial(
                        circuit,
                        trial=trial,
                        transpiler_backend=transpiler_backend,
                        basis_gates=basis_gates,
                        coupling_map=coupling_map,
                        optimization_level=int(optimization_level),
                        layout_method=layout_method,
                        routing_method=routing_method,
                        approximation_degree=approximation_degree,
                        iqm_strategy_name=iqm_strategy_name,
                    )

                qc_t = transpile_trial(qc)
                workload_metrics = None
                if measured_circuits is not None:
                    compiled_measured = tuple(
                        transpile_trial(circuit) for circuit in measured_circuits
                    )
                    physical_mappings = _compiled_measurement_physical_mappings(
                        compiled_measured
                    )
                    requested_physical_qubits = tuple(
                        sorted(
                            {
                                physical
                                for mapping in physical_mappings
                                for physical in mapping
                            }
                        )
                    )
                    workload_metrics = summarize_compiled_workload(
                        compiled_measured,
                        settings=measured_settings,
                        physical_mappings=physical_mappings,
                        requested_physical_qubits=requested_physical_qubits,
                    )
                ops = qc_t.count_ops()
                depth = int(qc_t.depth())
                size = int(qc_t.size())
                twoq = _count_two_qubit_gates_from_ops(ops)
                oneq = _count_one_qubit_gates(qc_t)
                native_ops = _count_native_ops(ops, native_operation_names)

                if workload_metrics is None:
                    rank_key = (
                        (depth, twoq, oneq, size)
                        if transpiler_backend is not None
                        else (depth, twoq, size)
                    )
                else:
                    aggregate = workload_metrics.aggregate
                    rank_key = (
                        aggregate["maximum_two_qubit_gate_count"],
                        aggregate["total_two_qubit_gate_count"],
                        aggregate["maximum_depth"],
                        aggregate["total_depth"],
                        aggregate["maximum_size"],
                        aggregate["total_size"],
                        trial,
                        strategy_index,
                    )

                depths.append(depth)
                sizes.append(size)
                twoq_counts.append(twoq)
                oneq_counts.append(oneq)
                row["successful_trials"] += 1
                successful.append(
                    {
                        "rank_key": rank_key,
                        "depth": depth,
                        "size": size,
                        "twoq": twoq,
                        "oneq": oneq,
                        "ops": dict(ops),
                        "native_ops": native_ops,
                        "num_qubits": qc_t.num_qubits,
                        "qc": qc_t,
                        "iqm_strategy_name": iqm_strategy_name or "",
                        "seed_transpiler": trial,
                        "workload_metrics": workload_metrics,
                    }
                )
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except BaseException as exc:
                # Catch BaseException (not just Exception) because Qiskit's Rust
                # backend can raise pyo3_runtime.PanicException (e.g. the known
                # TwoQubitWeylDecomposition bug, qiskit-terra issue #4159) which
                # does not inherit from Exception and would otherwise crash the
                # entire benchmark run instead of being treated as a failed trial.
                row["failed_trials"] += 1
                strategy_note = (
                    f" strategy={iqm_strategy_name}"
                    if iqm_strategy_name
                    else ""
                )
                last_trial_error = f"{type(exc).__name__}: {exc}".splitlines()[0]
                print(
                    f"  transpile trial {trial}{strategy_note} failed: {last_trial_error}",
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
    row["best_one_qubit_gate_count"] = best["oneq"]
    row["mean_one_qubit_gate_count"] = round(float(np.mean(oneq_counts)), 2)
    row["best_native_count_ops"] = json.dumps(best.get("native_ops", {}), sort_keys=True)
    row["num_qubits"] = best["num_qubits"]
    row["best_count_ops"] = best["ops"]
    row["iqm_transpiler_strategy"] = best.get("iqm_strategy_name", "")
    row["iqm_transpiler_seed"] = best.get("seed_transpiler")
    if best["workload_metrics"] is not None:
        aggregate = best["workload_metrics"].aggregate
        row["workload_circuit_count"] = aggregate["circuit_count"]
        row["workload_max_depth"] = aggregate["maximum_depth"]
        row["workload_total_depth"] = aggregate["total_depth"]
        row["workload_max_two_qubit_gate_count"] = aggregate[
            "maximum_two_qubit_gate_count"
        ]
        row["workload_total_two_qubit_gate_count"] = aggregate[
            "total_two_qubit_gate_count"
        ]
        row["workload_max_size"] = aggregate["maximum_size"]
        row["workload_total_size"] = aggregate["total_size"]

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
    ranking_workload: str = "state_preparation",
) -> dict:
    ranking_workload = _validate_ranking_workload(ranking_workload)
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
    row["ranking_workload"] = ranking_workload
    return row


def _attach_transpiler_run_metadata(
    row: dict,
    *,
    transpiler_backend=None,
    transpiler_metadata: dict | None = None,
    optimization_level: int = 3,
    layout_method: str | None = None,
    routing_method: str | None = None,
    iqm_strategy_names: Iterable[str] | None = None,
) -> dict:
    if transpiler_metadata:
        row.update(transpiler_metadata)
    elif transpiler_backend is not None:
        row["transpiler_backend"] = "iqm"
    row["optimization_level"] = int(optimization_level)
    row["layout_method"] = layout_method
    row["routing_method"] = routing_method
    return row


def _print_candidate_row_result(row: dict) -> None:
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


def _benchmark_direct_basis_candidate_group(
    *,
    state_name: str,
    candidate_index: int,
    total_candidates: int,
    candidate: DirectBasisCandidate,
    run_specs: list[tuple[float | None, str, bool]],
    n_qutrits: int | None,
    coupling_map=None,
    basis_gates=None,
    n_transpile_runs: int = 20,
    compute_fidelity: bool = True,
    max_fidelity_qubits: int = 10,
    quantum_circuits_dir: str | None = None,
    transpiler_backend=None,
    transpiler_metadata: dict | None = None,
    optimization_level: int = 3,
    layout_method: str | None = None,
    routing_method: str | None = None,
    iqm_strategy_names: Iterable[str] | None = None,
    ranking_workload: str = "state_preparation",
) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    for spec_index, (
        approximation_degree,
        run_label,
        legacy_exact_transpiled_filename,
    ) in enumerate(run_specs):
        print(
            f"[direct_basis_encoding] {candidate_index}/{total_candidates} "
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
                ranking_workload=ranking_workload,
            )
            _attach_transpiler_run_metadata(
                row,
                transpiler_backend=transpiler_backend,
                transpiler_metadata=transpiler_metadata,
                optimization_level=optimization_level,
                layout_method=layout_method,
                routing_method=routing_method,
                iqm_strategy_names=iqm_strategy_names,
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
                transpiler_backend=transpiler_backend,
                transpiler_metadata=transpiler_metadata,
                optimization_level=optimization_level,
                layout_method=layout_method,
                routing_method=routing_method,
                iqm_strategy_names=iqm_strategy_names,
                ranking_workload=ranking_workload,
            )

        _print_candidate_row_result(row)
        row_index = (candidate_index - 1) * len(run_specs) + spec_index
        rows.append((row_index, row))
    return rows


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
    transpiler_backend=None,
    transpiler_metadata: dict | None = None,
    optimization_level: int = 3,
    layout_method: str | None = None,
    routing_method: str | None = None,
    iqm_strategy_names: Iterable[str] | None = None,
    ranking_workload: str = "state_preparation",
    jobs: int = 1,
) -> tuple[pd.DataFrame, str | None]:
    ranking_workload = _validate_ranking_workload(ranking_workload)
    candidates = list(candidates)
    run_specs = _direct_basis_run_specs(approximation_degrees)
    jobs = max(int(jobs or 1), 1)
    if transpiler_backend is not None and jobs > 1:
        print(
            "[direct_basis_encoding] IQM transpiler backend is not thread-safe; "
            "running candidate jobs serially.",
            flush=True,
        )
        jobs = 1
    row_slots: list[dict | None] = [None] * (len(candidates) * len(run_specs))

    def store_group(group_rows: list[tuple[int, dict]]) -> None:
        for row_index, row in group_rows:
            row_slots[row_index] = row

    group_kwargs = {
        "state_name": state_name,
        "total_candidates": len(candidates),
        "run_specs": run_specs,
        "n_qutrits": n_qutrits,
        "coupling_map": coupling_map,
        "basis_gates": basis_gates,
        "n_transpile_runs": n_transpile_runs,
        "compute_fidelity": compute_fidelity,
        "max_fidelity_qubits": max_fidelity_qubits,
        "quantum_circuits_dir": quantum_circuits_dir,
        "transpiler_backend": transpiler_backend,
        "transpiler_metadata": transpiler_metadata,
        "optimization_level": optimization_level,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "iqm_strategy_names": iqm_strategy_names,
        "ranking_workload": ranking_workload,
    }

    def run_group(index: int, candidate: DirectBasisCandidate) -> list[tuple[int, dict]]:
        return _benchmark_direct_basis_candidate_group(
            candidate_index=index,
            candidate=candidate,
            **group_kwargs,
        )

    if jobs <= 1 or len(candidates) <= 1:
        for index, candidate in enumerate(candidates, start=1):
            store_group(run_group(index, candidate))
    else:
        print(
            f"[direct_basis_encoding] parallel candidate jobs={jobs}",
            flush=True,
        )
        max_workers = min(jobs, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_group, index, candidate)
                for index, candidate in enumerate(candidates, start=1)
            ]
            for future in as_completed(futures):
                store_group(future.result())

    rows = [row for row in row_slots if row is not None]

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
    is_w_matrix = basis_matrix.shape == (3, 3)
    e_npy = os.path.join(output_dir, "E.npy")
    w_npy = os.path.join(output_dir, "W.npy") if is_w_matrix else ""
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
        "basis_change_matrix_npy": w_npy or e_npy,
        "E_npy": e_npy,
        "W_npy": w_npy,
    }
    _save_qpy(f3_circuit, paths["f3_w_qpy"])
    _save_qpy(cz_circuit, paths["cz3_w_qpy"])
    _save_qpy(graph_state_circuit, paths["graph_state_qpy"])
    if is_w_matrix:
        _save_npy(basis_matrix, paths["W_npy"])
        _save_npy(encoding_embedding(basis_matrix), paths["E_npy"])
    else:
        _save_npy(basis_matrix, paths["E_npy"])
    if is_w_matrix and is_unitary(basis_matrix):
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
