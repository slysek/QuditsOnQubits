import argparse
import os
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from QuditsOnQubits.benchmark_encoding_bases import (
    ALL_CLASS_NAMES,
    BASIS_GATES,
    COUPLING_MAP,
    DEFAULT_FIDELITY_THRESHOLDS,
    benchmark_basis,
    generate_baseline,
    generate_clifford_wh_bases,
    generate_entangling_isometries,
    generate_finer_structured_grid,
    generate_fourier_like_bases,
    generate_haar_random_isometries,
    generate_householder_bases,
    generate_local_general_su2,
    generate_local_ry_only,
    generate_monomial_full_bases,
    generate_monomial_old_codespace_bases,
    generate_near_identity_isometries,
    generate_product_bases,
    generate_real_orthogonal_isometries,
    generate_structured_entangling_isometries,
    generate_two_cz_ansatz,
    generate_perturbed_isometries,
    write_multi_state_benchmark_report,
)
from QuditsOnQubits.benchmark_encoding_bases import (
    _DEFAULT_CIRCUITS_OUTPUT_DIR,
    _filter_candidates_by_preselection,
    _load_preselected_candidates,
    _print_single_state_summary,
    _resolve_circuits_output_dir,
    _save_top3_fidelity_circuits,
    _save_top3_per_class_csvs,
    _validate_preselection_coverage,
    _write_topk_tables_to_output_dir,
)
from QuditsOnQubits.create_ame_circuit import VALID_ENCODING_STRATEGIES
from QuditsOnQubits.project_paths import (
    benchmark_circuits_dir,
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
    prepared_w_benchmark_data_dir,
    prepared_w_benchmark_results_path,
)


def _workspace_tempdir():
    root = os.path.join(tempfile.gettempdir(), "qudits_parallel_benchmark_tests")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"bench_test_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=False)
    return path


def _normalize_class_filter(class_filter):
    if class_filter is None:
        return None
    if isinstance(class_filter, str):
        return {c.strip() for c in class_filter.split(",") if c.strip()}
    return set(class_filter)


def _generate_candidates_for_mode(mode, class_filter=None):
    candidates = []

    if mode in ("full", "original"):
        candidates += generate_baseline()
        candidates += generate_monomial_old_codespace_bases(max_candidates=None)
        candidates += generate_monomial_full_bases(max_candidates=None)
        candidates += generate_fourier_like_bases(max_candidates=80)
        candidates += generate_householder_bases(n_samples=20, seed=42)
        candidates += generate_clifford_wh_bases()
        candidates += generate_haar_random_isometries(n_samples=20, seed=100)
        candidates += generate_perturbed_isometries(n_samples_per_eps=8, seed=200)
        candidates += generate_entangling_isometries(n_samples=20, seed=300)
        candidates += generate_structured_entangling_isometries()

    if mode in ("full", "extended"):
        candidates += generate_product_bases(max_candidates=None)
        candidates += generate_local_ry_only(n_grid=10)
        candidates += generate_local_general_su2(n_samples=30, seed=600)
        candidates += generate_real_orthogonal_isometries(n_samples=20, seed=400)
        candidates += generate_near_identity_isometries(n_samples_per_eps=10, seed=500)
        candidates += generate_finer_structured_grid()
        candidates += generate_two_cz_ansatz(n_samples=50, seed=700)

    filter_set = _normalize_class_filter(class_filter)
    if filter_set:
        candidates = [(cls, name, e_new) for cls, name, e_new in candidates if cls in filter_set]

    return candidates


def _normalize_circuits_output_root(circuits_output_dir):
    if circuits_output_dir is _DEFAULT_CIRCUITS_OUTPUT_DIR:
        return benchmark_circuits_dir()
    return circuits_output_dir


def _build_candidate_tasks(
    candidates,
    state_name,
    n_transpile_runs,
    circuits_output_dir,
    approximation_values,
    fidelity_thresholds,
    approximation_seed,
    encoding_strategy,
):
    output_root = _normalize_circuits_output_root(circuits_output_dir)
    return [
        {
            "E_new": e_new,
            "class_name": cls,
            "candidate_name": name,
            "state_name": state_name,
            "coupling_map": COUPLING_MAP,
            "basis_gates": BASIS_GATES,
            "n_transpile_runs": n_transpile_runs,
            "circuits_output_dir": output_root,
            "approximation_values": approximation_values,
            "fidelity_thresholds": fidelity_thresholds,
            "approximation_seed": approximation_seed,
            "encoding_strategy": encoding_strategy,
        }
        for cls, name, e_new in candidates
    ]


def _benchmark_candidate_worker(task):
    return benchmark_basis(
        task["E_new"],
        task["class_name"],
        task["candidate_name"],
        state_name=task["state_name"],
        coupling_map=task["coupling_map"],
        basis_gates=task["basis_gates"],
        n_transpile_runs=task["n_transpile_runs"],
        circuits_output_dir=task["circuits_output_dir"],
        approximation_values=task["approximation_values"],
        fidelity_thresholds=task["fidelity_thresholds"],
        approximation_seed=task["approximation_seed"],
        encoding_strategy=task["encoding_strategy"],
    )


def _extract_fidelity_circuits_from_worker_row(row, class_name, candidate_name):
    fidelity_circuits = []
    for key in list(row.keys()):
        if key.startswith("_fid") and key.endswith("_best_qc"):
            label = key[1:].replace("_best_qc", "")
            fidelity_circuits.append({
                "class_name": class_name,
                "candidate_name": candidate_name,
                "label": label,
                "two_q": row.get(f"{label}_best_two_qubit_gate_count"),
                "depth": row.get(f"{label}_best_depth"),
                "qc": row.pop(key),
            })
    return fidelity_circuits


def _format_progress_suffix(row):
    if row["status"] == "ok":
        code_space = "old" if row["uses_old_codespace_only"] else "NEW"
        return (
            f"best_d={row['best_depth']:5d}  "
            f"mean_d={row['mean_depth']:7.1f}  "
            f"best_2q={row['best_two_qubit_gate_count']:5d}  "
            f"ent={row['avg_codeword_entanglement']:.3f}  "
            f"[{code_space}]"
        )
    return f"[{row['status']}]"


def _print_progress_update(index, total, class_name, candidate_name, row):
    print(
        f"  [{index:4d}/{total}]  {class_name:28s}  {candidate_name:30s}  "
        f"{_format_progress_suffix(row)}",
        flush=True,
    )


def _run_tasks_in_pool(tasks, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_benchmark_candidate_worker, task): (idx, task)
            for idx, task in enumerate(tasks)
        }
        for completed_index, future in enumerate(as_completed(future_to_task), start=1):
            task_index, task = future_to_task[future]
            yield completed_index, task_index, task, future.result()


def run_single_state_parallel_benchmark(
    state_name="ghz3",
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    class_filter=None,
    encoding_strategy="append_w",
    max_workers=None,
):
    filter_set = _normalize_class_filter(class_filter)
    filter_label = ",".join(sorted(filter_set)) if filter_set else "all"
    print("=" * 80)
    print(
        f"  Parallel benchmark baz kodowania qutrytu  "
        f"[state={state_name}]  [mode={mode}]  [class={filter_label}]"
    )
    print(
        f"  Transpilacja: {n_transpile_runs} prob na kandydata  "
        f"[workers={max_workers or os.cpu_count()}]"
    )
    print("=" * 80)

    if csv_path is None:
        csv_path = benchmark_state_results_path(state_name, mode)

    all_candidates = _generate_candidates_for_mode(mode, class_filter=filter_set)
    print(f"\n  Razem kandydatow:            {len(all_candidates)}")
    print("-" * 80)

    tasks = _build_candidate_tasks(
        candidates=all_candidates,
        state_name=state_name,
        n_transpile_runs=n_transpile_runs,
        circuits_output_dir=circuits_output_dir,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=approximation_seed,
        encoding_strategy=encoding_strategy,
    )

    results = []
    fidelity_circuits = []
    t0 = time.time()
    ordered_rows = [None] * len(tasks)

    for completed_index, task_index, task, row in _run_tasks_in_pool(tasks, max_workers=max_workers):
        fidelity_circuits.extend(
            _extract_fidelity_circuits_from_worker_row(
                row, task["class_name"], task["candidate_name"]
            )
        )
        ordered_rows[task_index] = row
        _print_progress_update(
            index=completed_index,
            total=len(tasks),
            class_name=task["class_name"],
            candidate_name=task["candidate_name"],
            row=row,
        )

    results = ordered_rows

    elapsed = time.time() - t0
    print(f"\nCzas benchmarku [{state_name}, parallel]: {elapsed:.1f} s")

    df = pd.DataFrame(results)

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wyniki zapisane do: {csv_path}")

    _save_top3_per_class_csvs(df, csv_path, fidelity_thresholds=fidelity_thresholds)

    resolved_circuits_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)
    if resolved_circuits_dir is not None and fidelity_circuits:
        _save_top3_fidelity_circuits(fidelity_circuits, resolved_circuits_dir)

    _print_single_state_summary(df, f"{state_name} [parallel]")

    return df, csv_path


def run_prepared_w_parallel_benchmark(
    state_name="ghz3",
    n_transpile_runs=20,
    csv_path=None,
    mode="full",
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    class_filter=None,
    preselected_candidates_file=None,
    output_dir=None,
    max_workers=None,
):
    if preselected_candidates_file is None:
        raise ValueError(
            "Tryb 'prepared_w_then_conjugated_entanglers' wymaga podania "
            "--preselected-candidates-file."
        )

    preselected_set = _load_preselected_candidates(preselected_candidates_file)
    all_candidates = _generate_candidates_for_mode(mode, class_filter=class_filter)
    filtered = _filter_candidates_by_preselection(all_candidates, preselected_set)
    _validate_preselection_coverage(preselected_set, filtered, preselected_candidates_file)

    if not filtered:
        print("  UWAGA: po preselekcji nie zostal zaden kandydat.")
        return pd.DataFrame(), None

    tasks = _build_candidate_tasks(
        candidates=filtered,
        state_name=state_name,
        n_transpile_runs=n_transpile_runs,
        circuits_output_dir=circuits_output_dir,
        approximation_values=approximation_values,
        fidelity_thresholds=fidelity_thresholds,
        approximation_seed=approximation_seed,
        encoding_strategy="prepared_w_then_conjugated_entanglers",
    )

    results = []
    fidelity_circuits = []
    t0 = time.time()
    ordered_rows = [None] * len(tasks)
    for completed_index, task_index, task, row in _run_tasks_in_pool(tasks, max_workers=max_workers):
        fidelity_circuits.extend(
            _extract_fidelity_circuits_from_worker_row(
                row, task["class_name"], task["candidate_name"]
            )
        )
        ordered_rows[task_index] = row
        _print_progress_update(
            index=completed_index,
            total=len(tasks),
            class_name=task["class_name"],
            candidate_name=task["candidate_name"],
            row=row,
        )

    results = ordered_rows

    elapsed = time.time() - t0
    print(f"\nCzas benchmarku [prepared_w, {state_name}, parallel]: {elapsed:.1f} s")

    df = pd.DataFrame(results)

    if output_dir is None:
        output_dir = prepared_w_benchmark_data_dir()
    os.makedirs(output_dir, exist_ok=True)

    if csv_path is None:
        csv_path = prepared_w_benchmark_results_path(state_name, mode)

    _write_topk_tables_to_output_dir(
        df,
        output_dir,
        file_prefix=f"benchmark_prepared_w_{state_name}_{mode}",
        fidelity_thresholds=fidelity_thresholds,
    )

    resolved_circuits_dir = _resolve_circuits_output_dir(state_name, circuits_output_dir)
    if resolved_circuits_dir is not None and fidelity_circuits:
        _save_top3_fidelity_circuits(fidelity_circuits, resolved_circuits_dir)

    _print_single_state_summary(df, f"{state_name} [prepared_w parallel]")

    return df, csv_path


def run_parallel_benchmark(
    mode="full",
    state_name="ghz3",
    n_transpile_runs=20,
    csv_path=None,
    circuits_output_dir=_DEFAULT_CIRCUITS_OUTPUT_DIR,
    approximation_values=None,
    fidelity_thresholds=DEFAULT_FIDELITY_THRESHOLDS,
    approximation_seed=0,
    class_filter=None,
    encoding_strategy="append_w",
    preselected_candidates_file=None,
    output_dir=None,
    max_workers=None,
    reuse_existing_ghz3=True,
    combined_report_path=None,
):
    if encoding_strategy == "prepared_w_then_conjugated_entanglers":
        return run_prepared_w_parallel_benchmark(
            state_name=state_name,
            n_transpile_runs=n_transpile_runs,
            csv_path=csv_path,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            class_filter=class_filter,
            preselected_candidates_file=preselected_candidates_file,
            output_dir=output_dir,
            max_workers=max_workers,
        )

    if state_name != "all":
        return run_single_state_parallel_benchmark(
            state_name=state_name,
            n_transpile_runs=n_transpile_runs,
            csv_path=csv_path,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            class_filter=class_filter,
            encoding_strategy=encoding_strategy,
            max_workers=max_workers,
        )

    state_frames = {}
    for single_state in ("two_qutrit", "ghz3", "ame43"):
        if single_state == "ghz3" and reuse_existing_ghz3:
            ghz_csv = benchmark_state_results_path("ghz3", mode)
            if os.path.exists(ghz_csv):
                state_frames["ghz3"] = pd.read_csv(ghz_csv)
                continue
        state_frames[single_state], _ = run_single_state_parallel_benchmark(
            state_name=single_state,
            n_transpile_runs=n_transpile_runs,
            mode=mode,
            circuits_output_dir=circuits_output_dir,
            approximation_values=approximation_values,
            fidelity_thresholds=fidelity_thresholds,
            approximation_seed=approximation_seed,
            class_filter=class_filter,
            encoding_strategy=encoding_strategy,
            max_workers=max_workers,
        )

    report_path = combined_report_path or multi_state_benchmark_report_path()
    write_multi_state_benchmark_report(state_frames, report_path)
    print(f"\nRaport markdown zapisany do: {report_path}")
    return state_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel benchmark encoding bases for qutrit states",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="extended",
        choices=["full", "original", "extended"],
        help="which candidate generators to use (default: extended)",
    )
    parser.add_argument(
        "state",
        nargs="?",
        default="ghz3",
        choices=["ghz3", "two_qutrit", "ame43", "all"],
        help="which state to benchmark (default: ghz3)",
    )
    parser.add_argument(
        "--class",
        dest="class_filter",
        default=None,
        help=(
            "comma-separated list of class names to benchmark. "
            f"Available: {', '.join(ALL_CLASS_NAMES)}"
        ),
    )
    parser.add_argument(
        "--encoding-strategy",
        dest="encoding_strategy",
        default="append_w",
        choices=list(VALID_ENCODING_STRATEGIES),
        help="Circuit build strategy.",
    )
    parser.add_argument(
        "--preselected-candidates-file",
        dest="preselected_candidates_file",
        default=None,
        help="CSV path required for prepared_w benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Custom output directory for prepared_w results.",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: os.cpu_count()",
    )
    parser.add_argument(
        "--no-circuit-export",
        dest="no_circuit_export",
        action="store_true",
        help="Disable QPY circuit export for faster parallel runs.",
    )
    args = parser.parse_args()

    circuits_output_dir = None if args.no_circuit_export else _DEFAULT_CIRCUITS_OUTPUT_DIR
    csv = None if args.state == "all" else benchmark_state_results_path(args.state, args.mode)
    run_parallel_benchmark(
        mode=args.mode,
        state_name=args.state,
        csv_path=csv,
        circuits_output_dir=circuits_output_dir,
        class_filter=args.class_filter,
        encoding_strategy=args.encoding_strategy,
        preselected_candidates_file=args.preselected_candidates_file,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
