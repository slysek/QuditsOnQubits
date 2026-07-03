"""Jednoetapowy pipeline benchmarkowy "suite".

Ten modul jest celowo wezszy niz :mod:`encoding_search_v2.runner`.
W :mod:`runner` zyje stary mechanizm dwuetapowy z preselekcja top-k.
Tu — w odpowiedzi na nocny benchmark dla rodzin grafow qutrytowych
(``ghz<n>``, ``path<n>``, ``cycle<n>``, ``wheel<n>``, ``complete<n>``,
``cluster<r>x<c>``) — chcemy:

* odpalic *jedna* komenda,
* zbenchmarkowac *kazdy* zarejestrowany stan ze sluga "suite",
* uzyc *wszystkich* klas baz kodowania zaimplementowanych w
  :mod:`QuditsOnQubits.benchmark_encoding_bases` (``mode="full"``),
* zapisac wyniki do osobnego folderu suite, rownolegle w wielu workerach,
* renderowac czytelny progres ("[done/total] ETA...") nadajacy sie do logu
  uruchomienia nocnego.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd

from qudits_on_qubits.encoding_search.paths import default_results_root
from qudits_on_qubits.encoding_search.results import write_result_bundle
from qudits_on_qubits.encoding_search.runner import (
    DEFAULT_FIDELITY_THRESHOLDS,
    _BenchmarkProgressReporter,
    _format_duration,
    _strip_internal_circuit_objects,
    benchmark_basis,
)
from qudits_on_qubits.encoding_search.states import (
    BenchmarkStateSpec,
    resolve_benchmark_state,
)
from qudits_on_qubits.encoding_search.triviality import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    _filter_trivial_candidates,
    candidate_metadata_fields,
)


SUITE_FILE_PREFIX = "encoding_search_v2_suite"
COMBINED_RESULTS_BASENAME = "suite_combined_results.csv"
SUITE_LOG_FILENAME = "suite_run.log"


# ───────────────────────── konfiguracja ────────────────────────


@dataclass(frozen=True)
class SuiteConfig:
    """Konfiguracja jednorazowego nocnego runa benchmarkowego."""

    suite_name: str
    states: tuple[str, ...]
    jobs: int = 1
    output_root: Optional[str] = None
    n_transpile_runs: int = 20
    candidate_mode: str = "full"
    class_filter: Optional[Sequence[str]] = None
    encoding_strategy: str = "append_w"
    approximation_values: Optional[Sequence[float]] = None
    fidelity_thresholds: Sequence[float] = DEFAULT_FIDELITY_THRESHOLDS
    approximation_seed: int = 0
    export_circuits: bool = False
    skip_existing: bool = False
    top_k: int = 30
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL


# ─────────── kandydaci: pelny zestaw klas baz kodowania ────────


def generate_all_class_candidates(
    mode: str = "full",
    class_filter: Optional[Iterable[str]] = None,
) -> list[tuple]:
    """Zwroc liste (class_name, candidate_name, E_new) dla wszystkich klas.

    Lazy-import :mod:`QuditsOnQubits.benchmark_encoding_bases_parallel` zeby
    uniknac ladowania ciezkich zaleznosci (qiskit) podczas importu modulu
    suite.
    """
    from qudits_on_qubits.core.benchmark_encoding_bases_parallel import (
        _generate_candidates_for_mode,
    )

    return _generate_candidates_for_mode(mode, class_filter=class_filter)


# ─────────── nizsze warstwy: budowa zadan i workery ────────────


def _strip_e_old_baseline_only_from_candidates(candidates):
    """Pozostaw kandydatow w ich naturalnej kolejnosci — bez modyfikacji.

    Zachowane jako wyrazny punkt wejscia, gdyby w przyszlosci suite
    chcial filtrowac np. baseline z innej rodziny.
    """
    return list(candidates)


def _suite_state_output_dir(config: SuiteConfig, state_id: str) -> str:
    root = config.output_root or default_results_root()
    return os.path.join(root, config.suite_name, state_id)


def _suite_combined_results_path(config: SuiteConfig) -> str:
    root = config.output_root or default_results_root()
    return os.path.join(root, config.suite_name, COMBINED_RESULTS_BASENAME)


def _suite_log_path(config: SuiteConfig) -> str:
    root = config.output_root or default_results_root()
    return os.path.join(root, config.suite_name, SUITE_LOG_FILENAME)


def _state_circuits_output_dir(config: SuiteConfig, state_id: str) -> Optional[str]:
    if not config.export_circuits:
        return None
    return os.path.join(_suite_state_output_dir(config, state_id), "circuits")


def _build_state_tasks(
    state_spec: BenchmarkStateSpec,
    benchmarked: list[tuple],
    config: SuiteConfig,
) -> list[dict]:
    circuits_output_dir = _state_circuits_output_dir(config, state_spec.state_id)
    return [
        {
            "E_new": e_new,
            "class_name": class_name,
            "candidate_name": candidate_name,
            "state_name": state_spec.state_id,
            "state_family": state_spec.state_family,
            "n_qutrits": state_spec.num_qutrits,
            "coupling_map": None,
            "basis_gates": None,
            "n_transpile_runs": config.n_transpile_runs,
            "circuits_output_dir": circuits_output_dir,
            "approximation_values": (
                tuple(config.approximation_values)
                if config.approximation_values is not None
                else None
            ),
            "fidelity_thresholds": tuple(config.fidelity_thresholds),
            "approximation_seed": config.approximation_seed,
            "encoding_strategy": config.encoding_strategy,
            "candidate_metadata": candidate_metadata_fields(
                class_name,
                candidate_name,
                e_new,
                atol=config.atol,
                rtol=config.rtol,
            ),
            "suite_name": config.suite_name,
        }
        for class_name, candidate_name, e_new in benchmarked
    ]


def _benchmark_candidate_worker(task: dict) -> dict:
    row = benchmark_basis(
        task["E_new"],
        task["class_name"],
        task["candidate_name"],
        state_name=task["state_name"],
        n_qutrits=task["n_qutrits"],
        coupling_map=task["coupling_map"],
        basis_gates=task["basis_gates"],
        n_transpile_runs=task["n_transpile_runs"],
        circuits_output_dir=task["circuits_output_dir"],
        approximation_values=task["approximation_values"],
        fidelity_thresholds=task["fidelity_thresholds"],
        approximation_seed=task["approximation_seed"],
        encoding_strategy=task["encoding_strategy"],
    )
    row["state_name"] = task["state_name"]
    row["state_family"] = task["state_family"]
    row["n_qutrits"] = task["n_qutrits"]
    row["suite_name"] = task["suite_name"]
    row.update(task["candidate_metadata"])
    return _strip_internal_circuit_objects(row)


# ─────────── stream-tee: jednoczesny stdout + plik logu ───────


class _TeeStream:
    """Lekki tee: rownolegle pisze do dwoch strumieni."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


# ─────────── glowna petla benchmarku po stanach ────────────────


def _state_results_already_exist(config: SuiteConfig, state_id: str) -> bool:
    state_dir = _suite_state_output_dir(config, state_id)
    main_csv = os.path.join(
        state_dir,
        f"{SUITE_FILE_PREFIX}_{state_id}_results.csv",
    )
    return os.path.isfile(main_csv) and os.path.getsize(main_csv) > 0


def _run_state(
    config: SuiteConfig,
    state_spec: BenchmarkStateSpec,
    candidates: list[tuple],
    *,
    log_stream,
) -> tuple[pd.DataFrame, dict[str, str], float]:
    benchmark_candidates, skipped_rows = _filter_trivial_candidates(
        candidates,
        state_name=state_spec.state_id,
        stage=1,
        atol=config.atol,
        rtol=config.rtol,
    )
    print(
        f"[suite {config.suite_name}] state {state_spec.state_id}: "
        f"{len(benchmark_candidates)} benchmarked, "
        f"{len(skipped_rows)} baseline-equivalent skipped, "
        f"{len(candidates)} candidates total",
        file=log_stream,
        flush=True,
    )

    tasks = _build_state_tasks(state_spec, benchmark_candidates, config)
    if not tasks:
        df = pd.DataFrame(skipped_rows)
        if not df.empty:
            df["state_name"] = state_spec.state_id
            df["state_family"] = state_spec.state_family
            df["n_qutrits"] = state_spec.num_qutrits
            df["suite_name"] = config.suite_name
        paths = _persist_state_results(config, state_spec, df)
        return df, paths, 0.0

    label = f"suite {config.suite_name} :: {state_spec.state_id}"
    reporter = _BenchmarkProgressReporter(len(tasks), label, stream=log_stream)
    reporter.start()

    rows: list[Optional[dict]] = [None] * len(tasks)
    started = time.time()
    jobs = max(int(config.jobs or 1), 1)

    if jobs <= 1:
        for index, task in enumerate(tasks):
            try:
                rows[index] = _benchmark_candidate_worker(task)
            except Exception:  # pragma: no cover - defensive
                rows[index] = _make_failure_row(task)
            reporter.update(task)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_index = {
                executor.submit(_benchmark_candidate_worker, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    rows[index] = future.result()
                except Exception:  # pragma: no cover - defensive
                    rows[index] = _make_failure_row(tasks[index])
                reporter.update(tasks[index])

    reporter.finish()

    elapsed = time.time() - started
    rows_clean = [row for row in rows if row is not None]

    df = pd.DataFrame(rows_clean + skipped_rows)
    if not df.empty:
        df["state_name"] = state_spec.state_id
        df["state_family"] = state_spec.state_family
        df["n_qutrits"] = state_spec.num_qutrits
        df["suite_name"] = config.suite_name

    paths = _persist_state_results(config, state_spec, df)
    return df, paths, elapsed


def _make_failure_row(task: dict) -> dict:
    return {
        "state_name": task["state_name"],
        "state_family": task["state_family"],
        "n_qutrits": task["n_qutrits"],
        "suite_name": task["suite_name"],
        "class_name": task["class_name"],
        "candidate_name": task["candidate_name"],
        "status": "worker_error",
        "error_message": traceback.format_exc(),
        "is_valid": False,
        "successful_trials": 0,
        "failed_trials": 0,
        "n_transpile_runs": 0,
    }


def _persist_state_results(
    config: SuiteConfig,
    state_spec: BenchmarkStateSpec,
    df: pd.DataFrame,
) -> dict[str, str]:
    output_dir = _suite_state_output_dir(config, state_spec.state_id)
    os.makedirs(output_dir, exist_ok=True)
    return write_result_bundle(
        df,
        output_dir=output_dir,
        file_prefix=f"{SUITE_FILE_PREFIX}_{state_spec.state_id}",
        top_k=config.top_k,
        fidelity_thresholds=config.fidelity_thresholds,
    )


def _write_combined_csv(
    config: SuiteConfig,
    state_frames: dict[str, pd.DataFrame],
) -> str:
    combined_path = _suite_combined_results_path(config)
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    if not state_frames:
        pd.DataFrame().to_csv(combined_path, index=False)
        return combined_path
    combined = pd.concat(
        [df for df in state_frames.values() if not df.empty],
        ignore_index=True,
        sort=False,
    ) if any(not df.empty for df in state_frames.values()) else pd.DataFrame()
    combined.to_csv(combined_path, index=False)
    return combined_path


# ─────────────────────── publiczne API ─────────────────────────


def run_benchmark_suite(
    config: SuiteConfig,
    *,
    log_stream=None,
) -> dict:
    """Uruchom suite benchmarkow dla wszystkich stanow w ``config.states``.

    Zwraca slownik z metadanymi runu (sciezki + per-state DataFrame).
    """
    suite_log_path = _suite_log_path(config)
    os.makedirs(os.path.dirname(suite_log_path), exist_ok=True)

    log_file = open(suite_log_path, "a", encoding="utf-8")
    user_stream = log_stream if log_stream is not None else sys.stdout
    tee = _TeeStream(user_stream, log_file)

    try:
        return _run_benchmark_suite_inner(config, tee)
    finally:
        log_file.close()


def _run_benchmark_suite_inner(config: SuiteConfig, tee) -> dict:
    started = time.time()

    state_specs: list[BenchmarkStateSpec] = []
    for state_name in config.states:
        state_specs.append(resolve_benchmark_state(state_name))

    candidates = generate_all_class_candidates(
        mode=config.candidate_mode,
        class_filter=config.class_filter,
    )
    candidates = _strip_e_old_baseline_only_from_candidates(candidates)

    print(
        f"[suite {config.suite_name}] starting; "
        f"states={len(state_specs)}, candidates_per_state={len(candidates)}, "
        f"jobs={config.jobs}, mode={config.candidate_mode}, "
        f"strategy={config.encoding_strategy}",
        file=tee,
        flush=True,
    )
    print(
        f"[suite {config.suite_name}] states: "
        + ", ".join(spec.state_id for spec in state_specs),
        file=tee,
        flush=True,
    )

    state_frames: dict[str, pd.DataFrame] = {}
    state_paths: dict[str, dict[str, str]] = {}
    state_durations: dict[str, float] = {}

    total_states = len(state_specs)
    for index, state_spec in enumerate(state_specs, start=1):
        if config.skip_existing and _state_results_already_exist(
            config, state_spec.state_id
        ):
            print(
                f"[suite {config.suite_name}] [{index}/{total_states}] "
                f"skipping {state_spec.state_id} (results already exist)",
                file=tee,
                flush=True,
            )
            csv_path = os.path.join(
                _suite_state_output_dir(config, state_spec.state_id),
                f"{SUITE_FILE_PREFIX}_{state_spec.state_id}_results.csv",
            )
            try:
                state_frames[state_spec.state_id] = pd.read_csv(csv_path)
            except Exception:
                state_frames[state_spec.state_id] = pd.DataFrame()
            continue

        state_started = time.time()
        print(
            f"[suite {config.suite_name}] [{index}/{total_states}] "
            f"=== state {state_spec.state_id} "
            f"(family={state_spec.state_family}, "
            f"qutrits={state_spec.num_qutrits}, "
            f"edges={len(state_spec.edges)}) ===",
            file=tee,
            flush=True,
        )

        df, paths, elapsed = _run_state(
            config, state_spec, candidates, log_stream=tee,
        )
        state_frames[state_spec.state_id] = df
        state_paths[state_spec.state_id] = paths
        state_durations[state_spec.state_id] = elapsed

        total_elapsed = time.time() - started
        avg_per_state = total_elapsed / max(index, 1)
        remaining = max(total_states - index, 0)
        suite_eta = avg_per_state * remaining
        print(
            f"[suite {config.suite_name}] [{index}/{total_states}] "
            f"finished {state_spec.state_id} "
            f"in {_format_duration(time.time() - state_started)}; "
            f"suite elapsed={_format_duration(total_elapsed)} "
            f"eta={_format_duration(suite_eta)} "
            f"remaining_states={remaining}",
            file=tee,
            flush=True,
        )

    combined_csv_path = _write_combined_csv(config, state_frames)

    total_elapsed = time.time() - started
    print(
        f"[suite {config.suite_name}] DONE in "
        f"{_format_duration(total_elapsed)}; "
        f"combined CSV: {combined_csv_path}",
        file=tee,
        flush=True,
    )

    return {
        "suite_name": config.suite_name,
        "states": [spec.state_id for spec in state_specs],
        "state_frames": state_frames,
        "state_paths": state_paths,
        "state_durations": state_durations,
        "combined_csv": combined_csv_path,
        "total_elapsed_seconds": total_elapsed,
    }


__all__ = [
    "COMBINED_RESULTS_BASENAME",
    "SUITE_FILE_PREFIX",
    "SuiteConfig",
    "generate_all_class_candidates",
    "run_benchmark_suite",
]
