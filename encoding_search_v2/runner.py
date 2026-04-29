import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from encoding_search_v2.candidates import (
    Candidate,
    CandidateSearchConfig,
    filter_candidates_for_stage2,
    format_candidate_counts,
    generate_stage1_candidates,
)
from encoding_search_v2.paths import stage_file_prefix, stage_output_dir
from encoding_search_v2.preselection import load_preselected_candidates, select_preselected_rows
from encoding_search_v2.results import write_result_bundle
from encoding_search_v2.states import BenchmarkStateSpec, resolve_benchmark_state
from encoding_search_v2.triviality import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    _filter_trivial_candidates,
    candidate_metadata_fields,
)


DEFAULT_FIDELITY_THRESHOLDS = (0.85, 0.90, 0.95)


def benchmark_basis(*args, **kwargs):
    from QuditsOnQubits.benchmark_encoding_bases import benchmark_basis as _benchmark_basis

    return _benchmark_basis(*args, **kwargs)


@dataclass(frozen=True)
class PipelineConfig:
    state_name: str
    stage: int
    n_qutrits: Optional[int] = None
    output_root: Optional[str] = None
    jobs: int = 1
    n_transpile_runs: int = 20
    top_k: int = 30
    ranking_csv: Optional[str] = None
    rank_by: str = "exact_depth"
    export_circuits: bool = False
    approximation_values: Optional[Sequence[float]] = None
    fidelity_thresholds: Sequence[float] = DEFAULT_FIDELITY_THRESHOLDS
    approximation_seed: int = 0
    max_monomial_full: Optional[int] = None
    max_product: Optional[int] = None
    include_product_grid: bool = False
    max_product_grid: Optional[int] = None
    product_grid_phase_steps: int = 4
    product_grid_polar_steps: int = 3
    include_near_identity: bool = False
    near_identity_samples_per_eps: int = 2
    near_identity_seed: int = 500
    limit_candidates: Optional[int] = None
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL

    def candidate_config(self) -> CandidateSearchConfig:
        return CandidateSearchConfig(
            max_monomial_full=self.max_monomial_full,
            max_product=self.max_product,
            include_product_grid=self.include_product_grid,
            max_product_grid=self.max_product_grid,
            product_grid_phase_steps=self.product_grid_phase_steps,
            product_grid_polar_steps=self.product_grid_polar_steps,
            include_near_identity=self.include_near_identity,
            near_identity_samples_per_eps=self.near_identity_samples_per_eps,
            near_identity_seed=self.near_identity_seed,
            limit_candidates=self.limit_candidates,
        )

    def state_spec(self) -> BenchmarkStateSpec:
        return resolve_benchmark_state(self.state_name, n_qutrits=self.n_qutrits)


def _stage_circuits_output_dir(config: PipelineConfig) -> Optional[str]:
    if not config.export_circuits:
        return None
    state_id = config.state_spec().state_id
    return os.path.join(
        stage_output_dir(state_id, config.stage, output_root=config.output_root),
        "circuits",
    )


def _build_benchmark_tasks(
    candidates: list[Candidate],
    config: PipelineConfig,
    encoding_strategy: str,
) -> list[dict]:
    state_spec = config.state_spec()
    circuits_output_dir = _stage_circuits_output_dir(config)
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
            "approximation_values": config.approximation_values,
            "fidelity_thresholds": tuple(config.fidelity_thresholds),
            "approximation_seed": config.approximation_seed,
            "encoding_strategy": encoding_strategy,
            "candidate_metadata": candidate_metadata_fields(
                class_name,
                candidate_name,
                e_new,
                atol=config.atol,
                rtol=config.rtol,
            ),
        }
        for class_name, candidate_name, e_new in candidates
    ]


def _strip_internal_circuit_objects(row: dict) -> dict:
    clean = dict(row)
    for key in list(clean):
        if key.startswith("_") and key.endswith("_best_qc"):
            clean.pop(key)
    return clean


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
    row.update(task["candidate_metadata"])
    return _strip_internal_circuit_objects(row)


def run_candidate_benchmarks(
    candidates: list[Candidate],
    config: PipelineConfig,
    encoding_strategy: str,
) -> list[dict]:
    tasks = _build_benchmark_tasks(candidates, config, encoding_strategy=encoding_strategy)
    if not tasks:
        return []

    jobs = int(config.jobs or 1)
    rows: list[Optional[dict]] = [None] * len(tasks)

    if jobs <= 1:
        for index, task in enumerate(tasks):
            rows[index] = _benchmark_candidate_worker(task)
        return [row for row in rows if row is not None]

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {
            executor.submit(_benchmark_candidate_worker, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            rows[future_to_index[future]] = _strip_internal_circuit_objects(future.result())

    return [row for row in rows if row is not None]


def _write_preselected_rows(config: PipelineConfig) -> str:
    state_id = config.state_spec().state_id
    selected = select_preselected_rows(
        csv_path=config.ranking_csv,
        state_name=state_id,
        top_k=config.top_k,
        rank_by=config.rank_by,
    )
    output_dir = stage_output_dir(state_id, config.stage, output_root=config.output_root)
    path = os.path.join(output_dir, "preselected_candidates.csv")
    os.makedirs(output_dir, exist_ok=True)
    selected.to_csv(path, index=False)
    return path


def _warn_on_missing_preselected(
    preselected: set[tuple[str, str]],
    filtered: list[Candidate],
    ranking_csv: str,
):
    regenerated = {(class_name, candidate_name) for class_name, candidate_name, _ in filtered}
    missing = sorted(preselected - regenerated)
    if not missing:
        return

    details = ", ".join(f"{class_name}/{candidate_name}" for class_name, candidate_name in missing)
    warnings.warn(
        f"{len(missing)} preselected candidates from {ranking_csv!r} were not regenerated "
        f"from the current stage-1 candidate configuration: {details}. "
        "Check that --include-product-grid, --include-near-identity, limits, and state match.",
        stacklevel=2,
    )


def run_stage1(config: PipelineConfig):
    state_spec = config.state_spec()
    candidates = generate_stage1_candidates(config.candidate_config())
    benchmark_candidates, skipped_rows = _filter_trivial_candidates(
        candidates,
        state_name=state_spec.state_id,
        stage=1,
        atol=config.atol,
        rtol=config.rtol,
    )
    print(
        f"Stage 1 [{state_spec.state_id}]: {len(benchmark_candidates)} benchmarked "
        f"candidates, {len(skipped_rows)} baseline-equivalent skipped"
    )
    print(f"Classes: {format_candidate_counts(benchmark_candidates)}")
    started = time.time()
    rows = run_candidate_benchmarks(benchmark_candidates, config, encoding_strategy="append_w")
    elapsed = time.time() - started
    print(f"Stage 1 [{state_spec.state_id}] benchmark time: {elapsed:.1f}s")

    df = pd.DataFrame(rows + skipped_rows)
    if not df.empty:
        df["state_name"] = state_spec.state_id
        df["state_family"] = state_spec.state_family
        df["n_qutrits"] = state_spec.num_qutrits
    output_dir = stage_output_dir(state_spec.state_id, 1, output_root=config.output_root)
    paths = write_result_bundle(
        df,
        output_dir=output_dir,
        file_prefix=stage_file_prefix(state_spec.state_id, 1),
        top_k=config.top_k,
        fidelity_thresholds=config.fidelity_thresholds,
    )
    return df, paths


def run_stage2(config: PipelineConfig):
    if not config.ranking_csv:
        raise ValueError("Stage 2 requires ranking_csv from stage 1.")

    state_spec = config.state_spec()
    preselected = load_preselected_candidates(
        csv_path=config.ranking_csv,
        state_name=state_spec.state_id,
        top_k=config.top_k,
        rank_by=config.rank_by,
    )
    candidates = generate_stage1_candidates(config.candidate_config())
    filtered = filter_candidates_for_stage2(candidates, preselected)
    _warn_on_missing_preselected(preselected, filtered, config.ranking_csv)
    benchmark_candidates, skipped_rows = _filter_trivial_candidates(
        filtered,
        state_name=state_spec.state_id,
        stage=2,
        atol=config.atol,
        rtol=config.rtol,
    )

    print(
        f"Stage 2 [{state_spec.state_id}]: {len(benchmark_candidates)} benchmarked "
        f"selected candidates, {len(skipped_rows)} baseline-equivalent skipped"
    )
    print(f"Selected classes: {format_candidate_counts(benchmark_candidates)}")
    preselected_path = _write_preselected_rows(config)

    started = time.time()
    rows = run_candidate_benchmarks(
        benchmark_candidates,
        config,
        encoding_strategy="prepared_w_then_conjugated_entanglers",
    )
    elapsed = time.time() - started
    print(f"Stage 2 [{state_spec.state_id}] benchmark time: {elapsed:.1f}s")

    df = pd.DataFrame(rows + skipped_rows)
    if not df.empty:
        df["state_name"] = state_spec.state_id
        df["state_family"] = state_spec.state_family
        df["n_qutrits"] = state_spec.num_qutrits
    output_dir = stage_output_dir(state_spec.state_id, 2, output_root=config.output_root)
    paths = write_result_bundle(
        df,
        output_dir=output_dir,
        file_prefix=stage_file_prefix(state_spec.state_id, 2),
        top_k=config.top_k,
        fidelity_thresholds=config.fidelity_thresholds,
    )
    paths["preselected_csv"] = preselected_path
    return df, paths
