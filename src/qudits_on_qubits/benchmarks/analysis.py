"""Provider-neutral aggregation using the existing Pareto implementation."""
from __future__ import annotations

import pandas as pd

from .direct_basis.pareto_selection import aggregate_strategy_statistics, rank_pareto_candidates

BOUNDARIES = ("comparison_id", "target_hash", "workload_hash")


def analyze_trials(trials: pd.DataFrame, *, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Require every planned seed and preserve independent target/protocol fronts."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if trials.empty:
        return rank_pareto_candidates(aggregate_strategy_statistics(trials))
    missing = set(BOUNDARIES) - set(trials.columns)
    if missing:
        raise ValueError(f"Missing comparison boundaries: {sorted(missing)}")
    if trials[list(BOUNDARIES)].isna().any().any():
        raise ValueError("Comparison boundaries cannot be missing")
    if not set(trials.seed_transpiler).issubset(set(seeds)):
        raise ValueError("Trial contains an unplanned seed")
    results = []
    for key, group in trials.groupby(list(BOUNDARIES), sort=True, dropna=False):
        statistics = aggregate_strategy_statistics(group)
        complete = (
            (statistics.total_trial_count == len(seeds))
            & (statistics.successful_trial_count == len(seeds))
            & (len(seeds) >= 2)
        )
        statistics["pareto_eligible"] = statistics.pareto_eligible & complete
        statistics.loc[~complete, "analysis_status"] = "incomplete_seed_set"
        statistics = rank_pareto_candidates(statistics)
        for column, value in zip(BOUNDARIES, key, strict=True):
            statistics[column] = value
        statistics["is_baseline"] = statistics.candidate_name.eq("canonical")
        results.append(statistics)
    return pd.concat(results, ignore_index=True)
