"""Hash-verified unified benchmark bundles and replayable analysis."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .analysis import analyze_trials
from .theta_continuation.artifacts import RunStore, atomic_write_json, fingerprint

SCHEMA = "encoding-benchmark-v1"
_BUNDLE = re.compile(r"(?:candidates/[0-9]{5}|trials/[0-9]{5}/[0-9]{10})\Z")


@dataclass
class BenchmarkResult:
    output_dir: Path
    manifest: dict
    trials: pd.DataFrame
    statistics: pd.DataFrame

    @property
    def pareto_front(self) -> pd.DataFrame:
        return self.statistics.loc[self.statistics.pareto_rank.eq(1).fillna(False)].copy()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_store(output_dir: Path, payload: dict) -> RunStore:
    manifest = {"schema": SCHEMA, **payload}
    manifest["fingerprint"] = fingerprint(manifest)
    return RunStore.create(output_dir, manifest)


def complete_run(store: RunStore, rows: list[dict], bundle_paths: list[str]) -> BenchmarkResult:
    hashes = {path: file_hash(store.root / path / "complete.json") for path in bundle_paths}
    store.write_bundle("results", metadata={"trials": rows, "bundle_hashes": hashes})
    atomic_write_json(store.root / "run-complete.json", {
        "schema": SCHEMA,
        "manifest_sha256": file_hash(store.root / "manifest.json"),
        "results_sha256": file_hash(store.root / "results/complete.json"),
    })
    return load_benchmark(store.root)


def load_benchmark(output_dir: str | Path) -> BenchmarkResult:
    """Verify the complete evidence graph, then recompute Pareto from saved trials."""
    store = RunStore.open(output_dir)
    manifest = store.manifest
    expected = dict(manifest)
    digest = expected.pop("fingerprint")
    if manifest.get("schema") != SCHEMA or fingerprint(expected) != digest:
        raise ValueError("Invalid benchmark manifest hash or schema")
    completion = json.loads((store.root / "run-complete.json").read_text(encoding="utf-8"))
    if (completion.get("schema") != SCHEMA
            or completion.get("manifest_sha256") != file_hash(store.root / "manifest.json")
            or completion.get("results_sha256") != file_hash(store.root / "results/complete.json")):
        raise ValueError("Benchmark completion hash mismatch")
    metadata = store.read_bundle("results")["metadata"]
    for relative, digest in metadata["bundle_hashes"].items():
        if not _BUNDLE.fullmatch(relative):
            raise ValueError("Invalid benchmark bundle path")
        directory = (store.root / relative).resolve()
        if not directory.is_relative_to(store.root) or file_hash(directory / "complete.json") != digest:
            raise ValueError("Benchmark bundle hash mismatch")
        store.read_bundle(relative)
    rows = metadata["trials"]
    for row in rows:
        for key in ("candidate_bundle", "trial_bundle"):
            if row.get(key) not in metadata["bundle_hashes"]:
                raise ValueError("Trial references an unverified bundle")
        saved_row = store.read_bundle(row["trial_bundle"])["metadata"]
        if saved_row != row:
            raise ValueError("Trial index does not match saved metadata")
    trials = pd.DataFrame(rows)
    statistics = analyze_trials(trials, seeds=tuple(manifest["config"]["transpiler_seeds"]))
    return BenchmarkResult(store.root, manifest, trials, statistics)
