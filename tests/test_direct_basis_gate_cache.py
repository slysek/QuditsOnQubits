from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest
from filelock import FileLock
from qiskit import qpy

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates as gates


def _cache_worker(cache, fixture, barrier, attempted, started, release, results):
    """Separate spawned interpreter with deterministic synthesis, real cache I/O."""
    def compile_cz3(embedding):
        with (Path(cache).parent / "syntheses.log").open("a") as handle:
            handle.write("synthesis\n")
        started.set()
        deadline = time.monotonic() + 20
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Test did not release the synthesis worker.")
            time.sleep(0.01)
        with open(fixture, "rb") as handle:
            return qpy.load(handle)[0]

    def lock(path):
        attempted.put(True)
        return FileLock(path)

    gates._compile_cz3 = compile_cz3
    gates.FileLock = lock
    barrier.wait(timeout=20)
    library = gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    results.put(library.benchmark_metrics())


def _visible_entries(cache):
    return [p for p in cache.iterdir() if p.is_dir() and not p.name.startswith(".")]


@pytest.mark.parametrize("terminate_holder", [False, True])
def test_cache_coordinates_processes_and_releases_dead_holder(
    tmp_path, exact_cz3_synthesis, terminate_holder,
):
    fixture = tmp_path / "exact_cz3.qpy"
    with fixture.open("wb") as handle:
        qpy.dump(gates._compile_cz3(np.eye(4, 3)), handle)
    cache = tmp_path / "shared"
    ctx = multiprocessing.get_context("spawn")
    count = 1 if terminate_holder else 2
    barrier = ctx.Barrier(count)
    attempted, results = ctx.Queue(), ctx.Queue()
    started = ctx.Event()
    # A killed worker can leave an Event's internal lock held. Use a file for
    # this test barrier so terminating the worker cannot deadlock test cleanup.
    release = tmp_path / "release"
    workers = [ctx.Process(target=_cache_worker, args=(
        str(cache), str(fixture), barrier, attempted, started, str(release), results,
    )) for _ in range(count)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert attempted.get(timeout=30)
        assert started.wait(timeout=30)
        assert not _visible_entries(cache)
        if terminate_holder:
            workers[0].terminate()
            workers[0].join(timeout=10)
            assert not workers[0].is_alive()
            lock_path = next(cache.glob("*.lock"))
            with FileLock(lock_path, timeout=3):
                pass  # OS releases the lock even without a Python finally block.
            repaired = gates.optimized_gate_library(np.eye(3), cache_dir=cache)
            assert not repaired.cache_hit
        else:
            release.touch()
            rows = [results.get(timeout=30) for _ in workers]
            for worker in workers:
                worker.join(timeout=10)
                assert worker.exitcode == 0
            assert sorted(row["gate_cache_hit"] for row in rows) == [False, True]
            assert rows[0]["N_2q"] == rows[1]["N_2q"]
            assert all(row["E_norm"] <= 1e-5 for row in rows)
            assert (tmp_path / "syntheses.log").read_text().splitlines() == ["synthesis"]
        assert len(_visible_entries(cache)) == 1
        assert gates.optimized_gate_library(np.eye(3), cache_dir=cache).cache_hit
    finally:
        release.touch()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        attempted.close()
        results.close()


@pytest.mark.parametrize("damage", ["manifest", "missing_manifest", "qpy", "missing_qpy", "encoding", "metadata"])
def test_invalid_cache_is_rebuilt(tmp_path, exact_cz3_synthesis, damage):
    cache = tmp_path / "cache"
    gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    directory, = _visible_entries(cache)
    manifest = directory / "synthesis.json"
    if damage == "manifest":
        manifest.write_text('{"incomplete":', encoding="utf-8")
    elif damage == "missing_manifest":
        manifest.unlink()
    elif damage == "qpy":
        (directory / "CZ3_W.qpy").write_bytes(b"QISKIT")
    elif damage == "missing_qpy":
        (directory / "CZ3_W.qpy").unlink()
    elif damage == "encoding":
        np.save(directory / "E.npy", np.zeros((4, 3)))
    else:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        del data["F3"]["leakage_phase"]
        manifest.write_text(json.dumps(data), encoding="utf-8")
    rebuilt = gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    assert not rebuilt.cache_hit
    assert len(exact_cz3_synthesis) == 2
    assert gates.optimized_gate_library(np.eye(3), cache_dir=cache).cache_hit
    assert not list(cache.glob(".*"))


def test_failed_publication_never_exposes_partial_cache(tmp_path, exact_cz3_synthesis, monkeypatch):
    cache = tmp_path / "cache"
    original_dump = qpy.dump
    calls = []

    def fail_second_dump(circuit, handle):
        # Even after the first QPY is written, no completed entry is visible.
        assert not _visible_entries(cache)
        calls.append(handle.name)
        if len(calls) == 2:
            raise OSError("Simulated disk full")
        original_dump(circuit, handle)

    with monkeypatch.context() as patch:
        patch.setattr(qpy, "dump", fail_second_dump)
        with pytest.raises(OSError, match="disk full"):
            gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    assert not _visible_entries(cache)
    assert not list(cache.glob(".*"))
    assert not gates.optimized_gate_library(np.eye(3), cache_dir=cache).cache_hit
    assert gates.optimized_gate_library(np.eye(3), cache_dir=cache).cache_hit
