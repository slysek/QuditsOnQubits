"""Reassess immutable theta artifacts; reconstruct only missing raw CZ3 circuits.

The historical continuation branch is preserved. Recovery repeats the original
BQSKit numerical call, then checkpoints its raw output *before* accepting or
compiling it. Thus even a rejected result survives interruption and resume.
"""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from .artifacts import RunStore, fingerprint, runtime_provenance
from .compilation import validate_cz3_candidate
from .cz3_continuation import ContinuationState
from .cz3_template import parameterize_cz3_circuit
from .encoding import _validate_integer, theta_embedding
from .models import ThetaBenchmarkConfig
from .runner import (
    _baseline, _check_metrics, _compile, _emit, _full_circuits, _validate_point,
)


PROCEDURE_VERSION = 1
_RAW_NAME = "raw_cz3.qpy"


def _unrestricted_metrics(circuit, theta):
    # A finite generous bound measures a raw candidate without accepting it.
    return validate_cz3_candidate(circuit, theta_embedding(theta), tolerance=100.0)


def _original_fallback_metrics(bundle):
    if "cz3_fallback.qpy" in bundle["circuits"]:
        return _unrestricted_metrics(bundle["circuits"]["cz3_fallback.qpy"], bundle["metadata"]["theta"])
    for error in bundle["metadata"]["errors"]:
        marker = "CZ3 failed gate validation: "
        if marker in error:
            try:
                metrics = ast.literal_eval(error.split(marker, 1)[1])
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Invalid historical CZ3 validation metrics") from exc
            if not isinstance(metrics, dict) or not {"E_norm", "L_norm", "N_2q"} <= metrics.keys():
                raise ValueError("Incomplete historical CZ3 validation metrics")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) or value < 0 for value in metrics.values()):
                raise ValueError("Invalid historical CZ3 validation metrics")
            return {**metrics, "N_CZ": metrics["N_2q"]}
    return {}


def _comparison(actual, original):
    keys = [key for key in ("E_norm", "L_norm", "N_CZ") if key in original]
    differences = {key: actual[key] - original[key] for key in keys}
    matches = bool(keys) and all(
        actual[key] == original[key] if key == "N_CZ" else
        np.isclose(actual[key], original[key], atol=1e-12, rtol=1e-8)
        for key in keys
    )
    return {"original_metrics": original, "actual_metrics": actual,
            "differences": differences, "matches_original": bool(matches) if keys else None,
            "comparison_atol": 1e-12, "comparison_rtol": 1e-8}


def _attempt_reassessment(bundle, template, tolerance):
    result = []
    for attempt in bundle["documents"]["attempts.json"]["attempts"]:
        fit = attempt["fit"]
        metrics = _unrestricted_metrics(template.bind(fit["parameters"]), attempt["theta"])
        _check_metrics({key: metrics[key] for key in ("E_norm", "L_norm")}, fit, "historical attempt")
        result.append({"id": attempt["id"], "theta": attempt["theta"],
                       "requested_grid_point": attempt["requested_grid_point"],
                       "historical_valid": fit["valid"],
                       "valid_at_new_tolerance": max(metrics["E_norm"], metrics["L_norm"]) <= tolerance,
                       "E_norm": metrics["E_norm"], "L_norm": metrics["L_norm"]})
    return result


def _load_source(source_run):
    store = RunStore.open(source_run)
    manifest = store.manifest
    if manifest.get("benchmark") != "theta_continuation_v1" or manifest.get("format_version") != 1:
        raise ValueError("Source must be an original theta_continuation_v1 run")
    if fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("Source manifest fingerprint does not match its content")
    config = ThetaBenchmarkConfig.from_dict(manifest["config"])
    if store.completed_point_indices() != list(range(len(config.grid()))):
        raise ValueError("Source must contain its complete configured theta grid")
    # Verify every completed bundle, including saved full circuits, before reuse.
    inventory = {}
    for marker in sorted(store.root.rglob("complete.json")):
        relative = marker.parent.relative_to(store.root).as_posix()
        inventory[relative] = store.read_bundle(relative)["complete"]
    baseline_bundle = store.read_bundle("baseline")
    template = parameterize_cz3_circuit(
        baseline_bundle["circuits"]["cz3_original.qpy"], baseline_sha256=config.baseline_sha256,
    )
    if manifest["template_id"] != template.template_id or manifest["baseline_sha256"] != config.baseline_sha256:
        raise ValueError("Source baseline identity mismatch")
    baseline = _baseline(store, template, config, readonly=True)
    prior = ContinuationState(0.0, template.initial_parameters.copy(), "baseline")
    bundles = []
    for index, theta in enumerate(config.grid()):
        bundle = store.read_bundle(f"points/{index:05d}")
        prior = _validate_point(bundle, prior, template, config, baseline, index, theta)
        _attempt_reassessment(bundle, template, config.cz3_tolerance)
        _original_fallback_metrics(bundle)
        bundles.append(bundle)
    return store, config, template, bundles, fingerprint(inventory)


def _manifest(source, config, inventory_hash):
    payload = {
        "format_version": 1, "benchmark": "theta_threshold_reassessment_v1",
        "procedure_version": PROCEDURE_VERSION, "config": config.to_dict(),
        "source_run": str(source.root), "source_fingerprint": source.manifest["fingerprint"],
        "source_inventory_fingerprint": inventory_hash,
        "source_cz3_tolerance": source.manifest["config"]["cz3_tolerance"],
        "baseline_sha256": source.manifest["baseline_sha256"],
        "template_id": source.manifest["template_id"],
        "continuation_policy": "reassess_saved_attempts_preserve_historical_branch",
        "recovery_protocol": {"target": "StateSystem_9_columns", "num_qubits": 4,
                              "gate_set": ["U3", "CZ"], "optimization_level": 2,
                              "max_synthesis_size": 4, "compiler_num_workers": 1,
                              "synthesis_epsilon": optimized_gates.SYNTHESIS_EPSILON,
                              "seed": optimized_gates.SYNTHESIS_SEED},
        "provenance": runtime_provenance(),
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _recover_raw(output_dir, expected_fingerprint, index, theta, original_metrics):
    """Spawn-safe worker. Publish raw QPY before any acceptance test."""
    store = RunStore.open(output_dir, expected_fingerprint=expected_fingerprint)
    relative = f"recovery/{index:05d}"
    if (store.root / relative / "complete.json").exists():
        store.read_bundle(relative)
        return index
    started = perf_counter()
    circuits = {}
    error = None
    try:
        # Same compiler/model/seed/epsilon as synthesize_cz3; its acceptance
        # wrapper is intentionally bypassed so rejected raw circuits survive.
        with optimized_gates._SYNTHESIS_LOCK:
            circuit = optimized_gates._compile_cz3(theta_embedding(theta))
        circuit.name = "CZ3_W"
        circuits[_RAW_NAME] = circuit
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    store.write_bundle(relative, metadata={
        "index": index, "theta": theta, "origin": "reconstructed",
        "source_fingerprint": store.manifest["source_fingerprint"],
        "recovery_protocol": store.manifest["recovery_protocol"],
        "original_metrics": original_metrics, "elapsed_seconds": perf_counter() - started,
        "status": "raw_saved" if circuits else "synthesis_failed", "error": error,
    }, circuits=circuits)
    return index


def _validate_recovery(store, index, theta, original_metrics):
    bundle = store.read_bundle(f"recovery/{index:05d}")
    metadata = bundle["metadata"]
    if (metadata["index"] != index or metadata["theta"] != theta
            or metadata["origin"] != "reconstructed"
            or metadata["source_fingerprint"] != store.manifest["source_fingerprint"]
            or metadata["recovery_protocol"] != store.manifest["recovery_protocol"]
            or metadata["original_metrics"] != original_metrics
            or not np.isfinite(metadata["elapsed_seconds"]) or metadata["elapsed_seconds"] < 0):
        raise ValueError("Recovery artifact identity mismatch")
    has_circuit = _RAW_NAME in bundle["circuits"]
    if (metadata["status"] != ("raw_saved" if has_circuit else "synthesis_failed")
            or (has_circuit and metadata["error"] is not None)
            or (not has_circuit and not isinstance(metadata["error"], str))):
        raise ValueError("Recovery artifact status mismatch")
    return bundle


def _recover_missing(store, source, bundles, workers, progress):
    pending = []
    for index, bundle in enumerate(bundles):
        if bundle["metadata"]["fallback_used"] and "cz3_fallback.qpy" not in bundle["circuits"]:
            theta = bundle["metadata"]["theta"]
            original_metrics = _original_fallback_metrics(bundle)
            if (store.root / f"recovery/{index:05d}/complete.json").exists():
                _validate_recovery(store, index, theta, original_metrics)
            else:
                pending.append((str(store.root), store.manifest["fingerprint"], index, theta, original_metrics))
    if not pending:
        return
    # A reconstruction under changed package versions is a new synthesis, not
    # a repeat of the recorded numerical protocol; do not silently call it one.
    if source.manifest["provenance"]["versions"] != store.manifest["provenance"]["versions"]:
        raise ValueError("Recovery requires the original numerical package versions")
    _emit(progress, f"recovering {len(pending)} missing raw CZ3 circuits; workers={workers}")
    if workers == 1:
        for args in pending:
            index = _recover_raw(*args)
            _emit(progress, f"raw CZ3 checkpoint saved: point {index}")
    else:
        # Independent historical fallbacks only. Continuation is never run in
        # parallel, nor restarted under a different warm-start branch.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_recover_raw, *args) for args in pending]
            for future in as_completed(futures):
                _emit(progress, f"raw CZ3 checkpoint saved: point {future.result()}")


def _build_point(source_bundle, store, template, baseline, config):
    old = source_bundle["metadata"]
    index, theta = old["index"], old["theta"]
    encoding = theta_embedding(theta)
    metadata, arrays, documents = deepcopy(old), deepcopy(source_bundle["arrays"]), deepcopy(source_bundle["documents"])
    circuits = {name: circuit.copy() for name, circuit in source_bundle["circuits"].items()
                if name != "cz3_selected.qpy"}
    details = {
        "source_point_fingerprint": fingerprint(source_bundle["complete"]),
        "source_correct": old["correct"], "source_status": old["status"],
        "source_errors": old["errors"], "source_cz3_metrics": old["cz3_metrics"],
        "source_cz3_tolerance": store.manifest["source_cz3_tolerance"],
        "cz3_tolerance": config.cz3_tolerance,
        "continuation_policy": store.manifest["continuation_policy"],
        "attempts": _attempt_reassessment(source_bundle, template, config.cz3_tolerance),
        "origin": "baseline" if index == 0 else "source_saved",
        "compilations": {}, "recovery_seconds": 0.0,
    }
    candidates, errors = [], []
    if index == 0:
        candidates.append((baseline, "baseline"))
    else:
        if old["fallback_used"] and "cz3_fallback.qpy" not in circuits:
            recovered = _validate_recovery(store, index, theta, _original_fallback_metrics(source_bundle))
            details["origin"] = "reconstructed"
            details["recovery_seconds"] = recovered["metadata"]["elapsed_seconds"]
            if _RAW_NAME in recovered["circuits"]:
                circuits["cz3_fallback.qpy"] = recovered["circuits"][_RAW_NAME].copy()
            else:
                errors.append("Recovery: " + recovered["metadata"]["error"])
        for name, method in (("cz3_continuation.qpy", "continuation"), ("cz3_fallback.qpy", "bqskit")):
            if name not in circuits:
                continue
            try:
                actual = _unrestricted_metrics(circuits[name], theta)
                if method == "bqskit":
                    details["fallback_comparison"] = _comparison(actual, _original_fallback_metrics(source_bundle))
                result = _compile(circuits[name], encoding, config)
                details["compilations"][method] = result.trials
                candidates.append((result, method))
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception as exc:
                errors.append(f"{method}: {type(exc).__name__}: {exc}")
                details["compilations"][method] = getattr(exc, "metrics", {}).get("trials", [])
    selected, method = (min(candidates, key=lambda candidate:
                            tuple(candidate[0].metrics[key] for key in ("N_CZ", "depth", "N_1q")))
                        if candidates else (None, None))
    metrics = {} if selected is None else selected.metrics
    if selected is not None:
        circuits["cz3_selected.qpy"] = selected.circuit
    correct = selected is not None and "f3_optimal.qpy" in circuits
    within_budget = None if selected is None else metrics["N_CZ"] <= baseline.metrics["N_CZ"]
    details["candidate_errors"] = errors
    metadata.update({
        "correct": correct, "status": "failed" if not correct else "ok" if within_budget else "above_baseline",
        "selected_method": method, "within_baseline_budget": within_budget, "cz3_metrics": metrics,
        "cz3_n_cz": metrics.get("N_CZ"), "cz3_depth": metrics.get("depth"),
        "cz3_n_1q": metrics.get("N_1q"), "cz3_E_norm": metrics.get("E_norm"), "cz3_L_norm": metrics.get("L_norm"),
        "errors": errors if not correct else [], "reassessment": details,
        # Old timings remain explicitly in details; no continuation/F3 fitting
        # took place in this new run.
        "continuation_seconds": 0.0, "f3_synthesis_seconds": 0.0,
        "fallback_seconds": details["recovery_seconds"],
    })
    details["source_timings"] = {key: old[key] for key in ("continuation_seconds", "f3_synthesis_seconds", "fallback_seconds")}
    documents["reassessment.json"] = details
    return {"metadata": metadata, "circuits": circuits, "arrays": arrays, "documents": documents}


def _validate_reassessed_point(bundle, source_bundle, prior, template, config, baseline, index, theta, store):
    details = bundle["metadata"]["reassessment"]
    original = source_bundle["metadata"]
    expected_origin = ("baseline" if index == 0 else "reconstructed"
                       if original["fallback_used"] and "cz3_fallback.qpy" not in source_bundle["circuits"]
                       else "source_saved")
    if (details != bundle["documents"]["reassessment.json"]
            or details["source_point_fingerprint"] != fingerprint(source_bundle["complete"])
            or details["cz3_tolerance"] != config.cz3_tolerance
            or details["source_cz3_tolerance"] != store.manifest["source_cz3_tolerance"]
            or details["source_correct"] != original["correct"]
            or details["source_status"] != original["status"]
            or details["source_errors"] != original["errors"]
            or details["source_cz3_metrics"] != original["cz3_metrics"]
            or details["origin"] != expected_origin
            or details["attempts"] != _attempt_reassessment(source_bundle, template, config.cz3_tolerance)):
        raise ValueError("Reassessment artifact identity or attempt metrics mismatch")
    if expected_origin == "reconstructed":
        recovery = _validate_recovery(store, index, theta, _original_fallback_metrics(source_bundle))
        has_raw = _RAW_NAME in recovery["circuits"]
        if (has_raw != ("cz3_fallback.qpy" in bundle["circuits"])
                or details["recovery_seconds"] != recovery["metadata"]["elapsed_seconds"]
                or (has_raw and not np.allclose(
                    Operator(recovery["circuits"][_RAW_NAME]).data,
                    Operator(bundle["circuits"]["cz3_fallback.qpy"]).data, atol=1e-13, rtol=0))):
            raise ValueError("Reassessed fallback does not match its raw recovery artifact")
    for key in ("continuation_valid", "fallback_used", "fallback_reason", "branch_state", "warm_start_parent_id"):
        if bundle["metadata"][key] != original[key]:
            raise ValueError("Reassessment changed historical continuation metadata")
    for name in ("attempts.json", "branch_state.json"):
        if bundle["documents"][name] != source_bundle["documents"][name]:
            raise ValueError("Reassessment changed the historical continuation branch")
    for name, original in source_bundle["circuits"].items():
        if name == "cz3_selected.qpy":
            continue
        if name not in bundle["circuits"] or not np.allclose(
            Operator(bundle["circuits"][name]).data, Operator(original).data, atol=1e-13, rtol=0,
        ):
            raise ValueError("Reassessment changed a saved source operator")
    if "cz3_fallback.qpy" in bundle["circuits"]:
        actual = _unrestricted_metrics(bundle["circuits"]["cz3_fallback.qpy"], theta)
        if details.get("fallback_comparison") != _comparison(actual, _original_fallback_metrics(source_bundle)):
            raise ValueError("Reassessment raw fallback comparison mismatch")
    return _validate_point(bundle, prior, template, config, baseline, index, theta)


def run_reassessment(source_run, output_dir, *, cz3_tolerance=5e-4, workers=3, resume=False, progress=None):
    """Recompile/revalidate gates and graph states under an explicit looser limit."""
    workers = _validate_integer(workers, "workers", 1, 3)
    source_path, output_path = Path(source_run).resolve(), Path(output_dir).resolve()
    if source_path == output_path or source_path in output_path.parents or output_path in source_path.parents:
        raise ValueError("Source and output directories must be separate and non-nested")
    _emit(progress, "validating immutable source artifacts and historical continuation")
    source, source_config, template, bundles, inventory_hash = _load_source(source_path)
    config = replace(source_config, cz3_tolerance=cz3_tolerance)
    if config.cz3_tolerance < source_config.cz3_tolerance:
        raise ValueError("Reassessment requires a CZ3 tolerance at least as large as the source tolerance")
    manifest = _manifest(source, config, inventory_hash)
    store = (RunStore.open(output_path, expected_fingerprint=manifest["fingerprint"])
             if resume else RunStore.create(output_path, manifest))
    if store.manifest != manifest:
        raise ValueError("Reassessment manifest content does not match its fingerprint")
    baseline = _baseline(store, template, config, readonly=False)
    indices = store.completed_point_indices()
    if indices != list(range(len(indices))) or len(indices) > len(bundles):
        raise ValueError("Completed reassessed points must form a contiguous grid prefix")
    prior = ContinuationState(0.0, template.initial_parameters.copy(), "baseline")
    points = []
    for index in indices:
        bundle = store.read_bundle(f"points/{index:05d}")
        prior = _validate_reassessed_point(bundle, bundles[index], prior, template, config, baseline, index, config.grid()[index], store)
        points.append(bundle["metadata"])
    checkpoint = store.load_checkpoint()
    if checkpoint is not None and (type(checkpoint.get("last_completed_index")) is not int
                                   or checkpoint.get("last_completed_index") not in indices
                                   or checkpoint.get("source_fingerprint") != manifest["source_fingerprint"]):
        raise ValueError("Reassessment checkpoint does not match completed points")
    _recover_missing(store, source, bundles, workers, progress)
    for index in range(len(indices), len(bundles)):
        bundle = _build_point(bundles[index], store, template, baseline, config)
        prior = _validate_reassessed_point(bundle, bundles[index], prior, template, config, baseline, index, config.grid()[index], store)
        store.write_bundle(f"points/{index:05d}", **bundle)
        store.save_checkpoint({"last_completed_index": index, "source_fingerprint": manifest["source_fingerprint"]})
        points.append(bundle["metadata"])
        point = bundle["metadata"]
        _emit(progress, f"reassessed point {index + 1}/{len(bundles)}: {point['status']}; "
                        f"CZ={point['cz3_n_cz']}; E_norm={point['cz3_E_norm']}; L_norm={point['cz3_L_norm']}")
    full_rows = _full_circuits(store, points, config, progress)
    from .reassessment_report import generate_reassessment_report
    report = generate_reassessment_report(store.root)
    counts = {"total": len(points), "correct": sum(point["correct"] for point in points),
              "failed": sum(not point["correct"] for point in points),
              "reconstructed": sum(point["reassessment"]["origin"] == "reconstructed" for point in points),
              "full_circuits": len(full_rows), "full_circuits_correct": sum(row["success"] for row in full_rows)}
    return {"output_dir": str(store.root), "report": str(report), "points": points,
            "full_circuits": full_rows, "counts": counts}
