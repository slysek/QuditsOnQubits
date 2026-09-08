"""Durable raw Bell execution with one persisted result per sampled block.

The schedule is fixed before submission. A submission marker is never deleted:
an interrupted attempt cannot silently turn back into an unsubmitted request.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import inspect
import secrets
from uuid import uuid4

import numpy as np

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .artifacts import load_basis_artifacts
from .backends import BackendIdentity, BackendCapabilities, ExecutionResult, SubmittedJob, create_backend_adapter
from .backends.aer import AerAdapter
from .block_estimation import evaluate_blocks
from .block_manifest import (
    MEASUREMENT_MODE, batch_path, value_hash, save_block_manifest,
    load_block_manifest, verify_block_artifacts,
)
from .errors import BackendCompatibilityError, ExperimentPersistenceError, ExperimentValidationError
from .measurement import RandomizedBlocks
from .models import AerIdeal, ExperimentSpec, ExperimentResult, ExperimentStatus, PiastQHardware, TranspilationConfig
from .preparation import prepare_measurements
from .safety import validate_persisted_strings
from .setting_schedule import generate_schedule
from .store import ExperimentStore


def _now():
    return datetime.now(timezone.utc)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _options(spec, timeout, run_options):
    from .runner import _validate_execution_options
    _validate_execution_options(timeout, run_options)
    if not isinstance(spec.measurement, RandomizedBlocks):
        raise ExperimentValidationError("randomized execution requires RandomizedBlocks")
    if any((spec.mitigation.readout, spec.mitigation.zne, spec.mitigation.circuit_twirling, spec.mitigation.force_recalibration)):
        raise ExperimentValidationError("randomized blocks v1 supports raw measurements only")
    options = dict(run_options or {})
    # Options cannot change the statistical unit or turn on mitigation. Keep a
    # deliberately small, inspectable provider execution surface in raw v1.
    allowed = {"max_execution_time", "execution"} if spec.backend.to_safe_dict()["kind"] == "ibm_hardware" else set()
    if set(options) - allowed:
        raise ExperimentValidationError("unsupported run_options in randomized raw mode")
    ExperimentStore.validate_plain_json(options)
    validate_persisted_strings(options, description="raw execution options", error_type=ExperimentValidationError)
    if isinstance(spec.backend, PiastQHardware) and (spec.transpilation != TranspilationConfig() or spec.workload_optimization is not None):
        raise BackendCompatibilityError("PIAST managed compilation requires default local compilation options")
    return options


def _write(store, run, doc, filename, value):
    store.write_plain_json(run, filename, value)
    doc["artifacts"][filename] = store.artifact_sha256(run, filename)


def _save(store, run, doc, clock):
    doc["updated_at"] = clock().isoformat()
    save_block_manifest(store, run, doc)


def _result(run, doc):
    preparation, execution, analysis = (doc[k] for k in ("preparation", "execution", "analysis"))
    if execution["status"] == "submission_unknown":
        status = ExperimentStatus.SUBMISSION_UNKNOWN
    elif execution["status"] == "complete" and analysis["status"] == "complete":
        status = ExperimentStatus.COMPLETED
    elif execution["status"] == "complete":
        status = ExperimentStatus.POSTPROCESSING
    else:
        status = ExperimentStatus.FAILED
    values = {"measurement_mode": MEASUREMENT_MODE, "raw": None, "conditional": None,
              **(doc.get("result") or {}), "preparation": dict(preparation),
              "execution": dict(execution), "analysis": dict(analysis)}
    replays = [b["local_replay"] for b in doc["batches"] if "local_replay" in b]
    if replays:
        values["execution"]["local_replays"] = replays
    return ExperimentResult(run.name, status, run, values, doc.get("backend") or {},
                            tuple(dict.fromkeys(job_id for b in doc["batches"]
                                  for job_id in (*b.get("previous_job_ids", ()), b.get("job_id")) if job_id)))


def _same_backend(left, right):
    # SDK/calibration versions may change between retrievals; provider and
    # configured target must not. Immutable requests retain the original data.
    return all(left.get(k) == right.get(k) for k in ("kind", "name", "provider", "emulates"))


def _safe_identity(identity):
    value = identity.to_safe_dict()
    for key in tuple(value):
        try:
            validate_persisted_strings({key: value[key]}, description="provider identity", error_type=ExperimentValidationError)
        except ExperimentValidationError:
            value[key] = {} if key == "metadata" else None
    return value


def _submission_path(store, run, index):
    recovery = batch_path(index, "recovery")
    return recovery if store.artifact_exists(run, recovery) else batch_path(index, "submission")


def _reconcile_local_replay(store, run, doc, batch, request):
    """Rebuild attempt history from durable evidence before updating job_id."""
    index = batch["batch_index"]
    marker_path = batch_path(index, "local-replay-attempt")
    if not store.artifact_exists(run, marker_path):
        return
    marker = store.read_plain_json(run, marker_path)
    if (marker.get("request_hash") != batch["request_hash"]
        or type(marker.get("additional_requested_shots")) is not int
        or marker["additional_requested_shots"] != len(request["positions"])*request["shots"]
        or marker.get("seed_simulator") != request["seed_simulator"]):
        raise ExperimentPersistenceError("local replay history does not match request")
    original_path = batch_path(index, "submission")
    original_id = marker.get("previous_job_id")
    if store.artifact_exists(run, original_path):
        original = store.read_plain_json(run, original_path)
        if original.get("request_hash") != batch["request_hash"]:
            raise ExperimentPersistenceError("original submission history mismatch")
        original_id = original["job_id"]
    marker["previous_job_id"] = original_id
    if original_id is not None:
        batch["previous_job_ids"] = list(dict.fromkeys((*batch.get("previous_job_ids", ()), original_id)))
    batch["local_replay"] = marker
    doc["artifacts"][marker_path] = store.artifact_sha256(run, marker_path)


def _batch_circuits(compiled, request):
    circuits = []
    for entry in request["positions"]:
        circuit = compiled[entry["catalog_index"]].copy()
        circuit.name = f"block_{entry['block_id']:08d}"
        circuits.append(circuit)
    return tuple(circuits)


def _batch_adapter(adapter, request):
    if type(adapter) is AerAdapter:
        return AerAdapter(AerIdeal(seed_simulator=request["seed_simulator"]))
    return adapter


def _validate_job(job, request, expected_id=None):
    if not isinstance(job, SubmittedJob):
        raise BackendCompatibilityError("adapter submit/restore must return SubmittedJob")
    if expected_id is not None and job.job_id != expected_id:
        raise BackendCompatibilityError("restored job identity mismatch")
    if not _same_backend(job.target_identity.to_safe_dict(), request["backend"]):
        raise BackendCompatibilityError("submitted backend mismatch")
    if job.circuit_count != len(request["positions"]) or job.shots != request["shots"]:
        raise BackendCompatibilityError("submitted circuit count or shots mismatch")


def _validate_evidence(evidence, request, catalog, job_id):
    if evidence.get("provider_validation_error") is True:
        raise ExperimentValidationError("provider reported invalid raw evidence")
    if evidence["job_id"] != job_id or not _same_backend(evidence["target_identity"], request["backend"]):
        raise ExperimentValidationError("result job/backend mismatch")
    counts = evidence["counts"]
    if len(counts) != len(request["positions"]):
        raise ExperimentValidationError("result circuit count mismatch")
    for entry, histogram in zip(request["positions"], counts, strict=True):
        width = catalog["classical_widths"][entry["catalog_index"]]
        if not isinstance(histogram, dict) or not histogram:
            raise ExperimentValidationError("missing raw counts")
        canonical = set()
        for bits, count in histogram.items():
            if not isinstance(bits, str):
                raise ExperimentValidationError("raw count key must be a bitstring")
            compact = bits.replace(" ", "")
            if len(compact) != width or set(compact) - {"0", "1"} or compact in canonical:
                raise ExperimentValidationError("raw bitstring width or uniqueness mismatch")
            canonical.add(compact)
            if type(count) is not int or count < 0:
                raise ExperimentValidationError("raw counts must be nonnegative integers")
        if sum(histogram.values()) != request["shots"]:
            raise ExperimentValidationError("raw block shots do not equal K")


def _read_batch(store, run, doc, batch, request, catalog):
    """Reconcile a counts checkpoint even if its receipt/manifest write failed."""
    index = batch["batch_index"]
    submission_path = _submission_path(store, run, index)
    if not store.artifact_exists(run, submission_path):
        return None
    submission = store.read_plain_json(run, submission_path)
    if submission["request_hash"] != batch["request_hash"] or not isinstance(submission["job_id"], str):
        raise ExperimentPersistenceError("submission does not match request")
    _reconcile_local_replay(store, run, doc, batch, request)
    batch["job_id"] = submission["job_id"]
    doc["artifacts"][submission_path] = store.artifact_sha256(run, submission_path)
    if not _valid_submission(submission, request):
        batch["status"] = "submission_unknown"
        return None
    counts_path = batch_path(index, "counts")
    if not store.artifact_exists(run, counts_path):
        return None
    evidence = store.read_plain_json(run, counts_path)
    receipt_path = batch_path(index, "receipt")
    digest = store.artifact_sha256(run, counts_path)
    receipt = None
    if store.artifact_exists(run, receipt_path):
        receipt = store.read_plain_json(run, receipt_path)
        if receipt["request_hash"] != batch["request_hash"] or receipt["counts_hash"] != digest or receipt["job_id"] != submission["job_id"]:
            raise ExperimentPersistenceError("receipt hash or identity mismatch")
    valid, reason = True, None
    try:
        _validate_evidence(evidence, request, catalog, submission["job_id"])
    except (KeyError, TypeError, ValueError, ExperimentValidationError):
        valid, reason = False, "invalid_raw_result"
    if receipt is not None and (receipt["valid"] is not valid or receipt["reason"] != reason):
        raise ExperimentPersistenceError("receipt validation mismatch")
    if receipt is None:
        store.write_plain_json(run, receipt_path, {"request_hash": batch["request_hash"], "counts_hash": digest,
                              "job_id": submission["job_id"], "valid": valid, "reason": reason})
    doc["artifacts"][counts_path] = digest
    doc["artifacts"][receipt_path] = store.artifact_sha256(run, receipt_path)
    batch["status"] = "completed" if valid else "invalid_result"
    if not valid:
        return None
    return {e["block_id"]: c for e, c in zip(request["positions"], evidence["counts"], strict=True)}


def _valid_submission(submission, request):
    return (isinstance(submission.get("target_identity"), dict)
            and _same_backend(submission["target_identity"], request["backend"])
            and type(submission.get("shots")) is int and submission["shots"] == request["shots"]
            and type(submission.get("circuit_count")) is int and submission["circuit_count"] == len(request["positions"]))


def _analyze(store, run, doc, schedule, catalog, counts, clock):
    if catalog is None:
        return
    try:
        reference = get_reference_experiment(schedule.state)
        values = evaluate_blocks(reference, schedule, counts,
            {tuple(s): pairs for s, pairs in zip(catalog["settings"], catalog["bit_indices"], strict=True)},
            {int(k): v for k, v in catalog["outcome_map"].items()})
        # Immutable analysis generations keep an old manifest valid if a crash
        # interrupts a partial-to-complete update. The alias is only for users.
        analysis_path = f"analysis/bell-{value_hash(values)}.json"
        _write(store, run, doc, analysis_path, values)
        store.write_plain_json(run, "analysis/bell.json", values)
        doc["result"] = values
        doc["analysis"] = {"status": "complete" if len(counts) == len(schedule.blocks) else "partial", "artifact": analysis_path}
    except ExperimentPersistenceError:
        raise
    except Exception:
        doc["analysis"] = {"status": "failed", "reason": "analysis_failed_raw_preserved"}
    _save(store, run, doc, clock)


def run_randomized_experiment(spec, *, adapter=None, repo_root=None, timeout=None, run_options=None,
                              _clock=_now, _randbelow=None, _source="system_secrets", _seed=None):
    options = _options(spec, timeout, run_options)
    reference = get_reference_experiment(spec.state)
    store = ExperimentStore(spec.output_root)
    run = store.create_run()
    doc = {"schema_version": 4, "measurement_mode": MEASUREMENT_MODE, "experiment_id": run.name,
           "spec": spec.to_safe_dict(), "reference_hash": reference.stable_hash(),
           "preparation": {"status": "preparing"}, "execution": {"status": "not_started"},
           "analysis": {"status": "not_started"}, "backend": None, "artifacts": {}, "batches": [],
           "result": None, "created_at": _clock().isoformat(), "execution_options": options, "options_hash": value_hash(options)}
    try:
        _write(store, run, doc, "reference.json", reference.to_dict())
        _save(store, run, doc, _clock)
        schedule = generate_schedule(reference, spec.measurement, _randbelow=_randbelow, _source=_source, _seed=_seed)
        _write(store, run, doc, "schedule.json", schedule.to_safe_dict())
        if not schedule.complete:
            doc["preparation"] = {"status": "coverage_limit_reached", "missing_pattern_indices": list(schedule.missing_pattern_indices)}
            doc["result"] = {"raw": None, "conditional": None, "raw_reason": "coverage_limit_reached", "conditional_reason": "coverage_limit_reached",
                "coverage": {"missing_pattern_indices": list(schedule.missing_pattern_indices), "complete": False,
                             "zero_contribution_blocks": sum(not b.matched_pattern_indices for b in schedule.blocks)},
                "budget": {"minimum_blocks": spec.measurement.setting_draws, "maximum_blocks": spec.measurement.max_setting_draws,
                           "scheduled_blocks": len(schedule.blocks), "shots_per_draw": spec.measurement.shots_per_draw,
                           "scheduled_shots": len(schedule.blocks)*spec.measurement.shots_per_draw, "completed_shots": 0}}
            _save(store, run, doc, _clock)
            return _result(run, doc)
        _save(store, run, doc, _clock)
        artifacts = load_basis_artifacts(spec.basis, spec.state, repo_root)
        prepared = prepare_measurements(artifacts, schedule=schedule)
        resolved = create_backend_adapter(spec.backend) if adapter is None else adapter
        identity = resolved.resolve()
        if not isinstance(identity, BackendIdentity):
            raise BackendCompatibilityError("adapter identity is invalid")
        from .runner import _compile_measurement_workload, _validate_adapter_target, _physical_qubit_mappings
        _validate_adapter_target(spec, identity)
        doc["backend"] = identity.to_safe_dict()
        capabilities = resolved.capabilities()
        if not isinstance(capabilities, BackendCapabilities):
            raise BackendCompatibilityError("adapter capabilities are invalid")
        if not capabilities.local and not capabilities.supports_resume:
            raise BackendCompatibilityError("remote blocks require retrievable job identifiers")
        if not resolved.availability().available:
            raise BackendCompatibilityError("backend unavailable before submission")
        settings = tuple(tuple(s) for s in prepared.metadata["setting_by_circuit_index"])
        selection = _compile_measurement_workload(resolved, prepared.circuits, settings, spec, expected_identity=identity)
        compiled = selection.batch.circuits
        for filename, circuits in (("source-state.qpy", (artifacts.state_circuit,)),
                                   ("logical-catalog.qpy", prepared.circuits), ("compiled-catalog.qpy", compiled)):
            doc["artifacts"][filename] = store.write_circuits(run, circuits, filename)
        store.write_json(run, "source-encoding.json", artifacts.encoding)
        doc["artifacts"]["source-encoding.json"] = store.artifact_sha256(run, "source-encoding.json")
        catalog_map = [settings.index(b.settings) for b in schedule.blocks]
        catalog = {"settings": [list(s) for s in settings], "catalog_index_by_block_id": catalog_map,
                   "bit_indices": [_plain(prepared.metadata["qutrit_bit_indices_by_setting"][s]) for s in settings],
                   "outcome_map": _plain(prepared.metadata["encoding_outcome_map"]),
                   "classical_widths": [c.num_clbits for c in compiled],
                   "compilation": _plain(selection.batch.metadata), "workload_optimization": _plain(selection.metadata),
                   "physical_qubit_mappings": _plain(selection.physical_mappings or _physical_qubit_mappings(compiled)),
                   "chronology": "provider may reorder circuits or shots within a job"}
        _write(store, run, doc, "catalog.json", catalog)
        limit = min(spec.measurement.max_circuits_per_job, capabilities.max_circuits or spec.measurement.max_circuits_per_job)
        seed_base = spec.backend.seed_simulator if isinstance(spec.backend, AerIdeal) else None
        if isinstance(spec.backend, AerIdeal) and seed_base is None:
            seed_base = secrets.randbits(32)
        for index, start in enumerate(range(0, len(schedule.blocks), limit)):
            positions = [{"position": p, "block_id": b, "catalog_index": catalog_map[b]}
                         for p, b in enumerate(range(start, min(start+limit, len(schedule.blocks))))]
            request = {"batch_index": index, "positions": positions, "shots": spec.measurement.shots_per_draw,
                       "backend": doc["backend"], "schedule_hash": doc["artifacts"]["schedule.json"],
                       "catalog_hash": doc["artifacts"]["compiled-catalog.qpy"], "mapping_hash": doc["artifacts"]["catalog.json"],
                       "seed_simulator": int(np.random.SeedSequence([seed_base, index]).generate_state(1)[0]) if seed_base is not None else None}
            _batch_adapter(resolved, request).preflight(_batch_circuits(compiled, request), request["shots"])
            validator = getattr(resolved, "validate_run_options", None)
            if callable(validator):
                validator(options)
            _write(store, run, doc, batch_path(index, "request"), request)
            doc["batches"].append({"batch_index": index, "request_hash": value_hash(request), "status": "not_submitted", "job_id": None})
        doc["preparation"] = {"status": "complete", "seed_simulator_base": seed_base, "capabilities": capabilities.to_safe_dict()}
        _save(store, run, doc, _clock)
        verify_block_artifacts(store, run, doc)
        return _execute(store, run, doc, spec, schedule, catalog, compiled, resolved, options, timeout, _clock)
    except ExperimentPersistenceError as error:
        setattr(error, "__qoq_artifact_dir__", run)
        raise
    except Exception as error:
        if doc["preparation"]["status"] == "complete":
            doc["execution"] = {"status": "partial", "reason": "execution_requires_attention"}
        else:
            doc["preparation"] = {"status": "failed", "reason": "preparation_failed", "error_type": type(error).__name__}
        _save(store, run, doc, _clock)
        return _result(run, doc)


def _execute(store, run, doc, spec, schedule, catalog, compiled, adapter, options, timeout, clock, _live_jobs=None):
    counts = {}
    for batch in doc["batches"]:
        index = batch["batch_index"]
        request = store.read_plain_json(run, batch_path(index, "request"))
        saved = _read_batch(store, run, doc, batch, request, catalog)
        if saved is not None:
            counts.update(saved)
            continue
        if batch["status"] == "invalid_result":
            doc["execution"] = {"status": "partial", "reason": "invalid_raw_result"}
            break
        submission_path = _submission_path(store, run, index)
        attempted = store.artifact_exists(run, batch_path(index, "attempt"))
        confirmed = store.artifact_exists(run, submission_path)
        if store.artifact_exists(run, batch_path(index, "local-replay-attempt")) and not store.artifact_exists(run, batch_path(index, "recovery")):
            _reconcile_local_replay(store, run, doc, batch, request)
            batch["status"] = "submission_unknown"
            doc["execution"] = {"status": "submission_unknown", "batch_index": index, "reason": "local_replay_unknown"}
            break
        if confirmed and not _valid_submission(store.read_plain_json(run, submission_path), request):
            batch["status"] = "submission_unknown"
            doc["execution"] = {"status": "submission_unknown", "batch_index": index, "reason": "submission_not_verified"}
            break
        if attempted and not confirmed:
            batch["status"] = "submission_unknown"
            doc["execution"] = {"status": "submission_unknown", "batch_index": index}
            break
        if adapter is None:
            adapter = create_backend_adapter(spec.backend)
        current = _batch_adapter(adapter, request)
        if not _same_backend(current.resolve().to_safe_dict(), request["backend"]):
            raise BackendCompatibilityError("resume adapter backend mismatch")
        circuits = _batch_circuits(compiled, request)
        if confirmed:
            submission = store.read_plain_json(run, submission_path)
            try:
                kwargs = {"circuit_count": len(circuits), "shots": request["shots"]}
                if "circuits" in inspect.signature(current.restore_job).parameters:
                    kwargs["circuits"] = circuits
                job = (_live_jobs or {}).get(index)
                if job is None:
                    job = current.restore_job(submission["job_id"], **kwargs)
                _validate_job(job, request, submission["job_id"])
            except Exception:
                doc["execution"] = {"status": "partial", "reason": "job_retrieval_requires_attention", "batch_index": index}
                break
        else:
            current.preflight(circuits, request["shots"])
            validator = getattr(current, "validate_run_options", None)
            if callable(validator):
                validator(options)
            marker = {"attempt_id": uuid4().hex, "request_hash": batch["request_hash"], "started_at": clock().isoformat()}
            if not store.write_exclusive_json(run, batch_path(index, "attempt"), marker):
                doc["execution"] = {"status": "submission_unknown", "batch_index": index}
                batch["status"] = "submission_unknown"
                break
            batch["status"] = "submission_unknown"
            doc["execution"] = {"status": "running"}
            _save(store, run, doc, clock)
            try:
                job = current.submit(circuits, request["shots"], options=options)
                # Persist the returned identity immediately, before validating
                # counts/K. Even a malformed response must not cause resubmit.
                if isinstance(job, SubmittedJob):
                    submission = {"job_id": job.job_id, "request_hash": batch["request_hash"],
                                  "attempt_id": marker["attempt_id"], "confirmed_at": clock().isoformat(),
                                  "target_identity": _safe_identity(job.target_identity), "shots": job.shots, "circuit_count": job.circuit_count}
                    _write(store, run, doc, submission_path, submission)
                    batch["job_id"] = job.job_id
                _validate_job(job, request)
            except ExperimentPersistenceError:
                raise
            except Exception:
                doc["execution"] = {"status": "submission_unknown", "batch_index": index, "reason": "submission_not_verified"}
                break
            batch["status"] = "submitted"
            _save(store, run, doc, clock)
        try:
            execution = current.result(job, timeout=timeout)
        except Exception as error:
            evidence = getattr(error, "raw_evidence", None)
            if isinstance(evidence, Mapping):
                evidence = _plain(evidence)
                evidence["provider_validation_error"] = True
                # Adapters expose only sanitized count evidence, never SDK
                # payloads. Repeat boundary checks before writing that evidence.
                ExperimentStore.validate_plain_json(evidence)
                validate_persisted_strings(evidence, description="partial provider counts", error_type=ExperimentPersistenceError)
                _write(store, run, doc, batch_path(index, "counts"), evidence)
                _read_batch(store, run, doc, batch, request, catalog)
                doc["execution"] = {"status": "partial", "batch_index": index, "reason": "invalid_raw_result"}
                break
            doc["execution"] = {"status": "partial", "batch_index": index, "reason": "result_unavailable_job_preserved"}
            break
        if not isinstance(execution, ExecutionResult):
            doc["execution"] = {"status": "partial", "batch_index": index, "reason": "invalid_result_type"}
            break
        evidence = execution.to_safe_dict()
        evidence["target_identity"] = _safe_identity(execution.target_identity)
        # Store the raw measurements even when provider metadata is unsafe.
        try:
            validate_persisted_strings(evidence, description="raw evidence", error_type=ExperimentValidationError)
        except ExperimentValidationError:
            evidence["metadata"] = {"reason": "unsafe_metadata_omitted"}
            evidence["timing"] = {}
        _write(store, run, doc, batch_path(index, "counts"), evidence)
        saved = _read_batch(store, run, doc, batch, request, catalog)
        _save(store, run, doc, clock)
        if saved is None:
            doc["execution"] = {"status": "partial", "batch_index": index, "reason": "invalid_raw_result"}
            break
        counts.update(saved)
    else:
        doc["execution"] = {"status": "complete"}
    _save(store, run, doc, clock)
    _analyze(store, run, doc, schedule, catalog, counts, clock)
    return _result(run, doc)


def resume_randomized_experiment(experiment_dir, *, adapter=None, timeout=None, run_options=None, spec=None, _clock=_now):
    store, run = ExperimentStore.open_existing_run(experiment_dir)
    doc = load_block_manifest(store, run)
    schedule, catalog, compiled = verify_block_artifacts(store, run, doc)
    if spec is not None and spec.to_safe_dict() != doc["spec"]:
        raise ExperimentValidationError("resume spec differs from saved preparation")
    if catalog is None:
        return _result(run, doc)
    # Completed runs require only local artifacts, regardless of adapter input.
    if doc["execution"]["status"] == "complete" and doc["analysis"]["status"] == "complete":
        for batch in doc["batches"]:
            request = store.read_plain_json(run, batch_path(batch["batch_index"], "request"))
            if _read_batch(store, run, doc, batch, request, catalog) is None:
                raise ExperimentPersistenceError("completed experiment has missing or invalid raw evidence")
        if store.read_plain_json(run, doc["analysis"]["artifact"]) != doc["result"]:
            raise ExperimentPersistenceError("analysis artifact and manifest disagree")
        return _result(run, doc)
    saved_spec = spec if spec is not None else ExperimentSpec.from_safe_dict(doc["spec"])
    options = _options(saved_spec, timeout, doc["execution_options"] if run_options is None else run_options)
    if value_hash(options) != doc["options_hash"]:
        raise ExperimentValidationError("resume requires the original raw execution options")
    return _execute(store, run, doc, saved_spec, schedule, catalog, compiled, adapter, options, timeout, _clock)


def recover_randomized_job(experiment_dir, *, batch_index, job_id, adapter, _clock=_now):
    """Attach a provider-verified job to an uncertain attempt, without submitting.

    Providers must prove remote circuit identity, K, backend and raw options.
    A local wrapper filled from this request is insufficient evidence. IBM's
    adapter implements this verifier; providers without it fail explicitly.
    Call resume_experiment afterwards to retrieve the job's existing results.
    """
    store, run = ExperimentStore.open_existing_run(experiment_dir)
    doc = load_block_manifest(store, run)
    _, catalog, compiled = verify_block_artifacts(store, run, doc)
    if type(batch_index) is not int or not 0 <= batch_index < len(doc["batches"]):
        raise ExperimentValidationError("invalid recovery batch_index")
    batch = doc["batches"][batch_index]
    if catalog is None or batch["status"] not in {"submission_unknown", "not_submitted"} or not store.artifact_exists(run, batch_path(batch_index, "attempt")):
        raise ExperimentValidationError("recovery requires an uncertain persisted submission attempt")
    if store.artifact_exists(run, batch_path(batch_index, "counts")):
        raise ExperimentValidationError("existing raw evidence cannot be replaced by recovery")
    verifier = getattr(adapter, "verify_restored_job", None)
    if not callable(verifier):
        raise BackendCompatibilityError("provider cannot verify remote circuits, shots, backend and raw options for unknown-job recovery")
    request = store.read_plain_json(run, batch_path(batch_index, "request"))
    circuits = _batch_circuits(compiled, request)
    job = adapter.restore_job(job_id, circuit_count=len(circuits), shots=request["shots"], circuits=circuits)
    _validate_job(job, request, job_id)
    verifier(job, circuits=circuits, shots=request["shots"])
    recovered = {"job_id": job.job_id, "request_hash": batch["request_hash"], "target_identity": _safe_identity(job.target_identity),
                 "shots": job.shots, "circuit_count": job.circuit_count, "confirmed_at": _clock().isoformat(), "recovery": "provider_verified"}
    filename = batch_path(batch_index, "recovery")
    if not store.write_exclusive_json(run, filename, recovered):
        raise ExperimentPersistenceError("a recovery record already exists; resume that job")
    doc["artifacts"][filename] = store.artifact_sha256(run, filename)
    batch.update(status="submitted", job_id=job.job_id)
    doc["execution"] = {"status": "partial", "reason": "verified_job_ready_for_retrieval"}
    _save(store, run, doc, _clock)
    return _result(run, doc)


def replay_randomized_aer_batch(experiment_dir, *, batch_index, timeout=None, _clock=_now):
    """Explicitly replay one lost local Aer batch using its saved seed and QPY.

    Remote targets and existing raw evidence are rejected. Original submission
    evidence and the additional requested shots remain in the replay record.
    A second uncertain replay cannot be submitted again by this operation.
    """
    store, run = ExperimentStore.open_existing_run(experiment_dir)
    doc = load_block_manifest(store, run)
    schedule, catalog, compiled = verify_block_artifacts(store, run, doc)
    spec = ExperimentSpec.from_safe_dict(doc["spec"])
    if not isinstance(spec.backend, AerIdeal) or doc["backend"]["kind"] != "aer_ideal":
        raise BackendCompatibilityError("explicit local replay is available only for AerIdeal")
    options = _options(spec, timeout, doc["execution_options"])
    if catalog is None or type(batch_index) is not int or not 0 <= batch_index < len(doc["batches"]):
        raise ExperimentValidationError("invalid local replay batch")
    if not store.artifact_exists(run, batch_path(batch_index, "attempt")) or any(
        store.artifact_exists(run, batch_path(batch_index, name)) for name in ("counts", "recovery")
    ):
        raise ExperimentValidationError("local replay requires a lost original attempt without raw or recovery evidence")
    batch = doc["batches"][batch_index]
    request = store.read_plain_json(run, batch_path(batch_index, "request"))
    adapter = AerAdapter(AerIdeal(seed_simulator=request["seed_simulator"]))
    if not _same_backend(adapter.resolve().to_safe_dict(), request["backend"]):
        raise BackendCompatibilityError("saved batch is not compatible with the local Aer target")
    circuits = _batch_circuits(compiled, request)
    adapter.preflight(circuits, request["shots"])
    original_path = batch_path(batch_index, "submission")
    original_id = batch.get("job_id")
    if store.artifact_exists(run, original_path):
        original = store.read_plain_json(run, original_path)
        if original.get("request_hash") != batch["request_hash"]:
            raise ExperimentPersistenceError("original submission history mismatch")
        original_id = original["job_id"]
    marker = {"request_hash": batch["request_hash"], "previous_job_id": original_id,
              "seed_simulator": request["seed_simulator"], "additional_requested_shots": len(circuits)*request["shots"],
              "reason": "explicit_deterministic_local_replay", "started_at": _clock().isoformat()}
    if not store.write_exclusive_json(run, batch_path(batch_index, "local-replay-attempt"), marker):
        raise ExperimentPersistenceError("a local replay attempt already exists; it will not be submitted again")
    _reconcile_local_replay(store, run, doc, batch, request)
    doc["execution"] = {"status": "submission_unknown", "batch_index": batch_index, "reason": "local_replay_in_progress"}
    _save(store, run, doc, _clock)
    try:
        job = adapter.submit(circuits, request["shots"], options=options)
        _validate_job(job, request)
    except Exception:
        doc["execution"] = {"status": "submission_unknown", "batch_index": batch_index, "reason": "local_replay_unknown"}
        _save(store, run, doc, _clock)
        return _result(run, doc)
    recovery = {**marker, "job_id": job.job_id, "target_identity": _safe_identity(job.target_identity),
                "shots": job.shots, "circuit_count": job.circuit_count, "recovery": "explicit_local_replay"}
    _write(store, run, doc, batch_path(batch_index, "recovery"), recovery)
    batch.update(status="submitted", job_id=job.job_id)
    doc["execution"] = {"status": "partial", "reason": "local_replay_submitted"}
    batch["local_replay"] = marker
    _save(store, run, doc, _clock)
    return _execute(store, run, doc, spec, schedule, catalog, compiled, adapter, options, timeout, _clock, {batch_index: job})
