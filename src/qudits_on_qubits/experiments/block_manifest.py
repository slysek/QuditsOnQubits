"""Schema 4 integrity checks for independently sampled Bell blocks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .errors import ExperimentPersistenceError, ExperimentValidationError
from .safety import validate_persisted_strings
from .setting_schedule import SettingSchedule


SCHEMA_VERSION = 4
MEASUREMENT_MODE = "independent_local_uniform_blocks"


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def batch_path(index, name):
    return f"batches/{index:04d}/{name}.json"


def validate_block_manifest(document):
    """Validate state independently of mutable provider objects."""
    try:
        if not isinstance(document, dict) or type(document["schema_version"]) is not int or document["schema_version"] != 4:
            raise ValueError("unsupported block schema")
        if document["measurement_mode"] != MEASUREMENT_MODE:
            raise ValueError("invalid measurement mode")
        if not isinstance(document["experiment_id"], str) or not document["experiment_id"]:
            raise ValueError("invalid experiment identity")
        if not isinstance(document["artifacts"], dict) or not isinstance(document["batches"], list):
            raise ValueError("invalid artifact index")
        if not isinstance(document["execution_options"], dict) or value_hash(document["execution_options"]) != document["options_hash"]:
            raise ValueError("invalid execution options hash")
        for path, digest in document["artifacts"].items():
            if not isinstance(path, str) or not _digest(digest):
                raise ValueError("invalid artifact hash")
        for index, batch in enumerate(document["batches"]):
            if type(batch["batch_index"]) is not int or batch["batch_index"] != index or not _digest(batch["request_hash"]):
                raise ValueError("invalid batch index")
            if batch["status"] not in {"not_submitted", "submission_unknown", "submitted", "completed", "invalid_result"}:
                raise ValueError("invalid batch status")
        if document["preparation"]["status"] not in {"preparing", "complete", "coverage_limit_reached", "failed"}:
            raise ValueError("invalid preparation status")
        if document["execution"]["status"] not in {"not_started", "running", "partial", "complete", "failed", "submission_unknown"}:
            raise ValueError("invalid execution status")
        if document["analysis"]["status"] not in {"not_started", "partial", "failed", "complete"}:
            raise ValueError("invalid analysis status")
        if document["execution"]["status"] == "complete" and (not document["batches"] or any(b["status"] != "completed" for b in document["batches"])):
            raise ValueError("complete execution requires completed batches")
        validate_persisted_strings(document, description="block manifest", error_type=ExperimentPersistenceError)
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentPersistenceError("invalid schema 4 block manifest") from error
    return document


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def save_block_manifest(store, run, document):
    validate_block_manifest(document)
    store.write_plain_json(run, "experiment.json", document)


def load_block_manifest(store, run):
    return validate_block_manifest(store.read_plain_json(run, "experiment.json"))


def verify_block_artifacts(store, run, document):
    """Recompute coverage and bind every request position to its saved circuit.

    Auxiliary submission/receipt files may be newer than the manifest after a
    crash. The runner verifies these against immutable request hashes on read.
    """
    validate_block_manifest(document)
    if document["experiment_id"] != run.name:
        raise ExperimentPersistenceError("experiment directory identity mismatch")
    for path, expected in document["artifacts"].items():
        if store.artifact_sha256(run, path) != expected:
            raise ExperimentPersistenceError(f"artifact hash mismatch: {path}")
    try:
        reference = get_reference_experiment(document["spec"]["state"])
        if "schedule.json" not in document["artifacts"] and document["preparation"]["status"] in {"preparing", "failed"} and not document["batches"]:
            return None, None, ()
        required = {"reference.json", "schedule.json"}
        if document["preparation"]["status"] == "complete":
            required.update({"catalog.json", "logical-catalog.qpy", "compiled-catalog.qpy", "source-state.qpy", "source-encoding.json"})
            required.update(batch_path(i, "request") for i in range(len(document["batches"])))
            for b in document["batches"]:
                if b["status"] == "completed":
                    required.update(batch_path(b["batch_index"], name) for name in ("counts", "receipt"))
                    if not any(batch_path(b["batch_index"], name) in document["artifacts"] for name in ("submission", "recovery")):
                        raise ValueError("completed batch lacks submission provenance")
        if not required.issubset(document["artifacts"]):
            raise ValueError("required artifact hashes missing")
        saved_reference = store.read_plain_json(run, "reference.json")
        if saved_reference != reference.to_dict() or document["reference_hash"] != reference.stable_hash():
            raise ValueError("reference changed; analysis requires the saved code version")
        schedule = SettingSchedule.from_safe_dict(store.read_plain_json(run, "schedule.json"))
        if schedule.state != reference.experiment_id or schedule.config.to_safe_dict() != document["spec"]["measurement"]:
            raise ValueError("schedule/spec mismatch")
        if not schedule.complete:
            if document["batches"] or document["preparation"]["status"] != "coverage_limit_reached":
                raise ValueError("incomplete schedule has execution requests")
            return schedule, None, ()
        if document["preparation"]["status"] != "complete":
            return schedule, None, ()
        catalog = store.read_plain_json(run, "catalog.json")
        logical = store.read_circuits(run, "logical-catalog.qpy")
        compiled = store.read_circuits(run, "compiled-catalog.qpy")
        settings = [list(s) for s in dict.fromkeys(b.settings for b in schedule.blocks)]
        if catalog["settings"] != settings or len(compiled) != len(settings) or len(logical) != len(settings):
            raise ValueError("catalogue settings mismatch")
        expected_map = [settings.index(list(b.settings)) for b in schedule.blocks]
        if catalog["catalog_index_by_block_id"] != expected_map:
            raise ValueError("block catalogue mapping mismatch")
        if len(catalog["bit_indices"]) != len(settings):
            raise ValueError("decoder catalogue length mismatch")
        expected_outcomes = {str(k): v for k, v in reference.outcome_convention.measurement_basis_index_map}
        if catalog["outcome_map"] != expected_outcomes or any(type(v) not in (int, type(None)) for v in catalog["outcome_map"].values()):
            raise ValueError("measurement outcome convention mismatch")
        for i, circuit in enumerate(compiled):
            if circuit.num_clbits != logical[i].num_clbits or circuit.num_clbits != catalog["classical_widths"][i]:
                raise ValueError("compiled classical width mismatch")
            indices = catalog["bit_indices"][i]
            flat = [x for pair in indices for x in pair]
            measured = [circuit.find_bit(item.clbits[0]).index for item in circuit.data if item.operation.name == "measure"]
            if len(indices) != len(schedule.local_settings) or any(len(pair) != 2 for pair in indices):
                raise ValueError("invalid decoder pairs")
            if any(type(x) is not int or not 0 <= x < circuit.num_clbits for x in flat) or len(set(flat)) != len(flat) or not set(flat).issubset(measured):
                raise ValueError("invalid decoder bit indices")
        visited = []
        provider_limit = document["preparation"]["capabilities"]["max_circuits"]
        limit = min(schedule.config.max_circuits_per_job, provider_limit or schedule.config.max_circuits_per_job)
        for index, batch in enumerate(document["batches"]):
            request = store.read_plain_json(run, batch_path(index, "request"))
            if value_hash(request) != batch["request_hash"]:
                raise ValueError("request hash mismatch")
            if request["batch_index"] != index or request["schedule_hash"] != document["artifacts"]["schedule.json"] or request["catalog_hash"] != document["artifacts"]["compiled-catalog.qpy"] or request["mapping_hash"] != document["artifacts"]["catalog.json"]:
                raise ValueError("request preparation mismatch")
            if type(request["shots"]) is not int or request["shots"] != schedule.config.shots_per_draw or request["backend"] != document["backend"]:
                raise ValueError("request backend/budget mismatch")
            if len(request["positions"]) != min(limit, len(schedule.blocks) - len(visited)):
                raise ValueError("invalid batch size")
            for position, entry in enumerate(request["positions"]):
                block_id = len(visited)
                if any(type(v) is not int for v in entry.values()) or entry != {"position": position, "block_id": block_id, "catalog_index": expected_map[block_id]}:
                    raise ValueError("request positions do not partition schedule")
                visited.append(block_id)
        if visited != list(range(len(schedule.blocks))):
            raise ValueError("requests do not cover schedule")
        return schedule, catalog, compiled
    except (KeyError, IndexError, TypeError, ValueError, ExperimentValidationError) as error:
        raise ExperimentPersistenceError("invalid saved block preparation") from error
