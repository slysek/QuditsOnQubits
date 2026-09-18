"""One encoding benchmark pipeline, independent of family and provider."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
from pathlib import Path
from uuid import uuid4

import numpy as np
from qiskit.exceptions import QiskitError

from .artifacts import complete_run, create_store
from .config import BenchmarkConfig
from .families import canonical_candidate
from .models import EncodingCandidate
from .report import write_report
from .synthesis import OptimizedSynthesis, SavedGateSynthesis, SynthesisArtifactError
from .targets import BackendTarget
from .theta_continuation.artifacts import fingerprint, runtime_provenance
from .validation import circuit_metrics, state_metrics
from .workloads import LogicalCircuit
from .direct_basis.optimized_gates import validate_code_space_gate
from .direct_basis.math_utils import qutrit_fourier, qutrit_cz


def _provenance():
    provenance = runtime_provenance()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        provenance["source_hashes"][f"qudits_on_qubits/benchmarks/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(package.joinpath("backend_adapters").glob("*.py")):
        provenance["source_hashes"][f"qudits_on_qubits/benchmarks/backend_adapters/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return provenance


def _gate_checks(library, encoding, config):
    if library.f3.num_qubits != 2 or library.cz3.num_qubits != 4:
        raise ValueError("Gate library has incompatible widths")
    values = {}
    for name, gate, basis, logical, tolerance in (
        ("f3", library.f3, encoding, qutrit_fourier(), config.f3_tolerance),
        ("cz3", library.cz3, np.kron(encoding, encoding), qutrit_cz(), config.cz3_tolerance),
    ):
        if gate.num_parameters or gate.num_clbits:
            raise ValueError("Gate library must contain bound unitary gates")
        metric = validate_code_space_gate(gate, basis, logical)
        if not np.isfinite([metric.E_norm, metric.L_norm]).all() or max(metric.E_norm, metric.L_norm) > tolerance:
            raise ValueError(f"{name} exceeds configured action/leakage tolerance")
        values[name] = asdict(metric)
    return values


def run_benchmark(
    circuit: LogicalCircuit, *, families, backend: BackendTarget,
    config: BenchmarkConfig | None = None, synthesis=None, baseline_synthesis=None,
    saved_candidates=(), output_dir: str | Path | None = None, progress=None,
):
    """Compile, route, validate and rank every candidate; never submit hardware jobs."""
    if not isinstance(circuit, LogicalCircuit):
        raise TypeError("circuit must be a LogicalCircuit")
    if not isinstance(backend, BackendTarget):
        raise TypeError("backend must be an explicit BackendTarget")
    config = config if config is not None else BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    if 2 * circuit.num_qutrits > min(12, config.max_validation_qubits, backend.num_qubits):
        raise ValueError("Workload exceeds target or validation qubit capacity")
    families = tuple(families)
    family_specs = [family.to_dict() for family in families]
    candidates = [candidate for family in families for candidate in family.generate()]
    candidates.extend(saved_candidates)
    if not candidates:
        raise ValueError("At least one encoding candidate is required")
    if any(not isinstance(candidate, EncodingCandidate) for candidate in candidates):
        raise TypeError("Families must produce EncodingCandidate objects")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers) or "canonical" in identifiers:
        raise ValueError("Candidate IDs must be unique and cannot reserve canonical")
    synthesis = synthesis if synthesis is not None else OptimizedSynthesis()
    baseline_synthesis = baseline_synthesis if baseline_synthesis is not None else OptimizedSynthesis()
    baseline = canonical_candidate()
    cohorts = {}
    for candidate in candidates:
        if isinstance(synthesis, Mapping):
            if candidate.family not in synthesis:
                raise ValueError(f"No synthesis strategy configured for {candidate.family}")
            strategy = synthesis[candidate.family]
        else:
            strategy = synthesis
        strategy_spec = strategy.to_dict()
        baseline_strategy = baseline_synthesis if isinstance(strategy, SavedGateSynthesis) else strategy
        protocol = {"synthesis": strategy_spec, "baseline_synthesis": baseline_strategy.to_dict(), "config": config.to_dict()}
        key = fingerprint(protocol)
        if key not in cohorts:
            cohorts[key] = dict(strategy=strategy, baseline_strategy=baseline_strategy,
                                protocol=protocol, candidates=[baseline])
        cohorts[key]["candidates"].append(candidate)
    backend_spec = backend.to_dict()
    target_hash = str(backend_spec.get("target_hash") or backend_spec.get("snapshot_hash") or fingerprint(backend_spec))
    workload_hash = circuit.stable_hash()
    cohorts = {fingerprint({"protocol": cohort["protocol"], "backend": backend_spec,
                            "workload_hash": workload_hash}): cohort for cohort in cohorts.values()}
    output = Path(output_dir) if output_dir is not None else Path("artifacts/encoding_benchmarks") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    schedule = []
    for key, cohort in cohorts.items():
        for candidate in cohort["candidates"]:
            schedule.append({"comparison_id": key, **candidate.to_dict(),
                             "encoding_sha256": hashlib.sha256(candidate.encoding.tobytes()).hexdigest()})
    store = create_store(output, {
        "created_at": datetime.now(timezone.utc).isoformat(), "workload": circuit.to_dict(),
        "workload_hash": workload_hash, "config": config.to_dict(),
        "backend": backend_spec, "target_hash": target_hash,
        "families": family_specs, "protocols": {key: value["protocol"] for key, value in cohorts.items()},
        "schedule": schedule, "software": _provenance(),
    })
    rows, bundles = [], []
    candidate_index = 0
    for comparison_id, cohort in cohorts.items():
        strategy, baseline_strategy = cohort["strategy"], cohort["baseline_strategy"]
        group = cohort["candidates"]
        prepare_group = group[1:] if isinstance(strategy, SavedGateSynthesis) else group
        strategy.prepare(prepare_group, store.root / "synthesis" / comparison_id, progress=progress)
        if baseline_strategy is not strategy:
            baseline_strategy.prepare([baseline], store.root / "synthesis" / comparison_id / "baseline", progress=progress)
        for candidate in group:
            if progress is not None:
                progress(f"candidate {candidate_index + 1}/{len(schedule)}: {candidate.candidate_id}")
            candidate_path = f"candidates/{candidate_index:05d}"
            selected_strategy = baseline_strategy if candidate.candidate_id == "canonical" else strategy
            source = None
            gate_metadata = {}
            candidate_error = None
            stage = "synthesis_failed"
            library = None
            try:
                library = selected_strategy.build(candidate)
                gate_metadata = _gate_checks(library, candidate.encoding, config)
                bound_source = getattr(library, "source_circuit", None)
                if bound_source is not None:
                    if getattr(library, "workload_hash", None) != workload_hash:
                        raise ValueError("Saved preparation workload hash mismatch")
                    source = bound_source.copy()
                else:
                    source = circuit.build(candidate.encoding, library)
                stage = "source_validation_failed"
                if source.num_qubits != 2 * circuit.num_qutrits:
                    raise ValueError("Preparation width does not match workload")
                expected = circuit.encoded_reference(candidate.encoding)
                before_metrics = state_metrics(source, expected, candidate.encoding, circuit.num_qutrits,
                                               max_qubits=config.max_validation_qubits)
                if before_metrics["fidelity"] < config.minimum_fidelity or before_metrics["leakage"] > config.maximum_leakage:
                    raise ValueError("Source preparation fails fidelity/leakage acceptance")
            except SynthesisArtifactError:
                raise
            except (ValueError, TypeError, RuntimeError, QiskitError) as error:
                candidate_error = f"{type(error).__name__}: {error}"
            saved_circuits = {}
            if library is not None:
                saved_circuits.update({"F3.qpy": library.f3, "CZ3.qpy": library.cz3})
            if source is not None:
                saved_circuits["source.qpy"] = source
            store.write_bundle(candidate_path, metadata={
                **candidate.to_dict(), "comparison_id": comparison_id, "gate_metrics": gate_metadata,
                "source_metrics": circuit_metrics(source) if source is not None else None,
                "error": candidate_error, "synthesis": selected_strategy.to_dict(),
                "synthesis_provenance": getattr(library, "provenance", {}) if library is not None else {},
                "source_state_metrics": before_metrics if candidate_error is None else None,
            }, arrays={"E.npy": candidate.encoding}, circuits=saved_circuits)
            bundles.append(candidate_path)
            for seed in config.transpiler_seeds:
                trial_path = f"trials/{candidate_index:05d}/{seed:010d}"
                row = {
                    "comparison_id": comparison_id, "target_hash": target_hash, "workload_hash": workload_hash,
                    "state_name": circuit.name, "class_name": candidate.family, "candidate_name": candidate.candidate_id,
                    "strategy_name": str(backend_spec.get("strategy", backend_spec.get("provider", "backend"))),
                    "seed_transpiler": seed, "success": False, "status": stage if candidate_error else "compilation_failed",
                    "error": candidate_error, "candidate_bundle": candidate_path, "trial_bundle": trial_path,
                    "graph_state_transpiled_qpy": None, "two_qubit_gate_count": None,
                    "one_qubit_gate_count": None, "size": None, "depth": None, "two_qubit_depth": None,
                    "fidelity": None, "leakage": None,
                }
                compiled = None
                if candidate_error is None:
                    try:
                        compiled = backend.compile(source.copy(), seed=seed)
                        backend.validate_native(compiled)
                        row.update(circuit_metrics(compiled))
                        row["status"] = "validation_failed"
                        row.update(state_metrics(compiled, expected, candidate.encoding, circuit.num_qutrits,
                                                 max_qubits=config.max_validation_qubits))
                        if row["fidelity"] < config.minimum_fidelity or row["leakage"] > config.maximum_leakage:
                            raise ValueError("Compiled preparation fails fidelity/leakage acceptance")
                        row.update(success=True, status="ok")
                    except (ValueError, TypeError, RuntimeError, QiskitError) as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                if compiled is not None:
                    row["graph_state_transpiled_qpy"] = f"{trial_path}/compiled.qpy"
                store.write_bundle(trial_path, metadata=row,
                                   circuits={"compiled.qpy": compiled} if compiled is not None else None)
                bundles.append(trial_path)
                rows.append(row)
            candidate_index += 1
    result = complete_run(store, rows, bundles)
    write_report(result)
    if progress is not None:
        progress(f"completed: {len(result.pareto_front)} Pareto alternatives; {store.root}")
    return result
