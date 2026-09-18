"""Sequential continuation, validated fallback and resumable theta benchmarks."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import CZGate, StatePreparation, UGate
from qiskit.quantum_info import Operator, Statevector, state_fidelity

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.benchmark import benchmark_direct_basis
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit, resolve_direct_state,
)
from .artifacts import RunStore, fingerprint, runtime_provenance
from .compilation import CompilationResult, compile_cz3_candidate, validate_cz3_candidate
from .cz3_continuation import ContinuationState, continue_cz3
from .cz3_template import load_cz3_template
from .encoding import theta_embedding, theta_leakage
from .f3 import synthesize_theta_f3
from .library import ThetaGateLibrary, validate_saved_f3
from .models import ThetaBenchmarkConfig


def _emit(progress, message):
    if progress is not None:
        progress(message)


def _state_dict(state, template):
    return {"theta": float(state.theta), "parameters": np.asarray(state.parameters).tolist(),
            "parent_id": state.parent_id, "template_id": template.template_id}


def _validated_state(payload, template, config):
    if payload["template_id"] != template.template_id:
        raise ValueError("Branch template identity mismatch.")
    if not isinstance(payload["parent_id"], str) or not payload["parent_id"]:
        raise ValueError("Branch parent identity is invalid.")
    circuit = template.bind(payload["parameters"])
    validate_cz3_candidate(circuit, theta_embedding(payload["theta"]), tolerance=config.cz3_tolerance)
    return ContinuationState(float(payload["theta"]), np.asarray(payload["parameters"], dtype=float), payload["parent_id"])


def _same_state(first, second):
    return (first.theta == second.theta and first.parent_id == second.parent_id
            and np.array_equal(first.parameters, second.parameters))


def _compile(circuit, encoding, config):
    return compile_cz3_candidate(circuit, encoding, tolerance=config.cz3_tolerance,
                                 seeds=config.transpiler_seeds, optimization_level=config.optimization_level)


def _manifest(config, template):
    payload = {"format_version": 1, "benchmark": "theta_continuation_v1", "config": config.to_dict(),
               "baseline_sha256": template.baseline_sha256, "template_id": template.template_id,
               "provenance": runtime_provenance()}
    return {**payload, "fingerprint": fingerprint(payload)}


def _baseline(store, template, config, *, readonly):
    if (store.root / "baseline/complete.json").exists():
        bundle = store.read_bundle("baseline")
        if bundle["documents"]["template.json"] != template.to_dict():
            raise ValueError("Persisted baseline template mismatch.")
        original = bundle["circuits"]["cz3_original.qpy"]
        if not np.allclose(Operator(original).data, Operator(template.original_circuit).data, atol=1e-13, rtol=0):
            raise ValueError("Persisted baseline circuit mismatch.")
        compiled = bundle["circuits"]["cz3_compiled.qpy"]
        metrics = validate_cz3_candidate(compiled, theta_embedding(0), tolerance=config.cz3_tolerance,
                                         source=original, compiled=True)
        saved = bundle["metadata"]["compiled_metrics"]
        _check_metrics(metrics, saved, "baseline")
        return CompilationResult(compiled, saved, bundle["documents"]["compilation.json"]["trials"])
    if readonly:
        raise ValueError("Circuits mode requires a complete baseline bundle.")
    result = _compile(template.original_circuit, theta_embedding(0), config)
    store.write_bundle("baseline", metadata={"template_id": template.template_id, "compiled_metrics": result.metrics},
                       circuits={"cz3_original.qpy": template.original_circuit, "cz3_compiled.qpy": result.circuit},
                       arrays={"initial_parameters.npy": template.initial_parameters},
                       documents={"template.json": template.to_dict(), "compilation.json": {"trials": result.trials}})
    return result


def _check_metrics(actual, saved, description):
    for key, value in actual.items():
        if key not in saved or not np.isclose(value, saved[key], atol=1e-12, rtol=1e-10):
            raise ValueError(f"Persisted {description} metric mismatch: {key}.")


def _replay_attempts(attempts, prior, template, config, index, theta):
    current = prior
    target_id = f"theta_{index:05d}"
    seen = set()
    for attempt in attempts:
        if (attempt["id"] in seen or not attempt["id"].startswith(target_id + "__")
                or attempt["warm_start_parent_id"] != current.parent_id
                or not np.array_equal(attempt["p_start"], current.parameters)
                or not current.theta <= attempt["theta"] <= theta
                or attempt["requested_grid_point"] != (attempt["theta"] == theta)):
            raise ValueError("Continuation attempt parent or theta history mismatch.")
        seen.add(attempt["id"])
        if attempt["fit"]["valid"]:
            current = _validated_state({"theta": attempt["theta"], "parameters": attempt["fit"]["parameters"],
                                        "parent_id": target_id if attempt["requested_grid_point"] else attempt["id"],
                                        "template_id": template.template_id}, template, config)
    return current


def _replay_branch(bundle, prior, template, config, index, theta):
    metadata = bundle["metadata"]
    if metadata["warm_start_parent_id"] != prior.parent_id:
        raise ValueError("Warm-start parent history mismatch.")
    attempt_document = bundle["documents"]["attempts.json"]
    if index == 0:
        current = ContinuationState(0.0, template.initial_parameters.copy(), "theta_00000")
    elif attempt_document.get("branch_rejected", False):
        current = prior
    else:
        current = _replay_attempts(attempt_document["attempts"], prior, template, config, index, theta)
    saved = bundle["documents"]["branch_state.json"]
    state = _validated_state(saved, template, config)
    if not _same_state(current, state) or saved != metadata["branch_state"]:
        raise ValueError("Persisted branch state does not match continuation history.")
    if not np.array_equal(bundle["arrays"]["branch_parameters.npy"], state.parameters):
        raise ValueError("Persisted branch parameter array mismatch.")
    if "cz3_branch.qpy" not in bundle["circuits"] or not np.allclose(
        Operator(bundle["circuits"]["cz3_branch.qpy"]).data,
        Operator(template.bind(state.parameters)).data, atol=1e-13, rtol=0,
    ):
        raise ValueError("Persisted branch circuit mismatch.")
    return state


def _validate_point(bundle, prior, template, config, baseline, index, theta):
    metadata = bundle["metadata"]
    if (metadata["index"] != index or metadata["theta"] != theta
            or metadata["template_id"] != template.template_id
            or metadata["baseline_n_cz"] != baseline.metrics["N_CZ"]):
        raise ValueError("Persisted gate point identity mismatch.")
    encoding = theta_embedding(theta)
    if not np.array_equal(bundle["arrays"]["E.npy"], encoding) or not np.array_equal(bundle["arrays"]["leakage.npy"], theta_leakage(theta)):
        raise ValueError("Persisted theta encoding mismatch.")
    state = _replay_branch(bundle, prior, template, config, index, theta)
    if metadata["continuation_valid"]:
        parameters = bundle["arrays"]["cz3_parameters.npy"]
        circuit = template.bind(parameters)
        validate_cz3_candidate(circuit, encoding, tolerance=config.cz3_tolerance)
        if state.theta != theta or not np.array_equal(state.parameters, parameters):
            raise ValueError("Continuation target and branch state mismatch.")
        validate_cz3_candidate(bundle["circuits"]["cz3_continuation.qpy"], encoding,
                                tolerance=config.cz3_tolerance, source=circuit)
    if "f3_optimal.qpy" in bundle["circuits"]:
        metrics = validate_saved_f3(bundle["circuits"]["f3_optimal.qpy"], encoding, metadata["f3_alpha"], tolerance=config.f3_tolerance)
        _check_metrics(metrics, metadata["f3_metrics"], "F3")
        metrics = validate_saved_f3(bundle["circuits"]["f3_alpha_zero.qpy"], encoding, 0.0, tolerance=config.f3_tolerance)
        _check_metrics(metrics, metadata["f3_alpha_zero_metrics"], "F3 alpha-zero")
    if metadata["selected_method"] is not None:
        selected = bundle["circuits"]["cz3_selected.qpy"]
        source_name = "cz3_fallback.qpy" if metadata["selected_method"] == "bqskit" else "cz3_continuation.qpy"
        metrics = validate_cz3_candidate(selected, encoding, tolerance=config.cz3_tolerance,
                                         source=bundle["circuits"][source_name], compiled=True)
        _check_metrics(metrics, metadata["cz3_metrics"], "CZ3")
    for prefix, saved_metrics, mapping in (
        ("cz3", metadata["cz3_metrics"], {"n_cz": "N_CZ", "depth": "depth", "n_1q": "N_1q", "E_norm": "E_norm", "L_norm": "L_norm"}),
        ("f3", metadata["f3_metrics"], {"n_cz": "N_CZ", "E_norm": "E_norm", "L_norm": "L_norm"}),
        ("f3_alpha_zero", metadata["f3_alpha_zero_metrics"], {"n_cz": "N_CZ"}),
    ):
        for suffix, metric_name in mapping.items():
            if metadata[f"{prefix}_{suffix}"] != saved_metrics.get(metric_name):
                raise ValueError(f"Persisted flattened metric mismatch: {prefix}_{suffix}.")
    expected_correct = "f3_optimal.qpy" in bundle["circuits"] and metadata["selected_method"] is not None
    within_budget = None if metadata["selected_method"] is None else metadata["cz3_n_cz"] <= baseline.metrics["N_CZ"]
    expected_status = "failed" if not expected_correct else "ok" if within_budget else "above_baseline"
    if (metadata["correct"] != expected_correct or metadata["within_baseline_budget"] != within_budget
            or metadata["status"] != expected_status):
        raise ValueError("Persisted correctness or budget status mismatch.")
    if metadata["correct"]:
        ThetaGateLibrary.from_bundle(bundle, encoding, f3_tolerance=config.f3_tolerance, cz3_tolerance=config.cz3_tolerance)
    return state


def _run_point(index, theta, prior, template, baseline, config, progress):
    encoding = theta_embedding(theta)
    target_id = f"theta_{index:05d}"
    errors, circuits, attempts = [], {}, []
    arrays = {"E.npy": encoding, "leakage.npy": theta_leakage(theta)}
    documents = {"attempts": attempts, "continuation_compilation": [], "fallback_compilation": []}
    f3_result, selected, selected_method = None, None, None
    continuation_valid, fallback_used = False, False
    continuation_seconds, fallback_seconds = 0.0, 0.0
    fallback_reason = None
    current = prior
    _emit(progress, f"point {index + 1}/{len(config.grid())}: theta={theta:.8g}; F3")
    try:
        f3_result = synthesize_theta_f3(encoding, tolerance=config.f3_tolerance,
                                        transpiler_seeds=config.transpiler_seeds, optimization_level=config.optimization_level)
        # The synthesis helper validates already; recomputation also protects injected results.
        optimal_metrics = validate_saved_f3(f3_result.optimal, encoding, f3_result.alpha, tolerance=config.f3_tolerance)
        zero_metrics = validate_saved_f3(f3_result.alpha_zero, encoding, 0.0, tolerance=config.f3_tolerance)
        if optimal_metrics["N_CZ"] > 2:
            raise ValueError("F3 optimal circuit exceeds the two-CZ limit.")
        f3_result = replace(f3_result, optimal_metrics={**f3_result.optimal_metrics, **optimal_metrics},
                            alpha_zero_metrics={**f3_result.alpha_zero_metrics, **zero_metrics})
        circuits.update({"f3_optimal.qpy": f3_result.optimal, "f3_alpha_zero.qpy": f3_result.alpha_zero})
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        f3_result = None
        errors.append(f"F3: {type(exc).__name__}: {exc}")
    _emit(progress, f"point {index + 1}/{len(config.grid())}: CZ3 continuation")
    if index == 0:
        current = ContinuationState(0.0, template.initial_parameters.copy(), target_id)
        continuation_valid = True
        circuits["cz3_continuation.qpy"] = template.original_circuit
        arrays["cz3_parameters.npy"] = template.initial_parameters.copy()
        selected, selected_method = baseline, "baseline"
    else:
        started = perf_counter()
        try:
            result = continue_cz3(template, prior, theta, target_id=target_id, max_nfev=config.max_nfev,
                                  tolerance=config.cz3_tolerance, max_subdivisions=config.max_subdivisions)
            attempts.extend(result.attempts)
            candidate_state = _validated_state(_state_dict(result.state, template), template, config)
            replayed = _replay_attempts(attempts, prior, template, config, index, theta)
            if not prior.theta <= candidate_state.theta <= theta or not _same_state(candidate_state, replayed):
                raise ValueError("Continuation branch parent or theta history mismatch.")
            current = candidate_state
            fit = result.target_fit
            if fit is not None:
                circuits["cz3_continuation.qpy"] = fit.circuit
                if fit.valid:
                    rebound = template.bind(fit.parameters)
                    validate_cz3_candidate(fit.circuit, encoding, tolerance=config.cz3_tolerance, source=rebound)
                    validate_cz3_candidate(rebound, encoding, tolerance=config.cz3_tolerance)
                    if current.theta != theta or not np.array_equal(current.parameters, fit.parameters):
                        raise ValueError("Continuation fit and branch state mismatch.")
                    continuation_valid = True
                    arrays["cz3_parameters.npy"] = np.asarray(fit.parameters).copy()
                    selected = _compile(fit.circuit, encoding, config)
                    selected_method = "continuation"
                    documents["continuation_compilation"] = selected.trials
            if not continuation_valid:
                fallback_reason = "continuation_invalid"
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            errors.append(f"Continuation: {type(exc).__name__}: {exc}")
            if current is prior and attempts:
                documents["branch_rejected"] = True
            documents["continuation_compilation"] = getattr(exc, "metrics", {}).get("trials", [])
            fallback_reason = "continuation_compile_failed" if continuation_valid else "continuation_invalid"
        continuation_seconds = perf_counter() - started
    if selected is not None and selected.metrics["N_CZ"] > baseline.metrics["N_CZ"]:
        fallback_reason = "above_baseline_budget"
    if selected is None or fallback_reason is not None:
        fallback_used = True
        _emit(progress, f"point {index + 1}/{len(config.grid())}: BQSKit fallback ({fallback_reason})")
        started = perf_counter()
        try:
            fallback = optimized_gates.synthesize_cz3(encoding, tolerance=config.cz3_tolerance)
            circuits["cz3_fallback.qpy"] = fallback
            candidate = _compile(fallback, encoding, config)
            documents["fallback_compilation"] = candidate.trials
            rank = lambda result: tuple(result.metrics[key] for key in ("N_CZ", "depth", "N_1q"))
            if selected is None or rank(candidate) < rank(selected):
                selected, selected_method = candidate, "bqskit"
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            errors.append(f"Fallback: {type(exc).__name__}: {exc}")
            documents["fallback_compilation"] = getattr(exc, "metrics", {}).get("trials", [])
        fallback_seconds = perf_counter() - started
    # The branch comes exclusively from the original parameterized topology.
    current = _validated_state(_state_dict(current, template), template, config)
    branch = _state_dict(current, template)
    arrays["branch_parameters.npy"] = current.parameters.copy()
    circuits["cz3_branch.qpy"] = template.bind(current.parameters)
    if selected is not None:
        circuits["cz3_selected.qpy"] = selected.circuit
    cz_metrics = {} if selected is None else selected.metrics
    f3_metrics = {} if f3_result is None else f3_result.optimal_metrics
    f3_zero_metrics = {} if f3_result is None else f3_result.alpha_zero_metrics
    correct = selected is not None and f3_result is not None
    within_budget = None if selected is None else selected.metrics["N_CZ"] <= baseline.metrics["N_CZ"]
    metadata = {
        "index": index, "theta": theta, "theta_over_pi": theta / np.pi, "template_id": template.template_id,
        "warm_start_parent_id": prior.parent_id, "status": "failed" if not correct else "ok" if within_budget else "above_baseline",
        "correct": correct, "continuation_valid": continuation_valid, "fallback_used": fallback_used,
        "fallback_reason": fallback_reason, "selected_method": selected_method, "within_baseline_budget": within_budget,
        "baseline_n_cz": baseline.metrics["N_CZ"],
        "cz3_n_cz": cz_metrics.get("N_CZ"), "cz3_depth": cz_metrics.get("depth"), "cz3_n_1q": cz_metrics.get("N_1q"),
        "cz3_E_norm": cz_metrics.get("E_norm"), "cz3_L_norm": cz_metrics.get("L_norm"),
        "f3_alpha": None if f3_result is None else f3_result.alpha, "f3_n_cz": f3_metrics.get("N_CZ"),
        "f3_alpha_zero_n_cz": f3_zero_metrics.get("N_CZ"), "f3_E_norm": f3_metrics.get("E_norm"), "f3_L_norm": f3_metrics.get("L_norm"),
        "continuation_seconds": continuation_seconds, "fallback_seconds": fallback_seconds,
        "f3_synthesis_seconds": 0.0 if f3_result is None else f3_result.synthesis_seconds,
        "errors": errors, "branch_state": branch, "cz3_metrics": cz_metrics, "f3_metrics": f3_metrics,
        "f3_alpha_zero_metrics": f3_zero_metrics,
    }
    return metadata, circuits, arrays, {"attempts.json": documents, "branch_state.json": branch}, current


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _zero_preparation_source(encoding):
    source = QuantumCircuit(2, name="zero_preparation")
    source.append(StatePreparation(encoding[:, 0], label="zero_W"), [0, 1])
    return source


def _preparation_metrics(circuit, encoding, config):
    """Validate the prepared state and the exact StatePreparation extension."""
    if (not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != 2
            or circuit.num_clbits or circuit.num_parameters
            or any(item.operation.base_class not in {UGate, CZGate} or item.clbits for item in circuit.data)):
        raise ValueError("Zero preparation requires a bound two-qubit U/CZ circuit.")
    actual = Operator(circuit).data
    expected = encoding[:, 0]
    state = actual[:, 0]
    state_phase = np.exp(1j * np.angle(np.vdot(state, expected)))
    target = Operator(_zero_preparation_source(encoding)).data
    full_phase = np.exp(1j * np.angle(np.vdot(actual, target)))
    metrics = {
        "N_CZ": int(circuit.count_ops().get("cz", 0)),
        "N_1q": sum(item.operation.num_qubits == 1 for item in circuit.data),
        "depth": int(circuit.depth()),
        "state_error_norm": float(np.linalg.norm(state_phase * state - expected)),
        "full_error_norm": float(np.linalg.norm(full_phase * actual - target, ord="fro")),
    }
    if (not np.isfinite(actual).all() or not np.isfinite(list(metrics.values())).all()
            or metrics["state_error_norm"] > config.f3_tolerance or metrics["full_error_norm"] > 1e-10):
        raise ValueError("Zero preparation failed prepared-state or exact-operator validation.")
    return metrics


def _compile_zero_preparation(encoding, config):
    """Compile one encoded-zero block for accounting before cross-block fusion."""
    source = _zero_preparation_source(encoding)
    candidates, trials = [], []
    for seed in config.transpiler_seeds:
        try:
            circuit = transpile(source, basis_gates=["u", "cz"], coupling_map=None,
                                routing_method="none", approximation_degree=1.0,
                                optimization_level=config.optimization_level, seed_transpiler=seed)
            metrics = {**_preparation_metrics(circuit, encoding, config), "seed": seed}
            circuit.metadata = metrics
            trials.append({"seed": seed, "valid": True, "metrics": metrics})
            rank = tuple(metrics[key] for key in ("N_CZ", "depth", "N_1q", "seed"))
            candidates.append((rank, circuit, metrics))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            trials.append({"seed": seed, "valid": False, "error": str(exc)})
    if not candidates:
        raise ValueError(f"No exact zero-preparation candidate passed validation: {trials}")
    _, circuit, metrics = min(candidates, key=lambda candidate: candidate[0])
    return CompilationResult(circuit, metrics, trials)


def _component_costs(state_name, preparation, gates):
    """Count actual block copies; duplicate non-self graph edges remain distinct."""
    state = resolve_direct_state(state_name)
    n_qutrits = state.num_qutrits
    n_edges = sum(left != right for left, right in state.edges)
    costs = {"component_num_qutrits": n_qutrits, "component_num_cz3_blocks": n_edges}
    for prefix, circuit, multiplicity in (
        ("preparation", preparation, n_qutrits),
        ("f3_blocks", gates.f3, n_qutrits),
        ("cz3_blocks", gates.cz3, n_edges),
    ):
        costs[f"{prefix}_n_cz"] = multiplicity * int(circuit.count_ops().get("cz", 0))
        costs[f"{prefix}_n_1q"] = multiplicity * sum(item.operation.num_qubits == 1 for item in circuit.data)
    for suffix in ("n_cz", "n_1q"):
        costs[f"unfused_{suffix}"] = sum(costs[f"{prefix}_{suffix}"] for prefix in ("preparation", "f3_blocks", "cz3_blocks"))
    return costs


def _validate_full_bundle(bundle, gate_bundle, config, state_name):
    row = bundle["metadata"]
    point = gate_bundle["metadata"]
    if (row != bundle["documents"]["row.json"] or row["index"] != point["index"]
            or row["theta"] != point["theta"] or row["state_name"] != state_name
            or row["gate_point_fingerprint"] != fingerprint(gate_bundle["complete"])):
        raise ValueError("Full-circuit artifact identity mismatch.")
    encoding = theta_embedding(point["theta"])
    preparation = bundle["circuits"]["zero_preparation.qpy"]
    preparation_metrics = _preparation_metrics(preparation, encoding, config)
    _check_metrics(preparation_metrics, bundle["documents"]["zero_preparation_compilation.json"]["metrics"], "zero preparation")
    gates = ThetaGateLibrary.from_bundle(gate_bundle, encoding, f3_tolerance=config.f3_tolerance, cz3_tolerance=config.cz3_tolerance)
    for key, expected in _component_costs(state_name, preparation, gates).items():
        if type(row.get(key)) is not int or row[key] != expected:
            raise ValueError(f"Saved component metric mismatch: {key}.")
    if row["success"]:
        circuit = bundle["circuits"]["graph_state_transpiled.qpy"]
        reference = build_direct_basis_graph_state_circuit(state_name, theta_embedding(point["theta"]))
        if (circuit.num_qubits != reference.num_qubits or circuit.num_clbits or circuit.num_parameters
                or not set(circuit.count_ops()) <= {"u", "cz"}):
            raise ValueError("Invalid saved full circuit.")
        fidelity = float(state_fidelity(Statevector.from_instruction(reference), Statevector.from_instruction(circuit)))
        if not np.isfinite(fidelity) or fidelity < 1 - max(1e-8, 100 * config.cz3_tolerance**2):
            raise ValueError("Saved full circuit failed independent reference fidelity.")
        for key, value in (("best_depth", circuit.depth()), ("best_two_qubit_gate_count", circuit.count_ops().get("cz", 0)),
                           ("best_one_qubit_gate_count", sum(item.operation.num_qubits == 1 for item in circuit.data))):
            if row[key] != value:
                raise ValueError(f"Saved full-circuit metric mismatch: {key}.")
    return row


def _full_circuits(store, points, config, progress):
    rows = []
    preparations = {}
    for state_name in config.states:
        for point in points:
            if not point["correct"]:
                continue
            index = point["index"]
            bundle = store.read_bundle(f"points/{index:05d}")
            relative = f"circuits/{state_name}/{index:05d}"
            if (store.root / relative / "complete.json").exists():
                saved = store.read_bundle(relative)
                rows.append(_validate_full_bundle(saved, bundle, config, state_name))
                document = saved["documents"]["zero_preparation_compilation.json"]
                preparations.setdefault(index, CompilationResult(saved["circuits"]["zero_preparation.qpy"], document["metrics"], document["trials"]))
                continue
            _emit(progress, f"full circuit: {state_name}; point {index + 1}/{len(points)}")
            encoding = theta_embedding(point["theta"])
            gates = ThetaGateLibrary.from_bundle(bundle, encoding, f3_tolerance=config.f3_tolerance, cz3_tolerance=config.cz3_tolerance)
            if index not in preparations:
                preparations[index] = _compile_zero_preparation(encoding, config)
            preparation = preparations[index]
            component_costs = _component_costs(state_name, preparation.circuit, gates)
            row = benchmark_direct_basis(
                state_name=state_name, basis_matrix=encoding, basis_candidate_name=f"theta_{index:05d}",
                basis_candidate_type="theta_continuation", gate_library=gates, cz3_tolerance=config.cz3_tolerance,
                coupling_map=None, basis_gates=["u", "cz"],
                transpiler_seeds=config.transpiler_seeds, rank_by_cz=True, compute_fidelity=True,
                optimization_level=config.optimization_level, routing_method="none", approximation_degree=1.0,
                quantum_circuits_dir=str(store.root / "circuits"),
            )
            row = _json_safe({**row, **component_costs, "index": index, "theta": point["theta"], "theta_over_pi": point["theta_over_pi"],
                              "state_name": state_name, "gate_point_fingerprint": fingerprint(bundle["complete"])})
            circuits = {"zero_preparation.qpy": preparation.circuit}
            for key, filename in (("graph_state_qpy", "graph_state.qpy"), ("graph_state_transpiled_qpy", "graph_state_transpiled.qpy")):
                if row.get(key) and Path(row[key]).is_file():
                    with Path(row[key]).open("rb") as stream:
                        loaded = qpy.load(stream)
                    if len(loaded) != 1:
                        raise ValueError("Expected one exported full circuit per QPY.")
                    circuits[filename] = loaded[0]
            documents = {"row.json": row, "zero_preparation_compilation.json": {"metrics": preparation.metrics, "trials": preparation.trials}}
            staged = {"metadata": row, "documents": documents, "circuits": circuits}
            _validate_full_bundle(staged, bundle, config, state_name)
            store.write_bundle(relative, metadata=row, circuits=circuits, documents=documents)
            rows.append(row)
    return rows


def run_theta_benchmark(config: ThetaBenchmarkConfig, output_dir: str | Path, *, mode="all", resume=False, progress=None) -> dict:
    """Run the unchanged theta-grid prefix, or consume its complete gate artifacts."""
    if not isinstance(config, ThetaBenchmarkConfig):
        raise TypeError("config must be a ThetaBenchmarkConfig")
    if mode not in {"gates", "circuits", "all"}:
        raise ValueError("mode must be gates, circuits, or all")
    _emit(progress, "validating pinned CZ3 baseline")
    template = load_cz3_template(config.baseline_qpy, expected_sha256=config.baseline_sha256, tolerance=config.cz3_tolerance)
    manifest = _manifest(config, template)
    store = (RunStore.open(output_dir, expected_fingerprint=manifest["fingerprint"])
             if resume or mode == "circuits" else RunStore.create(output_dir, manifest))
    if store.manifest != manifest:
        raise ValueError("Run manifest content does not match its fingerprint.")
    baseline = _baseline(store, template, config, readonly=mode == "circuits")
    indices = store.completed_point_indices()
    grid = config.grid()
    if indices != list(range(len(indices))) or len(indices) > len(grid):
        raise ValueError("Completed gate points must be a contiguous grid prefix.")
    if mode == "circuits" and len(indices) != len(grid):
        raise ValueError("Circuits mode requires the complete configured gate grid.")
    current = ContinuationState(0.0, template.initial_parameters.copy(), "baseline")
    points = []
    for index in indices:
        bundle = store.read_bundle(f"points/{index:05d}")
        current = _validate_point(bundle, current, template, config, baseline, index, grid[index])
        points.append(bundle["metadata"])
    checkpoint = store.load_checkpoint()
    if checkpoint is not None:
        checkpoint_index = checkpoint["last_completed_index"]
        if (not isinstance(checkpoint_index, int) or checkpoint_index not in indices
                or checkpoint["branch_state"] != points[checkpoint_index]["branch_state"]):
            raise ValueError("Checkpoint does not match completed gate history.")
    for index in range(len(indices), len(grid)):
        metadata, circuits, arrays, documents, current = _run_point(index, grid[index], current, template, baseline, config, progress)
        store.write_bundle(f"points/{index:05d}", metadata=metadata, circuits=circuits, arrays=arrays, documents=documents)
        store.save_checkpoint({"last_completed_index": index, "branch_state": metadata["branch_state"]})
        points.append(metadata)
        message = (f"point {index + 1}/{len(grid)}: {metadata['status']}; CZ={metadata['cz3_n_cz']}; "
                   f"method={metadata['selected_method']}; E_norm={metadata['cz3_E_norm']}; L_norm={metadata['cz3_L_norm']}")
        if not metadata["correct"] and metadata["errors"]:
            message += "; errors=" + "; ".join(error.splitlines()[0][:240] for error in metadata["errors"])
        _emit(progress, message)
    full_rows = _full_circuits(store, points, config, progress) if mode in {"all", "circuits"} else []
    counts = {"total": len(points), "correct": sum(point["correct"] for point in points),
              "failed": sum(not point["correct"] for point in points),
              "fallback": sum(point["fallback_used"] for point in points),
              "above_baseline": sum(point["within_baseline_budget"] is False for point in points)}
    return {"output_dir": str(store.root), "points": points, "full_circuits": full_rows,
            "counts": counts, **counts}
