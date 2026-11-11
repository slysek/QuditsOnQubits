"""Run bounded CZ3 fits from the saved baseline."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shlex
import sys
import tempfile
import time

import numpy as np
import qiskit
from qiskit import qpy
import scipy

from .evaluator import evaluate
from .encodings import tensor_encoding
from .models import EvaluationTask
from .optimizer import ANGLE_BOUND_RADIUS, fit_fixed_cz3, parameterize_cz3_circuit
from .targets import controlled_z_d

PROTOCOL_TOLERANCE = 1e-5
STRICT_TOLERANCE = 1e-10
SOLVER_CONFIG = {"method": "scipy_least_squares_trf",
                 "angle_bound_radius": float(ANGLE_BOUND_RADIUS),
                 "phase_bound_radius": float(np.pi)}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read_baseline(directory: Path, encoding_paths=None):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    summary_raw = (directory / "baseline_summary.json").read_bytes()
    if _hash(summary_raw) != manifest["files"].get("baseline_summary.json"):
        raise ValueError("The baseline summary hash does not match")
    summary = json.loads(summary_raw)
    cases = []
    for row in summary["results"]:
        if row["gate"] != "CZ3" or not row["is_valid"] or row["method"] not in ("saved_reference", "candidate"):
            continue
        names = [row["circuit_artifact"], *json.loads(row["encoding_artifacts"])]
        if len(names) != 3:
            raise ValueError("Each case must have one circuit and two encodings")
        values = []
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("The artifact name is not valid")
            raw = (directory / name).read_bytes()
            if _hash(raw) != manifest["files"].get(name):
                raise ValueError(f"The hash does not match for {name}")
            values.append(raw)
        circuit, = qpy.load(BytesIO(values[0]))
        encodings = [np.load(BytesIO(value), allow_pickle=False) for value in values[1:]]
        template = parameterize_cz3_circuit(circuit)
        task = EvaluationTask(3, 2, controlled_z_d(3), PROTOCOL_TOLERANCE / 3,
                              PROTOCOL_TOLERANCE**2 / 9, ("u", "u3", "cz"))
        if not evaluate(task, circuit, encodings).is_valid:
            raise ValueError("The source circuit does not pass the baseline tolerance")
        name = row["encoding"]
        if encoding_paths is not None:
            values[1:] = [Path(path).read_bytes() for path in encoding_paths]
            encodings = [np.load(BytesIO(value), allow_pickle=False) for value in values[1:]]
            tensor_encoding(encodings, d=3)
            name += " (new encodings)"
        rebound = template.bind(template.initial_parameters)
        from qiskit.quantum_info import Operator
        if not np.allclose(Operator(circuit).data, Operator(rebound).data, rtol=0, atol=1e-12):
            raise ValueError("The parameterized circuit does not match the source")
        cases.append({"name": name, "method": row["method"], "source_names": names,
                      "raw": values, "hashes": [_hash(value) for value in values],
                      "angles": template.initial_parameters.tolist()})
    if not cases:
        raise ValueError("The baseline does not contain a valid CZ3 circuit")
    return cases


def _worker(connection, circuit_bytes, encoding_bytes, parameters, max_nfev):
    try:
        circuit, = qpy.load(BytesIO(circuit_bytes))
        encodings = [np.load(BytesIO(raw), allow_pickle=False) for raw in encoding_bytes]
        result = fit_fixed_cz3(circuit, encodings, np.array(parameters), max_nfev=max_nfev,
                               tolerance=STRICT_TOLERANCE)
        data = result.to_dict()
        if data["solver_config"] != SOLVER_CONFIG:
            raise ValueError("The solver settings do not match the run settings")
        connection.send({"status": "completed", "solver": data["solver"],
                         "fit_evaluation": data["evaluation"],
                         "parameters": result.parameters.tolist()})
    except Exception as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _attempt(case, parameters, max_nfev, timeout):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, case["raw"][0], case["raw"][1:],
                                                       parameters, max_nfev))
    start = time.monotonic()
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            try:
                result = receiver.recv()
            except EOFError:
                result = {"status": "error", "error": "The process ended without a result"}
        else:
            result = {"status": "timeout", "error": "The process exceeded the time limit"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
    result["wall_seconds"] = time.monotonic() - start
    return result


def _code_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [*sorted((root / "src/qudits_on_qubits/evolution").glob("*.py")),
             *sorted((root / "src/qudits_on_qubits/benchmarks/theta_continuation").glob("*.py")),
             root / "src/qudits_on_qubits/benchmarks/validation.py",
             root / "src/qudits_on_qubits/benchmarks/direct_basis/optimized_gates.py",
             root / "src/qudits_on_qubits/benchmarks/direct_basis/math_utils.py"]
    return {str(path.relative_to(root)): _hash(path.read_bytes()) for path in paths}


def _config(cases, seeds, max_nfev, timeout):
    return {"format": 1, "target": "CZ3", "seeds": list(seeds), "max_nfev": max_nfev,
            "timeout_seconds": timeout, "tolerance_frobenius": PROTOCOL_TOLERANCE,
            "strict_tolerance_frobenius": STRICT_TOLERANCE,
            "solver_config": SOLVER_CONFIG,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "qiskit": qiskit.__version__},
            "code_sha256": _code_hashes(),
            "cases": [{"name": case["name"], "method": case["method"], "hashes": case["hashes"]}
                      for case in cases]}


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _completed(attempt_dir, output):
    result_path = attempt_dir / "result.json"
    commit_path = attempt_dir / "commit.json"
    if not commit_path.exists():
        return None
    result_bytes = result_path.read_bytes()
    commit = json.loads(commit_path.read_bytes())
    if _hash(result_bytes) != commit.get("result_sha256"):
        raise ValueError("A cached result does not match its hash")
    result = json.loads(result_bytes)
    if result.get("trial_path") != str(attempt_dir.relative_to(output)):
        raise ValueError("The cached result path does not match")
    if result.get("metrics", {}).get("is_valid") and set(result.get("artifact_hashes", {})) != {"circuit.qpy", "E_A.npy", "E_B.npy"}:
        raise ValueError("The cached result has missing artifacts")
    for name, expected in result.get("artifact_hashes", {}).items():
        if Path(name).name != name or _hash((attempt_dir / name).read_bytes()) != expected:
            raise ValueError("A cached artifact does not match its hash")
    return result if result.get("status") == "completed" else None


def _save_result(directory, result, case):
    result["artifact_hashes"] = {}
    if result["status"] == "completed":
        circuit, = qpy.load(BytesIO(case["raw"][0]))
        template = parameterize_cz3_circuit(circuit)
        circuit = template.bind(result["parameters"])
        encodings = [np.load(BytesIO(data), allow_pickle=False) for data in case["raw"][1:]]
        task = EvaluationTask(3, 2, controlled_z_d(3), PROTOCOL_TOLERANCE / 3,
                              PROTOCOL_TOLERANCE**2 / 9, ("u", "u3", "cz"))
        metrics = evaluate(task, circuit, encodings)
        result["metrics"] = asdict(metrics)
        leakage_norm = float(np.sqrt(9 * metrics.L))
        result["strict_valid"] = bool(metrics.error_norm <= STRICT_TOLERANCE and leakage_norm <= STRICT_TOLERANCE)
        result["leakage_frobenius"] = leakage_norm
        if metrics.is_valid:
            with (directory / "circuit.qpy").open("xb") as stream:
                qpy.dump(circuit, stream)
            (directory / "E_A.npy").write_bytes(case["raw"][1])
            (directory / "E_B.npy").write_bytes(case["raw"][2])
            for name in ("circuit.qpy", "E_A.npy", "E_B.npy"):
                result["artifact_hashes"][name] = _hash((directory / name).read_bytes())
    _atomic_json(directory / "result.json", result)
    _atomic_json(directory / "commit.json", {"result_sha256": _hash((directory / "result.json").read_bytes())})


def _setup(output, cases, config, config_hash):
    manifest = output / "manifest.json"
    pending = output / "pending.json"
    specification = {"configuration": config, "configuration_sha256": config_hash}
    if manifest.exists() or pending.exists():
        existing = json.loads((manifest if manifest.exists() else pending).read_bytes())
        if existing != specification:
            raise ValueError("The run configuration does not match")
    else:
        if any(item.name not in (".lock", "pending.json.tmp") for item in output.iterdir()):
            raise ValueError("The run has no configuration")
        _atomic_json(pending, specification)
    for index, case in enumerate(cases):
        folder = output / "inputs" / f"case_{index + 1:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, raw, expected in zip(("circuit.qpy", "E_A.npy", "E_B.npy"), case["raw"], case["hashes"], strict=True):
            path = folder / name
            if path.exists():
                if _hash(path.read_bytes()) != expected:
                    raise ValueError("A saved input does not match its hash")
            else:
                with tempfile.NamedTemporaryFile(dir=folder, prefix=".input-", delete=False) as stream:
                    stream.write(raw)
                    temporary = Path(stream.name)
                os.replace(temporary, path)
    if not manifest.exists():
        _atomic_json(manifest, specification)


def _best(records, *, strict=False):
    groups = {}
    for record in records:
        metrics = record.get("metrics", {})
        if record.get("status") != "completed" or not metrics.get("is_valid"):
            continue
        if strict and not record.get("strict_valid"):
            continue
        key = tuple(record["source_sha256"][1:])
        rank = (metrics["cz_count"] + metrics["cx_count"], metrics["two_qubit_depth"],
                metrics["depth"], metrics["one_qubit_gate_count"], metrics["error_norm"])
        if key not in groups or rank < groups[key][0]:
            groups[key] = (rank, {"encoding": record["encoding"], "case": record["case"],
                                  "source_encoding_sha256": list(key), "gate_set": ["u", "u3", "cz"],
                                  "circuit_artifact": record["trial_path"] + "/circuit.qpy",
                                  "metrics": metrics})
    return [value[1] for value in groups.values()]


def _write_readme(output, config):
    path = output / "README.md"
    if path.exists():
        return
    command = [sys.executable, "-m", "qudits_on_qubits.evolution.classical",
               "--baseline", config["baseline_path"], "--output", str(output.resolve()),
               "--max-nfev", str(config["max_nfev"]), "--timeout", str(config["timeout_seconds"])]
    for seed in config["seeds"]:
        command += ["--seed", str(seed)]
    if not config["seeds"]:
        command += ["--no-random-starts"]
    if config["encoding_paths"]:
        command += ["--encoding-a", config["encoding_paths"][0], "--encoding-b", config["encoding_paths"][1]]
    text = (
        "# Classical CZ3 run\n\n"
        "Run the command from the repository root. Use the same Python environment.\n"
        "Use a new output path to repeat the run.\n\n"
        "```sh\nPYTHONPATH=src " + shlex.join(command) + "\n```\n\n"
        "To continue this run, use the same command with `--resume`.\n"
        "The command needs the original baseline directory and any encoding paths.\n"
        "The `inputs` directory contains copies of the circuit and encoding files.\n\n"
        "The solver uses SciPy least_squares with method trf and linear loss.\n"
        "Each U3 angle is limited to its start value plus or minus 2π.\n"
        "The target phase is limited to its start value plus or minus π.\n"
        "The time limit applies separately to each start.\n"
        "The max-nfev value limits the solver count.\n"
        "The report also records the residual and Jacobian evaluation counts.\n"
        "The protocol and strict limits use unnormalized Frobenius norms.\n"
        "The protocol limit is 1e-5. The strict limit is 1e-10.\n\n"
        "Read `report_*.json` and `report_*.csv` for the results.\n"
        "Each attempt directory contains its result and any valid circuit.\n"
        "The report gives the relative paths of the best valid circuits.\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def run_classical(baseline: Path, output: Path, *, seeds=(0, 1, 2), max_nfev=200,
                  timeout=60.0, resume=False, encoding_paths=None):
    """Run one fit per start and keep the results for later use."""
    if (isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1
            or isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("The limits must be positive finite numbers")
    seeds = tuple(seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    baseline, output = Path(baseline), Path(output)
    if encoding_paths is not None and len(encoding_paths) != 2:
        raise ValueError("Supply two encoding paths")
    cases = _read_baseline(baseline, encoding_paths)
    config = _config(cases, seeds, max_nfev, float(timeout))
    baseline_manifest = json.loads((baseline / "manifest.json").read_bytes())
    config["git_commit"] = baseline_manifest.get("version", {}).get("git_commit")
    config["baseline_path"] = str(baseline.resolve())
    config["encoding_paths"] = ([str(Path(path).resolve()) for path in encoding_paths]
                                if encoding_paths is not None else None)
    config_hash = _hash(_json_bytes(config))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("The run directory does not exist")
    import fcntl
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _setup(output, cases, config, config_hash)
        _write_readme(output, config)
        records = []
        for index, case in enumerate(cases):
            case_name = f"case_{index + 1:03d}"
            starts = [("original", None, case["angles"])]
            starts += [(f"seed_{seed}", seed,
                        np.random.default_rng(seed).uniform(-np.pi, np.pi, len(case["angles"])).tolist())
                       for seed in seeds]
            for label, seed, parameters in starts:
                trial = output / case_name / label
                trial.mkdir(parents=True, exist_ok=True)
                attempts = sorted(path for path in trial.iterdir() if path.is_dir() and path.name.startswith("attempt_"))
                cached = next((value for path in reversed(attempts) if (value := _completed(path, output)) is not None), None)
                if cached is not None:
                    records.append(cached)
                    continue
                attempt_dir = trial / f"attempt_{len(attempts) + 1:04d}"
                attempt_dir.mkdir()
                result = _attempt(case, parameters, max_nfev, float(timeout))
                result.update({"case": case_name, "encoding": case["name"], "method": case["method"],
                               "start": label, "seed": seed, "initial_parameters": parameters,
                               "max_nfev": max_nfev, "timeout_seconds": timeout,
                               "solver_config": config["solver_config"],
                               "trial_path": str(attempt_dir.relative_to(output)),
                               "source_sha256": case["hashes"], "configuration_sha256": config_hash})
                _save_result(attempt_dir, result, case)
                records.append(result)
        report_number = 1
        while (output / f"report_{report_number:04d}.json").exists():
            report_number += 1
        report_path = output / f"report_{report_number:04d}.json"
        report_path.write_bytes(_json_bytes({"configuration_sha256": config_hash, "solver_config": config["solver_config"], "results": records,
                                             "best": _best(records),
                                             "best_strict": _best(records, strict=True)}))
        with (output / f"report_{report_number:04d}.csv").open("x", newline="", encoding="utf-8") as stream:
            fields = ["case", "encoding", "start", "seed", "status", "is_valid", "strict_valid", "F", "L", "eps", "cz_count", "nfev", "evaluations", "jacobian_evaluations", "wall_seconds"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                metrics = record.get("metrics", {})
                solver = record.get("solver", {})
                writer.writerow({key: record.get(key, metrics.get(key, solver.get(key))) for key in fields})
        return records
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description="Fit the saved CZ3 circuits.")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to the M1 report directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the M2 report directory.")
    parser.add_argument("--seed", type=int, action="append", help="Add a seed. The default seeds are 0, 1, and 2.")
    parser.add_argument("--no-random-starts", action="store_true", help="Run the saved start only.")
    parser.add_argument("--max-nfev", type=int, default=200, help="Set the solver function evaluation limit.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Set the time limit in seconds for each start.")
    parser.add_argument("--resume", action="store_true", help="Continue a run with the same configuration.")
    parser.add_argument("--encoding-a", type=Path, help="Set the encoding for qubits 0 and 1.")
    parser.add_argument("--encoding-b", type=Path, help="Set the encoding for qubits 2 and 3.")
    args = parser.parse_args()
    if (args.encoding_a is None) != (args.encoding_b is None):
        parser.error("Supply both encoding paths")
    if args.no_random_starts and args.seed is not None:
        parser.error("Use --seed or --no-random-starts")
    encodings = None if args.encoding_a is None else (args.encoding_a, args.encoding_b)
    seeds = () if args.no_random_starts else (tuple(args.seed) if args.seed is not None else (0, 1, 2))
    records = run_classical(args.baseline, args.output, seeds=seeds,
                            max_nfev=args.max_nfev, timeout=args.timeout, resume=args.resume,
                            encoding_paths=encodings)
    for record in records:
        metrics = record.get("metrics", {})
        print(f"{record['case']} {record['start']}: {record['status']} valid={metrics.get('is_valid')}")


if __name__ == "__main__":
    main()
