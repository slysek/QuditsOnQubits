"""Score literal OpenEvolve candidates for an encoded qudit state."""

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import logging
import math
import os
import struct
import subprocess
import sys
import threading
from time import perf_counter
from pathlib import Path

import numpy as np
from filelock import FileLock
from qiskit import QuantumCircuit, qpy

from .architectures import ARCHITECTURES, routed_costs
from .decoders import decoder_circuit, with_decoder_check
from .oe_candidates import MAX_OPERATIONS, candidate_angles, load_candidate, load_skeleton
from .skeleton_fit import MAX_RESTARTS, TAIL_PAIRS, warm_angles
from .state_workloads import STATES, evaluate_workload, layout, party_encodings

_LOG = logging.getLogger(__name__)
_STATES = {name: layout(name)[0] * layout(name)[2] for name in STATES}
REPRESENTATIONS = ("gates", "skeleton")
ENCODING_MODES = ("fixed", "free")
# cost: the routed success estimate (calibrated 1q, 2q and SWAP errors) ranks first; it needs an architecture.
OBJECTIVES = ("preparation", "total", "cost")
OBJECTIVE_ERROR = f"The objective must be one of {', '.join(OBJECTIVES)}"
LOGICAL_RANK = ("cz_count", "two_qubit_depth", "depth", "one_qubit_gate_count", "decoder_cz_count")
# The success estimate is only an approximation, so the gate counts after routing rank before it.
ROUTED_RANK = ("routed_two_qubit_count", "routed_two_qubit_depth", "routed_success_estimate", "cz_count",
               "routed_depth", "decoder_cz_count")
COST_RANK = ("routed_success_estimate", "routed_two_qubit_count", "routed_two_qubit_depth", "cz_count",
             "routed_depth", "decoder_cz_count")
SUCCESS_KEY = "routed_success_estimate"
# The success estimate ranks in steps of 1e-4.
SUCCESS_STEPS = 10_000
_BASE = MAX_OPERATIONS + 1
_COUNTERS = ("nfev", "evaluations", "jacobian_evaluations", "fit_restarts", "decoder_cz_count")
MAX_SECONDS = 600
# Restarts must end before the process limit, so that an invalid skeleton still returns its residual.
FIT_TIME_SHARE = 0.7
CACHE_DIRECTORY = "evaluation_cache"
# Valid structures with their angles, written at the campaign start: [{"pairs": ..., "angles": ...}].
WARM_STARTS = "warm_starts.json"
MAX_WARM_BYTES = 1 << 24
DUPLICATE_NOTE = ("DUPLICATE: this CZ structure was evaluated before in this search; the result below"
                  " is from the cache. An identical candidate is wasted.\n")


@dataclass(frozen=True)
class Assessment:
    """Keep the checked preparation, its metrics, the encodings and the optional decoder."""

    circuit: QuantumCircuit
    metrics: dict
    encodings: np.ndarray
    decoder: QuantumCircuit | None
    angles: np.ndarray | None = None


def _rank_names(metrics, objective):
    if "routed_two_qubit_count" not in metrics:
        return LOGICAL_RANK
    return COST_RANK if objective == "cost" else ROUTED_RANK


def rank_key(metrics, objective="preparation"):
    """Return the rank of a circuit as a tuple. A larger tuple is a better circuit.

    An invalid circuit ranks by its fidelity, below all valid circuits. A valid circuit ranks by
    LOGICAL_RANK: CZ count, two-qubit depth, depth, U count and decoder CZ count. Metrics with routed
    costs (a campaign with an architecture) rank by ROUTED_RANK: the two-qubit gates and the two-qubit
    depth after routing, then the success estimate, the logical CZ count, the routed depth and the
    decoder CZ count. With objective="total", the first cost includes the decoder CZs to the canonical
    encoding (not routed). With objective="cost", routed metrics rank by COST_RANK: the success estimate
    first, then the routed gate counts; logical metrics (no error data) rank by LOGICAL_RANK.
    """
    if objective not in OBJECTIVES:
        raise ValueError(OBJECTIVE_ERROR)
    if not metrics["is_valid"]:
        fidelity = float(metrics["F"])
        if not math.isfinite(fidelity) or not 0 <= fidelity <= 1:
            raise ValueError("The fidelity must be between zero and one")
        return (0, fidelity)
    key = [1]
    for index, name in enumerate(_rank_names(metrics, objective)):
        value = metrics[name]
        if name == SUCCESS_KEY:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("The success estimate must be between zero and one")
            key.append(round(value * SUCCESS_STEPS))
            continue
        if index == 0 and objective == "total":
            value += metrics["decoder_cz_count"]
        if name.startswith("routed_") and isinstance(value, int) and not isinstance(value, bool):
            # shortcut: routed costs above MAX_OPERATIONS tie at the limit; raise the base for larger circuits.
            value = min(value, MAX_OPERATIONS)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_OPERATIONS:
            raise ValueError("The circuit cost is outside the permitted range")
        key.append(MAX_OPERATIONS - value)
    return tuple(key)


def score_metrics(metrics, objective="preparation"):
    """Return rank_key as one number for OpenEvolve: at most 0.5 when invalid, from 1 to 2 when valid.

    shortcut: a float keeps about 16 digits, so the last routed key (decoder CZ count) can tie; the
    harness compares rank_key itself.
    """
    key = rank_key(metrics, objective)
    if key[0] == 0:
        return key[1] / 2
    value, size = 0, 1
    for digit, name in zip(key[1:], _rank_names(metrics, objective)):
        radix = SUCCESS_STEPS + 1 if name == SUCCESS_KEY else _BASE
        value, size = value * radix + digit, size * radix
    return 1.0 + value / size


def _fit_options(state, max_nfev, max_parameters, representation, encoding_mode, restarts, fit_seed):
    if state not in _STATES:
        raise ValueError(f"The state must be one of {', '.join(_STATES)}")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or not 0 <= max_nfev <= 200:
        raise ValueError("max_nfev must be an integer from 0 to 200")
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int) or not 0 <= max_parameters <= 24:
        raise ValueError("max_parameters must be an integer from 0 to 24")
    if representation not in REPRESENTATIONS or encoding_mode not in ENCODING_MODES:
        raise ValueError("Select a supported representation and encoding mode")
    if encoding_mode == "free" and representation != "skeleton":
        raise ValueError("Free encodings need the skeleton representation")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or not 1 <= restarts <= MAX_RESTARTS:
        raise ValueError(f"restarts must be an integer from 1 to {MAX_RESTARTS}")
    if isinstance(fit_seed, bool) or not isinstance(fit_seed, int) or fit_seed < 0:
        raise ValueError("fit_seed must be a nonnegative integer")


def _assess_inline(program_path, *, state, encoding, max_nfev=0, max_parameters=24,
                   representation="gates", encoding_mode="fixed", restarts=8, fit_seed=0,
                   time_budget=None, x0=None):
    """Check the candidate and return its assessment and fit counters.

    The caller sets the state, encoding and fit limits. The candidate cannot set them.
    """
    _fit_options(state, max_nfev, max_parameters, representation, encoding_mode, restarts, fit_seed)
    party_encodings(state, encoding)
    counters = dict.fromkeys(_COUNTERS, 0)
    if representation == "skeleton":
        from .skeleton_fit import cut_violations, fit_skeleton
        pairs = load_skeleton(program_path, expected_qubits=_STATES[state])
        for side, needed, present in cut_violations(state, pairs):
            raise ValueError(f"The CZ structure has {present} CZ across the cut between parties "
                             f"{', '.join(map(str, side))} and the others; it needs at least {needed}")
        fitted = fit_skeleton(state, pairs, None if encoding_mode == "free" else encoding,
                              restarts=restarts, seed=fit_seed, time_budget=time_budget, x0=x0)
        counters.update(nfev=fitted.nfev, fit_restarts=fitted.restarts,
                        decoder_cz_count=fitted.metrics["decoder_cz_count"])
        return Assessment(fitted.circuit, fitted.metrics, fitted.encodings, fitted.decoder, fitted.angles), counters
    circuit = load_candidate(program_path, expected_qubits=_STATES[state])
    if max_nfev:
        from .state_optimize import fit_state
        fitted = fit_state(state, circuit, encoding, max_nfev=max_nfev,
                           max_parameters=max_parameters, tolerance=1e-10)
        circuit = fitted.circuit
        counters.update(nfev=fitted.nfev, evaluations=fitted.evaluations,
                        jacobian_evaluations=fitted.jacobian_evaluations)
    encodings = party_encodings(state, encoding)
    valid = evaluate_workload(state, circuit, encodings, tolerance=1e-10)["is_valid"]
    return Assessment(circuit, {}, encodings, decoder_circuit(encodings) if valid else None), counters


def _limits(timeout_seconds, memory_mb, cpu_seconds):
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= MAX_SECONDS):
        raise ValueError(f"timeout_seconds must be greater than zero and at most {MAX_SECONDS}")
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or not 256 <= memory_mb <= 4096:
        raise ValueError("memory_mb must be an integer from 256 to 4096")
    if isinstance(cpu_seconds, bool) or not isinstance(cpu_seconds, int) or not 1 <= cpu_seconds <= MAX_SECONDS:
        raise ValueError(f"cpu_seconds must be an integer from 1 to {MAX_SECONDS}")


def _npy(array):
    data = BytesIO()
    np.save(data, array, allow_pickle=False)
    return base64.b64encode(data.getvalue()).decode("ascii")


def _worker():
    import resource

    payload = json.load(sys.stdin)
    _limits(payload["timeout_seconds"], payload["memory_mb"], payload["cpu_seconds"])
    memory = payload["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"]))
    try:
        encoding = np.load(BytesIO(base64.b64decode(payload["encoding"])), allow_pickle=False)
        result, counters = _assess_inline(
            payload["program_path"], state=payload["state"], encoding=encoding,
            max_nfev=payload["max_nfev"], max_parameters=payload["max_parameters"],
            representation=payload["representation"], encoding_mode=payload["encoding_mode"],
            restarts=payload["restarts"], fit_seed=payload["fit_seed"],
            time_budget=FIT_TIME_SHARE * payload["timeout_seconds"], x0=payload.get("x0"))
        output = BytesIO()
        qpy.dump([result.circuit] + ([result.decoder] if result.decoder is not None else []), output)
        response = {"ok": True, "circuits": base64.b64encode(output.getvalue()).decode("ascii"),
                    "encodings": _npy(result.encodings), "counters": counters,
                    "angles": None if result.angles is None else [float(v) for v in result.angles]}
    except (ValueError, TypeError, OSError, OverflowError, ArithmeticError) as exc:
        response = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(response, allow_nan=False))


def _decode_response(stdout, shape):
    try:
        response = json.loads(stdout)
        if not isinstance(response, dict):
            raise ValueError("The candidate process returned invalid data")
        if not response.get("ok"):
            raise ValueError(response.get("message", "The candidate process failed"))
        counters = {name: response["counters"][name] for name in _COUNTERS}
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counters.values()):
            raise ValueError("The candidate process returned invalid data")
        circuits = qpy.load(BytesIO(base64.b64decode(response["circuits"], validate=True)))
        encodings = np.load(BytesIO(base64.b64decode(response["encodings"], validate=True)),
                            allow_pickle=False)
        angles = None if response.get("angles") is None else np.asarray(response["angles"], dtype=float)
    except (KeyError, TypeError, json.JSONDecodeError, struct.error, EOFError, qpy.QpyError) as exc:
        raise ValueError("The candidate process returned invalid data") from exc
    if len(circuits) not in (1, 2) or encodings.shape != shape:
        raise ValueError("The candidate process returned an invalid circuit")
    if angles is not None and (angles.ndim != 1 or not np.isfinite(angles).all()):
        raise ValueError("The candidate process returned invalid angles")
    return circuits[0], (circuits[1] if len(circuits) == 2 else None), encodings, counters, angles


def assess(program_path, *, state, encoding, max_nfev=0, max_parameters=24, representation="gates",
           encoding_mode="fixed", restarts=8, fit_seed=0, timeout_seconds=30.0, memory_mb=2048,
           cpu_seconds=30, objective="preparation", x0=None, architecture=None):
    """Check one candidate in a process with time, CPU and memory limits.

    This process checks the returned state again, and the decoder too if there is one.
    x0 is an optional start point of the skeleton fit (see skeleton_fit.warm_angles).
    architecture (see architectures.ARCHITECTURES) adds the costs after layout and routing to a valid
    candidate, and the score uses them.
    """
    _fit_options(state, max_nfev, max_parameters, representation, encoding_mode, restarts, fit_seed)
    _limits(timeout_seconds, memory_mb, cpu_seconds)
    if objective not in OBJECTIVES:
        raise ValueError(OBJECTIVE_ERROR)
    if architecture is not None and architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")
    party_encodings(state, encoding)
    payload = {"program_path": str(Path(program_path).resolve()), "state": state,
               "encoding": _npy(encoding), "max_nfev": max_nfev, "max_parameters": max_parameters,
               "representation": representation, "encoding_mode": encoding_mode,
               "restarts": restarts, "fit_seed": fit_seed, "timeout_seconds": timeout_seconds,
               "memory_mb": memory_mb, "cpu_seconds": cpu_seconds,
               "x0": None if x0 is None else [float(v) for v in x0]}
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    started = perf_counter()
    try:
        process = subprocess.run([sys.executable, "-m", "qudits_on_qubits.evolution.oe_evaluator", "--worker"],
                                 input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=float(timeout_seconds), env=env)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("The candidate process exceeded the time limit") from exc
    if process.returncode != 0:
        raise ValueError("The candidate process failed")
    n, d, q = layout(state)
    circuit, decoder, encodings, counters, angles = _decode_response(process.stdout, (n, 1 << q, d))
    if encoding_mode == "fixed" and not np.allclose(encodings, encoding, rtol=0, atol=1e-12):
        raise ValueError("The candidate process changed the fixed encoding")
    metrics = {**evaluate_workload(state, circuit, encodings, tolerance=1e-10), **counters,
               "decoder_cz_count": 0}
    if metrics["is_valid"]:
        if decoder is None or decoder.num_qubits != circuit.num_qubits:
            raise ValueError("The candidate process returned no decoder for a valid state")
        metrics = with_decoder_check(state, circuit, decoder, metrics)
        if architecture is not None:
            metrics.update(routed_costs(circuit, architecture, objective=objective))
    metrics.update({"evaluation_failed": 0, "timed_out": 0,
                    "evaluation_seconds": perf_counter() - started,
                    "combined_score": score_metrics(metrics, objective)})
    return Assessment(circuit, metrics, encodings, decoder, angles)


def assess_candidate(program_path, **options):
    """Return the checked circuit and its metrics. See assess for the options."""
    result = assess(program_path, **options)
    return result.circuit, result.metrics


def _config():
    path = os.environ.get("QOQ_OE_CONFIG")
    if not path:
        raise ValueError("Set QOQ_OE_CONFIG to the host configuration path")
    with Path(path).open("rb") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("The host configuration is too large")
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("state") not in _STATES:
        raise ValueError("The host configuration must set a supported state")
    encoding_path = config.get("encoding_path")
    if not isinstance(encoding_path, str) or not encoding_path:
        raise ValueError("The host configuration must set encoding_path")
    with Path(encoding_path).open("rb") as stream:
        raw_encoding = stream.read(8193)
    if len(raw_encoding) > 8192:
        raise ValueError("The encoding file is too large")
    options = {"state": config["state"],
               "encoding": np.load(BytesIO(raw_encoding), allow_pickle=False),
               "max_nfev": config.get("max_nfev", 0), "max_parameters": config.get("max_parameters", 24),
               "representation": config.get("representation", "gates"),
               "encoding_mode": config.get("encoding_mode", "fixed"),
               "restarts": config.get("restarts", 8), "fit_seed": config.get("fit_seed", 0),
               "timeout_seconds": config.get("timeout_seconds", 30.0),
               "memory_mb": config.get("memory_mb", 2048), "cpu_seconds": config.get("cpu_seconds", 30),
               "objective": config.get("objective", "preparation"),
               "architecture": config.get("architecture")}
    if options["architecture"] is not None and options["architecture"] not in ARCHITECTURES:
        raise ValueError("The host configuration sets an unknown architecture")
    _fit_options(options["state"], options["max_nfev"], options["max_parameters"],
                 options["representation"], options["encoding_mode"], options["restarts"],
                 options["fit_seed"])
    party_encodings(options["state"], options["encoding"])
    _limits(options["timeout_seconds"], options["memory_mb"], options["cpu_seconds"])
    return options


def _summary(metrics):
    """One line for the next prompt. The fit residual shows how far an invalid skeleton is."""
    status = "VALID" if metrics["is_valid"] else "INVALID"
    line = (f"{status}: cz={metrics['cz_count']} two_qubit_depth={metrics['two_qubit_depth']} "
            f"depth={metrics['depth']} one_qubit_gates={metrics['one_qubit_gate_count']} "
            f"error_norm={metrics['error_norm']:.3e} leakage_norm={metrics['leakage_norm']:.3e} "
            f"fit_restarts={metrics['fit_restarts']} decoder_cz={metrics['decoder_cz_count']}")
    if "routed_two_qubit_count" in metrics:
        line += (f"\nAfter layout and routing: two_qubit_gates={metrics['routed_two_qubit_count']}"
                 f" two_qubit_depth={metrics['routed_two_qubit_depth']} depth={metrics['routed_depth']}"
                 f" success_estimate={metrics['routed_success_estimate']:.4f}")
    if not metrics["is_valid"]:
        line += (f"\nThe angle fit did not reach the state (fidelity F={metrics['F']:.4f})."
                 " This result does not prove that the topology has no solution."
                 " Continue from the best fitted angles or try another start."
                 " Add CZ gates if needed, or change the structure or construction algorithm.")
    return line


def _failure_metrics(state, exc, timeout_seconds):
    return {"combined_score": 0.0, "F": 0.0, "L": 1.0, "error_norm": 2.0,
            "leakage_norm": 1.0, "is_valid": 0.0, "strict_valid": 0.0,
            "cz_count": float(MAX_OPERATIONS), "cx_count": 0.0,
            "two_qubit_gate_count": float(MAX_OPERATIONS),
            "one_qubit_gate_count": float(MAX_OPERATIONS),
            "depth": float(MAX_OPERATIONS), "two_qubit_depth": float(MAX_OPERATIONS),
            "n": float(layout(state)[0]), "num_qubits": float(_STATES[state]),
            **dict.fromkeys(_COUNTERS, 0.0),
            "evaluation_failed": 1.0, "timed_out": float(isinstance(exc, TimeoutError)),
            "evaluation_seconds": float(timeout_seconds if isinstance(exc, TimeoutError) else 0.0),
            "cache_hit": 0.0}


def _cache_entry(program_path, options):
    """Return the cache file of this candidate, or None if the candidate cannot be read.

    The key is the CZ list of a skeleton with its optional angles (comments and layout do not change it)
    or the program text. An invalid result can be fitted again with a different start or fit budget.
    """
    try:
        if options["representation"] == "skeleton":
            pairs = load_skeleton(program_path, expected_qubits=_STATES[options["state"]])
            angles = candidate_angles(program_path)
            data = json.dumps(pairs if angles is None else [pairs, angles])
        else:
            data = Path(program_path).read_text(encoding="utf-8")
    except (ValueError, TypeError, OSError, SyntaxError, RecursionError):
        return None
    directory = Path(os.environ["QOQ_OE_CONFIG"]).parent / CACHE_DIRECTORY
    return directory / f"{hashlib.sha256(data.encode()).hexdigest()}.json"


def _cached(entry):
    try:
        data = json.loads(entry.read_text(encoding="utf-8"))
        metrics = {key: float(value) for key, value in data["metrics"].items()}
        return {**data, "metrics": metrics, "summary": str(data["summary"])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _cache_rank(metrics: dict, objective: str) -> tuple:
    """Restore integer costs from numeric cache values before ranking them."""
    costs = {name: int(metrics[name]) for name in set(LOGICAL_RANK + ROUTED_RANK) - {SUCCESS_KEY}
             if name in metrics and float(metrics[name]).is_integer()}
    return (*rank_key({**metrics, **costs}, objective), -metrics.get("error_norm", math.inf))


def _store(entry, metrics, summary, warm=None, fit_context=None, objective="preparation"):
    """Write the entry atomically, so that a parallel evaluation never reads half a file.

    warm holds the best fitted angles. Valid and invalid fits can supply start angles.
    """
    entry.parent.mkdir(exist_ok=True)
    with FileLock(str(entry) + ".lock"):
        previous = _cached(entry) if entry.is_file() else None
        current_rank = _cache_rank(metrics, objective)
        old_rank = _cache_rank(previous["metrics"], objective) if previous is not None else ()
        if old_rank > current_rank:
            metrics, summary = previous["metrics"], previous["summary"]
            warm = {key: previous[key] for key in ("pairs", "angles") if key in previous}
        temporary = entry.with_name(f"{entry.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps({"metrics": metrics, "summary": summary, "fit_context": fit_context,
                                         **(warm or {})}, allow_nan=False), encoding="utf-8")
        os.replace(temporary, entry)
    return metrics, summary


def _known(path):
    """Return the (pairs, angles) items of a warm-start file or cache entry. A bad file gives none."""
    try:
        if path.stat().st_size > MAX_WARM_BYTES:
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        return [(item["pairs"], item["angles"]) for item in items if isinstance(item, dict) and "angles" in item]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def _warm_start(program_path, options, directory=None, *, exact=False):
    """Return the candidate angles, else angles from the most similar fitted structure, else None.

    directory is the campaign directory; the default is the directory of QOQ_OE_CONFIG.
    Invalid angles in the candidate raise ValueError, so that the evaluation rejects the candidate.
    """
    if options["representation"] != "skeleton":
        return None
    n, _, q = layout(options["state"])
    try:
        pairs = load_skeleton(program_path, expected_qubits=n * q)
    except (ValueError, TypeError, OSError, SyntaxError, RecursionError):
        return None
    angles = candidate_angles(program_path)
    if angles is not None and not exact:
        return angles
    directory = Path(directory or Path(os.environ["QOQ_OE_CONFIG"]).parent)
    known = _known(directory / WARM_STARTS)
    entries = []
    for entry in (directory / CACHE_DIRECTORY).glob("*.json"):
        saved = _cached(entry)
        if saved is not None:
            entries.append((saved["metrics"].get("F", 0.0), entry))
    for _, entry in sorted(entries, key=lambda item: item[0], reverse=True):
        known += _known(entry)
    if exact:
        normalized = [tuple(sorted(pair)) for pair in pairs]
        known = [(p, a) for p, a in known if [tuple(sorted(pair)) for pair in p] == normalized]
    tail = n * len(TAIL_PAIRS[q]) if options["encoding_mode"] == "free" else 0
    fitted = warm_angles(pairs, known, n * q, tail)
    return fitted if fitted is not None else angles


def _fit_context(options: dict, x0: np.ndarray | list[float] | None) -> str:
    """Identify the start angles and settings of one numerical fit."""
    settings = {key: options[key] for key in ("fit_seed", "restarts", "timeout_seconds", "cpu_seconds",
                                             "memory_mb", "max_nfev", "max_parameters")}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    if x0 is not None:
        digest.update(np.asarray(x0, dtype="<f8").tobytes())
    return digest.hexdigest()


def evaluate(program_path):
    """Return numeric metrics and a text summary for OpenEvolve 0.4.0.

    A valid cached candidate needs no new fit. An invalid candidate can use an improved start.
    """
    from openevolve.evaluation_result import EvaluationResult

    options = _config()
    entry = _cache_entry(program_path, options)
    hit = _cached(entry) if entry is not None and entry.is_file() else None
    try:
        x0 = _warm_start(program_path, options)
        context = _fit_context(options, x0)
        if hit is not None and (hit["metrics"]["is_valid"] or hit.get("fit_context") == context):
            return EvaluationResult(metrics={**hit["metrics"], "evaluation_seconds": 0.0, "cache_hit": 1.0},
                                    artifacts={"evaluation": DUPLICATE_NOTE + hit["summary"]})
        result = assess(program_path, **options, x0=x0)
        metrics = result.metrics
    except (ValueError, TypeError, OSError, OverflowError, ArithmeticError) as exc:
        _LOG.warning("The candidate evaluation failed: %s", exc)
        return EvaluationResult(metrics=_failure_metrics(options["state"], exc, options["timeout_seconds"]),
                                artifacts={"evaluation": f"REJECTED: {exc}"})
    values = {key: float(value) for key, value in metrics.items()} | {"cache_hit": 0.0}
    spent = {key: values[key] for key in (*_COUNTERS, "evaluation_seconds") if key in values}
    summary = _summary(metrics)
    if entry is not None:
        warm = None
        if getattr(result, "angles", None) is not None and options["representation"] == "skeleton":
            warm = {"pairs": [list(p) for p in load_skeleton(program_path, expected_qubits=_STATES[options["state"]])],
                    "angles": [float(v) for v in result.angles]}
        try:
            values, summary = _store(entry, values, summary, warm, context, options["objective"])
        except (OSError, ValueError) as exc:
            # The cache only saves time; a failed write must not lose the evaluation.
            _LOG.warning("The evaluation cache write failed: %s", exc)
    return EvaluationResult(metrics={**values, **spent}, artifacts={"evaluation": summary})


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _worker()
