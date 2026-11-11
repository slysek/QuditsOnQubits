"""Give an agent session the angle fit of its campaign as a command (see oe_agent.py).

The session runs this module with QOQ_OE_CONFIG set to the evaluation.json of the campaign:

    python -m qudits_on_qubits.evolution.agent_tools fit --cz "[[0, 1], [1, 2]]" --write candidate.py
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from .architectures import routed_costs
from .oe_candidates import skeleton_program
from .oe_evaluator import CACHE_DIRECTORY, MAX_WARM_BYTES, WARM_STARTS, _config, _known
from .skeleton_fit import TAIL_PAIRS, cut_violations, fit_skeleton, warm_angles
from .state_workloads import layout

# The valid fits of this session in its working directory: [{"pairs": ..., "angles": ...}].
SESSION_FITS = "valid_fits.json"
SESSION_PROGRESS = "fit_progress.json"
MAX_FIT_SECONDS = 600


def _pairs(text):
    try:
        value = ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("--cz must be a literal list of qubit pairs") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("--cz must be a literal list of qubit pairs")
    return [tuple(pair) if isinstance(pair, (list, tuple)) else pair for pair in value]


def fit(pairs, *, restarts=8, seconds=60.0, seed=0, write=None):
    """Fit all angles of the CZ pairs for the campaign state and return a report.

    The first restart starts from the most similar valid structure of the campaign or of this session.
    A structure that fails a cut condition gets no fit. With write, a valid structure is written as a
    candidate file with its angles.
    """
    options = _config()
    state = options["state"]
    n, _, q = layout(state)
    violations = cut_violations(state, pairs)
    report = {"cz_count": len(pairs),
              "cut_violations": [{"parties": list(side), "needed": needed, "present": present}
                                 for side, needed, present in violations]}
    if violations:
        return {**report, "valid": False}
    if not 0 < seconds <= MAX_FIT_SECONDS:
        raise ValueError(f"--seconds must be greater than zero and at most {MAX_FIT_SECONDS}")
    progress_path = Path(SESSION_PROGRESS)
    if progress_path.exists() and progress_path.stat().st_size > MAX_WARM_BYTES:
        raise ValueError("The fit progress file is too large")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    if not isinstance(progress, list) or any(
            not isinstance(entry, dict) or not {"pairs", "angles", "F"} <= entry.keys()
            or not isinstance(entry["F"], (int, float)) or not 0 <= entry["F"] <= 1 for entry in progress):
        raise ValueError("The fit progress file must contain fitted pairs, angles and finite fidelity values")
    campaign = Path(os.environ["QOQ_OE_CONFIG"]).parent
    known = _known(campaign / WARM_STARTS) + _known(Path(SESSION_FITS)) + _known(Path(SESSION_PROGRESS))
    for entry in sorted((campaign / CACHE_DIRECTORY).glob("*.json")):
        known += _known(entry)
    free = options["encoding_mode"] == "free"
    x0 = warm_angles(pairs, known, n * q, n * len(TAIL_PAIRS[q]) if free else 0)
    started = perf_counter()
    fitted = fit_skeleton(state, pairs, None if free else options["encoding"], restarts=restarts, seed=seed,
                          time_budget=seconds, x0=x0)
    metrics = fitted.metrics
    report.update(valid=bool(metrics["is_valid"]), F=float(metrics["F"]), error_norm=float(metrics["error_norm"]),
                  two_qubit_depth=int(metrics["two_qubit_depth"]), restarts=fitted.restarts,
                  seconds=round(perf_counter() - started, 3))
    item = {"pairs": [list(p) for p in pairs], "angles": [float(v) for v in fitted.angles],
            "F": float(metrics["F"]), "error_norm": float(metrics["error_norm"]), "is_valid": bool(metrics["is_valid"])}
    previous = next((entry for entry in progress if entry["pairs"] == item["pairs"]), None)
    if (previous is None or (item["F"], -item["error_norm"]) > (previous["F"], -previous.get("error_norm", float("inf")))
            or item["is_valid"]):
        progress = [entry for entry in progress if entry["pairs"] != item["pairs"]] + [item]
        progress_path.write_text(json.dumps(progress, allow_nan=False))
    if metrics["is_valid"]:
        if options["architecture"] is not None:
            report.update(routed_costs(fitted.circuit, options["architecture"], objective=options["objective"]))
        angles = [float(v) for v in fitted.angles]
        items = [{"pairs": p, "angles": a} for p, a in _known(Path(SESSION_FITS))]
        Path(SESSION_FITS).write_text(json.dumps(items + [{"pairs": [list(p) for p in pairs], "angles": angles}]))
        if write is not None:
            Path(write).write_text(skeleton_program(n * q, pairs, angles), encoding="utf-8")
            report["written"] = write
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use the angle fit of this campaign.")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("fit", help="Fit all angles of a CZ structure and check the state.")
    command.add_argument("--cz", required=True, help='Set the ordered CZ pairs, for example "[[0, 1], [1, 2]]".')
    command.add_argument("--restarts", type=int, default=8, help="Set the fit restarts.")
    command.add_argument("--seconds", type=float, default=60.0, help="Set the fit time limit in seconds.")
    command.add_argument("--seed", type=int, default=0, help="Set the seed of the random restarts.")
    command.add_argument("--write", help="Write a valid structure with its angles to this candidate file.")
    args = parser.parse_args(argv)
    try:
        report = fit(_pairs(args.cz), restarts=args.restarts, seconds=args.seconds, seed=args.seed,
                     write=args.write)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
