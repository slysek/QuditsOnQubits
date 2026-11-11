"""Run the agent-network arm of the GA-versus-agents comparison.

K Codex agents search in parallel for the same time budget. They share one board file and can read the
valid fits of the other agents. There is no population, no selection and no evolution loop. The agents
use the same angle fit (agent_tools) and the same problem text (_skeleton_template) as the evolution
campaigns. At the end, an independent check (oe_evaluator.assess) measures the best valid structure, and
the script writes best.qpy, best_encodings.npy and decoder.qpy like a campaign.

    python scripts/agent_network.py --state ghz3 --output DIR --agents 3 --minutes 30
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from qiskit import qpy

from qudits_on_qubits.evolution import oe_agent
from qudits_on_qubits.evolution.agent_tools import SESSION_FITS
from qudits_on_qubits.evolution.harness import BASELINES
from qudits_on_qubits.evolution.oe_candidates import skeleton_program
from qudits_on_qubits.evolution.oe_evaluator import CACHE_DIRECTORY, WARM_STARTS, assess
from qudits_on_qubits.evolution.oe_run import _json, _skeleton_template, _source, _warm_pool, method_for
from qudits_on_qubits.evolution.polish import two_qubit_depth
from qudits_on_qubits.evolution.skeleton_fit import cut_seed, skeleton_of
from qudits_on_qubits.evolution.state_workloads import layout

POLL_SECONDS = 20
MIN_SESSION_SECONDS = 120
SHORT_SESSION_SECONDS = 60
MAX_SHORT_SESSIONS = 3
VERIFY_SECONDS = 600
# Fit settings per qubit count, the same as the evolution campaigns use.
FIT = {6: (8, 60), 8: (16, 150), 12: (1, 400)}
TASK = """# Agent {index} of {agents} in a team that searches for short exact qubit circuits

You are one agent in a team of {agents} agents. All agents work on the same problem at the same time.
There is no evolution loop: the team result is the best valid structure that any agent finds.
You have {minutes} minutes in this session. Work only in this directory and in {shared}.

## Result
Each time that you find a better valid structure, write it to {candidate} with the fit tool (--write).
The fit tool also adds each valid structure to {fits} in this directory.

## Team communication
- Shared board: {board}. Read it often. Append short lines: what you try, what fails, your best result.
- The valid structures of the other agents: {others} (read only; each file is a JSON list of
  {{"pairs": ..., "angles": ...}}). Start from the best one if it is better than yours.
- Split the work: try ideas that the board does not show yet.

## Tools
Fit all angles of a CZ structure and check the state:

    {python} -m qudits_on_qubits.evolution.agent_tools fit --cz "[[0, 1], [1, 2]]" --write {candidate}

Suggested options for this state: --restarts {restarts} --seconds {seconds}. The command prints one JSON
line (valid, F, error_norm, cz_count, two_qubit_depth, cut_violations). It writes the file only for a
valid structure. A structure that fails a cut condition gets no fit. The first restart starts from the
most similar valid structure.
You can write and run your own Python scripts with {python}. Useful modules:
qudits_on_qubits.evolution.skeleton_fit (fit_skeleton, cut_requirements, warm_angles) and
qudits_on_qubits.evolution.state_workloads (logical_state, layout).

## Rules
- An independent evaluator checks the full state of the best structure again at the end.
- Do not change files outside this directory and {shared}.
- The problem text below was written for the evolution loop. Ignore its "Current candidate" and
  "Evolution history" parts. Use its "Problem" part.

{problem}
"""


def _setup(state, output, encoding_mode, restarts, *, blank_start=False):
    """Write the campaign files that agent_tools reads and return (options, seed pairs, problem text)."""
    circuit, encoding, files, metrics = _source(Path(BASELINES[state]), state, method_for(state, "estar_direct"))
    output.mkdir(parents=True)
    (output / "encoding.npy").write_bytes(files[1])
    if not blank_start:
        (output / "source.qpy").write_bytes(files[0])
    (output / CACHE_DIRECTORY).mkdir()
    settings = {"state": state, "encoding_path": str(output / "encoding.npy"), "max_nfev": 0,
                "max_parameters": 12, "representation": "skeleton", "encoding_mode": encoding_mode,
                "restarts": restarts, "fit_seed": 0, "objective": "preparation",
                "timeout_seconds": VERIFY_SECONDS, "memory_mb": 4096, "cpu_seconds": VERIFY_SECONDS,
                "architecture": None}
    (output / "evaluation.json").write_bytes(_json(settings))
    (output / WARM_STARTS).write_bytes(_json(_warm_pool(state, encoding_mode, None if blank_start else circuit, encoding)))
    pairs = cut_seed(state) if blank_start else skeleton_of(circuit)
    seed = skeleton_program(circuit.num_qubits, pairs)
    problem = _skeleton_template(state, encoding_mode, known=() if blank_start else pairs).format(
        fitness_score="(none)", metrics=json.dumps({"cz_count": len(pairs), "two_qubit_depth": two_qubit_depth(pairs)}),
        improvement_areas="", artifacts="", evolution_history="(none)", language="python", current_program=seed)
    options = {k: v for k, v in settings.items() if k != "encoding_path"} | {"encoding": encoding}
    return options, pairs, problem


def _fits(output):
    """Return all valid fits of all agents as (agent, pairs, angles)."""
    found = []
    for path in sorted(output.glob(f"agent_*/{SESSION_FITS}")):
        try:
            items = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # the agent can be writing the file; the next poll reads it
        found += [(path.parent.name, [tuple(p) for p in item["pairs"]], item["angles"])
                  for item in items if isinstance(item, dict) and "pairs" in item and "angles" in item]
    return found


def _key(pairs):
    return (len(pairs), two_qubit_depth(pairs))


def _agent(index, agents, output, problem, deadline, agent, fit, blank_start=False):
    """Run sessions of one agent until the deadline; a session that ends early is started again."""
    home = output / f"agent_{index}"
    home.mkdir()
    env = oe_agent._tool_env()
    sessions = short = 0
    while (left := deadline - time.monotonic()) > MIN_SESSION_SECONDS and short < MAX_SHORT_SESSIONS:
        sessions += 1
        began = time.monotonic()
        task = TASK.format(index=index, agents=agents, minutes=int(left // 60), shared=output / "shared",
                           candidate=oe_agent.CANDIDATE, fits=SESSION_FITS, board=output / "shared" / "board.md",
                           others=output / f"agent_*/{SESSION_FITS}", python=sys.executable,
                           restarts=fit[0], seconds=fit[1], problem=problem)
        if blank_start:
            task = task.replace("(logical_state, layout)", "(layout)")
            task += ("\nUse only the supplied target vector and results from this run."
                     " Do not read previous circuits, baselines, records or benchmark results."
                     " Do not read state construction functions. No valid seed circuit is supplied.\n")
        session = home / f"session_{sessions}"
        session.mkdir()
        (session / "TASK.md").write_text(task, encoding="utf-8")
        # The agent may also write to the shared board directory.
        argv = oe_agent._argv(agent, home, env, writable=[output / "shared"])
        oe_agent._run(argv, task, session, int(left), env)
        # A session that ends at once (for example, a usage limit or a rejected login) must not restart in a loop.
        short = short + 1 if time.monotonic() - began < SHORT_SESSION_SECONDS else 0
    if short >= MAX_SHORT_SESSIONS:
        print(f"agent_{index}: stopped after {short} short sessions in a row", flush=True)
    return sessions


_session_usage = oe_agent.session_usage


def _usage(output):
    """Sum the token usage of all sessions."""
    total = {}
    for path in output.glob("agent_*/session_*/events.jsonl"):
        for key, value in _session_usage(path).items():
            total[key] = total.get(key, 0) + value
    return total


def _verify(output, options, ranked):
    """Check the best valid fits independently; write the campaign files of the first one that passes."""
    n, _, q = layout(options["state"])
    for agent, pairs, angles in ranked:
        program = output / "best_program.py"
        program.write_text(skeleton_program(n * q, pairs, angles), encoding="utf-8")
        result = assess(program, **options, x0=np.asarray(angles, dtype=float))
        if not result.metrics["is_valid"]:
            continue
        for name, circuit in (("best.qpy", result.circuit), ("decoder.qpy", result.decoder)):
            if circuit is not None:
                with (output / name).open("xb") as stream:
                    qpy.dump(circuit, stream)
        np.save(output / "best_encodings.npy", result.encodings, allow_pickle=False)
        return {"agent": agent, **{k: float(v) for k, v in result.metrics.items()}}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run K Codex agents with a shared board on one state.")
    parser.add_argument("--state", choices=tuple(BASELINES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--encoding-mode", choices=("fixed", "free"), default="free")
    parser.add_argument("--blank-start", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.agents <= 8 or not 0 < args.minutes <= 600:
        parser.error("--agents must be from 1 to 8 and --minutes from 0 to 600")
    started = time.monotonic()
    output = args.output.resolve()
    n, _, q = layout(args.state)
    fit = FIT.get(n * q, FIT[6])
    options, seed_pairs, problem = _setup(args.state, output, args.encoding_mode, fit[0], blank_start=args.blank_start)
    (output / "shared").mkdir()
    (output / "shared" / "board.md").write_text(f"# Team board\nSeed: {len(seed_pairs)} CZ.\n", encoding="utf-8")
    os.environ["QOQ_OE_CONFIG"] = str(output / "evaluation.json")
    agent = oe_agent.agent_settings()
    deadline = started + args.minutes * 60
    timeline, best = [], None if args.blank_start else _key(seed_pairs)
    stop = threading.Event()

    def poll():
        nonlocal best
        while not stop.wait(POLL_SECONDS):
            for name, pairs, _ in _fits(output):
                if best is None or _key(pairs) < best:
                    best = _key(pairs)
                    timeline.append({"seconds": round(time.monotonic() - started, 1), "agent": name,
                                     "cz_count": best[0], "two_qubit_depth": best[1]})
                    print(f"[{timeline[-1]['seconds']:7.0f} s] new best {best[0]} CZ, 2q depth {best[1]} ({name})",
                          flush=True)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    with ThreadPoolExecutor(args.agents) as pool:
        sessions = list(pool.map(lambda i: _agent(i, args.agents, output, problem, deadline, agent, fit, args.blank_start),
                                 range(1, args.agents + 1)))
    stop.set()
    watcher.join()
    ranked = sorted(_fits(output), key=lambda item: _key(item[1]))
    report = {"state": args.state, "agents": args.agents, "minutes": args.minutes, "agent_settings": agent,
              "blank_start": args.blank_start,
              "seed": {"cz_count": len(seed_pairs), "two_qubit_depth": two_qubit_depth(seed_pairs)},
              "sessions": sessions, "valid_fits": len(ranked), "timeline": timeline,
              "wall_seconds": round(time.monotonic() - started, 1), "usage": _usage(output),
              "best": _verify(output, options, ranked[:5])}
    (output / "network.json").write_bytes(_json(report))
    print(json.dumps({k: report[k] for k in ("best", "wall_seconds", "usage", "valid_fits")}), flush=True)
    return 0 if report["best"] else 1


if __name__ == "__main__":
    sys.exit(main())
