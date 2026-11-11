"""Run and record state experiments from one command."""

import argparse
from datetime import datetime
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import struct
import sys
from itertools import product
from time import monotonic

from .architectures import ARCHITECTURES, GROUPS, select
from .artifacts import main_checkout
from .checkpoints import resume_operation, save as _save, saved_run, saved_sessions
from .menu import (
    KEY_VARIABLE, apply_settings, ask, ask_int, ask_new_path, choose, pick_codex_effort, pick_codex_fast,
    pick_codex_model, pick_model, review, settings_menu,
)
from .oe_agent import MAX_SESSION_SECONDS
from .oe_evaluator import (ENCODING_MODES, MAX_RESTARTS, MAX_SECONDS, OBJECTIVE_ERROR, OBJECTIVES, REPRESENTATIONS,
                           _fit_options, _limits)
from .state_workloads import AME46_STATES, AME46_TARGETS, GENERATED_BASELINES
from .oe_run import (
    MAX_ITERATIONS, MAX_WORKERS, METHODS, OFFLINE_LABELS, OFFLINE_MODELS, read_prompt_notes, run_campaign,
)

ROOT = main_checkout(Path.cwd() if (Path.cwd() / "artifacts/evolution/openevolve").is_dir()
                     else Path(__file__).resolve().parents[3])
BASELINES = {state: ROOT / "artifacts/evolution/openevolve" / f"{state}_terra_baseline_20261102"
             for state in ("ghz3", "ame43")}
BASELINES["two_qutrit"] = ROOT / "artifacts/evolution/state_workloads/run_20261102_01"
BASELINES = {state: path if path.is_dir() else ROOT / "experiment_inputs/state_preparation"
             for state, path in BASELINES.items()}
# Written by `qoq-evolution ame46-import`.
BASELINES.update({state: AME46_TARGETS.parent / "baseline" for state in AME46_STATES})
BASELINES["ame44"] = ROOT / "artifacts/evolution/ame44/baseline"
BASELINES["ame64"] = ROOT / "artifacts/evolution/ame64/baseline"
BASELINES.update({state: ROOT / "artifacts/evolution" / state / "baseline" for state in GENERATED_BASELINES})
DEFAULT_STATES = ("ghz3", "ame43")
MODEL_CONFIGS = ROOT / "artifacts/evolution/model_configs"
NARRATOR_MODEL = "gpt-6.1-terra"
# Fixed families give only the baseline for AME(4,6) (the other seeds fail the fit), so a model proposes them.
ORCHESTRATOR_MODEL = "claude-opus-5-5"
COMMANDS = {"baseline": "baseline", "classical": "classical",
            "state-benchmark": "state_benchmark", "campaign": "oe_run", "discover": "discovery",
            "ame46-import": "ame46"}


def _check_baseline(baseline, state=None):
    if state in GENERATED_BASELINES:
        from .state_benchmark import ensure_state_baseline
        ensure_state_baseline(state, baseline)
    if not all((Path(baseline) / name).is_file() for name in ("manifest.json", "summary.json")):
        raise ValueError("The baseline must contain a manifest and a summary")


def run_batch(output, baselines, *, seeds=(0,), method="estar_direct", iterations=1,
              model_config=None, codex_auth_file=None, campaign_timeout_seconds=900,
              series_timeout_seconds=1800, timeout_seconds=30, cpu_seconds=30, memory_mb=2048,
              max_nfev=0, max_parameters=12, representation="gates", encoding_mode="fixed",
              restarts=8, workers=1, offline_model="fixed", objective="preparation", prompt_notes=None,
              narrator_model_config=None, architectures=(), blank_start=False, resume=False):
    """Run each state from its own baseline and stop after a failure.

    architectures are names or the groups all, ibm and iqm (see architectures.select). Each state and
    seed then gets one campaign per architecture, in order; none gives one campaign without routing.
    """
    parameters = locals().copy()
    started = monotonic()
    architectures = select(architectures)
    if not baselines or any(state not in BASELINES for state in baselines):
        raise ValueError("Select at least one supported state")
    seeds = tuple(seeds)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if method not in METHODS:
        raise ValueError("Select a supported preparation method")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"Iterations must be from 1 to {MAX_ITERATIONS}")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"Workers must be from 1 to {MAX_WORKERS}")
    if objective not in OBJECTIVES:
        raise ValueError(OBJECTIVE_ERROR)
    if objective == "cost" and not architectures:
        raise ValueError("The cost objective needs an architecture (the error data of a processor)")
    if prompt_notes is not None and representation != "skeleton":
        raise ValueError("Prompt notes need the skeleton representation")
    read_prompt_notes(prompt_notes)
    if offline_model not in OFFLINE_MODELS or (offline_model != "fixed" and representation != "skeleton"):
        raise ValueError("Random mutations and the agent need the skeleton representation")
    if blank_start and representation != "skeleton":
        raise ValueError("A blank start needs the skeleton representation")
    if model_config is None and offline_model == "fixed" and iterations != 1:
        raise ValueError("The offline model uses one iteration")
    if model_config is not None and not Path(model_config).is_file():
        raise ValueError("The model configuration file does not exist")
    if (isinstance(campaign_timeout_seconds, bool) or not isinstance(campaign_timeout_seconds, (int, float))
            or not 0 < campaign_timeout_seconds <= 1_000_000_000 or not math.isfinite(campaign_timeout_seconds)):
        raise ValueError("The campaign time limit must be positive")
    if (isinstance(series_timeout_seconds, bool) or not isinstance(series_timeout_seconds, (int, float))
            or not 0 < series_timeout_seconds <= 1_000_000_000 or not math.isfinite(series_timeout_seconds)):
        raise ValueError("The series time limit must be positive")
    for state in baselines:
        _fit_options(state, max_nfev, max_parameters, representation, encoding_mode, restarts, 0)
    _limits(timeout_seconds, memory_mb, cpu_seconds)
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError("The output directory already exists")
    for state, baseline in baselines.items():
        _check_baseline(baseline, state)
    parameters.update(seeds=seeds, architectures=architectures)
    with saved_run(output, "run", parameters, resume) as checkpoint:
        path = output / "batch.json"
        if resume:
            report = json.loads(path.read_text())
            if report["status"] == "completed":
                return 0
            report["status"] = "running"
            report.pop("error", None)
        else:
            report = {"status": "running",
                      "model_mode": OFFLINE_LABELS[offline_model] if model_config is None else "configured_model",
                      "iterations": iterations, "method": method, "campaign_timeout_seconds": campaign_timeout_seconds,
                      "representation": representation, "encoding_mode": encoding_mode, "restarts": restarts,
                      "workers": workers, "objective": objective, "architectures": list(architectures),
                      "blank_start": blank_start,
                      "series_timeout_seconds": series_timeout_seconds,
                      "runs": []}
        _save(path, report)
        for state, baseline in baselines.items():
            for seed, architecture in product(seeds, architectures or (None,)):
                remaining = series_timeout_seconds - checkpoint["elapsed_seconds"] - (monotonic() - started)
                if remaining <= 0:
                    report.update(status="failed", error="SeriesTimeout")
                    _save(path, report)
                    print("The series time limit is exceeded.", file=sys.stderr, flush=True)
                    return 1
                name = f"{state}_seed_{seed}" + (f"_{architecture}" if architecture else "")
                record = next((row for row in report["runs"] if row["directory"] == name), None)
                if record is not None and record["status"] == "completed":
                    continue
                if record is None:
                    record = {"state": state, "seed": seed, "architecture": architecture, "directory": name,
                              "status": "running"}
                    report["runs"].append(record)
                record["status"] = "running"
                record.pop("error", None)
                _save(path, report)
                print(f"\nCampaign {name}: start", flush=True)
                try:
                    result = (resume_operation(output / name) if (output / name / "resume.json").is_file() else
                              run_campaign(Path(baseline), output / name, state=state, method=method, seed=seed,
                                          iterations=iterations, model_config=model_config,
                                          codex_auth_file=codex_auth_file, campaign_timeout_seconds=min(campaign_timeout_seconds, remaining),
                                          timeout_seconds=timeout_seconds, cpu_seconds=cpu_seconds, memory_mb=memory_mb,
                                          max_nfev=max_nfev, max_parameters=max_parameters,
                                          representation=representation, encoding_mode=encoding_mode,
                                          restarts=restarts, workers=workers, offline_model=offline_model,
                                          objective=objective, prompt_notes=prompt_notes,
                                          narrator_model_config=narrator_model_config, architecture=architecture,
                                          blank_start=blank_start))
                    metrics = result["saved"]
                    if not metrics["is_valid"]:
                        raise ValueError("The saved state did not pass validation")
                    record.update({key: metrics[key] for key in ("cz_count", "two_qubit_depth", "depth", "one_qubit_gate_count", "F", "L")})
                    record.update({key: value for key, value in result.get("best", {}).items() if key.startswith("routed_")})
                    record["decoder_cz_count"] = metrics.get("decoder_cz_count", 0)
                    record["status"] = "completed"
                except KeyboardInterrupt:
                    record["status"] = report["status"] = "paused"
                    _save(path, report)
                    raise
                except Exception as exc:
                    error = "SeriesTimeout" if isinstance(exc, TimeoutError) and remaining <= campaign_timeout_seconds else type(exc).__name__
                    record.update(status="failed", error=error)
                    report["status"] = "failed"
                    _save(path, report)
                    print(f"{name} failed: {type(exc).__name__}: {str(exc)[:120]}. Full log: {output / name / 'campaign.log'}",
                          file=sys.stderr, flush=True)
                    return 1
                _save(path, report)
                if checkpoint["elapsed_seconds"] + monotonic() - started >= series_timeout_seconds:
                    report.update(status="failed", error="SeriesTimeout")
                    _save(path, report)
                    print("The series time limit is exceeded.", file=sys.stderr, flush=True)
                    return 1
        report["status"] = "completed"
        _save(path, report)
        return 0


def _readout(state, circuit_path, encoding_path, output, edges, seeds):
    from qiskit import qpy
    import numpy as np
    from .state_readout import compile_readout

    output = Path(output)
    if output.exists():
        raise FileExistsError("The output directory already exists")
    try:
        with Path(circuit_path).open("rb") as stream:
            circuits = qpy.load(stream)
    except (qpy.QpyError, EOFError, struct.error) as exc:
        raise ValueError("The QPY file is invalid") from exc
    if len(circuits) != 1:
        raise ValueError("The QPY file must contain one circuit")
    encoding = np.load(encoding_path, allow_pickle=False)
    result = compile_readout(state, circuits[0], encoding, edges, seeds=seeds)
    output.mkdir(parents=True, exist_ok=False)
    for name, circuit in result["circuits"].items():
        with (output / f"{name}.qpy").open("xb") as stream:
            qpy.dump(circuit, stream)
    _save(output / "summary.json", result["metadata"])
    print(f"Readout saved in {output}", flush=True)
    return 0


def _edge(value):
    try:
        left, right = value.split(",")
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use two qubit indices separated by a comma") from exc


def _parser():
    parser = argparse.ArgumentParser(description="Run state experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a series of state campaigns.")
    run.add_argument("--output", type=Path, required=True, help="Set a new output directory.")
    run.add_argument("--state", choices=tuple(BASELINES), action="append",
                     help="Select a state. The default is ghz3 and ame43.")
    run.add_argument("--seed", type=int, action="append", help="Add a seed. The default is 0.")
    for state in BASELINES:
        run.add_argument(f"--baseline-{state}", type=Path, default=BASELINES[state], help=f"Set the {state} baseline directory.")
    run.add_argument("--representation", choices=REPRESENTATIONS, default="gates",
                     help="Evolve gate lists or CZ skeletons. A skeleton gets all angles fitted.")
    run.add_argument("--encoding-mode", choices=ENCODING_MODES, default="fixed",
                     help="Keep the baseline encoding or fit one encoding per party (skeleton only).")
    run.add_argument("--restarts", type=int, default=8, help="Set the angle fit restarts per skeleton.")
    run.add_argument("--workers", type=int, default=1, help="Set the parallel OpenEvolve workers.")
    run.add_argument("--offline-model", choices=OFFLINE_MODELS, default="fixed",
                     help="Without --model-config: a fixed test candidate, random skeleton mutations, or one"
                          " Codex agent session per iteration (network, Codex quota; see QOQ_AGENT_*).")
    run.add_argument("--blank-start", action="store_true",
                     help="Start from the cut conditions of the target, without the baseline structure.")
    run.add_argument("--prompt-notes", type=Path,
                     help="Add a UTF-8 text file (at most 4096 bytes) to the skeleton prompt.")
    run.add_argument("--objective", choices=OBJECTIVES, default="preparation",
                     help="Rank by preparation CZs, by preparation plus decoder CZs, or (cost, needs"
                          " --architecture) by the routed success estimate before the gate counts.")
    run.add_argument("--method", choices=METHODS, default="estar_direct")
    run.add_argument("--iterations", type=int, default=1, help="Set the iterations per campaign.")
    run.add_argument("--model-config", type=Path, help="Set the model configuration. The default is an offline mock.")
    run.add_argument("--narrator-model-config", type=Path, help="Use this model for short progress summaries (optional).")
    run.add_argument("--codex-auth-file", type=Path, help="Set the Codex account file for quota checks.")
    run.add_argument("--campaign-timeout-seconds", type=float, default=900, help="Set the time limit per campaign.")
    run.add_argument("--series-timeout-seconds", type=float, default=1800, help="Set the time limit for the series.")
    run.add_argument("--timeout-seconds", type=float, default=30, help="Set the time limit per candidate.")
    run.add_argument("--cpu-seconds", type=int, default=30, help="Set the CPU time limit per candidate.")
    run.add_argument("--memory-mb", type=int, default=2048, help="Set the memory limit per candidate.")
    run.add_argument("--max-nfev", type=int, default=0, help="Set the angle solver call limit.")
    run.add_argument("--max-parameters", type=int, default=12, help="Set the angle solver parameter limit.")
    run.add_argument("--architecture", choices=(*GROUPS, *ARCHITECTURES), action="append",
                     help="Rank by the costs after layout and routing on a device; one campaign per device. "
                          "Use a name, or all, ibm or iqm; repeat for more. The default is no routing.")
    readout = sub.add_parser("readout", help="Decode, route and measure a state offline.")
    readout.add_argument("--state", choices=tuple(BASELINES), required=True)
    readout.add_argument("--circuit", type=Path, required=True, help="Set the input QPY file.")
    readout.add_argument("--encoding", type=Path, required=True, help="Set the input encoding file.")
    readout.add_argument("--output", type=Path, required=True, help="Set a new output directory.")
    readout.add_argument("--edge", type=_edge, action="append", required=True, help="Add a coupling edge as A,B.")
    readout.add_argument("--seed", type=int, action="append", help="Add a routing seed. The default is 0.")
    resume = sub.add_parser("resume", help="Resume a saved operation.")
    selection = resume.add_mutually_exclusive_group()
    selection.add_argument("output", type=Path, nargs="?", help="Set the saved output directory. Omit it to select a session.")
    selection.add_argument("--last", action="store_true", help="Resume the latest saved session without a selection menu.")
    sub.add_parser("menu", help="Enter the series settings in the terminal.")
    for command in COMMANDS:
        sub.add_parser(command, help=f"Run the {command} command.", add_help=False)
    return parser


# The other commands stay available on the command line (qoq-evolution --help).
EXPERIMENTS = (
    ("discover", "the whole pipeline: ideation -> structure families -> one evolution loop per family -> best"),
    ("settings", "save the CLIProxy key and address (kept outside the repository)"),
    ("resume", "select a saved session, with the latest session first"),
)
STATE_CHOICES = (("ghz3", "GHZ3 graph state: 3 qutrits on 6 qubits"),
                 ("ame43", "AME(4,3): 4 qutrits on 8 qubits; angle fits are slow"),
                 ("two_qutrit", "two-qutrit graph state: 2 qutrits on 4 qubits"),
                 ("Bruzda60", "AME(4,6), real, LU class of Pandya60: 4 parties on 12 qubits; very slow fits"),
                 ("Pandya60prime", "AME(4,6), real: 4 parties on 12 qubits; very slow fits"),
                 ("HZ96", "AME(4,6), real, LU class of HZ_rotated288_real: 4 parties on 12 qubits; very slow fits"),
                 ("Casas36", "AME(4,6), complex, arXiv:2504.05394: 4 parties on 12 qubits; very slow fits"),
                 ("aharonov3", "Aharonov singlet: 3 qutrits on 6 qubits"),
                 ("dicke4", "Dicke state with occupations (2,1,1): 4 qutrits on 8 qubits"),
                 ("cluster4", "Cluster state on a chain: 4 qutrits on 8 qubits"),
                 ("aklt4", "Spin-1 AKLT state on a ring: 4 qutrits on 8 qubits"),
                 ("ame45", "AME(4,5): 4 parties on 12 qubits with 3 unused levels per block"),
                 ("ame47", "AME(4,7): 4 parties on 12 qubits with 1 unused level per block"),
                 ("ame48", "AME(4,8): 4 parties on 12 qubits with no unused levels"))
ENCODING_CHOICES = (("fixed", "keep the baseline encoding of each party (qutrit or AME(4,6) party)"),
                    ("free", "fit one encoding per party; the decoder cost is reported apart"))
PROCESSOR_CHOICES = (("none", "no processor: rank by the logical CZ count"),
                     *((name, text) for name, (_, text) in ARCHITECTURES.items()))
OBJECTIVE_CHOICES = (("preparation", "fewest two-qubit gates first (after routing with a processor)"),
                     ("cost", "largest success estimate after routing first (calibrated 1q, 2q and SWAP errors);"
                              " needs a processor; more gates can win; records in <processor>_cost/"))
LOOP_CHOICES = (("agent", "recommended: evolve construction programs with Codex agent sessions"
                          " (network, Codex quota)"),
                ("model", "experimental local CZ strategies through CLIProxy; 16 proposals per request"),
                ("mutations", "random CZ edits without a network; a search baseline"))
AGENT_SESSION_MINUTES = 15
# Time for the evaluations and the final check of a campaign, on top of its agent sessions.
CAMPAIGN_MARGIN_SECONDS = 600


def _default_output(kind):
    return str(ROOT / "artifacts/evolution" / kind / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}")


def _search_settings(states, skeleton, single):
    """Ask the settings that every search uses and return them as command options.

    AME(4,6) fits from random angles do not converge; only the warm start from known circuits works,
    so one restart is the default there.
    """
    ame46 = any(state in AME46_STATES for state in states)
    slow = ame46 or any(state in ("ame43", "dicke4", "cluster4", "aklt4", "ame45", "ame47", "ame48") for state in states)
    options = []
    if not single:
        options += ["--workers", str(ask_int("Parallel workers", 4, 1, MAX_WORKERS,
                                             "Number of candidates that are evaluated at the same time."))]
    if skeleton:
        options += ["--restarts", str(ask_int("Angle-fit restarts per candidate", 1 if ame46 else 16 if slow else 8, 1, MAX_RESTARTS,
                                              "More restarts find a valid fit more often but take longer."))]
    seconds = ask_int("Time limit per candidate in seconds",
                      (MAX_SECONDS if ame46 else 150 if slow else 60) if skeleton else 30, 1, MAX_SECONDS,
                      "The evaluation of one candidate stops after this time (wall and CPU time).")
    return options + ["--timeout-seconds", str(seconds), "--cpu-seconds", str(seconds)]


def _loop_options():
    """Ask who writes the loop candidates. Return the command options and the agent environment."""
    loop = choose("Loop: who writes each candidate?", LOOP_CHOICES)
    if loop == "model":
        model = pick_model("Loop LLM", MODEL_CONFIGS, "random CZ edits without a network")
        return (["--model-config", model] if model else ["--offline-model", "mutations"]), {}
    if loop == "mutations":
        return ["--offline-model", "mutations"], {}
    model = pick_codex_model("Agent")
    effort = pick_codex_effort("Agent", model)
    fast = pick_codex_fast("Agent", model)
    minutes = ask_int("Agent: time limit per session in minutes", AGENT_SESSION_MINUTES, 1,
                      MAX_SESSION_SECONDS // 60, "A candidate that the session wrote before the limit counts.")
    # Empty values give the Codex defaults and replace values that the shell set before.
    return ["--offline-model", "agent"], {"QOQ_AGENT_MODEL": model,
                                          "QOQ_AGENT_EFFORT": effort, "QOQ_AGENT_FAST": "1" if fast else "",
                                          "QOQ_AGENT_SECONDS": str(60 * minutes)}


def _narrator_options():
    narrator = pick_model("Narrator (short summaries)", MODEL_CONFIGS, "no summaries; only short progress lines",
                          default_model=NARRATOR_MODEL, default_effort="low")
    return ["--narrator-model-config", narrator] if narrator else []


def _resume_output(last: bool = False) -> Path:
    """Select a saved session. An explicit path remains available."""
    sessions = saved_sessions(ROOT / "artifacts/evolution")
    if last and not sessions:
        raise ValueError("No saved sessions are available")
    if not sessions:
        return Path(ask("Saved output directory", note="No saved sessions are available in the session list."))
    choices = {}
    descriptions = []
    for index, session in enumerate(sessions, 1):
        output, options = session["output"], session["options"]
        state = options.get("state") or ", ".join(options.get("baselines", {})) or "unspecified state"
        status = session.get("status", "unknown")
        if status == "running":
            status = "running or interrupted"
        label = f"{output.name} [{index}]"
        updated = datetime.fromtimestamp(session["modified"]).strftime("%Y-%m-%d %H:%M")
        description = (f"{updated} | {session['kind']} | {state} | {status} | "
                       f"{session['elapsed_seconds']:.0f} s active | {output.parent}")
        choices[label] = session
        descriptions.append((label, description))
    label = next(iter(choices)) if last else choose("Saved sessions (newest first)",
                                                   [*descriptions, ("path", "enter a saved output directory")])
    if label == "path":
        return Path(ask("Saved output directory"))
    session = choices[label]
    print(f"Resuming the saved session in {session['output']} after {session['elapsed_seconds']:.0f} s of active time."
          " Completed stages remain saved.", flush=True)
    return session["output"]


def _menu():
    print("QuditsOnQubits evolution harness. Each step shows the options; Enter keeps the default.")
    print("Select 'resume' to list saved sessions. Use 'qoq-evolution resume --last' to resume the latest session.")
    if not os.environ.get(KEY_VARIABLE):
        print("No CLIProxy key yet: select 'settings' to save it, or use the Codex agent or random mutations.")
    while True:
        operation = choose("What do you want to run?", EXPERIMENTS)
        if operation == "settings":
            settings_menu()
        elif operation == "resume":
            return ["resume"]
        else:
            return _discover_menu()


def _discover_menu():
    """Ask the state, the encoding, the objective, the target processor, then the orchestrator, loop and
    narrator models, then the limits.

    The agent settings go to the environment of the started command, so the menu sets them after the start.
    """
    from .discovery import MAX_FAMILIES, MAX_ROUNDS

    state = choose("Which state do you want to optimize?", STATE_CHOICES)
    encoding = choose("Encoding", ENCODING_CHOICES)
    objective = choose("Optimize for", OBJECTIVE_CHOICES)
    # The cost needs the error data of a processor.
    processor = choose("Target quantum processor (the score then counts the gates after layout and routing)",
                       PROCESSOR_CHOICES[1:] if objective == "cost" else PROCESSOR_CHOICES)
    ideation = pick_model("Orchestrator (ideation)", MODEL_CONFIGS,
                          "fixed families without a model: baseline, hub and chain",
                          default_model=ORCHESTRATOR_MODEL, default_effort="high")
    loop_options, agent_env = _loop_options()
    loop_options += _narrator_options()
    families = ask_int("Families from ideation", 3, 1, MAX_FAMILIES, "Each family gets its own evolution loop.")
    rounds = ask_int("Rounds", 2, 1, MAX_ROUNDS, "From round 2, each family sees the best structures of the others.")
    iterations = ask_int("Iterations per family and round", 5 if agent_env else 20, 1, MAX_ITERATIONS,
                         "Each iteration makes one new candidate.")
    search = _search_settings([state], True, False)
    session = int(agent_env.get("QOQ_AGENT_SECONDS", 0))
    workers = int(search[search.index("--workers") + 1])
    # The workers run the agent sessions of one campaign in parallel batches.
    default_campaign = -(-iterations // workers) * session + CAMPAIGN_MARGIN_SECONDS if agent_env else 1800
    campaign = ask_int("Time limit per campaign in seconds", default_campaign, 1, 10**9)
    total = ask_int("Time limit for the whole discovery in seconds", campaign * families * rounds, 1, 10**9)
    output = ask_new_path("Output directory (new)", _default_output("discovery"))
    args = ["discover", "--output", output, "--state", state, "--encoding-mode", encoding]
    if processor != "none":
        args += ["--architecture", processor]
    args += ["--objective", objective]
    if state not in GENERATED_BASELINES and not (BASELINES[state] / "manifest.json").is_file():
        args += ["--baseline", ask(f"{state} baseline directory", note="The default baseline is missing.")]
    if ideation:
        args += ["--ideation-model-config", ideation]
    args += [*loop_options, "--families", str(families), "--rounds", str(rounds), "--iterations", str(iterations),
             *search, "--campaign-timeout-seconds", str(campaign), "--discovery-timeout-seconds", str(total)]
    notes = []
    if ideation:
        notes.append("1 orchestrator request for ideation")
    if "--model-config" in loop_options:
        notes.append(f"up to {families * rounds * iterations} loop strategy requests;"
                     " each strategy supplies at most 16 local proposals per worker and model")
    if agent_env:
        notes.append(f"about {families * rounds * iterations} Codex agent sessions of at most {session // 60} min"
                     " each (Codex quota)")
        notes.append("agent settings (put them before the command to run it again): "
                     + " ".join(f"{name}={value}" for name, value in agent_env.items() if value))
    if not review(args, notes):
        return None
    os.environ.update(agent_env)
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv) or ["menu"]
    if argv[0] in COMMANDS:
        module = "qudits_on_qubits.evolution." + COMMANDS[argv[0]]
        # The child receives Ctrl+C. The parent waits until the child saves its state.
        previous = signal.signal(signal.SIGINT, lambda *_: None)
        try:
            return subprocess.run([sys.executable, "-m", module, *argv[1:]], check=False).returncode
        finally:
            signal.signal(signal.SIGINT, previous)
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "menu":
            try:
                chosen = _menu()
            except KeyboardInterrupt:
                parser.exit(130, "Cancelled\n")
            return 0 if chosen is None else main(chosen)
        if args.command == "resume":
            result = resume_operation(args.output if args.output is not None else _resume_output(args.last))
            if isinstance(result, int):
                return result
            return 0 if result.get("status") == "completed" or result.get("saved", {}).get("is_valid") else 1
        if args.command == "readout":
            return _readout(args.state, args.circuit, args.encoding, args.output, args.edge, tuple(args.seed or [0]))
        states = args.state or list(DEFAULT_STATES)
        if len(states) != len(set(states)):
            raise ValueError("Select each state only once")
        selected = {state: getattr(args, f"baseline_{state}") for state in states}
        return run_batch(args.output, selected, seeds=tuple(args.seed or [0]), method=args.method,
                         iterations=args.iterations, model_config=args.model_config,
                         codex_auth_file=args.codex_auth_file, campaign_timeout_seconds=args.campaign_timeout_seconds,
                         series_timeout_seconds=args.series_timeout_seconds, timeout_seconds=args.timeout_seconds, cpu_seconds=args.cpu_seconds, memory_mb=args.memory_mb,
                         max_nfev=args.max_nfev, max_parameters=args.max_parameters,
                         representation=args.representation, encoding_mode=args.encoding_mode,
                         restarts=args.restarts, workers=args.workers, offline_model=args.offline_model,
                         objective=args.objective, prompt_notes=args.prompt_notes,
                         narrator_model_config=args.narrator_model_config,
                         architectures=tuple(args.architecture or ()), blank_start=args.blank_start)
    except KeyboardInterrupt:
        return 130
    except EOFError:
        parser.exit(2, "Error: Input ended\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def entrypoint():
    # Multithreaded BLAS makes the small angle fits up to 40 times slower; child processes inherit this.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        apply_settings()
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
