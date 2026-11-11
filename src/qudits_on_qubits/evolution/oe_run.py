"""Run a bounded OpenEvolve search for encoded qudit states."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace

import numpy as np
from qiskit import qpy
import qiskit
import scipy

from .architectures import ARCHITECTURES, route
from .checkpoints import latest_checkpoint, load as load_resume, save, saved_run
from .encodings import canonical_encoding
from .progress import LIVE_LOGS, Progress
from .oe_candidates import MAX_BYTES, candidate_angles, candidate_from_circuit, load_skeleton, skeleton_program
from .oe_evaluator import (
    _STATES, ENCODING_MODES, MAX_SECONDS, OBJECTIVE_ERROR, OBJECTIVES, REPRESENTATIONS, WARM_STARTS, _fit_options,
    _limits, _warm_start, assess,
)
from .polish import polish_depth
from .skeleton_fit import cut_requirements, cut_seed, known_angles, skeleton_of
from .state_workloads import (
    AME44, AME46_STATES, AME64, GENERATED_BASELINES, describe, encoded_target, evaluate_workload, layout, party_encodings,
)

MAX_ITERATIONS = 1000
MAX_WORKERS = 16
# Models without --model-config. "agent" runs one Codex agent session per iteration (see oe_agent.py);
# it needs the network and uses the Codex quota of the account.
OFFLINE_MODELS = ("fixed", "mutations", "agent")
POLL_SECONDS = 1
OFFLINE_LABELS = {"fixed": "offline_mock", "mutations": "offline_mutations", "agent": "codex_agent"}
MAX_NOTES_BYTES = 4096
INVALID_BEST = "best_invalid.json"
# Before the hard time limit, a stop signal lets OpenEvolve end and return its best program.
STOP_GRACE_SECONDS = 120
MAX_POLISH_SECONDS = 3600
# An invalid seed fits until 0.7 x timeout_seconds; a valid seed with a warm start fits in seconds.
SEED_CHECK_SECONDS = 60
COUNT_METRICS = ("cz_count", "decoder_cz_count", "two_qubit_depth", "depth", "one_qubit_gate_count")
SEED_NOTE = ("The seed of this search reached only fidelity F={F:.4f} in its first fit, so the search starts"
             " from the known valid structure. Follow the idea with small steps from this structure.")


def read_prompt_notes(path):
    """Read user notes for the skeleton prompt as UTF-8 text of at most 4096 bytes."""
    if path is None:
        return ""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_NOTES_BYTES + 1)
    if len(raw) > MAX_NOTES_BYTES:
        raise ValueError(f"The prompt notes must be at most {MAX_NOTES_BYTES} bytes")
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("The prompt notes must be UTF-8 text") from exc
# adapted_direct: an AME(4,6) circuit in the local basis of its source (see ame46.py).
METHODS = ("estar_direct", "canonical_direct", "canonical_f3", "adapted_direct")


def method_for(state, method):
    """Return the preparation method that the baseline supports."""
    if state in GENERATED_BASELINES:
        return "canonical_direct"
    return "adapted_direct" if state in AME46_STATES or state in (AME44, AME64) else method
GATES_MESSAGE = (
    "Return one Python assignment named CANDIDATE. "
    "Use only literal data. Keep num_qubits fixed. "
    "Use operations u, u3 and cz. The evaluator checks the full state. "
    "A valid state must have error and leakage norms at most 1e-10. "
    "First preserve correctness. Then reduce the CZ count and circuit depth."
)
SKELETON_MESSAGE = (
    "You design CZ structures for exact preparation of qudit states on qubits. "
    "Return one Python assignment named CANDIDATE with literal data only: "
    "num_qubits and an ordered list cz of qubit pairs. Do not write angles."
)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _source(baseline, state, method):
    if state in GENERATED_BASELINES:
        from .state_benchmark import ensure_state_baseline
        ensure_state_baseline(state, baseline)
    manifest = json.loads((baseline / "manifest.json").read_bytes())
    summary = json.loads((baseline / "summary.json").read_bytes())
    rows = [r for r in summary["results"] if r["workload"] == state and r["method"] == method and r["is_valid"]]
    if len(rows) != 1:
        raise ValueError("The baseline must contain one valid source circuit")
    row = rows[0]
    files = []
    for field in ("circuit_artifact", "encoding_artifact"):
        name = row[field]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("The source path is not valid")
        raw = (baseline / name).read_bytes()
        if _hash(raw) != manifest["files"].get(name):
            raise ValueError("The source hash does not match")
        files.append(raw)
    circuits = qpy.load(BytesIO(files[0]))
    if len(circuits) != 1:
        raise ValueError("The source must contain one circuit")
    encoding = np.load(BytesIO(files[1]), allow_pickle=False)
    metrics = evaluate_workload(state, circuits[0], encoding, tolerance=1e-10)
    if not metrics["is_valid"]:
        raise ValueError("The source circuit does not pass the limit")
    return circuits[0], encoding, files, metrics


def _admit(snapshot, candidate, island):
    """Keep failed evaluations out of the population so they are never parents."""
    return not candidate.metrics.get("evaluation_failed")


def _settings(model_config, iterations, seed, workers=1, system_message=GATES_MESSAGE, template_dir=None,
              offline_model="fixed"):
    from openevolve.config import Config, LLMConfig, LLMModelConfig, load_config
    if model_config is None:
        from .oe_agent import AgentClient
        from .oe_mock import MockLLM, SkeletonMutator
        client = {"fixed": MockLLM, "mutations": SkeletonMutator, "agent": AgentClient}[offline_model]
        model = LLMModelConfig(name="qoq-" + OFFLINE_LABELS[offline_model].replace("_", "-"), weight=1.0, init_client=client)
        config = Config(llm=LLMConfig(models=[model], evaluator_models=[model]))
    else:
        config = load_config(str(model_config))
        if not config.llm.models:
            raise ValueError("The model configuration must contain a model")
        if system_message == SKELETON_MESSAGE:
            from .oe_strategy import STRATEGY_MESSAGE, StrategyClient

            for model in config.llm.models:
                model.init_client = partial(StrategyClient, client_factory=model.init_client)
            system_message = STRATEGY_MESSAGE
    # A global seed is sent with every LLM call; equal prompts then return equal answers.
    config.random_seed = None
    config.database.random_seed = seed
    config.max_iterations = iterations
    config.checkpoint_interval = 1
    config.language = "python"
    config.max_code_length = MAX_BYTES
    config.diff_based_evolution = False
    config.enforce_evolve_blocks = False
    config.evaluator.parallel_evaluations = workers
    config.evaluator.cascade_evaluation = False
    config.evaluator.use_llm_feedback = False
    config.database.num_islands = 1
    # Each prompt repeats every listed program, so keep one top and one diverse copy.
    config.prompt.num_top_programs = 1
    config.prompt.num_diverse_programs = 1
    config.evolution_trace.enabled = True
    config.evolution_trace.include_code = True
    config.database.feature_dimensions = ["cz_count", "two_qubit_depth"]
    config.prompt.system_message = system_message
    config.prompt.template_dir = template_dir
    return config


def _skeleton_template(state, encoding_mode, objective="preparation", notes="", known=(), architecture=None,
                       *, strategy=False, agent=False):
    """Return the user prompt template. It describes the problem for this state.

    known is the CZ list of a valid structure (the baseline), shown as a fallback for invalid candidates.
    architecture names the device of the routed score (see architectures.ARCHITECTURES), or None.
    """
    problem, units = describe(state, encoding_mode)
    cuts = "\n".join(f"  - {units} {', '.join(map(str, side))} versus the others: at least {need} CZ across the cut"
                      for side, need in cut_requirements(state))
    cost = ("CZ gates of the preparation plus the decoder to the canonical encoding (decoder_cz)"
            if objective == "total" else "CZ gates of the preparation")
    ranking = f"fewer {cost} win,\nthen a smaller two-qubit depth."
    goal = "with fewer CZ gates, or with the same\ncount and a smaller two-qubit depth"
    if architecture is not None:
        ranking = (f"fewer two-qubit gates after layout and routing on {ARCHITECTURES[architecture][1]} win, then a"
                   f" smaller routed two-qubit depth, then fewer {cost}. CZ pairs that match the coupling map of the"
                   " device need no SWAP gates; heavy-hex and square lattices have no triangles.")
    if objective == "cost":
        goal = "with a larger routed success estimate (for example\nfewer SWAPs or fewer routed two-qubit gates)"
        ranking = ("a larger routed_success_estimate after layout and routing on"
                   f" {ARCHITECTURES[architecture][1]} wins (the product of 1 - calibrated gate error over all"
                   " native one- and two-qubit gates, SWAPs included), then fewer routed two-qubit gates, then a"
                   " smaller routed two-qubit depth. More CZ gates can win if they need fewer SWAP gates or use"
                   " better qubit pairs. CZ pairs that match the coupling map of the device need no SWAP gates;"
                   " heavy-hex and square lattices have no triangles.")
    reference = (f"A known valid structure with {len(known)} CZ: {json.dumps([list(p) for p in known])}\n"
                 if known else "")
    text = f"""# Current candidate
- Fitness: {{fitness_score}} (valid candidates score above 1; a better rank (see Problem) scores higher;
  an invalid candidate scores F/2)
- Metrics: {{metrics}}
- Focus areas: {{improvement_areas}}

{{artifacts}}

# Evolution history
{{evolution_history}}

# Current candidate
```{{language}}
{{current_program}}
```

# Problem
{problem}
The candidate lists only the CZ gates in time order. The evaluator puts a general one-qubit gate
on every qubit at the start and on both qubits after each CZ. Then it fits all angles numerically.
The first fit uses supplied angles, or the angles of the most similar fitted structure (the longest
common CZ subsequence). Invalid fits also supply angles. The other restarts start at random.
A candidate that keeps most CZs of a valid
structure in the same order is easier to fit. A candidate is valid if the fitted error and leakage
norms are at most 1e-10. A valid candidate always beats an invalid one. Among valid candidates, {ranking}
Necessary conditions from the Schmidt rank (each CZ across a cut at most doubles it):
{cuts}
{reference}
# Task
Make small steps: change 1 to 3 CZ gates per candidate and keep the other CZs in the same order.
The fit of a small change starts near the angles of a valid structure. The fit of a large change
starts far from them and often does not reach the state.
If the current candidate is VALID, propose a CZ structure {goal}. Useful moves: remove a CZ and move another one, change
which qubit of a block a CZ uses, reorder CZs, put CZs that act on different qubits in the same layer,
or use a CZ inside one block.
If the current candidate is INVALID, make it valid first: a valid candidate with more CZ gates beats
every invalid one. Add CZ gates, or go back to a valid structure and make a smaller change.
A higher fidelity F means that the fit came nearer to the state.
An identical candidate is wasted. Return the complete candidate in one ```{{language}} block.
"""
    if strategy:
        from .oe_strategy import STRATEGY_TASK

        text = text.split("# Task\n", 1)[0] + "# Task\n" + STRATEGY_TASK.replace("{", "{{").replace("}", "}}")
    if agent:
        text = text.split("# Task\n", 1)[0] + (
            "# Task\nFirst construct a valid circuit from the target vector. Then reduce its cost.\n"
            "You can replace the full construction algorithm. Keep and improve builder.py.\n"
            "Use the session tools to test the circuit and retain fitted angles.\n"
            "The session instructions define the output files.\n")
    if notes:
        # The template is formatted later, so literal braces in the notes must be doubled.
        text += "\n# Notes from earlier searches\n" + notes.replace("{", "{{").replace("}", "}}") + "\n"
    return text


def run_campaign(baseline: Path, output: Path, *, state="ghz3", method="estar_direct",
                 iterations=1, seed=0, max_nfev=0, max_parameters=12, timeout_seconds=30,
                 memory_mb=2048, cpu_seconds=30, model_config=None, campaign_timeout_seconds=900,
                 codex_auth_file=None, representation="gates", encoding_mode="fixed", restarts=8,
                 workers=1, offline_model="fixed", objective="preparation", prompt_notes=None,
                 initial_program=None, narrator_model_config=None, warm_starts=(), min_seed_fidelity=0.0,
                 architecture=None, blank_start=False, polish_seconds=0, resume=False):
    """Run the campaign in a process group with a wall-time limit.

    The worker writes its full output to campaign.log. This process prints short progress lines, and an
    optional narrator model (narrator_model_config) adds short summaries. Shortly before the limit, the
    worker gets SIGTERM: OpenEvolve then stops and the worker saves the best program. At the limit, the
    process group gets SIGKILL.
    polish_seconds > 0 lowers the two-qubit depth of the seed and of the best skeleton with exact CZ
    swaps (polish.py), for at most that time each.
    """
    parameters = locals().copy()
    started = monotonic()
    if (isinstance(campaign_timeout_seconds, bool)
            or not isinstance(campaign_timeout_seconds, (int, float))
            or not 0 < campaign_timeout_seconds <= 1_000_000_000
            or not math.isfinite(campaign_timeout_seconds)):
        raise ValueError("The campaign time limit must be between 0 and 1000000000 seconds")
    if (isinstance(polish_seconds, bool) or not isinstance(polish_seconds, (int, float))
            or not 0 <= polish_seconds <= MAX_POLISH_SECONDS or not math.isfinite(polish_seconds)):
        raise ValueError(f"The polish time limit must be from 0 to {MAX_POLISH_SECONDS} seconds")
    output = Path(output).resolve()
    previous_elapsed = load_resume(output, "campaign")["elapsed_seconds"] if resume else 0
    if resume and (output / "manifest.json").is_file() and (output / "summary.json").is_file():
        with saved_run(output, "campaign", parameters, resume=True):
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["files"].items():
                member = (output / name).resolve()
                if not member.is_relative_to(output) or _hash(member.read_bytes()) != digest:
                    raise ValueError("The saved campaign file hash does not match")
            return json.loads((output / "summary.json").read_text())
    time_left = campaign_timeout_seconds - previous_elapsed
    if time_left <= 0:
        raise TimeoutError("The campaign exceeded its time limit")
    args = {"baseline": str(Path(baseline).resolve()), "output": str(output),
            "state": state, "method": method, "iterations": iterations, "seed": seed,
            "max_nfev": max_nfev, "max_parameters": max_parameters,
            "timeout_seconds": timeout_seconds, "memory_mb": memory_mb,
            "cpu_seconds": cpu_seconds,
            "model_config": str(Path(model_config).resolve()) if model_config is not None else None,
            "campaign_timeout_seconds": campaign_timeout_seconds,
            "codex_auth_file": str(Path(codex_auth_file).resolve()) if codex_auth_file is not None else None,
            "representation": representation, "encoding_mode": encoding_mode, "restarts": restarts,
            "workers": workers, "offline_model": offline_model, "objective": objective,
            "prompt_notes": str(Path(prompt_notes).resolve()) if prompt_notes is not None else None,
            "initial_program": str(Path(initial_program).resolve()) if initial_program is not None else None,
            "warm_starts": [str(Path(d).resolve()) for d in warm_starts], "console_log": True,
            "min_seed_fidelity": min_seed_fidelity, "architecture": architecture, "blank_start": blank_start,
            "polish_seconds": polish_seconds, "resume": resume,
            "narrator_model_config": str(Path(narrator_model_config).resolve()) if narrator_model_config else None}
    # An output directory that exists before the start is not ours, so its old logs are not read.
    progress = None if output.exists() and not resume else Progress(output.name, output, iterations, narrator_model_config)
    worker = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from qudits_on_qubits.evolution.oe_run import _run_campaign\n"
        "args = json.load(sys.stdin)\n"
        "try:\n"
        "    result = {'result': _run_campaign(**args)}\n"
        "except KeyboardInterrupt:\n"
        "    result = {'paused': True}\n"
        "except Exception as exc:\n"
        "    result = {'error': type(exc).__name__, 'message': str(exc)}\n"
        "Path(sys.argv[1]).write_text(json.dumps(result, allow_nan=False))\n"
    )
    stop_at = time_left - min(STOP_GRACE_SECONDS, time_left / 10)
    stopping = False
    with tempfile.TemporaryDirectory(prefix="qoq-campaign-") as directory:
        result_path = Path(directory) / "result.json"
        process = subprocess.Popen([sys.executable, "-c", worker, str(result_path), str(Path(__file__).resolve().parents[2])],
                                   stdin=subprocess.PIPE, start_new_session=True)
        try:
            try:
                process.stdin.write(json.dumps(args).encode())
                process.stdin.close()
            except BrokenPipeError:
                pass
            while process.poll() is None:
                elapsed = monotonic() - started
                if elapsed >= time_left:
                    raise TimeoutError("The campaign exceeded its time limit")
                # best_program.py shows that OpenEvolve has ended; a signal then could stop the save.
                if elapsed >= stop_at and not stopping and not (output / "best_program.py").exists():
                    # Only the worker gets the signal: OpenEvolve stops its own pool and returns the best.
                    process.send_signal(signal.SIGTERM)
                    stopping = True
                    print(f"{output.name}: the time limit is near; the search stops and saves its best"
                          " candidate", flush=True)
                if progress is not None:
                    progress.poll()
                sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Pause requested. Wait until the harness saves the state before you stop the computer.", flush=True)
            if output.is_dir():
                (output / "pause.request").touch()
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=STOP_GRACE_SECONDS)
            except (ProcessLookupError, subprocess.TimeoutExpired, KeyboardInterrupt):
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if (output / "resume.json").is_file():
                checkpoint = load_resume(output, "campaign")
                checkpoint.update(status="paused", elapsed_seconds=previous_elapsed + monotonic() - started)
                save(output / "resume.json", checkpoint)
                print(f"Paused. Resume with: qoq-evolution resume {shlex.quote(str(output))}", flush=True)
            raise
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.returncode != 0 or not result_path.is_file():
            if stopping:
                # The stop signal came before OpenEvolve could catch it (for example during the diagnostics).
                raise TimeoutError("The campaign exceeded its time limit")
            raise RuntimeError("The campaign process failed")
        data = json.loads(result_path.read_text())
        if data.get("paused"):
            raise KeyboardInterrupt
        if "error" in data:
            errors = {"ValueError": ValueError, "FileExistsError": FileExistsError,
                      "FileNotFoundError": FileNotFoundError, "TimeoutError": TimeoutError,
                      "RuntimeError": RuntimeError}
            raise errors.get(data["error"], RuntimeError)(data.get("message", "The campaign process failed"))
    if progress is not None:
        progress.finish(data["result"]["saved"])
    return data["result"]


def _redirect_console(path):
    """Send stdout and stderr of this worker and its children to the campaign log."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)


def _file_hashes(output):
    """Hash the saved files. The live logs still grow after the manifest, so they are not hashed."""
    return {str(p.relative_to(output)): _hash(p.read_bytes()) for p in output.rglob("*")
            if p.is_file() and p.name not in (*LIVE_LOGS, "resume.json", ".run.lock", "pause.request")}


def _print_codex_quota(auth_file, label):
    from .codex_quota import read_codex_quota

    quota = read_codex_quota(auth_file)
    if quota is None:
        print(f"Codex quota ({label}): unavailable", flush=True)
        return
    fields = ("used_percent", "remaining_percent", "limit_window_seconds", "reset_at", "reset_after_seconds")
    for name in ("primary_window", "secondary_window"):
        window = quota.get(name)
        if window is not None:
            values = {field: window.get(field) for field in fields}
            print(f"Codex quota ({label}) {name}: {json.dumps(values)}", flush=True)


def _warm_pool(state, encoding_mode, circuit, encoding, directories=()):
    """Return the warm-start items of the baseline circuit and of the best circuits in the directories.

    Free encodings need tail angles, which a short fit finds for each party; an item without them is left out.
    circuit=None leaves out the baseline (a blank start).
    """
    sources = [] if circuit is None else [(circuit, party_encodings(state, encoding))]
    for directory in map(Path, directories):
        try:
            with (directory / "best.qpy").open("rb") as stream:
                best, = qpy.load(stream)
            encodings = np.load(directory / "best_encodings.npy", allow_pickle=False)
        except (OSError, ValueError, qpy.QpyError):
            continue
        sources.append((best, encodings))
    pool = []
    for source, encodings in sources:
        found = known_angles(source, encodings if encoding_mode == "free" else None)
        if found is not None:
            pool.append({"pairs": [list(p) for p in found[0]], "angles": [float(v) for v in found[1]]})
    return pool


def _skeleton_programs(width, pairs):
    """Return the seed program and two diagnostic children: without its first CZ and without its second CZ."""
    return (skeleton_program(width, pairs), skeleton_program(width, pairs[1:]),
            skeleton_program(width, pairs[:1] + pairs[2:]))


def _polish(output, label, options, deadline):
    """Lower the cost of a valid skeleton program with exact CZ swaps (polish.py, no model).

    Without an architecture the cost is the two-qubit depth, with one the costs after routing.
    A better valid circuit replaces the program, with its angles. Return the record of the change, or None.
    """
    path, width = output / f"{label}.py", _STATES[options["state"]]
    try:
        found = polish_depth(options["state"], load_skeleton(path, expected_qubits=width),
                             _warm_start(path, options, output),
                             encoding=None if options["encoding_mode"] == "free" else options["encoding"],
                             objective=options["objective"], deadline=deadline,
                             architecture=options["architecture"])
    except ValueError as exc:
        print(f"Polish of {label}: skipped ({exc})", flush=True)
        return None
    if found is None:
        return None
    from .oe_agent import builder_source, with_builder
    path.write_text(with_builder(skeleton_program(width, found["pairs"], found["angles"]),
                                 builder_source(path.read_text(encoding="utf-8"))), encoding="utf-8")
    record = {"program": f"{label}.py", "two_qubit_depth_before": found["two_qubit_depth_before"],
              "two_qubit_depth_after": int(found["metrics"]["two_qubit_depth"])}
    change = f"two-qubit depth {record['two_qubit_depth_before']} -> {record['two_qubit_depth_after']}"
    if "routed_two_qubit_count_before" in found:
        record.update(routed_two_qubit_count_before=found["routed_two_qubit_count_before"],
                      routed_two_qubit_count_after=int(found["metrics"]["routed_two_qubit_count"]))
        change += (f", routed two-qubit gates {record['routed_two_qubit_count_before']}"
                   f" -> {record['routed_two_qubit_count_after']}")
    print(f"Polish of {label}: {change} with exact CZ swaps", flush=True)
    return record


def _diagnose(output, label, options):
    """Fit one program of the campaign directory and return its record for diagnostics.json."""
    try:
        x0 = _warm_start(output / f"{label}.py", options, output)
        result = assess(output / f"{label}.py", **{**options, "max_nfev": 0}, x0=x0)
        if options["representation"] == "skeleton" and getattr(result, "angles", None) is not None:
            from .oe_agent import builder_source, with_builder
            path = output / f"{label}.py"
            path.write_text(with_builder(skeleton_program(_STATES[options["state"]],
                load_skeleton(path, expected_qubits=_STATES[options["state"]]), result.angles),
                builder_source(path.read_text(encoding="utf-8"))), encoding="utf-8")
        return {"program": f"{label}.py", "metrics": result.metrics}
    except (ValueError, TimeoutError) as exc:
        return {"program": f"{label}.py", "error": f"{type(exc).__name__}: {exc}"}


def check_options(*, state, method, iterations, seed, workers, max_nfev, max_parameters, representation,
                  encoding_mode, restarts, objective, prompt_notes, initial_program, timeout_seconds,
                  memory_mb, cpu_seconds, offline_model, model_config, architecture=None, blank_start=False):
    """Check the campaign options without side effects and return the prompt notes."""
    if state not in _STATES or method not in METHODS:
        raise ValueError("The state or preparation method is not supported")
    if architecture is not None and architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"The iteration count must be from 1 to {MAX_ITERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The seed must be a nonnegative integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"The worker count must be from 1 to {MAX_WORKERS}")
    _fit_options(state, max_nfev, max_parameters, representation, encoding_mode, restarts, seed)
    if objective not in OBJECTIVES:
        raise ValueError(OBJECTIVE_ERROR)
    if objective == "cost" and architecture is None:
        raise ValueError("The cost objective needs an architecture (the error data of a processor)")
    if prompt_notes is not None and representation != "skeleton":
        raise ValueError("Prompt notes need the skeleton representation")
    notes = read_prompt_notes(prompt_notes)
    if initial_program is not None and representation != "skeleton":
        raise ValueError("An initial program needs the skeleton representation")
    _limits(timeout_seconds, memory_mb, cpu_seconds)
    if offline_model not in OFFLINE_MODELS or (offline_model != "fixed" and representation != "skeleton"):
        raise ValueError("Random mutations and the agent need the skeleton representation")
    if model_config is None and offline_model == "fixed" and iterations != 1:
        raise ValueError("The offline test uses one iteration")
    if model_config is None and offline_model == "agent":
        from .oe_agent import agent_settings
        agent_settings()
    if blank_start and representation != "skeleton":
        raise ValueError("A blank start needs the skeleton representation")
    return notes


def _run_campaign(baseline: Path, output: Path, *, state="ghz3", method="estar_direct",
                  iterations=1, seed=0, max_nfev=0, max_parameters=12, timeout_seconds=30,
                  memory_mb=2048, cpu_seconds=30, model_config=None, campaign_timeout_seconds=900,
                  codex_auth_file=None, representation="gates", encoding_mode="fixed", restarts=8,
                  workers=1, offline_model="fixed", objective="preparation", prompt_notes=None,
                  initial_program=None, console_log=False, warm_starts=(), min_seed_fidelity=0.0,
                  architecture=None, blank_start=False, polish_seconds=0, resume=False,
                  narrator_model_config=None):
    """Run OpenEvolve and save the checked circuit.

    polish_seconds > 0 lowers the two-qubit depth of a valid seed (before OpenEvolve) and of the valid
    best skeleton (after it) with exact CZ swaps; see polish.py. Each polish stops before the time
    limit of the campaign.

    blank_start excludes the baseline structure and angles. The seed comes from initial_program,
    or from cut_seed when no initial program is supplied. The baseline sets only the encoding and comparison.

    architecture (see architectures.ARCHITECTURES) ranks valid candidates by their costs after layout
    and routing on that device, and saves the routed best circuit as best_routed.qpy.

    initial_program is an optional skeleton file. It replaces the baseline CZ structure as the start point.
    warm_starts are earlier campaign directories. Their best circuits, with the baseline, give the start
    angles of the skeleton fits.
    If the first fit of initial_program gives a fidelity below min_seed_fidelity, the search starts from
    the baseline CZ structure instead. The default 0 keeps every seed.
    """
    parameters = locals().copy()
    # The parent stops the campaign at its time limit; the final check needs the grace time.
    polish_end = perf_counter() + campaign_timeout_seconds - STOP_GRACE_SECONDS
    method = method_for(state, method)
    notes = check_options(state=state, method=method, iterations=iterations, seed=seed, workers=workers,
                          max_nfev=max_nfev, max_parameters=max_parameters, representation=representation,
                          encoding_mode=encoding_mode, restarts=restarts, objective=objective,
                          prompt_notes=prompt_notes, initial_program=initial_program,
                          timeout_seconds=timeout_seconds, memory_mb=memory_mb, cpu_seconds=cpu_seconds,
                          offline_model=offline_model, model_config=model_config, architecture=architecture,
                          blank_start=blank_start)
    if (isinstance(min_seed_fidelity, bool) or not isinstance(min_seed_fidelity, (int, float))
            or not 0 <= min_seed_fidelity <= 1):
        raise ValueError("The minimum seed fidelity must be from 0 to 1")
    from importlib.metadata import version as package_version
    if package_version("openevolve") != "0.4.0":
        raise ValueError("OpenEvolve version 0.4.0 is required")
    from openevolve.api import run_evolution
    if codex_auth_file is not None:
        _print_codex_quota(codex_auth_file, "before")

    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    model_config = Path(model_config).resolve() if model_config is not None else None
    with saved_run(output, "campaign", parameters, resume):
        if console_log:
            _redirect_console(output / "campaign.log")
        circuit, encoding, source, source_metrics = _source(baseline, state, method)
        if representation == "skeleton":
            pairs = (load_skeleton(initial_program, expected_qubits=circuit.num_qubits) if initial_program is not None
                     else cut_seed(state) if blank_start else skeleton_of(circuit))
            if len(pairs) < 2:
                raise ValueError("The source must contain at least two CZ gates")
            seed_program, child_program, rejected_program = _skeleton_programs(circuit.num_qubits, pairs)
            if initial_program is not None:
                from .oe_agent import builder_source, with_builder
                seed_program = with_builder(skeleton_program(circuit.num_qubits, pairs, candidate_angles(initial_program)),
                                            builder_source(Path(initial_program).read_text(encoding="utf-8")))
        else:
            candidate = candidate_from_circuit(circuit)
            cz_indices = [i for i, item in enumerate(candidate["operations"]) if item[0] == "cz"]
            if len(cz_indices) < 2:
                raise ValueError("The source must contain at least two CZ gates")
            seed_program = "CANDIDATE = " + repr(candidate) + "\n"
            child = dict(candidate)
            child["operations"] = [op for i, op in enumerate(candidate["operations"]) if i != cz_indices[0]]
            child_program = "CANDIDATE = " + repr(child) + "\n"
            rejected = dict(candidate)
            rejected["operations"] = [op for i, op in enumerate(candidate["operations"]) if i != cz_indices[1]]
            rejected_program = "CANDIDATE = " + repr(rejected) + "\n"
        if len(seed_program.encode()) > MAX_BYTES:
            raise ValueError("The source program is too large")
        template_dir = output / "prompts" if representation == "skeleton" else None
        local_strategy = model_config is not None and representation == "skeleton"
        config = _settings(model_config, iterations, seed, workers,
                           SKELETON_MESSAGE if representation == "skeleton" else GATES_MESSAGE,
                           str(template_dir) if template_dir else None, offline_model)
        config.evaluator.timeout = int(timeout_seconds) + 10
        config.database.db_path = str(output / "population")
        settings = {"state": state, "encoding_path": str(output / "encoding.npy"),
                    "max_nfev": max_nfev, "max_parameters": max_parameters,
                    "representation": representation, "encoding_mode": encoding_mode,
                    "restarts": restarts, "fit_seed": seed, "objective": objective,
                    "timeout_seconds": timeout_seconds, "memory_mb": memory_mb, "cpu_seconds": cpu_seconds,
                    "architecture": architecture}
        options = {key: value for key, value in settings.items() if key != "encoding_path"}
        options.update(encoding=encoding)
        if not resume or not (output / "configuration.json").is_file():
            (output / "encoding.npy").write_bytes(source[1])
            target, _ = encoded_target(state, canonical_encoding(layout(state)[1]) if encoding_mode == "free" else encoding)
            np.save(output / "target.npy", target, allow_pickle=False)
            if not blank_start:
                (output / "source.qpy").write_bytes(source[0])
            (output / "initial_program.py").write_text(seed_program, encoding="utf-8")
            (output / "evaluator.py").write_text(
                "from qudits_on_qubits.evolution.oe_evaluator import evaluate\n", encoding="utf-8")
            save(output / "evaluation.json", settings)
            if representation == "skeleton":
                (output / WARM_STARTS).write_bytes(_json(_warm_pool(state, encoding_mode, None if blank_start else circuit,
                                                                    encoding, warm_starts)))
            seed_options = options
            if initial_program is not None and (min_seed_fidelity or blank_start):
                seed_options = {**options, "timeout_seconds": min(timeout_seconds, SEED_CHECK_SECONDS),
                                "cpu_seconds": min(cpu_seconds, SEED_CHECK_SECONDS)}
            diagnostics = {"initial_program": _diagnose(output, "initial_program", seed_options)}
            template_notes = notes
            if min_seed_fidelity and initial_program is not None and not blank_start:
                # A seed far from every valid structure gets a poor warm start, and its fits do not reach the state.
                fidelity = float(diagnostics["initial_program"].get("metrics", {}).get("F", 0.0))
                diagnostics["seed_check"] = {"F": fidelity, "min_F": min_seed_fidelity,
                                             "replaced": fidelity < min_seed_fidelity}
                if fidelity < min_seed_fidelity:
                    pairs = skeleton_of(circuit)
                    seed_program, child_program, rejected_program = _skeleton_programs(circuit.num_qubits, pairs)
                    (output / "initial_program.py").write_text(seed_program, encoding="utf-8")
                    diagnostics["rejected_seed"] = diagnostics["initial_program"]
                    diagnostics["initial_program"] = _diagnose(output, "initial_program", options)
                    template_notes = "\n".join(filter(None, (notes, SEED_NOTE.format(F=fidelity))))
                    print(f"Seed check: the seed reached F={fidelity:.4f} (minimum {min_seed_fidelity}); the search"
                          f" starts from the known valid structure with {len(pairs)} CZ", flush=True)
            if polish_seconds and representation == "skeleton" and diagnostics["initial_program"].get("metrics", {}).get("is_valid"):
                polished = _polish(output, "initial_program", options, min(perf_counter() + polish_seconds, polish_end))
                if polished:
                    diagnostics["polish"] = polished
            (output / "mock_program.py").write_text(child_program, encoding="utf-8")
            (output / "diagnostic_program.py").write_text(rejected_program, encoding="utf-8")
            for label in ("mock_program", "diagnostic_program"):
                # These two fits only describe the children of the offline mock. A model or an agent writes its own.
                diagnostics[label] = (_diagnose(output, label, options) if model_config is None and offline_model != "agent"
                                      else {"program": f"{label}.py", "skipped": "a model writes the children"})
            (output / "diagnostics.json").write_bytes(_json(diagnostics))
            if template_dir:
                template_dir.mkdir(exist_ok=resume)
                (template_dir / "full_rewrite_user.txt").write_text(
                    _skeleton_template(state, encoding_mode, objective, template_notes,
                                       () if blank_start else skeleton_of(circuit), architecture, strategy=local_strategy,
                                       agent=model_config is None and offline_model == "agent"),
                    encoding="utf-8")
            root = Path(__file__).resolve().parents[3]
            code_paths = [*sorted((root / "src/qudits_on_qubits/evolution").glob("*.py")),
                          root / "src/qudits_on_qubits/reference_experiments.py",
                          root / "src/qudits_on_qubits/benchmarks/validation.py"]
            try:
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                        capture_output=True, text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                commit = json.loads((baseline / "manifest.json").read_bytes()).get("versions", {}).get("git_commit")
            agent = None
            if model_config is None and offline_model == "agent":
                from .oe_agent import agent_settings
                agent = agent_settings()
            run_config = {"state": state, "method": method, "iterations": iterations, "seed": seed,
                          "model_mode": OFFLINE_LABELS[offline_model] if model_config is None else "configured_model",
                          "model_config_sha256": _hash(model_config.read_bytes()) if model_config else None,
                          "max_nfev": max_nfev, "max_parameters": max_parameters,
                          "representation": representation, "encoding_mode": encoding_mode,
                          "restarts": restarts, "workers": workers, "objective": objective,
                          "architecture": architecture,
                          "architecture_description": ARCHITECTURES[architecture][1] if architecture else None,
                          "prompt_notes_sha256": _hash(notes.encode()) if notes else None,
                          "initial_program_sha256": _hash(seed_program.encode()) if initial_program else None,
                          "timeout_seconds": timeout_seconds, "memory_mb": memory_mb, "cpu_seconds": cpu_seconds,
                          "campaign_timeout_seconds": campaign_timeout_seconds, "min_seed_fidelity": min_seed_fidelity,
                          "blank_start": blank_start, "agent": agent, "polish_seconds": polish_seconds,
                          "local_strategy": local_strategy,
                          "codex_quota_enabled": codex_auth_file is not None,
                          "openevolve": "0.4.0", "encoding_sha256": _hash(source[1]),
                          "source_sha256": _hash(source[0]), "baseline": str(baseline),
                          "baseline_manifest_sha256": _hash((baseline / "manifest.json").read_bytes()),
                          "numpy": np.__version__, "scipy": scipy.__version__, "qiskit": qiskit.__version__,
                          "python": platform.python_version(), "git_commit": commit,
                          "code_sha256": {str(path.relative_to(root)): _hash(path.read_bytes()) for path in code_paths}}
            save(output / "configuration.json", run_config, sort_keys=True)
        else:
            run_config = json.loads((output / "configuration.json").read_text())
            if run_config["source_sha256"] != _hash(source[0]) or run_config["encoding_sha256"] != _hash(source[1]):
                raise ValueError("The baseline changed after the campaign started")
            if model_config is not None and run_config["model_config_sha256"] != _hash(model_config.read_bytes()):
                raise ValueError("The model configuration changed after the campaign started")
            agent = run_config["agent"]
        old_env = {key: os.environ.get(key) for key in ("QOQ_OE_CONFIG", "QOQ_OE_MOCK_PROGRAM")}
        os.environ["QOQ_OE_CONFIG"] = str(output / "evaluation.json")
        os.environ["QOQ_OE_MOCK_PROGRAM"] = str(output / "mock_program.py")
        try:
            from openevolve.population import PopulationStrategy
            result_path = output / "search_result.json"
            if resume and result_path.is_file():
                result = SimpleNamespace(**json.loads(result_path.read_text()))
            else:
                checkpoint_path, completed = latest_checkpoint(output) if resume else (None, 0)
                if completed >= iterations:
                    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
                    best = json.loads((checkpoint_path / "programs" / f"{metadata['best_program_id']}.json").read_text())
                    result = SimpleNamespace(best_code=best["code"], metrics=best["metrics"])
                else:
                    result = run_evolution(output / "initial_program.py", output / "evaluator.py",
                                           config=config, iterations=iterations - completed,
                                           output_dir=str(output / "openevolve"), cleanup=False,
                                           checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                                           population_strategy=PopulationStrategy(admit=_admit))
                if (output / "pause.request").exists():
                    raise KeyboardInterrupt
                save(result_path, {"best_code": result.best_code, "metrics": result.metrics})
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if agent is not None:
                from .oe_agent import agent_summary
                (output / "agent_usage.json").write_bytes(_json(agent_summary(output)))
        if not result.best_code:
            raise RuntimeError("OpenEvolve did not return a candidate")
        (output / "summary.json").unlink(missing_ok=True)
        (output / "manifest.json").unlink(missing_ok=True)
        (output / "best_program.py").write_text(result.best_code, encoding="utf-8")
        if representation == "skeleton":
            from .oe_agent import builder_source, with_builder
            best_path = output / "best_program.py"
            angles = _warm_start(best_path, options, output, exact=True)
            if angles is not None:
                best_path.write_text(with_builder(skeleton_program(circuit.num_qubits,
                    load_skeleton(best_path, expected_qubits=circuit.num_qubits), angles),
                    builder_source(result.best_code)), encoding="utf-8")
            builder = builder_source(result.best_code)
            if builder is not None:
                (output / "best_builder.py").write_text(builder, encoding="utf-8")
        if not result.metrics.get("is_valid"):
            # The metrics of an invalid best are known. A new fit could use all the time that is left after a stop.
            invalid = {**result.metrics, "is_valid": False,
                       **{key: int(result.metrics[key]) for key in COUNT_METRICS if key in result.metrics}}
            (output / INVALID_BEST).write_bytes(_json(invalid))
            raise RuntimeError("The best candidate did not pass the state check")
        polished = (_polish(output, "best_program", options, min(perf_counter() + polish_seconds, polish_end))
                    if polish_seconds and representation == "skeleton" else None)
        # The cache entry of a valid best skeleton holds its angles, so the warm start finds the same fit again.
        best = assess(output / "best_program.py", **{**options, "timeout_seconds": MAX_SECONDS,
                                                      "cpu_seconds": MAX_SECONDS},
                      x0=_warm_start(output / "best_program.py", options, output))
        metrics = best.metrics
        if not metrics["is_valid"]:
            # Discovery reads this file to start the next round from the best invalid structure.
            (output / INVALID_BEST).write_bytes(_json(metrics))
            raise RuntimeError("The best candidate did not pass the state check")
        if representation == "skeleton" and best.angles is not None:
            best_path.write_text(with_builder(skeleton_program(circuit.num_qubits,
                load_skeleton(best_path, expected_qubits=circuit.num_qubits), best.angles),
                builder_source(best_path.read_text(encoding="utf-8"))), encoding="utf-8")
        with (output / "best.qpy").open("wb" if resume else "xb") as stream:
            qpy.dump(best.circuit, stream)
        with (output / "best_encodings.npy").open("wb" if resume else "xb") as stream:
            np.save(stream, best.encodings, allow_pickle=False)
        if best.decoder is not None:
            with (output / "decoder.qpy").open("wb" if resume else "xb") as stream:
                qpy.dump(best.decoder, stream)
        with (output / "best.qpy").open("rb") as stream:
            saved, = qpy.load(stream)
        saved_encodings = np.load(output / "best_encodings.npy", allow_pickle=False)
        saved_metrics = evaluate_workload(state, saved, saved_encodings, tolerance=1e-10)
        if best.decoder is not None:
            with (output / "decoder.qpy").open("rb") as stream:
                decoder, = qpy.load(stream)
            decoded = evaluate_workload(state, saved.compose(decoder), canonical_encoding(layout(state)[1]),
                                        tolerance=1e-10)
            saved_metrics["is_valid"] = saved_metrics["is_valid"] and decoded["is_valid"]
            saved_metrics["decoder_cz_count"] = int(decoder.count_ops().get("cz", 0))
        if not saved_metrics["is_valid"]:
            raise RuntimeError("The saved circuit did not pass the state check")
        if architecture is not None:
            with (output / "best_routed.qpy").open("wb" if resume else "xb") as stream:
                qpy.dump(route(saved, architecture, metrics["routed_seed"]), stream)
        files = _file_hashes(output)
        summary = {"source": source_metrics, "openevolve_metrics": result.metrics,
                   "best": metrics, "saved": saved_metrics, "best_qpy": "best.qpy",
                   "best_routed_qpy": "best_routed.qpy" if architecture else None, "polish": polished,
                   "configuration_sha256": _hash(_json(run_config)), "files": files}
        if local_strategy:
            from .oe_strategy import strategy_summary

            summary["local_search"] = strategy_summary(output)
        if agent is not None:
            from .oe_agent import agent_summary
            summary["agent_search"] = agent_summary(output)
        save(output / "summary.json", summary)
        command = ("PYTHONPATH=src " + shlex.quote(sys.executable) + " -m qudits_on_qubits.evolution.oe_run"
                   + " --baseline " + shlex.quote(str(baseline)) + " --output NEW_OUTPUT"
                   + " --state " + state + " --method " + method + f" --iterations {iterations} --seed {seed}"
                   + f" --max-nfev {max_nfev} --max-parameters {max_parameters}"
                   + f" --timeout-seconds {timeout_seconds} --memory-mb {memory_mb} --cpu-seconds {cpu_seconds}"
                   + f" --representation {representation} --encoding-mode {encoding_mode}"
                   + f" --restarts {restarts} --workers {workers} --offline-model {offline_model}"
                   + f" --objective {objective}")
        command += f" --campaign-timeout-seconds {campaign_timeout_seconds}"
        if architecture is not None:
            command += f" --architecture {architecture}"
        if initial_program is not None:
            command += " --initial-program " + shlex.quote(str(output / "initial_program.py"))
        if blank_start:
            command += " --blank-start"
        if polish_seconds:
            command += f" --polish-seconds {polish_seconds}"
        if model_config:
            command += " --model-config " + shlex.quote(str(model_config))
        if codex_auth_file is not None:
            command += ' --codex-auth-file "$QOQ_CODEX_AUTH_FILE"'
        (output / "README.md").write_text(
            "# OpenEvolve state search\n\n"
            "Install the project with the `evolution` extra before the run.\n"
            "The extra installs OpenEvolve 0.4.0.\n\n"
            "```sh\npython -m pip install -e '.[evolution]'\n```\n\n"
            "The input circuit prepares the encoded state from physical zero.\n"
            "The candidate file contains literal circuit data. The evaluator does not run its code.\n"
            "The best circuit passes the state and leakage limits of 1e-10.\n"
            "best_encodings.npy holds one encoding per party (shape n x 2**q x d).\n"
            "decoder.qpy maps each encoding to the canonical one (the readout cost).\n"
            + ("The configured model selects a strategy for at most 16 local proposals per worker and model.\n"
               "strategy_requests/ stores model requests and token usage.\n"
               "strategy_candidates/ stores the generated CZ lists and their strategy identifiers.\n"
               "summary.json reports model calls and generated candidates separately in local_search.\n"
               "Iterations count candidates, not model calls. The model can use a paid API.\n" if local_strategy else
               "A configured model made the proposals. It can use a paid API.\n" if model_config else
               "Codex agent sessions made the proposals (agent_sessions/, discoveries.md). They use the network"
               " and the Codex quota; the QOQ_AGENT_* variables set them.\n" if offline_model == "agent" else
               "The offline mutation model makes random changes without a network request. It is not an"
               " LLM campaign and it is not deterministic.\n" if offline_model == "mutations" else
               "The mock is a deterministic test without a network request. It is not an LLM campaign.\n")
            + ("The score uses the costs after layout and routing on " + ARCHITECTURES[architecture][1] + ".\n"
               "best.qpy is the logical circuit; best_routed.qpy is the same circuit routed to that device"
               " (no decoding or measurement).\n" if architecture else
               "The circuit excludes decoding, measurement and hardware routing.\n")
            + "campaign.log holds the full console output; progress.log holds the short progress lines.\n"
            "Run this command from the repository root. Select a new output path.\n\n"
            f"```sh\n{command}\n```\n", encoding="utf-8")
        files = _file_hashes(output)
        save(output / "manifest.json", {"configuration": run_config,
                                        "configuration_sha256": _hash(_json(run_config)),
                                        "files": files})
        if codex_auth_file is not None:
            _print_codex_quota(codex_auth_file, "after")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run a bounded state search.")
    parser.add_argument("--baseline", type=Path, required=True, help="Set the state baseline directory.")
    parser.add_argument("--output", type=Path, required=True, help="Set a new output directory.")
    parser.add_argument("--state", choices=tuple(_STATES), default="ghz3")
    parser.add_argument("--method", choices=METHODS, default="estar_direct")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="gates",
                        help="Evolve gate lists or CZ skeletons. A skeleton gets all angles fitted.")
    parser.add_argument("--encoding-mode", choices=ENCODING_MODES, default="fixed",
                        help="Keep the baseline encoding or fit one encoding per party (skeleton only).")
    parser.add_argument("--restarts", type=int, default=8, help="Set the angle fit restarts per skeleton.")
    parser.add_argument("--workers", type=int, default=1, help="Set the parallel OpenEvolve workers.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=0)
    parser.add_argument("--max-parameters", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--campaign-timeout-seconds", type=float, default=900,
                        help="Set the maximum campaign time in seconds.")
    parser.add_argument("--codex-auth-file", type=Path, help="Set the Codex account file for quota checks.")
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--cpu-seconds", type=int, default=30)
    parser.add_argument("--model-config", type=Path, help="Use a configured model. This can send paid requests.")
    parser.add_argument("--offline-model", choices=OFFLINE_MODELS, default="fixed",
                        help="Without --model-config: a fixed test candidate, random skeleton mutations, or one"
                             " Codex agent session per iteration (network, Codex quota; see QOQ_AGENT_*).")
    parser.add_argument("--blank-start", action="store_true",
                        help="Start from the cut conditions of the target, without the baseline structure.")
    parser.add_argument("--prompt-notes", type=Path,
                        help="Add a UTF-8 text file (at most 4096 bytes) to the skeleton prompt.")
    parser.add_argument("--objective", choices=OBJECTIVES, default="preparation",
                        help="Rank by preparation CZs, by preparation plus decoder CZs, or (cost, needs"
                             " --architecture) by the routed success estimate before the gate counts.")
    parser.add_argument("--narrator-model-config", type=Path,
                        help="Use this model for short progress summaries (optional, can send paid requests).")
    parser.add_argument("--initial-program", type=Path,
                        help="Start from this skeleton file instead of the baseline CZ structure.")
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES),
                        help="Rank by the costs after layout and routing on this device (offline fake backend).")
    parser.add_argument("--polish-seconds", type=float, default=0,
                        help="Lower the two-qubit depth of the seed and of the best skeleton with exact CZ swaps,"
                             " for at most this time each (0 turns it off).")
    args = parser.parse_args()
    try:
        result = run_campaign(args.baseline, args.output, state=args.state, method=args.method,
                              iterations=args.iterations, seed=args.seed, max_nfev=args.max_nfev,
                              max_parameters=args.max_parameters, timeout_seconds=args.timeout_seconds,
                              memory_mb=args.memory_mb, cpu_seconds=args.cpu_seconds, model_config=args.model_config,
                              campaign_timeout_seconds=args.campaign_timeout_seconds, codex_auth_file=args.codex_auth_file,
                              representation=args.representation, encoding_mode=args.encoding_mode,
                              restarts=args.restarts, workers=args.workers, offline_model=args.offline_model,
                              objective=args.objective, prompt_notes=args.prompt_notes,
                              initial_program=args.initial_program, blank_start=args.blank_start,
                              narrator_model_config=args.narrator_model_config, architecture=args.architecture,
                              polish_seconds=args.polish_seconds)
    except KeyboardInterrupt:
        return 130
    print(f"CZ={result['saved']['cz_count']} F={result['saved']['F']:.16g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
