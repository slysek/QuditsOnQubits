"""Discovery pipeline: ideation, one evolution loop per structure family, and the global best.

1. Ideation: the model reads the task and proposes families of CZ structures, each with a seed skeleton.
2. Each family runs its own OpenEvolve campaign (LLM edits, angle fit, independent state check).
3. Migration: from the second round, the prompt of each family shows the best skeletons of the others.
4. The best valid circuit of all families wins.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys
from time import monotonic

from .architectures import ARCHITECTURES
from .checkpoints import resume_operation, saved_run
from .harness import BASELINES, _save
from .oe_candidates import load_skeleton, skeleton_program
from .oe_evaluator import _STATES, ENCODING_MODES, OBJECTIVES, rank_key
from .progress import ask_ideation_model as _ask_model
from .oe_run import (INVALID_BEST, MAX_NOTES_BYTES, METHODS, OFFLINE_LABELS, OFFLINE_MODELS, _source,
                     check_options, method_for, run_campaign)
from .polish import POLISH_SECONDS
from .records import STORE, as_baseline, as_warm_start, seed_records, update as update_record
from .skeleton_fit import cut_requirements, cut_seed, cut_violations, skeleton_of
from .state_workloads import describe, layout

MAX_FAMILIES = 8
MAX_ROUNDS = 10
MAX_IDEA_CHARS = 1000
MAX_ANSWER_CHARS = 65536
# A campaign whose seed fit stays below this fidelity starts from the known structure (see oe_run).
MIN_SEED_FIDELITY = 0.5
METRICS = ("is_valid", "cz_count","decoder_cz_count", "two_qubit_depth", "depth", "one_qubit_gate_count", "F", "L")
IDEATION_SYSTEM = ("You lead a search for short quantum circuits. You propose distinct families of"
                   " constructions. Each family later gets its own automated search. Answer with JSON only.")
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def ideation_prompt(state, encoding_mode, source_pairs, families, *, blank_start=False, agent=False):
    """Describe the task and ask for structure families with seed skeletons."""
    problem, units = describe(state, encoding_mode)
    cuts = "\n".join(f"  - {units} {', '.join(map(str, side))} versus the others: at least {need} CZ across"
                     for side, need in cut_requirements(state))
    reference = ("No valid circuit or fitted angles are supplied." if blank_start else
                 f"A known valid structure with {len(source_pairs)} CZ: {json.dumps([list(p) for p in source_pairs])}")
    seed_rules = ("Build each seed from the target and the cut conditions. Do not use existing circuits or records."
                  if blank_start else
                  "Use the known structure as the seed, or a small change of it that keeps most CZs in the same order. "
                  f"The search replaces a seed that gets a fidelity below {MIN_SEED_FIDELITY} in its first fit with the known structure.")
    fit_hint = ("No existing circuit supplies the start angles." if blank_start else
                "The first fit starts from the angles of the most similar valid structure (longest common CZ subsequence). "
                "For a large state such as AME(4,6), random angles rarely reach the state. "
                "A seed that keeps much of the known structure is more likely to become valid.")
    prompt = f"""# Task
{problem}
A candidate is an ordered list of CZ qubit pairs. A general one-qubit gate goes on every qubit at the
start and on both qubits after each CZ, and a numerical fit sets all angles. {fit_hint}
The cost is the CZ count, then the two-qubit depth.
Necessary conditions from the Schmidt rank (each CZ across a cut at most doubles it):
{cuts}
{reference}

# Request
Propose {families} families. Each family is a different strategy that changes a valid structure step by
step to fewer CZ gates or a smaller two-qubit depth, for example: remove the CZs of one block and repair
with CZs of other blocks, move CZs to a hub block, use more CZs inside one block, or use the structure of
the state itself. For each family give a short name, an idea of at most 600 characters that a later
search will follow, and one seed CZ list that satisfies all cut conditions. {seed_rules}
Return only one JSON list in a ```json block:
[{{"name": "...", "idea": "...", "cz": [[a, b], ...]}}]
"""
    if agent:
        prompt += ("\nEach family gets a coding agent with tools. Propose construction algorithms, not only local CZ edits.\n"
                   "The agent first constructs a valid circuit. It then compresses that circuit.\n"
                   "The seed CZ list is a provisional structure. It does not constrain the final algorithm.\n")
    return prompt


def _family(item, state, index):
    """Check one proposed family. A seed that breaks a cut condition cannot reach the state."""
    if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("idea"), str):
        raise ValueError("A family needs a name, an idea and a cz list")
    violations = cut_violations(state, item.get("cz"))
    if violations:
        raise ValueError(f"The seed has too few CZ across {len(violations)} cut(s)")
    pairs = [[int(a), int(b)] for a, b in item["cz"]]
    if len(pairs) < 2:
        raise ValueError("The seed must contain at least two CZ gates")
    slug = re.sub(r"[^a-z0-9]+", "_", item["name"].lower()).strip("_")[:32] or "family"
    return {"name": f"{index}_{slug}", "title": item["name"][:80], "idea": item["idea"][:MAX_IDEA_CHARS],
            "cz": pairs}


def select_families(items, state, families):
    """Return (accepted, rejected) families from the first `families` items."""
    if not isinstance(items, list):
        raise ValueError("The ideation answer must be a JSON list")
    accepted, rejected = [], []
    for item in items[:families]:
        try:
            accepted.append(_family(item, state, len(accepted)))
        except ValueError as exc:
            rejected.append({"item": repr(item)[:500], "reason": str(exc)})
    if not accepted:
        raise ValueError("The ideation answer has no usable family")
    return accepted, rejected


def parse_families(answer, state, families):
    """Read the JSON list of families from a model answer. The answer is data; nothing in it runs."""
    if not isinstance(answer, str) or len(answer) > MAX_ANSWER_CHARS:
        raise ValueError("The ideation answer is empty or too long")
    if not answer.strip():
        raise ValueError("The ideation model returned no text. The provider may have exhausted its output budget"
                         " at the selected reasoning effort")
    match = _JSON_BLOCK.search(answer)
    try:
        items = json.loads(match.group(1) if match else answer)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The ideation answer is not valid JSON") from exc
    return select_families(items, state, families)


def offline_families(state, source_pairs):
    """Return fixed families without a model: the baseline, a hub block and a qubit chain."""
    n, _, q = layout(state)
    hub = [[j, j + 1] for j in range(q - 1)] + [[j, q * k + j] for k in range(1, n) for j in range(q)]
    chain = [[b, b + 1] for b in range(q * n - 1)]
    return [{"name": "source", "idea": "Start from the known baseline structure.", "cz": [list(p) for p in source_pairs]},
            {"name": "hub", "idea": "Block 0 makes a superposition of its levels and copies it to every other block.",
             "cz": hub},
            {"name": "chain", "idea": "Pass the entanglement along a chain of neighbouring qubits.", "cz": chain}]


def family_notes(family, others, user_notes=""):
    """Write the prompt notes of one family: its idea, then the best skeletons of the other families."""
    lines = [f"Family \"{family['title']}\": {family['idea']}",
             "Follow this family idea. Find a valid construction before compression."]
    if others:
        lines.append("Best structures of the other families (you can reuse parts of them):")
        for o in others:
            status = "valid" if o["is_valid"] else f"INVALID with fidelity F={o['F']:.4f}"
            lines.append(f"- {o['title']}: {o['cz_count']} CZ, two-qubit depth {o['two_qubit_depth']},"
                         f" {status}: {json.dumps(o['cz'])}")
            if o.get("builder"):
                lines.append(f"Construction program: {o['builder']}")
    if user_notes:
        lines.append(user_notes)
    # shortcut: long notes lose their tail (user notes first); split the byte budget per section if that bites.
    return "\n".join(lines).encode()[:MAX_NOTES_BYTES].decode("utf-8", errors="ignore")


def _score(metrics, objective):
    return rank_key({**metrics, "decoder_cz_count": metrics.get("decoder_cz_count", 0)}, objective)


def _update_records(store, output, report, state, objective):
    """Offer the run winner to the record of the processor and to the logical record (not for a cost run)."""
    winner = report["best"]
    provenance = {"source": {"run": output.name, "family": winner["family"],
                             "directory": str(output / winner["directory"])},
                  "settings": {key: report[key] for key in ("model_mode", "ideation_mode", "rounds", "iterations",
                                                            "method", "encoding_mode", "objective", "architecture")}}
    results = []
    # The cost needs the error data of a processor, so a cost run has no logical record.
    for processor in dict.fromkeys((report["architecture"],) + (() if objective == "cost" else (None,))):
        try:
            result = update_record(store, state, processor, output / winner["directory"], objective=objective,
                                   provenance=provenance)
        except Exception as exc:  # the report is saved; a record error must not stop the run
            result = {"record": str(store.directory(state, processor, objective)), "error": f"{type(exc).__name__}: {exc}"}
        outcome = result.get("error") or result.get("skipped") or ("replaced" if result["accepted"] else "kept")
        print(f"Record {result['record']}: {outcome}; run winner {result.get('new')}, record {result.get('old')}",
              flush=True)
        results.append(result)
    return results


def _invalid_best(directory, width, objective):
    """Return (pairs, metrics) of a campaign that found no valid candidate, or None.

    The next round of the family starts from this structure, and the other families see it.
    """
    try:
        saved = json.loads((directory / INVALID_BEST).read_text(encoding="utf-8"))
        metrics = {key: saved.get(key, 0) for key in METRICS}
        _score(metrics, objective)  # rejects a missing or non-finite fidelity
        return load_skeleton(directory / "best_program.py", expected_qubits=width), metrics
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _positive_seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1_000_000_000 or not math.isfinite(value):
        raise ValueError(f"The {label} time limit must be positive")


def run_discovery(baseline, output, *, state="ghz3", method="estar_direct", families=3, rounds=1,
                  model_config=None, offline_model="fixed", encoding_mode="fixed", objective="preparation",
                  iterations=1, seed=0, workers=1, restarts=8, timeout_seconds=30, cpu_seconds=30,
                  memory_mb=2048, prompt_notes=None, campaign_timeout_seconds=900,
                  discovery_timeout_seconds=3600, ideation_model_config=None, narrator_model_config=None,
                  architecture=None, polish_seconds=POLISH_SECONDS, record_store=None, from_baseline=False,
                  blank_start=False, resume=False):
    """Run ideation, then `rounds` rounds of one campaign per family. Return the report.

    record_store (see records.RecordStore) turns on the records: the run starts from the record of the
    processor (else the logical record) instead of the baseline, the other records give warm starts, and
    the winner can replace the record of the processor and the logical record. from_baseline=True starts
    from the baseline; the records still give warm starts and get the winner.

    ideation_model_config selects the orchestrator model. The default is model_config (the loop model).
    polish_seconds > 0 lets each campaign lower the two-qubit depth of its seed and of its best skeleton
    with exact CZ swaps (polish.py, no model), so the loop can work on the CZ structure.
    architecture names a processor (see architectures.ARCHITECTURES). Then the campaigns and the family
    selection rank the valid circuits by their gate costs after layout and routing on that processor.
    """
    parameters = locals().copy()
    started = monotonic()
    method = method_for(state, method)
    for value, limit, label in ((families, MAX_FAMILIES, "family count"), (rounds, MAX_ROUNDS, "round count")):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"The {label} must be from 1 to {limit}")
    _positive_seconds(campaign_timeout_seconds, "campaign")
    _positive_seconds(discovery_timeout_seconds, "discovery")
    ideation_config = ideation_model_config if ideation_model_config is not None else model_config
    for config in (model_config, ideation_config):
        if config is not None and not Path(config).is_file():
            raise ValueError("The model configuration file does not exist")
    campaign = {"state": state, "method": method, "iterations": iterations, "seed": seed, "workers": workers,
                "representation": "skeleton", "encoding_mode": encoding_mode, "restarts": restarts,
                "objective": objective, "timeout_seconds": timeout_seconds, "cpu_seconds": cpu_seconds,
                "memory_mb": memory_mb, "offline_model": offline_model, "model_config": model_config,
                "architecture": architecture}
    user_notes = check_options(**campaign, max_nfev=0, max_parameters=12, prompt_notes=prompt_notes,
                               initial_program=None)
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError("The output directory already exists")
    with saved_run(output, "discover", parameters, resume) as checkpoint:
        setup_path = output / "discovery_inputs.json"
        if resume and setup_path.is_file():
            inputs_config = json.loads(setup_path.read_text())
            baseline = Path(inputs_config["baseline"])
            record_warm = [Path(path) for path in inputs_config["record_warm"]]
        else:
            seed_record, other_records = (None, [])
            if record_store is not None and not blank_start:
                seed_record, other_records = seed_records(record_store, state, architecture, {
                    "encoding_mode": encoding_mode, "objective": objective, "method": method},
                    run_roots=() if from_baseline else (output.parent, record_store.archive.parent / "discovery"))
            if seed_record is not None and from_baseline:
                seed_record, other_records = None, [seed_record, *other_records]
            if seed_record is not None:
                baseline = as_baseline(seed_record, output / "inputs" / "record", state, method)
                print(f"Start from the record {seed_record.directory} ({seed_record.metrics['cz_count']} CZ)", flush=True)
            record_warm = [as_warm_start(record, output / "inputs" / "records" / record.directory.name)
                           for record in other_records]
            inputs_config = {"baseline": str(Path(baseline).resolve()),
                             "record_warm": [str(path.resolve()) for path in record_warm],
                             "seed_record": None if seed_record is None else str(seed_record.directory),
                             "record_warm_starts": [str(record.directory) for record in other_records]}
            _save(setup_path, inputs_config)
        circuit = _source(Path(baseline), state, method)[0]
        source_pairs = cut_seed(state) if blank_start else skeleton_of(circuit)
        width = _STATES[state]

        record_config = None if ideation_config is None else str(Path(ideation_config).resolve())
        ideation_path = output / "ideation.json"
        saved_ideation = json.loads(ideation_path.read_text()) if resume and ideation_path.is_file() else {}
        if resume and not saved_ideation.get("families"):
            checkpoint.setdefault("ideation_seconds", checkpoint["elapsed_seconds"])
        if saved_ideation.get("families"):
            accepted, rejected = select_families(
                [{"name": family["title"], "idea": family["idea"], "cz": family["cz"]}
                 for family in saved_ideation["families"]], state, families)
            prompt, answer = saved_ideation["prompt"], saved_ideation["answer"]
        elif ideation_config is None:
            prompt = answer = None
            accepted, rejected = select_families(offline_families(state, source_pairs), state, families)
        else:
            prompt = ideation_prompt(state, encoding_mode, source_pairs, families, blank_start=blank_start,
                                     agent=model_config is None and offline_model == "agent")
            print("Ideation: waiting for the model to return the families. No harness time limit.", flush=True)
            ideation_started = monotonic()
            try:
                answer = _ask_model(ideation_config, IDEATION_SYSTEM, prompt)
            finally:
                checkpoint["ideation_seconds"] = (checkpoint.get("ideation_seconds", 0)
                                                   + monotonic() - ideation_started)
            try:
                accepted, rejected = parse_families(answer, state, families)
            except ValueError:
                output.mkdir(parents=True, exist_ok=True)
                _save(output / "ideation.json", {"model_config": record_config, "prompt": prompt, "answer": answer,
                                                 "families": [], "rejected": []})
                raise
        output.mkdir(parents=True, exist_ok=True)
        if not saved_ideation.get("families"):
            _save(ideation_path, {"model_config": record_config, "prompt": prompt, "answer": answer,
                                  "families": accepted, "rejected": rejected})
        source = "The orchestrator proposed" if ideation_config else "Offline ideation gives"
        print(f"{source} {len(accepted)} families ({len(rejected)} rejected; details in ideation.json):", flush=True)
        for family in accepted:
            print(f"  {family['name']}: seed {len(family['cz'])} CZ. {family['idea'][:110]}", flush=True)

        report = {"status": "running", "state": state, "method": method, "encoding_mode": encoding_mode,
                  "objective": objective,
                  "blank_start": blank_start,
                  "architecture": architecture,
                  "model_mode": "configured_model" if model_config else OFFLINE_LABELS[offline_model],
                  "ideation_mode": "configured_model" if ideation_config else "offline_families",
                  "rounds": rounds, "iterations": iterations, "families": [f["name"] for f in accepted],
                  "seed_record": inputs_config["seed_record"],
                  "record_warm_starts": inputs_config["record_warm_starts"],
                  "runs": [], "family_best": {}, "best": None}
        path = output / "discovery.json"
        if resume and path.is_file():
            report = json.loads(path.read_text())
            if report["status"] == "completed":
                return report
            report["status"] = "running"
            report.pop("error", None)
        _save(path, report)
        best = report["family_best"]
        for round_index in range(1, rounds + 1):
            for family in accepted:
                elapsed = checkpoint["elapsed_seconds"] + monotonic() - started - checkpoint.get("ideation_seconds", 0)
                remaining = discovery_timeout_seconds - elapsed
                if remaining <= 0:
                    report["error"] = "DiscoveryTimeout"
                    break
                name = f"round_{round_index}_{family['name']}"
                record = next((row for row in report["runs"] if row["directory"] == name), None)
                if record is not None and record["status"] in ("completed", "failed", "invalid"):
                    continue
                current = best.get(family["name"])
                others = [b for key, b in best.items() if key != family["name"]]
                if model_config is None and offline_model == "agent":
                    others = [{**b, "builder": str((output / b["directory"] / "best_builder.py").resolve())}
                              if (output / b["directory"] / "best_builder.py").is_file() else b for b in others]
                inputs = output / "inputs" / name
                inputs.mkdir(parents=True, exist_ok=resume)
                if not (inputs / "seed.py").is_file():
                    if current is not None:
                        (inputs / "seed.py").write_bytes((output / current["directory"] / "best_program.py").read_bytes())
                    elif not blank_start and (Path(baseline) / "initial_program.py").is_file():
                        (inputs / "seed.py").write_bytes((Path(baseline) / "initial_program.py").read_bytes())
                    else:
                        (inputs / "seed.py").write_text(skeleton_program(width, family["cz"]), encoding="utf-8")
                    (inputs / "notes.txt").write_text(family_notes(family, others, user_notes), encoding="utf-8")
                if record is None:
                    record = {"round": round_index, "family": family["name"], "directory": name, "status": "running"}
                    report["runs"].append(record)
                record["status"] = "running"
                _save(path, report)
                print(f"\nFamily {family['name']}, round {round_index}/{rounds}: start", flush=True)
                try:
                    # The valid best circuits of all families give the start angles of similar skeletons.
                    warm = [output / b["directory"] for b in best.values() if b.get("is_valid")] + record_warm
                    result = (resume_operation(output / name) if (output / name / "resume.json").is_file() else
                              run_campaign(Path(baseline), output / name, **campaign, warm_starts=warm,
                                          prompt_notes=inputs / "notes.txt", initial_program=inputs / "seed.py",
                                          campaign_timeout_seconds=min(campaign_timeout_seconds, remaining),
                                          narrator_model_config=narrator_model_config,
                                          min_seed_fidelity=0.0 if blank_start or (model_config is None and offline_model == "agent")
                                          else MIN_SEED_FIDELITY,
                                          blank_start=blank_start, polish_seconds=polish_seconds))
                except KeyboardInterrupt:
                    record["status"] = report["status"] = "paused"
                    _save(path, report)
                    raise
                except Exception as exc:
                    # One family can fail (invalid seed, time limit) without stopping the others.
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    found = _invalid_best(output / name, width, objective)
                    if found is None:
                        _save(path, report)
                        print(f"Family {family['name']}, round {round_index}: no result ({type(exc).__name__}:"
                              f" {str(exc)[:120]}). Full log: {output / name / 'campaign.log'}",
                              file=sys.stderr, flush=True)
                        continue
                    pairs, metrics = found
                    record.update(status="invalid", **metrics)
                    print(f"Family {family['name']}, round {round_index}: no valid circuit; best {metrics['cz_count']} CZ"
                          f" with fidelity {metrics['F']:.4f}. Full log: {output / name / 'campaign.log'}", flush=True)
                else:
                    metrics = {key: result["saved"].get(key, 0) for key in METRICS}
                    # With an architecture, the routed costs come first in _score, as in the campaign.
                    metrics.update({k: v for k, v in result.get("best", {}).items() if k.startswith("routed_")})
                    pairs = load_skeleton(output / name / "best_program.py", expected_qubits=width)
                    record.update(status="completed", **metrics)
                if current is None or _score(metrics, objective) > _score(current, objective):
                    best[family["name"]] = {"title": family["title"], "directory": name,
                                            "cz": [list(p) for p in pairs], **metrics}
                _save(path, report)
            if "error" in report:
                break
        # family_best can hold invalid structures (they seed the next round); only a valid one can win.
        valid = [key for key in best if best[key]["is_valid"]]
        if valid:
            winner = max(valid, key=lambda key: _score(best[key], objective))
            report["best"] = {"family": winner, **best[winner]}
        report["status"] = "completed" if valid and "error" not in report else "failed"
        _save(path, report)
        if record_store is not None and not blank_start and report["best"] is not None:
            report["records"] = _update_records(record_store, output, report, state, objective)
            _save(path, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run ideation, one evolution loop per family, and select the best.")
    parser.add_argument("--output", type=Path, required=True, help="Set a new output directory.")
    parser.add_argument("--state", choices=tuple(BASELINES), default="ghz3")
    parser.add_argument("--baseline", type=Path, help="Set the baseline directory. The default depends on the state.")
    parser.add_argument("--method", choices=METHODS, default="estar_direct")
    parser.add_argument("--families", type=int, default=3, help="Set the number of families from ideation.")
    parser.add_argument("--rounds", type=int, default=1, help="Set the rounds. Families exchange skeletons between rounds.")
    parser.add_argument("--model-config", type=Path, help="Use a configured model. This can send paid requests.")
    parser.add_argument("--ideation-model-config", type=Path,
                        help="Use this model for ideation. The default is --model-config.")
    parser.add_argument("--narrator-model-config", type=Path,
                        help="Use this model for short progress summaries (optional).")
    parser.add_argument("--offline-model", choices=OFFLINE_MODELS, default="fixed",
                        help="Without --model-config: a fixed test candidate, random skeleton mutations, or one"
                             " Codex agent session per iteration (network, Codex quota; see QOQ_AGENT_*).")
    parser.add_argument("--encoding-mode", choices=ENCODING_MODES, default="fixed")
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES),
                        help="Rank the circuits by their gate costs after layout and routing on this processor.")
    parser.add_argument("--objective", choices=OBJECTIVES, default="preparation",
                        help="Rank by preparation CZs, by preparation plus decoder CZs, or (cost, needs"
                             " --architecture) by the routed success estimate before the gate counts.")
    parser.add_argument("--iterations", type=int, default=1, help="Set the iterations per family and round.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30, help="Set the time limit per candidate.")
    parser.add_argument("--cpu-seconds", type=int, default=30, help="Set the CPU time limit per candidate.")
    parser.add_argument("--memory-mb", type=int, default=2048, help="Set the memory limit per candidate.")
    parser.add_argument("--prompt-notes", type=Path, help="Add a UTF-8 text file to the prompt of each family.")
    parser.add_argument("--campaign-timeout-seconds", type=float, default=900, help="Set the time limit per campaign.")
    parser.add_argument("--discovery-timeout-seconds", type=float, default=3600, help="Set the total time limit.")
    parser.add_argument("--polish-seconds", type=float, default=POLISH_SECONDS,
                        help="Lower the cost of each seed and best skeleton with exact CZ swaps, for at most this"
                             " time each (0 turns it off). With --architecture, the cost is after routing.")
    parser.add_argument("--from-baseline", action="store_true",
                        help="Start from the baseline, not from the record in quantum_circuits/<state>/. A given"
                             " --baseline also does this. The run winner can still replace the record.")
    parser.add_argument("--blank-start", action="store_true",
                        help="Use no existing circuit or fitted angles. Do not read or update circuit records.")
    args = parser.parse_args(argv)
    try:
        report = run_discovery(args.baseline or BASELINES[args.state], args.output, state=args.state,
                               method=args.method, families=args.families, rounds=args.rounds,
                               model_config=args.model_config, offline_model=args.offline_model,
                               encoding_mode=args.encoding_mode, objective=args.objective,
                               iterations=args.iterations, seed=args.seed, workers=args.workers,
                               restarts=args.restarts, timeout_seconds=args.timeout_seconds,
                               cpu_seconds=args.cpu_seconds, memory_mb=args.memory_mb,
                               prompt_notes=args.prompt_notes, campaign_timeout_seconds=args.campaign_timeout_seconds,
                               discovery_timeout_seconds=args.discovery_timeout_seconds,
                               ideation_model_config=args.ideation_model_config,
                               narrator_model_config=args.narrator_model_config,
                               architecture=args.architecture, polish_seconds=args.polish_seconds,
                               record_store=STORE,
                               from_baseline=args.from_baseline or args.baseline is not None,
                               blank_start=args.blank_start)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    best = report["best"]
    result = (f"family {best['family']}, {best['cz_count']} CZ (2q depth {best['two_qubit_depth']},"
              f" decoder {best['decoder_cz_count']} CZ)" if best else "no valid circuit")
    if best and "routed_two_qubit_count" in best:
        result += (f"; on {args.architecture}: {best['routed_two_qubit_count']} two-qubit gates"
                   f" (2q depth {best['routed_two_qubit_depth']}) after routing")
    print(f"\nBest: {result}. Report: {args.output / 'discovery.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
