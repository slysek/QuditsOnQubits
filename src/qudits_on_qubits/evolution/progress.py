"""Short progress lines and optional model summaries for a running campaign.

The campaign worker writes its full console output to campaign.log. Progress reads the new lines, prints
a few short lines (seed, new best, status, errors) and appends them to progress.log. An optional narrator
model writes one or two sentences about the state from time to time and at the end.
"""

import asyncio
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
from time import monotonic

ITERATION = re.compile(r"Iteration (\d+): Program \S+ \(parent: \S+\) completed in")
METRICS = re.compile(r" - Metrics: (.+)$")
EVALUATED = re.compile(r"Evaluated program \S+ in [\d.]+s: (.+)$")
NEW_BEST = re.compile(r"New best solution found at iteration (\d+)")
ADMISSION_REJECTED = re.compile(r"Admission strategy rejected program")
REJECTED = re.compile(r"The candidate evaluation failed: (?:\w+: )?(.+)$")
ERROR = re.compile(r" - ERROR - (.+)$")
PASS_THROUGH = ("Codex quota", "Seed check")
LIVE_LOGS = ("campaign.log", "progress.log")
STATUS_PARTS = 4
STATUS_SECONDS = 120
NARRATE_SECONDS = 300
NARRATOR_WAIT_SECONDS = 60
NARRATOR_LANGUAGE = "Polish"
NARRATOR_SYSTEM = ("You write short status notes about a running search for short quantum circuits. The cost is"
                   " the CZ count (fewer is better). 'valid' means that the circuit prepares the target state."
                   " Write plain text: no lists, no markdown.")
MAX_TEXT = 160
# CLIProxy defaults Fable to 64,000 tokens. The API permits 128,000 tokens for Fable 5.
FABLE_OUTPUT_TOKENS = 128_000


def ask_model(model_config: str | Path, system: str, prompt: str, *, unbounded: bool = False) -> str:
    """Send one request with the OpenEvolve client of the configured model and return the text."""
    from openevolve.config import load_config
    from openevolve.llm.ensemble import LLMEnsemble

    config = load_config(str(model_config))
    if not config.llm.models:
        raise ValueError("The model configuration must contain a model")
    if unbounded:
        for model in config.llm.models:
            model.timeout = None
            model.max_tokens = FABLE_OUTPUT_TOKENS if model.name.startswith("claude-fable-5") else None
    messages = [{"role": "user", "content": prompt}]
    return asyncio.run(LLMEnsemble(config.llm.models).generate_with_context(system, messages))


def ask_ideation_model(model_config: str | Path, system: str, prompt: str) -> str:
    """Wait for the families without a client timeout. Use the provider output budget."""
    # A separate process lets Ctrl+C stop the HTTP thread when its timeout is disabled.
    worker = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from qudits_on_qubits.evolution.progress import ask_model\n"
        "try:\n"
        "    result = {'answer': ask_model(**json.load(sys.stdin))}\n"
        "except Exception as exc:\n"
        "    result = {'error': f'{type(exc).__name__}: {exc}'}\n"
        "Path(sys.argv[1]).write_text(json.dumps(result), encoding='utf-8')\n"
    )
    with tempfile.TemporaryDirectory(prefix="qoq-ideation-") as directory:
        result_path = Path(directory) / "answer.json"
        subprocess.run([sys.executable, "-c", worker, str(result_path), str(Path(__file__).resolve().parents[2])],
                       input=json.dumps({"model_config": str(model_config), "system": system, "prompt": prompt,
                                         "unbounded": True}), text=True, check=True, start_new_session=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if "error" in result:
        raise ValueError(result["error"])
    return result["answer"]


def _values(text):
    values = {}
    for part in text.split(", "):
        name, _, value = part.partition("=")
        try:
            values[name.strip()] = float(value)
        except ValueError:
            pass
    return values


def _cost(values):
    return (f"{int(values['cz_count'])} CZ, 2q depth {int(values.get('two_qubit_depth', 0))},"
            f" decoder {int(values.get('decoder_cz_count', 0))} CZ")


def _duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class Progress:
    """Turn the campaign log into short lines. Call poll() often and finish() at the end."""

    def __init__(self, label, output, iterations, narrator_config=None, narrate_seconds=NARRATE_SECONDS):
        self.label, self.output, self.iterations = label, output, iterations
        self.narrator_config, self.narrate_seconds = narrator_config, narrate_seconds
        self.started = self.last_narration = monotonic()
        self.offset, self.partial = 0, ""
        self.counts, self.rejections, self.errors = Counter(), Counter(), Counter()
        self.iteration, self.values, self.seed, self.best = 0, {}, None, None
        self.repeated = 0
        self.history = []
        self.step = max(1, iterations // STATUS_PARTS)
        self.last_status = monotonic()
        self.changed = False
        self.lock = threading.Lock()
        self.thread = None

    def say(self, text):
        line = f"[{self.label}] {text}"
        with self.lock:
            print(line, flush=True)
            if self.output.is_dir():
                with (self.output / "progress.log").open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")

    def poll(self):
        """Read the new lines of campaign.log and report what changed."""
        path = self.output / "campaign.log"
        if not path.is_file():
            return
        with path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        self.offset += len(data)
        lines = (self.partial + data.decode("utf-8", errors="replace")).split("\n")
        self.partial = lines.pop()
        for line in lines:
            self._line(line.rstrip("\r"))
        if self.narrator_config and self.changed and monotonic() - self.last_narration >= self.narrate_seconds:
            self.narrate()

    def _line(self, line):
        if line.startswith(PASS_THROUGH):
            print(line, flush=True)
        elif match := ITERATION.search(line):
            self.iteration = int(match.group(1))
        elif match := METRICS.search(line):
            self._candidate(_values(match.group(1)))
        elif (match := EVALUATED.search(line)) and self.seed is None and self.iteration == 0:
            self.seed = _values(match.group(1))
            state = "valid" if self.seed.get("is_valid") else f"invalid (fidelity {self.seed.get('F', 0):.4f})"
            self.say(f"seed: {int(self.seed.get('cz_count', 0))} CZ, {state}; {self.iterations} iterations")
        elif match := NEW_BEST.search(line):
            if self.values.get("is_valid"):
                self.best = self.values
                self.history.append({"iteration": int(match.group(1)), "cz": int(self.values["cz_count"]),
                                     "two_qubit_depth": int(self.values.get("two_qubit_depth", 0))})
                self.say(f"iteration {match.group(1)}: new best, {_cost(self.values)}")
        elif ADMISSION_REJECTED.search(line):
            self._count("rejected")
        elif match := REJECTED.search(line):
            self.rejections[match.group(1)[:MAX_TEXT]] += 1
        elif match := ERROR.search(line):
            message = match.group(1)[:MAX_TEXT]
            self.errors[message] += 1
            if self.errors[message] == 1:
                self.say(f"error: {message}")

    def _candidate(self, values):
        self.values = values
        self.repeated += bool(values.get("cache_hit"))
        self._count("valid" if values.get("is_valid") else "invalid")

    def _count(self, kind):
        """Count one finished iteration. Workers finish out of order, so count instead of reading its number."""
        self.counts[kind] += 1
        self.changed = True
        done = sum(self.counts.values())
        if done % self.step == 0 or done == self.iterations or monotonic() - self.last_status >= STATUS_SECONDS:
            self.last_status = monotonic()
            self.say(self.status())

    def status(self):
        counts = ", ".join(f"{self.counts[k]} {k}" for k in ("valid", "invalid", "rejected"))
        text = f"{sum(self.counts.values())}/{self.iterations} iterations: {counts}"
        if self.repeated:
            text += f"; {self.repeated} repeated (from the cache)"
        if self.best:
            text += f"; best {int(self.best['cz_count'])} CZ"
        if self.rejections:
            reason = textwrap.shorten(self.rejections.most_common(1)[0][0], 90, placeholder="...")
            text += f"; most common rejection: {reason}"
        return text + f" ({_duration(monotonic() - self.started)})"

    def narrate(self, final=False):
        """Ask the narrator model for a short summary in a background thread."""
        self.changed, self.last_narration = False, monotonic()
        if self.thread is not None and self.thread.is_alive():
            if not final:
                return
            self.thread.join(NARRATOR_WAIT_SECONDS)
        data = {"campaign": self.label, "finished": final, "iterations_done": sum(self.counts.values()),
                "iterations_total": self.iterations, "elapsed_seconds": int(monotonic() - self.started),
                "candidates": dict(self.counts), "seed_cz": self.seed and int(self.seed.get("cz_count", 0)),
                "seed_valid": bool(self.seed and self.seed.get("is_valid")), "best_history": self.history[-5:],
                "common_rejections": self.rejections.most_common(3), "errors": self.errors.most_common(2)}
        prompt = (f"Data (JSON): {json.dumps(data)}\nWrite at most two short sentences in {NARRATOR_LANGUAGE}"
                  " about what the search does now and how it goes. Use only facts from the data.")
        self.thread = threading.Thread(target=self._narrate, args=(prompt,), daemon=True)
        self.thread.start()
        if final:
            self.thread.join(NARRATOR_WAIT_SECONDS)

    def _narrate(self, prompt):
        try:
            text = ask_model(self.narrator_config, NARRATOR_SYSTEM, prompt)
        except Exception as exc:  # The narrator is optional; it must never stop the search.
            self.say(f"narrator unavailable: {type(exc).__name__}")
            return
        self.say("summary: " + " ".join(str(text).split())[:400])

    def finish(self, saved):
        """Read the last lines, print the result and wait for the last summary."""
        self.poll()
        self.say(f"done in {_duration(monotonic() - self.started)}: {_cost(saved)}."
                 f" Full log: {self.output / 'campaign.log'}")
        if self.narrator_config:
            self.narrate(final=True)
