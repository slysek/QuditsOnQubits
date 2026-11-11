"""Run one Codex agent session for each OpenEvolve iteration (see docs/architektura-agentowa.md).

The session works in its own directory, in the Codex workspace-write sandbox without network access.
It can run the campaign fit (agent_tools) and its own scripts. It writes candidate.py (a CZ skeleton
with optional angles) and at most three DISCOVERY lines in notes.txt. The evaluator reads the
candidate as data; it never runs code of the session.

Settings come from the environment: QOQ_AGENT_BINARY (default codex), QOQ_AGENT_MODEL,
QOQ_AGENT_EFFORT, QOQ_AGENT_FAST (1 = the Fast service tier) and QOQ_AGENT_SECONDS (time limit of one
session, default 900).
"""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

from .oe_candidates import MAX_BYTES
from .oe_evaluator import CACHE_DIRECTORY, WARM_STARTS
from .oe_mock import _CURRENT

SESSIONS = "agent_sessions"
DISCOVERIES = "discoveries.md"
CANDIDATE = "candidate.py"
NOTES = "notes.txt"
BUILDER = "builder.py"
BUILDER_PREFIX = "QOQ_BUILDER\n"
MAX_BUILDER_BYTES = 32768
MAX_NOTES = 3
MAX_NOTE_CHARS = 300
MAX_NOTES_READ = 65536
MAX_DISCOVERY_BYTES = 4096
DEFAULT_SECONDS = 900
MAX_SESSION_SECONDS = 7200
KILL_GRACE_SECONDS = 10
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_MODEL = re.compile(r"[A-Za-z0-9._:/-]{1,100}")
TASK = """# One iteration of a circuit search

You are one iteration of an evolutionary search for short exact qubit circuits that prepare a qudit state.
You have {minutes} minutes. Work only in this directory. You have no network access.

## Phase: {phase}
Construction: first create a valid circuit from the target vector. You can change the full algorithm.
Compression: reduce the cost of a valid circuit. Keep a valid circuit if a change fails.
A low fidelity does not prove that a topology has no solution. The angle fit can fail.
Read parent_candidate.py if it exists. Read builder.py if it exists.
The builder contains the inherited construction algorithm. Improve this program, not only its CZ list.
Load the physical target vector from {target} with numpy.load(..., allow_pickle=False).
In free encoding mode, this vector uses the canonical encoding. The fit tool also fits decoder tails.

## Result
Before the time limit, write {candidate} in this directory:

    CANDIDATE = {{"num_qubits": W, "cz": [[a, b], ...], "angles": [...]}}

- cz is the ordered list of CZ pairs. Between the CZs, the circuit has general one-qubit gates.
- angles are the fitted angles. They are optional. The fit tool writes the file with them.
- Goal: a valid structure that ranks higher than the current candidate (the ranking is in the search
  state below; without a processor: fewer CZ gates, or the same CZ count and a smaller two-qubit
  depth). You can test large changes with the fit tool before you write the file.
- If you find no improvement, write the best valid structure that you found.

Also write builder.py. This file must contain the complete circuit construction program.
Include all required constants and helper functions in this file. Do not depend on earlier session files.
Run the builder in this session. It must write candidate.py with the constructed circuit and its angles.
The archive keeps this source with the literal candidate. The evaluator never executes this source.

Then write at most 3 lines in {notes} that start with "DISCOVERY:". Each line is one fact that later
sessions must know, for example a structure family that fails or a rule that works.

## Tools
Fit all angles of a CZ structure and check the state:

    {python} -m qudits_on_qubits.evolution.agent_tools fit --cz "[[0, 1], [1, 2]]" --write {candidate}

Options: --restarts N, --seconds S, --seed K. The command prints one JSON line (valid, F, error_norm,
cz_count, two_qubit_depth, cut_violations; with a processor also the routed_* costs). It writes the file only for a valid structure. A structure
that fails a cut condition gets no fit.
The tool keeps the best angles of invalid fits in fit_progress.json. Later fits can continue from them.
You can write and run your own Python scripts with {python}. Useful modules:
qudits_on_qubits.evolution.skeleton_fit (fit_skeleton, cut_requirements, warm_angles) and
qudits_on_qubits.evolution.state_workloads (logical_state, layout).
All evaluated structures of this search and their results are in {cache}/*.json. Read them only.

## Rules
- An independent evaluator checks the full state of your candidate again. It reads the file as data.
- Do not change files outside this directory.
- The search state below is written for a model without tools. Write your result to {candidate}.

{discoveries}## Search state
{prompt}
"""


def builder_source(code: str) -> str | None:
    """Read the archived builder string without executing the candidate."""
    if len(code.encode("utf-8")) > MAX_BYTES:
        raise ValueError("The candidate file is too large")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError("The candidate syntax is invalid") from exc
    text = ast.get_docstring(tree, clean=False)
    if text is None or not text.startswith(BUILDER_PREFIX):
        return None
    source = text[len(BUILDER_PREFIX):]
    if len(source.encode("utf-8")) > MAX_BUILDER_BYTES:
        raise ValueError("The builder source is too large")
    return source


def with_builder(code: str, source: str | None) -> str:
    """Attach builder source as data to a literal candidate."""
    if source is None:
        return code
    if len(source.encode("utf-8")) > MAX_BUILDER_BYTES:
        raise ValueError("The builder source is too large")
    if not source.strip():
        raise ValueError("The builder source must not be empty")
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError("The builder syntax is invalid") from exc
    builder_source(code)
    tree = ast.parse(code)
    if ast.get_docstring(tree, clean=False) is not None:
        code = "\n".join(code.splitlines()[tree.body[0].end_lineno:]) + "\n"
    result = repr(BUILDER_PREFIX + source).replace("`", "\\x60") + "\n" + code
    if len(result.encode("utf-8")) > MAX_BYTES:
        raise ValueError("The candidate with builder source is too large")
    return result


def _phase(campaign: Path) -> str:
    """Select compression only when the campaign has a checked valid circuit."""
    try:
        initial = json.loads((campaign / "diagnostics.json").read_text())["initial_program"]
        if initial.get("metrics", {}).get("is_valid"):
            return "compression"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    for path in (campaign / CACHE_DIRECTORY).glob("*.json"):
        try:
            if json.loads(path.read_text())["metrics"]["is_valid"]:
                return "compression"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    try:
        if any(item.get("is_valid", True) for item in json.loads((campaign / WARM_STARTS).read_text())):
            return "compression"
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return "construction"


def session_usage(path: Path) -> dict[str, int]:
    """Read token counts from Codex or Claude session events."""
    turns, result, messages = [], None, {}
    if not path.exists():
        return {}
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn.completed":
                turns.append(event.get("usage") or {})
            elif event.get("type") == "result":
                result = event.get("usage") or {}
            elif event.get("type") == "assistant" and isinstance(event.get("message"), dict):
                messages[event["message"].get("id")] = event["message"].get("usage") or {}
    total = {}
    # shortcut: interrupted Claude sessions report partial message counts, so their output count can be too low.
    for usage in turns or ([result] if result is not None else messages.values()):
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    return total


def agent_summary(campaign: Path) -> dict:
    """Count sessions and tokens, including sessions without a candidate."""
    sessions = list((campaign / SESSIONS).glob("session_*"))
    usage = {}
    missing = 0
    for session in sessions:
        counts = session_usage(session / "events.jsonl")
        missing += not bool(counts)
        for key, value in counts.items():
            usage[key] = usage.get(key, 0) + value
    return {"sessions": len(sessions), "sessions_without_usage": missing, "usage": usage}


def agent_settings():
    """Read and check the agent settings from the environment."""
    try:
        seconds = int(os.environ.get("QOQ_AGENT_SECONDS", DEFAULT_SECONDS))
    except ValueError as exc:
        raise ValueError("QOQ_AGENT_SECONDS must be an integer") from exc
    if not 1 <= seconds <= MAX_SESSION_SECONDS:
        raise ValueError(f"QOQ_AGENT_SECONDS must be from 1 to {MAX_SESSION_SECONDS}")
    model = os.environ.get("QOQ_AGENT_MODEL") or None
    if model is not None and not _MODEL.fullmatch(model):
        raise ValueError("QOQ_AGENT_MODEL is not a valid model name")
    effort = os.environ.get("QOQ_AGENT_EFFORT") or None
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"QOQ_AGENT_EFFORT must be one of {', '.join(EFFORTS)}")
    fast = os.environ.get("QOQ_AGENT_FAST", "")
    if fast not in ("", "0", "1"):
        raise ValueError("QOQ_AGENT_FAST must be 0 or 1")
    binary = os.environ.get("QOQ_AGENT_BINARY", "codex")
    if shutil.which(binary) is None:
        raise ValueError(f"The agent program {binary!r} is not found")
    return {"binary": binary, "model": model, "effort": effort, "fast": fast == "1", "seconds": seconds}


def _tool_env():
    """Environment of the session commands: the campaign settings, the package path and one BLAS thread."""
    source = str(Path(__file__).resolve().parents[2])
    return {"QOQ_OE_CONFIG": os.environ["QOQ_OE_CONFIG"],
            "PYTHONPATH": os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH")))),
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"}


def _claude_argv(settings, session, writable):
    """Headless Claude Code in a bwrap jail: the file system is read-only except the session, the writable
    directories, a private /tmp and the Claude state; user hooks, plugins and MCP servers are not loaded."""
    home = Path.home()
    jail = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in (session, *writable, home / ".claude", home / ".claude.json"):
        jail += ["--bind", str(path), str(path)]
    argv = jail + ["--chdir", str(session), settings["binary"], "-p", "--output-format", "stream-json", "--verbose",
                   "--permission-mode", "dontAsk", "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
                   "--setting-sources", "project,local", "--strict-mcp-config", "--no-session-persistence"]
    if settings["model"]:
        argv += ["--model", settings["model"]]
    if settings["effort"]:
        argv += ["--effort", settings["effort"]]
    for path in writable:
        argv += ["--add-dir", str(path)]
    return argv


def _argv(settings, session, env, writable=()):
    """Return the session command. writable lists more directories that the session can write."""
    if Path(settings["binary"]).name == "claude":
        return _claude_argv(settings, session, writable)
    argv = [settings["binary"], "exec", "--json", "--skip-git-repo-check", "--ignore-user-config",
            "-C", str(session), "-s", "workspace-write"]
    if writable:
        argv += ["-c", f"sandbox_workspace_write.writable_roots={json.dumps([str(p) for p in writable])}"]
    if settings["model"]:
        argv += ["-m", settings["model"]]
    if settings["effort"]:
        argv += ["-c", f"model_reasoning_effort={json.dumps(settings['effort'])}"]
    if settings.get("fast"):
        argv += ["-c", 'service_tier="fast"']  # the value that the Codex /fast command writes
    table = "{" + ", ".join(f"{key} = {json.dumps(value)}" for key, value in env.items()) + "}"
    return argv + ["-c", "sandbox_workspace_write.network_access=false",
                   "-c", f"shell_environment_policy.set={table}",
                   "-o", str(session / "last_message.txt"), "-"]


def _run(argv, task, session, seconds, env):
    """Run the session and return True if it reached its time limit.

    shortcut: the session stays in the process group of the campaign, so the hard campaign limit also stops
    its child processes; the session limit stops only the agent process. Give the session its own process
    group if orphaned tool processes become a problem.
    """
    with (session / "events.jsonl").open("wb") as out, (session / "stderr.txt").open("wb") as err:
        # The session needs no LLM proxy key; keep it out of the reach of the agent shell.
        inherited = {key: value for key, value in os.environ.items() if key != "QOQ_PROXY_KEY"}
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=session,
                                   env={**inherited, **env})
        try:
            process.communicate(task.encode(), timeout=seconds)
            return False
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return True


def _keep_discoveries(session, path):
    """Append at most MAX_NOTES DISCOVERY lines of the session to the discoveries of the campaign."""
    try:
        with (session / NOTES).open("rb") as stream:
            text = stream.read(MAX_NOTES_READ).decode("utf-8", errors="replace")
    except OSError:
        return
    lines = [line.strip()[:MAX_NOTE_CHARS] for line in text.splitlines() if line.strip().startswith("DISCOVERY:")]
    if lines:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("".join(f"- {session.name}: {line}\n" for line in lines[:MAX_NOTES]))


def _discoveries(path):
    try:
        data = path.read_bytes()[-MAX_DISCOVERY_BYTES:].decode("utf-8", errors="replace")
    except OSError:
        return ""
    return f"## Discoveries of earlier sessions\n{data}\n"


class AgentClient:
    """OpenEvolve model client: each request runs one Codex agent session and returns its candidate."""

    def __init__(self, config):
        self.model = config.name
        self.settings = agent_settings()
        self.last_usage = None

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.last_usage = None
        campaign = Path(os.environ["QOQ_OE_CONFIG"]).parent
        (campaign / SESSIONS).mkdir(exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix="session_", dir=campaign / SESSIONS))
        parent = _CURRENT.search(messages[-1]["content"])
        if parent is not None:
            code = parent[1]
            (session / "parent_candidate.py").write_text(code, encoding="utf-8")
            source = builder_source(code)
            if source is not None:
                (session / BUILDER).write_text(source, encoding="utf-8")
        phase = _phase(campaign)
        task = TASK.format(minutes=max(1, self.settings["seconds"] // 60), candidate=CANDIDATE, notes=NOTES,
                           python=sys.executable, cache=campaign / CACHE_DIRECTORY,
                           phase=phase, target=campaign / "target.npy",
                           discoveries=_discoveries(campaign / DISCOVERIES), prompt=messages[-1]["content"])
        if os.environ.get("QOQ_VECTOR_ONLY"):
            task = task.replace("(logical_state, layout)", "(layout)")
            task += ("\nUse only the supplied target and results from this run."
                     " Do not read previous circuits, records, benchmarks or state construction functions.\n")
        (session / "TASK.md").write_text(task, encoding="utf-8")
        env = _tool_env()
        started = perf_counter()
        timed_out = _run(_argv(self.settings, session, env), task, session, self.settings["seconds"], env)
        self.last_usage = session_usage(session / "events.jsonl")
        (session / "usage.json").write_text(json.dumps({"usage": self.last_usage, "timed_out": timed_out,
            "seconds": perf_counter() - started, "phase": phase}), encoding="utf-8")
        _keep_discoveries(session, campaign / DISCOVERIES)
        candidate = session / CANDIDATE
        if not candidate.is_file():
            reason = "reached its time limit" if timed_out else "ended"
            raise RuntimeError(f"The agent session {session.name} {reason} without {CANDIDATE}")
        if candidate.stat().st_size > MAX_BYTES:
            raise RuntimeError(f"The {CANDIDATE} of agent session {session.name} is too large")
        code = candidate.read_text(encoding="utf-8")
        builder = session / BUILDER
        if not builder.is_file() or builder.is_symlink() or builder.stat().st_size > MAX_BUILDER_BYTES:
            raise RuntimeError("The session must supply a bounded builder.py file")
        code = with_builder(code, builder.read_text(encoding="utf-8"))
        return "```python\n" + code + "\n```"

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("", [{"role": "user", "content": prompt}])
