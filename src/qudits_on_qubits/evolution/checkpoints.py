"""Save operation settings and resume completed work after an interruption."""

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
from time import monotonic
from typing import Iterator

from filelock import FileLock, Timeout

AGENT_ENV = ("QOQ_AGENT_BINARY", "QOQ_AGENT_MODEL", "QOQ_AGENT_EFFORT", "QOQ_AGENT_FAST", "QOQ_AGENT_SECONDS")
PATH_OPTIONS = ("baseline", "output", "model_config", "codex_auth_file", "prompt_notes", "initial_program",
                "narrator_model_config", "ideation_model_config")


def save(path: Path, document: dict, *, sort_keys: bool = False) -> None:
    """Replace a JSON file after all bytes reach the disk."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=sort_keys, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _options(parameters: dict) -> dict:
    options = {key: value for key, value in parameters.items() if key not in ("resume", "console_log")}
    for key in PATH_OPTIONS:
        if options.get(key) is not None:
            options[key] = str(Path(options[key]).resolve())
    if "baselines" in options:
        options["baselines"] = {state: str(Path(path).resolve()) for state, path in options["baselines"].items()}
    if "warm_starts" in options:
        options["warm_starts"] = [str(Path(path).resolve()) for path in options["warm_starts"]]
    if options.get("record_store") is not None:
        store = options["record_store"]
        options["record_store"] = (None if options.get("blank_start") else
                                   {key: str(getattr(store, key).resolve()) for key in ("root", "archive")})
    return json.loads(json.dumps(options))


def load(output: Path, kind: str | None = None) -> dict:
    """Read saved settings. Reject unsupported operations and invalid fields."""
    try:
        document = json.loads((output / "resume.json").read_text(encoding="utf-8"))
        if (not isinstance(document, dict) or document.get("version") != 1
                or document.get("kind") not in ("run", "discover", "campaign")
                or (kind is not None and document["kind"] != kind)
                or not isinstance(document.get("options"), dict)):
            raise ValueError
        options = document["options"]
        if any(options.get(key) is not None and not isinstance(options[key], str) for key in PATH_OPTIONS):
            raise ValueError
        if "baselines" in options and (not isinstance(options["baselines"], dict)
                                      or any(not isinstance(path, str) for path in options["baselines"].values())):
            raise ValueError
        for elapsed in (document["elapsed_seconds"], document.get("ideation_seconds", 0)):
            if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError
        environment = document["environment"]
        if (not isinstance(environment, dict) or set(environment) != set(AGENT_ENV)
                or any(value is not None and not isinstance(value, str) for value in environment.values())
                or Path(document["options"]["output"]).resolve() != output.resolve()):
            raise ValueError
        return document
    except (KeyError, TypeError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("The saved operation settings are invalid") from exc


def history_directory() -> Path:
    """Return the directory for saved session paths."""
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "qoq-evolution/sessions"


def saved_sessions(root: Path) -> list[dict]:
    """List valid sessions by update time. Include existing runs in the default output directory."""
    outputs = {path.parent.resolve() for path in root.rglob("resume.json")}
    for entry in history_directory().glob("*.json"):
        try:
            document = json.loads(entry.read_text(encoding="utf-8"))
            output = document["output"]
            if isinstance(output, str) and Path(output).is_absolute():
                outputs.add(Path(output).resolve())
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            continue
    sessions = {}
    for output in outputs:
        try:
            sessions[output] = {**load(output), "output": output,
                                "modified": (output / "resume.json").stat().st_mtime}
        except (OSError, ValueError):
            continue
    return sorted((session for output, session in sessions.items()
                   if not any(parent in sessions for parent in output.parents)),
                  key=lambda session: (session["modified"], str(session["output"])), reverse=True)


@contextmanager
def saved_run(output: Path, kind: str, parameters: dict, resume: bool = False) -> Iterator[dict]:
    """Save settings and active time. Hold the directory lock until the operation stops."""
    options = _options(parameters)
    if resume and not output.is_dir():
        raise FileNotFoundError("The saved output directory does not exist")
    output.mkdir(parents=True, exist_ok=resume)
    lock = FileLock(output / ".run.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ValueError("This operation is already running") from exc
    try:
        document = load(output, kind) if resume else {
            "version": 1, "kind": kind, "options": options, "elapsed_seconds": 0,
            "environment": {key: os.environ.get(key) for key in AGENT_ENV}}
        if document["options"] != options:
            raise ValueError("Resume with the saved operation settings")
        started, previous = monotonic(), document["elapsed_seconds"]
        document["status"] = "running"
        save(output / "resume.json", document)
        try:
            history = history_directory()
            history.mkdir(mode=0o700, parents=True, exist_ok=True)
            key = hashlib.sha256(os.fsencode(output.resolve())).hexdigest()
            save(history / f"{key}.json", {"output": str(output.resolve())})
        except OSError as exc:
            print(f"Cannot save session history: {exc}", flush=True)
        if kind == "campaign":
            (output / "pause.request").unlink(missing_ok=True)
        try:
            yield document
        except KeyboardInterrupt:
            document["status"] = "paused"
            raise
        except BaseException:
            document["status"] = "failed"
            raise
        else:
            document["status"] = "completed"
        finally:
            document["elapsed_seconds"] = previous + monotonic() - started
            save(output / "resume.json", document)
            if document["status"] == "paused":
                print(f"\nPaused. Resume with: qoq-evolution resume {shlex.quote(str(output.resolve()))}", flush=True)
    finally:
        lock.release()


def resume_operation(output: Path | str) -> dict | int:
    """Resume one supported operation with its saved settings and agent environment."""
    output = Path(output).resolve()
    document = load(output)
    options = dict(document["options"])
    if document["kind"] == "run":
        from .harness import run_batch as run
    elif document["kind"] == "discover":
        from .discovery import run_discovery as run
        from .records import RecordStore
        store = options.get("record_store")
        if store is not None:
            if (not isinstance(store, dict) or set(store) != {"root", "archive"}
                    or any(not isinstance(value, str) for value in store.values())):
                raise ValueError("The saved record directories are invalid")
            options["record_store"] = RecordStore(**{key: Path(value) for key, value in store.items()})
    else:
        from .oe_run import run_campaign as run
    try:
        inspect.signature(run).bind(**options, resume=True)
    except TypeError as exc:
        raise ValueError("The saved operation options are invalid") from exc
    previous = {key: os.environ.get(key) for key in AGENT_ENV}
    try:
        for key, value in document["environment"].items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return run(**options, resume=True)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def latest_checkpoint(output: Path, *, update_metadata: bool = True) -> tuple[Path | None, int]:
    """Select the newest complete population. Ignore an interrupted checkpoint write.

    Set update_metadata=False to keep the checkpoint files unchanged.
    """
    checkpoints = list((output / "openevolve/checkpoints").glob("checkpoint_*"))
    complete = []
    for path in sorted(checkpoints, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        try:
            metadata = json.loads((path / "metadata.json").read_text())
            iteration = metadata["last_iteration"]
            if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
                continue
            members = {member for island in metadata["islands"] for member in island}
            members.add(metadata["best_program_id"])
            for member in members:
                if not isinstance(member, str) or Path(member).name != member:
                    raise ValueError
                program = json.loads((path / "programs" / f"{member}.json").read_text())
                if program["id"] != member:
                    raise ValueError
            complete.append((path, metadata))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if complete:
        path, metadata = complete[0]
        # Parallel iterations can finish in a different order. Continue after the largest saved number.
        last = max(item[1]["last_iteration"] for item in complete)
        if update_metadata and metadata["last_iteration"] != last:
            save(path / "metadata.json", {**metadata, "last_iteration": last})
        return path, len(complete)
    if checkpoints:
        raise ValueError("The campaign has no complete population checkpoint")
    return None, 0
