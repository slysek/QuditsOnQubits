"""Run one provider at a time and preserve its checkpoint."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path

from .protocol import RESULT_SCHEMA, ClaudeProtocol, CodexProtocol, Protocol
from .state import PROVIDERS, Store, atomic_json, choose_provider, positive_number, workspace_lock


class Child:
    """Keep provider output and terminate the provider process group."""

    def __init__(self, command: list[str], workspace: str, log: Path,
                 env: dict[str, str], lock_fd: int) -> None:
        self.selector = selectors.DefaultSelector()
        self.buffer = b""
        self.bytes_read = 0
        self.logs = {}
        self.process = None
        try:
            self.logs["stdout"] = log.open("wb")
            self.logs["stderr"] = log.with_suffix(".stderr").open("wb")
            self.process = subprocess.Popen(
                command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, pass_fds=(lock_fd,),
            )
            for name in self.logs:
                pipe = getattr(self.process, name)
                os.set_blocking(pipe.fileno(), False)
                self.selector.register(pipe, selectors.EVENT_READ, name)
        except BaseException:
            try:
                if self.process is not None:
                    self.signal(signal.SIGKILL)
                    self.process.wait(timeout=5)
            finally:
                self.close_streams()
            raise

    def close_streams(self) -> None:
        self.selector.close()
        for handle in self.logs.values():
            handle.close()
        if self.process is not None:
            for name in ("stdin", "stdout", "stderr"):
                getattr(self.process, name).close()

    def send(self, message: dict | None) -> None:
        if message is not None:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            self.process.stdin.flush()

    def poll(self, timeout: float = 0.2) -> list[dict]:
        events = []
        for key, _ in self.selector.select(timeout):
            data = os.read(key.fileobj.fileno(), 65536)
            if not data:
                self.selector.unregister(key.fileobj)
                if key.data == "stdout" and self.buffer.strip():
                    raise ValueError("The provider ended with an incomplete JSON event.")
                continue
            self.bytes_read += len(data)
            if self.bytes_read > 64 * 1024 * 1024:
                raise ValueError("The provider exceeded the 64 MiB output limit for one work period.")
            self.logs[key.data].write(data)
            self.logs[key.data].flush()
            if key.data != "stdout":
                continue
            self.buffer += data
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.strip():
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("The provider event must be a JSON object.")
                    events.append(event)
            if len(self.buffer) > 4 * 1024 * 1024:
                raise ValueError("The provider event exceeded 4 MiB.")
        return events

    def signal(self, number: int) -> None:
        try:
            os.killpg(self.process.pid, number)
        except ProcessLookupError:
            pass

    def close(self, grace: float) -> None:
        try:
            self.signal(signal.SIGINT)
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            # The group can contain teammates after the lead process exits.
            self.signal(signal.SIGKILL)
            self.process.wait(timeout=5)
        finally:
            self.close_streams()


def run_slice(store: Store, provider: str, deadline: float,
              stop: threading.Event, lock_fd: int) -> Protocol:
    config = store.config
    protocol = ClaudeProtocol() if provider == "claude" else CodexProtocol()
    session = store.state["sessions"][provider]
    prompt = store.prompt(provider, deadline)
    env = os.environ.copy()
    if provider == "claude":
        command = [*config.claude_command, "-p", "--verbose", "--input-format", "stream-json",
                   "--output-format", "stream-json", "--json-schema", json.dumps(RESULT_SCHEMA),
                   "--permission-mode", "auto", "--permission-prompts", "none"]
        if config.claude_teams:
            env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
            command += ["--teammate-mode", "in-process"]
        if session:
            command += ["--resume", session]
        first = {"type": "user", "message": {"role": "user", "content": prompt},
                 "parent_tool_use_id": None, "session_id": session or ""}
    else:
        command = [*config.codex_command, "app-server", "--listen", "stdio://"]
        first = protocol.start(Path(config.workspace), prompt, session)
    log = store.directory / "logs" / f"{store.state['sequence']:06d}-{provider}.jsonl"
    child = Child(command, config.workspace, log, env, lock_fd)
    store.state["active"] = {"provider": provider, "pid": child.process.pid, "log": str(log)}
    monotonic_end = time.monotonic() + max(0, deadline - time.time())
    nudge_at = monotonic_end - min(config.grace_seconds, max(0, deadline - time.time()) / 2)
    nudged = False
    try:
        store.save()
        child.send(first)
        while protocol.outcome is None:
            if stop.is_set() or (store.directory / "STOP").exists():
                protocol.outcome = "stopped"
                break
            now = time.monotonic()
            if now >= monotonic_end:
                protocol.outcome = "interrupted"
                break
            if isinstance(protocol, CodexProtocol) and not nudged and now >= nudge_at:
                child.send(protocol.steer())
                nudged = True
            for event in child.poll(min(0.2, monotonic_end - now)):
                for reply in protocol.receive(event):
                    child.send(reply)
                if protocol.session and protocol.session != store.state["sessions"][provider]:
                    store.state["sessions"][provider] = protocol.session
                    store.save()
                if protocol.outcome:
                    break
            if child.process.poll() is not None and not child.selector.get_map() and not protocol.outcome:
                protocol.fail(f"The provider exited without a result. Exit code: {child.process.returncode}.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        protocol.fail(f"The provider protocol failed: {exc}")
    finally:
        if isinstance(protocol, CodexProtocol):
            try:
                child.send(protocol.interrupt())
            except (OSError, ValueError):
                pass
        child.close(config.grace_seconds)
        store.state["active"] = None
        store.save()
    return protocol


def run(store: Store, hours: float, stop: threading.Event | None = None,
        *, execute=run_slice, wall_clock=time.time, monotonic=time.monotonic,
        wait=None) -> str:
    if not positive_number(hours):
        raise ValueError("The run duration must be a finite positive number of hours.")
    stop = stop or threading.Event()
    wait = wait or stop.wait
    with workspace_lock(Path(store.config.workspace)) as lock_fd:
        # Reload after the lock to avoid overwriting another supervisor's last state.
        store.state = Store.open(store.directory).state
        if store.state["status"] == "complete":
            return "complete"
        if (store.directory / "STOP").exists():
            raise ValueError("The STOP file exists. Remove it before you resume this run.")
        if store.state["status"] == "blocked":
            raise ValueError("The run is blocked. Read the logs and use the resume command after you correct the problem.")
        end = monotonic() + hours * 3600
        store.event("run_started", hours=hours)
        try:
            while monotonic() < end and not stop.is_set():
                if (store.directory / "STOP").exists():
                    break
                now = wall_clock()
                eligible = {name: store.state["cooldowns"][name]
                            if store.state["failures"][name] < store.config.max_failures
                            else now + max(1, end - monotonic()) for name in PROVIDERS}
                if all(store.state["failures"][name] >= store.config.max_failures for name in PROVIDERS):
                    store.state["status"] = "blocked"
                    store.event("run_blocked", reason="Both providers failed repeatedly.")
                    return "blocked"
                provider = choose_provider(eligible, now)
                if provider is None:
                    store.state["status"] = "waiting"
                    store.save()
                    delay = min(1, max(0, min(eligible.values()) - now), end - monotonic())
                    wait(delay)
                    continue
                deadline = now + min(store.config.slice_seconds, end - monotonic())
                if provider == "codex" and store.state["failures"]["claude"] < store.config.max_failures:
                    deadline = min(deadline, store.state["cooldowns"]["claude"])
                store.state["status"] = "running"
                store.state["sequence"] += 1
                store.save()
                store.event("provider_started", provider=provider, deadline=deadline)
                try:
                    result = execute(store, provider, deadline, stop, lock_fd)
                except OSError as exc:
                    result = Protocol()
                    result.fail(f"The provider could not start: {exc}")
                now = wall_clock()
                if result.session:
                    store.state["sessions"][provider] = result.session
                if result.checkpoint:
                    checkpoint = f"checkpoints/{store.state['sequence']:06d}-{provider}.json"
                    atomic_json(store.directory / checkpoint, result.checkpoint)
                    store.state["last_checkpoint"] = checkpoint
                store.state["last_error"] = result.error
                store.event("provider_finished", provider=provider, outcome=result.outcome,
                            error=result.error, reset_at=result.reset_at)
                if result.outcome == "limited":
                    reset = result.reset_at if result.reset_at > now else now + store.config.retry_seconds
                    store.state["cooldowns"][provider] = reset + store.config.reset_padding
                    store.state["failures"][provider] = 0
                elif result.outcome == "failed":
                    store.state["failures"][provider] += 1
                    store.state["cooldowns"][provider] = now + store.config.retry_seconds
                elif result.outcome in ("complete", "blocked"):
                    store.state["status"] = result.outcome
                    return result.outcome
                elif result.outcome == "stopped":
                    break
                else:
                    store.state["failures"][provider] = 0
                store.save()
                # Prevent a broken or empty provider from creating a tight process loop.
                wait(min(1, max(0, end - monotonic())))
            store.state["status"] = "paused"
            store.event("run_paused")
            return "paused"
        finally:
            if store.state["status"] in ("running", "waiting"):
                store.state["status"] = "paused"
            store.save()
