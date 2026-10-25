"""Control the local research relay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .runner import run
from .state import Config, Store, workspace_lock


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Transfer research work between Claude Code and Codex.")
    commands = root.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Create a run without starting an agent.")
    initialize.add_argument("--workspace", type=Path, default=Path.cwd())
    initialize.add_argument("--run-dir", type=Path, required=True)
    initialize.add_argument("--goal-file", type=Path, required=True)
    initialize.add_argument("--slice-minutes", type=float, default=10)
    initialize.add_argument("--retry-minutes", type=float, default=15)
    initialize.add_argument("--claude-bin", default="claude")
    initialize.add_argument("--codex-bin", default="codex")
    initialize.add_argument("--no-claude-teams", action="store_true")
    for command in ("run", "resume", "status", "stop", "doctor"):
        item = commands.add_parser(command)
        item.add_argument("--run-dir", type=Path, required=True)
        if command in ("run", "resume"):
            item.add_argument("--hours", type=float, default=8)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if os.name != "posix":
        print("The relay requires Linux or WSL.", file=sys.stderr)
        return 2
    try:
        if args.command == "init":
            config = Config(workspace=str(args.workspace), slice_seconds=args.slice_minutes * 60,
                            retry_seconds=args.retry_minutes * 60,
                            claude_command=(args.claude_bin,), codex_command=(args.codex_bin,),
                            claude_teams=not args.no_claude_teams)
            store = Store.create(args.run_dir, config, args.goal_file.read_text())
            print(f"Run created: {store.directory}")
            return 0
        store = Store.open(args.run_dir)
        if args.command == "status":
            print(json.dumps(store.state, indent=2))
            return 0
        if args.command == "stop":
            (store.directory / "STOP").touch(mode=0o600)
            print("Stop requested. The active provider will stop before another provider starts.")
            return 0
        if args.command == "doctor":
            for provider in ("claude", "codex"):
                command = getattr(store.config, f"{provider}_command")
                if not shutil.which(command[0]):
                    raise ValueError(f"The {provider} command is not available: {command[0]}")
                result = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise ValueError(f"The {provider} version check failed.")
                print(f"{provider}: {result.stdout.strip()}")
            with workspace_lock(Path(store.config.workspace)):
                pass
            print("Local checks passed. No model request was sent. Account access is not checked.")
            return 0
        if args.command == "resume":
            with workspace_lock(Path(store.config.workspace)):
                store = Store.open(args.run_dir)
                if store.state["status"] == "complete":
                    raise ValueError("The goal is complete. Create a new run for another goal.")
                (store.directory / "STOP").unlink(missing_ok=True)
                store.state["status"] = "paused"
                store.state["failures"] = {"claude": 0, "codex": 0}
                store.save()
        stop = threading.Event()
        old_handlers = {}
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, lambda *_: stop.set())
        try:
            return 2 if run(store, args.hours, stop) == "blocked" else 0
        finally:
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Relay error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
