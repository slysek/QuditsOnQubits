"""Stage events and periodic, log-friendly progress for long candidate jobs."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import queue
import multiprocessing
import sys
import threading
import time

import psutil


_REPORTER = ContextVar("candidate_stage_reporter", default=None)


@contextmanager
def stage_scope(reporter):
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


def report_stage(stage: str, completed: int | None = None):
    reporter = _REPORTER.get()
    if reporter is not None:
        reporter(stage, completed)


@contextmanager
def monitor_progress(interval, total_candidates, *, parallel):
    if not interval or not total_candidates:
        yield None
        return
    events = multiprocessing.get_context("spawn").Queue() if parallel else queue.Queue()
    try:
        with ProgressMonitor(events, interval=interval, total_candidates=total_candidates):
            yield events
    finally:
        if parallel:
            events.close()
            events.join_thread()


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _single_line(value):
    return "".join(char if char.isprintable() else " " for char in str(value))


class ProcessActivity:
    """Observe CPU consumption, including BQSKit children, without guessing health."""

    def __init__(self, pid):
        self.pid = pid
        self.previous = None
        self.root_created = None

    def sample(self):
        try:
            root = psutil.Process(self.pid)
            created = root.create_time()
            if self.root_created is not None and self.root_created != created:
                return "process exited (PID reused)"
            self.root_created = created
            if not root.is_running() or root.status() == psutil.STATUS_ZOMBIE:
                return "process exited"
            processes = [root, *root.children(recursive=True)]
            current = {}
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    current[(process.pid, process.create_time())] = cpu.user + cpu.system
                except psutil.NoSuchProcess:
                    continue
            if self.previous is None:
                activity = "CPU sampling"
            else:
                delta = sum(max(0.0, cpu - self.previous.get(key, 0.0)) for key, cpu in current.items())
                activity = f"CPU +{delta:.1f}s since last display"
                if delta < 0.01:
                    activity += " (no CPU activity observed; may be waiting)"
            self.previous = current
            return f"{len(current)} processes | {activity}"
        except psutil.NoSuchProcess:
            return "process exited"
        except psutil.Error:
            return "CPU activity unavailable"


@dataclass
class CandidateProgress:
    label: str
    pid: int
    started: float
    total: int
    stage: str = "starting"
    completed: int = 0
    stage_started: float = field(init=False)
    sampler: ProcessActivity = field(init=False)

    def __post_init__(self):
        self.stage_started = self.started
        self.sampler = ProcessActivity(self.pid)

    def update(self, stage, now, completed=None):
        if stage != self.stage:
            self.stage_started = now
        self.stage = stage
        if completed is not None:
            self.completed = completed

    def render(self, now, activity=None):
        activity = self.sampler.sample() if activity is None else activity
        if self.total and (self.completed or self.stage.startswith("transpiling")):
            fraction = min(1.0, self.completed / self.total)
            filled = int(20 * fraction)
            bar = "#" * filled + "-" * (20 - filled)
            progress = f"[{bar}] {self.completed}/{self.total} trials, {fraction:.0%}, {self.total - self.completed} remaining"
        else:
            progress = f"[....................] ETA unknown | {self.completed}/{self.total} trials recorded"
        return (
            f"{_single_line(self.label)} | PID {self.pid} | elapsed {_duration(now - self.started)}\n"
            f"  {_single_line(self.stage)} | stage elapsed {_duration(now - self.stage_started)}\n"
            f"  {progress} | {activity}"
        )


class ProgressMonitor:
    """One parent-owned output stream; heartbeat runs even without stage events."""

    def __init__(self, events, *, interval, total_candidates, stream=None):
        self.events = events
        self.interval = interval
        self.total_candidates = total_candidates
        self.stream = sys.stderr if stream is None else stream
        self.active = {}
        self.finished = 0
        self.thread = threading.Thread(target=self._run, name="candidate-progress", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.events.put(None)
        self.thread.join()

    def _write(self, text):
        try:
            self.stream.write(text + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            pass  # A closed output stream must not abort synthesis or exports.

    def _handle(self, event):
        key = event["key"]
        if event["kind"] == "start":
            progress = CandidateProgress(event["label"], event["pid"], event["at"], event["total"])
            self.active[key] = progress
            self._write("[progress] " + progress.render(time.monotonic()))
        elif event["kind"] == "stage" and key in self.active:
            self.active[key].update(event["stage"], event["at"], event["completed"])
        elif event["kind"] == "finish":
            progress = self.active.pop(key, None)
            if progress is not None:
                self.finished += 1
                self._write(
                    f"[progress] {_single_line(progress.label)} | {event['status']} | "
                    f"elapsed {_duration(event['at'] - progress.started)} | "
                    f"trials {progress.completed}/{progress.total}"
                )

    def _run(self):
        next_display = time.monotonic() + self.interval
        while True:
            try:
                event = self.events.get(timeout=max(0.001, next_display - time.monotonic()))
            except queue.Empty:
                event = {}
            if event is None:
                return
            if event:
                self._handle(event)
            now = time.monotonic()
            if now >= next_display:
                if self.active:
                    lines = [f"[progress] finished {self.finished}/{self.total_candidates} | active {len(self.active)}"]
                    lines.extend(progress.render(now) for progress in self.active.values())
                    self._write("\n".join(lines))
                elif self.finished < self.total_candidates:
                    self._write(f"[progress] finished {self.finished}/{self.total_candidates} | waiting for worker start/results")
                next_display = now + self.interval
