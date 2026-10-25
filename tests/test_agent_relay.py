from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.agent_relay.protocol import ClaudeProtocol, CodexProtocol, quota_reset
from tools.agent_relay.protocol import Protocol
from tools.agent_relay.runner import Child, run, run_slice
from tools.agent_relay.__main__ import main
from tools.agent_relay.state import Config, Store, choose_provider, workspace_lock

pytestmark = pytest.mark.skipif(os.name != "posix", reason="The relay requires Linux or WSL.")
FAKE = Path(__file__).parent / "fixtures" / "relay_provider.py"


def make_store(tmp_path, claude="limit", codex="complete", **options):
    config = Config(
        workspace=str(tmp_path), grace_seconds=0.05, reset_padding=0.05,
        claude_command=(sys.executable, str(FAKE), "claude", claude),
        codex_command=(sys.executable, str(FAKE), "codex", codex), **options,
    )
    return Store.create(tmp_path / "artifacts" / "night", config, "Check the relay.")


def test_claude_quota_uses_reported_reset_and_ignores_normal_text():
    protocol = ClaudeProtocol()
    protocol.receive({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "The paper discusses rate limits."}
    ]}})
    assert protocol.outcome is None
    protocol.receive({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed_warning", "resetsAt": 2000
    }})
    assert protocol.outcome is None
    protocol.receive({"type": "rate_limit_event", "rate_limit_info": {
        "status": "rejected", "resetsAt": 2000
    }})
    assert protocol.outcome == "limited"
    assert protocol.reset_at == 2000


def test_claude_result_requires_current_structured_output():
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "is_error": False, "result": "Done."})
    assert protocol.outcome == "failed"
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "is_error": False, "structured_output": {
        "status": "complete", "summary": "The checks passed.", "next_steps": []
    }})
    assert protocol.outcome == "complete"


def test_codex_quota_waits_for_all_exhausted_windows():
    payload = {"rateLimits": {"primary": {"usedPercent": 100, "resetsAt": 2000},
                              "secondary": {"usedPercent": 100, "resetsAt": 9000}}}
    assert quota_reset(payload) == 9000
    payload["rateLimits"]["secondary"]["usedPercent"] = 25
    assert quota_reset(payload) == 2000
    payload["rateLimits"]["primary"]["usedPercent"] = 25
    assert quota_reset(payload) is None


def test_codex_protocol_initializes_then_resumes_and_starts_turn():
    protocol = CodexProtocol()
    first = protocol.start(Path("/tmp"), "Continue.", "thread-1")
    assert first["method"] == "initialize"
    requests = protocol.receive({"id": first["id"], "result": {}})
    assert requests[0]["method"] == "initialized"
    assert requests[1]["method"] == "account/rateLimits/read"
    requests = protocol.receive({"id": requests[1]["id"], "result": {
        "rateLimits": {"primary": None, "secondary": None}
    }})
    assert requests[0]["method"] == "thread/resume"
    assert requests[0]["params"]["threadId"] == "thread-1"
    requests = protocol.receive({"id": requests[0]["id"], "result": {
        "thread": {"id": "thread-1"}
    }})
    assert requests[0]["method"] == "turn/start"
    assert requests[0]["params"]["input"][0]["text"] == "Continue."
    assert requests[0]["params"]["outputSchema"]["required"]


def test_provider_selection_prefers_claude_and_waits_for_both():
    assert choose_provider({"claude": 100, "codex": 0}, 50) == "codex"
    assert choose_provider({"claude": 100, "codex": 200}, 50) is None
    assert choose_provider({"claude": 100, "codex": 200}, 100) == "claude"


def test_store_persists_state_and_refuses_overwrite(tmp_path):
    run = tmp_path / "artifacts" / "night"
    config = Config(workspace=str(tmp_path), slice_seconds=60)
    store = Store.create(run, config, "Compare the two methods.")
    store.state["cooldowns"]["claude"] = 1234
    store.save()
    assert Store.open(run).state["cooldowns"]["claude"] == 1234
    assert "Compare the two methods." in (run / "GOAL.md").read_text()
    with pytest.raises(ValueError):
        Store.create(run, config, "Replace the goal.")


def test_workspace_lock_prevents_two_runs(tmp_path):
    with workspace_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already"):
            with workspace_lock(tmp_path):
                pytest.fail("A second supervisor acquired the workspace lock.")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_config_rejects_invalid_slice_duration(value, tmp_path):
    with pytest.raises(ValueError):
        Config(workspace=str(tmp_path), slice_seconds=value)


def test_run_directory_must_be_in_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError):
        Store.create(tmp_path / "elsewhere", Config(workspace=str(workspace)), "Research.")


class Clock:
    def __init__(self):
        self.value = 1000.0

    def now(self):
        return self.value

    def wait(self, seconds):
        self.value += seconds


def test_scheduler_returns_to_claude_when_its_limit_resets(tmp_path):
    store = make_store(tmp_path, slice_seconds=60)
    clock = Clock()
    calls = []

    def execute(store, provider, deadline, stop, lock_fd):
        calls.append((provider, deadline))
        result = Protocol()
        if len(calls) == 1:
            result.outcome = "limited"
            result.reset_at = 1010
        elif len(calls) == 2:
            assert deadline == 1010.05
            clock.value = deadline
            result.outcome = "interrupted"
        else:
            result.result({"status": "complete", "summary": "The goal is complete.", "next_steps": []})
        return result

    assert run(store, 1, execute=execute, wall_clock=clock.now, monotonic=clock.now, wait=clock.wait) == "complete"
    assert [name for name, _ in calls] == ["claude", "codex", "claude"]
    assert Store.open(store.directory).state["status"] == "complete"


def test_scheduler_waits_when_both_providers_are_limited(tmp_path):
    store = make_store(tmp_path)
    clock = Clock()
    calls = []

    def execute(store, provider, deadline, stop, lock_fd):
        calls.append((provider, clock.now()))
        result = Protocol()
        if len(calls) <= 2:
            result.outcome = "limited"
            result.reset_at = 1010 if provider == "claude" else 1020
        else:
            result.result({"status": "complete", "summary": "The goal is complete.", "next_steps": []})
        return result

    run(store, 1, execute=execute, wall_clock=clock.now, monotonic=clock.now, wait=clock.wait)
    assert calls[2][0] == "claude"
    assert calls[2][1] >= 1010.05


def test_scheduler_stops_after_repeated_failures(tmp_path):
    store = make_store(tmp_path, max_failures=1, retry_seconds=1)
    clock = Clock()

    def execute(*_):
        raise FileNotFoundError("The test binary is absent.")

    assert run(store, 1, execute=execute, wall_clock=clock.now, monotonic=clock.now, wait=clock.wait) == "blocked"
    with pytest.raises(ValueError, match="blocked"):
        run(Store.open(store.directory), 1)


def test_scheduler_unknown_reset_uses_backoff_and_preserves_it_on_restart(tmp_path):
    store = make_store(tmp_path, retry_seconds=100)
    clock = Clock()

    def execute(*_):
        result = Protocol()
        result.outcome = "limited"
        return result

    assert run(store, 3 / 3600, execute=execute, wall_clock=clock.now, monotonic=clock.now, wait=clock.wait) == "paused"
    saved = Store.open(store.directory)
    assert saved.state["cooldowns"]["claude"] == 1100.05
    assert saved.state["cooldowns"]["codex"] == 1101.05
    assert saved.state["status"] == "paused"


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_real_subprocess_protocol_completes_and_saves_session(tmp_path, provider):
    store = make_store(tmp_path, claude="complete")
    with workspace_lock(tmp_path) as lock_fd:
        result = run_slice(store, provider, time.time() + 5, threading.Event(), lock_fd)
    assert result.outcome == "complete", result.error
    assert store.state["sessions"][provider] == provider + "-test"
    assert store.state["active"] is None
    assert list((store.directory / "logs").glob("*.jsonl"))


def test_subprocess_handoff_cycle(tmp_path):
    store = make_store(tmp_path)
    # Keep Claude unavailable long enough for Codex to return its result.
    store.config = Config(**{**store.config.__dict__, "reset_padding": 2})
    assert run(store, 0.01) == "complete"
    events = [json.loads(line) for line in (store.directory / "events.jsonl").read_text().splitlines()]
    started = [event["provider"] for event in events if event["event"] == "provider_started"]
    assert started == ["claude", "codex"]
    assert list((store.directory / "checkpoints").glob("*-codex.json"))


def test_subprocess_full_cycle_returns_to_claude(tmp_path):
    store = make_store(tmp_path, claude="limit-once", codex="wait")
    assert run(store, 0.01) == "complete"
    events = [json.loads(line) for line in (store.directory / "events.jsonl").read_text().splitlines()]
    started = [event["provider"] for event in events if event["event"] == "provider_started"]
    assert started == ["claude", "codex", "claude"]
    assert store.state["sessions"] == {"claude": "claude-test", "codex": "codex-test"}


@pytest.mark.parametrize("behavior", ["bad-json", "crash"])
def test_provider_protocol_errors_are_explicit(tmp_path, behavior):
    store = make_store(tmp_path, claude=behavior)
    with workspace_lock(tmp_path) as fd:
        result = run_slice(store, "claude", time.time() + 5, threading.Event(), fd)
    assert result.outcome == "failed"
    assert result.error


def test_deadline_kills_provider_and_descendant(tmp_path):
    store = make_store(tmp_path, claude="hang")
    start = time.monotonic()
    with workspace_lock(tmp_path) as fd:
        result = run_slice(store, "claude", time.time() + 0.3, threading.Event(), fd)
    assert result.outcome == "interrupted"
    assert time.monotonic() - start < 4
    pid = int((tmp_path / "relay-descendant.pid").read_text())
    # A killed orphan can remain a zombie until the host reaps it.
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        assert stat.read_text().split()[2] == "Z"
    with workspace_lock(tmp_path):
        pass


def test_codex_deadline_interrupts_turn(tmp_path):
    store = make_store(tmp_path, codex="wait")
    with workspace_lock(tmp_path) as fd:
        result = run_slice(store, "codex", time.time() + 0.3, threading.Event(), fd)
    assert result.outcome == "interrupted"


def test_stop_file_prevents_restart_until_explicit_resume(tmp_path):
    store = make_store(tmp_path, claude="complete")
    assert main(["stop", "--run-dir", str(store.directory)]) == 0
    with pytest.raises(ValueError, match="STOP"):
        run(store, 1)
    assert main(["resume", "--run-dir", str(store.directory), "--hours", "0.01"]) == 0
    assert Store.open(store.directory).state["status"] == "complete"
    assert main(["resume", "--run-dir", str(store.directory)]) == 2


def test_cli_init_status_doctor_and_overwrite_guard(tmp_path, capsys):
    goal = tmp_path / "goal.md"
    goal.write_text("Compare the two methods.")
    directory = tmp_path / "artifacts" / "night"
    args = ["init", "--workspace", str(tmp_path), "--run-dir", str(directory), "--goal-file", str(goal)]
    assert main(args) == 0
    assert main(args) == 2
    assert main(["status", "--run-dir", str(directory)]) == 0
    assert '"status": "ready"' in capsys.readouterr().out
    store = Store.open(directory)
    config = json.loads((directory / "config.json").read_text())
    config["claude_command"] = [sys.executable, str(FAKE), "claude", "complete"]
    config["codex_command"] = [sys.executable, str(FAKE), "codex", "complete"]
    (directory / "config.json").write_text(json.dumps(config))
    assert main(["doctor", "--run-dir", str(directory)]) == 0
    assert main(["run", "--run-dir", str(directory), "--hours", "0.01"]) == 0
    assert main(["run", "--run-dir", str(directory), "--hours", "0.01"]) == 0


def test_codex_interactive_request_blocks_without_approving():
    protocol = CodexProtocol()
    response = protocol.receive({"id": 55, "method": "item/commandExecution/requestApproval", "params": {}})
    assert protocol.outcome == "blocked"
    assert "error" in response[0]
    assert "result" not in response[0]


@pytest.mark.parametrize("code", ["tooManyDenials", "cyberPolicy", "misalignmentPolicyViolation"])
def test_codex_terminal_denial_blocks_and_preserves_error(code):
    protocol = CodexProtocol()
    protocol.session = "main"
    protocol.turn = "main-turn"
    protocol.receive({"method": "turn/completed", "params": {"threadId": "main", "turn": {
        "id": "main-turn", "status": "failed", "error": {
            "codexErrorInfo": code, "message": "The required action was denied."
        }
    }}})
    assert protocol.outcome == "blocked"
    assert code in protocol.error
    assert "The required action was denied." in protocol.error
    assert protocol.checkpoint is None


@pytest.mark.parametrize("code", ["tooManyDenials", "cyberPolicy", "misalignmentPolicyViolation"])
def test_scheduler_terminal_denial_prevents_retry_and_transfer(tmp_path, code):
    store = make_store(tmp_path)
    clock = Clock()
    store.state["cooldowns"]["claude"] = clock.now() + 60
    store.save()
    calls = []

    def execute(store, provider, deadline, stop, lock_fd):
        calls.append(provider)
        if provider == "codex":
            clock.value = deadline
            protocol = CodexProtocol()
            protocol.session = "codex-main"
            protocol.turn = "codex-turn"
            protocol.receive({"method": "turn/completed", "params": {
                "threadId": "codex-main", "turn": {
                    "id": "codex-turn", "status": "failed", "error": {
                        "codexErrorInfo": code, "message": "The required action was denied."
                    }
                }
            }})
            return protocol
        protocol = Protocol()
        protocol.result({"status": "complete", "summary": "The test goal is complete.", "next_steps": []})
        return protocol

    assert run(store, 1, execute=execute, wall_clock=clock.now, monotonic=clock.now,
               wait=clock.wait) == "blocked"
    assert calls == ["codex"]
    saved = Store.open(store.directory)
    assert saved.state["status"] == "blocked"
    assert saved.state["sequence"] == 1
    assert saved.state["sessions"]["claude"] is None
    assert saved.state["failures"] == {"claude": 0, "codex": 0}
    assert code in saved.state["last_error"]


@pytest.mark.parametrize("result", [None, {}, {"status": "complete", "summary": "", "next_steps": []},
                                   {"status": "complete", "summary": "ok", "next_steps": [1]}])
def test_invalid_results_do_not_complete_a_goal(result):
    protocol = Protocol()
    protocol.result(result)
    assert protocol.outcome == "failed"


def test_codex_failed_turn_distinguishes_quota_from_authentication():
    for info, expected in [("usageLimitExceeded", "limited"), ("rateLimitExceeded", "limited"),
                           ("unauthorized", "failed")]:
        protocol = CodexProtocol()
        protocol.session = "main"
        protocol.turn = "main-turn"
        protocol.receive({"method": "turn/completed", "params": {"threadId": "main", "turn": {
            "id": "main-turn", "status": "failed", "error": {"codexErrorInfo": info}
        }}})
        assert protocol.outcome == expected


def test_codex_missing_quota_endpoint_still_starts_thread():
    protocol = CodexProtocol()
    first = protocol.start(Path("/tmp"), "Continue.", None)
    request = protocol.receive({"id": first["id"], "result": {}})[1]
    response = protocol.receive({"id": request["id"], "error": {"code": -1, "message": "Unavailable"}})
    assert response[0]["method"] == "thread/start"
    protocol.receive({"id": response[0]["id"], "error": {"code": -1, "message": "Authentication failed."}})
    assert protocol.outcome == "failed"


def test_codex_quota_notification_does_not_require_a_failed_turn():
    protocol = CodexProtocol()
    protocol.receive({"method": "account/rateLimits/updated", "params": {
        "rateLimits": {"primary": {"usedPercent": 100, "resetsAt": 10000}}
    }})
    assert protocol.outcome == "limited"
    assert protocol.reset_at == 10000


def test_claude_permission_denial_and_errors():
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "permission_denials": [{"tool_name": "Bash"}]})
    assert protocol.outcome == "blocked"
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "is_error": True, "errors": ["Authentication failed."]})
    assert protocol.outcome == "failed"
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "is_error": True, "errors": ["You've hit your limit."]})
    assert protocol.outcome == "limited"


@pytest.mark.parametrize("status", ["complete", "continue", "blocked"])
def test_claude_historical_denial_preserves_valid_final_checkpoint(status):
    checkpoint = {"status": status, "summary": "The agent used a permitted tool.",
                  "next_steps": [] if status == "complete" else ["Check the next task."]}
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "subtype": "success", "is_error": False,
                      "permission_denials": [{"tool_name": "Bash", "tool_use_id": "denied-attempt"}],
                      "structured_output": checkpoint})
    assert protocol.outcome == status
    assert protocol.checkpoint == checkpoint
    if status != "blocked":
        assert protocol.error == ""


def test_claude_failed_result_with_permission_denial_does_not_trigger_fallback():
    protocol = ClaudeProtocol()
    protocol.receive({"type": "result", "is_error": True,
                      "permission_denials": [{"tool_name": "Bash"}],
                      "errors": ["The required action needs permission."]})
    assert protocol.outcome == "blocked"
    assert "permission" in protocol.error.lower()


@pytest.mark.parametrize("behavior,expected", [("complete-after-denial", "complete"),
                                               ("blocked-after-denial", "blocked")])
def test_subprocess_denial_history_preserves_checkpoint_and_never_bypasses(tmp_path, behavior, expected):
    store = make_store(tmp_path, claude=behavior)
    assert run(store, 0.01) == expected
    saved = Store.open(store.directory)
    assert saved.state["status"] == expected
    assert saved.state["sequence"] == 1
    assert saved.state["sessions"]["codex"] is None
    checkpoint = json.loads((store.directory / saved.state["last_checkpoint"]).read_text())
    assert checkpoint["status"] == expected
    events = [json.loads(line) for line in (store.directory / "logs" / "000001-claude.jsonl").read_text().splitlines()]
    assert events[-1]["permission_denials"][0]["tool_name"] == "Bash"


def test_stop_event_interrupts_an_active_provider(tmp_path):
    store = make_store(tmp_path, codex="wait")
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    try:
        with workspace_lock(tmp_path) as fd:
            result = run_slice(store, "codex", time.time() + 5, stop, fd)
        assert result.outcome == "stopped"
    finally:
        timer.join()


def test_corrupt_state_and_invalid_config_are_rejected(tmp_path):
    store = make_store(tmp_path)
    for key, value in [("cooldowns", {"claude": float("nan"), "codex": 0}),
                       ("sequence", -1), ("version", 99)]:
        state = {**store.state, key: value}
        (store.directory / "state.json").write_text(json.dumps(state))
        with pytest.raises(ValueError):
            Store.open(store.directory)
    with pytest.raises(ValueError):
        Config(workspace=str(tmp_path), claude_command="claude")
    with pytest.raises(ValueError):
        Config(workspace=str(tmp_path), max_failures=0)
    with pytest.raises(ValueError):
        Store.create(tmp_path / "empty", store.config, " ")


@pytest.mark.parametrize("thread_id,turn_id", [("worker", "worker-turn"), ("main", "old-turn")])
def test_codex_ignores_other_threads_and_old_turns(thread_id, turn_id):
    protocol = CodexProtocol()
    protocol.session = "main"
    protocol.turn = "main-turn"
    protocol.receive({"method": "turn/started", "params": {
        "threadId": thread_id, "turn": {"id": turn_id, "status": "inProgress"}
    }})
    protocol.receive({"method": "item/completed", "params": {
        "threadId": thread_id, "turnId": turn_id, "item": {"type": "agentMessage", "text": json.dumps({
            "status": "complete", "summary": "The worker task is complete.", "next_steps": []
        })}
    }})
    protocol.receive({"method": "turn/completed", "params": {
        "threadId": thread_id, "turn": {"id": turn_id, "status": "completed"}
    }})
    assert protocol.outcome is None
    assert protocol.text == ""
    assert protocol.turn == "main-turn"
    assert protocol.interrupt()["params"] == {"threadId": "main", "turnId": "main-turn"}


def test_claude_subagent_result_cannot_complete_main_task_or_replace_session():
    protocol = ClaudeProtocol()
    protocol.receive({"type": "system", "subtype": "init", "session_id": "main"})
    protocol.receive({"type": "result", "session_id": "worker", "parent_tool_use_id": "tool-1",
                      "structured_output": {"status": "complete", "summary": "The worker task is complete.", "next_steps": []}})
    assert protocol.session == "main"
    assert protocol.outcome is None


def test_child_does_not_start_if_log_creation_fails(tmp_path, monkeypatch):
    def unexpected_start(*args, **kwargs):
        pytest.fail("The provider started before the log was ready.")

    monkeypatch.setattr(subprocess, "Popen", unexpected_start)
    with workspace_lock(tmp_path) as fd:
        with pytest.raises(OSError):
            Child([sys.executable], str(tmp_path), tmp_path / "absent" / "log.jsonl", os.environ.copy(), fd)


def test_child_cleans_process_and_pipes_after_selector_failure(tmp_path, monkeypatch):
    import selectors

    processes = []
    original = subprocess.Popen

    def record_process(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def fail_register(*args, **kwargs):
        raise OSError("The selector could not register the pipe.")

    monkeypatch.setattr(subprocess, "Popen", record_process)
    monkeypatch.setattr(selectors.DefaultSelector, "register", fail_register)
    with workspace_lock(tmp_path) as fd:
        with pytest.raises(OSError, match="selector"):
            Child([sys.executable, "-c", "import time; time.sleep(300)"], str(tmp_path),
                  tmp_path / "log.jsonl", os.environ.copy(), fd)
    assert len(processes) == 1
    process = processes[0]
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed
    with workspace_lock(tmp_path):
        pass
