"""Provide local CLI test doubles. Do not send model requests."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def emit(value):
    print(json.dumps(value), flush=True)


provider, behavior = sys.argv[1:3]
if "--version" in sys.argv:
    print("relay-test-provider 1")
    raise SystemExit(0)

if behavior == "crash":
    print("The test provider could not start.", file=sys.stderr)
    raise SystemExit(4)
if behavior == "bad-json":
    print("{broken", flush=True)
    raise SystemExit(0)
if behavior == "hang":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path("relay-descendant.pid").write_text(str(child.pid))
    time.sleep(300)

result = {"status": "complete", "summary": "The test goal is complete.", "next_steps": []}
if behavior in ("blocked", "blocked-after-denial"):
    result["status"] = "blocked"
if provider == "claude":
    request = json.loads(sys.stdin.readline())
    assert request["message"]["role"] == "user"
    emit({"type": "system", "subtype": "init", "session_id": "claude-test"})
    first_visit = not Path("relay-claude-visited").exists()
    if behavior == "limit-once":
        Path("relay-claude-visited").touch()
    if behavior == "limit" or (behavior == "limit-once" and first_visit):
        emit({"type": "rate_limit_event", "rate_limit_info": {
            "status": "rejected", "resetsAt": time.time() + (2.3 if behavior == "limit-once" else 0.2)
        }})
    else:
        event = {"type": "result", "is_error": False, "structured_output": result}
        if behavior in ("complete-after-denial", "blocked-after-denial"):
            event["permission_denials"] = [{"tool_name": "Bash", "tool_use_id": "denied-attempt"}]
        emit(event)
else:
    for line in sys.stdin:
        request = json.loads(line)
        method = request["method"]
        reply = {}
        if method == "initialized":
            continue
        if method == "account/rateLimits/read":
            reply = {"rateLimits": {"primary": None, "secondary": None}}
            if behavior == "limit":
                reply["rateLimits"]["primary"] = {"usedPercent": 100, "resetsAt": time.time() + 1}
        elif method in ("thread/start", "thread/resume"):
            assert request["params"]["sandbox"] == "workspace-write"
            assert request["params"]["approvalsReviewer"] == "auto_review"
            reply = {"thread": {"id": "codex-test"}}
        elif method == "turn/start":
            assert request["params"]["outputSchema"]["required"]
            reply = {"turn": {"id": "turn-test"}}
        emit({"id": request["id"], "result": reply})
        if method == "turn/start":
            if behavior == "wait":
                continue
            emit({"method": "item/completed", "params": {"threadId": "codex-test", "turnId": "turn-test", "item": {
                "type": "agentMessage", "text": json.dumps(result)
            }}})
            emit({"method": "turn/completed", "params": {"threadId": "codex-test", "turn": {
                "id": "turn-test", "status": "completed"
            }}})
        elif method == "turn/interrupt":
            break
