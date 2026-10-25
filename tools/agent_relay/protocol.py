"""Read provider events without treating research text as control messages."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["continue", "complete", "blocked"]},
        "summary": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "summary", "next_steps"],
}


def timestamp(value: object) -> float:
    return float(value) if (type(value) in (float, int) and math.isfinite(value) and value > 0) else 0


def quota_reset(payload: dict) -> float | None:
    limits = payload.get("rateLimits") or {}
    exhausted = [window for name in ("primary", "secondary")
                 if isinstance(window := limits.get(name), dict)
                 and type(window.get("usedPercent")) in (int, float)
                 and window["usedPercent"] >= 100]
    if exhausted:
        return max(timestamp(window.get("resetsAt")) for window in exhausted)
    if payload.get("ordinaryUsageAllowed") is False or limits.get("rateLimitReachedType"):
        return 0
    return None


class Protocol:
    def __init__(self) -> None:
        self.outcome: str | None = None
        self.reset_at = 0.0
        self.session: str | None = None
        self.checkpoint: dict | None = None
        self.error = ""

    def result(self, value: object) -> None:
        if (not isinstance(value, dict)
                or value.get("status") not in ("continue", "complete", "blocked")
                or not isinstance(value.get("summary"), str)
                or not value["summary"].strip()
                or not isinstance(value.get("next_steps"), list)
                or not all(isinstance(x, str) for x in value["next_steps"])):
            self.fail("The provider did not return a valid checkpoint.")
            return
        self.checkpoint = value
        self.outcome = value["status"]

    def fail(self, message: str) -> None:
        self.outcome = "failed"
        self.error = message


class ClaudeProtocol(Protocol):
    def receive(self, event: dict) -> list[dict]:
        kind = event.get("type")
        if kind != "rate_limit_event" and event.get("parent_tool_use_id") is not None:
            return []
        if (kind == "system" and event.get("subtype") == "init"
                and isinstance(event.get("session_id"), str)):
            self.session = event["session_id"]
        if kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            if (info.get("status") == "rejected" or info.get("isUsingOverage") is True
                    or info.get("overageInUse") is True):
                self.outcome = "limited"
                self.reset_at = timestamp(info.get("resetsAt"))
        elif kind == "assistant" and event.get("error") == "rate_limit":
            self.outcome = "limited"
        elif kind == "result" and self.outcome != "limited":
            denials = event.get("permission_denials")
            if event.get("is_error"):
                message = str(event.get("errors") or event.get("result") or event.get("subtype"))
                if denials:
                    self.outcome = "blocked"
                    self.error = "Claude failed after a permission denial. Check the provider log."
                elif re.search(r"rate.?limit|usage limit|you.ve hit your limit", message, re.I):
                    self.outcome = "limited"
                else:
                    self.fail(message[:1000])
            else:
                self.result(event.get("structured_output"))
                # Earlier denied attempts can precede a successful permitted action.
                if denials and self.outcome in ("failed", "blocked"):
                    self.outcome = "blocked"
                    self.error = "Claude could not confirm progress after a permission denial. Check the provider log."
        return []


class CodexProtocol(Protocol):
    def __init__(self) -> None:
        super().__init__()
        self.pending: dict[int, str] = {}
        self.counter = 0
        self.turn: str | None = None
        self.text = ""

    def request(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        self.pending[self.counter] = method
        return {"id": self.counter, "method": method, "params": params or {}}

    def start(self, workspace: Path, prompt: str, session: str | None) -> dict:
        self.workspace = workspace
        self.prompt = prompt
        self.session = session
        return self.request("initialize", {"clientInfo": {
            "name": "qoq_agent_relay", "version": "0.1.0"
        }})

    def thread_request(self) -> dict:
        params = {"cwd": str(self.workspace), "sandbox": "workspace-write",
                  "approvalPolicy": "on-request", "approvalsReviewer": "auto_review"}
        if self.session:
            params["threadId"] = self.session
        return self.request("thread/resume" if self.session else "thread/start", params)

    def limit(self, payload: dict) -> None:
        reset = quota_reset(payload)
        if reset is not None:
            self.outcome = "limited"
            self.reset_at = max(self.reset_at, reset)

    def interrupt(self) -> dict | None:
        if self.session and self.turn:
            return self.request("turn/interrupt", {"threadId": self.session, "turnId": self.turn})
        return None

    def steer(self) -> dict | None:
        if self.session and self.turn:
            return self.request("turn/steer", {
                "threadId": self.session, "expectedTurnId": self.turn,
                "input": [{"type": "text", "text":
                           "Save your current findings and task state now. Return the required JSON checkpoint."}]
            })
        return None

    def receive(self, event: dict) -> list[dict]:
        method = event.get("method")
        if method and "id" in event:
            self.outcome = "blocked"
            self.error = f"Codex requested interactive input: {method}."
            return [{"id": event["id"], "error": {
                "code": -32601, "message": "The relay cannot supply human approval or input."
            }}]
        if "id" in event:
            request = self.pending.pop(event["id"], None)
            if "error" in event:
                if request == "account/rateLimits/read":
                    # API-key sessions do not provide ChatGPT quota data.
                    return [self.thread_request()]
                if request not in ("turn/interrupt", "turn/steer"):
                    self.fail(str(event["error"])[:1000])
                return []
            result = event.get("result") or {}
            if request == "initialize":
                return [{"method": "initialized", "params": {}}, self.request("account/rateLimits/read")]
            if request == "account/rateLimits/read":
                self.limit(result)
                return [] if self.outcome else [self.thread_request()]
            if request in ("thread/start", "thread/resume"):
                self.session = result["thread"]["id"]
                return [self.request("turn/start", {"threadId": self.session,
                        "input": [{"type": "text", "text": self.prompt}], "outputSchema": RESULT_SCHEMA})]
            if request == "turn/start":
                self.turn = result["turn"]["id"]
            return []
        params = event.get("params") or {}
        if method in ("turn/started", "item/completed", "turn/completed"):
            if not self.session or params.get("threadId") != self.session:
                return []
            event_turn = (params.get("turn") or {}).get("id") if method.startswith("turn/") else params.get("turnId")
            if method == "turn/started":
                if self.turn is not None and event_turn != self.turn:
                    return []
            elif self.turn is None or event_turn != self.turn:
                return []
        if method == "account/rateLimits/updated":
            self.limit(params)
        elif method == "turn/started":
            self.turn = params["turn"]["id"]
        elif method == "item/completed":
            item = params.get("item") or {}
            if item.get("type") == "agentMessage":
                self.text = item.get("text", "")
        elif method == "turn/completed" and self.outcome != "limited":
            turn = params.get("turn") or {}
            if turn.get("status") == "completed":
                try:
                    self.result(json.loads(self.text))
                except (ValueError, TypeError):
                    self.fail("Codex did not return a JSON checkpoint.")
            elif turn.get("status") == "interrupted":
                self.outcome = "interrupted"
            else:
                error = turn.get("error") or {}
                info = error.get("codexErrorInfo")
                if isinstance(info, str) and info in ("tooManyDenials", "cyberPolicy", "misalignmentPolicyViolation"):
                    self.outcome = "blocked"
                    self.error = f"Codex stopped after a permission or policy denial: {str(error)[:1000]}"
                elif isinstance(info, str) and info in ("usageLimitExceeded", "rateLimitExceeded"):
                    self.outcome = "limited"
                elif isinstance(info, dict) and "usageLimitExceeded" in info:
                    self.outcome = "limited"
                    self.reset_at = timestamp((info["usageLimitExceeded"] or {}).get("resetsAt"))
                else:
                    self.fail(str(error)[:1000])
        return []
