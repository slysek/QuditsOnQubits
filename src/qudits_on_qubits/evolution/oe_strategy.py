"""Use one model strategy for several local CZ proposals."""

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable
from uuid import uuid4

from .oe_candidates import _check_width, _read_literal, load_skeleton, skeleton_program
from .oe_mock import MUTATIONS, _CURRENT, mutate_skeleton

LOCAL_STEPS = 16
MAX_ATTEMPTS = 64
MAX_STRATEGY_BYTES = 8192
REQUESTS = "strategy_requests"
CANDIDATES = "strategy_candidates"
STRATEGY_MESSAGE = (
    "Select a strategy for local CZ changes. Return one JSON object with moves and qubits. "
    "Do not return a circuit, angles or executable code."
)
STRATEGY_TASK = """Select a strategy for the next local proposals from the current parent.
The local generator uses this strategy for at most 16 proposals. The evaluator checks each proposal.
Choose one to eight move sequences. Each sequence contains one to three operations.
The generator selects one sequence for each proposal. It selects the gate indices and qubits locally.
Supported operations:
- remove: remove one CZ.
- retarget: change one end of a CZ.
- swap: exchange two adjacent CZs.
- move: move one CZ to another position.
- insert: insert one CZ.
Select at least two qubits. The generator edits only CZ pairs inside this set.
An inserted CZ also uses this set. A moved CZ can change position in the full list.
Use all qubits to search the full circuit. Use a smaller set to search one region.
Combine operations when one change cannot improve the structure.
Return JSON only. Example:
{"moves": [["remove"], ["remove", "retarget"], ["move"]], "qubits": [0, 1, 2, 3]}
"""


def _fields(items: list[tuple[str, Any]]) -> dict:
    if len({key for key, _ in items}) != len(items):
        raise ValueError("The strategy has duplicate fields")
    return dict(items)


def parse_strategy(answer: str, width: int) -> dict:
    """Check the operation sequences and qubit indices in a model response."""
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > MAX_STRATEGY_BYTES:
        raise ValueError("The strategy response is invalid or too large")
    block = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", answer, re.DOTALL)
    try:
        plan = json.loads(block[1] if block else answer, object_pairs_hook=_fields)
    except (ValueError, RecursionError) as exc:
        raise ValueError("The strategy must contain valid JSON with unique fields") from exc
    if not isinstance(plan, dict) or set(plan) != {"moves", "qubits"}:
        raise ValueError("The strategy must have only moves and qubits")
    moves, qubits = plan["moves"], plan["qubits"]
    if (not isinstance(moves, list) or not 1 <= len(moves) <= 8
            or any(not isinstance(move, list) or not 1 <= len(move) <= 3
                   or any(operation not in MUTATIONS for operation in move) for move in moves)):
        raise ValueError("The strategy must have one to eight sequences of one to three supported operations")
    if (not isinstance(qubits, list) or not 2 <= len(qubits) <= width
            or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < width for q in qubits)
            or len(set(qubits)) != len(qubits)):
        raise ValueError("The strategy must select at least two different qubits inside the circuit")
    return plan


def _parent(messages: list[dict]) -> tuple[int, tuple]:
    match = _CURRENT.search(messages[-1]["content"]) if messages else None
    if match is None:
        raise ValueError("The prompt has no current candidate")
    with tempfile.TemporaryDirectory(prefix="qoq-strategy-") as directory:
        path = Path(directory) / "parent.py"
        path.write_text(match[1], encoding="utf-8")
        width = _read_literal(path).get("num_qubits")
        _check_width(width)
        return width, load_skeleton(path, expected_qubits=width)


def _key(pairs: list | tuple) -> tuple:
    return tuple(tuple(sorted(pair)) for pair in pairs)


def _record(folder: str, name: str, data: dict) -> None:
    directory = Path(os.environ["QOQ_OE_CONFIG"]).parent / folder
    directory.mkdir(exist_ok=True)
    temporary = directory / f"{name}.tmp"
    temporary.write_text(json.dumps(data, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / f"{name}.json")


class StrategyClient:
    """Request one strategy, then return local candidates through the model interface."""

    def __init__(self, config: Any, client_factory: Callable | None = None) -> None:
        from openevolve.llm.ensemble import _create_model

        base = replace(config, init_client=None)
        self.client = (client_factory or _create_model)(base)
        self.model = config.name
        self.random = random.Random(config.random_seed)
        self.remaining = self.width = 0
        self.plan: dict = {}
        self.strategy_id = ""
        self.exhausted = False
        self.last_usage: dict | None = None
        self.seen: set[tuple] = set()

    async def _request(self, system_message: str, messages: list[dict], width: int, kwargs: dict) -> dict:
        if self.plan:
            reason = ("No new variant remained. Select different operations or a larger qubit set."
                      if self.exhausted else "The local budget ended. Use the evaluation history to select the next strategy.")
            content = messages[-1]["content"] + "\nPrevious strategy: " + json.dumps(self.plan) + "\n" + reason
            messages = [*messages[:-1], {**messages[-1], "content": content}]
        self.strategy_id = uuid4().hex
        record = {"model": self.model, "system": system_message, "messages": messages, "response": None, "usage": None}
        _record(REQUESTS, self.strategy_id, record)
        try:
            record["response"] = await self.client.generate_with_context(system_message, messages, **kwargs)
            self.last_usage = record["usage"] = getattr(self.client, "last_usage", None)
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            _record(REQUESTS, self.strategy_id, record)
        return parse_strategy(record["response"], width)

    async def generate_with_context(self, system_message: str, messages: list[dict], **kwargs: Any) -> str:
        self.last_usage = None
        width, pairs = _parent(messages)
        if not self.remaining or width != self.width:
            self.plan = await self._request(system_message, messages, width, kwargs)
            self.width, self.remaining = width, LOCAL_STEPS
            self.exhausted = False
        self.seen.add(_key(pairs))
        for _ in range(MAX_ATTEMPTS):
            moves = self.random.choice(self.plan["moves"])
            child = list(pairs)
            for kind in moves:
                child = mutate_skeleton(width, child, self.random, kind=kind, qubits=self.plan["qubits"])
            key = _key(child)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.remaining -= 1
            _record(CANDIDATES, uuid4().hex, {"strategy_id": self.strategy_id, "moves": moves,
                    "num_qubits": width, "parent": pairs, "cz": child})
            code = f"# Strategy {self.strategy_id}\n" + skeleton_program(width, child)
            return "```python\n" + code + "```"
        self.remaining = 0
        self.exhausted = True
        raise ValueError("The strategy produced no new CZ structure within the attempt limit")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(STRATEGY_MESSAGE, [{"role": "user", "content": prompt}], **kwargs)


def strategy_summary(output: Path) -> dict:
    """Count model requests and local proposals. Sum usage only from model requests."""
    requests = list((output / REQUESTS).glob("*.json"))
    usage: dict[str, int] = {}
    for path in requests:
        for key, value in (json.loads(path.read_text())["usage"] or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return {"model_calls": len(requests), "generated_candidates": len(list((output / CANDIDATES).glob("*.json"))),
            "usage": usage}
