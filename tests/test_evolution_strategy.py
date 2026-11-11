"""Check local CZ proposals from one model strategy."""

import asyncio
from functools import partial
import json
import random

import pytest

from qudits_on_qubits.evolution.oe_candidates import load_skeleton, skeleton_program
from qudits_on_qubits.evolution.oe_mock import mutate_skeleton
from qudits_on_qubits.evolution.oe_strategy import (
    LOCAL_STEPS, STRATEGY_MESSAGE, StrategyClient, parse_strategy, strategy_summary,
)

PAIRS = ((0, 1), (0, 2), (1, 3), (4, 5))
PLAN = {"moves": [["retarget"], ["remove", "insert"]], "qubits": [0, 1, 2, 3]}


class FakePlanner:
    def __init__(self, config):
        self.calls = 0
        self.last_usage = None

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.calls += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return json.dumps(PLAN)


def _prompt(pairs=PAIRS):
    return [{"role": "user", "content": "# Current candidate\n```python\n"
             + skeleton_program(6, pairs) + "```\n"}]


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("QOQ_OE_CONFIG", str(tmp_path / "evaluation.json"))
    pytest.importorskip("openevolve")
    from openevolve.config import LLMModelConfig
    return StrategyClient(LLMModelConfig(name="test", random_seed=3), client_factory=FakePlanner)


def test_strategy_parser_accepts_a_bounded_plan():
    assert parse_strategy("```json\n" + json.dumps(PLAN) + "\n```", 6) == PLAN


@pytest.mark.parametrize("answer", [
    "not JSON", "[]", "null", '{"moves": [], "qubits": [0, 1]}',
    '{"moves": [["execute"]], "qubits": [0, 1]}',
    '{"moves": [["remove"]], "qubits": [true, 1]}',
    '{"moves": [["remove"]], "qubits": [0, 6]}',
    '{"moves": [["remove"]], "qubits": [0, 0]}',
    '{"moves": [["remove"]], "qubits": [0]}',
    '{"moves": [["remove"]], "qubits": [0, 1], "code": "pass"}',
    '{"moves": [["remove"]], "moves": [["insert"]], "qubits": [0, 1]}',
    json.dumps({"moves": [["remove"] * 4], "qubits": [0, 1]}),
    json.dumps({"moves": [["remove"]] * 9, "qubits": [0, 1]}),
    " " * 8193, None,
])
def test_strategy_parser_rejects_invalid_data(answer):
    with pytest.raises(ValueError):
        parse_strategy(answer, 6)


@pytest.mark.parametrize("kind", ["remove", "retarget", "swap", "move", "insert"])
def test_local_mutations_preserve_qubits_outside_the_plan(kind):
    for seed in range(20):
        result = mutate_skeleton(6, PAIRS, random.Random(seed), kind=kind, qubits=[0, 1, 2, 3])
        assert (4, 5) in result and PAIRS == ((0, 1), (0, 2), (1, 3), (4, 5))
        assert all(a != b and 0 <= a < 6 and 0 <= b < 6 for a, b in result)
        assert all(pair == (4, 5) or set(pair) <= {0, 1, 2, 3} for pair in result)


def test_one_model_call_drives_many_checked_literal_proposals(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    seen, usage = set(), []
    for step in range(LOCAL_STEPS):
        reply = asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt()))
        path = tmp_path / "candidate.py"
        path.write_text(reply.removeprefix("```python\n").removesuffix("```"))
        pairs = load_skeleton(path, expected_qubits=6)
        assert pairs != PAIRS and pairs not in seen and (4, 5) in pairs
        seen.add(pairs)
        usage.append(client.last_usage)
    assert client.client.calls == 1
    assert usage[0]["total_tokens"] == 120 and all(item is None for item in usage[1:])
    assert strategy_summary(tmp_path) == {
        "model_calls": 1, "generated_candidates": LOCAL_STEPS,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt()))
    assert client.client.calls == 2


def test_strategy_uses_the_current_parent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt()))
    changed = (*PAIRS[:-1], (3, 5))
    reply = asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt(changed)))
    path = tmp_path / "candidate.py"
    path.write_text(reply.removeprefix("```python\n").removesuffix("```"))
    assert (3, 5) in load_skeleton(path, expected_qubits=6)
    assert client.client.calls == 1 and client.last_usage is None


def test_invalid_strategy_records_the_request_without_a_candidate(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    async def invalid(*args, **kwargs):
        return '{"moves": [["run_python"]], "qubits": [0, 1]}'

    monkeypatch.setattr(client.client, "generate_with_context", invalid)
    with pytest.raises(ValueError):
        asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt()))
    assert strategy_summary(tmp_path)["model_calls"] == 1
    assert strategy_summary(tmp_path)["generated_candidates"] == 0


def test_failed_request_has_no_stale_usage(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    client.client.last_usage = {"total_tokens": 999}

    async def failed(*args, **kwargs):
        raise TimeoutError("The model request exceeded its time limit")

    monkeypatch.setattr(client.client, "generate_with_context", failed)
    with pytest.raises(TimeoutError):
        asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, _prompt()))
    assert client.last_usage is None
    assert strategy_summary(tmp_path) == {"model_calls": 1, "generated_candidates": 0, "usage": {}}
    request, = (tmp_path / "strategy_requests").glob("*.json")
    assert json.loads(request.read_text())["error"] == "TimeoutError"


def test_exhausted_strategy_refreshes_without_an_unbounded_loop(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    messages = []

    async def remove_only(system, request, **kwargs):
        messages.append(request[-1]["content"])
        return json.dumps({"moves": [["remove"]], "qubits": [0, 1]})

    monkeypatch.setattr(client.client, "generate_with_context", remove_only)
    prompt = _prompt(((0, 1), (4, 5)))
    asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, prompt))
    with pytest.raises(ValueError, match="no new CZ structure"):
        asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, prompt))
    assert len(messages) == 1 and client.remaining == 0
    with pytest.raises(ValueError, match="no new CZ structure"):
        asyncio.run(client.generate_with_context(STRATEGY_MESSAGE, prompt))
    assert len(messages) == 2 and "Select different operations" in messages[-1]


def test_generator_rejects_an_untrusted_parent_before_a_model_request(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no current candidate"):
        asyncio.run(client.generate("no circuit"))
    with pytest.raises(ValueError, match="CANDIDATE assignment"):
        asyncio.run(client.generate("# Current candidate\n```python\nimport os\n```"))
    assert client.client.calls == 0 and strategy_summary(tmp_path)["model_calls"] == 0


def test_insertion_cannot_exceed_the_skeleton_limit():
    from qudits_on_qubits.evolution.skeleton_fit import MAX_SKELETON_CZ

    pairs = [(0, 1)] * MAX_SKELETON_CZ
    assert mutate_skeleton(4, pairs, random.Random(0), kind="insert") == pairs
    assert mutate_skeleton(4, pairs, random.Random(0), kind="remove", qubits=[2, 3]) == pairs
    assert mutate_skeleton(4, [(0, 1)], random.Random(0), kind="swap") == [(0, 1)]
    with pytest.raises(ValueError, match="not supported"):
        mutate_skeleton(4, pairs, random.Random(0), kind="execute")


def test_configured_skeleton_models_use_strategies_only_in_the_loop(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from openevolve.config import Config, LLMConfig, LLMModelConfig
    from qudits_on_qubits.evolution import oe_run

    def config(_):
        return Config(llm=LLMConfig(models=[LLMModelConfig(name="test", init_client=FakePlanner)]))

    monkeypatch.setattr("openevolve.config.load_config", config)
    settings = oe_run._settings(tmp_path / "model.yaml", 32, 0, system_message=oe_run.SKELETON_MESSAGE)
    factory = settings.llm.models[0].init_client
    assert isinstance(factory, partial) and factory.func is StrategyClient
    assert factory.keywords["client_factory"] is FakePlanner
    assert settings.prompt.system_message == STRATEGY_MESSAGE
    direct = oe_run._settings(tmp_path / "model.yaml", 32, 0)
    assert direct.llm.models[0].init_client is FakePlanner
    offline = oe_run._settings(None, 32, 0, system_message=oe_run.SKELETON_MESSAGE, offline_model="mutations")
    assert offline.prompt.system_message == oe_run.SKELETON_MESSAGE


def test_strategy_prompt_replaces_the_single_candidate_task():
    from qudits_on_qubits.evolution.oe_run import _skeleton_template
    fields = dict.fromkeys(("fitness_score", "metrics", "improvement_areas", "artifacts",
                            "evolution_history", "language", "current_program"), "")
    prompt = _skeleton_template("ghz3", "fixed", strategy=True).format(**fields)
    assert '"moves"' in prompt and '"qubits"' in prompt
    assert "Return the complete candidate" not in prompt
    assert "An identical candidate is wasted" not in prompt


def test_real_campaign_checks_local_candidates_and_saves_the_winner(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    from openevolve.config import Config, LLMConfig, LLMModelConfig
    from qiskit import qpy
    from test_evolution_oe_run import _baseline, _socketpair_works
    from qudits_on_qubits.evolution import oe_run
    from qudits_on_qubits.evolution.encodings import canonical_encoding
    from qudits_on_qubits.evolution.state_workloads import evaluate_workload

    if not _socketpair_works():
        pytest.skip("The sandbox blocks the asyncio socket pair")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("{}")

    def config(_):
        return Config(llm=LLMConfig(models=[LLMModelConfig(name="test", init_client=FakePlanner)]))

    monkeypatch.setattr("openevolve.config.load_config", config)
    output = tmp_path / "run"
    result = oe_run._run_campaign(baseline, output, state="ghz3", method="canonical_direct",
                                 model_config=model, representation="skeleton", encoding_mode="fixed",
                                 iterations=2, workers=1, restarts=1, timeout_seconds=5, cpu_seconds=5,
                                 campaign_timeout_seconds=90)
    assert result["saved"]["is_valid"]
    assert result["local_search"] == {
        "model_calls": 1, "generated_candidates": 2,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    with (output / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    assert evaluate_workload("ghz3", circuit, canonical_encoding(3), tolerance=1e-10)["is_valid"]
    assert json.loads((output / "configuration.json").read_text())["local_strategy"]
    events = [json.loads(line) for line in (output / "openevolve/evolution_trace.jsonl").read_text().splitlines()]
    assert len(events) == 2 and all("# Strategy " in event["child_code"] for event in events)
    assert sum((event["metadata"].get("token_usage") or {}).get("total_tokens", 0) for event in events) == 120
    request, = (output / "strategy_requests").glob("*.json")
    assert '"moves"' in json.loads(request.read_text())["messages"][-1]["content"]
