"""Check the short progress lines and the optional narrator summaries."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from qudits_on_qubits.evolution import progress

INFO = "2026-11-03 12:00:00,000 - INFO - "
METRICS = "is_valid={valid}, F={f}, cz_count={cz}.0000, two_qubit_depth=5.0000, decoder_cz_count=4.0000, evaluation_failed={failed}"


def _log(tmp_path, *lines, end="\n"):
    with (tmp_path / "campaign.log").open("a") as stream:
        stream.write("\n".join(lines) + end)


def test_progress_prints_short_lines_and_keeps_them(tmp_path, capsys):
    tracker = progress.Progress("ghz3_seed_0", tmp_path, 4)
    _log(tmp_path, INFO + "Evaluated program s in 3.10s: " + METRICS.format(valid=1, f=1, cz=9, failed=0),
         INFO + "Iteration 1: Program a (parent: s) completed in 5.00s",
         INFO + "Metrics: " + METRICS.format(valid=1, f=1, cz=8, failed=0),
         INFO + "🌟 New best solution found at iteration 1: a",
         "2026 - WARNING - The candidate evaluation failed: ValueError: The CZ structure has 1 CZ across the cut",
         "2026 - ERROR - LLM generation failed: timeout",
         "2026 - ERROR - LLM generation failed: timeout",
         "Codex quota (after): unavailable",
         INFO + "Admission strategy rejected program c",
         INFO + "Iteration 4: Program b (parent: a) completed in 4.00s")
    _log(tmp_path, INFO + "Metrics: " + METRICS.format(valid=0, f=0.5, cz=6, failed=0)[:40], end="")
    tracker.poll()
    _log(tmp_path, METRICS.format(valid=0, f=0.5, cz=6, failed=0)[40:])
    tracker.poll()
    out = capsys.readouterr().out
    assert "[ghz3_seed_0] seed: 9 CZ, valid; 4 iterations" in out
    assert "iteration 1: new best, 8 CZ, 2q depth 5, decoder 4 CZ" in out
    assert out.count("error: LLM generation failed: timeout") == 1
    assert "Codex quota (after): unavailable" in out
    assert "1/4 iterations: 1 valid, 0 invalid, 0 rejected" in out
    assert "3/4 iterations: 1 valid, 1 invalid, 1 rejected; best 8 CZ; most common rejection: The CZ structure" in out
    assert "Metrics:" not in out
    tracker.finish({"cz_count": 8, "two_qubit_depth": 5, "decoder_cz_count": 4})
    saved = (tmp_path / "progress.log").read_text()
    assert "new best" in saved and "done in" in saved and str(tmp_path / "campaign.log") in saved


def test_narrator_summaries_and_failures_never_stop_the_search(tmp_path, monkeypatch, capsys):
    prompts = []
    monkeypatch.setattr(progress, "ask_model", lambda cfg, system, prompt: prompts.append(prompt) or "Idzie\n dobrze.")
    tracker = progress.Progress("run", tmp_path, 10, narrator_config="narrator.yaml", narrate_seconds=0)
    _log(tmp_path, INFO + "Iteration 1: Program a (parent: s) completed in 5.00s",
         INFO + "Metrics: " + METRICS.format(valid=1, f=1, cz=7, failed=0))
    tracker.poll()
    tracker.thread.join(5)
    tracker.finish({"cz_count": 7})
    assert '"iterations_done": 1' in prompts[0] and "Polish" in prompts[0] and '"finished": true' in prompts[-1]
    assert capsys.readouterr().out.count("summary: Idzie dobrze.") == 2

    def broken(*_args):
        raise RuntimeError("proxy down")
    monkeypatch.setattr(progress, "ask_model", broken)
    tracker = progress.Progress("run", tmp_path, 10, narrator_config="narrator.yaml")
    tracker.finish({"cz_count": 7})
    assert "narrator unavailable: RuntimeError" in capsys.readouterr().out


def test_progress_ignores_a_missing_log(tmp_path, capsys):
    progress.Progress("x", tmp_path / "missing", 3).poll()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ideation,model,expected_tokens", [
    (True, "claude-fable-5-2", 128000),
    (True, "gpt-6.1-terra", None),
    (False, "claude-fable-5-2", 16000),
])
def test_ideation_has_no_client_timeout_or_harness_token_cap(tmp_path, monkeypatch,
                                                          ideation, model, expected_tokens):
    pytest.importorskip("openevolve")
    import httpx
    import openai

    config = tmp_path / "model.yaml"
    config.write_text(f"llm:\n  api_key: test-key\n  api_base: http://proxy.test/v1\n"
                      f"  max_tokens: 64000\n  timeout: 1\n  retries: 0\n  reasoning_effort: xhigh\n"
                      f"  models:\n    - name: {model}\n      timeout: 1\n      max_tokens: 16000\n")
    requests = []

    def respond(request):
        requests.append(request)
        if ideation:
            time.sleep(1.05)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "families"}}],
        })

    real_client = openai.OpenAI

    def client(**options):
        return real_client(**options, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "OpenAI", client)
    assert progress.ask_model(config, "Return the families.", "Find five families.", unbounded=ideation) == "families"
    body = json.loads(requests[0].content)
    assert body.get("max_tokens") == expected_tokens
    assert body["reasoning_effort"] == "xhigh"
    assert requests[0].extensions["timeout"]["read"] == (None if ideation else 1)


@pytest.mark.parametrize("result", [{"answer": "families"}, {"error": "The model failed"}])
def test_ideation_worker_returns_the_answer_or_error(monkeypatch, result):
    def run(command, **options):
        assert options["start_new_session"] and options["check"]
        assert json.loads(options["input"])["unbounded"] is True
        Path(command[3]).write_text(json.dumps(result))

    monkeypatch.setattr(progress.subprocess, "run", run)
    if "error" in result:
        with pytest.raises(ValueError, match="The model failed"):
            progress.ask_ideation_model("model.yaml", "system", "prompt")
    else:
        assert progress.ask_ideation_model("model.yaml", "system", "prompt") == "families"


def test_ideation_can_stop_a_worker_without_an_http_timeout(monkeypatch):
    popen = subprocess.Popen
    processes = []

    def interrupted_worker(command, **options):
        process = popen([sys.executable, "-c", "import time; time.sleep(60)"], **options)
        process.communicate = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
        processes.append(process)
        return process

    monkeypatch.setattr(progress.subprocess, "Popen", interrupted_worker)
    with pytest.raises(KeyboardInterrupt):
        progress.ask_ideation_model("model.yaml", "system", "prompt")
    assert processes[0].poll() is not None
