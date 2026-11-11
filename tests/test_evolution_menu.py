"""Check the terminal menu: lists, checked questions, the proxy model picker and the menu flows."""

import io
import json
import urllib.error

import pytest

from qudits_on_qubits.evolution import harness, menu


def _answers(monkeypatch, *values):
    values = iter(values)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(values))


def _menu_setup(tmp_path, monkeypatch, key=None, models=()):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(harness, "BASELINES", {state: tmp_path for state in harness.BASELINES})
    monkeypatch.setattr(harness, "MODEL_CONFIGS", tmp_path / "configs")
    if key is None:
        monkeypatch.delenv("QOQ_PROXY_KEY", raising=False)
    else:
        monkeypatch.setenv("QOQ_PROXY_KEY", key)
        monkeypatch.setattr(menu, "list_models", lambda base, key: list(models))


def _capture_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(harness.subprocess, "run",
                        lambda command, check: calls.append(command) or type("Result", (), {"returncode": 0})())
    return calls


def test_choose_shows_descriptions_and_reads_a_number(monkeypatch, capsys):
    _answers(monkeypatch, "2", "", "9")
    assert menu.choose("Pick", [("a", "first"), ("bb", "second")]) == "bb"
    assert "1. a   first" in capsys.readouterr().out
    assert menu.choose("Pick", ["a", "b", "c"], default=2) == "c"
    with pytest.raises(ValueError, match="number"):
        menu.choose("Pick", ["a", "b"])


def test_questions_check_numbers_and_paths(tmp_path, monkeypatch):
    _answers(monkeypatch, "", "x")
    assert menu.ask_int("Count", 4, 1, 8) == 4
    with pytest.raises(ValueError, match="1 to 8"):
        menu.ask_int("Count", 4, 1, 8)
    monkeypatch.setattr(menu, "interactive", lambda: True)
    _answers(monkeypatch, "0", "99", "5", str(tmp_path), str(tmp_path / "new"))
    assert menu.ask_int("Count", 4, 1, 8) == 5
    assert menu.ask_new_path("Output", "x") == str(tmp_path / "new")


def _respond(monkeypatch, body):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False
    seen = {}
    def urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        if isinstance(body, Exception):
            raise body
        return Response(body)
    monkeypatch.setattr(menu.urllib.request, "urlopen", urlopen)
    return seen


def test_list_models_sorts_text_models_with_owner(monkeypatch):
    items = [{"id": "gpt-b", "owned_by": "openai"}, {"id": "gpt-image-2"}, {"id": "claude-a", "owned_by": "anthropic"},
             {"id": 5}, "bad"]
    seen = _respond(monkeypatch, json.dumps({"data": items}).encode())
    assert menu.list_models("http://proxy/v1", "secret") == [("claude-a", "anthropic"), ("gpt-b", "openai")]
    assert seen == {"url": "http://proxy/v1/models", "auth": "Bearer secret"}


@pytest.mark.parametrize("body", [b"not json", b'{"data": 3}', b"[]", urllib.error.URLError("refused"),
                                  b" " * (menu.MAX_LIST_BYTES + 1)])
def test_list_models_rejects_bad_answers(monkeypatch, body):
    _respond(monkeypatch, body)
    with pytest.raises(ValueError):
        menu.list_models("http://proxy/v1", "secret")


def test_written_configuration_has_no_secret_and_loads(tmp_path, monkeypatch):
    openevolve_config = pytest.importorskip("openevolve.config")
    monkeypatch.setenv("QOQ_PROXY_KEY", "secret-value")
    path = menu.write_model_config(tmp_path, "claude-opus-5-5", "xhigh")
    assert "secret-value" not in path.read_text() and "${QOQ_PROXY_KEY}" in path.read_text()
    assert menu.write_model_config(tmp_path, "claude-opus-5-5", "xhigh") == path
    model, = openevolve_config.load_config(str(path)).llm.models
    assert (model.name, model.reasoning_effort, model.api_base) == ("claude-opus-5-5", "xhigh", menu.PROXY_BASE)
    default, = openevolve_config.load_config(str(menu.write_model_config(tmp_path, "m/x", "default"))).llm.models
    assert default.reasoning_effort is None
    with pytest.raises(ValueError):
        menu.write_model_config(tmp_path, "m", "huge")


def test_pick_model_uses_the_proxy_list_or_a_path(tmp_path, monkeypatch, capsys):
    pytest.importorskip("yaml")
    monkeypatch.delenv("QOQ_PROXY_KEY", raising=False)
    _answers(monkeypatch, "my.yaml", "")
    assert menu.pick_model("Loop", tmp_path) == "my.yaml"
    assert menu.pick_model("Loop", tmp_path) is None
    monkeypatch.setenv("QOQ_PROXY_KEY", "secret")
    _respond(monkeypatch, urllib.error.URLError("refused"))
    _answers(monkeypatch, "")
    assert menu.pick_model("Loop", tmp_path) is None
    assert "Start it" in capsys.readouterr().out
    monkeypatch.setenv("QOQ_PROXY_KEY", "secret")
    monkeypatch.setattr(menu, "list_models", lambda base, key: [("claude-a", "anthropic"), ("gpt-b", "openai")])
    _answers(monkeypatch, "3", "", "")
    text = open(menu.pick_model("Loop", tmp_path)).read()
    assert "name: gpt-b" in text and "reasoning_effort: high" in text
    assert menu.pick_model("Loop", tmp_path) is None


def test_discover_menu_passes_both_models(tmp_path, monkeypatch):
    _menu_setup(tmp_path, monkeypatch)
    calls = _capture_subprocess(monkeypatch)
    # discover, ghz3, free, objective cost, processor 4 (iqm_garnet; no "none" for cost), ideation path, loop
    # mutations, narrator path, families, rounds, iterations, workers, restarts, seconds, campaign limit, total
    # limit, output, start
    _answers(monkeypatch, "", "", "2", "2", "4", "ideas.yaml", "3", "tell.yaml", "", "3", "10", "", "", "", "", "",
             str(tmp_path / "new"), "")
    assert harness.main(["menu"]) == 0
    command = calls[0]
    assert command[2] == "qudits_on_qubits.evolution.discovery"
    options = dict(zip(command[3::2], command[4::2]))
    assert options["--architecture"] == "iqm_garnet" and options["--objective"] == "cost"
    assert options["--ideation-model-config"] == "ideas.yaml" and options["--offline-model"] == "mutations"
    assert options["--narrator-model-config"] == "tell.yaml"
    assert options["--rounds"] == "3" and options["--iterations"] == "10" and options["--encoding-mode"] == "free"
    assert options["--discovery-timeout-seconds"] == str(1800 * 3 * 3) and "--model-config" not in options


def test_discover_menu_with_a_proxy_loop_model_and_cancel(tmp_path, monkeypatch, capsys):
    pytest.importorskip("yaml")
    _menu_setup(tmp_path, monkeypatch, key="secret", models=[("gpt-b", "openai")])
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: pytest.fail("cancel must not run"))
    # discover, ame43, fixed, objective preparation, no processor, no orchestrator, loop model, gpt-b, effort
    # xhigh, no narrator, families, rounds, iterations, workers, restarts, seconds, campaign limit, total limit,
    # output, cancel
    _answers(monkeypatch, "", "2", "", "", "", "", "2", "2", "6", "", "", "", "5", "", "", "", "", "",
             str(tmp_path / "new"), "2")
    assert harness.main(["menu"]) == 0
    out = capsys.readouterr().out
    assert "--model-config" in out and "up to 30 loop strategy requests" in out and "--restarts" in out
    assert "at most 16 local proposals" in out
    assert "--timeout-seconds 150" in out and "qoq-evolution discover" in out
    configs = list((tmp_path / "configs").glob("gpt-b_xhigh_*.yaml"))
    assert len(configs) == 1


def _codex_cache(directory, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(directory))
    (directory / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "gpt-x", "visibility": "list", "priority": 2, "description": "X"},
        {"slug": "gpt-hidden", "visibility": "hide", "priority": 0},
        {"slug": "gpt-y", "visibility": "list", "priority": 1}, "bad"]}))


def test_codex_models_come_from_the_codex_cache_in_priority_order(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert menu.codex_models() == []
    _answers(monkeypatch, "typed")
    assert menu.pick_codex_model("Agent") == "typed"
    _codex_cache(tmp_path, monkeypatch)
    assert menu.codex_models() == [("gpt-y", ""), ("gpt-x", "X")]
    _answers(monkeypatch, "", "3")
    assert menu.pick_codex_model("Agent") == ""
    assert menu.pick_codex_model("Agent") == "gpt-x"


def test_discover_menu_with_the_codex_agent_sets_its_environment(tmp_path, monkeypatch, capsys):
    _menu_setup(tmp_path, monkeypatch)
    _codex_cache(tmp_path, monkeypatch)
    calls = _capture_subprocess(monkeypatch)
    for name in ("QOQ_AGENT_MODEL", "QOQ_AGENT_EFFORT", "QOQ_AGENT_FAST", "QOQ_AGENT_SECONDS"):
        monkeypatch.setenv(name, "before")  # monkeypatch removes the values that the menu sets
    # discover, ghz3, fixed, objective preparation, no processor, no orchestrator, loop agent, model gpt-x
    # from the Codex list, effort high, 10 minutes, no narrator, families, rounds, iterations, workers,
    # restarts, seconds, campaign limit, total limit, output, start
    _answers(monkeypatch, "", "", "", "", "", "", "", "3", "", "10", "", "1", "1", "", "", "", "", "", "",
             str(tmp_path / "new"), "")
    assert harness.main(["menu"]) == 0
    options = dict(zip(calls[0][3::2], calls[0][4::2]))
    assert options["--offline-model"] == "agent" and options["--iterations"] == "5"
    assert "--architecture" not in options and options["--objective"] == "preparation"
    assert harness.os.environ["QOQ_AGENT_FAST"] == ""  # gpt-x lists no Fast tier, so the menu does not ask
    # Two batches of four parallel sessions of 600 s, plus the margin for the checks.
    assert options["--campaign-timeout-seconds"] == str(2 * 600 + harness.CAMPAIGN_MARGIN_SECONDS)
    assert (harness.os.environ["QOQ_AGENT_MODEL"], harness.os.environ["QOQ_AGENT_EFFORT"],
            harness.os.environ["QOQ_AGENT_SECONDS"]) == ("gpt-x", "high", "600")
    assert "about 5 Codex agent sessions of at most 10 min" in capsys.readouterr().out


def test_codex_efforts_and_fast_tier_come_from_the_model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "gpt-z", "visibility": "list",
         "supported_reasoning_levels": [{"effort": "high", "description": "deep"}, {"effort": "max"}, "bad"],
         "service_tiers": [{"id": "priority", "name": "Fast", "description": "2x speed"}]},
        {"slug": "gpt-plain", "visibility": "list"}]}))
    # gpt-z: default, high, max; "" keeps high, then max; speed: standard, fast
    _answers(monkeypatch, "", "3", "2")
    assert menu.pick_codex_effort("Agent", "gpt-z") == "high"
    assert menu.pick_codex_effort("Agent", "gpt-z") == "max"
    assert menu.pick_codex_fast("Agent", "gpt-z") is True
    # Without cache levels: the proxy efforts, max last; no Fast tier: no question.
    _answers(monkeypatch, str(len(menu.EFFORTS)))
    assert menu.pick_codex_effort("Agent", "gpt-plain") == "max"
    assert menu.pick_codex_fast("Agent", "gpt-plain") is False and menu.pick_codex_fast("Agent", "") is False


def test_menu_cancel_key_exits_cleanly(monkeypatch):
    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(harness, "choose", cancel)
    with pytest.raises(SystemExit) as stop:
        harness.main(["menu"])
    assert stop.value.code == 130


def test_settings_are_private_and_the_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("QOQ_PROXY_KEY", raising=False)
    monkeypatch.setenv("QOQ_PROXY_BASE", "http://other/v1")
    assert menu.load_settings() == {}
    menu.save_settings({"proxy_key": "k1", "proxy_base": "http://saved/v1"})
    path = tmp_path / "qoq-evolution" / "settings.json"
    assert path.stat().st_mode & 0o777 == 0o600
    menu.apply_settings()
    assert menu.os.environ["QOQ_PROXY_KEY"] == "k1" and menu.os.environ["QOQ_PROXY_BASE"] == "http://other/v1"
    path.write_text("{bad")
    with pytest.raises(ValueError, match="settings file"):
        menu.load_settings()


def test_settings_menu_saves_a_hidden_key_and_keeps_it_when_empty(tmp_path, monkeypatch, capsys):
    import getpass
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("QOQ_PROXY_KEY", raising=False)
    monkeypatch.setattr(menu, "list_models", lambda base, key: [("m", "o")] if key == "k1" else [])
    keys = iter(["k1", ""])
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: next(keys))
    _answers(monkeypatch, "", "http://proxy/v1/")
    menu.settings_menu()
    menu.settings_menu()
    assert menu.load_settings() == {"proxy_key": "k1", "proxy_base": "http://proxy/v1"}
    assert menu.os.environ["QOQ_PROXY_KEY"] == "k1"
    out = capsys.readouterr().out
    assert "lists 1 text models" in out and "k1" not in out


def test_no_arguments_open_the_menu_and_settings_return_to_it(tmp_path, monkeypatch, capsys):
    _menu_setup(tmp_path, monkeypatch)
    opened = []
    monkeypatch.setattr(harness, "settings_menu", lambda: opened.append("settings"))
    monkeypatch.setattr(harness, "_discover_menu", lambda: opened.append("discover"))
    _answers(monkeypatch, "2", "1")
    assert harness.main([]) == 0
    assert opened == ["settings", "discover"]
    out = capsys.readouterr().out
    assert "1. discover" in out and "2. settings" in out and "3. resume" in out


@pytest.mark.parametrize("raw", ['"k1"', "'k1'", " k1 ", "Bearer k1", '"Bearer k1"\n'])
def test_quoted_or_bearer_keys_are_cleaned(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    menu.save_settings({"proxy_key": raw})
    assert menu.load_settings()["proxy_key"] == "k1"
    monkeypatch.setenv("QOQ_PROXY_KEY", raw)
    menu.apply_settings()
    assert menu.os.environ["QOQ_PROXY_KEY"] == "k1"
