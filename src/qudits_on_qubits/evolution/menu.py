"""Terminal widgets for the harness menu and the CLIProxy model picker.

choose() shows a list with a description for each option. The arrow keys move and Enter selects.
Without a terminal (pipes, tests), every widget reads plain text lines instead.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import urllib.error
import urllib.request

PROXY_BASE = "http://127.0.0.1:8317/v1"
KEY_VARIABLE = "QOQ_PROXY_KEY"
EFFORTS = (("default", "send no effort; the model uses its own default"),
           ("minimal", "fastest answers, least reasoning"),
           ("low", "fast, some reasoning"),
           ("medium", "balanced"),
           ("high", "more reasoning; good for structure search"),
           ("xhigh", "more reasoning; slow and expensive"),
           ("max", "most reasoning; slowest and most expensive"))
EFFORT_NAMES = tuple(name for name, _ in EFFORTS)
OFFLINE = "offline"
# The key stays a ${VAR} reference, so the configuration file holds no secret.
# max_tokens limits the output, and the reasoning tokens count too: at xhigh, 16384 left no answer text.
# The requests do not stream, so a long answer needs the longer timeout.
PROVIDER = {"api_key": "${" + KEY_VARIABLE + "}", "max_tokens": 64000, "timeout": 1200, "retries": 0}
MAX_LIST_BYTES = 1_000_000
HINT = "Up/Down: move   Enter: select   q: cancel"


def interactive():
    return sys.stdin.isatty() and sys.stdout.isatty()


def _items(options):
    return [option if isinstance(option, tuple) else (option, "") for option in options]


def _choose_by_number(title, items, default):
    width = max(len(label) for label, _ in items)
    print(f"\n{title}")
    for number, (label, text) in enumerate(items, 1):
        print(f"  {number:>2}. {label.ljust(width)}  {text}".rstrip())
    answer = input(f"Number [{default + 1}]: ").strip()
    if not answer:
        return default
    if not answer.isdigit() or not 1 <= int(answer) <= len(items):
        raise ValueError("Select a number from the list")
    return int(answer) - 1


def _choose_with_keys(title, items, default):
    import curses

    os.environ.setdefault("ESCDELAY", "25")  # a lone Esc cancels without the default one-second wait
    width_label = max(len(label) for label, _ in items)

    def run(screen):
        curses.curs_set(0)
        curses.use_default_colors()
        index = default
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            visible = max(1, height - 3)
            top = max(0, min(index - visible // 2, len(items) - visible))
            screen.addnstr(0, 0, title, width - 1, curses.A_BOLD)
            for row, (label, text) in enumerate(items[top:top + visible], 1):
                selected = top + row - 1 == index
                screen.addnstr(row, 0, ("> " if selected else "  ") + label.ljust(width_label), width - 1,
                               curses.A_REVERSE if selected else curses.A_NORMAL)
                if text and width > width_label + 6:
                    screen.addnstr(row, width_label + 4, text, width - width_label - 5, curses.A_DIM)
            screen.addnstr(min(height - 1, visible + 1), 0, HINT, width - 1, curses.A_DIM)
            key = screen.getch()
            if key in (curses.KEY_UP, ord("k")):
                index = (index - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j")):
                index = (index + 1) % len(items)
            elif key in (curses.KEY_ENTER, 10, 13):
                return index
            elif key == 27:
                # Some terminals send ESC [ A/B or ESC O A/B that curses does not decode; a lone ESC cancels.
                screen.nodelay(True)
                follow = (screen.getch(), screen.getch())
                screen.nodelay(False)
                if follow[0] in (ord("["), ord("O")) and follow[1] in (ord("A"), ord("B")):
                    index = (index + (1 if follow[1] == ord("B") else -1)) % len(items)
                elif follow[0] == -1:
                    raise KeyboardInterrupt
            elif key == ord("q"):
                raise KeyboardInterrupt

    try:
        return curses.wrapper(run)
    except curses.error:
        # A terminal without cursor control (for example TERM=dumb) gets the numbered list.
        return _choose_by_number(title, items, default)


def choose(title, options, default=0):
    """Return the label of one option. Each option is a label or a (label, description) pair."""
    items = _items(options)
    index = _choose_with_keys(title, items, default) if interactive() else _choose_by_number(title, items, default)
    print(f"{title}: {items[index][0]}")
    return items[index][0]


def ask(prompt, default="", note=""):
    """Read one text answer. An empty answer gives the default."""
    if note:
        print(f"  {note}")
    shown = f" [{default}]" if default != "" else ""
    return input(f"{prompt}{shown}: ").strip() or str(default)


def _retry(message):
    if not interactive():
        raise ValueError(message)
    print(f"  {message}")


def ask_int(prompt, default, low, high, note=""):
    """Read a whole number from low to high. In a terminal, ask again after a bad answer."""
    while True:
        answer = ask(prompt, default, note)
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        _retry(f"Enter a whole number from {low} to {high}")
        note = ""


def ask_new_path(prompt, default, note=""):
    """Read a path that does not exist yet."""
    while True:
        answer = ask(prompt, default, note)
        if not Path(answer).exists():
            return answer
        _retry("This path already exists; enter a new one")
        note = ""


def review(args, notes=()):
    """Show the settings and the equivalent command, then ask to start. Return True to start."""
    print(f"\nSummary: {args[0]}")
    pairs = args[1:]
    for index in range(0, len(pairs), 2):
        print(f"  {pairs[index]:<30} {pairs[index + 1] if index + 1 < len(pairs) else ''}")
    for note in notes:
        print(f"  Note: {note}")
    print("\nEquivalent command (you can run it again without the menu):")
    print("  qoq-evolution " + shlex.join(args))
    return choose("Start now?", [("start", "run with these settings"), ("cancel", "exit without running")]) == "start"


def settings_path():
    """Return the user settings file. It is outside the repository, so the key never enters Git."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "qoq-evolution" / "settings.json"


def clean_key(value):
    """Remove spaces, quotes and a Bearer prefix that come with a key copied from YAML or a header."""
    value = value.strip().strip("'\"").strip()
    return value[len("bearer "):].strip() if value.lower().startswith("bearer ") else value


def load_settings():
    """Return the saved proxy settings: proxy_key and proxy_base, when present."""
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"The settings file {path} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"The settings file {path} must contain a JSON object")
    settings = {key: data[key].strip() for key in ("proxy_key", "proxy_base") if isinstance(data.get(key), str)}
    if "proxy_key" in settings:
        settings["proxy_key"] = clean_key(settings["proxy_key"])
    return {key: value for key, value in settings.items() if value}


def save_settings(settings):
    """Write the settings readable only by the user (mode 600)."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def apply_settings():
    """Put the saved key and address into the environment. Values that are already set win."""
    settings = load_settings()
    for name, variable in (("proxy_key", KEY_VARIABLE), ("proxy_base", "QOQ_PROXY_BASE")):
        if name in settings:
            os.environ.setdefault(variable, settings[name])
    if os.environ.get(KEY_VARIABLE):
        os.environ[KEY_VARIABLE] = clean_key(os.environ[KEY_VARIABLE])


def settings_menu():
    """Ask for the proxy key (hidden input) and address, check them and save them."""
    import getpass

    settings = load_settings()
    print(f"\nSettings file: {settings_path()}")
    print(f"  CLIProxy key: {'saved' if settings.get('proxy_key') else 'not saved'}")
    key = clean_key(getpass.getpass("CLIProxy key (input is hidden; empty keeps the current key): "))
    base = ask("CLIProxy address", settings.get("proxy_base", PROXY_BASE)).rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("The address must start with http:// or https://")
    updated = {**settings, "proxy_base": base, **({"proxy_key": key} if key else {})}
    if updated.get("proxy_key"):
        try:
            print(f"  Connection is good: the proxy lists {len(list_models(base, updated['proxy_key']))} text models.")
        except ValueError as exc:
            print(f"  Warning: {exc}. The settings are saved anyway.")
    save_settings(updated)
    if updated.get("proxy_key"):
        os.environ[KEY_VARIABLE] = updated["proxy_key"]
    os.environ["QOQ_PROXY_BASE"] = base
    print("  Saved.")


def proxy_base():
    """Return the proxy URL from QOQ_PROXY_BASE, or the local default."""
    base = os.environ.get("QOQ_PROXY_BASE", PROXY_BASE).rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("QOQ_PROXY_BASE must start with http:// or https://")
    return base


def list_models(base, key, timeout=10):
    """Return sorted (name, owner) pairs of the text models that the proxy lists at /models."""
    request = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_LIST_BYTES + 1)
        if len(raw) > MAX_LIST_BYTES:
            raise ValueError("The proxy model list is too large")
        data = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"The proxy model list is not available: {exc}") from exc
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("The proxy model list has an unknown format")
    models = {item["id"]: str(item.get("owned_by", ""))[:40] for item in items
              if isinstance(item, dict) and isinstance(item.get("id"), str) and len(item["id"]) <= 200}
    return sorted((name, owner) for name, owner in models.items() if "image" not in name)


def write_model_config(directory, model, effort):
    """Write an OpenEvolve configuration for one proxy model and effort and return its path."""
    import yaml

    if effort not in EFFORT_NAMES:
        raise ValueError("Select a supported effort")
    llm = {**PROVIDER, "api_base": proxy_base(), "models": [{"name": model}]}
    if effort != "default":
        llm["reasoning_effort"] = effort
    text = yaml.safe_dump({"llm": llm}, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    path = Path(directory) / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}_{effort}_{digest}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


def pick_model(role, directory, offline_text="no model, no network, no cost", default_model=None,
               default_effort="high"):
    """Pick a proxy model and an effort for one role. Return a configuration path, or None for offline.

    Without the proxy list (no key, proxy down), ask for a configuration path instead.
    """
    key = os.environ.get(KEY_VARIABLE)
    try:
        if not key:
            raise ValueError("no CLIProxy key; save it with 'settings' in the main menu")
        models = list_models(proxy_base(), key)
    except ValueError as exc:
        print(f"\n{role}: {exc}")
        if key:
            print(f"  CLIProxy does not answer at {os.environ.get('QOQ_PROXY_BASE', PROXY_BASE)}. Start it and open"
                  " the menu again to select a model from its list; the runs need it too.")
        return ask(f"{role}: model configuration path (empty for offline)") or None
    names = [name for name, _ in models]
    start = names.index(default_model) + 1 if default_model in names else 0
    model = choose(f"{role}: model", [(OFFLINE, offline_text), *models], default=start)
    if model == OFFLINE:
        return None
    effort = choose(f"{role}: reasoning effort for {model}", EFFORTS, default=EFFORT_NAMES.index(default_effort))
    return str(write_model_config(directory, model, effort))


def _codex_cache():
    """Return the model entries of the Codex model cache, or [] when the cache is not available."""
    path = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "models_cache.json"
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_LIST_BYTES + 1)
        if len(raw) > MAX_LIST_BYTES:
            return []
        return [m for m in json.loads(raw)["models"] if isinstance(m, dict) and isinstance(m.get("slug"), str)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def codex_models():
    """Return (name, description) of the models that Codex lists for its login, from the Codex model cache."""
    models = _codex_cache()
    models.sort(key=lambda m: m["priority"] if isinstance(m.get("priority"), int) else MAX_LIST_BYTES)
    return [(m["slug"], str(m.get("description", ""))[:60]) for m in models if m.get("visibility") == "list"]


def pick_codex_model(role):
    """Pick a model of the Codex login. Return "" for the Codex default."""
    models = codex_models()
    if not models:
        return ask(f"{role}: Codex model name (empty for the default)", note="The Codex model cache is not available.")
    model = choose(f"{role}: Codex model", [("default", "the default model of Codex"), *models])
    return "" if model == "default" else model


def _codex_entry(model):
    return next((m for m in _codex_cache() if m["slug"] == model), {})


def pick_codex_effort(role, model, default="high"):
    """Pick an effort that the Codex model lists in the cache (else the proxy efforts). Return "" for the default."""
    levels = [(level["effort"], str(level.get("description", ""))[:60])
              for level in _codex_entry(model).get("supported_reasoning_levels") or ()
              if isinstance(level, dict) and isinstance(level.get("effort"), str)]
    options = [EFFORTS[0], *(levels or EFFORTS[1:])]
    names = [name for name, _ in options]
    effort = choose(f"{role}: reasoning effort", options, default=names.index(default) if default in names else 0)
    return "" if effort == "default" else effort


def pick_codex_fast(role, model):
    """Ask for the Fast service tier when the Codex model lists one. Return True for fast."""
    tiers = _codex_entry(model).get("service_tiers") or ()
    fast = next((t for t in tiers if isinstance(t, dict) and str(t.get("name", "")).lower() == "fast"), None)
    if fast is None:
        return False
    return choose(f"{role}: speed", [("standard", "normal speed and usage"),
                                     ("fast", str(fast.get("description", "faster, more usage"))[:60])]) == "fast"
