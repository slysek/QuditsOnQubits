"""Read the quota for one selected Codex account."""

import json
from http.client import HTTPException
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _window(value):
    if not isinstance(value, dict):
        raise ValueError
    used = value.get("used_percent")
    if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used) or not 0 <= used <= 100:
        raise ValueError
    result = {"used_percent": used, "remaining_percent": 100 - used}
    for name in ("limit_window_seconds", "reset_at", "reset_after_seconds"):
        number = value.get(name)
        if number is not None and (isinstance(number, bool) or not isinstance(number, int)
                                   or not 0 <= number <= 2**63 - 1
                                   or (name == "limit_window_seconds" and number == 0)):
            raise ValueError
        result[name] = number
    return result


def read_codex_quota(auth_file):
    """Return quota windows for the selected account, or None if unavailable."""
    try:
        with Path(auth_file).open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return None
        auth = json.loads(raw)
        if not isinstance(auth, dict) or auth.get("type") != "codex" or auth.get("disabled", False):
            return None
        token, account = auth.get("access_token"), auth.get("account_id")
        if not isinstance(token, str) or not token or not isinstance(account, str) or not account:
            return None
        request = Request("https://chatgpt.com/backend-api/wham/usage", headers={
            "Authorization": "Bearer " + token,
            "ChatGPT-Account-Id": account,
        })
        with build_opener(_NoRedirect()).open(request, timeout=5) as response:
            if response.status != 200:
                return None
            raw = response.read(65537)
        if len(raw) > 65536:
            return None
        data = json.loads(raw)
        limits = data.get("rate_limit") if isinstance(data, dict) else None
        if not isinstance(limits, dict):
            return None
        windows = {name: _window(limits[name]) for name in ("primary_window", "secondary_window")
                   if limits.get(name) is not None}
        return windows or None
    except (OSError, URLError, HTTPException, ValueError, TypeError, OverflowError):
        return None
