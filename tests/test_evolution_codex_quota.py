"""Check Codex quota reporting without network access."""

import json
from urllib.error import HTTPError, URLError

import pytest

from qudits_on_qubits.evolution import codex_quota


@pytest.fixture
def auth_file(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"type": "codex", "access_token": "secret-token", "account_id": "secret-account"}))
    return path


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, length):
        return self.data[:length]


class Opener:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request, self.timeout = request, timeout
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_reads_both_windows_and_sends_credentials_only_to_expected_endpoint(auth_file, monkeypatch):
    data = {"rate_limit": {"primary_window": {"used_percent": 14, "limit_window_seconds": 604800,
                                               "reset_at": 1793788881, "reset_after_seconds": 128383},
                           "secondary_window": {"used_percent": 0.5, "limit_window_seconds": 18000,
                                                 "reset_at": None, "reset_after_seconds": 20}}}
    opener = Opener(Response(json.dumps(data).encode()))
    monkeypatch.setattr(codex_quota, "build_opener", lambda *args: opener)
    result = codex_quota.read_codex_quota(auth_file)
    assert result["primary_window"]["remaining_percent"] == 86
    assert result["secondary_window"]["remaining_percent"] == 99.5
    assert result["primary_window"]["reset_at"] == 1793788881
    assert result["secondary_window"]["reset_after_seconds"] == 20
    assert opener.request.full_url == "https://chatgpt.com/backend-api/wham/usage"
    assert opener.request.get_method() == "GET"
    assert opener.timeout == 5
    assert "secret-token" not in repr(result)
    assert "secret-account" not in repr(result)


def test_one_window_is_valid(auth_file, monkeypatch):
    data = {"rate_limit": {"primary_window": {"used_percent": 100, "limit_window_seconds": 600}}}
    monkeypatch.setattr(codex_quota, "build_opener", lambda *args: Opener(Response(json.dumps(data).encode())))
    result = codex_quota.read_codex_quota(auth_file)
    assert result["primary_window"]["remaining_percent"] == 0
    assert "secondary_window" not in result


@pytest.mark.parametrize("body", [
    b"bad", b"[]", b"{}", b'{"rate_limit": null}',
    b'{"rate_limit": {"primary_window": {"used_percent": -1}}}',
    b'{"rate_limit": {"primary_window": {"used_percent": 101}}}',
    b'{"rate_limit": {"primary_window": {"used_percent": true}}}',
    b'{"rate_limit": {"primary_window": {"used_percent": NaN}}}',
    b'{"rate_limit": {"primary_window": {"used_percent": 2, "reset_at": -1}}}',
    b'{"rate_limit": {"primary_window": {"used_percent": 2, "limit_window_seconds": 0}}}',
    b"x" * 65537,
])
def test_malformed_responses_are_unavailable(auth_file, monkeypatch, body):
    monkeypatch.setattr(codex_quota, "build_opener", lambda *args: Opener(Response(body)))
    assert codex_quota.read_codex_quota(auth_file) is None


@pytest.mark.parametrize("response", [HTTPError("https://chatgpt.com/", 302, "redirect", {}, None),
                                     URLError("secret-token"), TimeoutError("secret-token"),
                                     Response(b'{}', status=302)])
def test_network_errors_are_unavailable_without_output(auth_file, monkeypatch, capsys, response):
    monkeypatch.setattr(codex_quota, "build_opener", lambda *args: Opener(response))
    assert codex_quota.read_codex_quota(auth_file) is None
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("data", [b"bad", b"[]", b"{}", b'{"type": "codex", "disabled": true}', b"x" * 65537])
def test_invalid_auth_file_does_not_request(auth_file, monkeypatch, data):
    auth_file.write_bytes(data)
    monkeypatch.setattr(codex_quota, "build_opener", lambda *args: pytest.fail("Unexpected request"))
    assert codex_quota.read_codex_quota(auth_file) is None


def test_redirects_are_rejected():
    assert codex_quota._NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com") is None
