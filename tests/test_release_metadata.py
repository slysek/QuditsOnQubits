"""Release metadata in pyproject, CITATION.cff and CHANGELOG must not drift apart.

A version bump touches three files by hand. Without this guard a stale value in
one of them ships in a release, which has happened before.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
REPOSITORY = "github.com/slysek/QuditsOnQubits"

RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", re.M)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def latest_release() -> tuple[str, str]:
    match = RELEASE_HEADING.search(CHANGELOG.read_text(encoding="utf-8"))
    assert match is not None, "CHANGELOG has no released version heading"
    return match.group(1), match.group(2)


def test_citation_version_matches_pyproject():
    assert citation()["version"] == project_version()


def test_changelog_documents_the_released_version():
    version, _ = latest_release()
    assert version == project_version()


def test_citation_release_date_matches_changelog():
    _, date = latest_release()
    assert citation()["date-released"] == date


def test_citation_release_date_is_a_string():
    # CFF 1.2.0 requires a string; an unquoted YAML date becomes datetime.date
    # and fails schema validation.
    assert isinstance(citation()["date-released"], str)


def test_changelog_keeps_an_unreleased_section():
    assert "## [Unreleased]" in CHANGELOG.read_text(encoding="utf-8")


def test_citation_version_is_a_string():
    # A two-component version such as `0.2` parses as float unless quoted, which
    # would make the comparison above fail as 0.2 != "0.2".
    assert isinstance(citation()["version"], str)


def test_changelog_link_references_resolve():
    text = CHANGELOG.read_text(encoding="utf-8")
    version = project_version()
    reference = re.search(rf"^\[{re.escape(version)}\]: (\S+)$", text, re.M)
    assert reference is not None, f"missing link reference for {version}"
    assert reference.group(1).endswith(f"...v{version}"), reference.group(1)
    assert REPOSITORY in reference.group(1), reference.group(1)

    unreleased = re.search(r"^\[Unreleased\]: (\S+)$", text, re.M)
    assert unreleased is not None, "missing [Unreleased] link reference"
    assert unreleased.group(1).endswith(f"compare/v{version}...HEAD"), unreleased.group(1)
