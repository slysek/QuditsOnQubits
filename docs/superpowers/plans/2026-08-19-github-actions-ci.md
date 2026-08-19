# GitHub Actions CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only GitHub Actions CI that runs the full supported-Python test matrix on Ubuntu and a Python 3.12 smoke suite on Windows for every pull request to `main`.

**Architecture:** One repository workflow owns all automated tests. Three independent Ubuntu matrix jobs run the full suite for Python 3.11 through 3.13, while one Windows job covers repository hygiene, Aer integration, and scientific regressions. A dedicated CI pull request lands before the SZY-42 pull request is synchronized and made ready.

**Tech Stack:** GitHub Actions, `actions/checkout@v7`, `actions/setup-python@v7`, pip cache, pytest, actionlint 1.7.12, PowerShell.

---

## File Structure

- Create `.github/workflows/ci.yml`: the only runtime CI configuration; defines triggers, permissions, concurrency, Ubuntu matrix, and Windows smoke job.
- Existing `pyproject.toml`: authoritative Python range and `dev` dependencies; read but do not modify.
- Existing `tests/test_clean_repo_smoke.py`: Windows repository-hygiene smoke coverage.
- Existing `tests/test_experiment_aer_integration.py`: Windows real Aer smoke coverage.
- Existing `tests/test_reference_regressions.py`: Windows scientific-regression smoke coverage.
- Existing `docs/superpowers/specs/2026-08-19-github-actions-ci-design.md`: accepted design; do not modify during implementation unless a real CI constraint contradicts it.

### Task 1: Add the GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Read: `pyproject.toml`

Configuration files are the explicit TDD exception accepted in the design. Validation uses actionlint, local test commands, and a real GitHub Actions run instead of a synthetic production-code unit test.

- [ ] **Step 1: Confirm the workflow is absent on the CI branch**

Run:

```powershell
Test-Path -LiteralPath .github\workflows\ci.yml
```

Expected: `False`.

- [ ] **Step 2: Create the workflow with the exact approved contract**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
    branches:
      - main
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  tests:
    name: Tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        python-version:
          - "3.11"
          - "3.12"
          - "3.13"
    steps:
      - name: Check out repository
        uses: actions/checkout@v7

      - name: Set up Python
        uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml

      - name: Install project
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"

      - name: Run full test suite
        run: python -m pytest -q

  windows-smoke:
    name: Windows smoke (Python 3.12)
    runs-on: windows-latest
    timeout-minutes: 30
    steps:
      - name: Check out repository
        uses: actions/checkout@v7

      - name: Set up Python
        uses: actions/setup-python@v7
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml

      - name: Install project
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"

      - name: Run Windows smoke suite
        run: >-
          python -m pytest -q
          tests/test_clean_repo_smoke.py
          tests/test_experiment_aer_integration.py
          tests/test_reference_regressions.py
```

- [ ] **Step 3: Download and verify actionlint in a temporary directory**

Run from PowerShell:

```powershell
$qoqActionlintRoot = [System.IO.Path]::GetTempPath()
$qoqActionlintTemp = (
    New-Item -ItemType Directory -Path (
        Join-Path $qoqActionlintRoot ("qoq-actionlint-" + [guid]::NewGuid())
    )
).FullName
if (-not $qoqActionlintTemp.StartsWith(
    $qoqActionlintRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Unsafe temporary actionlint path"
}

try {
    gh release download v1.7.12 `
        --repo rhysd/actionlint `
        --pattern actionlint_1.7.12_windows_amd64.zip `
        --dir $qoqActionlintTemp
    $qoqArchive = Join-Path $qoqActionlintTemp "actionlint_1.7.12_windows_amd64.zip"
    $qoqExpectedSha = "6e7241b51e6817ea6a047693d8e6fed13b31819c9a0dd6c5a726e1592d22f6e9"
    $qoqActualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $qoqArchive).Hash.ToLowerInvariant()
    if ($qoqActualSha -ne $qoqExpectedSha) {
        throw "actionlint archive SHA-256 mismatch"
    }
    Expand-Archive -LiteralPath $qoqArchive -DestinationPath $qoqActionlintTemp
    & (Join-Path $qoqActionlintTemp "actionlint.exe") .github\workflows\ci.yml
    if ($LASTEXITCODE -ne 0) {
        throw "actionlint failed with exit code $LASTEXITCODE"
    }
} finally {
    if (Test-Path -LiteralPath $qoqActionlintTemp) {
        Remove-Item -Recurse -Force -LiteralPath $qoqActionlintTemp
    }
}
```

Expected: exit code `0`, no actionlint diagnostics, and only the verified temporary directory is removed.

- [ ] **Step 4: Review the workflow diff and security boundary**

Run:

```powershell
git diff --check
git diff -- .github/workflows/ci.yml
rg -n "permissions:|contents: read|pull_request:|push:|workflow_dispatch:|ubuntu-latest|windows-latest|3\.10|3\.11|3\.12|3\.13" .github/workflows/ci.yml
rg -n "write|secret|token|pull_request_target" .github/workflows/ci.yml
```

Expected:

- `git diff --check` exits `0`;
- all approved triggers, runners, and Python versions are present;
- the last search returns no matches.

- [ ] **Step 5: Commit the workflow only**

Run:

```powershell
git add .github/workflows/ci.yml
git diff --cached --check
git diff --cached --stat
git commit -m "ci: add pull request test matrix"
```

Expected: one created workflow file; no generated metadata, user files, or SZY-42 files staged.

### Task 2: Verify the workflow commands locally

**Files:**
- Test: `tests/test_clean_repo_smoke.py`
- Test: `tests/test_experiment_aer_integration.py`
- Test: `tests/test_reference_regressions.py`
- Verify: `.github/workflows/ci.yml`

- [ ] **Step 1: Run the exact Windows smoke command**

Run:

```powershell
$env:PYTHONPATH = "C:\Users\szymon\QuditsOnQubits\.worktrees\github-actions-ci\src"
python -m pytest -q `
    tests/test_clean_repo_smoke.py `
    tests/test_experiment_aer_integration.py `
    tests/test_reference_regressions.py
```

Expected: exit code `0`, no failures, and only the existing intentional skips if any.

- [ ] **Step 2: Run the complete local regression suite**

Run:

```powershell
$env:PYTHONPATH = "C:\Users\szymon\QuditsOnQubits\.worktrees\github-actions-ci\src"
python -m pytest -q
```

Expected baseline: `702 passed, 3 skipped, 315 subtests` with exit code `0`.

- [ ] **Step 3: Verify repository state before publication**

Run:

```powershell
git status -sb
git diff origin/main...HEAD --check
git log --oneline origin/main..HEAD
```

Expected:

- clean worktree;
- exactly the design, implementation-plan, and workflow commits ahead of `origin/main`;
- no diff-check errors.

### Task 3: Publish the dedicated CI pull request and verify real jobs

**Files:**
- Publish: branch `agent/github-actions-ci`
- Target: `main`

- [ ] **Step 1: Confirm GitHub authentication and PR scope**

Run:

```powershell
gh auth status
gh repo view --json nameWithOwner,defaultBranchRef
git status -sb
git diff --stat origin/main...HEAD
```

Expected: authenticated as a user with push access; repository `slysek/QuditsOnQubits`; default branch `main`; clean worktree containing only the CI design and workflow.

- [ ] **Step 2: Push the CI branch**

Run:

```powershell
git push -u origin agent/github-actions-ci
```

Expected: remote tracking branch created without changing `main`.

- [ ] **Step 3: Open a draft CI pull request through the GitHub connector**

Use these exact values:

```text
repository_full_name: slysek/QuditsOnQubits
base: main
head: agent/github-actions-ci
draft: true
title: ci: add pull request test matrix
```

PR body:

```markdown
## Summary

- run the full pytest suite on Ubuntu for Python 3.11, 3.12, and 3.13
- run repository, Aer, and scientific smoke tests on Windows Python 3.12
- use read-only permissions, pip caching, timeouts, and stale-run cancellation

## Why

The repository currently has no automated pull-request checks. This CI bootstrap must land on `main` before SZY-42 is made ready for review.

## Local validation

- actionlint 1.7.12: passed
- Windows smoke command: passed
- full suite: `702 passed, 3 skipped, 315 subtests`
```

Expected: one new draft PR targeting `main`.

- [ ] **Step 4: Wait for the real workflow and inspect every job**

Run after GitHub schedules the workflow:

```powershell
$qoqCiPr = gh pr view agent/github-actions-ci --json number --jq '.number'
if (-not $qoqCiPr) {
    throw "CI pull request number was not resolved"
}
gh pr checks $qoqCiPr --watch --interval 10
gh pr view $qoqCiPr --json statusCheckRollup
```

Expected four successful checks:

```text
Tests (Python 3.11)
Tests (Python 3.12)
Tests (Python 3.13)
Windows smoke (Python 3.12)
```

If the workflow is not scheduled within two minutes, inspect:

```powershell
gh run list --branch agent/github-actions-ci --limit 10
gh workflow list --all
```

Do not claim CI is operational until a real run completes successfully.

- [ ] **Step 5: Mark the CI PR ready after all four checks pass**

Run:

```powershell
$qoqCiPr = gh pr view agent/github-actions-ci --json number --jq '.number'
if (-not $qoqCiPr) {
    throw "CI pull request number was not resolved"
}
gh pr ready $qoqCiPr
gh pr view $qoqCiPr --json url,isDraft,state,statusCheckRollup
```

Expected: open, non-draft PR with all four checks successful. Stop for user review and merge; do not merge without explicit instruction.

### Task 4: Roll CI onto SZY-42 after the CI PR is merged

**Files:**
- Verify: `origin/main:.github/workflows/ci.yml`
- Update: branch `feature/szy-42-two-qutrit-vertical-slice`
- Preserve: uncommitted `src/qudits_on_qubits.egg-info/*` changes in the SZY-42 worktree

- [ ] **Step 1: Verify CI exists on remote `main`**

Run only after the user confirms the CI PR was merged:

```powershell
git fetch origin main
git show origin/main:.github/workflows/ci.yml
gh run list --branch main --workflow CI --limit 5
```

Expected: workflow exists on `origin/main`, and the merge-triggered `main` run is visible.

- [ ] **Step 2: Wait for the `main` CI run**

Run:

```powershell
$qoqMainRun = gh run list --branch main --workflow CI --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $qoqMainRun --exit-status
gh run view $qoqMainRun --json conclusion,jobs,url
```

Expected: `conclusion` is `success` and all four jobs passed.

- [ ] **Step 3: Audit the dirty SZY-42 worktree before merging `main`**

Run in `C:\Users\szymon\QuditsOnQubits\.worktrees\szy-42`:

```powershell
git status --short
git diff --name-only
git diff --name-only origin/main...HEAD
```

Expected: existing uncommitted changes remain limited to `src/qudits_on_qubits.egg-info/*`. Do not stage, delete, reset, or overwrite them. If another user-owned path appears, stop and request direction.

- [ ] **Step 4: Merge updated `main` without rewriting SZY-42 history**

Run in the SZY-42 worktree:

```powershell
git merge --no-edit origin/main
git status -sb
git push origin feature/szy-42-two-qutrit-vertical-slice
```

Expected: merge commit contains the CI workflow; the uncommitted egg-info changes remain unstaged and preserved; PR #4 receives a `synchronize` event.

- [ ] **Step 5: Verify PR #4 under the new CI**

Run:

```powershell
gh pr checks 4 --watch --interval 10
gh pr view 4 --json url,isDraft,state,statusCheckRollup
```

Expected: the same four CI checks pass against the SZY-42 merge result.

- [ ] **Step 6: Mark PR #4 ready**

Run only after all checks pass:

```powershell
gh pr ready 4
gh pr view 4 --json url,isDraft,state,statusCheckRollup
```

Expected: PR #4 is open, non-draft, targets `main`, and has four successful CI checks. Do not merge PR #4 without a separate explicit user instruction.
