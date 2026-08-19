# GitHub Actions CI Design

**Date:** 2026-08-19
**Repository:** `slysek/QuditsOnQubits`
**Target branch:** `main`

## Goal

Add repository-level continuous integration that runs automatically for every pull request targeting `main`, validates every supported Python version, and provides one Windows cross-platform smoke check.

The CI bootstrap ships in a dedicated pull request before the SZY-42 pull request is made ready for review.

## Current State

- The repository has no `.github/workflows` directory.
- Pull request #4 targets `main` but currently has no status checks.
- `pyproject.toml` supports Python `>=3.10,<3.14` and defines the `dev` extra with pytest and pytest-cov.
- A clean `origin/main` baseline passes locally: `702 passed, 3 skipped, 315 subtests`.

## Scope

Create one workflow:

```text
.github/workflows/ci.yml
```

The workflow owns automated Python test execution only. It will not publish packages, deploy artifacts, access hardware backends, use repository secrets, or change branch-protection settings.

## Triggers

Run the workflow for:

- every pull request whose base branch is `main`;
- every push to `main`;
- manual `workflow_dispatch` runs.

The `pull_request` trigger covers opening a PR, pushing new commits to it, and reopening it. A concurrency group cancels an older in-progress run when the same PR or branch receives a newer commit.

## Security

Set workflow-level permissions to:

```yaml
permissions:
  contents: read
```

The workflow will use no credentials or write permissions. Tests that represent private hardware or optional provider integrations remain mocked or skipped according to the existing test suite.

## Actions and Dependency Installation

Use the current official major releases:

- `actions/checkout@v7`;
- `actions/setup-python@v7`.

`setup-python` enables its built-in pip cache with `pyproject.toml` as the dependency cache key. Each job installs the project with:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No Conda environment is assumed in GitHub Actions.

## Job 1: Ubuntu Supported-Python Matrix

Run on `ubuntu-latest` with `strategy.fail-fast: false` for:

- Python `3.10`;
- Python `3.11`;
- Python `3.12`;
- Python `3.13`.

Every matrix entry runs the complete suite:

```bash
python -m pytest -q
```

Each job has a 30-minute timeout. Failure in one Python version does not cancel the remaining versions, so the PR reports the complete compatibility picture.

## Job 2: Windows Python 3.12 Smoke

Run on `windows-latest` with Python `3.12`. Install the same project and development dependencies, then run:

```powershell
python -m pytest -q tests/test_clean_repo_smoke.py tests/test_experiment_aer_integration.py tests/test_reference_regressions.py
```

This checks repository hygiene, a real local Aer integration path, and frozen scientific reference values on Windows without duplicating the full four-version matrix.

The Windows job has a 30-minute timeout.

## Failure Behavior

- Installation failure fails only the affected job.
- Test failure produces the normal pytest traceback and nonzero exit status.
- Matrix jobs continue independently because fail-fast is disabled.
- A newer commit cancels stale work for the same PR or branch.
- CI never retries tests automatically, avoiding hidden flaky-test behavior.

## Validation

CI configuration is a configuration-file change, so production-code TDD does not apply. Validation consists of:

1. `actionlint` against `.github/workflows/ci.yml`;
2. local execution of the Windows smoke command;
3. local full-suite baseline evidence;
4. a real GitHub Actions run on the dedicated CI pull request;
5. confirmation that all five jobs are present: four Ubuntu matrix jobs and one Windows smoke job.

## Rollout

1. Commit the workflow on branch `agent/github-actions-ci`.
2. Push the branch and open a dedicated draft PR to `main`.
3. Verify the real GitHub Actions run and resolve any workflow-only failures.
4. Mark the CI PR ready and merge it to `main` after review.
5. Merge the updated `main` into `feature/szy-42-two-qutrit-vertical-slice` without rewriting its published history.
6. Push the synchronized SZY-42 branch and verify that PR #4 receives the new CI runs.
7. Mark PR #4 ready only after its CI passes.

Branch-protection enforcement is intentionally separate. The workflow will run for every PR to `main`; making its checks mandatory before merge can be enabled after the check names exist on `main`.

## Acceptance Criteria

- The workflow file exists on `main` after the CI PR is merged.
- Every PR to `main` schedules the supported-Python matrix and Windows smoke job.
- Pushes to `main` and manual dispatches schedule the same jobs.
- Workflow permissions are read-only.
- Python `3.10`, `3.11`, `3.12`, and `3.13` each run the full pytest suite on Ubuntu.
- Windows Python `3.12` runs the defined smoke suite.
- Dependency caching uses `pyproject.toml`.
- Stale runs for the same PR or branch are cancelled.
- No secrets, hardware credentials, publishing, deployment, or automatic retries are introduced.

## References

- GitHub Actions pull request events: <https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request>
- `actions/checkout` releases: <https://github.com/actions/checkout/releases>
- `actions/setup-python` releases: <https://github.com/actions/setup-python/releases>
