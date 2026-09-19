---
name: Bug report
about: Report a compilation, simulation, backend integration, or reproducibility problem.
title: "[Bug] "
labels: ""
assignees: ""
---

Read the [contribution guide](https://github.com/slysek/QuditsOnQubits/blob/main/CONTRIBUTING.md).
For vulnerabilities, follow the [security policy](https://github.com/slysek/QuditsOnQubits/blob/main/SECURITY.md)
instead of posting details here. Remove credentials, tokens, and personal paths
from the material you share.

## Expected and observed behavior

Describe what you expected and what happened, including incorrect numerical
results or failed state-preservation checks even when there is no exception.

## Environment

- OS and version:
- Python version (`python --version`):
- Package version (`python -m pip show qudits-on-qubits`):
- Source commit, if using a checkout (`git rev-parse HEAD`):
- Qiskit version and relevant provider package versions:
- Backend: local simulation, IQM, IBM, PiastQ, or other; include the named target
  and whether the run used hardware:

## Reproduction

Provide a small input and the exact command. For a notebook, give its path, the
relevant cells, and configuration changes. Prefer a local simulation or provider
test double when it reproduces the problem.

```bash
# Paste the exact command here.
```

## Benchmark and artifact evidence

If applicable, include the logical workload, encoding family, synthesis policy,
compiler seeds, tolerances, backend target, and qubit layout. Provide the artifact
manifest SHA-256 or a repository-relative run directory, and the relevant manifest
fields. Mark this section not applicable if no artifacts were produced.

Keep generated runs under `artifacts/`; share only the small files needed to
reproduce the issue, not a large campaign or credential files.

## Full traceback and output

Paste the complete traceback and relevant output, with secrets removed. If there
was no exception, state that and include the incorrect result.

```text
```
