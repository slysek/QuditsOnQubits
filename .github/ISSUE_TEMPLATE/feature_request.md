---
name: Feature request
about: Propose a workload, encoding family, synthesis policy, backend, or workflow improvement.
title: "[Feature] "
labels: ""
assignees: ""
---

Read the [contribution guide](https://github.com/slysek/QuditsOnQubits/blob/main/CONTRIBUTING.md)
and describe substantial changes here before implementing them.

## Research or contributor need

What experiment or contribution is difficult today? Include a small example of
the logical circuit, workload, or workflow when relevant.

## Proposed behavior

Describe the desired API, command, or notebook behavior. Identify the affected
encoding family, synthesis policy, backend (local, IQM, IBM, or another provider),
or Bell execution workflow.

## Validation and scientific scope

How could this be checked with local simulation or provider test doubles, without
hardware access or credentials? For benchmark changes, describe the canonical
baseline, compiler budget, seeds, tolerances, and success metrics. Keep distinct
backend and synthesis protocols separate; circuit cost improvements alone do not
establish improved hardware performance.

## Alternatives and limitations

Describe workarounds, tradeoffs, dependencies, and any limitations on the claims
the new feature would support.
