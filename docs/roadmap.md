# Roadmap

The next public milestone is a reproducible encoding benchmark that other
researchers can run, inspect and extend. This roadmap describes intended work;
it is not a claim of awarded funding or guaranteed hardware results.

## Available now

- One benchmark API for a logical F3/CZ3 circuit, encoding families and an explicit backend.
- LocalSU2 sampling and SchmidtTheta scanning, including verified replay of 41 saved gate sets.
- Canonical baselines, native compilation and routing, independent fidelity/leakage checks,
  and Pareto ranking across a common compiler-seed budget.
- A short offline notebook, saved QPY/JSON evidence and reanalysis.
- Separate Bell experiment tools and a historical hardware dataset with offline replay.

## Proposed next milestones

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| Expand workloads | Small graph-state benchmarks beyond the two-qutrit example. | Independent logical references; canonical comparisons; reproducible local runs. |
| Compare backends | Matched IQM Garnet/Emerald and IBM compilation campaigns. | Frozen targets, equal search budgets, native gate counts, depth and validated final layouts. |
| Extend candidate search | Additional families and synthesis policies through the existing contracts. | Deterministic generation tests, gate-action checks and full Pareto reports, including failed trials. |
| Improve distribution | Smaller optional dependency groups and a versioned release. | Fresh wheel/sdist installation, CI across supported Python versions and a tested notebook. |

Hardware execution remains a separate experiment, subject to provider access.
Compilation improvements must be re-evaluated on each target; they do not imply
better Bell values or lower experimental error by themselves.

For a [Unitary Foundation microgrant](https://unitary.foundation/grants), a focused
proposal can select milestones with public code, reproducible artifacts and clear
acceptance criteria. The [program FAQ](https://unitary.foundation/faqs) describes
the preferred duration as 3–6 months. Final scope, budget and schedule belong in
the application and should reflect the maintainer's available time.
