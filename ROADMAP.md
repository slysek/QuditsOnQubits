# QuditsOnQubits — Completion and Unitary Foundation Roadmap

**Status:** strategic implementation plan  
**Last updated:** 2026-07-19  
**Target:** a reproducible, hardware-agnostic toolkit for running qutrit Bell experiments on qubit-based quantum computers

## 1. Executive decision

The next milestone should not be framed as “finish every possible qudit feature.” The project should be narrowed to a credible, testable public release:

> **QuditsOnQubits v1.0:** an open-source toolkit that defines qutrit encodings on qubits, prepares selected and user-supplied qutrit states, compiles qutrit Bell experiments, executes them through simulator/IQM/PiastQ adapters, and stores fully reproducible results.

The strongest Unitary Foundation microgrant proposal is therefore:

> **Reproducible qutrit Bell experiments on qubit hardware, with IQM and PiastQ as two reference hardware integrations.**

The grant scope should contain four concrete outputs:

1. Three audited reference experiments: two-qutrit, three-qutrit GHZ, and AME(4,3).
2. A unified experiment API with Aer, IQM, and PiastQ adapters.
3. A state-preparation interface for arbitrary small pure qutrit states and scalable structured state families.
4. Public documentation, tests, reproducible result manifests, and a packaged release.

This is a better proposal than promising unrestricted “arbitrary state support.” Exact preparation of a generic N-qutrit state requires exponentially many amplitudes and generally exponential circuit resources. The library should support:

- exact arbitrary pure states for small N;
- structured states supplied as graph states or qutrit circuits;
- extensible state-preparation plugins;
- mixed-state preparation only as a later milestone.

## 2. Current repository assessment

### What is already strong

- The repository already has separate modules for Bell functionals, Bell measurements, encoding search, direct-basis benchmarks, and backend-specific compilation.
- The Bell counts pipeline is implemented for `two_qutrit`, `ghz3`, and `ame43`.
- IQM transpiler strategies, backend loading, candidate selection, and circuit exports exist.
- PiastQ/AQT is an optional integration and can submit all Bell-setting circuits in one job.
- The graph-state registry already covers GHZ/star, path, cycle, wheel, complete, and cluster graph families.
- The low-level encoding layer can mathematically apply an isometry `E^(tensor N)` to an arbitrary pure qutrit state vector.
- There are unit and regression tests around the newer IQM and PiastQ components.

### Main gaps

1. **The public API is not yet a product API.** The public `QuditsOnQubits` class is effectively empty, while users must know internal modules, scripts, artifact paths, and provider-specific details.
2. **The scientific workflow is notebook-driven.** Hardware selection, mitigation, execution, and result comparison are partly embedded in large working notebooks.
3. **Result provenance is insufficient.** Existing artifacts do not consistently record whether a result came from real hardware, a noisy fake backend, or an ideal simulator.
4. **The three reference experiments are implemented but not frozen as reproducible benchmark specifications.**
5. **PiastQ execution is synchronous and assumes availability.** There is no persistent queue, resume mechanism, idempotency key, or availability probe.
6. **Bell scenarios are hard-coded around three candidates.** The general graph-state layer and general Bell-experiment layer are not yet connected through a stable abstraction.
7. **Packaging metadata and open-source infrastructure are incomplete.** The project needs a license, contribution guidelines, citation metadata, CI, documentation deployment, release automation, and a smaller core dependency set.
8. **Security must be fixed before publication.** A working notebook currently contains a hard-coded IQM credential as a default argument.

## 3. Immediate P0 security and integrity work

Complete this before making the repository public or applying for funding.

- [ ] Revoke/rotate the exposed IQM credential.
- [ ] Remove every hard-coded token, key, and credential from source and notebooks.
- [ ] Rewrite Git history if the credential was valid at any point; deleting only the current line is not sufficient.
- [ ] Add `.env.example` containing names only, never values.
- [ ] Add automated secret scanning in CI and a local pre-commit hook.
- [ ] Replace silent fallback from real IQM hardware to a fake backend with an explicit error.
- [ ] Require `execution_mode` to be one of `ideal_simulator`, `noisy_simulator`, or `hardware`.
- [ ] Require the resolved backend identity to be stored in every run manifest.

**Acceptance criterion:** no execution can silently present simulator output as hardware output, and repository history passes secret scanning.

## 4. Target architecture

### 4.1 Core domain objects

Create small typed objects rather than one large stateful class.

```python
QutritEncoding
QutritStateSpec
StatePreparationSpec
BellScenario
ExperimentSpec
CompiledExperiment
ExperimentRun
ExperimentResult
RunManifest
```

Suggested responsibilities:

#### `QutritEncoding`

- logical dimension;
- physical qubits per qutrit;
- isometry `E`;
- leakage subspace and decoding map;
- validation of `E^dagger E = I`;
- serialization and stable identifier/hash.

The first release may support only `d=3` and two qubits per qutrit, but the object must not hide these assumptions.

#### `QutritStateSpec`

Support three explicit construction modes:

1. `from_statevector(amplitudes)` — exact small pure states;
2. `from_graph(graph, weights)` — scalable graph states;
3. `from_circuit(qutrit_circuit)` — structured user-defined preparations.

Validate normalization, dimension, qutrit count, and compatibility with the encoding.

#### `BellScenario`

Contain:

- party count and local dimensions;
- measurement labels and observables;
- Bell terms and coefficients;
- classical bound;
- expected ideal quantum value or reference value when known;
- conventions for powers, roots of unity, bit order, and outcome mapping.

The existing three Bell candidates should become registered `BellScenario` instances rather than branches spread across multiple modules.

#### `ExperimentSpec`

Contain only provider-independent intent:

- state specification;
- encoding;
- Bell scenario;
- shots;
- random seed;
- setting scheduling/randomization policy;
- mitigation configuration;
- requested backend capabilities.

#### `CompiledExperiment`

Contain:

- transpiled/prepared circuits;
- setting-to-circuit map;
- physical layout;
- compiler settings;
- validation report;
- source `ExperimentSpec` hash.

#### `ExperimentRun` and `ExperimentResult`

Separate submission from result retrieval. A run must be serializable immediately after submission so it can be resumed in a new process.

### 4.2 Public API

Target a minimal API similar to:

```python
from qudits_on_qubits import BellExperiment, QutritEncoding
from qudits_on_qubits.backends import IQMBackendAdapter

experiment = BellExperiment.reference(
    "ghz3",
    encoding=QutritEncoding.default_two_qubit(),
    shots=20_480,
)

compiled = experiment.compile(IQMBackendAdapter("garnet"))
run = compiled.submit()
result = run.result()
result.save("results/ghz3-iqm")
```

For a small user-defined pure state:

```python
experiment = BellExperiment.from_statevector(
    amplitudes=psi_qutrit,
    scenario=my_scenario,
    encoding=encoding,
)
```

The high-level API must not expose artifact-directory conventions or require notebook code.

### 4.3 Backend adapter protocol

Define one protocol implemented by:

- `AerBackendAdapter`;
- `IQMBackendAdapter`;
- `PiastQBackendAdapter`.

Required methods:

```python
capabilities()
health()
compile(experiment_spec)
submit(compiled_experiment)
status(run_id)
retrieve(run_id)
cancel(run_id)
```

Provider-specific compiler and sampler objects remain internal to adapters.

The protocol must preserve provider-native job IDs and metadata rather than flattening them away.

## 5. Scientific scope: the three reference experiments

For each of `two_qutrit`, `ghz3`, and `ame43`, create a frozen reference package containing:

- state definition;
- encoding definition;
- Bell functional and normalization convention;
- measurement observables;
- classical bound;
- ideal statevector;
- ideal Bell value computed independently by matrix methods;
- expected number of unique measurement circuits;
- bit-order and qutrit-outcome convention;
- leakage treatment;
- simulator regression data;
- one canonical end-to-end example.

### Required validation ladder

For every reference experiment:

1. Compare the generated encoded state with the ideal encoded statevector.
2. Compare Bell value from the Bell operator with Bell value reconstructed from exact probabilities.
3. Compare exact probabilities with high-shot Aer sampling within a statistical tolerance.
4. Confirm conjugate-term and power conventions.
5. Confirm physical bit ordering and qutrit decoding.
6. Report leakage before any postselection.
7. Report both unconditional and conditional-on-code-space quantities when postselection is used.
8. Estimate uncertainty using bootstrap or an analytic estimator.
9. Compare hardware output only after the same manifest and postprocessing pipeline passes simulator checks.

Do not use “Bell violation” in result summaries unless the implemented normalization and classical bound are explicitly verified for the exact scenario.

## 6. IQM completion plan

### 6.1 Extract the experiment pipeline from notebooks

Move reusable logic for the following into package modules:

- backend and calibration retrieval;
- layout selection;
- dynamical-decoupling configuration;
- readout mitigation;
- zero-noise extrapolation;
- job submission and retrieval;
- result-table generation.

Notebooks should call public functions and contain interpretation/plots, not own execution logic.

### 6.2 Normalize historical IQM work

Create a one-time migration script that scans selected notebooks and local artifact folders and emits manifests with:

- `unknown` for any metadata that cannot be proven;
- explicit `noisy_simulator` for fake-backend results;
- explicit `hardware` only when a provider job ID and backend identity are available.

Never infer hardware execution from a notebook filename.

### 6.3 Final IQM deliverables

- [ ] One simulator reference result per state.
- [ ] One IQM compilation report per state and encoding candidate.
- [ ] At least one intentionally submitted IQM hardware run per state, subject to access.
- [ ] Raw counts, run manifest, compiled circuits, and analysis report saved together.
- [ ] Cross-state summary table with uncertainty, leakage, depth, CZ count, shots, backend, date, and calibration identifier.

## 7. PiastQ/AQT completion plan

PiastQ availability must be treated as a normal system state rather than an exceptional crash.

### 7.1 Availability probe

Create an interface that can combine:

1. the user-supplied availability/status webpage;
2. a lightweight provider/API health check;
3. backend metadata/capability retrieval;
4. optional manual override.

Do not scrape a page in the core package. Implement webpage parsing as an optional provider-specific plugin with tested selectors and a safe `unknown` state.

Possible states:

```text
available
busy
maintenance
offline
unknown
authentication_error
```

### 7.2 Persistent submission queue

Implement a local SQLite queue or equivalent transactional store.

Each queued item must include:

- experiment hash;
- compiled circuit hashes;
- desired backend/mode;
- shots;
- creation time and expiry;
- current status;
- retry count and next retry time;
- provider job ID when submitted;
- last error category;
- artifact directory.

Required behavior:

- idempotent submission;
- exponential backoff with jitter;
- no retry for invalid circuits or authentication errors;
- resume after process restart;
- retrieve previously submitted jobs without resubmission;
- explicit cancellation;
- dry-run mode;
- maximum retry/expiry policy.

### 7.3 PiastQ deliverables

- [ ] `piastq status` command.
- [ ] `piastq enqueue <experiment>` command.
- [ ] `piastq worker` command that checks availability and submits eligible jobs.
- [ ] `piastq resume`/`retrieve` command.
- [ ] Fake-client contract tests covering offline, timeout, duplicate submission, partial result, and recovery.
- [ ] One end-to-end simulator/fake-provider example.
- [ ] One real PiastQ run for each feasible reference experiment, subject to access and machine capacity.

## 8. General state support

The phrase “arbitrary qutrit state” must be split into precise supported cases.

### Tier A — arbitrary small pure states

- Accept a normalized vector of length `3**N`.
- Encode it through a validated local isometry.
- Produce an exact state-preparation circuit using Qiskit synthesis.
- Warn and report synthesis cost.
- Set a documented practical limit based on tests, not an arbitrary claim.

### Tier B — structured states

Create a small qutrit circuit intermediate representation supporting at least:

- local qutrit Fourier gate;
- generalized Pauli `X` and `Z`;
- controlled-phase/CZ;
- arbitrary local qutrit unitary;
- graph-state construction;
- composition and inversion where defined.

Compile each logical operation through the selected encoding and backend adapter.

### Tier C — later extensions

Keep out of the first grant milestone unless Tier A/B finish early:

- mixed states and channels;
- different encodings for different parties;
- generic qudit dimension `d > 3`;
- variational state preparation;
- tensor-network state import;
- automatic Bell-inequality discovery.

## 9. Reproducible result model

Replace the minimal CSV-only manifest with a versioned JSON schema plus a flat CSV summary.

### Mandatory manifest fields

- schema version;
- experiment/run UUID;
- UTC creation, submission, completion timestamps;
- Git commit SHA and package version;
- Python/Qiskit/provider package versions;
- state specification and hash;
- Bell scenario specification and hash;
- encoding matrix/hash;
- backend provider, backend name, and execution mode;
- provider job ID;
- backend capability snapshot;
- calibration snapshot/reference when permitted;
- shots and seed;
- transpiler name, settings, seed, layout, and optimization strategy;
- circuit hashes, depth, size, and gate counts;
- raw count file and checksum;
- leakage and postselection policy;
- Bell value, uncertainty, classical bound, and violation margin;
- mitigation configuration and unmitigated result;
- status, warnings, and errors.

### Artifact layout

```text
runs/<run_uuid>/
  manifest.json
  experiment.json
  circuits/
  counts.json.gz
  analysis.json
  summary.md
  plots/
```

Large result data should live in versioned GitHub releases, Zenodo, or a dedicated data repository rather than being committed indiscriminately to the source repository.

## 10. Package and open-source readiness

### Packaging

- [ ] Split dependencies into `core`, `iqm`, `piastq`, `notebooks`, and `dev` extras.
- [ ] Remove Jupyter, plotting, IBM Runtime, and unrelated heavy packages from the base install where possible.
- [ ] Add project authors, license expression, README, repository/docs URLs, keywords, and classifiers.
- [ ] Add console entry points instead of relying only on `scripts/`.
- [ ] Pin provider compatibility ranges deliberately and test them.
- [ ] Publish pre-release `0.2.0`, then stable `1.0.0` only after the public API and schemas are frozen.

### Repository files

- [ ] `LICENSE` — choose a permissive license such as Apache-2.0 or BSD-3-Clause after confirming institutional constraints.
- [ ] `CONTRIBUTING.md`.
- [ ] `CODE_OF_CONDUCT.md`.
- [ ] `SECURITY.md`.
- [ ] `CITATION.cff`.
- [ ] `CHANGELOG.md`.
- [ ] issue and pull-request templates.
- [ ] architecture decision records for encoding, result schema, and backend protocol.

### Documentation

Use MkDocs or Sphinx and publish documentation automatically.

Required pages:

1. What problem the library solves.
2. Installation and optional provider extras.
3. Encoding conventions and leakage.
4. Two-qutrit Bell quickstart.
5. GHZ3 and AME(4,3) tutorials.
6. Custom pure-state tutorial.
7. Custom graph/circuit tutorial.
8. IQM setup and execution.
9. PiastQ setup, status, queue, and recovery.
10. Result manifests and reproducibility.
11. API reference.
12. Scientific assumptions and limitations.

Remove `docs/` from `.gitignore` before adopting it as the maintained documentation source.

### Continuous integration

CI should include:

- Python 3.10–3.13 test matrix where provider dependencies permit;
- formatter/linter;
- static typing on the public/core modules;
- unit tests and coverage;
- secret scanning;
- package build and install test;
- documentation build;
- simulator integration tests;
- mocked IQM/PiastQ contract tests;
- notebook smoke tests for selected tutorial notebooks;
- release workflow to PyPI and GitHub releases.

Hardware tests must be manual/scheduled and must never run automatically on an untrusted pull request.

## 11. Unitary Foundation microgrant strategy

### Recommended application title

**Reproducible qutrit Bell experiments on qubit quantum computers**

### One-sentence pitch

QuditsOnQubits will provide an open, vendor-neutral Python toolkit for encoding qutrit systems into qubits and running reproducible Bell experiments across simulators, IQM superconducting hardware, and the PiastQ/AQT trapped-ion system.

### Why the project fits

- It is an open-source quantum software/research infrastructure project.
- It converts substantial existing research code into a reusable public tool.
- It connects higher-dimensional quantum-information experiments to hardware most researchers can access only through qubit APIs.
- It provides cross-provider reproducibility rather than a one-off notebook.
- It is specialized enough that conventional commercial funding is unlikely, but useful to researchers and educators.

### Proposed 3–6 month grant deliverables

1. Stable typed API and data schema.
2. Audited two-qutrit, GHZ3, and AME(4,3) reference experiments.
3. Simulator, IQM, and resilient PiastQ adapters.
4. Small arbitrary pure-state and structured qutrit-circuit support.
5. Public documentation, CI, examples, and package release.
6. A versioned public dataset of canonical simulation/hardware runs where access permits.

### Suggested USD 4,000 budget

| Item | Amount | Purpose |
|---|---:|---|
| Developer/research time | $2,800 | API refactor, tests, adapters, scientific validation |
| Hardware/compute access | $600 | IQM/PiastQ runs, simulator compute, data storage |
| Documentation and dissemination | $400 | tutorials, diagrams, release materials, publication preparation |
| Contingency | $200 | provider/API changes or additional validation runs |

Adjust this budget if hardware access is free; move unused hardware funds to development time and documentation rather than inventing costs.

### Evidence to prepare before applying

Complete a short pre-application sprint first:

- [ ] rotate and purge credentials;
- [ ] make the repository public, subject to institutional approval;
- [ ] add license and citation metadata;
- [ ] enable green CI;
- [ ] publish one clean simulator tutorial end to end;
- [ ] publish one versioned result manifest;
- [ ] add an architecture diagram;
- [ ] open the roadmap as GitHub milestones/issues;
- [ ] tag an initial `v0.2.0` release;
- [ ] record a two-minute demo video.

### Two-minute video outline

- **0:00–0:20:** the problem — qutrit experiments are useful, but most accessible hardware and SDKs expose qubits.
- **0:20–0:50:** current proof — show the three Bell scenarios and compiled circuits/results.
- **0:50–1:20:** what is missing — fragmented notebooks, provider differences, reproducibility, PiastQ availability.
- **1:20–1:45:** grant deliverables — unified API, two hardware adapters, state support, tests, docs, data.
- **1:45–2:00:** public impact and the $4,000 request.

### Success metrics

- three reference experiments pass the full validation ladder;
- one command/API path works on Aer and compiles for both IQM and PiastQ;
- every published result has a complete versioned manifest;
- core public modules have meaningful unit coverage, with a target of at least 80%;
- installation and three tutorials work in a clean environment;
- package and documentation are publicly released;
- at least one external researcher can reproduce a simulator result without private guidance.

## 12. Proposed delivery schedule

### Phase 0 — Security and truthfulness (week 1)

- rotate/remove secrets;
- eliminate silent simulator fallback;
- define execution modes and provenance rules;
- add secret scanning.

### Phase 1 — Freeze scientific contracts (weeks 2–3)

- formalize the three reference scenarios;
- verify classical/ideal values and conventions;
- define the manifest schema;
- add exact and sampling regression tests.

### Phase 2 — Public API and packaging (weeks 4–6)

- implement domain objects and backend protocol;
- create Aer adapter;
- refactor reference experiments onto the new API;
- split dependencies and add CLI entry points.

### Phase 3 — IQM normalization (weeks 7–9)

- implement IQM adapter;
- extract notebook logic;
- migrate historical results conservatively;
- execute/record canonical IQM runs when access permits.

### Phase 4 — PiastQ resilience (weeks 10–12)

- implement health/availability interface;
- implement persistent queue and resume;
- add provider contract tests;
- execute/record canonical PiastQ runs when available.

### Phase 5 — General state support (weeks 13–14)

- exact small pure-state API;
- structured qutrit-circuit API;
- resource estimates and limits;
- custom-state tutorials.

### Phase 6 — Public release (weeks 15–16)

- documentation and examples;
- external clean-install reproduction;
- PyPI/GitHub release;
- Zenodo/data release where appropriate;
- Unitary Foundation application and demo video.

## 13. Prioritized implementation backlog

### P0 — blockers

1. Credential rotation and history cleanup.
2. Explicit hardware/simulator provenance.
3. Freeze and test the three reference Bell scenarios.
4. Versioned run manifest schema.
5. Green CI and clean installation.

### P1 — grant core

6. Stable public experiment API.
7. Aer adapter.
8. IQM adapter and notebook extraction.
9. PiastQ adapter with persistent queue/resume.
10. Canonical results and reports for all three states.
11. Small arbitrary pure-state preparation.
12. Structured qutrit-circuit preparation.
13. Documentation, tutorials, license, and release metadata.

### P2 — after the core release

14. Generic graph-derived Bell scenarios.
15. Per-party heterogeneous encodings.
16. Mixed-state/channel workflows.
17. Dimension `d > 3`.
18. Tensor-network/variational preparation.
19. Additional hardware providers.
20. JOSS or equivalent software paper.

## 14. Suggested GitHub milestones and issues

### Milestone A — `v0.2.0-reproducible-core`

- Remove leaked credentials and add secret scanning.
- Add execution provenance and disable silent fallback.
- Define `ExperimentSpec` and `RunManifest` schemas.
- Freeze reference Bell scenarios.
- Add Aer end-to-end reference tests.

### Milestone B — `v0.3.0-provider-adapters`

- Implement backend adapter protocol.
- Port IQM flow to adapter.
- Port PiastQ flow to adapter.
- Add persistent PiastQ queue and status probe.
- Add provider contract tests.

### Milestone C — `v0.4.0-general-state-preparation`

- Add arbitrary small pure-state preparation.
- Add qutrit circuit IR.
- Add resource estimation.
- Add custom-state tutorials.

### Milestone D — `v1.0.0-public-release`

- Stabilize API/schema.
- Publish docs and package.
- Publish canonical result dataset.
- Complete external reproduction test.
- Submit Unitary Foundation application.

## 15. Definition of done for v1.0

The library is ready for a stable public release and a strong grant application when all of the following are true:

- no credentials are present in current files or reachable history;
- the repository has a clear permissive license and contribution/citation files;
- a new user can install the core package without provider-specific heavy dependencies;
- the three reference experiments run from one public API on Aer;
- the same experiment specifications compile through IQM and PiastQ adapters;
- PiastQ jobs can be queued, resumed, and retrieved after process restart;
- every result is unambiguously labeled as ideal simulator, noisy simulator, or hardware;
- every published result includes raw counts, circuit hashes, provider job ID when applicable, software versions, uncertainty, leakage, and Bell normalization/bound;
- arbitrary small pure states and structured qutrit circuits have documented supported paths and explicit resource limits;
- CI, documentation, and release automation are green;
- at least one external person reproduces a tutorial in a clean environment;
- the two-minute grant demo shows a real end-to-end workflow rather than only notebook code.

## 16. External references for the grant/application phase

- Unitary Foundation microgrants: https://unitary.foundation/grants/
- Unitary Foundation microgrant FAQ: https://unitary.foundation/grants/faq/
- Previously funded projects: https://unitary.foundation/grants/
- PiastQ information (PCSS): https://quantum.psnc.pl/
- EuroHPC Piast-Q system information: https://eurohpc-ju.europa.eu/

Verify current application wording, URLs, provider APIs, and hardware access conditions immediately before submission because these can change.
