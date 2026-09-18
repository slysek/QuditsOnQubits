# Unified encoding benchmark

The public entry point is `qudits_on_qubits.benchmarks.run_benchmark`. It accepts a logical circuit, encoding families, an explicit backend, and a synthesis policy. It produces actual compiled circuits, acceptance checks, and a Pareto front **after native compilation and routing**. It never submits QPU jobs.

The [short English notebook](../notebooks/encoding_benchmark.ipynb) shows five code cells: imports, configuration, `run_benchmark`, a baseline/Pareto table, and a plot. Its default run reuses all 41 saved theta gate sets and recompiles them on a synthetic four-qubit line. No prior `artifacts/` directory or provider account is required.

Use **Restart Kernel and Run All Cells** in the project environment. Replace `backend` to select IQM or IBM. The commented `families` and `synthesis` assignments start a fresh LocalSU2/SchmidtTheta search. Archive verification and plotting live in `examples/encoding_benchmark_demo.py`; the benchmark call remains visible in the notebook. Optimized synthesis can take substantially longer.

## Quick local run

Install the package with `python -m pip install -e .`, then:

```powershell
qoq-benchmark --backend local --family local-su2 --family schmidt-theta --local-samples 2 --theta-points 3 --synthesis exact --seeds 0 1 2 --output-dir artifacts/encoding_benchmarks/quick
```

Without reinstalling an existing environment, use `python -m qudits_on_qubits.benchmarks.cli` with the same arguments and `PYTHONPATH=src`. The destination must be new or empty. `exact` explicitly uses dense embedded gates as a quick control; the default `optimized` policy uses the existing F3/CZ3 synthesis and cache, which can take substantially longer. These are different synthesis protocols, with independent Pareto fronts. There is no silent synthesis or backend fallback.

Run `examples/two_qutrit_encoding_benchmark.py` for a small real routed example on a synthetic four-qubit line. The synthetic local target is a control, not an IQM or IBM hardware claim.

## Python API

```python
from qudits_on_qubits.benchmarks import (
    BackendTarget, BenchmarkConfig, LocalSU2, SchmidtTheta,
    OptimizedSynthesis, run_benchmark, two_qutrit_graph_circuit,
)

result = run_benchmark(
    two_qutrit_graph_circuit(),
    families=[LocalSU2(samples=20, seed=42), SchmidtTheta(points=41)],
    backend=BackendTarget.iqm("garnet"),
    config=BenchmarkConfig(transpiler_seeds=tuple(range(20))),
    synthesis=OptimizedSynthesis(),
    output_dir="artifacts/encoding_benchmarks/garnet-example",
    progress=print,
)
print(result.pareto_front)
```

Constructing an IQM/IBM target without a supplied backend may read provider metadata and require an existing account. All compilation remains local. Existing credentials/loaders are reused; credentials are not put in artifacts.

Backend choices use the same runner:

```python
garnet = BackendTarget.iqm("garnet", strategy="transpile_to_iqm_exact")
emerald = BackendTarget.iqm("emerald", strategy="transpile_to_iqm_exact")
ibm = BackendTarget.ibm("YOUR_BACKEND_NAME")
```

For offline testing supply an SDK backend object: `BackendTarget.iqm("garnet", backend=IQMFakeGarnet())` or `BackendTarget.ibm(fake.name, backend=fake)`. A fake device validates compilation behavior; it does not represent current hardware calibration.

Factories accept explicit `initial_layout`, `layout_method`, `routing_method`, and `optimization_level`. Backend targets are captured once, then shared by all candidates. IQM state-preparation profiles must preserve relative phases; probability-only profiles are rejected. The same policy and routing search budget apply to every candidate.

Hardware JSON snapshots currently preserve target/protocol evidence for auditing; they are not executable provider-backend serialization. `BackendTarget.from_snapshot(snapshot)` reconstructs the explicit synthetic local profile only. Hardware replay requires a compatible supplied backend and checking its snapshot identity.

## Reuse the theta scan and its actual QPY

The theta family specifies encodings. Continuation is a separately selected synthesis strategy:

```python
from qudits_on_qubits.benchmarks import ThetaContinuationSynthesis
from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig

source = "artifacts/theta_benchmark/full-41points-tol5e-4"
theta_config = ThetaBenchmarkConfig.from_dict(RunStore.open(source).manifest["config"])

result = run_benchmark(
    two_qutrit_graph_circuit(),
    families=[SchmidtTheta(points=theta_config.theta_points)],
    backend=BackendTarget.iqm("garnet"),
    config=BenchmarkConfig(
        transpiler_seeds=(0, 1, 2),
        f3_tolerance=theta_config.f3_tolerance,
        cz3_tolerance=theta_config.cz3_tolerance,
    ),
    synthesis=ThetaContinuationSynthesis(theta_config, source_run=source),
    output_dir="artifacts/encoding_benchmarks/theta-replay",
)
```

This reads verified point bundles, including reassessment campaigns, and reuses saved F3/CZ3 QPY. It does not resynthesize imported gates. Failed scientific points remain diagnostics; corrupt artifacts stop the run. A truncated historical grid needs a matching explicit subset generator, not a differently spaced grid.

To perform a new continuation, omit `source_run` and supply a `ThetaBenchmarkConfig`. Its default canonical baseline is bundled inside the package, including wheel installations; the same pinned inputs are archived under `experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/`. To use another baseline, specify its QPY path and SHA-256 explicitly. The CLI requires an explicit `--theta-baseline-qpy` for a new scan so its provenance is deliberate. Continuation and fallback use the existing sequential runner.

Mixed policies are explicit:

```python
synthesis = {
    "local_su2": OptimizedSynthesis(),
    "schmidt_theta": ThetaContinuationSynthesis(theta_config, source_run=source),
}
```

Pass this mapping as `synthesis=`. Every policy cohort has a canonical reference and its own comparison boundary. Canonical and theta=0 retain separate labelled records when supplied by different families; equality is not presented as an improvement.

## Preserve a previously found complete preparation

```python
from qudits_on_qubits.benchmarks import SavedGateSynthesis, saved_candidate

workload = two_qutrit_graph_circuit()
directory = "PATH_TO_SAVED_BASIS"
candidate = saved_candidate(directory, candidate_id="previous_best")
saved = SavedGateSynthesis(directory, workload_hash=workload.stable_hash())

result = run_benchmark(
    workload,
    families=[],
    saved_candidates=[candidate],
    synthesis=saved,
    baseline_synthesis=OptimizedSynthesis(),
    backend=BackendTarget.iqm("garnet"),
    output_dir="artifacts/encoding_benchmarks/saved-best",
)
```

The directory must contain `E.npy`, `F3_W.qpy`, and `CZ3_W.qpy`. With explicit workload binding, an available `graph_state_direct_basis.qpy` is reused verbatim as the preparation; it still undergoes independent state validation. Without binding, preparation is assembled from the imported gates. The reference policy is explicit and recorded because imported noncanonical gates cannot serve as canonical gates.

CLI equivalents: `--saved-basis PATH` and `--reuse-saved-preparation`. These add a saved alternative to the selected families.

## Metrics and correctness

Each trial preserves:

- Candidate E, generation parameters, synthesis identity and source hashes.
- Gate action error and leakage over the full logical subspace, with one common global phase.
- Preparation fidelity and leakage before and after backend compilation.
- Native 1q/2q gate counts, total depth, 2q depth, actual physical output mapping and active wires.
- Backend target data, calibration information used by compilation, routing options, library versions and source fingerprints.
- Success or failure and its stage; failures are never encoded as zero-cost successes.

The objectives are mean native 2q count, mean depth and population depth standard deviation. At least two distinct planned seeds are required; every planned seed must succeed for Pareto eligibility. Trials from different targets, workloads, synthesis protocols or routing configurations are never silently combined.

Gate tolerance, full-state fidelity and full-state leakage are separate thresholds. A historical CZ3 tolerance of 5e-4 must be explicitly selected and does not imply zero error or automatically loosen the full-state checks.

Validation follows final layout, simulates active wires only and traces routing ancillas. Default active-wire limit is 12, configurable up to 16; logical input width is limited to 12 physical qubits. An exceeded limit is a diagnostic failure, not an unverified accepted result.

Pareto optimality is relative to this finite search and protocol. A run may find no improvement over canonical. A best concrete seed circuit is linked separately from aggregate means.

## Results and reanalysis

A run contains:

```text
manifest.json
candidates/00000/{metadata.json,E.npy,F3.qpy,CZ3.qpy,source.qpy,complete.json}
trials/00000/0000000000/{metadata.json,compiled.qpy,complete.json}
results/{metadata.json,complete.json}
run-complete.json
trials.csv
statistics.csv
pareto.csv
pareto.png
pareto.pdf
report.md
```

Files for failed stages may be absent; the metadata states why. JSON/NPY/QPY bundles and the final index are hash-verified. CSV, figures and Markdown are derived outputs.

```python
from qudits_on_qubits.benchmarks import load_benchmark
from qudits_on_qubits.benchmarks.report import write_report

result = load_benchmark("artifacts/encoding_benchmarks/quick")
write_report(result)
```

Reanalysis reads verified saved evidence and does not synthesize gates. General unified-run computation resume is not implemented; interrupted runs retain partial bundles. The existing theta runner keeps its own resumable artifacts.

## Where to extend

| Change | Module and contract |
| --- | --- |
| New supported logical circuit | `workloads.py`: construct `LogicalCircuit` with ordered `LogicalOperation` values. No runner changes for another F3/CZ3 circuit. |
| New encoding family | `families.py`: implement `generate()` yielding `EncodingCandidate`, and JSON `to_dict()`. Pass the object directly. |
| New synthesis algorithm | `synthesis.py`: implement `to_dict()`, `prepare(candidates, output_dir, progress=None)`, `build(candidate)` returning F3/CZ3 circuits. |
| New device/profile | `targets.py`: freeze its target, compile through its provider adapter, preserve final layout and enforce native gate constraints. |
| New selection/report | `analysis.py` adapts existing Pareto routines; `report.py` renders derived outputs. |

Legacy commands remain available for historical campaigns and specialized investigations. The unified API is the recommended entry point for new encoding searches. It reuses their mathematical/synthesis algorithms instead of copying them.
