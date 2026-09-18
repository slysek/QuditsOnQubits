# Legacy direct-basis benchmarks

For new searches, use the [unified encoding benchmark](encoding_benchmark.md).
These tools remain available for previous campaigns and specialized investigations.

## Direct-Basis Top-K Selection

Run a full direct-basis benchmark for one Bell-supported state and copy selected circuits:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state ghz3 `
  --candidate-set all-qutrit-u3 `
  --n-transpile-runs 20 `
  --jobs 4 `
  --approximation-thresholds 0.99,0.95,0.90 `
  --select-top-k 5
```

This runs `exact`, `fid099`, `fid095`, and `fid090`. Threshold labels pass `approximation_degree` into Qiskit transpilation; selected threshold rows must also satisfy `fidelity >= threshold`. `--jobs` runs independent candidates concurrently while keeping each candidate's exact/threshold exports serialized. Selected circuits are written under `artifacts/direct_basis_runs/selected_best/<state>/<run_id>/`.

For a fast smoke run:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state two_qutrit `
  --candidate-set sanity `
  --limit-candidates 3 `
  --n-transpile-runs 1 `
  --jobs 2 `
  --local-line-coupling `
  --approximation-thresholds 0.99,0.95,0.90 `
  --select-top-k 2
```

Small smoke candidate sets may warn that a threshold label selected fewer than `top-k` rows; that means the measured fidelity did not pass that threshold. The `exact` label still selects the best depth-ranked circuits.

Load the rank-1 transpiled circuit from a selected run:

```powershell
python scripts/load_best_circuit.py `
  --run-kind direct_basis_runs `
  --state two_qutrit `
  --run-id <printed_run_id> `
  --selection-label exact `
  --rank 1
```

## Direct-Basis Rerun Candidate Selection

Use preliminary benchmark CSVs to create per-state rerun inputs:

```powershell
python scripts/select_top_rerun_candidates.py `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_all_qutrit_u3_runs4_<timestamp>.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ghz3_all_qutrit_u3_runs4_<timestamp>.csv `
  --input-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_ame43_all_qutrit_u3_runs1_<timestamp>.csv `
  --top-k 10 `
  --run-id stage2_20260706
```

By default this writes one CSV per `state_name` under `artifacts/iqm_runs/processed/rerun_selection/<run_id>/`. The `candidate` rows are the unique Top-K non-baseline-equivalent candidates by depth ranking. Baseline-equivalent and unresolved rows are still kept in the same file as diagnostics with `selection_role` values such as `baseline_equivalent_excluded` and `unresolved_candidate`; they are not rerun by `from-old-csv`.

Rerun one state with the selected baseline plus candidates:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state ghz3 `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/processed/rerun_selection/stage2_20260706/direct_basis_ghz3_stage2_20260706_top10_rerun_candidates.csv `
  --iqm-backend garnet `
  --n-transpile-runs 20 `
  --jobs 4
```

Repeat the rerun command for each generated state CSV. The rerun selector always includes the chosen baseline row, so each state is compared against its own rerun baseline.

## IQM Direct-Basis Transpilation

Create `.env` in the repository root:

```env
IQM_SERVER_URL=https://resonance.iqm.tech/
IQM_TOKEN=replace-with-your-iqm-api-token
```

Run a small IQM-backed direct-basis benchmark:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --jobs 4
```

The `--iqm-backend` value is the IQM quantum computer name or alias. `garnet` is only an example. When this flag is present, the script loads one IQM backend for the whole run. For the default `state_preparation` workload, it compiles each candidate with strategies that preserve relative phases:

```text
preset_exact
transpile_to_iqm_exact
```

For each candidate and seed, the benchmark tries the selected strategies and keeps the best transpiled circuit by `(depth, two_qubit_gate_count, one_qubit_gate_count, size)`. The output CSV records the winning `iqm_transpiler_strategy` and `iqm_transpiler_seed`.

`preset_default` and `transpile_to_iqm_default` remove final RZ rotations, preserving computational-basis measurement probabilities but generally changing state fidelity. They are rejected for `state_preparation`, including with `--no-fidelity`. They remain available in the transpiler harness and in the default strategy set for `--ranking-workload bell_measurements`.

Optional transpiler controls:

```powershell
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --layout-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --routing-method sabre
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-use-metrics
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-strategy preset_exact
python scripts/run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --iqm-backend garnet --iqm-legacy-pass-manager
```

IQM output defaults to `artifacts/iqm_runs/raw`, and QPY exports default to `artifacts/iqm_runs/raw/quantum_circuits/<backend>/`.

## IQM Transpiler Harness

Use the harness to compare IQM-aware transpilation strategies for candidates
selected by earlier benchmark CSVs:

```powershell
python scripts/run_iqm_transpiler_harness.py `
  --state two_qutrit `
  --candidate-set from-old-csv `
  --old-csv artifacts/iqm_runs/raw/direct_basis_iqm_garnet_two_qutrit_from_old_csv_runs20_20260706_204350.csv `
  --iqm-backend garnet `
  --n-transpile-runs 3
```

Before transpilation, the harness deduplicates candidate matrices that differ
only by a global phase. The representative is transpiled and the removed
candidates remain traceable in the phase-audit CSV.

The harness only transpiles circuits. It does not submit jobs to IQM hardware.
It writes:

```text
artifacts/iqm_runs/processed/transpiler_harness/<run_id>/
  all_trials.csv
  best_by_candidate.csv
  candidate_global_phase_duplicates.csv
  strategy_statistics.csv
  pareto_ranked.csv
  state_equivalence_groups.csv
  recommended_circuits.csv
  summary.json
  quantum_circuits/<state>/<class>__<candidate>/
    F3_W.qpy
    CZ3_W.qpy
    graph_state_direct_basis.qpy
    graph_state_direct_basis_transpiled_<strategy>_seed<seed>.qpy
    E.npy
    W.npy
```

Pass `--quantum-circuits-dir` to override the artifact directory, or
`--no-export-quantum-circuits` to write only CSV/JSON outputs.

Built-in strategies:

```text
preset_default
preset_exact
transpile_to_iqm_default
transpile_to_iqm_exact
```

`best_by_candidate.csv` chooses the best successful trial by
`(depth, cz_count, r_count, size)` and flags warning thresholds such as
`depth_gt_100` and `cz_gt_50`. This is the legacy depth-first view and its
selection order is unchanged.

The statistical outputs are computed after IQM transpilation from the
successful rows in `all_trials.csv`. Each candidate/strategy pair remains a
separate statistical alternative across transpiler seeds. `pareto_ranked.csv`
assigns rank 1 to the nondominated front over mean 2Q-gate count, mean depth,
and depth standard deviation. Within each Pareto rank, `ideal_score` uses
weights 0.50/0.30/0.20 for those objectives; the score never overrides the
Pareto rank.

`state_equivalence_groups.csv` groups alternatives that prepare the same
compiled logical state. Physical costs are evaluated before this grouping,
and `recommended_circuits.csv` then keeps one recommendation per
state-equivalence group.

Existing `all_trials.csv` results can be analyzed again without rerunning the
harness:

```powershell
python scripts/analyze_iqm_transpiler_harness.py --all-trials artifacts/iqm_runs/processed/transpiler_harness/20260902_120000/all_trials.csv
```

Neither the harness CLI nor this standalone analysis CLI submits hardware
jobs.

To synthesize gates and transpile several encodings concurrently, use `--jobs`
on the IQM harness:

```powershell
python scripts/run_iqm_transpiler_harness.py `
  --state two_qutrit `
  --candidate-set all-qutrit-u3 `
  --iqm-backend garnet `
  --n-transpile-runs 20 `
  --strategy preset_exact `
  --strategy transpile_to_iqm_exact `
  --jobs 4 `
  --run-id optimized_f3_cz3_two_qutrit_parallel
```

`--jobs` defaults to 1. Each worker is a separate spawned process, so different
encodings can run CZ3 synthesis concurrently on Windows and Linux. Seeds and
strategies for one encoding remain sequential. The parent loads the IQM backend
once; workers rebuild its compilation targets from the same architecture and
calibration metrics without fetching a new calibration or carrying a hardware
client. Qiskit/Rayon nested parallelism is disabled inside workers, and each
BQSKit synthesis uses one worker. Only BQSKit's fixed-port startup handshake is
serialized; the searches themselves run concurrently.

Start with `--jobs 4` and adjust for available CPU and RAM. Gate-cache locking
still prevents duplicate synthesis of the same encoding. Completed candidates
are reported immediately; CSV row order and ranking tie-breaking retain input
order. All Pareto, equivalence, recommendation and summary reports are generated
after the workers finish, with `jobs` recorded in `summary.json`. Use a distinct
run ID for each run. Colliding candidate artifact directories are rejected
before worker startup. At most `jobs` candidates are in flight; an exception or
interrupt stops further scheduling and cancels pending work. Already active
candidates are allowed to finish or handle their interrupt before shutdown, so
stopping can still take time but will not drain the entire candidate pool.
An already running process must be restarted to use this
option; completed validated gate-cache entries can be reused. For Python API
calls with `jobs > 1`, use an `if __name__ == "__main__":` entry-point guard and
module-level, serializable custom strategy runners.

The IQM harness prints a progress block for each active candidate every 15
seconds, including its PID, elapsed time, stage duration and observed CPU-time
increase across the worker and its BQSKit descendants. Change the interval with
`--progress-interval 10`, or disable it with `--progress-interval 0`. This works
with both `--jobs 1` and parallel runs and uses plain lines suitable for
PowerShell and redirected cloud logs.

Stages distinguish gate-cache lock waits, cached-gate validation, F3 synthesis,
BQSKit startup-lock waits, runtime startup, CZ3 synthesis, exports and each
transpilation strategy/seed. Transpilation displays a completed-trial bar and
remaining trial count (20 seeds and 2 strategies mean 40 trials per candidate).
CZ3 synthesis has no fixed total or trustworthy ETA, so it shows `ETA unknown`
instead of a fabricated percentage. Periodic output means the monitor is alive;
an increasing CPU time means observed computation, not guaranteed optimizer
improvement. Zero CPU activity may mean a lock or other wait, and is not itself
proof of a hang. The progress monitor does not interrupt slow syntheses. A running
older process must be restarted to load the new reporting code.

## Optimized F3 and CZ3 in the benchmark

The existing direct-basis benchmark and the IQM/PiastQ transpiler harnesses
now use optimized F3 and CZ3 by default for every encoding. No extra flag is
needed. Each circuit prepares encoded logical zero, applies the optimized F3
to each qutrit, and composes the synthesized CZ3 on each graph edge.

F3 uses the analytic leakage phase for monomial encodings. For other bases,
including dense `B = B_s W`, a numerical phase solve enforces the two-CNOT
invariant before local synthesis. Every accepted F3 must have **at most two
CNOTs** and code-space error and leakage at most `1e-10`. The CNOT limit is
for the isolated F3 block, before hardware routing or whole-circuit optimization.
The numerical criterion is described by
[Shende, Bullock and Markov](https://arxiv.org/abs/quant-ph/0308045).

CZ3 follows `notebooks/CZ3_bqckit_optimalization.ipynb`: a BQSKit `StateSystem`
maps all nine columns of `B2 = kron(B, B)` to `B2 @ CZ3_logical`, leaving the
other seven dimensions unconstrained. It uses `U3Gate` and `CZGate`,
`optimization_level=2`, `max_synthesis_size=4`, `synthesis_epsilon=1e-8`, and
`seed=0`. Conversion to Qiskit reconciles the matrix bit order. The explicit
U3/CZ circuit is composed into the graph circuit and exported as `CZ3_W.qpy`.
`F3_W.qpy` likewise contains the validated elementary F3 decomposition.

Every benchmark row records the isolated CZ3 metrics:

- `E_norm = ||phase * U @ B2 - B2 @ CZ3_logical||_F`, with a **single global
  phase shared by all nine columns**, so relative-phase errors are detected.
- `L_norm = ||(I - B2 @ B2.conj().T) @ U @ B2||_F`.
- `N_2q`: the number of two-qubit gates in that synthesized CZ3 block.

These are the unnormalized Frobenius norms used in the notebook. Both CZ3
norms must be at most `1e-5`. The analogous F3 fields are `f3_E_norm`,
`f3_L_norm`, and `f3_N_2q`; phase, synthesis method, seed, epsilon, synthesis
time, cache status, and gate-library version are also recorded. Existing
whole-circuit gate counts keep their original meaning. Main benchmark
`fidelity` includes synthesis error by comparing against the ideal logical
state preparation. A failed acceptance check produces `gate_validation_failed`
and skips transpilation for that candidate; it never selects a legacy gate.

Validated gates are cached under `artifacts/direct_basis_runs/optimized_gates/`
by the exact encoding, synthesis settings and dependency versions, and reused
across states, seeds and approximation thresholds. Cache loads recheck both
errors and gate counts. Per-encoding OS file locks coordinate independent
benchmark processes, and complete cache directories are published by atomic
rename. Invalid or incomplete cache entries are rebuilt. The cache format is
versioned so older, unlocked writers cannot interfere with the new entries.
BQSKit searches within one process remain serialized; the existing candidate
jobs still handle transpilation.
The first synthesis for a new basis can take substantially longer than a
transpiler trial. BQSKit is included in the project dependencies.

The historical `--compare-optimal-f3-leakage` diagnostic remains available for
reproducing old phase comparisons, but its results do not supply the primary
gates or determine ranking. The optimized library is always the primary input.

To also remove encodings that differ only by a global phase, add
`--deduplicate-global-phase`. This reuses the IQM harness deduplication, prefers
`baseline` as the representative of its group, and runs before transpilation
on the pool remaining after candidate filters and limits. For the complete
`v2-stage1` / `monomial_full` pool, 649 entries become 216 representatives
(including the baseline). Relative phases and different physical supports
remain distinct. Unsupported candidates remain in the pool for error reporting.

For the combined comparison on IQM Garnet:

```powershell
python scripts/run_direct_basis_benchmarks.py `
  --state two_qutrit `
  --candidate-set v2-stage1 `
  --candidate-class monomial_full `
  --deduplicate-global-phase `
  --compare-optimal-f3-leakage `
  --iqm-backend garnet `
  --n-transpile-runs 20
```

Deduplication is opt-in and also works without the F3 comparison or IQM backend.
The benchmark CSV contains representatives only; a sibling
`<output-stem>_global_phase_duplicates.csv` records each removed candidate, its
representative, and the detected phase factor. The audit is written before
transpilation, even when no duplicates are found. Any Top-K selection operates
on the remaining representatives using the historical ranking described below.
The IQM backend currently runs candidates serially even if `--jobs` is greater
than one.
