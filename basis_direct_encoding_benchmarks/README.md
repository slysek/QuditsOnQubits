# Direct Basis Encoding Benchmarks

This folder contains the benchmark path where a qutrit encoding is used
directly when preparing graph states.

## Old Method: Append W

The legacy method prepares the graph state in the canonical `Z` encoding:

```text
|0> -> |00>
|1> -> |01>
|2> -> |10>
```

Then it appends physical two-qubit `W` gates on every encoded qutrit.  That
means the cost of `W` is counted as an explicit final basis-change layer.

## New Method: Direct Basis Encoding

The new method does not append `W` or another encoding-change gate as a
separate physical layer.  Instead, the candidate encoding defines the
physical qutrit codewords directly.

For a passive basis change inside the canonical code space, the encoding is

```text
|tilde_j> = W |j>
E_Z = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
E_W = E_Z @ W
```

The benchmark also accepts general encoding isometries:

```text
E_new in C^(4x3)
E_new^dag @ E_new = I_3
```

These candidates may change the encoded subspace itself, for example by
using the physical level `|11>` as one of the qutrit codewords.

For each qutrit, the local target is prepared directly:

```text
physical target = E_new @ |+>
```

For each logical qutrit gate `G`, the physical two-qubit gate acts as `G`
on the encoded subspace and as identity on its orthogonal complement:

```text
G_phys = E_new @ G @ E_new^dag + (I_4 - E_new @ E_new^dag)
```

For each graph edge, the logical qutrit `CZ` is embedded analogously:

```text
CZ_phys = (E_new kron E_new) @ CZ @ (E_new kron E_new)^dag
          + (I_16 - (E_new kron E_new) @ (E_new kron E_new)^dag)
```

This means the encoding is not charged as a final appended layer.  Its cost
can still appear because the direct local state and the encoded entangling
gate may compile to harder physical circuits.

## Files

- `math_utils.py` constructs `E_Z`, `E_W`, `|+_W>`, qutrit `CZ`,
  encoded one- and two-qutrit gates, and leakage-complement identities.
- `candidates.py` generates small sanity candidates and converts
  `E_new = E_Z @ W` candidates from the legacy and `encoding_search_v2`
  pools into direct encoding candidates.
- `circuits.py` builds direct-basis graph-state circuits.
- `benchmark.py` transpiles direct circuits and writes CSV-compatible rows.
- `comparison.py` joins old and direct CSV files and computes metric deltas.

## Run

All commands are launched from the repository root.

### Candidate Sets

Use `--candidate-set` to choose what is benchmarked:

- `sanity` - small smoke-test set: `I`, `F3`, `F3dg`, a few diagonal
  phases, permutations, and optional Haar-random `U(3)` bases.
- `all-qutrit-u3` - the full benchmark set.  The name is kept for CLI
  compatibility, but this now includes the full `encoding_search_v2`
  stage-1 pool, including candidates that change the encoded subspace, plus
  non-duplicated legacy qutrit classes.  The legacy `monomial_old_codespace`
  class is omitted here because it is exactly the `sup012` subset of
  `monomial_full`.
- `old_qutrit` - legacy qutrit `U(3)` classes only: baseline,
  `monomial_old_codespace`, `fourier_like`, sampled Householder/Haar
  `U(3)`, and `clifford_wh`.
- `v2-stage1` - the raw `encoding_search_v2` stage-1 pool converted for
  this direct benchmark.
- `from-old-csv` - regenerate only candidates listed in an existing CSV.

### Common Flags

- `--state two_qutrit|ghz3|ame43|ghz_star` selects the benchmark state.
- `--n-qutrits N` is required for variable-size `ghz_star` runs.
- `--random-count N` controls how many random Haar bases are added to the
  `sanity` set.
- `--seed N` controls random candidate generation for the `sanity` set.
- `--candidate-class CLASS` benchmarks only candidates with this
  `class_name`, for example `monomial_full` or `product`.  Pass it multiple
  times, or use a comma-separated list, to include several classes.  The
  `baseline` candidate is included automatically whenever this filter is
  used, and it does not count against `--limit-candidates`.
- `--n-transpile-runs N` controls how many transpiler seeds are tried per
  candidate.  Use `1` for a quick check and `20` for the standard full
  benchmark.
- `--limit-candidates N` truncates the loaded candidate list.  Do not pass
  it for a full benchmark.
- `--old-csv PATH` is required with `--candidate-set from-old-csv`.
- `--output-dir DIR` writes the timestamped result CSV under `DIR`.
- `--output-csv PATH` writes to an exact result CSV path.
- `--no-fidelity` skips fidelity calculation.
- `--max-fidelity-qubits N` skips fidelity once the transpiled circuit has
  more than `N` qubits.
- `--local-line-coupling` uses a nearest-neighbor line over the physical
  qubits, useful for quick local checks.
- `--quantum-circuits-dir DIR` chooses where QPY exports are written.
- `--no-export-quantum-circuits` disables per-candidate QPY exports.

### Quick Smoke Runs

Fast sanity run on a local four-qubit line:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --random-count 1 --n-transpile-runs 1 --local-line-coupling --max-fidelity-qubits 4
```

Quick check of the full candidate loader without exporting QPY files:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set all-qutrit-u3 --limit-candidates 5 --n-transpile-runs 1 --local-line-coupling --no-export-quantum-circuits --max-fidelity-qubits 4
```

Run only selected encoding classes:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set all-qutrit-u3 --candidate-class monomial_full --n-transpile-runs 10
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set all-qutrit-u3 --candidate-class monomial_full --candidate-class product --n-transpile-runs 10
```

### Full Benchmarks

Use `all-qutrit-u3` without `--limit-candidates` for the full benchmark.
This benchmarks the full `encoding_search_v2` stage-1 pool and the legacy
qutrit classes that are not already covered by that pool.  The stage-1 pool
includes `monomial_full` and `product` candidates that may change the encoded
subspace, not only passive `U(3)` basis changes inside the canonical code
space.  `monomial_old_codespace` is not added separately because every one
of its candidates appears in `monomial_full` with support `sup012`.

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set all-qutrit-u3 --n-transpile-runs 20
python run_direct_basis_benchmarks.py --state ghz3 --candidate-set all-qutrit-u3 --n-transpile-runs 20
python run_direct_basis_benchmarks.py --state ame43 --candidate-set all-qutrit-u3 --n-transpile-runs 20
```

For longer GHZ/star variants, pass the state and size explicitly:

```bash
python run_direct_basis_benchmarks.py --state ghz_star --n-qutrits 5 --candidate-set all-qutrit-u3 --n-transpile-runs 20
```

### Raw encoding_search_v2 Stage-1 Pool

To run only the raw `encoding_search_v2` stage-1 candidate pool, use:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set v2-stage1 --n-transpile-runs 20
```

The v2 candidate controls are also available:

- `--max-monomial-full N` limits `monomial_full` generation.
- `--max-product N` limits the discrete product-unitary library.
- `--include-product-grid` adds the SU(2) product grid.
- `--max-product-grid N` limits that grid.
- `--product-grid-phase-steps N` and `--product-grid-polar-steps N`
  control the grid resolution.
- `--include-near-identity` adds near-identity isometries.
- `--near-identity-samples-per-eps N` controls near-identity samples.
- `--near-identity-seed N` controls the near-identity RNG seed.

Example bounded v2 run:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set v2-stage1 --max-monomial-full 100 --max-product 32 --include-product-grid --max-product-grid 16
```

For a full raw v2-stage1 run, omit all `--max-*` and `--limit-candidates`
flags.  Add `--include-product-grid` or `--include-near-identity` only when
you also want those opt-in v2 classes included.

### CSV-Driven Runs

Run direct benchmarks for candidates listed in an old CSV:

```bash
python run_direct_basis_benchmarks.py --state ame43 --candidate-set from-old-csv --old-csv encoding_search_v2/results/ame43/stage2/encoding_search_v2_ame43_stage2_top30_by_depth.csv --n-transpile-runs 1
```

Compare:

```bash
python compare_old_vs_direct_basis.py --old-csv encoding_search_v2/results/ame43/stage2/encoding_search_v2_ame43_stage2_top30_by_depth.csv --direct-csv basis_direct_encoding_benchmarks/results/direct_basis_benchmarks_TIMESTAMP.csv
```

Results are written to timestamped files under:

```text
basis_direct_encoding_benchmarks/results/
```

By default, result CSV names include the state, candidate set, optional
candidate limit, transpiler run count, and timestamp:

```text
direct_basis_<state>_<candidate_set>[_limitN]_runs<RUNS>_<YYYYMMDD_HHMMSS>.csv
```

Examples:

```text
direct_basis_two_qutrit_all_qutrit_u3_runs20_20260511_153000.csv
direct_basis_ghz_star_5_all_qutrit_u3_runs20_20260511_153000.csv
direct_basis_two_qutrit_v2_stage1_limit5_runs1_20260511_153000.csv
```

Use `--output-csv <path>` when you want to choose the exact filename.

For each successfully built candidate, the runner also exports QPY circuits
under:

```text
basis_direct_encoding_benchmarks/quantum_circuits/<state>/<class>__<candidate>/
```

Each candidate folder contains:

```text
F3_W.qpy                         # two-qubit embedding of F3^(W) = W F3 W^dag
CZ3_W.qpy                        # four-qubit embedding of CZ3^(W)
graph_state_direct_basis.qpy     # full direct-basis graph-state preparation
```

Use `--quantum-circuits-dir <path>` to choose another export root, or
`--no-export-quantum-circuits` to disable these QPY files for a run.
