# Direct Basis Encoding Benchmarks

This folder contains the benchmark path where a qutrit unitary `W in U(3)`
defines the logical qutrit basis directly.

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

The new method does not append `W` as a separate physical gate.  Instead,
`W` defines the qutrit basis:

```text
|tilde_j> = W |j>
E_Z = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
E_W = E_Z @ W
```

For each qutrit, the local target is prepared directly:

```text
|+_W> = W |+>
physical target = E_Z @ W @ |+>
```

For each graph edge, the qutrit entangler is conjugated:

```text
CZ^(W) = (W kron W) CZ (W^dag kron W^dag)
```

The four-qubit physical gate is built by embedding `CZ^(W)` into the
canonical code subspace
`span{|00>, |01>, |10>} kron span{|00>, |01>, |10>}`.  The leakage subspace
where either encoded qutrit is `|11>` is kept as identity and is not coupled
to the code subspace.

This means `W` is not charged as a final appended layer.  Its cost can still
appear because the direct local state `|+_W>` and the edge gate `CZ^(W)` may
compile to harder physical circuits.

## Files

- `math_utils.py` constructs `E_Z`, `E_W`, `|+_W>`, qutrit `CZ`,
  `CZ^(W)`, and the 16x16 leakage-identity embedding.
- `candidates.py` generates small sanity candidates and converts old
  `E_new = E_Z @ W` candidates into direct `U(3)` candidates.
- `circuits.py` builds direct-basis graph-state circuits.
- `benchmark.py` transpiles direct circuits and writes CSV-compatible rows.
- `comparison.py` joins old and direct CSV files and computes metric deltas.

## Run

Small sanity run on a local four-qubit line:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set sanity --random-count 1 --n-transpile-runs 1 --local-line-coupling --max-fidelity-qubits 4
```

Full direct-basis run over all regenerated qutrit `U(3)` basis candidates,
without taking a top-k list from an old CSV:

```bash
python run_direct_basis_benchmarks.py --state two_qutrit --candidate-set all-qutrit-u3 --n-transpile-runs 20
python run_direct_basis_benchmarks.py --state ghz3 --candidate-set all-qutrit-u3 --n-transpile-runs 20
python run_direct_basis_benchmarks.py --state ame43 --candidate-set all-qutrit-u3 --n-transpile-runs 20
```

For longer GHZ/star variants, pass the state and size explicitly:

```bash
python run_direct_basis_benchmarks.py --state ghz_star --n-qutrits 5 --candidate-set all-qutrit-u3 --n-transpile-runs 20
```

`all-qutrit-u3` benchmarks candidates where the old embedding is truly
`E_Z @ W` for a qutrit `W in U(3)`: baseline, old-code-space monomial
phase/permutation bases, Fourier-like bases, sampled Haar/Householder
`U(3)` bases, and qutrit Clifford-Weyl-Hadamard candidates.  General
4x3 isometries that use the physical `|11>` level as part of the code are
not passive qutrit basis changes, so they are not included in this full
`U(3)` command.

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
