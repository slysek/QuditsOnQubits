# encoding_search_v2

Separate staged pipeline for comparing qutrit-on-two-qubit encoding bases.

Stage 1 benchmarks the existing base circuit plus final local `W` gates
(`append_w`). The default candidate pool is:

- `baseline`: `E_old`
- `monomial_full`: all monomial embeddings using any 3 of 4 computational states,
  all logical permutations, and phases from `{1, omega, omega^2}`
- `product`: a finite local one-qubit unitary library for `E_new = (U kron V) E_old`

Optional Stage 1 classes:

- product SU(2) grid via `--include-product-grid`
- bounded near-identity perturbations via `--include-near-identity`

Stage 2 reads a Stage 1 ranking CSV, selects stable `(class_name, candidate_name)`
pairs, and benchmarks only those candidates with
`prepared_w_then_conjugated_entanglers`: local `W|+>` state preparation plus
explicit `(W kron W) CZ (Wdag kron Wdag)` entangler blocks.

Before either stage benchmarks candidates, the pipeline keeps the explicit
`baseline / E_old` reference and skips other candidates that are equivalent to
baseline within `--atol` / `--rtol`. The practical checks cover exact embedding
equality, equality up to one global phase, and `W` equal to identity up to
tolerance. Skipped rows remain in the full results CSV with
`status=skipped_baseline_equivalent`, `is_trivial_identity`,
`is_baseline_equivalent`, and `skip_reason`.

Examples:

```powershell
python -m encoding_search_v2 --state two_qutrit --stage 1 --jobs 32
python -m encoding_search_v2 --state ghz3 --stage 1 --jobs 32
python -m encoding_search_v2 --state ame43 --stage 1 --jobs 32
```

Then run Stage 2 from the Stage 1 ranking:

```powershell
python -m encoding_search_v2 --state ghz3 --stage 2 --jobs 32 --top-k 30 --ranking-csv encoding_search_v2\results\ghz3\stage1\encoding_search_v2_ghz3_stage1_top30_by_depth.csv
```

For a quick candidate-count check without transpilation:

```powershell
python -m encoding_search_v2 --state ghz3 --stage 1 --dry-run
```
