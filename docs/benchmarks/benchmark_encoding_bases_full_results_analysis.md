# Analysis of `data/benchmarks/benchmark_encoding_bases_full_results.csv`

## Scope

This note summarizes the benchmark file `data/benchmarks/benchmark_encoding_bases_full_results.csv` with focus on:

- overall result quality,
- ranking of encoding classes by transpiled circuit depth,
- best-performing candidates,
- links between encoding properties and transpilation cost.

## Dataset overview

- Total rows: `768`
- Successful rows: `768`
- Failed rows: `0`
- Status distribution: all rows have `status = ok`
- Benchmark size per candidate: `20` transpilation runs each

The dataset is therefore complete and internally consistent enough for direct class-by-class comparison.

## Executive summary

The baseline encoding `E_old` is clearly the global winner. Its `mean_depth = 53.9` and `best_depth = 47` are far ahead of every alternative.

Among non-baseline encodings, the best results come from very structured, low-entanglement families:

- `monomial`
- `clifford_wh`
- `structured_entangling` with trivial parameters
- `local_ry_only`

The best non-baseline result is `mean_depth = 81.15`, which is still `27.25` depth units worse than baseline, or about `50.6%` higher.

The broad pattern is very strong: lower two-qubit gate count almost directly predicts lower depth. The correlation between `mean_depth` and `mean_two_qubit_gate_count` is `0.9635`.

Highly entangling or fully generic encodings are usually much worse. Several classes collapse to the same transpiled cost plateau around:

- `mean_depth = 122.55`
- `best_depth = 108`
- `mean_two_qubit_gate_count = 60.5`

This suggests that for those classes the transpiler sees essentially the same implementation complexity, despite different encoding parameters.

## Best global candidates

### Top 10 by `mean_depth`

| Rank | Class | Candidate | Mean depth | Best depth | Mean 2Q count |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `baseline` | `E_old` | 53.90 | 47 | 33.05 |
| 2 | `monomial` | `P012_ph000` | 81.15 | 74 | 42.50 |
| 3 | `clifford_wh` | `X0Z0F0` | 81.15 | 74 | 42.50 |
| 4 | `structured_entangling` | `t0.00_p0.00_a0.00` | 81.15 | 74 | 42.50 |
| 5 | `monomial` | `P012_ph110` | 81.75 | 74 | 42.50 |
| 6 | `monomial` | `P012_ph220` | 81.75 | 74 | 42.50 |
| 7 | `monomial` | `P012_ph101` | 81.95 | 76 | 42.50 |
| 8 | `monomial` | `P012_ph202` | 81.95 | 76 | 42.50 |
| 9 | `local_ry_only` | `ry_0.000_3.142` | 82.00 | 76 | 42.50 |
| 10 | `local_ry_only` | `ry_3.142_0.000` | 82.05 | 74 | 42.50 |

### Best non-baseline candidates

The best non-baseline plateau is `81.15`, reached by three structurally different but effectively equivalent points:

- `monomial / P012_ph000`
- `clifford_wh / X0Z0F0`
- `structured_entangling / t0.00_p0.00_a0.00`

This is a useful signal: once the encoding reduces to a very simple structured transformation, the transpiled hardware cost becomes competitive with the best non-baseline families.

## Ranking by class

### Best candidate per class

| Class | Best candidate | Best mean depth | Best depth | Mean 2Q count | Avg entanglement | Overlap with old codespace |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `E_old` | 53.90 | 47 | 33.05 | 0.000 | 1.000 |
| `clifford_wh` | `X0Z0F0` | 81.15 | 74 | 42.50 | 0.000 | 1.000 |
| `monomial` | `P012_ph000` | 81.15 | 74 | 42.50 | 0.000 | 1.000 |
| `structured_entangling` | `t0.00_p0.00_a0.00` | 81.15 | 74 | 42.50 | 0.000 | 1.000 |
| `local_ry_only` | `ry_0.000_3.142` | 82.00 | 76 | 42.50 | 0.000 | 0.667 |
| `local_general_su2` | `lsu2_000` | 101.65 | 91 | 51.50 | 0.000 | 0.699 |
| `finer_structured` | `fine_t0.00_p1.32_a1.10` | 105.00 | 94 | 51.50 | 0.000 | 0.817 |
| `real_orthogonal` | `real_019` | 107.65 | 96 | 54.50 | 0.203 | 0.720 |
| `fourier_like` | `D010_F3_D110` | 111.30 | 102 | 54.50 | 0.550 | 1.000 |
| `two_cz_ansatz` | `2cz_045` | 114.50 | 101 | 57.50 | 0.770 | 0.763 |
| `entangling_isometry` | `ent_000` | 122.55 | 108 | 60.50 | 0.388 | 0.704 |
| `haar_random_isometry` | `haar_000` | 122.55 | 108 | 60.50 | 0.497 | 0.729 |
| `householder_random` | `rand_000` | 122.55 | 108 | 60.50 | 0.456 | 1.000 |
| `near_identity` | `nearid_eps0.01_00` | 122.55 | 108 | 60.50 | 0.001 | 1.000 |
| `perturbed_isometry` | `pert_eps0.01_00` | 122.55 | 108 | 60.50 | 0.003 | 1.000 |

### Class-level averages

Sorted by minimum `mean_depth` inside each class:

| Class | Candidates | Class mean of mean depth | Best mean depth in class | Mean 2Q count |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1 | 53.90 | 53.90 | 33.05 |
| `clifford_wh` | 27 | 104.95 | 81.15 | 53.28 |
| `monomial` | 120 | 101.19 | 81.15 | 53.30 |
| `structured_entangling` | 125 | 99.99 | 81.15 | 49.20 |
| `local_ry_only` | 99 | 90.68 | 82.00 | 46.14 |
| `local_general_su2` | 30 | 102.48 | 101.65 | 51.50 |
| `finer_structured` | 100 | 105.20 | 105.00 | 51.50 |
| `real_orthogonal` | 20 | 111.33 | 107.65 | 55.70 |
| `fourier_like` | 64 | 121.08 | 111.30 | 59.38 |
| `two_cz_ansatz` | 50 | 114.79 | 114.50 | 57.50 |
| `entangling_isometry` | 20 | 122.55 | 122.55 | 60.50 |
| `haar_random_isometry` | 20 | 122.55 | 122.55 | 60.50 |
| `householder_random` | 20 | 122.55 | 122.55 | 60.50 |
| `near_identity` | 40 | 122.55 | 122.55 | 60.50 |
| `perturbed_isometry` | 32 | 122.55 | 122.55 | 60.50 |

## Main patterns

### 1. Baseline remains unmatched

`E_old` is not just the best candidate; it is in a separate regime. The nearest competitors are roughly `50%` deeper on average.

### 2. Structure beats genericity

The strongest non-baseline classes are those with strong algebraic structure:

- monomial permutations and phases,
- Clifford/Weyl-Heisenberg-like constructions,
- simple structured-entangling points,
- local `Ry`-only families.

These families can still remain close to the old codespace or at least avoid introducing costly entangling structure into the synthesized circuit.

### 3. Two-qubit count is the dominant cost driver

Correlations with `mean_depth`:

- `mean_two_qubit_gate_count`: `0.9635`
- `best_two_qubit_gate_count`: `0.9639`
- `mean_size`: `0.9907`
- `best_depth`: `0.9962`

The main message is straightforward: if a basis change increases the two-qubit gate count, depth rises almost automatically.

### 4. Codeword entanglement is usually harmful here

Global correlation between `avg_codeword_entanglement` and `mean_depth` is `0.5599`.

A particularly sharp split appears when comparing zero-entanglement and nonzero-entanglement candidates:

- zero entanglement: average `mean_depth = 99.50`
- nonzero entanglement: average `mean_depth = 119.78`

So in this benchmark, codeword entanglement is generally associated with noticeably worse transpilation cost.

### 5. Overlap with the old codespace helps only weakly

The correlation between `overlap_with_old_codespace` and `mean_depth` is only `0.2432`.

That means geometric closeness to the old codespace alone is not enough. Some candidates remain close to the old codespace yet still land on the worst transpilation plateau. The detailed circuit structure matters much more than overlap by itself.

## Old-codespace-only vs full-space encodings

Using the `uses_old_codespace_only` flag:

- old-codespace-only candidates: `233`
- full-space candidates: `535`

Aggregate comparison:

| Group | Count | Average mean depth | Best mean depth | Average mean 2Q count |
| --- | ---: | ---: | ---: | ---: |
| Old-codespace-only | 233 | 108.64 | 53.90 | 55.45 |
| Full-space / not old-only | 535 | 105.95 | 82.00 | 52.59 |

Interpretation:

- On average, the full-space families are slightly better than the old-only families.
- But this is driven by a few strong structured families such as `structured_entangling` and `local_ry_only`.
- The best single result overall still comes from the original old-codespace baseline.

## Class stability and sensitivity

Some classes vary strongly across candidates, which means parameter tuning matters:

- `clifford_wh`
- `monomial`
- `structured_entangling`
- `local_ry_only`

Other classes are nearly or exactly flat, meaning tuning buys almost nothing:

- `entangling_isometry`
- `haar_random_isometry`
- `householder_random`
- `near_identity`
- `perturbed_isometry`

For these flat classes, every tested candidate produces essentially the same transpiled cost:

- `mean_depth = 122.55`
- `best_depth = 108`
- `mean_two_qubit_gate_count = 60.5`

This is one of the most useful practical conclusions in the table: large parts of the generic-isometry search space appear not worth exploring further under the present transpilation pipeline.

## Practical recommendations

If the goal is low transpiled depth, the most promising families to continue exploring are:

1. `monomial`
2. `clifford_wh`
3. `structured_entangling`
4. `local_ry_only`

If the goal is scientific comparison rather than optimization, then the flat bad-performing classes are still useful as control groups, because they define a clear "generic hard encoding" regime.

If compute budget is limited, the least promising classes for further random search are:

- `entangling_isometry`
- `haar_random_isometry`
- `near_identity`
- `perturbed_isometry`

They currently show no evidence of producing anything meaningfully better than the same worst-case plateau.

## Bottom line

The benchmark strongly favors simple, low-entanglement, highly structured basis changes. The original encoding `E_old` remains the best option by a wide margin. When moving away from the baseline, the only families that stay reasonably competitive are those whose circuit realization remains close to cheap structured transformations. Generic or entangling isometries do not pay off under the present transpilation setting.
