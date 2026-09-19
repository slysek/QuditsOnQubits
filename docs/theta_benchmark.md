# E_theta benchmark: F3 and CZ3 continuation

The benchmark studies the family of isometries

```text
E_theta = [[ cos(theta), 0, 0],
           [          0, 1, 0],
           [          0, 0, 1],
           [-sin(theta), 0, 0]]
l_theta = [sin(theta), 0, 0, cos(theta)]
```

over the range `0 <= theta <= pi/4`. All qutrits of a given point share the same encoding. The computation is local; the CLI does not run jobs on a QPU.

F3 is given a free leakage phase in the extension `E_theta F3 E_theta† + exp(i alpha)|l_theta><l_theta|`. The existing algorithm selects `alpha`; the result must use at most two two-qubit CZ gates and satisfy the action and leakage tolerances. The control cost at `alpha = 0` is recorded separately.

CZ3 starts from the pinned circuit `experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy`. Its SHA-256 is `a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791`; the neighbouring file `E.npy` must contain `eye(4, 3)`. The original has six CZ and eleven U3, i.e. 33 angle parameters. We add no gates and no connections. We preserve the qubit order and the global phase.

## Running

Run the PowerShell commands from the repository or worktree directory that contains an existing `.venv`. The script adds the local `src` relative to its own path; an explicit `PYTHONPATH` is also useful when working with the API and the tests.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --help
```

A pilot over the first three points of the target 41-point grid, with the full numerical settings:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode gates --theta-points 41 --limit-points 3 --output-dir artifacts/theta_pilot
```

`--limit-points 3` selects a prefix of the 41-point grid; it does not turn it into three points stretched out to `pi/4`. It is part of the immutable configuration. The pilot and the full run are saved in separate directories.

A full run of the gates and of the states `two_qutrit`, `ghz3`, `ame43`:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode all --output-dir artifacts/theta_full
```

Resuming after the process was interrupted, with the configuration read from the manifest:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --resume artifacts/theta_full --mode all
```

A separate full-circuit stage for existing, saved gate results:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode circuits --resume artifacts/theta_pilot
```

An equivalent form of the last command uses `--mode circuits --output-dir artifacts/theta_pilot`. The `circuits` mode requires an existing manifest and does not re-synthesize the gates. The `all` mode can extend a finished `gates` stage with full circuits within the same run. `--resume` must not be combined with `--output-dir`. Explicitly supplied, contradictory configuration parameters are rejected; they are not ignored.

If the environment is already activated, `python` can be used instead of `.venv/Scripts/python.exe`. The POSIX equivalent is:

```bash
PYTHONPATH=src python scripts/run_theta_benchmark.py --mode all --output-dir artifacts/theta_full
```

A new run requires a new or empty directory. Without `--output-dir` a separate directory is created under `artifacts/theta_continuation/`, identified by a UTC timestamp and a random suffix. On Windows, custom programs that use the runner require the `if __name__ == '__main__':` guard; the provided CLI includes it for the BQSKit runtime.

## Protocol parameters

| Argument | Default | Meaning |
| --- | --- | --- |
| `--mode` | `all` | `gates`, `circuits` or both stages |
| `--theta-points` | `41` | Uniform grid from 0 to pi/4, at least 2 points |
| `--limit-points` | full range | Prefix of the grid, from 1 to `theta-points` |
| `--max-nfev` | `3000` | Angle-fitting budget within a single attempt |
| `--max-subdivisions` | `4` | Maximum depth of step halving, from 0 to 4 |
| `--f3-tolerance` | `1e-10` | Threshold for both F3 norms |
| `--cz3-tolerance` | `1e-5` | Threshold for both CZ3 norms |
| `--transpiler-seeds` | `0 1 2` | Explicit set of distinct transpilation seeds |
| `--optimization-level` | `3` | Transpilation level, from 0 to 3 |
| `--baseline-qpy` | the pinned file above | Local QPY with a single four-qubit circuit |
| `--baseline-sha256` | the pinned hash above | Exact SHA-256 of the source QPY |
| `--states` | `two_qutrit ghz3 ame43` | States for the full-circuit stage |

The baseline and the candidates are transpiled identically: `basis_gates=['u','cz']`, full connectivity, `approximation_degree=1.0`. Among valid circuits the winner is decided, in order, by the smallest CZ count, depth, 1q gate count and seed. The cost of the original template and the cost after transpilation are separate quantities. Routing on a specific processor is not part of this comparison.

For CZ3, `B_theta = E_theta tensor E_theta`. The fit covers the action `C(p) B_theta = exp(i gamma) B_theta CZ3`, with a single shared global phase for all nine columns. The final `E_norm` and `L_norm` are computed independently of the optimizer status. A separate phase per column is not allowed: it could hide incorrect logical phases.

After a failed direct attempt the runner halves the step down to the given depth. It also records intermediate attempts as `requested_grid_point=false`; it does not append them to the primary grid of results. The fallback uses the existing BQSKit StateSystem and its `synthesis_epsilon=1e-8`. That parameter does not replace the final CZ3 norm tolerance.

The fallback is triggered by an invalid fit, an invalid transpilation, or by exceeding the comparable baseline cost. The cheapest valid circuit among the available attempts is selected. A valid result more expensive than the baseline is marked `above_baseline`; the absence of a valid result is marked `failed`.

A BQSKit result does not change the continuation template. When the template was valid before transpilation, its parameters may remain the starting point of the next point even if the fallback is chosen. Otherwise the runner returns to the actual last valid state of the branch, including an intermediate point. Neighbouring theta values are executed sequentially.

## Artifacts and report

The manifest persists the configuration, the baseline source, the fingerprint, the library versions and the source hashes. Every completed bundle has a `complete.json` with the JSON, NPY and QPY hashes. A resume reads the same branch parameters; it does not reconstruct them from rounded tables. A corrupted completed bundle is an artifact error, not an optimization failure.

| File or directory | Contents |
| --- | --- |
| `manifest.json` | Immutable protocol and provenance |
| `baseline/` | Copy of the baseline, its transpilation data, template and parameter description |
| `points/00000/` etc. | Results for the target theta, circuits, matrices, all attempts and the branch state |
| `circuits/{state}/{index}/` | Full circuits from the saved F3/CZ3 library and verified metrics |
| `checkpoints/latest.json` | Last persisted resume state |
| `points.csv` | All completed points, including `failed` and `above_baseline` |
| `attempts.csv` | Attempts, intermediate points, actual parents and fit metrics |
| `parameters.csv` | Angle, instruction, qubit, template, raw and separately unwrapped parameters |
| `operator_distances.csv` | Distances of the local U3 between neighbouring valid points |
| `full_circuits.csv` | Costs, statuses and fidelities of the full circuits |
| `figures/` | Standalone PNG and PDF figures, for further analysis or publication |
| `report.md` | Report with the protocol, results, limitations and figures |

The CLI prints the current stage and the final report path. Exit code `0` means the protocol completed, including when scientifically failed points occurred. Code `2` means a configuration or artifact error, `1` an unexpected runtime error, and `130` an interruption by the user.

Regenerating the report without running the synthesis:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -c "from qudits_on_qubits.benchmarks.theta_continuation.report import generate_report; print(generate_report('artifacts/theta_full'))"
```

The report reads all completed bundles through the validating artifact store. The CSV files and the figures are derived data and can be regenerated from the JSON/NPY/QPY.

## Preparation and block costs

The report tabulates the cost of preparing the encoded zero, of all F3 blocks and of all CZ3 blocks within a given full circuit. These are the summed costs of separately compiled components before the blocks are fused, under the same exact U/CZ transpilation. Next to the sum stands the actual cost of the whole transpiled circuit. The sum may differ from that result, because optimizing the whole circuit also simplifies gates across block boundaries.

In `full_circuits.csv` the components carry the fields `preparation_n_cz`, `f3_blocks_n_cz`, `cz3_blocks_n_cz`, and their sum the field `unfused_n_cz`. The corresponding single-qubit gate costs are stored as `preparation_n_1q`, `f3_blocks_n_1q`, `cz3_blocks_n_1q` and `unfused_n_1q`. The number of qutrits and of CZ3 blocks is in `component_num_qutrits` and `component_num_cz3_blocks`. The final whole-circuit costs are `two_qubit_gate_count` and `one_qubit_gate_count`.

Older bundles may not contain this breakdown. The report then shows —, and the CSV leaves the value missing; missing does not mean zero cost. We do not derive component costs from the final gate count and we do not modify existing artifacts while reporting.

## Interpreting the parameters

The raw 33 angles of the fixed template are grouped in threes according to the actual U3 instructions and qubits. Every `template_id` gets its own figures. The absence of valid parameters for the current theta means a gap, even when the parameters of the previous valid parent were recorded or the fallback found a different valid circuit.

`unwrap` removes representation jumps by multiples of `2*pi` only within a connected segment of valid points. It does not cross a failed point or a change of template. The unwrapped plot alone does not prove smoothness: Euler angles have an ambiguous representation and coordinate singularities.

For that reason the report also computes `min_gamma ||U_j(theta_i) - exp(i gamma) U_j(theta_(i-1))||_F` for the local U3 stored in the artifacts. This measure respects periodicity and equivalent representations of the same U3 up to a global phase. It applies to neighbouring valid points of the same template; it does not remove all the gauge freedom that can be moved between different gates, and it does not prove smoothness of the full-circuit operator.

The known baseline with six CZ is not a proof of a global minimum. Finding an F3 with at most two CZ does not prove a minimum over all `alpha`. A continuation failure means only that the given template, starting point and finite numerical budget did not produce a result meeting the thresholds. The report preserves negative results and costs larger than the baseline.

## Local tests

The CLI tests use stubs for the expensive runner. The report tests write real, small artifact bundles and verify parameter gaps, U3 periodicity, template distinctness and the rejection of corrupted data.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -m pytest tests/test_theta_benchmark_cli.py tests/test_theta_benchmark_report.py -q
```

F3 requires `0 < --f3-tolerance <= 1e-10`. CZ3 accepts an explicit, positive and finite `--cz3-tolerance`; the default is still `1e-5`. The threshold is passed to the synthesis, the gate validation and the independent validation of the full-circuit library. It does not change the BQSKit epsilon, seed or algorithm. Changing the threshold requires a new run and a new manifest.


## Reassessment at the 5e-4 threshold

A separate script `scripts/reassess_theta_benchmark.py` reads an existing campaign and produces a new report. It does not modify the source directory. For CZ3 it checks both conditions: `E_norm <= 5e-4` and `L_norm <= 5e-4`; for F3 it keeps `1e-10`.

```powershell
.venv/Scripts/python.exe scripts/reassess_theta_benchmark.py --source-run artifacts/theta_benchmark/full-41points --output-dir artifacts/theta_benchmark/full-41points-tol5e-4 --cz3-tolerance 5e-4 --workers 3
```

A resume uses the same command with `--resume` appended. The configuration, library versions, code hashes and source-bundle hashes must match the new manifest.

The saved F3, baseline and raw CZ3 are reused after their hashes and operator action have been validated. If the old synthesis stored only the metrics of a rejected result, the circuit is reconstructed with the same BQSKit settings. The new report marks such reconstructions and compares their norms and CZ count against the previous values. A resume reads completed recovery bundles instead of repeating their synthesis.

This is a reassessment of saved attempts, not a new continuation: the historical starting parameters and parents remain unchanged. Independent BQSKit reconstructions may run in parallel. They must not be interpreted as a parallel continuation of successive theta values.

If all standard CZ3 transpilations change the full operator by more than `1e-10`, the compilation attempts an exact replacement of every U3 by a U with identical parameters. It preserves the wires, the CZ order and the global phase. This variant is explicitly marked; the threshold for preserving the full operator is still `1e-10`.

The report contains the actual CZ and single-qubit gate counts, the total depth, the depth of the CZ layers, the error and leakage norms, the full-state fidelities and the split of the cost into zero preparation, F3 and CZ3. The figures are produced from the saved QPY. Acceptance at `5e-4` means that this threshold was met, not that the error is zero.
