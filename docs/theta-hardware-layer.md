# Additional hardware layer of the theta benchmark

The layer reads saved circuits from `artifacts/theta_benchmark/full-41points-tol5e-4`.
It does not change the original results. New results go to a separate directory
`artifacts/theta_benchmark/full-41points-iqm-layer`.

## Local run

From the project directory, in an environment containing the project dependencies:

```powershell
python scripts/run_theta_hardware_layer.py `
  --source-run artifacts/theta_benchmark/full-41points-tol5e-4 `
  --output-dir artifacts/theta_benchmark/full-41points-iqm-layer `
  --snapshot artifacts/theta_benchmark/iqm_emerald_snapshot_20260917.json
```

`--resume` resumes exactly the same configuration and checks the hashes of the
saved bundles. A change of code, sources, snapshot or configuration requires a new
directory. Partial attempt: `--indices 0 40 --states two_qutrit`; index 0 is required
for the baseline comparison. The CLI does not connect to a provider and does not
submit jobs.

## Additional numbers

- `gates.csv`: original F3/CZ3, native R/CZ, after routing; the full operator
  is checked once the output permutation has been taken into account.
- `summary.csv`: old preparation costs, native preparation, preparation
  after routing, full Bell before/after routing; CZ, R, depth and CZ depth.
- `bell_settings.csv`: a separate row for every setting, the shared prefix,
  local measurements, repair SWAPs, and the path to the actual QPY.
- `report.md`, `hardware_costs.png`, `hardware_costs.pdf`, `ranking.json`:
  a comparison of all angles and candidates by full Bell cost.

SWAPs are decomposed into native gates and are already counted in CZ. The final
permutation stays in the metadata; merely relabelling the output indices does not
require inverting it. The repair guarantees connectivity inside each qutrit encoding
pair. The final local preparation gates are merged into the measurement blocks. Each
side uses the same local block for a given basis regardless of the other sides'
bases; the shared prefix stays identical across all settings.

## Scope of the comparison

41 angles, 3 states, 123 preparations, 1394 full Bell settings and 82 gates.
Within a state, every angle uses the same Emerald subgraph and compilation budget.
Routing compares optimization levels 3 and 1 for each of the seeds 0, 1, 2. Level 0
is the fallback mode when both change the result beyond the 1e-10 tolerance. This is
a resource comparison under fixed constraints, without global optimization of qubit
selection, noise model and pulse schedule. DD/ZNE are not included.
Averages over settings carry equal weights; they do not define a shot allocation.

The theta=0 baseline comes from the same F3/CZ3 library. The historical shortened
Bell baseline with 5-7 CZ and the prepared theta40 campaign are separate artifacts.
This layer does not change them. IQM jobs require separate user approval.

Before finishing, the runner checks that all source files are unchanged.
Agreement of the ideal histograms verifies compilation correctness; it does not
predict the hardware Bell value. Leakage carries zero weight and stays in the
denominator.
