# IQM state preparation and QPY fidelity

`scripts/run_direct_basis_benchmarks.py --iqm-backend garnet` uses
`preset_exact` and `transpile_to_iqm_exact` by default for
`--ranking-workload state_preparation`. These strategies retain final RZ
rotations and therefore preserve relative phases. Explicit phase-dropping
strategies are rejected for this workload, including with `--no-fidelity`.

`preset_default` and `transpile_to_iqm_default` remove terminal RZ rotations.
This preserves computational-basis measurement probabilities but generally
changes state fidelity. They remain available in the transpiler comparison
harness and for `bell_measurements`, where the complete measurement circuits
are compiled before their resource costs are ranked.

IQM-transpiled circuits are normalized when their ordered bit lists disagree
with `find_bit` indices. The normalization preserves wire order, operations,
global phase, registers, classical data, and layout. It runs before benchmark
metrics and before QPY export, so CSV indices and loaded QPY circuits agree.

Historical CSV and QPY files are not automatically repaired. In particular,
the GHZ3 run `20260907_012249` selected phase-dropping strategies. Rerun its
state-preparation benchmark with the new defaults, using a new output CSV and
circuit directory. Changing the old fidelity column alone cannot restore the
missing rotations or correct the associated resource counts.

The local diagnostic in
`artifacts/iqm_runs/analysis/ghz3_20260907_012249_exact_verified/verification.csv`
checks all 216 encodings using `IQMFakeGarnet`, `preset_exact`, and one seed per
encoding taken from the original CSV. It checks fidelity after QPY reload and
provides newly generated QPY files. It is not a repeat of the original
20-seed search and does not use the historical live calibration.
