# GHZ3 IQM notebook cells design

## Goal

Extend `notebooks/ghz3_bell_canonical_baseline.ipynb` with an opt-in IQM
Garnet baseline matching the established two-qutrit notebook structure.

## Notebook behavior

- Keep the unguarded AerIdeal baseline unchanged.
- Import `IQMHardware` and `MitigationConfig` from the experiment API.
- Define one shared `SHOTS = 100` value for Aer and IQM.
- Define `HARDWARE_MITIGATION` with readout mitigation enabled, ZNE enabled,
  and factors `(1, 3, 5)`.
- Add a dedicated IQM Garnet section guarded by `RUN_IQM = False`.
- When `RUN_IQM` is changed to `True`, submit the GHZ3 canonical basis
  experiment to Garnet and store the result under `RESULTS["iqm_garnet"]`.
- A default Run All must execute Aer only and print an explicit IQM skip
  message without contacting hardware.
- Expand the final summary to include both `aer_ideal` and `iqm_garnet`,
  preserving raw, readout-mitigated, ZNE, combined ZNE/readout, diagnostics,
  leakage, reference bounds, status, and artifact directory values.

## Safety and credentials

The notebook must not embed tokens, provider URLs, or local credential paths.
IQM credentials continue to come from the existing provider/environment
configuration. Hardware submission remains an explicit user action.

## Tests

- Update structural notebook tests to require the IQM imports and opt-in cells.
- Assert the exact Garnet backend, 100 shots, readout policy, and ZNE factors.
- Assert the default path does not call hardware.
- Assert summary rows preserve runner-produced Aer and IQM values.
- Keep the existing environment-gated IQM Garnet connected smoke test.
- Run focused notebook/pipeline tests and the relevant regression suite before
  updating pull request #14.
