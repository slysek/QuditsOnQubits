# Graph-state baselines

These files preserve the initial baselines from `run_20261102_01`.
They cover `two_qutrit`, `ghz3`, and `ame43` with methods `canonical_f3`,
`canonical_direct`, and `estar_direct`.

`canonical_f3` synthesizes the complete F₃ gate for initial preparation.
The direct methods prepare only the required initial state.
`estar_direct` uses the E* encoding. All costs include preparation from physical zero.

The harness can use these files when the original local baselines are absent.
`summary.json` identifies each circuit and encoding. `manifest.json` contains their hashes.
The [published comparison](../../benchmarks/state_preparation/README.md) uses
`canonical_direct` as its baseline and reports later search results separately.
