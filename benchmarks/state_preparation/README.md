# State preparation results

Snapshot: **11 November 2026**. The manifest covers **58 saved records** and the comparison inputs.

## Canonical preparation and harness results

The baseline directly prepares the required initial state in the canonical encoding.
It does not synthesize a complete F₃ gate for that preparation.
The harness results use free local encodings.
All costs include preparation from physical zero, before routing or measurement.

| State | Canonical CZ | Harness CZ | Reduction | Decoder CZ | Preparation + decoder CZ |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Two-qutrit graph state](../../quantum_circuits/two_qutrit/logical/results.json) | 8 | **3** | 62.5% | 3 | 6 |
| [GHZ₃ graph state](../../quantum_circuits/ghz3/logical/results.json) | 15 | **5** | 66.7% | 4 | 9 |
| [AME(4,3)](../../quantum_circuits/ame43/logical/results.json) | 28 | **17** | 39.3% | 6 | 23 |

The saved fidelities exceed `1 - 1e-12`. Final leakage probability is below `1e-20`.
Acceptance requires both state error and leakage norms at most `1e-10`.
The decoder restores the canonical encoding. Its cost is separate from preparation.
The last column adds preparation and decoder counts without further compilation.

These are cumulative best results from different searches, including random mutations and agents.
They do not compare search methods under equal budgets or establish global minima.
Circuit costs alone do not establish an improvement on hardware.

## AME(4,6)

The public inputs include four tensors, local basis maps, and known 12-qubit circuits.
The imported counts below already occur in the source circuits.
The harness did not discover a lower CZ count for these imported baselines.

| State | Imported CZ | Imported 2q depth | Saved logical CZ | Saved 2q depth | Decoder CZ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bruzda60 | 46 | 30 | 46 | 26 | 52 |
| Pandya60prime | 64 | 41 | — | — | — |
| HZ96 | 69 | 39 | — | — | — |
| Casas36 | 37 | 22 | 37 | 19 | 52 |

The tests check all imported circuits against their source tensors.
They also check the three complementary two-party partitions against `I₃₆/36`.
The [input guide](../../experiment_inputs/ame46/README.md) gives the import command.

## Other saved logical results

These rows report available circuits without a matched baseline comparison.
They include analytic constructions and results from later searches.
Changing the qudit dimension also changes the target. Cross-dimension counts do not isolate the effect of leakage.

| State | Qubits | Preparation CZ | 2q depth | Decoder CZ |
| --- | ---: | ---: | ---: | ---: |
| [aharonov3](../../quantum_circuits/aharonov3/logical/results.json) | 6 | 9 | 4 | 0 |
| [aklt4](../../quantum_circuits/aklt4/logical/results.json) | 8 | 16 | 11 | 4 |
| [ame44](../../quantum_circuits/ame44/logical/results.json) | 8 | 8 | 2 | 0 |
| [ame45](../../quantum_circuits/ame45/logical/results.json) | 12 | 41 | 33 | 52 |
| [ame47](../../quantum_circuits/ame47/logical/results.json) | 12 | 84 | 31 | 52 |
| [ame48](../../quantum_circuits/ame48/logical/results.json) | 12 | 13 | 3 | 52 |
| [ame64](../../quantum_circuits/ame64/logical/results.json) | 12 | 18 | 3 | 0 |
| [cluster4](../../quantum_circuits/cluster4/logical/results.json) | 8 | 9 | 5 | 4 |
| [dicke4](../../quantum_circuits/dicke4/logical/results.json) | 8 | 15 | 7 | 8 |

## Offline processor results

These counts describe preparation after layout and routing, without the decoder or measurement.
Each processor can select a different circuit. They are not measurements on a QPU.
IBM models use captured calibration data. Some IQM models use generic or synthetic errors.

| State | Heron r2 | Heron r3 | Nighthawk | Garnet | Emerald | Sirius |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| two_qutrit | [3](../../quantum_circuits/two_qutrit/ibm_heron_r2/results.json) | [3](../../quantum_circuits/two_qutrit/ibm_heron_r3/results.json) | [3](../../quantum_circuits/two_qutrit/ibm_nighthawk/results.json) | [3](../../quantum_circuits/two_qutrit/iqm_garnet/results.json) | [3](../../quantum_circuits/two_qutrit/iqm_emerald/results.json) | [9](../../quantum_circuits/two_qutrit/iqm_sirius/results.json) |
| ghz3 | [5](../../quantum_circuits/ghz3/ibm_heron_r2/results.json) | [6](../../quantum_circuits/ghz3/ibm_heron_r3/results.json) | [5](../../quantum_circuits/ghz3/ibm_nighthawk/results.json) | [5](../../quantum_circuits/ghz3/iqm_garnet/results.json) | [5](../../quantum_circuits/ghz3/iqm_emerald/results.json) | [13](../../quantum_circuits/ghz3/iqm_sirius/results.json) |
| ame43 | [34](../../quantum_circuits/ame43/ibm_heron_r2/results.json) | [37](../../quantum_circuits/ame43/ibm_heron_r3/results.json) | [20](../../quantum_circuits/ame43/ibm_nighthawk/results.json) | [20](../../quantum_circuits/ame43/iqm_garnet/results.json) | [22](../../quantum_circuits/ame43/iqm_emerald/results.json) | [50](../../quantum_circuits/ame43/iqm_sirius/results.json) |
| ame44 | [12](../../quantum_circuits/ame44/ibm_heron_r2/results.json) | [12](../../quantum_circuits/ame44/ibm_heron_r3/results.json) | [8](../../quantum_circuits/ame44/ibm_nighthawk/results.json) | [8](../../quantum_circuits/ame44/iqm_garnet/results.json) | [8](../../quantum_circuits/ame44/iqm_emerald/results.json) | [20](../../quantum_circuits/ame44/iqm_sirius/results.json) |
| ame64 | [50](../../quantum_circuits/ame64/ibm_heron_r2/results.json) | [50](../../quantum_circuits/ame64/ibm_heron_r3/results.json) | [33](../../quantum_circuits/ame64/ibm_nighthawk/results.json) | [33](../../quantum_circuits/ame64/iqm_garnet/results.json) | [33](../../quantum_circuits/ame64/iqm_emerald/results.json) | [50](../../quantum_circuits/ame64/iqm_sirius/results.json) |
| Bruzda60 | [148](../../quantum_circuits/Bruzda60/ibm_heron_r2/results.json) | [148](../../quantum_circuits/Bruzda60/ibm_heron_r3/results.json) | [94](../../quantum_circuits/Bruzda60/ibm_nighthawk/results.json) | [89](../../quantum_circuits/Bruzda60/iqm_garnet/results.json) | [95](../../quantum_circuits/Bruzda60/iqm_emerald/results.json) | [134](../../quantum_circuits/Bruzda60/iqm_sirius/results.json) |
| Casas36 | [84](../../quantum_circuits/Casas36/ibm_heron_r2/results.json) | [84](../../quantum_circuits/Casas36/ibm_heron_r3/results.json) | [57](../../quantum_circuits/Casas36/ibm_nighthawk/results.json) | [48](../../quantum_circuits/Casas36/iqm_garnet/results.json) | [54](../../quantum_circuits/Casas36/iqm_emerald/results.json) | [111](../../quantum_circuits/Casas36/iqm_sirius/results.json) |

Values count native two-qubit operations. Sirius counts include MOVE operations as well as CZ.
A `success_estimate` in a record is a model output, not a measured fidelity.
Use the same model, routing budget, and objective for controlled comparisons.

## Check the evidence

Use a repository checkout. The checks require no model, credentials, or QPU access.

```bash
python -m pip install -e ".[dev,evolution]"
python -m pytest -q tests/test_published_state_preparation.py
```

The tests check SHA-256 hashes, canonical comparisons, all published logical preparations,
available decoders, native operation counts, and AME(4,6) input imports.
This checks the saved evidence. It does not repeat the original searches.

- [summary.json](summary.json): comparison costs, thresholds, and validation versions.
- [manifest.json](manifest.json): published records and SHA-256 hashes.
- [Canonical inputs](../../experiment_inputs/state_preparation/summary.json): baseline circuits and encodings.
- [AME(4,6) inputs](../../experiment_inputs/ame46/README.md): tensors, circuits, and provenance.
- [Saved circuit format](../../quantum_circuits/README.md) and [harness guide](../../docs/harness.md).

Original run labels in records can refer to local campaign archives.
The files in the manifest suffice for the checks above.
