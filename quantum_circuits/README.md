# Saved state preparation circuits

The state preparation harness uses `quantum_circuits/<state>/<processor>/` for its records.
`logical/` contains a result before hardware routing.
Processor directories contain independently selected results for offline processor models.

| File | Contents |
| --- | --- |
| `results.json` | State, settings, provenance, costs, validation metrics, and encoding matrices. |
| `circuit.qpy` | Preparation from physical zero in the saved output encoding. |
| `decoder.qpy` | Optional decoder from that encoding to the canonical encoding. |
| `circuit_native.qpy` | Optional preparation after layout and routing to a processor model. |
| `circuit_native.json` | Native gate list and physical qubit layout. |

Preparation circuits contain no measurements. Decoder costs are separate.
The `native` section describes offline compilation. A `success_estimate` is a
model output and does not establish performance on hardware.

The [result snapshot](../benchmarks/state_preparation/README.md) lists the published
costs and provides checks for the circuits and their encodings.
AME(4,6) records need the [published tensors](../experiment_inputs/ame46/README.md).
The [harness guide](../docs/harness.md) describes further searches and record updates.
