# AME(4,6) inputs

`targets.npz` contains four source tensors: `Bruzda60`, `Pandya60prime`, `HZ96`,
and `Casas36`. Each array has shape `6 x 6 x 6 x 6`, with axes A, B, C, D.
The harness normalizes each tensor before use. Each party occupies three qubits.

`circuits/<state>.json` contains a known preparation circuit and its local basis.
The import checks the tensor hash, local unitary matrices, full output state,
and final leakage. The input circuits use the `ame46-c18-ops-v1` format.
The local basis uses `ame46-local-basis-v1` with direction `source_to_output`.

These files are copies of the inputs used by the local AME(4,6) baseline.
Their original circuit hashes appear in `provenance.json`.
The [result manifest](../../benchmarks/state_preparation/manifest.json) contains
the hashes of all published input files.

Import from the repository root:

```bash
qoq-evolution ame46-import --targets experiment_inputs/ame46/targets.npz \
  --circuits experiment_inputs/ame46/circuits
```

See the [harness guide](../../docs/harness.md#ame46) for further searches.
