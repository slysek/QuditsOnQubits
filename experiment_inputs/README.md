# Experiment inputs

Deterministic, reusable source inputs live here. Runtime results stay in `artifacts/`.

Layout:

```
reference_bases/<state>/<encoding>/
```

Each reference-basis bundle contains:

- `graph_state_direct_basis.qpy`: one unmeasured prepared graph-state circuit.
- `E.npy`: the encoding isometry.
- `metadata.json`: bundle identity, dimensions, and file hashes.

Bundles are immutable. A validation failure requires explicit inspection or removal before a new bundle may be materialized.
