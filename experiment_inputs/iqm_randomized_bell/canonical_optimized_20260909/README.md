# Pinned canonical F3/CZ3 library

Copied without modification from
`artifacts/iqm_runs/processed/transpiler_harness/optimized_f3_cz3_two_qutrit_20260909/quantum_circuits/two_qutrit/baseline__E_old`.

`E.npy` is the canonical 4-by-3 embedding. Both QPY files carry
`gate_library = code_space_optimized_v1`. The notebook reconstructs the graph
state from these gates rather than loading an opaque prepared state.

| Gate | Two-qubit count | Code-space Frobenius error | Leakage Frobenius norm | SHA-256 |
| --- | --- | --- | --- | --- |
| F3 | 2 CX | 2.945892585688694e-15 | 1.5375600943694801e-15 | `4f337e4c1e92a48cd84be7a1d9cc42dbd2904cf65ec2b7bc513d9b75018167af` |
| CZ3 | 6 CZ | 1.4566641475869985e-06 | 6.417337310175103e-07 | `a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791` |

Errors are recomputed at notebook load, allowing one global phase. CZ3 is an
available optimized synthesis, not a proven globally minimal circuit.
