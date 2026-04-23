# Encoded AME Circuit Basis-Change Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace matrix-transformed encoded AME gates with explicit `W -> gate -> Wdag` circuit blocks while preserving the initial `W` layer for every `E_new` basis change.

**Architecture:** Keep all public behavior in `create_ame_circuit` unchanged, but move the `E_new` path from synthesized transformed matrices to helper-driven subcircuit assembly. Add a small regression test module that proves the new explicit blocks are operator-equivalent to the previous matrix conjugation formulas and that `create_ame_circuit(..., E_new=...)` still returns a valid circuit.

**Tech Stack:** Python, NumPy, Qiskit, unittest

---

## File Structure

- Modify: `QuditsOnQubits/create_ame_circuit.py`
  Responsibility: replace transformed-gate synthesis with decomposed `W` and `Wdag` subcircuits plus helper-based block assembly.
- Create: `tests/test_create_ame_circuit.py`
  Responsibility: regression coverage for encoded `Fgate`/`CZgate` block equivalence and basic encoded-circuit smoke coverage.

### Task 1: Add Regression Tests For Explicit Encoded Blocks

**Files:**
- Create: `tests/test_create_ame_circuit.py`
- Test: `tests/test_create_ame_circuit.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import unittest

import numpy as np
from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from QuditsOnQubits.create_ame_circuit import (
    _append_encoded_czgate,
    _append_encoded_fgate,
    _build_encoding_change_circuits,
    create_ame_circuit,
)


def _repo_root():
    return os.path.dirname(os.path.dirname(__file__))


def _load_gate(relative_path):
    with open(os.path.join(_repo_root(), relative_path), "rb") as fd:
        return qpy.load(fd)[0]


def _sample_encoding():
    a = 1.0 / np.sqrt(2.0)
    return np.array(
        [
            [a, a, 0],
            [a, -a, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=complex,
    )


class EncodedAmeCircuitTests(unittest.TestCase):
    def test_encoded_blocks_match_previous_matrix_conjugation(self):
        e_new = _sample_encoding()
        w, w_qc, wdag_qc = _build_encoding_change_circuits(e_new)
        wdag = w.conj().T

        fgate = _load_gate("quantum_circuits/Fgate3.qpy")
        czgate = _load_gate("quantum_circuits/CZgate3.qpy")

        fgate_old = w @ Operator(fgate).data @ wdag
        cz_old = np.kron(w, w) @ Operator(czgate).data @ np.kron(wdag, wdag)

        f_qc = QuantumCircuit(2)
        _append_encoded_fgate(f_qc, [0, 1], fgate, w_qc, wdag_qc)

        cz_qc = QuantumCircuit(4)
        _append_encoded_czgate(cz_qc, [0, 1, 2, 3], czgate, w_qc, wdag_qc)

        self.assertTrue(np.allclose(Operator(f_qc).data, fgate_old))
        self.assertTrue(np.allclose(Operator(cz_qc).data, cz_old))

    def test_create_ame_circuit_with_encoding_change_builds_valid_circuit(self):
        qc, graph = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=_sample_encoding())

        self.assertEqual(graph.vcount(), 2)
        self.assertEqual(qc.num_qubits, 4)
        self.assertGreater(qc.size(), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p "test_create_ame_circuit.py" -v`
Expected: FAIL because `_build_encoding_change_circuits`, `_append_encoded_fgate`, and `_append_encoded_czgate` do not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_create_ame_circuit.py
git commit -m "test: add encoded AME circuit regression coverage"
```

### Task 2: Implement Helper-Based Encoded Circuit Assembly

**Files:**
- Modify: `QuditsOnQubits/create_ame_circuit.py`
- Test: `tests/test_create_ame_circuit.py`

- [ ] **Step 1: Write the minimal implementation**

```python
def _build_encoding_change_circuits(E_new):
    from encoding_change_unitary import build_encoding_change_unitary

    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W ma wymiar {W.shape}, oczekiwano (4, 4)"
    W_qc = TwoQubitWeylDecomposition(W).circuit()
    Wdag_qc = TwoQubitWeylDecomposition(W.conj().T).circuit()
    return W, W_qc, Wdag_qc


def _append_encoded_fgate(qc, pair, Fgate, W_qc, Wdag_qc):
    qc.append(W_qc, pair)
    qc.append(Fgate, pair)
    qc.append(Wdag_qc, pair)


def _append_encoded_czgate(qc, edge, CZgate, W_qc, Wdag_qc):
    left = edge[:2]
    right = edge[2:]
    qc.append(W_qc, left)
    qc.append(W_qc, right)
    qc.append(CZgate, edge)
    qc.append(Wdag_qc, left)
    qc.append(Wdag_qc, right)
```

Update `_create_circuit_from_graph(...)` so that:

```python
W_qc = None
Wdag_qc = None
if E_new is not None:
    _, W_qc, Wdag_qc = _build_encoding_change_circuits(E_new)

if W_qc is not None:
    for pair in qubitList:
        qc.append(W_qc, pair)

for pair in qubitList:
    if W_qc is None:
        qc.append(Fgate, pair)
    else:
        _append_encoded_fgate(qc, pair, Fgate, W_qc, Wdag_qc)

for edge in edgeList:
    if W_qc is None:
        qc.append(CZgate, edge)
    else:
        _append_encoded_czgate(qc, edge, CZgate, W_qc, Wdag_qc)
```

Also remove the old `_transform_gates_to_new_encoding(...)` helper and update the `E_new` docstring/comments so they describe explicit circuit blocks rather than transformed `UnitaryGate` matrices.

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p "test_create_ame_circuit.py" -v`
Expected: PASS with 2 tests run, 0 failures.

- [ ] **Step 3: Run the smoke verification**

Run:

```powershell
@'
import numpy as np
from QuditsOnQubits.create_ame_circuit import create_ame_circuit

a = 1.0 / np.sqrt(2.0)
e_new = np.array([
    [a,  a, 0],
    [a, -a, 0],
    [0,  0, 1],
    [0,  0, 0],
], dtype=complex)

qc, graph = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=e_new)
print(graph.vcount(), qc.num_qubits, qc.size())
'@ | python -
```

Expected: prints `2 4 <positive integer>` and exits with code 0.

- [ ] **Step 4: Commit**

```bash
git add QuditsOnQubits/create_ame_circuit.py tests/test_create_ame_circuit.py
git commit -m "refactor: build encoded AME circuits from explicit W blocks"
```

### Task 3: Final Verification Review

**Files:**
- Modify: `QuditsOnQubits/create_ame_circuit.py`
- Test: `tests/test_create_ame_circuit.py`

- [ ] **Step 1: Re-run the full verification commands**

Run: `python -m unittest discover -s tests -p "test_create_ame_circuit.py" -v`
Expected: PASS with 2 tests run, 0 failures.

Run:

```powershell
@'
import numpy as np
from qiskit.quantum_info import Operator
from QuditsOnQubits.create_ame_circuit import create_ame_circuit

a = 1.0 / np.sqrt(2.0)
e_new = np.array([
    [a,  a, 0],
    [a, -a, 0],
    [0,  0, 1],
    [0,  0, 0],
], dtype=complex)

qc, _ = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=e_new)
print(Operator(qc).data.shape)
'@ | python -
```

Expected: prints `(16, 16)` and exits with code 0.

- [ ] **Step 2: Commit**

```bash
git add QuditsOnQubits/create_ame_circuit.py tests/test_create_ame_circuit.py
git commit -m "chore: verify encoded AME circuit rewrite"
```
