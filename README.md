# QuditsOnQubits

A Python framework for encoding higher-dimensional quantum systems (qudits) on standard qubit hardware, with a focus on preparing and verifying **absolutely maximally entangled (AME) states** on IBM Quantum processors.

## Overview

Quantum information encoded in dimensions *d > 2* (qutrits, ququarts) offers advantages in entanglement certification and Bell-type experiments, but current hardware only provides qubits. This project bridges that gap by:

- **Encoding qutrits/ququarts into qubit pairs** via configurable isometries $E: \mathbb{C}^d \to \mathbb{C}^4$
- **Constructing AME-state circuits from graphs** — vertex-local unitaries (F gates) and edge entangling operations (generalized CZ) loaded from a precompiled gate library
- **Measuring multipartite entanglement** through the $B_\text{AME}$ witness built from graph-derived stabilizer-like Pauli strings
- **Running Bell/CHSH-type inequality tests** on simulators and real IBM Quantum backends, with and without error mitigation
- **Benchmarking encoding bases** by transpiling circuits to a realistic coupling map and comparing gate depths

## Key Components

| Module | Description |
|---|---|
| `create_ame_circuit` | Builds Qiskit circuits for AME states from an igraph graph; supports encoding changes via unitary conjugation and Weyl decomposition |
| `generate_b_ame` | Computes the $B_\text{AME}$ entanglement witness — stabilizer coefficients, expectation values, and hardware-ready observables |
| `prepare_op_to_ibm` | Converts stabilizer circuits into Hermitian SparsePauliOp observables with ISA-compatible qubit layout |
| `benchmark_encoding_bases` | Sweeps over encoding isometries (Fourier, Haar-random, monomial, CZ-ansatz, …) and records transpiled circuit depth |
| `draw_graph` | Visualises the underlying graph structure of an AME state |

## Notebooks

| Notebook | Purpose |
|---|---|
| `2qutrit_RyGates` | CHSH-style parameter scans with Ry rotations on a 2-qutrit AME state, run on IBM hardware |
| `2qutrit_wittness` | Entanglement witness measurement with and without error mitigation |
| `calc_fidelity_2qtr` | Two-qutrit state fidelity analysis |
| `calc_fidelity_ghz` | GHZ-like qutrit state tomography using qiskit-experiments |
| `ghz_ry_chsh_recovered` | Recovered CHSH witness values for GHZ-type qutrit states |
| `ame43_random_chsh` | CHSH experiments on AME(4,3) graph states |
| `encode_basis_change` | Demonstrates encoding-change unitary construction and fidelity validation |

## Tech Stack

- **Qiskit** + **Qiskit IBM Runtime** — circuit construction, transpilation, and execution on IBM Quantum hardware
- **Qiskit Aer** — local statevector and noisy simulation
- **NumPy / SciPy / SymPy** — linear algebra, matrix exponentials, symbolic display
- **igraph** + **Matplotlib** — graph construction and visualisation
- **Pandas** — benchmark data collection

## Getting Started

```bash
pip install -r requirements.txt
```

Some notebooks additionally require `qiskit-aer`, `qiskit-experiments`, and `scipy`.

To run experiments on IBM Quantum hardware you need a valid IBM Quantum account configured with `qiskit-ibm-runtime`.

## License

See repository for license details.
