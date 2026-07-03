from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qiskit import qpy


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_selected_circuit(run_kind: str, state: str, candidate: str):
    base = repo_root() / "artifacts" / run_kind / "selected_best" / state / candidate
    with (base / "graph_state_direct_basis.qpy").open("rb") as f:
        graph_state = qpy.load(f)[0]
    with (base / "graph_state_direct_basis_transpiled.qpy").open("rb") as f:
        transpiled = qpy.load(f)[0]
    E = np.load(base / "E.npy")
    return graph_state, transpiled, E


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a selected best direct-basis circuit artifact."
    )
    parser.add_argument(
        "--run-kind",
        default="direct_basis_runs",
        choices=["direct_basis_runs", "iqm_runs"],
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    graph_state, transpiled, E = load_selected_circuit(
        args.run_kind,
        args.state,
        args.candidate,
    )
    print(f"graph_state_qubits={graph_state.num_qubits}")
    print(f"transpiled_qubits={transpiled.num_qubits}")
    print(f"E_shape={E.shape}")


if __name__ == "__main__":
    main()
