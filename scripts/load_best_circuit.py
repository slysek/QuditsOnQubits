from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qiskit import qpy


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_selected_candidate_dir(
    run_kind: str,
    state: str,
    run_id: str,
    selection_label: str,
    *,
    rank: int | None = None,
    candidate: str | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root) if repo_root is not None else globals()["repo_root"]()
    base = root / "artifacts" / run_kind / "selected_best" / state / run_id / selection_label
    if candidate:
        path = base / candidate
        if not path.is_dir():
            raise FileNotFoundError(f"selected candidate directory does not exist: {path}")
        return path
    if rank is None:
        raise ValueError("rank is required when candidate is not provided")
    matches = sorted(base.glob(f"rank{int(rank):02d}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one selected candidate for rank {int(rank):02d} under {base}, found {len(matches)}"
        )
    return matches[0]


def load_selected_circuit(
    run_kind: str,
    state: str,
    run_id: str,
    selection_label: str,
    *,
    rank: int | None = None,
    candidate: str | None = None,
    circuit_kind: str = "transpiled",
    repo_root: str | Path | None = None,
):
    base = resolve_selected_candidate_dir(
        run_kind,
        state,
        run_id,
        selection_label,
        rank=rank,
        candidate=candidate,
        repo_root=repo_root,
    )
    if circuit_kind == "transpiled":
        circuit_path = base / "graph_state_direct_basis_transpiled.qpy"
    elif circuit_kind == "raw":
        circuit_path = base / "graph_state_direct_basis.qpy"
    else:
        raise ValueError("circuit_kind must be 'transpiled' or 'raw'")
    e_path = base / "E.npy"
    if not e_path.is_file():
        raise FileNotFoundError(f"required selected encoding is missing: {e_path}")
    with circuit_path.open("rb") as handle:
        circuit = qpy.load(handle)[0]
    E = np.load(e_path)
    return circuit, E


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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection-label", required=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--circuit-kind",
        choices=("transpiled", "raw"),
        default="transpiled",
    )
    args = parser.parse_args()

    circuit, E = load_selected_circuit(
        args.run_kind,
        args.state,
        args.run_id,
        args.selection_label,
        rank=args.rank,
        candidate=args.candidate,
        circuit_kind=args.circuit_kind,
    )
    print(f"circuit_kind={args.circuit_kind}")
    print(f"circuit_qubits={circuit.num_qubits}")
    print(f"E_shape={E.shape}")


if __name__ == "__main__":
    main()
