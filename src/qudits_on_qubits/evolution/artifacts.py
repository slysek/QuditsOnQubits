"""Load pinned input files and check their hashes and dimensions."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from qiskit import qpy

from .encodings import tensor_encoding


def file_hash(data: bytes) -> str:
    """Return the SHA-256 hash of the bytes."""
    return hashlib.sha256(data).hexdigest()


def main_checkout(path: Path) -> Path:
    """Return the main git working tree when path is a linked worktree. Otherwise, return path.

    Runs in a linked worktree then write their results to the main checkout. The results stay
    when you remove the worktree.
    """
    marker = path / ".git"
    if not marker.is_file():
        return path
    gitdir = path / marker.read_text(encoding="utf-8").removeprefix("gitdir:").strip()
    if gitdir.parent.name != "worktrees" or gitdir.parent.parent.name != ".git":
        return path
    return gitdir.parent.parent.parent


def load_bytes(path: Path, expected_hash: str) -> bytes:
    """Read a file and check its SHA-256 hash."""
    data = path.read_bytes()
    if file_hash(data) != expected_hash:
        raise ValueError(f"The SHA-256 hash of {path} is not correct")
    return data


def load_encoding(path: Path, expected_hash: str, d: int) -> np.ndarray:
    """Load one isometry from a pinned NPY file."""
    matrix = np.load(BytesIO(load_bytes(path, expected_hash)), allow_pickle=False)
    tensor_encoding([matrix], d=d)
    return matrix


def load_circuit(path: Path, expected_hash: str, num_qubits: int):
    """Load one circuit from a pinned QPY file."""
    circuits = qpy.load(BytesIO(load_bytes(path, expected_hash)))
    if len(circuits) != 1 or circuits[0].num_qubits != num_qubits:
        raise ValueError(f"{path} must contain one circuit with {num_qubits} qubits")
    return circuits[0]
