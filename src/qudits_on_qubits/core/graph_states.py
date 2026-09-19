"""Graph-state edge generators and benchmark state-name resolution.

A single, self-contained place in the repository defining:

* edge-list generators for graph families (star / path / cycle / wheel /
  complete / 2D cluster grid),
* a parser for textual benchmark state names (e.g. ``"path5"``,
  ``"cluster2x3"``, ``"ghz_star_4"``, ``"ame43"``),
* predefined benchmark suite registries used by the single-stage pipeline
  in :mod:`encoding_search_v2.suite`.

This module is intentionally outside :mod:`encoding_search_v2` and
:mod:`QuditsOnQubits.benchmark_encoding_bases` so both layers can share
a single definition of graph families without circular imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from qudits_on_qubits.reference_experiments import get_reference_experiment


EdgeList = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class GraphStateSpec:
    """Static description of a qutrit graph state for benchmarking."""

    state_id: str
    state_family: str
    num_qutrits: int
    graph_type: str
    edges: EdgeList


# ───────────────────── edge generators ────────────────────────

def star_edges(n: int) -> EdgeList:
    n = int(n)
    if n < 2:
        raise ValueError(f"star graph requires n>=2, got {n}.")
    return tuple((0, leaf) for leaf in range(1, n))


def path_edges(n: int) -> EdgeList:
    n = int(n)
    if n < 2:
        raise ValueError(f"path graph requires n>=2, got {n}.")
    return tuple((i, i + 1) for i in range(n - 1))


def cycle_edges(n: int) -> EdgeList:
    n = int(n)
    if n < 3:
        raise ValueError(f"cycle graph requires n>=3, got {n}.")
    return tuple((i, (i + 1) % n) for i in range(n))


def wheel_edges(n: int) -> EdgeList:
    """Wheel: node 0 is the center, nodes 1..n-1 form the outer cycle.

    Requires n>=4 (at least 1 center + 3 cycle nodes).
    """
    n = int(n)
    if n < 4:
        raise ValueError(f"wheel graph requires n>=4 (1 center + 3 outer), got {n}.")
    spokes = tuple((0, leaf) for leaf in range(1, n))
    outer_path = tuple((i, i + 1) for i in range(1, n - 1))
    closing = ((n - 1, 1),)
    return spokes + outer_path + closing


def complete_edges(n: int) -> EdgeList:
    n = int(n)
    if n < 2:
        raise ValueError(f"complete graph requires n>=2, got {n}.")
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def cluster_edges(rows: int, cols: int) -> EdgeList:
    """2D cluster (rectangular grid) with nearest-neighbor adjacency."""
    rows = int(rows)
    cols = int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(
            f"cluster grid requires rows>=1 and cols>=1, got rows={rows}, cols={cols}."
        )
    if rows * cols < 2:
        raise ValueError(
            f"cluster grid requires at least 2 nodes, got {rows}x{cols}."
        )

    def idx(r: int, c: int) -> int:
        return r * cols + c

    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return tuple(edges)


# ───────────────────── state-name parser ──────────────────────

_GHZ_STAR_PATTERN = re.compile(r"^(?:ghz_star|ghz_n)_(\d+)$")
_GHZ_SHORT_PATTERN = re.compile(r"^ghz(\d+)$")
_PATH_PATTERN = re.compile(r"^path(\d+)$")
_CYCLE_PATTERN = re.compile(r"^cycle(\d+)$")
_WHEEL_PATTERN = re.compile(r"^wheel(\d+)$")
_COMPLETE_PATTERN = re.compile(r"^complete(\d+)$")
_CLUSTER_PATTERN = re.compile(r"^cluster(\d+)x(\d+)$")


_FIXED_REFERENCE_GRAPH_METADATA = {
    "two_qutrit": ("two_qutrit", "star"),
    "ghz3": ("ghz_star", "star"),
    "ame43": ("ame43", "ame43"),
}


def _fixed_reference_graph_state(name: str) -> GraphStateSpec:
    state = get_reference_experiment(name).state
    family, graph_type = _FIXED_REFERENCE_GRAPH_METADATA[name]
    return GraphStateSpec(
        name,
        family,
        state.num_parties,
        graph_type,
        state.legacy_edges(),
    )


def resolve_graph_state(
    state_name: str,
    n_qutrits: Optional[int] = None,
) -> Optional[GraphStateSpec]:
    """Return a GraphStateSpec for a recognized state name, or None.

    Unknown names return None so callers (CLI or runner) can report an error
    with the appropriate context, such as the available state names."""
    name = str(state_name).strip()
    if not name:
        return None

    if name in _FIXED_REFERENCE_GRAPH_METADATA:
        spec = _fixed_reference_graph_state(name)
        if n_qutrits is not None and int(n_qutrits) != spec.num_qutrits:
            raise ValueError(
                f"{name} has fixed n_qutrits={spec.num_qutrits}, got {n_qutrits}."
            )
        return spec

    if name in ("ghz_star", "ghz_n"):
        if n_qutrits is None:
            raise ValueError(f"{name} requires n_qutrits.")
        n = int(n_qutrits)
        return GraphStateSpec(
            f"ghz_star_{n}", "ghz_star", n, "star", star_edges(n),
        )

    match = _GHZ_STAR_PATTERN.match(name)
    if match:
        n_from_slug = int(match.group(1))
        if n_qutrits is not None and int(n_qutrits) != n_from_slug:
            raise ValueError(
                f"{name} already encodes n_qutrits={n_from_slug}, got {n_qutrits}."
            )
        return GraphStateSpec(
            f"ghz_star_{n_from_slug}", "ghz_star", n_from_slug, "star",
            star_edges(n_from_slug),
        )

    match = _GHZ_SHORT_PATTERN.match(name)
    if match:
        n = int(match.group(1))
        if n_qutrits is not None and int(n_qutrits) != n:
            raise ValueError(
                f"{name} already encodes n_qutrits={n}, got {n_qutrits}."
            )
        return GraphStateSpec(
            f"ghz{n}", "ghz_star", n, "star", star_edges(n),
        )

    for pattern, family, edge_fn in (
        (_PATH_PATTERN, "path", path_edges),
        (_CYCLE_PATTERN, "cycle", cycle_edges),
        (_WHEEL_PATTERN, "wheel", wheel_edges),
        (_COMPLETE_PATTERN, "complete", complete_edges),
    ):
        match = pattern.match(name)
        if not match:
            continue
        n = int(match.group(1))
        if n_qutrits is not None and int(n_qutrits) != n:
            raise ValueError(
                f"{name} already encodes n_qutrits={n}, got {n_qutrits}."
            )
        return GraphStateSpec(
            state_id=f"{family}{n}",
            state_family=family,
            num_qutrits=n,
            graph_type=family,
            edges=edge_fn(n),
        )

    match = _CLUSTER_PATTERN.match(name)
    if match:
        rows = int(match.group(1))
        cols = int(match.group(2))
        n = rows * cols
        if n_qutrits is not None and int(n_qutrits) != n:
            raise ValueError(
                f"{name} encodes n_qutrits={n} (={rows}*{cols}), got {n_qutrits}."
            )
        return GraphStateSpec(
            state_id=f"cluster{rows}x{cols}",
            state_family="cluster",
            num_qutrits=n,
            graph_type="cluster",
            edges=cluster_edges(rows, cols),
        )

    return None


def resolve_graph_state_or_raise(
    state_name: str,
    n_qutrits: Optional[int] = None,
) -> GraphStateSpec:
    spec = resolve_graph_state(state_name, n_qutrits=n_qutrits)
    if spec is None:
        raise ValueError(
            f"Unknown benchmark state: {state_name!r}. "
            "Expected one of two_qutrit, ghz3, ame43, ghz<n>, ghz_star_<n>, "
            "path<n>, cycle<n>, wheel<n>, complete<n>, cluster<r>x<c>."
        )
    return spec


def is_known_graph_state(state_name: str) -> bool:
    """Return True for every name supported by the resolver."""
    try:
        return resolve_graph_state(state_name) is not None
    except ValueError:
        # States such as "ghz_star" that require n_qutrits.
        return True


# ───────────────────── "suite" registry ────────────────────────

#: Extended set of qutrit graph states for overnight benchmarking.
#: States ``two_qutrit``, ``ghz3``, ``ame43`` are *intentionally* excluded
#: from this set — they are already benchmarked separately in the older pipeline.
EXTENDED_GRAPH_STATES: Tuple[str, ...] = (
    # Qutrit GHZ / star graph
    "ghz4", "ghz5", "ghz6", "ghz7", "ghz8", "ghz9",
    # Paths / chains
    "path4", "path5", "path6", "path7", "path8", "path9",
    # Cycles / rings
    "cycle4", "cycle5", "cycle6", "cycle7", "cycle8", "cycle9",
    # Wheels
    "wheel5", "wheel6", "wheel7", "wheel8", "wheel9",
    # Complete graphs
    "complete4", "complete5", "complete6",
    # 2D cluster grids
    "cluster2x2", "cluster2x3",
)


BENCHMARK_SUITES: dict[str, Tuple[str, ...]] = {
    "graph_states_extended": EXTENDED_GRAPH_STATES,
}


def list_suites() -> Tuple[str, ...]:
    return tuple(sorted(BENCHMARK_SUITES))


def get_suite_states(suite_name: str) -> Tuple[str, ...]:
    if suite_name not in BENCHMARK_SUITES:
        raise ValueError(
            f"Unknown benchmark suite: {suite_name!r}. "
            f"Expected one of {list_suites()}."
        )
    return BENCHMARK_SUITES[suite_name]
