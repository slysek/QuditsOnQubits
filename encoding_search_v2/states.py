from dataclasses import dataclass
from typing import Optional

from QuditsOnQubits.graph_states import (
    GraphStateSpec,
    resolve_graph_state,
    resolve_graph_state_or_raise,
)


@dataclass(frozen=True)
class BenchmarkStateSpec:
    state_name: str
    state_id: str
    state_family: str
    num_qutrits: int
    graph_type: str
    edges: list[tuple[int, int]]


def _validate_n_qutrits(n_qutrits: int, *, state_name: str) -> int:
    n = int(n_qutrits)
    if n < 2:
        raise ValueError(f"{state_name} requires n_qutrits >= 2, got {n}.")
    return n


def star_graph_edges(n_qutrits: int) -> list[tuple[int, int]]:
    n = _validate_n_qutrits(n_qutrits, state_name="ghz_star")
    return [(0, leaf) for leaf in range(1, n)]


def parse_n_values(value: Optional[str]) -> tuple[int, ...]:
    if value is None:
        return ()
    values = tuple(
        _validate_n_qutrits(int(item.strip()), state_name="ghz_star")
        for item in value.split(",")
        if item.strip()
    )
    if not values:
        raise ValueError("--n-values must contain at least one integer.")
    return values


def _to_benchmark_state_spec(
    raw_name: str,
    spec: GraphStateSpec,
) -> BenchmarkStateSpec:
    """Konwertuj :class:`GraphStateSpec` -> :class:`BenchmarkStateSpec`.

    Argument ``raw_name`` to oryginalna nazwa wpisana przez uzytkownika
    (np. ``"ghz_star"`` dla wariantu z osobnym ``--n-values``); pozostaje
    w polu :attr:`BenchmarkStateSpec.state_name`, podczas gdy stabilny
    identyfikator wynikow trafia do :attr:`BenchmarkStateSpec.state_id`.
    """
    return BenchmarkStateSpec(
        state_name=raw_name,
        state_id=spec.state_id,
        state_family=spec.state_family,
        num_qutrits=spec.num_qutrits,
        graph_type=spec.graph_type,
        edges=list(spec.edges),
    )


def resolve_benchmark_state(
    state_name: str,
    n_qutrits: Optional[int] = None,
) -> BenchmarkStateSpec:
    """Resolver dla pipeline'u v2 — opakowuje wspoldzielony rejestr.

    Akceptuje:
    * stale: ``two_qutrit``, ``ghz3``, ``ame43``,
    * GHZ/star: ``ghz_star``, ``ghz_n``, ``ghz_star_<n>``, ``ghz<n>``,
    * pozostale rodziny grafow z :mod:`QuditsOnQubits.graph_states`
      (``path<n>``, ``cycle<n>``, ``wheel<n>``, ``complete<n>``,
      ``cluster<r>x<c>``).
    """
    name = str(state_name).strip()

    if name in ("ghz_star", "ghz_n"):
        if n_qutrits is None:
            raise ValueError(f"{name} requires --n-qutrits or --n-values.")

    spec = resolve_graph_state_or_raise(name, n_qutrits=n_qutrits)

    if name in ("ghz_star", "ghz_n"):
        # Zachowaj historyczne pole ``state_name`` ("ghz_star"),
        # podczas gdy state_id jest stabilnym slug-iem.
        return _to_benchmark_state_spec(raw_name="ghz_star", spec=spec)

    return _to_benchmark_state_spec(raw_name=name, spec=spec)


def list_known_state_aliases() -> tuple[str, ...]:
    """Zwroc czytelna liste przykladow nazw stanow, dla komunikatow CLI."""
    return (
        "two_qutrit",
        "ghz3",
        "ame43",
        "ghz<n>  (np. ghz5)",
        "ghz_star_<n>  (np. ghz_star_5)",
        "path<n>",
        "cycle<n>",
        "wheel<n>",
        "complete<n>",
        "cluster<r>x<c>  (np. cluster2x3)",
    )


__all__ = [
    "BenchmarkStateSpec",
    "list_known_state_aliases",
    "parse_n_values",
    "resolve_benchmark_state",
    "star_graph_edges",
]
