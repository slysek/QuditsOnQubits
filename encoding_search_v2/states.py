import re
from dataclasses import dataclass
from typing import Optional


_GHZ_STAR_ALIASES = {"ghz_star", "ghz_n"}
_GHZ_STAR_SLUG_RE = re.compile(r"^(ghz_star|ghz_n)_(\d+)$")


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


def _validate_fixed_n(state_name: str, n_qutrits: Optional[int], expected: int):
    if n_qutrits is not None and int(n_qutrits) != expected:
        raise ValueError(
            f"{state_name} has fixed n_qutrits={expected}; got {n_qutrits}."
        )


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


def _resolve_ghz_star_n(state_name: str, n_qutrits: Optional[int]) -> Optional[int]:
    if state_name in _GHZ_STAR_ALIASES:
        if n_qutrits is None:
            raise ValueError(f"{state_name} requires --n-qutrits or --n-values.")
        return _validate_n_qutrits(n_qutrits, state_name=state_name)

    match = _GHZ_STAR_SLUG_RE.match(state_name)
    if not match:
        return None

    n_from_slug = _validate_n_qutrits(int(match.group(2)), state_name=state_name)
    if n_qutrits is not None and int(n_qutrits) != n_from_slug:
        raise ValueError(
            f"{state_name} already encodes n_qutrits={n_from_slug}; got {n_qutrits}."
        )
    return n_from_slug


def resolve_benchmark_state(
    state_name: str,
    n_qutrits: Optional[int] = None,
) -> BenchmarkStateSpec:
    name = str(state_name).strip()

    if name == "two_qutrit":
        _validate_fixed_n(name, n_qutrits, expected=2)
        return BenchmarkStateSpec(
            state_name="two_qutrit",
            state_id="two_qutrit",
            state_family="two_qutrit",
            num_qutrits=2,
            graph_type="star",
            edges=star_graph_edges(2),
        )

    if name == "ghz3":
        _validate_fixed_n(name, n_qutrits, expected=3)
        return BenchmarkStateSpec(
            state_name="ghz3",
            state_id="ghz3",
            state_family="ghz_star",
            num_qutrits=3,
            graph_type="star",
            edges=star_graph_edges(3),
        )

    if name == "ame43":
        _validate_fixed_n(name, n_qutrits, expected=4)
        return BenchmarkStateSpec(
            state_name="ame43",
            state_id="ame43",
            state_family="ame43",
            num_qutrits=4,
            graph_type="ame43",
            edges=[(0, 1), (0, 1), (1, 2), (2, 3), (3, 0)],
        )

    ghz_star_n = _resolve_ghz_star_n(name, n_qutrits)
    if ghz_star_n is not None:
        return BenchmarkStateSpec(
            state_name="ghz_star",
            state_id=f"ghz_star_{ghz_star_n}",
            state_family="ghz_star",
            num_qutrits=ghz_star_n,
            graph_type="star",
            edges=star_graph_edges(ghz_star_n),
        )

    raise ValueError(
        f"Unknown benchmark state: {state_name!r}. "
        "Expected two_qutrit, ghz3, ame43, ghz_star, ghz_n, or ghz_star_<n>."
    )
