from __future__ import annotations

import inspect
import math
import string
from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import numpy as np

from qudits_on_qubits.reference_experiments import get_reference_experiment

from .basis import omega, ordered_qutrit_eigenbasis


def build_general_graph_bell_settings(
    n: int,
    graph: Any,
    root_edge: tuple[int, int] | None = None,
    lam_fn: Callable[..., complex] | None = None,
    weight_attr: str = "weight",
    split_coefficients: str = "uniform",
    drop_conjugate_half: bool = False,
) -> dict[str, Any]:
    """Generate graph-local Bell terms and the global settings they require.

    This function only produces metadata for Sampler/backend-count workflows. It
    does not build a global Bell operator and does not tensor local observables.

    For every graph edge ``(u, v)`` it creates all ``n x n`` endpoint settings.
    The powers ``1, ..., n - 1`` are stored in term metadata and do not create
    additional measurement settings. When ``drop_conjugate_half=True``, only the
    first half of graph powers is kept, while the full setting list is retained.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if split_coefficients != "uniform":
        raise ValueError("only split_coefficients='uniform' is supported")

    num_parties = _graph_vcount(graph)
    party_order = tuple(range(num_parties))
    edges = _graph_edges(graph, weight_attr)
    if not edges:
        raise ValueError("graph must contain at least one edge")
    edges = _order_edges(edges, root_edge)

    if n == 3 and num_parties == 2 and len(edges) == 1:
        return _two_qutrit_pdf_settings(
            edge=edges[0],
            root_edge=root_edge,
            split_coefficients=split_coefficients,
            drop_conjugate_half=drop_conjugate_half,
        )

    all_powers = tuple(range(1, n))
    term_powers = (
        all_powers[: max(1, (n - 1 + 1) // 2)]
        if drop_conjugate_half
        else all_powers
    )

    measurement_settings_set: set[tuple[str | None, ...]] = set()
    terms: list[dict[str, Any]] = []
    for edge_index, (u, v, weight) in enumerate(edges):
        for endpoint_settings in product(range(n), repeat=2):
            setting = _edge_setting(num_parties, u, v, endpoint_settings)
            measurement_settings_set.add(setting)
            for graph_power in term_powers:
                powers = [0] * num_parties
                powers[u] = graph_power
                powers[v] = graph_power
                coefficient = _term_coefficient(
                    weight=weight,
                    graph_power=graph_power,
                    n=n,
                    num_edges=len(edges),
                    lam_fn=lam_fn,
                )
                terms.append(
                    {
                        "coeff": coefficient,
                        "settings": setting,
                        "powers": tuple(powers),
                        "source": f"edge_{edge_index}:{u}-{v}",
                        "graph_power": graph_power,
                    }
                )

    measurement_settings = sorted(
        measurement_settings_set,
        key=lambda setting: tuple("" if item is None else str(item) for item in setting),
    )
    return {
        "terms": terms,
        "measurement_settings": measurement_settings,
        "party_order": party_order,
        "edges": tuple((u, v, weight) for u, v, weight in edges),
        "root_edge": root_edge,
        "local_dimension": n,
        "drop_conjugate_half": drop_conjugate_half,
        "split_coefficients": split_coefficients,
    }


def _edge_setting(
    num_parties: int,
    u: int,
    v: int,
    endpoint_settings: tuple[int, int],
) -> tuple[str | None, ...]:
    setting: list[str | None] = [None] * num_parties
    setting[u] = f"{_party_label(u)}{endpoint_settings[0]}"
    setting[v] = f"{_party_label(v)}{endpoint_settings[1]}"
    return tuple(setting)


def _party_label(party: int) -> str:
    if party < len(string.ascii_uppercase):
        return string.ascii_uppercase[party]
    return f"P{party}"


def _graph_vcount(graph: Any) -> int:
    if not hasattr(graph, "vcount"):
        raise TypeError("graph must be an igraph-like object with vcount()")
    return int(graph.vcount())


def _graph_edges(graph: Any, weight_attr: str) -> list[tuple[int, int, complex]]:
    if not hasattr(graph, "es"):
        raise TypeError("graph must be an igraph-like object with es")

    weighted = weight_attr in set(graph.es.attributes())
    edges: list[tuple[int, int, complex]] = []
    for edge in graph.es:
        u, v = tuple(edge.tuple)
        weight = edge[weight_attr] if weighted else 1
        edges.append((int(u), int(v), complex(weight)))
    return edges


def _order_edges(
    edges: Sequence[tuple[int, int, complex]],
    root_edge: tuple[int, int] | None,
) -> list[tuple[int, int, complex]]:
    ordered = list(edges)
    if root_edge is None:
        return ordered

    normalized_root = tuple(sorted(root_edge))
    for index, edge in enumerate(ordered):
        if tuple(sorted(edge[:2])) == normalized_root:
            return [ordered[index], *ordered[:index], *ordered[index + 1 :]]
    raise ValueError(f"root_edge {root_edge!r} is not present in graph")


def _term_coefficient(
    weight: complex,
    graph_power: int,
    n: int,
    num_edges: int,
    lam_fn: Callable[..., complex] | None,
) -> complex:
    lam = _call_lam_fn(lam_fn, graph_power, n)
    return complex(weight) / (num_edges * n * lam)


def _call_lam_fn(
    lam_fn: Callable[..., complex] | None,
    graph_power: int,
    n: int,
) -> complex:
    if lam_fn is None:
        return 1.0 + 0.0j

    try:
        parameters = inspect.signature(lam_fn).parameters
    except (TypeError, ValueError):
        return complex(lam_fn(graph_power, n))

    if len(parameters) >= 2:
        return complex(lam_fn(graph_power, n))
    return complex(lam_fn(graph_power))


def _two_qutrit_pdf_settings(
    edge: tuple[int, int, complex],
    root_edge: tuple[int, int] | None,
    split_coefficients: str,
    drop_conjugate_half: bool,
) -> dict[str, Any]:
    u, v, weight = edge
    spec = get_reference_experiment("two_qutrit")
    observables_by_label = {
        observable.label: observable.as_array()
        for observable in spec.observables
    }

    terms: list[dict[str, Any]] = []
    for term in spec.bell_functional.terms:
        if not term.factors:
            continue
        graph_power = int(term.factors[0].outcome_power) % 3
        if drop_conjugate_half and graph_power != 1:
            continue
        terms.append(
            {
                "coeff": complex(weight) * term.sampling_coefficient(),
                "settings": spec.setting_for_term(term),
                "powers": spec.powers_for_term(term),
                "source": f"two_qutrit_pdf:{u}-{v}",
                "graph_power": graph_power,
            }
        )

    measurement_settings = sorted(
        {tuple(term["settings"]) for term in terms},
        key=lambda setting: tuple("" if item is None else str(item) for item in setting),
    )
    return {
        "terms": terms,
        "measurement_settings": measurement_settings,
        "candidate": spec.experiment_id,
        "spec_hash": spec.stable_hash(),
        "party_order": spec.state.party_order,
        "edges": (edge,),
        "root_edge": root_edge,
        "local_dimension": 3,
        "drop_conjugate_half": drop_conjugate_half,
        "split_coefficients": split_coefficients,
        "observables_by_label": observables_by_label,
        "physical_to_logical_outcome_map": dict(
            spec.outcome_convention.measurement_basis_index_map
        ),
        "construction": "two_qutrit_pdf",
    }


def _make_XZ_qutrit_d3() -> tuple[np.ndarray, np.ndarray]:
    w = omega(3)
    X = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        X[(j + 1) % 3, j] = 1.0
    Z = np.diag([w**j for j in range(3)]).astype(complex)
    return X, Z


def _qutrit_lambda_d3(power: int) -> complex:
    if power == 1:
        return complex(np.exp(1j * np.pi / 18))
    if power == 2:
        return complex(np.exp(-1j * np.pi / 18))
    raise ValueError("qutrit lambda is defined only for powers 1 and 2")


def _replacement_phase_exponent(k: int) -> int:
    return k * (k + 1)


def _measurement_observables_qutrit_d3(power: int) -> list[np.ndarray]:
    X, Z = _make_XZ_qutrit_d3()
    w = omega(3)
    lam = _qutrit_lambda_d3(power)
    observables: list[np.ndarray] = []
    for t in range(3):
        matrix = np.zeros((3, 3), dtype=complex)
        for k in range(3):
            phase = w ** (power * t * k)
            replacement_phase = w ** (power * _replacement_phase_exponent(k))
            xzk = X @ np.linalg.matrix_power(Z, k)
            matrix += phase * replacement_phase * np.linalg.matrix_power(xzk, power)
        observables.append(lam * matrix / math.sqrt(3))
    return observables


def _root_expectation_scale(
    measurement: np.ndarray,
    desired: np.ndarray,
    power: int,
) -> complex:
    V, _ = ordered_qutrit_eigenbasis(measurement)
    diagonal = V.conj().T @ desired @ V
    roots = np.array([omega(3) ** ((power * a) % 3) for a in range(3)])
    values = np.diag(diagonal) / roots
    if not np.allclose(values, values[0], atol=1e-8):
        raise ValueError("desired observable is not diagonal in the measurement basis")
    off_diagonal = diagonal - np.diag(np.diag(diagonal))
    if not np.allclose(off_diagonal, 0.0, atol=1e-8):
        raise ValueError("desired observable has off-diagonal terms in measurement basis")
    return complex(values[0])
