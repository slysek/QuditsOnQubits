"""Classical depth polish of a valid skeleton, without a model.

Two CZs that follow each other on a qubit b commute when the U gate between them on b is diagonal, or
anti-diagonal (then CZ X_b CZ = X_b Z_c adds a Z on the other qubit c). Such a swap keeps the circuit
exactly, with new U angles, and can remove a dependency between two-qubit layers. A bounded search over
these swaps finds the smallest two-qubit depth; one short fit from the new angles then checks the result.
"""
from collections import deque
from time import perf_counter

import numpy as np
from qiskit.synthesis import OneQubitEulerDecomposer

from .architectures import routed_costs
from .oe_evaluator import rank_key
from .skeleton_fit import _circuits, _u_matrix, fit_skeleton
from .state_workloads import layout

# Off-diagonal (or diagonal) entries below this count as zero; the final fit removes the small error.
MONOMIAL_TOLERANCE = 1e-6
MAX_STATES = 5000
# With an architecture, the polish routes this many of the shallowest variants (about 1 s each at 12 qubits).
ROUTED_VARIANTS = 16
# Default time limit of one polish in a discovery campaign; an AME(4,6) polish takes a few seconds.
POLISH_SECONDS = 120
_IDENTITY = np.eye(2)
_Z = np.diag([1.0, -1.0])
_EULER = OneQubitEulerDecomposer("U")


def two_qubit_depth(pairs):
    """Return the number of parallel CZ layers when each CZ goes as early as possible."""
    level, depth = {}, 0
    for a, b in pairs:
        level[a] = level[b] = max(level.get(a, 0), level.get(b, 0)) + 1
        depth = max(depth, level[a])
    return depth


def _slot(width, k, side):
    """Return the angle slice of the U gate on qubit pairs[k][side] after CZ k."""
    start = 3 * width + 6 * k + 3 * side
    return slice(start, start + 3)


def _swapped(pairs, x, width, i, j, b):
    """Return the angles after CZ j goes before CZ i (both on qubit b, nothing on b between them), or None."""
    ib, jb = pairs[i].index(b), pairs[j].index(b)
    between = _u_matrix(*x[_slot(width, i, ib)])
    if max(abs(between[0, 1]), abs(between[1, 0])) < MONOMIAL_TOLERANCE:
        z = _IDENTITY
    elif max(abs(between[0, 0]), abs(between[1, 1])) < MONOMIAL_TOLERANCE:
        z = _Z
    else:
        return None
    y = np.array(x, dtype=float)
    # Before: CZ_i, T on b, CZ_j, U_b and U_c. After: CZ_j, Z on c, U_c, then CZ_i, U_b T on b.
    y[_slot(width, i, ib)] = _EULER.angles(_u_matrix(*x[_slot(width, j, jb)]) @ between)
    y[_slot(width, j, jb)] = 0.0
    y[_slot(width, j, 1 - jb)] = _EULER.angles(_u_matrix(*x[_slot(width, j, 1 - jb)]) @ z)
    return y


def _reorder(pairs, x, width, order):
    """Return pairs and angles in the CZ order `order`; the first layer and the tail stay."""
    start, count = 3 * width, len(pairs)
    blocks = np.asarray(x[start:start + 6 * count]).reshape(count, 6)[order]
    return [pairs[k] for k in order], np.concatenate([x[:start], blocks.ravel(), x[start + 6 * count:]])


def exact_swaps(pairs, x, width):
    """Return (pairs, angles) for each exact swap of two CZs that follow each other on one qubit.

    CZ j moves to just before CZ i if no CZ between them uses its other qubit c, and CZ i moves to just
    after CZ j if no CZ between them uses its other qubit a. Both give the same circuit.
    """
    pairs = [tuple(p) for p in pairs]
    result, last = [], {}
    for j, pair in enumerate(pairs):
        for b in pair:
            i = last.get(b)
            if i is None:
                continue
            a, c = pairs[i][1 - pairs[i].index(b)], pair[1 - pair.index(b)]
            if a == c:
                continue
            y = _swapped(pairs, x, width, i, j, b)
            if y is None:
                continue
            between = pairs[i + 1:j]
            rest = list(range(i + 1, j))
            if not any(c in p for p in between):
                result.append(_reorder(pairs, y, width, [*range(i), j, i, *rest, *range(j + 1, len(pairs))]))
            if not any(a in p for p in between):
                result.append(_reorder(pairs, y, width, [*range(i), *rest, j, i, *range(j + 1, len(pairs))]))
        last[pair[0]] = last[pair[1]] = j
    return result


def variants(pairs, x, width, *, deadline, max_states=MAX_STATES):
    """Search the exact swaps breadth first. Return [(two-qubit depth, pairs, angles)], the start first."""
    pairs = [tuple(p) for p in pairs]
    found = [(two_qubit_depth(pairs), pairs, np.asarray(x, dtype=float))]
    seen, queue = {tuple(pairs)}, deque([found[0][1:]])
    while queue and len(seen) < max_states and perf_counter() < deadline:
        for swapped, y in exact_swaps(*queue.popleft(), width):
            if tuple(swapped) not in seen:
                seen.add(tuple(swapped))
                queue.append((swapped, y))
                found.append((two_qubit_depth(swapped), swapped, y))
    return found


def shallowest(pairs, x, width, *, deadline, max_states=MAX_STATES):
    """Return (two-qubit depth, pairs, angles) of the shallowest variant, the first one found on a tie."""
    return min(variants(pairs, x, width, deadline=deadline, max_states=max_states), key=lambda item: item[0])


def _best_routed(found, base_metrics, width, architecture, objective, deadline):
    """Return (rank, pairs, angles) of the shallow variant with the best routed costs, or None."""
    best = None
    for _, pairs, angles in sorted(found[1:], key=lambda item: item[0])[:ROUTED_VARIANTS]:
        if perf_counter() >= deadline:
            break
        circuit = _circuits(width, pairs, (), angles, 0.0)[0]
        rank = rank_key({**base_metrics, **routed_costs(circuit, architecture, objective=objective)}, objective)
        if best is None or rank > best[0]:
            best = (rank, pairs, angles)
    return best


def polish_depth(state, pairs, x0, *, encoding=None, objective="preparation", deadline, architecture=None):
    """Lower the cost of a valid skeleton with exact CZ swaps.

    A first fit from the start angles x0 gives exact angles; the swaps need them. Without an architecture,
    the shallowest swapped circuit is the candidate. With one, the ROUTED_VARIANTS shallowest swapped
    circuits are routed and the best routed one is the candidate. A last fit checks the candidate.
    encoding=None means free encodings. Return {"pairs", "angles", "metrics", "two_qubit_depth_before"}
    (and "routed_two_qubit_count_before" with an architecture) for a valid circuit with a better rank_key,
    else None. The work stops at the perf_counter() deadline.
    """
    n, _, q = layout(state)
    left = deadline - perf_counter()
    if left <= 0:
        return None
    base = fit_skeleton(state, pairs, encoding, restarts=1, x0=x0, time_budget=left)
    if not base.metrics["is_valid"]:
        return None
    base_metrics = base.metrics
    found = variants(pairs, base.angles, n * q, deadline=deadline)
    if architecture is None:
        depth, chosen, angles = min(found, key=lambda item: item[0])
        if depth >= base.metrics["two_qubit_depth"]:
            return None
    else:
        base_metrics = {**base.metrics, **routed_costs(base.circuit, architecture, objective=objective)}
        best = _best_routed(found, base.metrics, n * q, architecture, objective, deadline)
        if best is None or best[0] <= rank_key(base_metrics, objective):
            return None
        _, chosen, angles = best
    left = deadline - perf_counter()
    if left <= 0:
        return None
    fitted = fit_skeleton(state, chosen, encoding, restarts=1, x0=angles, time_budget=left)
    metrics = fitted.metrics
    if architecture is not None and metrics["is_valid"]:
        metrics = {**metrics, **routed_costs(fitted.circuit, architecture, objective=objective)}
    if not metrics["is_valid"] or rank_key(metrics, objective) <= rank_key(base_metrics, objective):
        return None
    found = {"pairs": chosen, "angles": fitted.angles, "metrics": metrics,
             "two_qubit_depth_before": int(base.metrics["two_qubit_depth"])}
    if architecture is not None:
        found["routed_two_qubit_count_before"] = int(base_metrics["routed_two_qubit_count"])
    return found
