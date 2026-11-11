"""Fit every angle of a CZ skeleton to an encoded qudit state.

The ansatz puts a U gate on each qubit, then a U gate on both qubits after each CZ.
With free encodings, each party block also gets a tail of CZs and U layers that can map any
encoding to the canonical one: three CZs reach any two-qubit unitary (qutrits), and the
three-qubit tail of an AME(4,6) party is checked numerically. The fit then uses the canonical
target, and the tail gives the encoding of each party.
For each valid state, a separate short decoder to the canonical encoding gives the readout cost.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from scipy.optimize import least_squares

from .decoders import decoder_circuit, with_decoder_check
from .encodings import canonical_encoding
from .state_workloads import encoded_target, evaluate_workload, layout, logical_state, party_encodings

DECODER_CZ_PER_QUTRIT = 3
# CZ pairs of the free-encoding tail of one party, in block qubit indices. For three qubits,
# 13 CZs mapped each random 8 x 6 isometry to the canonical one (residual 4e-15); 12 did not
# (residual 3e-2). An 8 x 6 isometry has 60 parameters, and each CZ adds at most 4.
TAIL_PAIRS = {2: ((0, 1),) * DECODER_CZ_PER_QUTRIT, 3: (((0, 1), (1, 2), (0, 2)) * 5)[:13]}
MAX_SKELETON_CZ = 128
MAX_RESTARTS = 64
STRICT_TOLERANCE = 1e-10
CHUNK_NFEV = 50
# A restart that improves its error by less than 0.5 % in one chunk is in a local minimum.
STALL_RATIO = 0.995


@dataclass(frozen=True)
class SkeletonFit:
    """Keep the checked preparation, the decoder and the fit counters.

    encodings has shape (n, 2**q, d). decoder maps them to the canonical encoding; it is None
    for an invalid state. angles are the fitted ansatz angles, a warm start for similar skeletons.
    """

    circuit: QuantumCircuit
    decoder: QuantumCircuit | None
    encodings: np.ndarray
    metrics: dict
    restarts: int
    nfev: int
    fit_error: float
    angles: np.ndarray | None = None


class _Ansatz:
    """Exact statevector of the U-sandwich ansatz with its forward-mode Jacobian."""

    def __init__(self, width, pairs):
        self.width, self.dim = width, 1 << width
        indices = np.arange(self.dim)
        self.ops = [("u", q) for q in range(width)]
        for a, b in pairs:
            self.ops += [("cz", ((indices >> a) & (indices >> b) & 1) == 1), ("u", a), ("u", b)]
        self.parameters = 3 * sum(kind == "u" for kind, _ in self.ops)

    def run(self, x, jacobian=False):
        columns = 1 + self.parameters if jacobian else 1
        state = np.zeros((self.dim, columns), dtype=complex)
        state[0, 0] = 1
        slot = 0
        for kind, arg in self.ops:
            if kind == "cz":
                state[arg] *= -1
                continue
            theta, phi, lam = x[3 * slot:3 * slot + 3]
            cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
            ep, el = np.exp(1j * phi), np.exp(1j * lam)
            a, b, c, d = cosine, -el * sine, ep * sine, ep * el * cosine
            view = state.reshape(1 << (self.width - 1 - arg), 2, 1 << arg, columns)
            # Columns of later angles are still zero, so the gate skips them.
            used = 1 + 3 * slot if jacobian else 1
            zero, one = view[:, 0, :, :used].copy(), view[:, 1, :, :used].copy()
            view[:, 0, :, :used] = a * zero + b * one
            view[:, 1, :, :used] = c * zero + d * one
            if jacobian:
                derivatives = ((-sine / 2, -el * cosine / 2, ep * cosine / 2, -ep * el * sine / 2),
                               (0, 0, 1j * c, 1j * d), (0, 1j * b, 0, 1j * d))
                for offset, (da, db, dc, dd) in enumerate(derivatives):
                    view[:, 0, :, used + offset] = da * zero[..., 0] + db * one[..., 0]
                    view[:, 1, :, used + offset] = dc * zero[..., 0] + dd * one[..., 0]
            slot += 1
        return state


class _Deadline(Exception):
    """The time budget ended during a least-squares call."""


def _fit(ansatz, target, x0, max_nfev, deadline=np.inf):
    # The best point so far: a fit that stops at the deadline returns it.
    best = {"cost": np.inf, "x": np.asarray(x0, dtype=float), "calls": 0}

    def residual(x):
        if perf_counter() > deadline:
            raise _Deadline
        state = ansatz.run(x)[:, 0]
        # Remove the target component so that the global phase is free.
        r = state - target * np.vdot(target, state)
        values = np.r_[r.real, r.imag]
        cost = float(values @ values)
        best["calls"] += 1
        if cost < best["cost"]:
            best.update(cost=cost, x=np.array(x, dtype=float))
        return values

    def jacobian(x):
        if perf_counter() > deadline:
            raise _Deadline
        columns = ansatz.run(x, jacobian=True)[:, 1:]
        columns = columns - np.outer(target, target.conj() @ columns)
        return np.vstack((columns.real, columns.imag))

    method = "lm" if 2 * ansatz.dim >= ansatz.parameters else "trf"
    try:
        solution = least_squares(residual, x0, jac=jacobian, method=method, max_nfev=max_nfev,
                                 ftol=1e-15, xtol=1e-15, gtol=1e-15)
        x, nfev = solution.x, int(solution.nfev)
    except np.linalg.LinAlgError:
        # LAPACK can fail on an ill-conditioned Jacobian. This counts as one failed restart.
        return x0, np.inf, 0.0, max_nfev
    except _Deadline:
        # One Jacobian of a 12-qubit skeleton can take seconds, so the time check is per call, not per chunk.
        x, nfev = best["x"], best["calls"]
    # 4*pi is a period of all three U angles. Flat directions let the solver drift to angles
    # near 1e7, where Qiskit matrices are no longer unitary to 1e-10. The error below uses
    # the wrapped angles, so any precision loss shows in the result.
    x = np.mod(x, 4 * np.pi)
    if not np.isfinite(x).all():
        return x0, np.inf, 0.0, nfev
    return (x, *_error(ansatz, target, x), nfev)


def _error(ansatz, target, x):
    """Return the phase-aligned error norm and the phase of the overlap."""
    state = ansatz.run(x)[:, 0]
    overlap = np.vdot(target, state)
    return float(np.linalg.norm(state - np.exp(1j * np.angle(overlap)) * target)), float(np.angle(overlap))


def _restart(ansatz, target, x, max_nfev, deadline=np.inf):
    """Fit in chunks and stop early when the error stalls above the tolerance or time ends.

    A start point that already reaches the state (a warm start from the same structure) is kept without a fit.
    """
    error, phase = _error(ansatz, target, x)
    if error <= STRICT_TOLERANCE:
        return x, error, phase, 0
    used, previous = 0, np.inf
    while True:
        budget = min(CHUNK_NFEV, max_nfev - used)
        x, error, phase, nfev = _fit(ansatz, target, x, budget, deadline)
        used += nfev
        converged = nfev < budget
        stalled = error > STALL_RATIO * previous or perf_counter() > deadline
        if error <= STRICT_TOLERANCE or converged or used >= max_nfev or stalled:
            return x, error, phase, used
        previous = error


def _circuits(width, pairs, decoder_pairs, x, phase):
    angles = iter(np.asarray(x, dtype=float).reshape(-1, 3))
    preparation = QuantumCircuit(width, global_phase=-phase)
    for q in range(width):
        preparation.u(*next(angles), q)
    for a, b in pairs:
        preparation.cz(a, b)
        preparation.u(*next(angles), a)
        preparation.u(*next(angles), b)
    if not decoder_pairs:
        return preparation, None
    decoder = QuantumCircuit(width)
    for a, b in decoder_pairs:
        decoder.cz(a, b)
        decoder.u(*next(angles), a)
        decoder.u(*next(angles), b)
    return preparation, decoder


def _tail_blocks(tail, n, q=2):
    """Split the fit tail into one q-qubit circuit D_k per party."""
    blocks = []
    for k in range(n):
        block = QuantumCircuit(q)
        for item in tail.data:
            wires = [tail.find_bit(b).index - q * k for b in item.qubits]
            if all(0 <= w < q for w in wires):
                block.append(item.operation, wires)
        blocks.append(block)
    return blocks


def _validate_pairs(pairs, width):
    if not isinstance(pairs, (list, tuple)) or len(pairs) > MAX_SKELETON_CZ:
        raise ValueError(f"The skeleton must be a list of at most {MAX_SKELETON_CZ} CZ pairs")
    checked = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each CZ must be a pair of qubit indices")
        a, b = pair
        if any(isinstance(q, bool) or not isinstance(q, Integral) or not 0 <= q < width for q in pair) or a == b:
            raise ValueError("Each CZ needs two different qubits inside the circuit")
        checked.append((int(a), int(b)))
    return tuple(checked)


def fit_skeleton(name, pairs, encoding=None, *, restarts=8, seed=0, max_nfev=400, time_budget=None, x0=None):
    """Fit all angles of the skeleton and check the state.

    With encoding=None, each party gets a free encoding and a decoder to the canonical one.
    Otherwise the target uses the given encoding (one matrix or one per party). Restarts stop
    after the first success, and the seed fixes their start points. After time_budget seconds,
    the current restart stops at its next chunk and no new restart starts.
    x0 (see warm_angles) is the start point of the first restart; the other restarts start at random.
    """
    n, d, q = layout(name)
    width = q * n
    pairs = _validate_pairs(pairs, width)
    if isinstance(restarts, bool) or not isinstance(restarts, Integral) or not 1 <= restarts <= MAX_RESTARTS:
        raise ValueError(f"restarts must be an integer from 1 to {MAX_RESTARTS}")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or not 1 <= max_nfev <= 10_000:
        raise ValueError("max_nfev must be an integer from 1 to 10000")
    if time_budget is not None and (isinstance(time_budget, bool) or not isinstance(time_budget, Real)
                                    or not np.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("time_budget must be a positive number of seconds")
    deadline = np.inf if time_budget is None else perf_counter() + time_budget
    free = encoding is None
    decoder_pairs = tuple((q * k + a, q * k + b) for k in range(n) for a, b in TAIL_PAIRS[q]) if free else ()
    target, _ = encoded_target(name, canonical_encoding(d) if free else encoding)
    ansatz = _Ansatz(width, pairs + decoder_pairs)
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (ansatz.parameters,) or not np.isfinite(x0).all():
            raise ValueError(f"x0 must hold {ansatz.parameters} finite angles")
    best = None
    nfev = 0
    for restart in range(1, int(restarts) + 1):
        start = (x0 if restart == 1 and x0 is not None
                 else np.random.default_rng([int(seed), restart]).uniform(0, 2 * np.pi, ansatz.parameters))
        x, error, phase, used = _restart(ansatz, target, start, int(max_nfev), deadline)
        nfev += used
        if best is None or error < best[1]:
            best = (x, error, phase)
        if error <= STRICT_TOLERANCE or perf_counter() > deadline:
            break
    x, error, phase = best
    circuit, tail = _circuits(width, pairs, decoder_pairs, x, phase)
    blocks = _tail_blocks(tail, n, q) if free else None
    if free:
        # E_k = D_k^dagger B, with D_k the tail of party k and B the canonical encoding.
        encodings = np.asarray([Operator(b).data.conj().T @ canonical_encoding(d) for b in blocks])
    else:
        encodings = party_encodings(name, encoding)
    metrics = evaluate_workload(name, circuit, encodings, tolerance=STRICT_TOLERANCE)
    metrics["decoder_cz_count"] = 0
    decoder = None
    if metrics["is_valid"]:
        decoder = decoder_circuit(encodings, blocks)
        metrics = with_decoder_check(name, circuit, decoder, metrics)
    return SkeletonFit(circuit, decoder, encodings, metrics, restart, nfev, error, x)


def skeleton_of(circuit):
    """Return the CZ pairs of a U/U3/CZ circuit in gate order."""
    return tuple(tuple(circuit.find_bit(q).index for q in item.qubits)
                 for item in circuit.data if item.operation.name == "cz")


def _u_matrix(theta, phi, lam):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def skeleton_angles(circuit):
    """Return the CZ pairs of a U/U3/CZ circuit and ansatz angles that give the same state up to a phase.

    The one-qubit gates of a qubit before its first CZ become its first U. The gates after a CZ,
    up to the next CZ on the same qubit, become the U after that CZ.
    """
    width = circuit.num_qubits
    pending = [np.eye(2, dtype=complex) for _ in range(width)]
    slot, slots, pairs = list(range(width)), {}, []
    for item in circuit.data:
        qubits = [circuit.find_bit(b).index for b in item.qubits]
        if item.operation.name == "cz":
            for offset, b in enumerate(qubits):
                slots[slot[b]], pending[b] = pending[b], np.eye(2, dtype=complex)
                slot[b] = width + 2 * len(pairs) + offset
            pairs.append(tuple(qubits))
        else:
            pending[qubits[0]] = np.asarray(item.operation.to_matrix()) @ pending[qubits[0]]
    for b in range(width):
        slots[slot[b]] = pending[b]
    euler = OneQubitEulerDecomposer("U")
    return tuple(pairs), np.concatenate([euler.angles(slots[s]) for s in range(width + 2 * len(pairs))])


def _block_unitary(y, q):
    """Return the unitary of a U layer on q qubits followed by the tail of TAIL_PAIRS[q]. Qubit 0 is the low bit."""
    angles = iter(y.reshape(-1, 3))
    indices = np.arange(1 << q)
    unitary = np.eye(1 << q, dtype=complex)

    def gate(b):
        return np.kron(np.kron(np.eye(1 << (q - 1 - b)), _u_matrix(*next(angles))), np.eye(1 << b))

    for b in range(q):
        unitary = gate(b) @ unitary
    for a, b in TAIL_PAIRS[q]:
        unitary = np.where(((indices >> a) & (indices >> b) & 1)[:, None] == 1, -unitary, unitary)
        unitary = gate(a) @ unitary
        unitary = gate(b) @ unitary
    return unitary


def _fit_tail(encoding, restarts=4):
    """Return (U layer matrices, tail angles) with tail . layer . E = B up to a phase, or None."""
    rows, d = encoding.shape
    q = rows.bit_length() - 1
    canonical = canonical_encoding(d)

    def residual(y):
        r = _block_unitary(y[:-1], q) @ encoding - np.exp(1j * y[-1]) * canonical
        return np.r_[r.real.ravel(), r.imag.ravel()]

    for restart in range(restarts):
        y0 = np.random.default_rng([q, restart]).uniform(0, 2 * np.pi, 3 * (q + 2 * len(TAIL_PAIRS[q])) + 1)
        solution = least_squares(residual, y0, method="trf", max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        if np.linalg.norm(solution.fun) <= 1e-11:
            y = np.mod(solution.x[:-1], 4 * np.pi)
            return [_u_matrix(*y[3 * b:3 * b + 3]) for b in range(q)], y[3 * q:]
    return None


def known_angles(circuit, encodings=None):
    """Return (pairs, angles) of a valid circuit for warm starts, or None.

    With encodings (free mode), each party block also gets the angles of its tail. The first U layer
    of the tail fit moves into the last U of the circuit on each qubit of the block.
    """
    pairs, x = skeleton_angles(circuit)
    if encodings is None:
        return pairs, x
    encodings = np.asarray(encodings, dtype=complex)
    q = encodings.shape[1].bit_length() - 1
    width = circuit.num_qubits
    last = list(range(width))
    for i, (a, b) in enumerate(pairs):
        last[a], last[b] = width + 2 * i, width + 2 * i + 1
    euler, tails = OneQubitEulerDecomposer("U"), []
    for k, encoding in enumerate(encodings):
        fitted = _fit_tail(encoding)
        if fitted is None:
            return None
        for b, layer in enumerate(fitted[0]):
            s = last[q * k + b]
            x[3 * s:3 * s + 3] = euler.angles(layer @ _u_matrix(*x[3 * s:3 * s + 3]))
        tails.append(fitted[1])
    return pairs, np.concatenate([x, *tails])


def _common(a, b):
    """Return the index pairs (i, j) of a longest common subsequence of a and b."""
    table = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            table[i, j] = table[i + 1, j + 1] + 1 if a[i] == b[j] else max(table[i + 1, j], table[i, j + 1])
    i = j = 0
    matches = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            matches.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1, j] >= table[i, j + 1]:
            i += 1
        else:
            j += 1
    return matches


def warm_angles(pairs, known, width, tail=0):
    """Return start angles for pairs from the known (pairs, angles) item with the most common CZs, or None.

    The common CZs (a longest common subsequence) keep their U angles; a new CZ gets identity U gates.
    The first U layer and the angles of the tail CZs (tail > 0 for free encodings) are copied.
    """
    pairs = [tuple(p) for p in pairs]
    best = None
    for known_pairs, angles in known:
        try:
            known_pairs, angles = [tuple(map(int, p)) for p in known_pairs], np.asarray(angles, dtype=float)
        except (TypeError, ValueError):
            continue  # A damaged item only loses its warm start.
        if angles.shape != (3 * width + 6 * (len(known_pairs) + tail),) or not np.isfinite(angles).all():
            continue
        matches = _common([tuple(sorted(p)) for p in pairs], [tuple(sorted(p)) for p in known_pairs])
        # Most common CZs first, then fewest unmatched known CZs: the entry of the same skeleton wins.
        rank = (len(matches), len(matches) - len(known_pairs))
        if best is None or rank > best[0]:
            best = (rank, known_pairs, angles, matches)
    if best is None:
        return None
    _, known_pairs, angles, matches = best
    x = np.zeros(3 * width + 6 * (len(pairs) + tail))
    x[:3 * width] = angles[:3 * width]
    for i, j in matches:
        source = angles[3 * width + 6 * j:3 * width + 6 * j + 6]
        # The two U gates after a CZ follow its qubit order.
        x[3 * width + 6 * i:3 * width + 6 * i + 6] = source if pairs[i] == known_pairs[j] else np.r_[source[3:], source[:3]]
    # Keep the U gates after removed CZs. Merge each U into the previous slot on its wire.
    matched = {j: i for i, j in matches}
    slots = list(range(width))
    euler = OneQubitEulerDecomposer("U")
    for j, pair in enumerate(known_pairs):
        if j in matched:
            i = matched[j]
            for wire in pair:
                slots[wire] = width + 2 * i + pairs[i].index(wire)
        else:
            for k, wire in enumerate(pair):
                slot = 3 * slots[wire]
                old = angles[3 * width + 6 * j + 3 * k:3 * width + 6 * j + 3 * k + 3]
                x[slot:slot + 3] = euler.angles(_u_matrix(*old) @ _u_matrix(*x[slot:slot + 3]))
    if tail:
        x[-6 * tail:] = angles[-6 * tail:]
    return x


def cut_requirements(name):
    """Return (parties, minimum CZs across the cut) for each bipartition of the state.

    One CZ across the cut at most doubles the Schmidt rank r, and a change of encoding does
    not change it. So each valid circuit has at least ceil(log2(r)) CZs across the cut.
    For a qutrit graph state, r is 3**k with k the rank mod 3 of the cut adjacency block.
    """
    n, d, _ = layout(name)
    # Axis j of the tensor is party n-1-j, because party 0 is the lowest digit.
    tensor = logical_state(name).reshape((d,) * n)
    result = []
    for mask in range(1, 1 << (n - 1)):
        side = [k for k in range(n) if mask >> k & 1]
        rest = [k for k in range(n) if not mask >> k & 1]
        matrix = tensor.transpose([n - 1 - k for k in side + rest]).reshape(d ** len(side), -1)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(singular > 1e-9 * singular[0]))
        result.append((tuple(side), int(np.ceil(np.log2(rank) - 1e-12))))
    return result


def cut_violations(name, pairs):
    """Return (parties, needed, present) for each cut with too few crossing CZs.

    Qubit b belongs to party b // q. A CZ crosses a cut if its parties are on different sides.
    A skeleton with a violation cannot reach the state with any angles or encodings.
    """
    n, _, q = layout(name)
    pairs = _validate_pairs(pairs, q * n)
    result = []
    for side, needed in cut_requirements(name):
        present = sum((a // q in side) != (b // q in side) for a, b in pairs)
        if present < needed:
            result.append((side, needed, present))
    return result


def cut_seed(name):
    """Return a CZ list that meets every cut condition, from the target state alone (no known circuit).

    Each step adds one CZ across the first cut with too few CZs; the qubit inside each party changes
    in turn. The result is a start point for a search, usually not a valid circuit.
    """
    n, _, q = layout(name)
    pairs = []
    while violations := cut_violations(name, pairs):
        side = violations[0][0]
        other = next(k for k in range(n) if k not in side)
        pairs.append((q * side[0] + len(pairs) % q, q * other + len(pairs) % q))
    return tuple(pairs)
