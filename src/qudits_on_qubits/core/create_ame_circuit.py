from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, StatePreparation
import numpy as np
from qiskit import qpy
from igraph import Graph
from qiskit.synthesis import TwoQubitWeylDecomposition

from qudits_on_qubits.core.project_paths import quantum_circuits_resource


VALID_ENCODING_STRATEGIES = ("append_w", "prepared_w_then_conjugated_entanglers")


def create_ame_circuit(n=None, dim=3, graph_type="star", graph=None,
                       basis=None, E_new=None,
                       encoding_strategy="append_w"):
    """Construct an AME graph-state circuit.

    Parameters
    ----------
    E_new : np.ndarray, shape (4, 3), optional
        New qutrit encoding isometry C^3 -> C^4, supported only for dim=3.
        With append_w, build in the base encoding and append an encoding-change
        gate W to each qutrit.
    encoding_strategy : str
        "append_w" (default): standard circuit followed by local W gates.
        "prepared_w_then_conjugated_entanglers": prepare W|+> from |00>
        locally using StatePreparation, with entanglers (W ⊗ W) CZ (W† ⊗ W†)."""

    if encoding_strategy not in VALID_ENCODING_STRATEGIES:
        raise ValueError(
            f"Unknown encoding_strategy: {encoding_strategy!r}. "
            f"Allowed: {VALID_ENCODING_STRATEGIES}"
        )

    if graph is None and n is None:
        raise ValueError("Provide either `graph` or the vertex count `n`.")

    if E_new is not None and dim != 3:
        raise ValueError("Encoding changes (E_new) are supported only for dim=3.")

    if graph is None:
        if graph_type == "star":
            edges = [[0, i] for i in range(1, n)]
        elif graph_type == "line":
            edges = [[i, i + 1] for i in range(n - 1)]
        else:
            raise ValueError(
                f"Unknown graph type: {graph_type}. "
                "Allowed values are 'star' or 'line'."
            )
        graph = Graph(n, edges=edges)

    if encoding_strategy == "append_w":
        qc = _build_circuit_append_w(graph, dim, E_new=E_new)
    else:
        qc = _build_circuit_prepared_w_then_conjugated_entanglers(
            graph, dim, E_new=E_new,
        )

    if basis is not None:
        T = change_basis(basis, dim)
        T_qc = TwoQubitWeylDecomposition(T).circuit()

        for i in range(graph.vcount()):
            qc.append(T_qc, [2 * i, 2 * i + 1])

    return qc, graph


def _load_qpy_gate(filename):
    with quantum_circuits_resource(filename).open("rb") as fd:
        return qpy.load(fd)[0]


def _build_circuit_append_w(graph, dim, E_new=None):
    """Apply F to each qutrit, CZ on graph edges, and local W gates last."""
    Fgate, CZgate = _load_gates_for_dim(dim)

    W_qc = None
    if E_new is not None:
        _, W_qc, _ = _build_encoding_change_circuits(E_new)

    n = graph.vcount()
    qubit_list = [[2 * i, 2 * i + 1] for i in range(n)]
    edge_list = _build_edge_list(graph, qubit_list)

    qc = QuantumCircuit(2 * n)

    for pair in qubit_list:
        qc.append(Fgate, pair)

    for edge in edge_list:
        qc.append(CZgate, edge)

    if W_qc is not None:
        for pair in qubit_list:
            qc.append(W_qc, pair)

    return qc


def _build_circuit_prepared_w_then_conjugated_entanglers(graph, dim, E_new=None):
    """Prepare W|+> from |00> on each qutrit using StatePreparation.
    Apply (W ⊗ W) CZ (W† ⊗ W†) on each graph edge."""
    if dim != 3:
        raise ValueError(
            "Strategy 'prepared_w_then_conjugated_entanglers' requires dim=3."
        )

    _, CZgate = _load_gates_for_dim(dim)
    Fgate, _ = _load_gates_for_dim(dim)

    n = graph.vcount()
    qubit_list = [[2 * i, 2 * i + 1] for i in range(n)]
    edge_list = _build_edge_list(graph, qubit_list)

    qc = QuantumCircuit(2 * n)

    if E_new is not None:
        W, W_qc, Wdag_qc = _build_encoding_change_circuits(E_new)

        local_prep = _build_local_w_plus_preparation(W, Fgate)
        for pair in qubit_list:
            qc.append(local_prep, pair)

        conjugated_cz = _build_conjugated_cz_block(W_qc, Wdag_qc, CZgate)
        for edge in edge_list:
            qc.append(conjugated_cz, edge)
    else:
        for pair in qubit_list:
            qc.append(Fgate, pair)
        for edge in edge_list:
            qc.append(CZgate, edge)

    return qc


def _load_gates_for_dim(dim):
    """Return (Fgate, CZgate) for the given qudit dimension."""
    if dim == 3:
        return _load_qpy_gate("Fgate3.qpy"), _load_qpy_gate("CZgate3.qpy")
    if dim == 4:
        return _load_qpy_gate("Fgate4.qpy"), _load_qpy_gate("CZgate4cor.qpy")
    raise ValueError(f"Unsupported dimension: {dim}")


def _build_edge_list(graph, qubit_list):
    """Build the 4-qubit edge index list from graph edges."""
    edge_list = []
    for u, v in graph.get_edgelist():
        if u != v:
            edge_list.append([
                qubit_list[u][0],
                qubit_list[u][1],
                qubit_list[v][0],
                qubit_list[v][1],
            ])
    return edge_list


def _build_local_w_plus_preparation(W, Fgate):
    """Prepare W|+> from |00> as a 2-qubit StatePreparation.

    |+> in qutrit-on-2-qubit encoding = Fgate|00>,
    so the target local state is W @ Fgate|00> = W|+>.
    """
    from qiskit.quantum_info import Operator

    F_op = Operator(Fgate).data
    plus_state = F_op @ np.array([1, 0, 0, 0], dtype=complex)
    psi_local = W @ plus_state
    psi_local = psi_local / np.linalg.norm(psi_local)

    return StatePreparation(psi_local, label="W|+>")


def _build_conjugated_cz_block(W_qc, Wdag_qc, CZgate):
    """Build the 4-qubit entangler: (W ⊗ W) CZ (W† ⊗ W†)."""
    qc = QuantumCircuit(4, name="WW_CZ_WdagWdag")

    qc.append(Wdag_qc, [0, 1])
    qc.append(Wdag_qc, [2, 3])

    qc.append(CZgate, [0, 1, 2, 3])

    qc.append(W_qc, [0, 1])
    qc.append(W_qc, [2, 3])

    return qc


def _build_encoding_change_circuits(E_new):
    """Build W together with unitary gate blocks for W and Wdag."""
    from .encoding_change_unitary import build_encoding_change_unitary

    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W has shape {W.shape}; expected (4, 4)"

    W_qc = UnitaryGate(W, label="W")
    Wdag_qc = UnitaryGate(W.conj().T, label="Wdag")

    W_qc.name = "W"
    Wdag_qc.name = "Wdag"
    return W, W_qc, Wdag_qc


def change_basis(mtx, dim):
    qubit0 = np.array([[1], [0]])
    qubit1 = np.array([[0], [1]])

    q0 = np.kron(qubit0, qubit0)
    q1 = np.kron(qubit0, qubit1)
    q2 = np.kron(qubit1, qubit0)
    q3 = np.kron(qubit1, qubit1)

    if np.array_equal((mtx.transpose().conjugate() @ mtx).round(6), np.identity(4)):
        pi_old = np.column_stack((q0, q1, q2, q3))
        pi_new = mtx
        T = pi_new @ pi_old.conjugate().transpose()
        return T

    raise ValueError("The supplied matrix is not unitary", mtx)
