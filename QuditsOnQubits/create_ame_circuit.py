from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, StatePreparation
import numpy as np
from qiskit import qpy
from igraph import Graph
from qiskit.synthesis import TwoQubitWeylDecomposition

from QuditsOnQubits.project_paths import quantum_circuits_path


VALID_ENCODING_STRATEGIES = ("append_w", "prepared_w_then_conjugated_entanglers")


def create_ame_circuit(n=None, dim=3, graph_type="star", graph=None,
                       basis=None, E_new=None,
                       encoding_strategy="append_w"):
    """
    Tworzy obwod AME z grafu.

    Parametry
    ---------
    E_new : np.ndarray, shape (4,3), optional
        Nowa mapa kodowania qutrytu (izometria C^3 -> C^4).
        Jesli podana, obwod jest budowany w bazowym kodowaniu, a na koncu
        kazdego qutrytu dodawana jest bramka zmiany kodowania W.
        Dziala tylko dla dim=3.

    encoding_strategy : str
        Strategia budowy obwodu z kodowaniem W:
        - "append_w": standardowy obwod + lokalne W na koncu (domyslnie)
        - "prepared_w_then_conjugated_entanglers":
          lokalnie przygotowuje W|+> z |00> (StatePreparation),
          a entanglery sa budowane jako (W ⊗ W) CZ (W† ⊗ W†)
    """

    if encoding_strategy not in VALID_ENCODING_STRATEGIES:
        raise ValueError(
            f"Nieznana encoding_strategy: {encoding_strategy!r}. "
            f"Dozwolone: {VALID_ENCODING_STRATEGIES}"
        )

    if graph is None and n is None:
        raise ValueError("Nalezy podac albo `graph`, albo liczbe wierzcholkow `n`.")

    if E_new is not None and dim != 3:
        raise ValueError("Zmiana kodowania (E_new) jest obslugiwana tylko dla dim=3.")

    if graph is None:
        if graph_type == "star":
            edges = [[0, i] for i in range(1, n)]
        elif graph_type == "line":
            edges = [[i, i + 1] for i in range(n - 1)]
        else:
            raise ValueError(
                f"Nieznany typ grafu: {graph_type}. "
                "Dozwolone wartosci to 'star' lub 'line'."
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
    with open(quantum_circuits_path(filename), "rb") as fd:
        return qpy.load(fd)[0]


def _build_circuit_append_w(graph, dim, E_new=None):
    """Tryb 'append_w': F na kazdym qutrycie, CZ na krawedziach, W na koncu."""
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
    """Tryb 'prepared_w_then_conjugated_entanglers':
    1. Lokalne przygotowanie W|+> z |00> (StatePreparation) na kazdym qutrycie
    2. Entanglery budowane jako (W ⊗ W) CZ (W† ⊗ W†) na kazdej krawedzi
    """
    if dim != 3:
        raise ValueError(
            "Strategia 'prepared_w_then_conjugated_entanglers' wymaga dim=3."
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
    raise ValueError(f"Nieobslugiwany wymiar: {dim}")


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
    from encoding_change_unitary import build_encoding_change_unitary

    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W ma wymiar {W.shape}, oczekiwano (4, 4)"

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

    raise ValueError("Podana macierz nie jest unitarna", mtx)
