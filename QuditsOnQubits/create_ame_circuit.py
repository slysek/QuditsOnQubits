from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit import qpy
from igraph import Graph
from qiskit.synthesis import TwoQubitWeylDecomposition

from QuditsOnQubits.project_paths import quantum_circuits_path


def create_ame_circuit(n=None, dim=3, graph_type="star", graph=None,
                       basis=None, E_new=None):
    """
    Tworzy obwod AME z grafu.

    Parametry
    ---------
    E_new : np.ndarray, shape (4,3), optional
        Nowa mapa kodowania qutrytu (izometria C^3 -> C^4).
        Jesli podana, obwod dostaje poczatkowa warstwe W na kazdym qutrycie,
        a nastepnie kazda bramka Fgate i CZgate jest realizowana jawnie jako
        blok zlozony z W, bramki bazowej i Wdag.
        Dziala tylko dla dim=3.
    """

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

    qc = _create_circuit_from_graph(graph, dim, E_new=E_new)

    if basis is not None:
        T = change_basis(basis, dim)
        T_qc = TwoQubitWeylDecomposition(T).circuit()

        for i in range(graph.vcount()):
            qc.append(T_qc, [2 * i, 2 * i + 1])

    return qc, graph


def _create_circuit_from_graph(graph, dim, E_new=None):
    with open(quantum_circuits_path("Fgate3.qpy"), "rb") as fd:
        Fgate3 = qpy.load(fd)[0]

    with open(quantum_circuits_path("CZgate3.qpy"), "rb") as fd:
        CZgate3 = qpy.load(fd)[0]

    with open(quantum_circuits_path("Fgate4.qpy"), "rb") as fd:
        Fgate4 = qpy.load(fd)[0]

    with open(quantum_circuits_path("CZgate4cor.qpy"), "rb") as fd:
        CZgate4 = qpy.load(fd)[0]

    if dim == 3:
        Fgate = Fgate3
        CZgate = CZgate3
    elif dim == 4:
        Fgate = Fgate4
        CZgate = CZgate4
    else:
        raise ValueError(f"Nieobslugiwany wymiar: {dim}")

    W_qc = None
    Wdag_qc = None
    if E_new is not None:
        _, W_qc, Wdag_qc = _build_encoding_change_circuits(E_new)

    n = graph.vcount()
    qubit_list = [[2 * i, 2 * i + 1] for i in range(n)]

    edge_list = []
    for u, v in graph.get_edgelist():
        if u != v:
            edge_list.append([
                qubit_list[u][0],
                qubit_list[u][1],
                qubit_list[v][0],
                qubit_list[v][1],
            ])

    qc = QuantumCircuit(2 * n)

    if W_qc is not None:
        for pair in qubit_list:
            qc.append(W_qc, pair)

    for pair in qubit_list:
        if W_qc is None:
            qc.append(Fgate, pair)
        else:
            _append_encoded_fgate(qc, pair, Fgate, W_qc, Wdag_qc)

    for edge in edge_list:
        if W_qc is None:
            qc.append(CZgate, edge)
        else:
            _append_encoded_czgate(qc, edge, CZgate, W_qc, Wdag_qc)

    return qc


def _build_encoding_change_circuits(E_new):
    """Build W together with decomposed circuit blocks for W and Wdag."""
    from encoding_change_unitary import build_encoding_change_unitary

    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W ma wymiar {W.shape}, oczekiwano (4, 4)"

    W_qc = TwoQubitWeylDecomposition(W).circuit()
    Wdag_qc = TwoQubitWeylDecomposition(W.conj().T).circuit()

    W_qc.name = "W"
    Wdag_qc.name = "W†"
    return W, W_qc, Wdag_qc


def _append_encoded_fgate(qc, pair, Fgate, W_qc, Wdag_qc):
    """Append the explicit W -> Fgate -> Wdag block for one encoded qutrit."""
    qc.append(W_qc, pair)
    qc.append(Fgate, pair)
    qc.append(Wdag_qc, pair)


def _append_encoded_czgate(qc, edge, CZgate, W_qc, Wdag_qc):
    """Append the explicit encoded CZ block across two encoded qutrits."""
    left_pair = edge[:2]
    right_pair = edge[2:]

    qc.append(W_qc, left_pair)
    qc.append(W_qc, right_pair)
    qc.append(CZgate, edge)
    qc.append(Wdag_qc, left_pair)
    qc.append(Wdag_qc, right_pair)


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
