from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
import numpy as np
from qiskit import qpy
from igraph import Graph
from qiskit.synthesis import TwoQubitWeylDecomposition


def create_ame_circuit(n=None, dim=3, graph_type='star', graph=None,
                       basis=None, E_new=None):
    """
    Tworzy obwód AME z grafu.

    Parametry
    ---------
    E_new : np.ndarray, shape (4,3), optional
        Nowa mapa kodowania qutrytu (izometria C³→C⁴).
        Jeśli podana, bramki Fgate i CZgate zostają przetransformowane
        unitarną macierzą zmiany kodowania W: G' = W G W†  (lub ⊗W dla CZ).
        Działa tylko dla dim=3.
    """

    if graph is None and n is None:
        raise ValueError("Należy podać albo `graph`, albo liczbę wierzchołków `n`.")

    if E_new is not None and dim != 3:
        raise ValueError("Zmiana kodowania (E_new) jest obsługiwana tylko dla dim=3.")

    if graph is None:
        if graph_type == 'star':
            edges = [[0, i] for i in range(1, n)]
        elif graph_type == 'line':
            edges = [[i, i + 1] for i in range(n - 1)]
        else:
            raise ValueError(f"Nieznany typ grafu: {graph_type}. "
                             "Dozwolone wartości to 'star' lub 'line'.")
        graph = Graph(n, edges=edges)

    qc = _create_circuit_from_graph(graph, dim, E_new=E_new)

    if basis is not None:
        # sprawdzenie i utworzenie bramki dwubitowej z macierzy 'basis'
        T = change_basis(basis, dim)
        T_qc = TwoQubitWeylDecomposition(T).circuit()

        for i in range(graph.vcount()):
            qc.append(T_qc, [2 * i, 2 * i + 1])

    return qc, graph


def _create_circuit_from_graph(graph, dim, E_new=None):

    # Qutrits gates
    with open('quantum_circuits/Fgate3.qpy', 'rb') as fd:
        Fgate3 = qpy.load(fd)[0]

    with open('quantum_circuits/CZgate3.qpy', 'rb') as fd:
        CZgate3 = qpy.load(fd)[0]

    # Ququarts gates

    with open('quantum_circuits/Fgate4.qpy', 'rb') as fd:
        Fgate4 = qpy.load(fd)[0]

    with open('quantum_circuits/CZgate4cor.qpy', 'rb') as fd:
        CZgate4 = qpy.load(fd)[0]

    if dim == 3:
        Fgate = Fgate3
        CZgate = CZgate3
    elif dim == 4:
        Fgate = Fgate4
        CZgate = CZgate4
    else:
        raise ValueError(f"Nieobsługiwany wymiar: {dim}")

    # ── Opcjonalna zmiana bazy kodowania (tylko dim=3) ──
    # Przy E_new najpierw aplikujemy W na każdym qutrycie, a następnie
    # używamy przekształconych bramek F i CZ.
    W_gate = None
    if E_new is not None:
        W, W_gate = _build_encoding_change_gate(E_new)
        Fgate, CZgate = _transform_gates_to_new_encoding(Fgate, CZgate, E_new, W=W)

    n = graph.vcount()
    qubitList = [[2 * i, 2 * i + 1] for i in range(n)]
    
    edgeList = []
    for edge in graph.get_edgelist():
        u, v = edge[0], edge[1]
        if u != v:
            edgeList.append([
                qubitList[u][0],
                qubitList[u][1],
                qubitList[v][0],
                qubitList[v][1]
            ])
    
    qc = QuantumCircuit(2 * n)

    if W_gate is not None:
        for pair in qubitList:
            qc.append(W_gate, pair)

    for pair in qubitList:
        qc.append(Fgate, pair)

    for edge in edgeList:
        qc.append(CZgate, edge)
    
    return qc


def _build_encoding_change_gate(E_new):
    """Buduje macierz W i odpowiadającą jej 2-qubitową bramkę zmiany kodowania."""
    from encoding_change_unitary import build_encoding_change_unitary

    W = build_encoding_change_unitary(E_new)
    assert W.shape == (4, 4), f"W ma wymiar {W.shape}, oczekiwano (4, 4)"
    return W, UnitaryGate(W, label='W_enc')


def _transform_gates_to_new_encoding(Fgate, CZgate, E_new, W=None):
    """
    Transformuje bramki Fgate (4×4) i CZgate (16×16) do nowej bazy kodowania.

    Nowe bramki obliczane przez sprzężenie unitarne macierzą zmiany kodowania W:
        Fgate_new  = W  @ Fgate  @ W†
        CZgate_new = (W⊗W) @ CZgate @ (W⊗W)†
    """
    from encoding_change_unitary import build_encoding_change_unitary

    if W is None:
        W = build_encoding_change_unitary(E_new)
    Wdag = W.conj().T

    # ── Fgate: 4×4 (1 qutryt = 2 qubity) ──
    Fgate_mtx = Operator(Fgate).data
    assert Fgate_mtx.shape == (4, 4), f"Fgate ma wymiar {Fgate_mtx.shape}, oczekiwano (4, 4)"

    Fgate_new_mtx = W @ Fgate_mtx @ Wdag
    assert Fgate_new_mtx.shape == (4, 4)

    # ── CZgate: 16×16 (2 qutryty = 4 qubity) ──
    CZgate_mtx = Operator(CZgate).data
    assert CZgate_mtx.shape == (16, 16), f"CZgate ma wymiar {CZgate_mtx.shape}, oczekiwano (16, 16)"

    WW = np.kron(W, W)
    WWdag = np.kron(Wdag, Wdag)
    CZgate_new_mtx = WW @ CZgate_mtx @ WWdag
    assert CZgate_new_mtx.shape == (16, 16)

    # Zamień macierze z powrotem na bramki Qiskit
    Fgate_new = UnitaryGate(Fgate_new_mtx, label='Fgate_enc')
    CZgate_new = UnitaryGate(CZgate_new_mtx, label='CZgate_enc')

    return Fgate_new, CZgate_new

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

    else:
        raise ValueError("Podana macierz nie jest unitarna", mtx)