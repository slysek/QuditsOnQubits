from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit import qpy
from igraph import Graph
from qiskit.synthesis import TwoQubitWeylDecomposition


def create_ame_circuit(n=None, dim=3, graph_type='star', graph=None, basis=None):

    if graph is None and n is None:
        raise ValueError("Należy podać albo `graph`, albo liczbę wierzchołków `n`.")

    if graph is None:
        if graph_type == 'star':
            edges = [[0, i] for i in range(1, n)]
        elif graph_type == 'line':
            edges = [[i, i + 1] for i in range(n - 1)]
        else:
            raise ValueError(f"Nieznany typ grafu: {graph_type}. "
                             "Dozwolone wartości to 'star' lub 'line'.")
        graph = Graph(n, edges=edges)

    qc = _create_circuit_from_graph(graph, dim)

    if basis is not None:
        # sprawdzenie i utworzenie bramki dwubitowej z macierzy 'basis'
        T = change_basis(basis, dim)
        T_qc = TwoQubitWeylDecomposition(T).circuit()

        for i in range(graph.vcount()):
            qc.append(T_qc, [2 * i, 2 * i + 1])

    return qc, graph


def _create_circuit_from_graph(graph, dim):

    # Qutrits gates
    with open('Fgate3.qpy', 'rb') as fd:
        Fgate3 = qpy.load(fd)[0]

    with open('CZgate3.qpy', 'rb') as fd:
        CZgate3 = qpy.load(fd)[0]

    # Ququatrs gates

    with open('Fgate4.qpy', 'rb') as fd:
        Fgate4 = qpy.load(fd)[0]

    with open('CZgate4cor.qpy', 'rb') as fd:
        CZgate4 = qpy.load(fd)[0]

    if dim == 3:
        Fgate = Fgate3
        CZgate = CZgate3
    elif dim == 4:
        Fgate = Fgate4
        CZgate = CZgate4
    else:
        raise ValueError(f"Nieobsługiwany wymiar: {dim}")
    
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

    for pair in qubitList:
        qc.append(Fgate, pair)

    for edge in edgeList:
        qc.append(CZgate, edge)
    
    return qc

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