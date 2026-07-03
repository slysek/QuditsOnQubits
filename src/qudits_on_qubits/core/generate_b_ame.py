from pathlib import Path
from qiskit import qpy
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp
from qiskit.circuit import QuantumCircuit
import numpy as np
from igraph import Graph, plot
import matplotlib.pyplot as plt
from .create_ame_circuit import create_ame_circuit
from .project_paths import quantum_circuits_path
from IPython.display import display, Math
from fractions import Fraction

_QC_DIR = Path(quantum_circuits_path())

with open(_QC_DIR / 'Xgate3.qpy', 'rb') as fd:
    Xgate3 = qpy.load(fd)[0]

with open(_QC_DIR / 'Zgate3.qpy', 'rb') as fd:
    Zgate3 = qpy.load(fd)[0]

with open(_QC_DIR / 'Xgate4.qpy', 'rb') as fd:
    Xgate4 = qpy.load(fd)[0]

with open(_QC_DIR / 'Zgate4.qpy', 'rb') as fd:
    Zgate4 = qpy.load(fd)[0]

with open(_QC_DIR / 'Xgate3dag.qpy', 'rb') as fd:
    Xgate3dag = qpy.load(fd)[0]

with open(_QC_DIR / 'Zgate3dag.qpy', 'rb') as fd:
    Zgate3dag = qpy.load(fd)[0]

with open(_QC_DIR / 'Xgate4dag.qpy', 'rb') as fd:
    Xgate4dag = qpy.load(fd)[0]

with open(_QC_DIR / 'Zgate4dag.qpy', 'rb') as fd:
    Zgate4dag = qpy.load(fd)[0]

class MemoryLimitExceeded(Exception):
    """Exception raised when circuit requires too much memory for computation"""
    pass

class generate_b_ame:
    """
    Class for generating and computing the B_AME (Bell-AME) indicator for quantum states.

    B_AME is a measure of multipartite entanglement based on graph stabilizers.
    """

    def __init__(self, dim=None, graph=None, circuit=None, thval=True):
        """
        Initializes the generate_b_ame object.

        Args:
            dim (int, optional): Local dimension of the quantum system (3 or 4)
            graph (igraph.Graph, optional): Graph defining the entanglement structure
            circuit (QuantumCircuit, optional): Ready quantum circuit for analysis
            thval (bool): Whether to calculate theoretical values (True) or use provided circuit
        """
        self.dim = dim
        self.graph = graph
        self.circuit = circuit
        self.qc_stab_list = None
        self.exp_val = None
        self.bame_val = None
        self.stab_joined_cleaned = None
        self.bame_str = ''

        if dim is not None and graph is not None and thval:
            self._calculate(thval)

    def _calculate(self, thval=True):
        """
        Main computational method - generates stabilizers and computes B_AME.

        Process:
        1. Generates all types of stabilizers (G1, G2, G3)
        2. Creates quantum circuits for each stabilizer
        3. Computes expectation values
        4. Sums results with appropriate coefficients

        Args:
            thval (bool): Whether to use theoretical AME circuit or provided circuit
        """
        g1n = G1n(self.graph, self.dim)
        g0 = g1n[0]
        g1list = g1n[1:]
        g2list = G2n(self.graph, self.dim)
        g3list = G3n(self.graph, self.dim)

        n1, n2 = find_n1_n2(self.graph)

        k1_list = [k for k in range(1, self.dim)]
        k2_list = n1_ng(self.graph, n1, n2)
        k3_list = n1_notconnected(self.graph, n1)


        ck_list, j_list = find_ck(self.graph, self.dim)

        g0_list = {}
        g1g2k_list = {}
        g1gik_list = {}
        gk_list = {}

        g0.insert(0, 1)
        g0_list[f'(G1)'] = g0

        for k in k1_list:
            g1list[k - 1].insert(0, ck_list[k - 1][0])
            g1g2k_list[f'(G1G2^{k})'] = g1list[k - 1]

        for j in j_list.keys():
            for i, k2 in zip(range(len(j_list[j])), k2_list):
                g2list[i][j - 1].insert(0, ck_list[j - 1][i + 1])
                g1gik_list[f'(G1G{k2})'] = g2list[i][j - 1]


        for k, i in zip(k3_list, range(len(k3_list))):
            gk_list[f'(G{k})'] = g3list[i][0]
            g3list[i][0].insert(0, 1)

        g0_list_n = {}
        g1g2k_list_n = {}
        g1gik_list_n = {}
        gk_list_n = {}

        for n in range(1, self.dim):
            g0_list_n[f'(G1)^{n}'] = np.strings.multiply(g0[1:], n).astype(object).tolist()
            g0_list_n[f'(G1)^{n}'].insert(0, g0[0])

        for i in g1g2k_list.keys():
            for n in range(1, self.dim):
                g1g2k_list_n[f'{i}^{n}'] = np.strings.multiply(g1g2k_list[i][1:], n).astype(object).tolist()
                g1g2k_list_n[f'{i}^{n}'].insert(0, g1g2k_list[i][0])

        for i in g1gik_list.keys():
            for n in range(1, self.dim):
                g1gik_list_n[f'{i}^{n}'] = np.strings.multiply(g1gik_list[i][1:], n).astype(object).tolist()
                g1gik_list_n[f'{i}^{n}'].insert(0, g1gik_list[i][0])

        for i in gk_list.keys():
            for n in range(1, self.dim):
                gk_list_n[f'{i}^{n}'] = np.strings.multiply(gk_list[i][1:], n).astype(object).tolist()
                gk_list_n[f'{i}^{n}'].insert(0, gk_list[i][0])

        stab_joined = {**g0_list_n, **g1g2k_list_n, **g1gik_list_n, **gk_list_n}

        self.bame_str = ''

        for i in stab_joined.keys():
            if i != list(stab_joined.keys())[-1]:
                self.bame_str += f'{stab_joined[i][0]} * {i} + '
            else:
                self.bame_str += f'{stab_joined[i][0]} * {i}'

        self.stab_joined_cleaned = {}

        for op in stab_joined.keys():
            self.stab_joined_cleaned[op] = clean_id([stab_joined[op][1:]])[0]
            self.stab_joined_cleaned[op].insert(0, stab_joined[op][0])


        self.qc_stab_list = [create_stab_from_list(stab_list[1:], self.dim) for stab_list in self.stab_joined_cleaned.values()]

        if self.dim == 3:
            keys = list(self.stab_joined_cleaned.keys())

            for i in range(len(keys[::2])):
                self.qc_stab_list[2 * i + 1] = create_stab_from_list(self.stab_joined_cleaned[keys[2*i]][1:], self.dim, dag=True)


        if self.circuit is None:
            ame_qc, _ = create_ame_circuit(dim=self.dim, graph=self.graph)
            try:
                if thval:
                    self.exp_val = th_values(ame_qc, self.qc_stab_list)
                else:
                    self.exp_val = th_values(self.circuit, self.qc_stab_list)
            except MemoryLimitExceeded as e:
                print(f"Error: {e}")
                self.exp_val = []
                return

        else:
            self.exp_val = th_values(self.circuit, self.qc_stab_list)

        self.ck_final = []
        for stab_list in self.stab_joined_cleaned.values():
            self.ck_final.append(stab_list[0])

        self.bame_val = sum([ck*ev for ck, ev in zip(self.ck_final, self.exp_val)])

        if thval:
            print(f'B_ame = {self.bame_val}')

    def show_eq(self, method=None):
        """
        Displays the B_AME equation in different formats.

        Args:
            method (str, optional): Display format:
                - 'latex': LaTeX format with mathematical rendering
                - None: Plain text
        """
        if method == 'latex':
            ltx = create_latex(self.stab_joined_cleaned)
            display(Math(ltx))
        else:
            print(f'B_AME = {self.bame_str}')

    def get_dict(self):
        """
        Returns dictionary containing all stabilizers with coefficients.

        Returns:
            dict: Dictionary {stabilizer_name: [coefficient, operator_list]}
        """
        return self.stab_joined_cleaned

    def get_value(self):
        """
        Returns the computed B_AME value.

        Returns:
            float: B_AME indicator value
        """
        return self.bame_val

    def get_stabilizers(self):
        """
        Returns list of quantum circuits representing stabilizers.

        Returns:
            list[QuantumCircuit]: List of stabilizer circuits
        """
        return self.qc_stab_list

    def get_expectation_values(self):
        """
        Returns list of expectation values for all stabilizers.

        Returns:
            list[float]: List of expectation values
        """
        return self.exp_val

    def bame_jobresult(self, result_list):
        """
        Computes B_AME based on results from real quantum device.

        Args:
            result_list (list[float]): List of expectation values from experiment

        Returns:
            float: B_AME value computed from experimental results
        """
        bamesum = 0
        for ck, ev in zip(self.ck_final, result_list):
            bamesum += ck * ev
        return bamesum

    def get_n1n2(self):
        """
        Returns the two nodes with highest degree in the graph.

        Returns:
            tuple[int, int]: Indices of nodes (n1, n2) with highest degrees
        """
        return find_n1_n2(self.graph)
def G0_tylda(graph):
    n1, n2 = find_n1_n2(graph)


def create_latex(op_dict):
    """
    Creates LaTeX representation of the B_AME equation.

    Converts stabilizer dictionary to formatted mathematical equation
    with proper subscripts, powers, and fractions.

    Args:
        op_dict (dict): Dictionary of stabilizers with coefficients

    Returns:
        str: String with LaTeX code representing the B_AME equation
    """
    n_list = [n[-1] for n in op_dict.keys()]
    op_clean = [n[1:-3] for n in op_dict.keys()]
    ck_list = [ck[0] for ck in op_dict.values()]

    latexstr = ''

    for op, k in zip(op_clean, range(len(op_clean))):
        str = ''
        if '^' in op:
            for i in range(len(op[:-2])):
                if i % 2 == 1:
                    str += '_' + f'{{{op[i]}}}'
                else:
                    str += op[i]
            if int(op[-1]) != 1:
                str += f'^{{{op[-1]}}}'
        else:
            for i in range(len(op)):
                if i % 2 == 1:
                    str += '_' + f'{{{op[i]}}}'
                else:
                    str += op[i]


        if n_list[k] != '1':
            str = '(' + str + f')^{{{n_list[k]}}}'

        if ck_list[k] != 1:
            latexstr += f'\\frac{{{Fraction(float(ck_list[k])).numerator}}}{{{Fraction(float(ck_list[k])).denominator}}}' + str
        else:
            latexstr += str
        if k != len(op_clean) - 1:
            latexstr += ' + '

    return 'B_{ame} = ' + latexstr
def th_values(main_qc, qc_list):
    """
    Computes theoretical expectation values of stabilizers for given state.

    First checks if circuits are not too large (memory limit), then
    creates operators and computes expectation values.

    Args:
        main_qc (QuantumCircuit): Main quantum circuit (AME state)
        qc_list (list[QuantumCircuit]): List of stabilizer circuits

    Returns:
        list[float]: List of expectation values for each stabilizer

    Raises:
        MemoryLimitExceeded: When circuit requires too much memory
    """
    # Check size of each circuit before creating operator
    for qc_stab in qc_list:
        if hasattr(qc_stab, 'num_qubits'):
            num_qubits = qc_stab.num_qubits
            if num_qubits > 18:  # 2^18 = 262144 - reasonable limit
                raise MemoryLimitExceeded(
                    f"Circuit has {num_qubits} qubits. This requires {2**num_qubits}x{2**num_qubits} "
                    f"matrix ({(2**(2*num_qubits) * 16) / (1024**4):.2f} TB memory). "
                    f"Too much for regular computer."
                )

    try:
        operators = [Operator(qc_stab) for qc_stab in qc_list]
    except MemoryError:
        max_qubits = max(qc.num_qubits for qc in qc_list if hasattr(qc, 'num_qubits'))
        raise MemoryLimitExceeded(
            f"Cannot create operator for circuit with {max_qubits} qubits. "
            f"Too much memory needed for computation on regular computer."
        )

    exp_val = []

    for op in operators:
        exp_val.append(Statevector(main_qc).expectation_value(op).round(4))

    return exp_val

def find_n1_n2(graph):
    """
    Finds two nodes with highest degree in the graph.

    Algorithm searches for node n1 with maximum degree, then among its
    neighbors searches for node n2 with highest degree.

    Args:
        graph (igraph.Graph): Input graph

    Returns:
        tuple[int, int]: Indices of nodes (n1, n2) with highest degrees
    """
    n1 = graph.vs[0]
    n2 = graph.vs[1]

    for i in range(graph.vcount()):
        if graph.neighborhood_size(i) - 1 > len(n1.neighbors()):
            n1 = graph.vs[i]
            ng = graph.neighbors(n1.index)
            n2 = graph.vs[ng[0]]
            for j in ng:
                if graph.neighborhood_size(j) - 1 > len(n2.neighbors()) and j != n1.index:
                    n2 = graph.vs[j]
    return n1.index + 1, n2.index + 1

def n1_ng(graph, n1, n2):
    """
    Returns neighbors of node n1 excluding node n2.

    Args:
        graph (igraph.Graph): Input graph
        n1 (int): Index of first node (1-indexed)
        n2 (int): Index of second node to exclude (1-indexed)

    Returns:
        numpy.ndarray: Array of neighbor indices of n1 without n2 (1-indexed)
    """
    ng = graph.neighbors(n1 - 1)
    ng = np.add(ng, 1)

    idxs = np.where(ng == n2)
    ng = np.delete(ng, idxs)
    return ng

def n1_notconnected(graph, n1):
    """
    Returns all nodes not directly connected to node n1.

    Args:
        graph (igraph.Graph): Input graph
        n1 (int): Index of reference node (1-indexed)

    Returns:
        numpy.ndarray: Array of node indices not connected to n1 (1-indexed)
    """
    n1_notconnected = [i for i in range(graph.vcount())]

    for j in range(len(n1_notconnected)):
        if j in graph.neighbors(n1 - 1) or j == n1 - 1:
            n1_notconnected.remove(j)

    n1_notconnected = np.add(n1_notconnected, 1)

    return n1_notconnected

def is_repetition_of(s: str, char: str) -> bool:
    """
    Checks if string consists only of repetitions of one character.

    Args:
        s (str): String to check
        char (str): Character to compare

    Returns:
        bool: True if string is repetitions of char, False otherwise
    """
    return bool(s) and s == char * len(s)

def clean_id(op_list):
    """
    Removes identity operators (I) from Pauli operator strings.

    Cleans string representation of operators by removing redundant 'I'.
    If entire string is all 'I', leaves one 'I'.

    Args:
        op_list (list[list[str]]): List of lists of operator strings

    Returns:
        list[list[str]]: Cleaned list of operators without redundant 'I'
    """
    cleaned_op_list = []
    for i in op_list:
        cleaned = []
        for j in i:
            if not is_repetition_of(j, 'I'):
                j = j.replace('I', "")
                cleaned.append(j)
            else:
                cleaned.append('I')
        cleaned_op_list.append(cleaned)
    return cleaned_op_list

def G1op(graph, n1, n2):
    """
    Generates G1 operator - basic stabilizer generator.

    Creates operator of form X_{n1} ⊗ Z_{n2}^{r12} ⊗ ∏Z_{i}^{r1i},
    where r_ij is the number of edges between nodes i,j.

    Args:
        graph (igraph.Graph): Graph defining structure
        n1 (int): Index of first node (0-indexed)
        n2 (int): Index of second node (0-indexed)

    Returns:
        numpy.ndarray: Array of strings representing operator on each qubit
    """
    opstr = np.full(graph.vcount(), 'I', dtype='U10')
    opstr[n1] = 'X'
    num_edges = len(graph.es.select(_between=([n1], [n2])))
    opstr[n2] = 'Z' * num_edges

    n1_without_n2 = n1_ng(graph, n1 + 1, n2 + 1)

    for i in n1_without_n2:
        num_edges2 = len(graph.es.select(_between=([n1], [i - 1])))
        opstr[i - 1] = 'Z' * num_edges2

    return opstr

def G2op(graph, n1, n2):
    """
    Generates G2 operator - second stabilizer generator.

    Creates operator of form X_{n2} ⊗ Z_{n1}^{r12} ⊗ ∏Z_{i}^{r2i},
    where connections of n2 with other nodes are considered.

    Args:
        graph (igraph.Graph): Graph defining structure
        n1 (int): Index of first node (0-indexed)
        n2 (int): Index of second node (0-indexed)

    Returns:
        numpy.ndarray: Array of strings representing operator on each qubit
    """
    opstr = np.full(graph.vcount(), 'I', dtype='U10')
    num_edges = len(graph.es.select(_between=([n1], [n2])))
    opstr[n2] = 'X'
    opstr[n1] = 'Z' * num_edges
    n1_without_n2 = n1_ng(graph, n1 + 1, n2 + 1)

    for i in n1_without_n2:
        num_edges2 = len(graph.es.select(_between=([n2], [i - 1])))
        if num_edges2 > 0:
            opstr[i - 1] = 'Z' * num_edges2

    n1_nc = n1_notconnected(graph, n1 + 1)

    for i in n1_nc:
        num_edges3 = len(graph.es.select(_between=([n2], [i - 1])))
        if num_edges3 > 0:
            opstr[i - 1] = 'Z' * num_edges3

    return opstr

def G1n(graph, dim):
    """
    Generates all powers of G1 operators and G1*G2^k combinations.

    Creates complete set of first type generators for given dimension.

    Args:
        graph (igraph.Graph): Graph defining structure
        dim (int): Local dimension of system

    Returns:
        list[list[str]]: List of all G1 type operators with their powers
    """
    n1, n2 = find_n1_n2(graph)

    g1str = G1op(graph, n1 - 1, n2 - 1)
    g2str = G2op(graph, n1 - 1, n2 - 1)
    g1list = []
    n = [i for i in range(1, dim)]
    k = [i for i in range(dim)]
    for i in n:
        for j in k:
            g1 = np.strings.multiply(g1str + np.strings.multiply(g2str, j), i)
            g1list.append(g1)
    cleaned_list = clean_id(g1list)
    return cleaned_list

def Gi1(graph, n1, n2):
    """
    Generates G_i operators for nodes neighboring n1 (excluding n2).

    For each neighbor i of node n1 creates operator X_i with appropriate Z on
    nodes connected to i.

    Args:
        graph (igraph.Graph): Graph defining structure
        n1 (int): Index of first node (0-indexed)
        n2 (int): Index of second node (0-indexed)

    Returns:
        list[numpy.ndarray]: List of operators for each neighbor of n1
    """
    n1_without_n2 = n1_ng(graph, n1 + 1, n2 + 1)
    n1_without_n2 = np.add(n1_without_n2, -1)
    opstr_list = []

    for i in n1_without_n2:

        opstr = np.full(graph.vcount(), 'I', dtype='U10')
        num_edges = len(graph.es.select(_between=([n1], [i])))
        opstr[i] = 'X'
        opstr[n1] = 'Z' * num_edges
        num_edges2 = len(graph.es.select(_between=([n2], [i])))
        if num_edges2 > 0:
            opstr[n2.index] = 'Z' * num_edges2

        ni = n1_without_n2.copy().tolist()
        ni.remove(graph.vs[i].index)
        for j in ni:
            num_edges3 = len(graph.es.select(_between=([graph.vs[i].index], [j])))
            if num_edges3 > 0:
                opstr[j] = 'Z' * num_edges3

        n1_nc = np.add(n1_notconnected(graph, n1 + 1), -1)

        for k in n1_nc:
            num_edges4 = len(graph.es.select(_between=([graph.vs[i].index], [k])))
            if num_edges4 > 0:
                opstr[k] = 'Z' * num_edges4
        opstr_list.append(opstr)

    return opstr_list

def Gi2(graph, n1, n2):
    """
    Generates G_i operators for nodes not connected to n1.

    For each node not connected to n1 creates operator X_i with appropriate
    Z operators on nodes connected to it.

    Args:
        graph (igraph.Graph): Graph defining structure
        n1 (int): Index of first node (0-indexed)
        n2 (int): Index of second node (0-indexed)

    Returns:
        list[numpy.ndarray]: List of operators for nodes not connected to n1
    """
    opstr_list = []
    n1_nc = np.add(n1_notconnected(graph, n1 + 1), -1)
    n1_without_n2 = np.add(n1_ng(graph, n1 + 1, n2 + 1), -1)
    for i in n1_nc:
        opstr = np.full(graph.vcount(), 'I', dtype='U10')
        opstr[i] = 'X'
        num_edges = len(graph.es.select(_between=([n2], [i])))

        if num_edges > 0:
            opstr[n2] = 'Z' * num_edges

        for k in n1_without_n2:
            num_edges3 = len(graph.es.select(_between=([graph.vs[i].index], [k])))
            if num_edges3 > 0:
                opstr[k] = 'Z' * num_edges3

        ni = n1_nc.copy().tolist()
        ni.remove(i)
        for j in ni:
            num_edges2 = len(graph.es.select(_between=([graph.vs[i].index], [j])))
            if num_edges2 > 0:
                opstr[j] = 'Z' * num_edges2
        opstr_list.append(opstr)

    return opstr_list

def G2n(graph, dim):
    """
    Generates G1*G_i type operators for neighbors of n1.

    Combines basic G1 operator with G_i operators for each
    neighbor of node n1.

    Args:
        graph (igraph.Graph): Graph defining structure
        dim (int): Local dimension of system

    Returns:
        list[list]: List of G1*G_i type operators
    """
    n1, n2 = find_n1_n2(graph)

    g1str = G1op(graph, n1 - 1, n2 - 1)
    gistr = Gi1(graph, n1 - 1, n2 - 1)
    templist2 = []
    n = [i for i in range(1, dim)]

    for gi in gistr:
        templist = []
        g1 = np.strings.multiply(g1str + gi, 1)
        templist.append(g1)
        cleaned_list = clean_id(templist)
        templist2.append(cleaned_list)

    return templist2

def G3n(graph, dim):
    """
    Generates all powers of G_i operators for nodes not connected to n1.

    Creates powers of operators G_i^k for k=1,2,...,dim-1 for each
    node not connected to n1.

    Args:
        graph (igraph.Graph): Graph defining structure
        dim (int): Local dimension of system

    Returns:
        list[list]: List of all powers of G_i operators
    """
    n1, n2 = find_n1_n2(graph)

    n = [i for i in range(1, dim)]
    gi2str = Gi2(graph, n1 - 1, n2 - 1)
    op_list = []

    for i in gi2str:
        g_list = []
        for k in n:
            gtemp = np.strings.multiply(i, k)
            g_list.append(gtemp)
        cleaned = clean_id(g_list)
        op_list.append(cleaned)
    if not op_list:
        return op_list
    else:
        return op_list

def is_j(graph, dim):
    """
    Identifies nodes satisfying edge relation condition r_{1j} = k*r_{12}.

    Finds nodes j such that number of edges between n1 and j is multiple
    of number of edges between n1 and n2.

    Args:
        graph (igraph.Graph): Graph defining structure
        dim (int): Local dimension of system

    Returns:
        dict: Dictionary {k: [list_of_nodes]} satisfying condition for each k
    """
    n1, n2 = find_n1_n2(graph)
    n1_without_n2 = n1_ng(graph, n1, n2)
    j_list_all = {}
    k_list = [k for k in range(1, dim)]

    for k in k_list:
        j_list = []
        for j in n1_without_n2:
            r1j = len(graph.es.select(_between=([graph.vs[j - 1].index], [n1 - 1])))
            r12 = len(graph.es.select(_between=([n2 - 1], [n1 - 1])))
            if r1j == k * r12:
                j_list.append(j)
        j_list_all[k] = j_list

    return j_list_all

def find_ck(graph, dim):
    """
    Computes coefficients c_k for operator normalization in B_AME equation.

    Solves system of equations to ensure proper weights for each
    operator type in B_AME sum.

    Args:
        graph (igraph.Graph): Graph defining structure
        dim (int): Local dimension of system

    Returns:
        tuple: (coefficient_list, j_nodes_dictionary)
    """
    n1, n2 = find_n1_n2(graph)
    k = [k for k in range(1, dim)]
    k2 = n1_ng(graph, n1, n2)

    c1_list = np.full(len(k), 1).reshape(1, -1)
    c2_list = []
    j_list = is_j(graph, dim)


    for i in range(1, dim):
        k_list = np.full(len(k2), 0)
        for j in j_list[i]:
            k_list[j - k2[0]] = 1
        c2_list.append(k_list)

    c2_list = np.array(c2_list)

    A = np.column_stack((c1_list.transpose(), c2_list))

    c_list = []

    for i in A:
        x = i / np.sum(i)
        c_list.append(x)
    c_list = np.array(c_list)

    return c_list, j_list

def create_stab_from_str(str, dim, dag=False):
    """
    Creates quantum circuit of stabilizer from Pauli operator string.

    Converts string representation (e.g. "XZ") to quantum circuit
    using appropriate gates for given dimension.

    Args:
        str (str): String of Pauli operators (X, Z)
        dim (int): Local dimension (3 or 4)
        dag (bool): Whether to use conjugate gates (†)

    Returns:
        QuantumCircuit: Quantum circuit representing stabilizer
    """
    qc = QuantumCircuit(2)
    if not dag:
        if dim == 3:
            Xgate = Xgate3
            Zgate = Zgate3
        elif dim == 4:
            Xgate = Xgate4
            Zgate = Zgate4
        qc.name = str
    else:
        if dim == 3:
            Xgate = Xgate3dag
            Zgate = Zgate3dag
        elif dim == 4:
            Xgate = Xgate4dag
            Zgate = Zgate4dag
        str = str[::-1]
        qc.name = "dag".join(str + " ")

    for i in str:
        if i == 'X':
            qc.append(Xgate, [0, 1])
        elif i == 'Z':
            qc.append(Zgate, [0, 1])

    return qc

def create_stab_from_list(list, dim, dag=False):
    """
    Creates quantum circuit for list of stabilizers acting on different qubits.

    Combines multiple stabilizers into one circuit, where each stabilizer
    acts on dedicated pair of qubits.

    Args:
        list (list[str]): List of operator strings for each qubit
        dim (int): Local dimension (3 or 4)
        dag (bool): Whether to use conjugate gates (†)

    Returns:
        QuantumCircuit: Combined quantum circuit of all stabilizers
    """
    n = len(list)
    qc = QuantumCircuit(2 * n)

    for i, k in zip(list, range(n)):
        stab = create_stab_from_str(i, dim, dag)
        qc.append(stab, [2 * k, 2 * k + 1])
    return qc
