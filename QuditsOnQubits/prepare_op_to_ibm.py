from qiskit.quantum_info import Operator, SparsePauliOp

def prepare_op_to_ibm(base_qc, qc_list):
    operators = [Operator(qc_stab) for qc_stab in qc_list]

    isa_A_list = []

    for op in operators:
        A_op = (0.5 * (op + op.adjoint()))

        A = SparsePauliOp.from_operator(A_op)

        isa_A_list.append(A.apply_layout(layout=base_qc.layout))

    return isa_A_list