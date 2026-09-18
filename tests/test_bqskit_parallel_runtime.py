"""Exercise real BQSKit startup ports without a costly CZ3 search."""
import multiprocessing
import traceback

import numpy as np
import pytest


def _runtime_worker(barrier, results, qubit):
    from bqskit.ir import Circuit
    from bqskit.ir.gates import XGate
    from bqskit.passes import NOOPPass
    from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import _create_bqskit_compiler

    try:
        with _create_bqskit_compiler() as compiler:
            # Two private runtimes must be live at the same time.
            barrier.wait(timeout=60)
            circuit = Circuit(2)
            circuit.append_gate(XGate(), qubit)
            result = compiler.compile(circuit, [NOOPPass()])
            results.put((qubit, np.asarray(result.get_unitary()), ""))
    except BaseException:
        results.put((qubit, None, traceback.format_exc()))


def test_two_real_bqskit_runtimes_can_coexist():
    pytest.importorskip("bqskit")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [ctx.Process(target=_runtime_worker, args=(barrier, results, q)) for q in (0, 1)]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            qubit, unitary, error = results.get(timeout=100)
            assert not error, error
            x = np.array([[0, 1], [1, 0]])
            expected = np.kron(x, np.eye(2)) if qubit == 0 else np.kron(np.eye(2), x)
            np.testing.assert_allclose(unitary, expected)
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        results.close()
