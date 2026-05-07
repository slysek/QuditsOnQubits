"""Tests for find_best_diagonal_decomposition.

The script under test lives at the repo root (`QuditsOnQubits/`), not inside
the QuditsOnQubits package, so we import it as a top-level module. Pytest's
default rootdir-based sys.path injection makes this work when run from
`QuditsOnQubits/`.
"""

import numpy as np
import pytest

import find_best_diagonal_decomposition as fbdd


class TestUnitariesEqualUpToGlobalPhase:
    def test_identity_matches_itself(self):
        I = np.eye(4, dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(I, I)

    def test_global_phase_factor_matches(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        phase = np.exp(1j * np.pi / 7)
        assert fbdd.unitaries_equal_up_to_global_phase(A, phase * A)

    def test_negated_diagonal_matches_up_to_phase(self):
        D = np.diag([1.0, 1j, -1.0, -1j])
        assert fbdd.unitaries_equal_up_to_global_phase(D, -D)

    def test_different_unitaries_do_not_match(self):
        I = np.eye(4, dtype=complex)
        Z_diag = np.diag([1.0, -1.0, 1.0, 1.0])
        assert not fbdd.unitaries_equal_up_to_global_phase(I, Z_diag)

    def test_zero_matrices_match(self):
        Z = np.zeros((4, 4), dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(Z, Z)

    def test_zero_vs_nonzero_does_not_match(self):
        I = np.eye(4, dtype=complex)
        Z = np.zeros((4, 4), dtype=complex)
        assert not fbdd.unitaries_equal_up_to_global_phase(I, Z)

    def test_atol_controls_strictness(self):
        I = np.eye(4, dtype=complex)
        perturbed = I + 1e-10 * np.ones((4, 4), dtype=complex)
        assert fbdd.unitaries_equal_up_to_global_phase(I, perturbed, atol=1e-8)
        assert not fbdd.unitaries_equal_up_to_global_phase(I, perturbed, atol=1e-12)


class TestValidateDiagonal:
    def test_accepts_canonical_6_qubit_example(self):
        D = np.array([1, -1, 1j, -1j] * 16, dtype=complex)
        n = fbdd._validate_diagonal(D, atol=1e-8)
        assert n == 6

    def test_rejects_2d_array(self):
        with pytest.raises(ValueError, match="1-D"):
            fbdd._validate_diagonal(np.eye(4, dtype=complex), atol=1e-8)

    def test_rejects_non_power_of_two_length(self):
        with pytest.raises(ValueError, match="power of two"):
            fbdd._validate_diagonal(np.array([1, 1, 1], dtype=complex), atol=1e-8)

    def test_rejects_length_one(self):
        with pytest.raises(ValueError, match="at least 2"):
            fbdd._validate_diagonal(np.array([1.0], dtype=complex), atol=1e-8)

    def test_rejects_non_unit_modulus(self):
        D = np.array([1, 1, 1, 0.5], dtype=complex)
        with pytest.raises(ValueError, match="magnitude"):
            fbdd._validate_diagonal(D, atol=1e-8)

    def test_rejects_non_array(self):
        with pytest.raises(ValueError, match="ndarray"):
            fbdd._validate_diagonal([1, -1, 1, -1], atol=1e-8)


def _bitmask(S):
    m = 0
    for i in S:
        m |= 1 << i
    return m


class TestZPhaseCoefficients:
    def test_constant_diagonal_has_no_nonzero_subsets(self):
        D = np.ones(4, dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        assert result["num_qubits"] == 2
        assert result["coefficients"] == {}
        assert result["num_nonzero"] == 0
        assert result["total_terms"] == 3
        assert result["max_weight"] == 0
        assert result["weight_histogram"] == {}
        assert abs(result["constant"]) < 1e-10

    def test_z0_pattern_has_single_nonzero_coefficient(self):
        # diag(1, -1, 1, -1) on 2 qubits = exp(i pi/2 * Z_0) up to global phase
        D = np.array([1, -1, 1, -1], dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        assert result["num_qubits"] == 2
        assert set(result["coefficients"].keys()) == {frozenset({0})}
        c = result["coefficients"][frozenset({0})]
        assert abs(abs(c) - np.pi / 2) < 1e-10

    def test_z0z1_pattern(self):
        # diag(1, -1, -1, 1) corresponds to exp(i pi/2 * Z_0 Z_1) up to global phase
        D = np.array([1, -1, -1, 1], dtype=complex)
        result = fbdd.z_phase_coefficients_from_diag(D)
        nonzero = result["coefficients"]
        assert set(nonzero.keys()) == {frozenset({0, 1})}

    def test_round_trip_random_sparse(self):
        # Coefficients are bounded so the worst-case sum |sum_S c_S (+/-1)|
        # stays inside (-pi, pi]; otherwise the phases wrap when stored as
        # exp(i theta) and the recovered c_S differ by a 2pi/N multiple.
        rng = np.random.default_rng(0)
        n = 4
        chosen_subsets = [
            frozenset({0, 1}),
            frozenset({2}),
            frozenset({1, 3}),
            frozenset({0, 1, 2, 3}),
        ]
        bound = np.pi / (2 * len(chosen_subsets))
        chosen_coeffs = {S: rng.uniform(-bound, bound) for S in chosen_subsets}

        N = 2**n
        theta = np.zeros(N)
        for k in range(N):
            for S, cS in chosen_coeffs.items():
                sign = 1 - 2 * (bin(_bitmask(S) & k).count("1") % 2)
                theta[k] += cS * sign
        D = np.exp(1j * theta)

        result = fbdd.z_phase_coefficients_from_diag(D)
        recovered = result["coefficients"]
        for S, cS in chosen_coeffs.items():
            assert S in recovered
            assert abs(recovered[S] - cS) < 1e-10
        assert set(recovered.keys()) == set(chosen_subsets)


class TestPermutationHelpers:
    def test_bit_permute_identity_is_identity(self):
        n = 3
        perm = (0, 1, 2)
        for k in range(8):
            assert fbdd._bit_permute(k, perm, n) == k

    def test_bit_permute_swap_first_two(self):
        # perm=(1,0,2): bit at position i of new int comes from position
        # perm[i] of old. k=0b001 (bit 0 set) -> new bit 0 = old bit
        # perm[0]=1 = 0; new bit 1 = old bit perm[1]=0 = 1. So 0b001 -> 0b010.
        n = 3
        perm = (1, 0, 2)
        assert fbdd._bit_permute(0b001, perm, n) == 0b010
        assert fbdd._bit_permute(0b010, perm, n) == 0b001
        assert fbdd._bit_permute(0b101, perm, n) == 0b110

    def test_permute_diagonal_identity_returns_copy(self):
        D = np.array([1, -1, 1j, -1j], dtype=complex)
        out = fbdd._permute_diagonal(D, (0, 1))
        np.testing.assert_array_equal(out, D)

    def test_permute_diagonal_swap_qubits(self):
        # n=2, swap qubits.
        D = np.array([10, 20, 30, 40], dtype=complex)
        out = fbdd._permute_diagonal(D, (1, 0))
        # new[0b00] = D[0b00] = 10
        # new[0b01] = D[bit_permute(0b01,(1,0),2)] = D[0b10] = 30
        # new[0b10] = D[bit_permute(0b10,(1,0),2)] = D[0b01] = 20
        # new[0b11] = D[0b11] = 40
        np.testing.assert_array_equal(
            out, np.array([10, 30, 20, 40], dtype=complex)
        )

    def test_enumerate_permutations_identity_only(self):
        perms = list(
            fbdd._enumerate_permutations(num_qubits=4, try_perms=False, rng_seed=0)
        )
        assert perms == [(0, 1, 2, 3)]

    def test_enumerate_permutations_full_for_n_le_4(self):
        perms = list(
            fbdd._enumerate_permutations(num_qubits=4, try_perms=True, rng_seed=0)
        )
        assert len(perms) == 24
        assert (0, 1, 2, 3) in perms

    def test_enumerate_permutations_sampled_for_n_5_or_6(self):
        perms = list(
            fbdd._enumerate_permutations(num_qubits=6, try_perms=True, rng_seed=0)
        )
        assert (0, 1, 2, 3, 4, 5) in perms
        assert 1 < len(perms) <= 25
        assert all(sorted(p) == [0, 1, 2, 3, 4, 5] for p in perms)

    def test_enumerate_permutations_identity_only_for_n_gt_6(self):
        with pytest.warns(UserWarning, match="permutation"):
            perms = list(
                fbdd._enumerate_permutations(
                    num_qubits=8, try_perms=True, rng_seed=0
                )
            )
        assert perms == [tuple(range(8))]


class TestMetrics:
    def test_count_metrics_simple_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.rz(0.5, 0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.7, 2)
        m = fbdd._count_metrics(qc)
        assert m["total_gates"] == 4
        assert m["two_qubit"] == 2
        assert m["depth"] == qc.depth()
        assert m["breakdown"]["rz"] == 2
        assert m["breakdown"]["cx"] == 2

    def test_count_metrics_empty_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        m = fbdd._count_metrics(qc)
        assert m["total_gates"] == 0
        assert m["two_qubit"] == 0
        assert m["depth"] == 0
        assert m["breakdown"] == {}

    def test_count_metrics_excludes_barriers(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(1, 0)
        m = fbdd._count_metrics(qc)
        assert m["two_qubit"] == 2
        assert m["total_gates"] == 2


class TestScore:
    def test_score_two_qubit_then_depth_returns_tuple(self):
        m = {"two_qubit": 5, "depth": 12, "total_gates": 30}
        assert fbdd._score(m, "two_qubit_then_depth") == (5, 12, 30)

    def test_score_orders_by_two_qubit_first(self):
        a = {"two_qubit": 3, "depth": 100, "total_gates": 100}
        b = {"two_qubit": 4, "depth": 1, "total_gates": 1}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_orders_by_depth_when_tied(self):
        a = {"two_qubit": 3, "depth": 5, "total_gates": 100}
        b = {"two_qubit": 3, "depth": 6, "total_gates": 1}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_orders_by_total_when_tied(self):
        a = {"two_qubit": 3, "depth": 5, "total_gates": 9}
        b = {"two_qubit": 3, "depth": 5, "total_gates": 10}
        assert fbdd._score(a, "two_qubit_then_depth") < fbdd._score(b, "two_qubit_then_depth")

    def test_score_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            fbdd._score({"two_qubit": 1, "depth": 1, "total_gates": 1}, "unknown")


from qiskit.quantum_info import Operator  # noqa: E402  (used by tests below)


class TestDiagonalGateStrategy:
    def test_default_basis_gates_returned_when_neither_supplied(self):
        bg = fbdd._resolve_basis_gates(backend=None, basis_gates=None)
        assert bg == ["rz", "sx", "x", "cx"]

    def test_explicit_basis_gates_passes_through(self):
        bg = fbdd._resolve_basis_gates(backend=None, basis_gates=["rz", "cz"])
        assert bg == ["rz", "cz"]

    def test_diagonal_gate_strategy_is_exact(self):
        D = np.array([1, -1, 1j, -1j], dtype=complex)  # n=2
        qc = fbdd._strategy_diagonal_gate(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        target = np.diag(D)
        assert fbdd.unitaries_equal_up_to_global_phase(U, target, atol=1e-8)

    def test_diagonal_gate_strategy_normalize_phase(self):
        D = 1j * np.array([1, -1, 1j, -1j], dtype=complex)
        qc = fbdd._strategy_diagonal_gate(
            D_diag_perm=D,
            normalize_phase=True,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        # When normalized, the circuit implements D / D[0]; up to global
        # phase that is the same operator as diag(D).
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)


def _canonical_d_diag():
    """The 6-qubit example from the user spec."""
    return np.array([
         1,  1,  1j,  1, -1j, -1,  1j,  1,
        -1j, -1, -1, -1,  1j, -1j,  1, -1,
         1j,  1j, -1j, -1j,  1, -1j, -1j,  1j,
        -1j,  1, -1,  1j, -1,  1,  1j, -1,
        -1,  1, -1,  1j, -1,  1,  1j, -1j,
         1,  1, -1j, -1, -1,  1,  1j, -1j,
        -1, -1, -1, -1, -1,  1, -1j,  1,
        -1, -1, -1, -1,  1j,  1j, -1j, -1
    ], dtype=complex)


class TestSparsePhasePolyStrategy:
    def test_sparse_strategy_z0_diagonal(self):
        # diag(1, -1, 1, -1) = exp(i pi/2 * Z_0) up to a global phase.
        D = np.array([1, -1, 1, -1], dtype=complex)
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)

    def test_sparse_strategy_two_term_diagonal(self):
        # Hand-crafted: c_{0,1} = pi/3, c_{0,1,2} = -pi/5, no others.
        n = 3
        N = 1 << n
        c = {frozenset({0, 1}): np.pi / 3, frozenset({0, 1, 2}): -np.pi / 5}
        theta = np.zeros(N)
        for k in range(N):
            for S, cS in c.items():
                m = 0
                for i in S:
                    m |= 1 << i
                sign = 1 - 2 * (bin(m & k).count("1") % 2)
                theta[k] += cS * sign
        D = np.exp(1j * theta)
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-8)

    def test_sparse_strategy_canonical_6_qubit_is_exact(self):
        D = _canonical_d_diag()
        qc = fbdd._strategy_sparse_phase_poly(
            D_diag_perm=D,
            normalize_phase=False,
            basis_gates=["rz", "sx", "x", "cx"],
            backend=None,
            optimization_level=1,
            seed=0,
        )
        U = Operator(qc).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, np.diag(D), atol=1e-7)


class TestDriver:
    def test_driver_canonical_example_validates_and_returns_best(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=False,
            verbose=False,
        )
        assert "best_circuit" in result
        assert "best_score" in result
        assert "best_metadata" in result
        assert "all_candidates" in result
        assert "diagnostics" in result
        # best circuit must reproduce D up to global phase
        U = Operator(result["best_circuit"]).data
        assert fbdd.unitaries_equal_up_to_global_phase(
            U, np.diag(D), atol=1e-7
        )
        # at least one candidate from each strategy must have validated
        strategies_seen = {c["strategy"] for c in result["all_candidates"]}
        assert "diagonal_gate" in strategies_seen
        assert "sparse_phase_poly" in strategies_seen
        assert all(c["validation"] == "ok" for c in result["all_candidates"])

    def test_driver_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fbdd.find_best_diagonal_decomposition(
                np.array([1, 1, 1], dtype=complex), verbose=False
            )

    def test_driver_score_orders_correctly(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=False,
            verbose=False,
        )
        scores = [
            (c["two_qubit"], c["depth"], c["total_gates"])
            for c in result["all_candidates"]
        ]
        assert scores == sorted(scores)
        assert result["best_score"] == scores[0]

    def test_driver_diagnostics_contain_phase_coefficients(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(1),
            try_qubit_permutations=False,
            verbose=False,
        )
        diag = result["diagnostics"]
        assert diag["num_qubits"] == 6
        assert "phase_coefficients" in diag
        assert "num_candidates_attempted" in diag

    def test_driver_handles_failed_candidate_without_crashing(self, monkeypatch):
        # Force every candidate to fail validation by patching the equality helper.
        D = _canonical_d_diag()
        monkeypatch.setattr(
            fbdd, "unitaries_equal_up_to_global_phase",
            lambda U, V, atol=1e-8: False
        )
        with pytest.raises(RuntimeError, match="all candidates failed"):
            fbdd.find_best_diagonal_decomposition(
                D, seeds=range(1), try_qubit_permutations=False, verbose=False
            )


class TestComparisonTable:
    def test_table_contains_required_columns(self, capsys):
        D = _canonical_d_diag()
        fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        for col in ("strategy", "seed", "perm", "norm", "rz", "cx",
                    "2q", "total", "depth", "validation"):
            assert col in out, f"missing column header: {col!r}"

    def test_table_contains_best_marker(self, capsys):
        D = _canonical_d_diag()
        fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        assert "BEST" in out

    def test_table_rows_match_candidate_count(self, capsys):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D, seeds=range(1), try_qubit_permutations=False, verbose=True
        )
        out = capsys.readouterr().out
        candidate_count = len(result["all_candidates"])
        # A "data row" is any line that mentions one of the strategy names AND
        # ends in a validation status (ok/skipped). The BEST-marked row also
        # matches; the trailing summary line and the header do not.
        data_row_lines = [
            line for line in out.splitlines()
            if (("diagonal_gate" in line or "sparse_phase_poly" in line)
                and line.rstrip().endswith(("ok", "skipped"))
                # exclude the trailing "BEST -> " summary line
                and not line.lstrip().startswith("BEST -> "))
        ]
        assert len(data_row_lines) == candidate_count


class TestEndToEndWithPermutations:
    @pytest.mark.slow
    def test_canonical_with_permutations_validates(self):
        D = _canonical_d_diag()
        result = fbdd.find_best_diagonal_decomposition(
            D,
            seeds=range(2),
            try_qubit_permutations=True,
            verbose=False,
        )
        # All accepted candidates must be marked validation == "ok"
        assert all(c["validation"] == "ok" for c in result["all_candidates"])
        # The best candidate's circuit must reproduce its declared
        # permuted diagonal up to global phase.
        best = result["best_metadata"]
        D_perm = fbdd._permute_diagonal(D, best["permutation"])
        if best["normalize_phase"]:
            target = np.diag(D_perm) / D_perm[0]
        else:
            target = np.diag(D_perm)
        U = Operator(result["best_circuit"]).data
        assert fbdd.unitaries_equal_up_to_global_phase(U, target, atol=1e-7)
