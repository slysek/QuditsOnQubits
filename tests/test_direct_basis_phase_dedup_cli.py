from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.run_direct_basis_benchmarks import _load_candidates, build_parser, main
from qudits_on_qubits.benchmarks.direct_basis.phase_equivalence import PHASE_DUPLICATE_COLUMNS


MONOMIAL_ARGS = [
    "--state", "two_qutrit", "--candidate-set", "v2-stage1",
    "--candidate-class", "monomial_full", "--compare-optimal-f3-leakage",
]


@pytest.mark.parametrize("iqm", [False, True])
def test_f3_benchmark_deduplicates_full_pool_and_audits_removed_candidates(tmp_path, capsys, iqm):
    output = tmp_path / "nested" / "results.csv"
    raw = _load_candidates(build_parser().parse_args(MONOMIAL_ARGS))
    assert len(raw) == 649
    args = MONOMIAL_ARGS + ["--deduplicate-global-phase", "--output-csv", str(output)]
    if iqm:
        args += ["--iqm-backend", "garnet"]

    with (
        patch("scripts.run_direct_basis_benchmarks.load_iqm_backend", return_value=object()),
        patch("scripts.run_direct_basis_benchmarks.backend_metadata", return_value={}),
        patch("scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates",
              return_value=(pd.DataFrame(), str(output))) as benchmark,
    ):
        assert main(args) == 0

    kwargs = benchmark.call_args.kwargs
    representatives = kwargs["candidates"]
    assert kwargs["compare_optimal_f3_leakage"] is True
    assert len(representatives) == 216
    assert representatives[0].class_name == "baseline"
    assert representatives[0].candidate_name == "E_old"
    # Preserve all four physical supports, with 54 representatives each.
    supports = {}
    for candidate in representatives:
        matrix = candidate.matrix
        if matrix.shape == (3, 3):
            matrix = np.vstack([matrix, np.zeros((1, 3))])
        support = tuple(np.flatnonzero(np.any(np.abs(matrix) > 1e-9, axis=1)))
        supports[support] = supports.get(support, 0) + 1
    assert len(supports) == 4
    assert set(supports.values()) == {54}

    audit = pd.read_csv(output.with_name("results_global_phase_duplicates.csv"))
    assert tuple(audit.columns) == PHASE_DUPLICATE_COLUMNS
    assert len(audit) == 433
    by_key = {(c.class_name, c.candidate_name): c for c in raw}
    representative_keys = {(c.class_name, c.candidate_name) for c in representatives}
    duplicate_keys = set()
    for row in audit.itertuples(index=False):
        rep_key = (row.representative_class_name, row.representative_candidate_name)
        dup_key = (row.duplicate_class_name, row.duplicate_candidate_name)
        assert rep_key in representative_keys
        duplicate_keys.add(dup_key)
        phase = complex(row.detected_phase_real, row.detected_phase_imag)
        np.testing.assert_allclose(by_key[dup_key].matrix, phase * by_key[rep_key].matrix, atol=1e-9)
    assert duplicate_keys.isdisjoint(representative_keys)
    assert duplicate_keys | representative_keys == set(by_key)
    stdout = capsys.readouterr().out
    assert "649 -> 216" in stdout
    assert "candidates=216" in stdout


def test_phase_dedup_is_opt_in(tmp_path):
    output = tmp_path / "results.csv"
    with patch("scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates",
               return_value=(pd.DataFrame(), str(output))) as benchmark:
        assert main(MONOMIAL_ARGS + ["--output-csv", str(output)]) == 0
    assert len(benchmark.call_args.kwargs["candidates"]) == 649
    assert not output.with_name("results_global_phase_duplicates.csv").exists()


def test_phase_dedup_writes_empty_audit_without_f3_comparison(tmp_path):
    output = tmp_path / "results.csv"
    with patch("scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates",
               return_value=(pd.DataFrame(), str(output))) as benchmark:
        assert main([
            "--state", "two_qutrit", "--candidate-set", "v2-stage1",
            "--candidate-class", "baseline", "--limit-candidates", "0",
            "--deduplicate-global-phase", "--output-csv", str(output),
        ]) == 0
    assert len(benchmark.call_args.kwargs["candidates"]) == 1
    assert benchmark.call_args.kwargs["compare_optimal_f3_leakage"] is False
    audit = pd.read_csv(output.with_name("results_global_phase_duplicates.csv"))
    assert tuple(audit.columns) == PHASE_DUPLICATE_COLUMNS
    assert audit.empty
