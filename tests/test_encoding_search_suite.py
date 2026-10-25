from dataclasses import replace
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from qudits_on_qubits.encoding_search import suite


def test_small_suite_keeps_baseline_and_resumes_saved_results(tmp_path, monkeypatch):
    config = suite.SuiteConfig(
        suite_name="small", states=("two_qutrit",), output_root=str(tmp_path),
        n_transpile_runs=1, approximation_values=(1.0,), export_circuits=True,
    )
    candidates = [("baseline", "E_old", None), ("product", "identity", np.eye(4, 3))]
    monkeypatch.setattr(suite, "generate_all_class_candidates", lambda **kwargs: candidates)

    result = suite.run_benchmark_suite(config, log_stream=StringIO())

    frame = result["state_frames"]["two_qutrit"].set_index("candidate_name")
    assert frame.loc["E_old", "status"] == "ok"
    assert frame.loc["E_old", "successful_trials"] == 1
    assert frame.loc["identity", "status"] == "skipped_baseline_equivalent"
    original = pd.read_csv(result["combined_csv"])
    monkeypatch.setattr(suite, "benchmark_basis", lambda *args, **kwargs: pytest.fail("Unexpected recompilation"))

    resumed = suite.run_benchmark_suite(replace(config, skip_existing=True), log_stream=StringIO())

    pd.testing.assert_frame_equal(pd.read_csv(resumed["combined_csv"]), original)


def test_corrupt_saved_results_do_not_replace_combined_results(tmp_path, monkeypatch):
    config = suite.SuiteConfig(
        suite_name="saved", states=("two_qutrit",),
        output_root=str(tmp_path), skip_existing=True,
    )
    directory = tmp_path / "saved" / "two_qutrit"
    directory.mkdir(parents=True)
    (directory / "encoding_search_v2_suite_two_qutrit_results.csv").write_text(
        "candidate_name,depth\na,1\nb,2,3\n", encoding="utf-8",
    )
    combined = tmp_path / "saved" / suite.COMBINED_RESULTS_BASENAME
    combined.write_text("previous results\n", encoding="utf-8")
    monkeypatch.setattr(suite, "generate_all_class_candidates", lambda **kwargs: [])

    with pytest.raises(pd.errors.ParserError):
        suite.run_benchmark_suite(config, log_stream=StringIO())

    assert combined.read_text(encoding="utf-8") == "previous results\n"
