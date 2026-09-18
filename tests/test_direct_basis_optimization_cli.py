"""The legacy CLI must preserve the requested compilation protocol."""
import contextlib
import io
from unittest.mock import patch

import pytest

from scripts.run_direct_basis_benchmarks import main


@pytest.mark.parametrize("provider", ["local", "iqm", "ibm"])
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_optimization_level_reaches_compilation_and_metadata(provider, level):
    module = "scripts.run_direct_basis_benchmarks."
    options = {"local": [], "iqm": ["--iqm-backend", "garnet"],
               "ibm": ["--ibm-backend", "ibm_test"]}[provider]
    with (
        patch(module + "_load_candidates", return_value=[object()]),
        patch(module + "load_iqm_backend", return_value=object()),
        patch(module + "load_ibm_backend", return_value=object()),
        patch(module + "backend_metadata", return_value={"optimization_level": level}) as iqm_metadata,
        patch(module + "ibm_backend_metadata", return_value={"transpiler_backend": "ibm"}),
        patch(module + "benchmark_direct_basis_candidates", return_value=(None, "out.csv")) as run,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert main(["--state", "two_qutrit", "--output-csv", "out.csv",
                     "--no-export-quantum-circuits", "--optimization-level", str(level), *options]) == 0
    assert run.call_args.kwargs["optimization_level"] == level
    if provider == "iqm":
        assert iqm_metadata.call_args.kwargs["optimization_level"] == level
        assert run.call_args.kwargs["transpiler_metadata"]["optimization_level"] == level


@pytest.mark.parametrize("level", ["-1", "4", "1.5"])
def test_invalid_optimization_level_fails_before_candidate_loading(level):
    with patch("scripts.run_direct_basis_benchmarks._load_candidates") as load:
        with pytest.raises(SystemExit) as caught:
            main(["--state", "two_qutrit", "--optimization-level", level])
        assert caught.value.code == 2
        load.assert_not_called()