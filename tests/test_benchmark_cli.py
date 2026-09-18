import pytest
from qudits_on_qubits.benchmarks.cli import main


def test_backend_is_required():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_invalid_theta_family_is_rejected_without_loading_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("backend should not be loaded")
    monkeypatch.setattr("qudits_on_qubits.benchmarks.cli.BackendTarget.iqm", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["--backend", "iqm", "--device", "garnet", "--synthesis", "theta",
              "--family", "local-su2"])
    assert error.value.code == 2


def test_cli_calls_same_api_and_propagates_configuration(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    captured = {}
    def fake(circuit, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path, trials=pd.DataFrame({"success":[True,True]}),
                               pareto_front=pd.DataFrame({"candidate":[1]}))
    monkeypatch.setattr("qudits_on_qubits.benchmarks.cli.run_benchmark", fake)
    assert main(["--backend","local","--family","schmidt-theta","--theta-points","3",
                 "--seeds","4","7","--synthesis","exact","--output-dir",str(tmp_path)]) == 0
    assert captured["config"].transpiler_seeds == (4,7)
    assert captured["families"][0].points == 3
    assert captured["backend"].provider == "local"


def test_public_api_exports():
    import qudits_on_qubits.benchmarks as api
    for name in ("run_benchmark", "load_benchmark", "LocalSU2", "SchmidtTheta", "BackendTarget",
                 "BenchmarkConfig", "LogicalCircuit", "LogicalOperation", "ExactSynthesis",
                 "OptimizedSynthesis", "ThetaContinuationSynthesis", "SavedGateSynthesis",
                 "saved_candidate", "two_qutrit_graph_circuit"):
        assert callable(getattr(api, name))


@pytest.mark.parametrize("args", [
    ["--backend", "local", "--family", "local-su2", "--family", "local-su2"],
    ["--backend", "iqm"],
    ["--backend", "local", "--device", "garnet"],
    ["--backend", "local", "--theta-source-run", "unused"],
    ["--backend", "local", "--reuse-saved-preparation"],
])
def test_incompatible_cli_arguments_rejected(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_new_continuation_requires_pinned_baseline(capsys):
    assert main(["--backend", "local", "--synthesis", "theta"]) == 2
    assert "--theta-baseline-qpy" in capsys.readouterr().err


def test_cli_can_add_saved_basis_to_generated_families(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from qiskit import qpy
    from qudits_on_qubits.benchmarks import LocalSU2, ExactSynthesis, SavedGateSynthesis

    candidate = next(LocalSU2(samples=1).generate())
    library = ExactSynthesis().build(candidate)
    np.save(tmp_path / "E.npy", candidate.encoding)
    for name, circuit in (("F3_W.qpy", library.f3), ("CZ3_W.qpy", library.cz3)):
        with (tmp_path / name).open("wb") as handle:
            qpy.dump(circuit, handle)
    captured = {}
    def run(circuit, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path, trials=pd.DataFrame({"success":[True, True]}),
                               pareto_front=pd.DataFrame({"candidate":[1]}))
    monkeypatch.setattr("qudits_on_qubits.benchmarks.cli.run_benchmark", run)
    assert main(["--backend", "local", "--family", "local-su2", "--local-samples", "1",
                 "--synthesis", "exact", "--saved-basis", str(tmp_path)]) == 0
    assert len(captured["saved_candidates"]) == 1
    assert isinstance(captured["synthesis"]["saved"], SavedGateSynthesis)
    assert isinstance(captured["synthesis"]["local_su2"], ExactSynthesis)
