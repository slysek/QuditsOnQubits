"""Small offline encoding benchmark using the public API and exact synthesis."""
from qudits_on_qubits.benchmarks import (
    BackendTarget, BenchmarkConfig, ExactSynthesis, LocalSU2, SchmidtTheta,
    run_benchmark, two_qutrit_graph_circuit,
)


def main():
    result = run_benchmark(
        two_qutrit_graph_circuit(),
        families=[LocalSU2(samples=2, seed=42), SchmidtTheta(points=3)],
        backend=BackendTarget.local(num_qubits=4, coupling_map=[
            (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
        ]),
        config=BenchmarkConfig(transpiler_seeds=(0, 1, 2)),
        synthesis=ExactSynthesis(),
        progress=print,
    )
    print(result.pareto_front[["candidate_name", "mean_two_qubit_gate_count", "mean_depth", "std_depth"]])
    print(result.output_dir / "report.md")


if __name__ == "__main__":
    main()
