from __future__ import annotations

import os
import sys
from pprint import pprint

from igraph import Graph
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from qutrit_bell_measurements import (  # noqa: E402
    build_sampler_circuits_for_candidate,
    build_sampler_circuits_from_graph,
    canonical_Ez,
)


graph = Graph()
graph.add_vertices(2)
graph.add_edge(0, 1, weight=1)

state_circuit = QuantumCircuit(4)
state_circuit.h(0)
state_circuit.cx(0, 2)

sampler_circuits, metadata = build_sampler_circuits_from_graph(
    state_circuit=state_circuit,
    graph=graph,
    E=canonical_Ez(),
    d=3,
)

print(f"number of circuits: {len(sampler_circuits)}")
print("measurement settings:")
pprint(metadata["measurement_settings"])
print("\nfirst two circuits:")
for circuit in sampler_circuits[:2]:
    print(circuit)
print("\nfirst three Bell terms:")
pprint(metadata["terms"][:3])

ghz_state_circuit = QuantumCircuit(6)
ghz_circuits, ghz_metadata = build_sampler_circuits_for_candidate(
    candidate="ghz3",
    state_circuit=ghz_state_circuit,
    E=canonical_Ez(),
    d=3,
)

print("\nGHZ3 candidate API:")
print(f"number of circuits: {len(ghz_circuits)}")
print("first three settings:")
pprint(ghz_metadata["measurement_settings"][:3])
print("first three Bell terms:")
pprint(ghz_metadata["terms"][:3])
