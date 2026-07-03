"""Direct qutrit-basis encoding benchmarks.

This package keeps the direct-basis method separate from the legacy
"append W as a physical basis-change gate" pipeline.
"""

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    benchmark_direct_basis,
    benchmark_direct_basis_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    DirectBasisCandidate,
    generate_all_qutrit_u3_candidates,
    generate_extended_legacy_candidates,
    generate_sanity_basis_candidates,
    generate_v2_stage1_direct_candidates,
)

__all__ = [
    "DirectBasisCandidate",
    "benchmark_direct_basis",
    "benchmark_direct_basis_candidates",
    "generate_all_qutrit_u3_candidates",
    "generate_extended_legacy_candidates",
    "generate_sanity_basis_candidates",
    "generate_v2_stage1_direct_candidates",
]
