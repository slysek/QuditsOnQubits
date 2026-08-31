from __future__ import annotations

import qudits_on_qubits
from qudits_on_qubits.experiments import (
    IQMQubitSelectorConfig,
    WorkloadOptimizationConfig,
)


def test_top_level_workload_optimization_config_alias_matches_experiments_export():
    assert qudits_on_qubits.WorkloadOptimizationConfig is WorkloadOptimizationConfig


def test_top_level_iqm_qubit_selector_config_alias_matches_experiments_export():
    assert qudits_on_qubits.IQMQubitSelectorConfig is IQMQubitSelectorConfig
