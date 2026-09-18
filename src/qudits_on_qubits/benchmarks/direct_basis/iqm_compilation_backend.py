"""Offline IQM backend snapshot for spawned transpilation workers."""
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from qiskit.providers import Options


class IqmCompilationBackend(IQMBackendBase):
    """Preserve IQM target construction without carrying a network client."""

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, *args, **kwargs):
        raise RuntimeError("Compilation snapshots cannot submit hardware jobs.")

    def __reduce__(self):
        # Qiskit's Target pickle drops IQM-specific attributes. Rebuild targets
        # from the same architecture and metrics in each spawned interpreter.
        return (_restore_compilation_backend, (self.architecture, self.metrics, self.name))


def _restore_compilation_backend(architecture, metrics, name):
    return IqmCompilationBackend(architecture, metrics=metrics, name=name)


def snapshot_compilation_backend(backend):
    if not isinstance(backend, IQMBackendBase):
        return backend
    return IqmCompilationBackend(
        backend.architecture, metrics=backend.metrics, name=backend.name,
    )
