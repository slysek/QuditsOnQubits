"""Typed errors raised by experiment specification and execution code."""


class ExperimentError(Exception):
    """Base error for all experiment runner failures."""


class ExperimentValidationError(ExperimentError, ValueError):
    """Raised when an experiment input is invalid or unsafe to persist."""


class OptionalDependencyError(ExperimentError, ImportError):
    """Raised when an optional backend integration is unavailable."""


class BackendUnavailableError(ExperimentError):
    """Raised when a requested backend cannot be reached."""


class BackendCompatibilityError(ExperimentError):
    """Raised when a backend cannot run the requested experiment."""


class JobSubmissionError(ExperimentError):
    """Raised when a backend rejects a job submission."""


class JobResultError(ExperimentError):
    """Raised when a submitted job cannot provide a usable result."""


class ExperimentPersistenceError(ExperimentError):
    """Raised when experiment artifacts cannot be safely persisted."""
