"""Base executor interface for structured logging, timing, and benchmarking."""

from abc import ABC, abstractmethod
from typing import Any, Callable


class Executor(ABC):
    """Abstract base class for workflow executors.

    An executor provides structured logging, timing, and result collection
    for runnable workflows. Subclasses implement concrete logging backends
    (e.g. console logging, benchmark metrics, file output).

    Workflow classes should accept an optional ``Executor`` instance in
    their constructor and delegate logging and timing to it. This keeps
    business logic decoupled from output formatting.
    """

    @abstractmethod
    def section(self, title: str) -> None:
        """Log a section header.

        Args:
            title: Human-readable section title.
        """

    @abstractmethod
    def log_result(self, key: str, value: Any) -> None:
        """Log a named result.

        Args:
            key: Result identifier.
            value: Result value (any JSON-serialisable type).
        """

    @abstractmethod
    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting.

        Args:
            name: Metric identifier.
            value: Numeric value.
            fmt: Python format specification for the value.
        """

    @abstractmethod
    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run an operation and log its elapsed time.

        Args:
            name: Human-readable operation name.
            operation: Callable taking no arguments.

        Returns:
            The return value of ``operation``.
        """
