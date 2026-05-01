"""Example executor providing structured logging and timing for demonstrations."""

import logging
import time
from typing import Any, Callable, Dict

from laker.executor import Executor

logger = logging.getLogger(__name__)


class ExampleExecutor(Executor):
    """Executes example workflows with structured logging and timing."""

    def __init__(self, name: str = "LAKER Example"):
        self.name = name
        self.results: Dict[str, Any] = {}

    def section(self, title: str) -> None:
        """Log a section header."""
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)

    def log_result(self, key: str, value: Any) -> None:
        """Log a named result."""
        self.results[key] = value
        logger.info("%s: %s", key, value)

    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run an operation and log its elapsed time."""
        logger.info("Starting: %s", name)
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        logger.info("Completed: %s in %.3fs", name, elapsed)
        return result

    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting."""
        formatted = format(value, fmt)
        self.results[name] = value
        logger.info("%s: %s", name, formatted)
