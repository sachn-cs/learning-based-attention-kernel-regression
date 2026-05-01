"""Benchmark executor providing standardized timing and logging."""

import logging
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

from laker.executor import Executor

logger = logging.getLogger(__name__)


class BenchmarkExecutor(Executor):
    """Executes benchmarks with standardized timing, warmup, and logging."""

    def __init__(self, warmup: int = 20, trials: int = 50):
        self.warmup = warmup
        self.trials = trials
        self.results: List[Dict[str, Any]] = []

    def run(
        self,
        name: str,
        operation: Callable[[], Any],
        trials: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run a benchmark operation with warmup and multiple trials.

        Args:
            name: Identifier for this benchmark.
            operation: Callable taking no arguments to benchmark.
            trials: Number of measurement iterations. Defaults to self.trials.
            warmup: Number of warmup iterations. Defaults to self.warmup.

        Returns:
            Dictionary with mean_ms, std_ms, min_ms, max_ms, and trials.
        """
        trials = trials if trials is not None else self.trials
        warmup = warmup if warmup is not None else self.warmup

        logger.info("Starting benchmark '%s' (%d warmup, %d trials)", name, warmup, trials)

        for i in range(warmup):
            operation()

        times = []
        for i in range(trials):
            start = time.perf_counter()
            operation()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0

        result = {
            "name": name,
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": min(times),
            "max_ms": max(times),
            "trials": trials,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: mean=%.3fms std=%.3fms", name, mean, std)

        return result

    def run_once(self, name: str, operation: Callable[[], Any]) -> Dict[str, float]:
        """Run a benchmark operation once and record the time.

        Args:
            name: Identifier for this benchmark.
            operation: Callable taking no arguments to benchmark.

        Returns:
            Dictionary with mean_ms, std_ms, min_ms, max_ms, and trials.
        """
        logger.info("Starting single-run benchmark '%s'", name)

        start = time.perf_counter()
        operation()
        elapsed = (time.perf_counter() - start) * 1000

        result = {
            "name": name,
            "mean_ms": elapsed,
            "std_ms": 0.0,
            "min_ms": elapsed,
            "max_ms": elapsed,
            "trials": 1,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: %.3fms", name, elapsed)

        return result

    def section(self, title: str) -> None:
        """Log a section header."""
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)

    def log_result(self, key: str, value: Any) -> None:
        """Log a named result."""
        logger.info("%s: %s", key, value)

    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting."""
        formatted = format(value, fmt)
        logger.info("%s: %s", name, formatted)

    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run an operation and log its elapsed time."""
        logger.info("Starting: %s", name)
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        logger.info("Completed: %s in %.3fs", name, elapsed)
        return result

    def run_repeated(
        self, name: str, operation: Callable[[], Any], repetitions: int = 20
    ) -> Dict[str, float]:
        """Run an operation multiple times in a single timed block and average.

        Args:
            name: Identifier for this benchmark.
            operation: Callable taking no arguments to benchmark.
            repetitions: Number of repetitions within the timed block.

        Returns:
            Dictionary with mean_ms per repetition, std_ms=0, min_ms, max_ms, and trials.
        """
        logger.info("Starting repeated benchmark '%s' (%d repetitions)", name, repetitions)

        start = time.perf_counter()
        for i in range(repetitions):
            operation()
        elapsed = (time.perf_counter() - start) / repetitions * 1000

        result = {
            "name": name,
            "mean_ms": elapsed,
            "std_ms": 0.0,
            "min_ms": elapsed,
            "max_ms": elapsed,
            "trials": repetitions,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: %.3fms per iteration", name, elapsed)

        return result
