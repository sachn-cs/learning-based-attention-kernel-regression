# LAKER Design Patterns

This document describes the architectural patterns used throughout the LAKER
project. All new code should follow these conventions.

---

## 1. Executor Pattern

### What it is

The **Executor Pattern** decouples workflow logic from output formatting.
Workflow classes (e.g. examples, benchmarks) accept an optional ``Executor``
instance in their constructor and delegate all logging, timing, and result
collection to it.

### Why we use it

- **Consistency**: Every workflow emits the same structured output format.
- **Testability**: Workflows can be run with a silent or mock executor in tests.
- **Reusability**: The same workflow can be run in a notebook, a benchmark
  suite, or a CI job without code changes.

### Base interface

The abstract base class lives in ``laker.executor.Executor``:

```python
from abc import ABC, abstractmethod
from typing import Any, Callable

class Executor(ABC):
    @abstractmethod
    def section(self, title: str) -> None: ...
    @abstractmethod
    def log_result(self, key: str, value: Any) -> None: ...
    @abstractmethod
    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None: ...
    @abstractmethod
    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any: ...
```

### Concrete executors

- ``examples.executor.ExampleExecutor`` — structured logging for demonstrations.
- ``benchmarks.executor.BenchmarkExecutor`` — timing, warmup, and statistical
  aggregation for performance measurement.

### Writing a workflow that uses an executor

```python
from typing import Optional
from laker.executor import Executor

class MyWorkflow:
    def __init__(self, executor: Optional[Executor] = None):
        self.executor = executor if executor is not None else ExampleExecutor()

    def run(self) -> None:
        self.executor.section("My Workflow")
        result = self.executor.time_operation("heavy step", self.heavy_step)
        self.executor.log_metric("accuracy", result.accuracy)
```

Constructor rule: always accept ``executor: Optional[Executor] = None`` and
fall back to a sensible default.

---

## 2. Class + Convenience Wrapper

### Rule: every public workflow must have a class implementation

Standalone free functions are **not** acceptable as the primary API for
workflows, benchmarks, or examples. They may exist as thin convenience
wrappers, but a class must always be available.

### Pattern

```python
class MyBenchmark:
    def __init__(self, executor: Optional[Executor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()

    def run(self) -> dict:
        ...

    @classmethod
    def run_default(cls) -> dict:
        """Convenience entry point."""
        return cls().run()
```

### When to keep a free-function wrapper

Free-function wrappers are permitted **only** in the public API package
(``laker/``) for heavily-used one-liners such as ``generate_radio_field`` or
``plot_radio_map``. These wrappers must:

1. Be a single line: instantiate the class and forward arguments.
2. Carry a docstring that explicitly states:
   ``"Convenience wrapper around <Class>.<method>()."``
3. Never contain business logic.

---

## 3. Naming Conventions

- **No semi-private naming**: leading underscores are not used for functions,
  methods, classes, or variables. If something is truly internal, document it
  in the docstring rather than hiding it with an underscore.
- **Descriptive names**: avoid single-letter or cryptic abbreviations.
  ``chunk_size`` is preferred over ``cs``; ``preconditioner`` over ``pre``.
- **PEP 8**: all identifiers are ``snake_case``; classes are ``PascalCase``.

---

## 4. Logging

All diagnostic output must use the ``logging`` module. ``print()`` is forbidden
in library code, examples, and benchmarks. Documentation code blocks should
likewise demonstrate ``logger.info(...)`` rather than ``print(...)``.
