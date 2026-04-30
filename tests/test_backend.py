"""Tests for backend utilities."""

import pytest
import torch

from laker.backend import (
    get_default_device,
    get_default_dtype,
    maybe_compile,
    set_default_device,
    set_default_dtype,
    to_tensor,
)


def test_get_default_dtype():
    """Default dtype should be float32 after our changes."""
    assert get_default_dtype() == torch.float32


def test_set_default_dtype():
    """set_default_dtype should mutate the global default."""
    old = get_default_dtype()
    set_default_dtype(torch.float64)
    assert get_default_dtype() == torch.float64
    set_default_dtype(old)
    assert get_default_dtype() == old


def test_to_tensor_from_numpy():
    """to_tensor should convert numpy arrays."""
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0])
    t = to_tensor(arr)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == get_default_dtype()


def test_to_tensor_from_list():
    """to_tensor should convert lists."""
    t = to_tensor([1.0, 2.0, 3.0])
    assert isinstance(t, torch.Tensor)
    assert t.tolist() == [1.0, 2.0, 3.0]


def test_maybe_compile_returns_callable():
    """maybe_compile should return a callable."""

    def dummy(x):
        return x + 1

    compiled = maybe_compile(dummy)
    assert callable(compiled)
    # Does not raise
    result = compiled(torch.tensor(1.0))
    assert result.item() == 2.0
