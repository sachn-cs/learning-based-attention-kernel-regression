"""Tests for backend utilities."""

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


def test_set_default_device_auto():
    """set_default_device(None) should auto-select a device."""
    old = get_default_device()
    device = set_default_device(None)
    assert isinstance(device, torch.device)
    set_default_device(old)


def test_set_default_device_string():
    """set_default_device should accept a string."""
    old = get_default_device()
    device = set_default_device("cpu")
    assert device == torch.device("cpu")
    assert get_default_device() == torch.device("cpu")
    set_default_device(old)


def test_to_tensor_explicit_device_dtype():
    """to_tensor should respect explicit device and dtype."""
    t = to_tensor([1.0, 2.0], device=torch.device("cpu"), dtype=torch.float64)
    assert t.dtype == torch.float64
    assert t.device == torch.device("cpu")


def test_to_tensor_from_tensor():
    """to_tensor should handle existing tensors."""
    src = torch.tensor([1.0, 2.0], dtype=torch.float32)
    t = to_tensor(src, dtype=torch.float64)
    assert t.dtype == torch.float64
    assert torch.equal(t, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_backend_env_vars():
    """Backend should read LAKER_DEVICE and LAKER_DTYPE from environment."""
    import importlib
    import os

    old_device = os.environ.get("LAKER_DEVICE")
    old_dtype = os.environ.get("LAKER_DTYPE")

    try:
        os.environ["LAKER_DEVICE"] = "cpu"
        os.environ["LAKER_DTYPE"] = "float64"

        # Re-import to trigger env var initialization
        from laker import backend

        importlib.reload(backend)

        assert backend.get_default_device() == torch.device("cpu")
        assert backend.get_default_dtype() == torch.float64
    finally:
        if old_device is not None:
            os.environ["LAKER_DEVICE"] = old_device
        else:
            os.environ.pop("LAKER_DEVICE", None)
        if old_dtype is not None:
            os.environ["LAKER_DTYPE"] = old_dtype
        else:
            os.environ.pop("LAKER_DTYPE", None)

        # Restore defaults for other tests
        importlib.reload(backend)
        backend.set_default_dtype(torch.float32)
