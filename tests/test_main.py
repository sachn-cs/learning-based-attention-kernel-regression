"""Tests for the CLI entry point in laker.__main__."""

import sys
from unittest.mock import patch

import numpy
import pytest
import torch

from laker.__main__ import (
    cmd_fit,
    cmd_predict,
    load_tensor,
    main,
    setup_logging,
)
from laker.models import LAKERRegressor


def test_setup_logging_verbose():
    """setup_logging with verbose=True should not raise."""
    setup_logging(verbose=True)


def test_setup_logging_quiet():
    """setup_logging with verbose=False should not raise."""
    setup_logging(verbose=False)


def test_load_tensor_npy(tmp_path):
    """load_tensor should read .npy files."""
    path = tmp_path / "data.npy"
    arr = numpy.array([1.0, 2.0, 3.0])
    numpy.save(path, arr)
    tensor = load_tensor(str(path))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == [1.0, 2.0, 3.0]


def test_load_tensor_pt(tmp_path):
    """load_tensor should read .pt files."""
    path = tmp_path / "data.pt"
    tensor = torch.tensor([1.0, 2.0, 3.0])
    torch.save(tensor, path)
    loaded = load_tensor(str(path))
    assert torch.equal(loaded, tensor)


def test_load_tensor_unsupported(tmp_path):
    """load_tensor should raise ValueError for unsupported extensions."""
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tensor(str(path))


def test_load_tensor_missing():
    """load_tensor should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_tensor("/nonexistent/path/data.npy")


def test_main_version(capsys):
    """main with --version should print version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        with patch.object(sys, "argv", ["laker", "--version"]):
            main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "laker" in captured.out.lower() or "0." in captured.out


def test_main_no_subcommand(capsys):
    """main without subcommand should print help and exit with code 1."""
    with pytest.raises(SystemExit) as exc_info:
        with patch.object(sys, "argv", ["laker"]):
            main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "fit" in captured.out


def test_cmd_fit(tmp_path):
    """cmd_fit should fit a model and save it."""
    torch.manual_seed(42)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    loc_path = tmp_path / "locations.pt"
    meas_path = tmp_path / "measurements.pt"
    out_path = tmp_path / "model.pt"

    torch.save(x, loc_path)
    torch.save(y, meas_path)

    class Args:
        locations = str(loc_path)
        measurements = str(meas_path)
        output = str(out_path)
        lambda_reg = 1e-2
        gamma = 1e-1
        embedding_dim = 6
        num_probes = 30
        device = "cpu"
        dtype = "float64"
        verbose = False

    cmd_fit(Args())
    assert out_path.exists()
    model = LAKERRegressor.load(str(out_path))
    assert model.alpha is not None


def test_cmd_predict(tmp_path):
    """cmd_predict should load a model and save predictions."""
    torch.manual_seed(42)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        device="cpu",
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    model_path = tmp_path / "model.pt"
    model.save(str(model_path))

    query_path = tmp_path / "query.pt"
    x_query = torch.rand(10, 2, dtype=torch.float64) * 100.0
    torch.save(x_query, query_path)

    out_path = tmp_path / "predictions.pt"

    class Args:
        model = str(model_path)
        locations = str(query_path)
        output = str(out_path)

    cmd_predict(Args())
    assert out_path.exists()
    preds = torch.load(out_path)
    assert preds.shape == (10,)


def test_main_predict_subcommand(tmp_path):
    """main with predict subcommand should run successfully."""
    torch.manual_seed(42)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=10,
        device="cpu",
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    model_path = tmp_path / "model.pt"
    model.save(str(model_path))

    query_path = tmp_path / "query.pt"
    x_query = torch.rand(5, 2, dtype=torch.float64) * 100.0
    torch.save(x_query, query_path)

    out_path = tmp_path / "predictions.pt"

    argv = [
        "laker",
        "predict",
        "--model",
        str(model_path),
        "--locations",
        str(query_path),
        "--output",
        str(out_path),
    ]
    with patch.object(sys, "argv", argv):
        main()

    assert out_path.exists()
    preds = torch.load(out_path)
    assert preds.shape == (5,)


def test_main_fit_subcommand(tmp_path):
    """main with fit subcommand should run successfully."""
    torch.manual_seed(42)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    loc_path = tmp_path / "locs.pt"
    meas_path = tmp_path / "meas.pt"
    out_path = tmp_path / "model.pt"

    torch.save(x, loc_path)
    torch.save(y, meas_path)

    argv = [
        "laker",
        "fit",
        "--locations",
        str(loc_path),
        "--measurements",
        str(meas_path),
        "--output",
        str(out_path),
        "--lambda-reg",
        "0.01",
        "--gamma",
        "0.1",
        "--embedding-dim",
        "4",
        "--num-probes",
        "10",
        "--dtype",
        "float64",
    ]
    with patch.object(sys, "argv", argv):
        main()

    assert out_path.exists()
    model = LAKERRegressor.load(str(out_path))
    assert model.alpha is not None
