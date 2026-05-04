"""Residual corrector models for LAKER."""

import torch
import torch.nn as nn


class ResidualCorrector(nn.Module):
    """Tiny MLP that predicts the residual ``y - y_hat_laker``.

    The corrector operates on raw spatial coordinates (or optionally on
    embedding vectors) and adds its output to the base LAKER prediction.
    It is trained with strong L2 regularisation and early stopping to avoid
    overfitting the residual.

    Args:
        input_dim: Dimensionality of the input features (e.g. spatial coords).
        output_dim: Dimensionality of the target. Default 1 for scalar regression.
        hidden_dim: Width of the single hidden layer. Default 32.
        dropout: Dropout probability on the hidden layer. Default 0.1.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the residual corrector network."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residual correction for inputs ``x``.

        Args:
            x: Tensor of shape ``(m, input_dim)``.

        Returns:
            Tensor of shape ``(m, output_dim)``.

        """
        return self.net(x)
