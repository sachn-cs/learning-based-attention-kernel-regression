"""Custom embedding module for save/load testing."""

import torch
import torch.nn as nn


class CustomEmbedding(nn.Module):
    """Simple custom embedding for testing save/load."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 10, dtype=torch.float64)

    def forward(self, x):
        return torch.tanh(self.fc(x))
