"""MLP architecture (Section 7): 6 -> 64 -> 64 -> 1 with Softplus output.

Softplus structurally enforces Δt > 0, so the loss needs no clamping.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import HIDDEN, N_FEATURES


class TTAModel(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES, hidden: int | list[int] = HIDDEN):
        super().__init__()
        widths = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        layers += [nn.Linear(prev, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
