"""Pinball (quantile) loss at τ = 0.05 (Section 5).

τ = 0.05 penalises over-prediction 19x more than under-prediction, so the model
learns to hug the lower envelope -- the fastest reliably observed arrival times.
"""
from __future__ import annotations

import torch

from ..config import PINBALL_TAU


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = PINBALL_TAU) -> torch.Tensor:
    err = target - pred
    return torch.mean(torch.maximum(tau * err, (tau - 1.0) * err))
