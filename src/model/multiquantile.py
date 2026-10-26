"""Experimental multi-quantile TTA model.

Jointly fits several pinball quantiles with a monotone (non-crossing) output,
to test whether shared statistical strength from the well-populated central
quantiles stabilises the sparsely-populated tau=0.01 extreme tail -- the
suspected cause of the reachable-area degradation at long horizons /
high coverage (see outputs/ffpc_match/experiment_multiquantile*.log).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import HIDDEN, N_FEATURES


class MultiQuantileTTAModel(nn.Module):
    """in_dim -> hidden -> hidden -> K, one output per quantile in `quantiles`
    (ascending). Outputs are constrained non-decreasing in quantile index via a
    cumulative-softplus parameterisation: Q_1 = softplus(z_1), Q_k = Q_{k-1} +
    softplus(z_k) for k>1. This makes quantile crossing structurally
    impossible and guarantees Delta t > 0 for every quantile, the same
    guarantee the single-quantile TTAModel's Softplus output gives.
    """

    def __init__(self, quantiles, in_dim: int = N_FEATURES, hidden: int | list[int] = HIDDEN):
        super().__init__()
        self.quantiles = list(quantiles)
        assert self.quantiles == sorted(self.quantiles), "quantiles must be ascending"
        k = len(self.quantiles)
        widths = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        layers += [nn.Linear(prev, k)]
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        increments = F.softplus(self.trunk(x))   # (N, K), each > 0
        return torch.cumsum(increments, dim=1)    # (N, K), non-decreasing in k


def multi_pinball_loss(preds: torch.Tensor, target: torch.Tensor, taus) -> torch.Tensor:
    """Mean pinball loss averaged over K quantiles. preds: (N,K); target: (N,)."""
    y = target.unsqueeze(1)
    tau = torch.as_tensor(taus, dtype=preds.dtype, device=preds.device).unsqueeze(0)
    err = y - preds
    return torch.maximum(tau * err, (tau - 1.0) * err).mean()
