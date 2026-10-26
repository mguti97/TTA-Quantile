"""Differentiable soft reachable-area auxiliary loss (canonical since 2026-07-03).

The player TTA surface is translation/rotation invariant, so its reachable-area
fraction can be computed on a fixed heading-frame Cartesian grid: the first
three feature columns (d/D_NORM, cos, sin) come from the grid geometry and every
remaining column (speed, accel components, optional profile feats) is broadcast
from the sampled state. Minimising

    soft_area = mean_cells sigmoid((h - tta_cell) / beta)

can only push predicted times UP (out of the set), so the term is pure
area-shrinking pressure, opposed by the pinball asymmetry wherever the data
contains fast arrivals. Validated in scripts/experiment_arealoss.py; the
lam=0.01 operating point trades far-tail feature-conditioned coverage for a
~30% matched-coverage area gain at long horizons (see config.AREA_LAM notes).
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import AREA_BETA, AREA_GRID_HALF, AREA_GRID_STEP, AREA_HORIZONS, D_NORM


def heading_grid(device, offset=(0.0, 0.0), step: float = AREA_GRID_STEP,
                 half: float = AREA_GRID_HALF) -> torch.Tensor:
    """Heading-frame grid -> (d/D_NORM, cos, sin) columns, shape (G, 3).

    ``offset`` translates the whole grid (metres); the default is the fixed
    canonical grid, a random sub-cell offset gives the jittered grid of
    `stochastic_aux`.
    """
    ax = np.arange(-half, half + 1e-6, step)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    x = xx.ravel() + offset[0]
    y = yy.ravel() + offset[1]
    d = np.hypot(x, y)
    keep = d >= 2.0          # drop the degenerate near-origin cell(s)
    x, y, d = x[keep], y[keep], d[keep]
    cols = np.stack([d / D_NORM, x / d, y / d], axis=1).astype(np.float32)
    return torch.from_numpy(cols).to(device)


def stochastic_aux(rng: np.random.Generator, device, horizons,
                   step: float = AREA_GRID_STEP, half: float = AREA_GRID_HALF):
    """One Monte-Carlo draw of the soft-area geometry: a grid translated by a
    uniform sub-cell offset and ``len(horizons)`` horizons drawn uniformly in
    [min(horizons), max(horizons)].

    The fixed grid + fixed horizon list pin the area pressure to fixed radii
    and level sets, which the network traces as creases/walls in the TTA
    surface (steps at the 4 m cell radii, walls just below each penalised
    horizon). Re-drawing both every step keeps the expected loss -- the mean
    soft area over the horizon range -- but leaves no geometry to pin to.
    """
    off = rng.uniform(-step / 2, step / 2, size=2)
    grid_cols = heading_grid(device, offset=off, step=step, half=half)
    lo, hi = float(min(horizons)), float(max(horizons))
    hs = tuple(rng.uniform(lo, hi, size=len(horizons)).tolist())
    return grid_cols, hs


def soft_area(model, grid_cols: torch.Tensor, state_cols: torch.Tensor,
              horizons=AREA_HORIZONS, beta: float = AREA_BETA) -> torch.Tensor:
    """Mean soft reachable-area fraction over states and horizons.

    grid_cols: (G, 3) fixed geometry columns; state_cols: (S, in_dim - 3), the
    non-geometry feature columns of the sampled states (X[:, 3:]).
    """
    S = state_cols.size(0)
    G = grid_cols.size(0)
    in_dim = 3 + state_cols.size(1)
    Xg = torch.empty(S, G, in_dim, device=state_cols.device)
    Xg[:, :, :3] = grid_cols.unsqueeze(0)
    Xg[:, :, 3:] = state_cols.unsqueeze(1)
    tta = model(Xg.reshape(S * G, in_dim)).reshape(S, G)
    total = 0.0
    for h in horizons:
        total = total + torch.sigmoid((h - tta) / beta).mean()
    return total / len(horizons)
