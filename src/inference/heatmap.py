"""Full-pitch time-to-arrival heatmaps (Section 9).

For one frame, every tracked player's kinematic state is broadcast against a
fixed pitch grid, transformed into the 6-feature representation, and pushed
through the MLP in a single batched forward pass.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import GRID_RES, PASS_N_FEATURES, PITCH_L, PITCH_W, PITCH_X0, PITCH_Y0
from ..pipeline.ball_baseline import ball_min_time
from ..pipeline.features import transform
from ..pipeline.suvat import suvat_time


def pitch_grid(res: float = GRID_RES):
    """Return (cells (G,2), nx, ny) of cell-centre coordinates over the pitch."""
    xs = np.arange(PITCH_X0 + res / 2, PITCH_X0 + PITCH_L, res)
    ys = np.arange(PITCH_Y0 + res / 2, PITCH_Y0 + PITCH_W, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return cells, len(xs), len(ys)


def features_for_state(pos, vel, acc, cells):
    """6-feature matrix for one player's state against all grid cells."""
    G = cells.shape[0]
    state_pos = np.broadcast_to(pos, (G, 2))
    vel_b = np.broadcast_to(vel, (G, 2))
    acc_b = np.broadcast_to(acc, (G, 2))
    return transform(state_pos, vel_b, acc_b, cells)


@torch.no_grad()
def heatmaps_for_frame(model, states, device, res: float = GRID_RES, return_suvat=False):
    """Predict per-player heatmaps for a frame.

    `states` is a list of (pos, vel, acc) tuples (one per tracked player), each a
    length-2 array. Returns (heatmaps, nx, ny) with heatmaps shape (n_players,
    nx, ny). If return_suvat, also returns the SUVAT baseline heatmaps.
    """
    cells, nx, ny = pitch_grid(res)
    G = cells.shape[0]
    feats = [features_for_state(p, v, a, cells) for (p, v, a) in states]
    X = np.concatenate([f["X"] for f in feats], axis=0)

    model.eval()
    pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    heat = pred.reshape(len(states), nx, ny)

    if return_suvat:
        suv = np.concatenate(
            [suvat_time(f["v"], f["cos_t"], f["d"]) for f in feats]
        ).reshape(len(states), nx, ny)
        return heat, suv, nx, ny
    return heat, nx, ny


@torch.no_grad()
def ball_arrival_heatmap(model, ball_pos, passer_vel, device, res: float = GRID_RES,
                         return_baseline: bool = False):
    """Ball-pipeline analog of `heatmaps_for_frame` for a single situation.

    Given the ball's release position and the PASSER's velocity (the heading that
    orients the geometry), predict the minimum pass-arrival time (learned τ=0.05
    lower envelope) to every cell of the pitch grid. The ball release position is
    the "state", each grid cell is the "target"; acceleration is unused and ball
    velocity is excluded by design (only the first PASS_N_FEATURES columns are fed).

    Returns (heat, nx, ny), or (heat, baseline, nx, ny) with the analytic
    d/BALL_V_MAX straight-line bound if `return_baseline`. Heatmaps are (nx, ny)
    in pitch-grid orientation (x along axis-0), in seconds.
    """
    cells, nx, ny = pitch_grid(res)
    feats = features_for_state(np.asarray(ball_pos, float), np.asarray(passer_vel, float),
                               np.zeros(2), cells)
    X = feats["X"][:, :PASS_N_FEATURES]

    model.eval()
    heat = model(torch.from_numpy(X).to(device)).cpu().numpy().reshape(nx, ny)

    if return_baseline:
        base = ball_min_time(feats["d"]).reshape(nx, ny)
        return heat, base, nx, ny
    return heat, nx, ny
