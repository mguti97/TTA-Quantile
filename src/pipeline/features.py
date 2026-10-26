"""Coordinate transform + feature extraction (Sections 3 & 6.4), vectorised.

Given a player's kinematic state and a target position (both in pitch frame),
produce the 6 normalised input features the model consumes:

    [ d/120, cosθ, sinθ, v/12, ax'/10, ay'/10 ]

The transform translates the player to the origin and rotates so the heading
(velocity vector) aligns with +x. By construction vx' = v, vy' = 0.
"""
from __future__ import annotations

import numpy as np

from ..config import A_NORM, D_NORM, V_NORM


def transform(state_pos, vel, acc, target_pos):
    """Vectorised transform for N samples.

    All inputs are (N, 2) except as noted. Returns a dict of (N,) arrays with
    both the raw geometric quantities (d, v, theta, ax', ay') and the normalised
    6-feature matrix `X` of shape (N, 6).
    """
    state_pos = np.asarray(state_pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    dx = target_pos[:, 0] - state_pos[:, 0]
    dy = target_pos[:, 1] - state_pos[:, 1]

    vx, vy = vel[:, 0], vel[:, 1]
    v = np.hypot(vx, vy)                       # current speed; vx' == v
    phi = np.arctan2(vy, vx)                   # heading angle
    cphi, sphi = np.cos(phi), np.sin(phi)

    # rotate target displacement by -phi into the heading frame
    d = np.hypot(dx, dy)
    rx = dx * cphi + dy * sphi
    ry = -dx * sphi + dy * cphi
    safe_d = np.where(d > 0, d, 1.0)
    cos_t = rx / safe_d
    sin_t = ry / safe_d

    # rotate acceleration by -phi -> longitudinal / lateral components
    ax_p = acc[:, 0] * cphi + acc[:, 1] * sphi
    ay_p = -acc[:, 0] * sphi + acc[:, 1] * cphi

    X = np.stack(
        [
            d / D_NORM,
            cos_t,
            sin_t,
            v / V_NORM,
            ax_p / A_NORM,
            ay_p / A_NORM,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "X": X,
        "d": d,
        "v": v,
        "cos_t": cos_t,
        "sin_t": sin_t,
        "theta_deg": np.degrees(np.arctan2(sin_t, cos_t)),
        "ax_p": ax_p,
        "ay_p": ay_p,
    }
