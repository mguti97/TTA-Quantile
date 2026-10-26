"""SUVAT two-phase physics baseline (Section 8).

Used for (1) the noise-floor filter during preprocessing and (2) as the
benchmark at evaluation. Fully vectorised over numpy arrays.

Inputs are described in the player-heading frame:
    v     current speed (m/s), >= 0
    cos_t cosine of angle to target (theta); v*cos_t is the speed component
          toward the target
    d     distance to target (m), > 0
"""
from __future__ import annotations

import numpy as np

from ..config import SUVAT_A_MAX, SUVAT_V_MAX


def suvat_time(v, cos_t, d, a_max: float = SUVAT_A_MAX, v_max: float = SUVAT_V_MAX):
    """Minimum arrival time under the two-phase (redirect + sprint) model.

    Accepts scalars or numpy arrays; returns the same shape.
    """
    v = np.asarray(v, dtype=np.float64)
    cos_t = np.asarray(cos_t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    # --- Phase 1: redirect ------------------------------------------------
    v_toward = v * cos_t
    moving_away = v_toward < 0
    t_stop = np.where(moving_away, np.abs(v_toward) / a_max, 0.0)
    v0 = np.where(moving_away, 0.0, v_toward)  # useful momentum retained

    # --- Phase 2: sprint from v0 over distance d --------------------------
    d_accel = (v_max**2 - v0**2) / (2.0 * a_max)  # distance to reach v_max

    # branch a: already at/above v_max -> pure cruise
    t_cruise_only = d / v_max
    # branch b: never reaches v_max -> solve d = v0 t + 0.5 a t^2
    t_quad = (-v0 + np.sqrt(np.maximum(v0**2 + 2.0 * a_max * d, 0.0))) / a_max
    # branch c: accelerate to v_max, then cruise the remainder
    t_accel = (v_max - v0) / a_max
    t_accel_cruise = t_accel + (d - d_accel) / v_max

    t_sprint = np.where(
        v0 >= v_max,
        t_cruise_only,
        np.where(d_accel >= d, t_quad, t_accel_cruise),
    )

    return t_stop + t_sprint
