"""Straight-line ball baseline for the pass pipeline (analog of suvat.py).

A pass cannot travel faster than a maximum ball speed, so the fastest possible
arrival over a straight-line distance ``d`` is simply ``d / v_max``. This serves
the same dual role SUVAT plays in the player pipeline:

  1. noise floor -- any observed travel time below this is a mislabeled
     release/reception pairing and is discarded during preprocessing;
  2. physics benchmark -- the analytic lower bound the learned τ=0.05 model is
     compared against at evaluation.

We deliberately do NOT model release speed or aerodynamic drag here (v1): the
ball's own velocity is excluded by design, so this is a pure geometric bound.
"""
from __future__ import annotations

import numpy as np

from ..config import BALL_V_MAX


def ball_min_time(d, v_max: float = BALL_V_MAX):
    """Minimum straight-line ball travel time over distance ``d`` (m) -> seconds.

    Accepts scalars or numpy arrays; returns the same shape.
    """
    d = np.asarray(d, dtype=np.float64)
    return d / v_max
