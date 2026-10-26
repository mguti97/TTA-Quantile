"""Pass-event sample extraction + label quality filter (ball-pipeline analog of
sampling.py).

Each successful pass yields exactly ONE sample (unlike the player pipeline, which
fans every state out over many horizons): the ball goes from its release position
to its reception position, and the label is the observed flight time. The τ=0.05
pinball loss then learns the lower envelope -- the *minimum* arrival time for a
pass of that geometry and passer state.

The feature transform is reused verbatim from the player pipeline: the ball's
release position is the "state", the reception position is the "target", and the
PASSER's velocity supplies the heading. We keep only the first four columns
([d, cosθ, sinθ, v]) -- acceleration is dropped, and ball velocity is never used.
"""
from __future__ import annotations

import numpy as np

from ..config import PASS_D_MIN, PASS_DT, PASS_N_FEATURES, PASS_T_MAX, PASS_T_MIN
from .ball_baseline import ball_min_time
from .features import transform


def _empty():
    return {
        "state_pos": np.empty((0, 2)), "vel": np.empty((0, 2)),
        "acc": np.empty((0, 2)), "target_pos": np.empty((0, 2)),
        "label": np.empty(0), "frame": np.empty(0, dtype=int),
        "passer": np.empty(0, dtype=int), "segment": np.empty(0, dtype=int),
    }


def samples_from_passes(passes, ball, posG, velG, accG, validG):
    """Build raw (state, target, label) columns from one match's pass events.

    `ball` is (T,2); `posG/velG/accG` are (T,P,2) cleaned per-frame player
    kinematics; `validG` is (T,P) bool (passer usable as a state at that frame).
    A pass survives only if the ball is visible at both endpoints and the passer
    has a valid cleaned velocity at release.
    """
    if not passes:
        return _empty()

    rel = np.array([p.release_g for p in passes], dtype=np.int64)
    rec = np.array([p.reception_g for p in passes], dtype=np.int64)
    pas = np.array([p.passer_idx for p in passes], dtype=np.int64)
    seg = np.array([p.period for p in passes], dtype=np.int64)

    T = ball.shape[0]
    ok = (pas >= 0) & (rel >= 0) & (rec > rel) & (rec < T)
    if not ok.any():
        return _empty()
    rel, rec, pas, seg = rel[ok], rec[ok], pas[ok], seg[ok]

    state_pos = ball[rel]
    target_pos = ball[rec]
    vel = velG[rel, pas]
    acc = accG[rel, pas]
    label = (rec - rel) * PASS_DT

    keep = (
        validG[rel, pas]
        & np.isfinite(state_pos).all(1)
        & np.isfinite(target_pos).all(1)
        & np.isfinite(vel).all(1)
        & np.isfinite(acc).all(1)
    )
    if not keep.any():
        return _empty()

    return {
        "state_pos": state_pos[keep],
        "vel": vel[keep],
        "acc": acc[keep],
        "target_pos": target_pos[keep],
        "label": label[keep].astype(float),
        "frame": rel[keep],
        "passer": pas[keep],
        "segment": seg[keep],
    }


def build_pass_features_and_filter(raw: dict):
    """Apply the transform and the ball noise-floor / sanity filter.

    Returns (X, y, meta, (n_in, n_out)). X has PASS_N_FEATURES columns:
    [d/D_NORM, cosθ, sinθ, v_passer/V_NORM].
    """
    tf = transform(raw["state_pos"], raw["vel"], raw["acc"], raw["target_pos"])
    t_ball = ball_min_time(tf["d"])

    keep = (
        (raw["label"] >= t_ball)            # not faster than physically possible
        & (raw["label"] >= PASS_T_MIN)
        & (raw["label"] <= PASS_T_MAX)
        & (tf["d"] >= PASS_D_MIN)
    )

    X = tf["X"][keep][:, :PASS_N_FEATURES]
    y = raw["label"][keep].astype(np.float32)
    meta = {
        "frame": raw["frame"][keep],
        "passer": raw["passer"][keep],
        "segment": raw["segment"][keep],
        "d": tf["d"][keep].astype(np.float32),
        "v": tf["v"][keep].astype(np.float32),          # passer speed at release
        "theta_deg": tf["theta_deg"][keep].astype(np.float32),
        "t_ball": t_ball[keep].astype(np.float32),       # straight-line baseline
    }
    return X, y, meta, (int(raw["label"].size), int(keep.sum()))
