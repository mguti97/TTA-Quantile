"""Dense sample extraction + label quality filter (Sections 6.2 & 6.3).

For every valid state frame of every player we emit one sample per future
horizon (0.25..8 s), whose target is the player's *real* smoothed position at that
horizon and whose label is the horizon itself. We then apply the SUVAT noise
floor: any sample whose observed Δt is faster than physically possible is a
tracking artifact and is discarded -- critical because the τ=0.05 pinball loss
learns from the bottom of the distribution.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    FPS,
    HORIZON_MAX_F,
    HORIZON_MIN_F,
    HORIZON_MODE,
    HORIZONS_F,
    N_RANDOM_HORIZONS,
    SEED,
    STATE_STRIDE,
    SUVAT_FILTER,
)
from .cleaning import CleanTrack
from .features import transform
from .suvat import suvat_time


def _horizon_offsets(state_idx, track):
    """Yield (frame_offset_array, label_array) pairs to sample for each state.

    In "fixed" mode every state gets the same integer-second offsets. In "random"
    mode each state draws N_RANDOM_HORIZONS continuous offsets, so labels span the
    range densely instead of snapping to 1/2/3/4/5 s.
    """
    if HORIZON_MODE == "fixed":
        for kf in HORIZONS_F:
            yield np.full(state_idx.size, kf), np.full(state_idx.size, kf / FPS)
        return
    # deterministic per-track rng so rebuilds are reproducible
    rng = np.random.default_rng(SEED + 1009 * int(track.player) + int(track.segment))
    for _ in range(N_RANDOM_HORIZONS):
        offs = rng.integers(HORIZON_MIN_F, HORIZON_MAX_F + 1, size=state_idx.size)
        yield offs, offs / FPS


def samples_from_track(track: CleanTrack, global_frame0: int):
    """Extract raw (state, target, label) tuples for one CleanTrack.

    `global_frame0` is the global frame index of local index 0, used so each
    sample carries an absolute frame (needed for the time-block split).

    Returns a dict of column arrays (empty arrays if the track yields nothing).
    """
    L = track.L
    state_idx_all = np.where(track.state_valid)[0]
    if STATE_STRIDE > 1:
        # keep evenly time-spaced states (modulo on the segment-local frame
        # index, so spacing is constant regardless of where gaps fall)
        state_idx_all = state_idx_all[state_idx_all % STATE_STRIDE == 0]

    cols = {k: [] for k in ("state_pos", "vel", "acc", "target_pos", "label",
                            "frame", "player", "segment", "horizon")}
    if state_idx_all.size == 0:
        return _empty()

    pos_ok = ~np.isnan(track.pos[:, 0])  # target frame must have a smoothed pos
    for offs, labels in _horizon_offsets(state_idx_all, track):
        tgt_idx = state_idx_all + offs
        keep = (tgt_idx < L)
        si = state_idx_all[keep]
        ti = tgt_idx[keep]
        lab = labels[keep]
        good = pos_ok[ti]
        si, ti, lab = si[good], ti[good], lab[good]
        if si.size == 0:
            continue
        cols["state_pos"].append(track.pos[si])
        cols["vel"].append(track.vel[si])
        cols["acc"].append(track.acc[si])
        cols["target_pos"].append(track.pos[ti])
        cols["label"].append(lab.astype(float))
        cols["frame"].append(global_frame0 + si)
        cols["player"].append(np.full(si.size, track.player))
        cols["segment"].append(np.full(si.size, track.segment))
        cols["horizon"].append(lab.astype(np.float32))

    if not cols["label"]:
        return _empty()
    return {k: np.concatenate(v) for k, v in cols.items()}


def _empty():
    return {
        "state_pos": np.empty((0, 2)), "vel": np.empty((0, 2)),
        "acc": np.empty((0, 2)), "target_pos": np.empty((0, 2)),
        "label": np.empty(0), "frame": np.empty(0, dtype=int),
        "player": np.empty(0, dtype=int), "segment": np.empty(0, dtype=int),
        "horizon": np.empty(0, dtype=int),
    }


def build_features_and_filter(raw: dict, player_vmax: float | None = None,
                              player_amax: float | None = None):
    """Apply the coordinate transform and the SUVAT noise-floor filter.

    Returns (X, y, meta) where meta holds per-sample diagnostics for stratified
    evaluation, plus the SUVAT baseline prediction for each surviving sample.
    `player_vmax` / `player_amax`, when given, are this track's player-specific
    max speed (m/s) / max acceleration (m/s^2); each is broadcast onto every
    surviving sample as the `vmax` / `amax` meta column so the player-profile
    model can consume it as an extra feature.
    """
    tf = transform(raw["state_pos"], raw["vel"], raw["acc"], raw["target_pos"])
    t_suvat = suvat_time(tf["v"], tf["cos_t"], tf["d"])

    # Section 6.3: discard physically-impossible observations (and d==0 degenerates).
    # config.SUVAT_FILTER == "off" keeps the physically-impossible ones (the
    # reviewer sensitivity variant); the d==0 degenerate guard always applies.
    below_suvat = raw["label"] < t_suvat
    valid_d = tf["d"] > 1e-6
    if SUVAT_FILTER == "on":
        keep = (~below_suvat) & valid_d
    else:
        keep = valid_d
    # counted in both modes so the build log always reports what the physics
    # model *would* have discarded
    n_below = int((below_suvat & valid_d).sum())

    X = tf["X"][keep]
    y = raw["label"][keep].astype(np.float32)
    n_out = int(keep.sum())
    meta = {
        "frame": raw["frame"][keep],
        "player": raw["player"][keep],
        "segment": raw["segment"][keep],
        "horizon": raw["horizon"][keep],
        "d": tf["d"][keep].astype(np.float32),
        "v": tf["v"][keep].astype(np.float32),
        "theta_deg": tf["theta_deg"][keep].astype(np.float32),
        "t_suvat": t_suvat[keep].astype(np.float32),
    }
    if player_vmax is not None:
        meta["vmax"] = np.full(n_out, player_vmax, dtype=np.float32)
    if player_amax is not None:
        meta["amax"] = np.full(n_out, player_amax, dtype=np.float32)
    n_in = raw["label"].size
    return X, y, meta, (n_in, n_out, n_below)
