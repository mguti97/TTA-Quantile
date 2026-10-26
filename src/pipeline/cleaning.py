"""Track cleaning: gap-aware segmentation, smoothing, derivatives, clipping.

The dominant data reality here is occlusion: ~43% of in-field frames are NaN.
So the core idea is to split each player's timeline (within a segment/half) into
*continuous runs* -- maximal stretches of consecutive, tracked frames with no
implausible jumps -- and only ever smooth / differentiate *inside* a run. We
never differentiate across a gap (Section 6.1).

Output per (player, segment) is a `CleanTrack`: arrays aligned to the segment's
local frame index, NaN where the player is untracked or in a too-short run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from ..config import (
    A_CLIP,
    DT,
    DV_CLIP,
    JUMP_BREAK_M,
    MIN_RUN_LEN,
    SAVGOL_POLY,
    SAVGOL_WINDOW,
    V_CLIP,
)


@dataclass
class CleanTrack:
    player: int
    segment: int
    pos: np.ndarray        # (L, 2) smoothed position, NaN outside usable runs
    vel: np.ndarray        # (L, 2) m/s
    acc: np.ndarray        # (L, 2) m/s^2
    state_valid: np.ndarray  # (L,) bool: usable as a *state* (passed clipping)

    @property
    def L(self) -> int:
        return self.pos.shape[0]


def _find_runs(valid: np.ndarray, pos: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index ranges of continuous, jump-free runs.

    A run breaks on a missing frame OR a single-frame displacement exceeding
    JUMP_BREAK_M (gross discontinuity / ID swap).
    """
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(len(valid)):
        if not valid[i]:
            if start is not None:
                runs.append((start, i))
                start = None
            continue
        if start is None:
            start = i
        elif np.hypot(*(pos[i] - pos[i - 1])) > JUMP_BREAK_M:
            # jump: close the current run, open a new one at i
            runs.append((start, i))
            start = i
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def clean_track(pos_raw: np.ndarray, in_field: np.ndarray, player: int, segment: int) -> CleanTrack:
    """Clean one player's trajectory over one segment.

    `pos_raw` is (L, 2) raw positions (NaN where untracked); `in_field` is (L,)
    1 when the player is on the pitch.
    """
    L = pos_raw.shape[0]
    valid_raw = (in_field == 1) & ~np.isnan(pos_raw[:, 0])

    pos = np.full((L, 2), np.nan)
    vel = np.full((L, 2), np.nan)
    acc = np.full((L, 2), np.nan)
    state_valid = np.zeros(L, dtype=bool)

    for a, b in _find_runs(valid_raw, pos_raw):
        n = b - a
        if n < MIN_RUN_LEN:
            continue  # too short to smooth/differentiate reliably
        win = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
        if win <= SAVGOL_POLY:
            continue
        seg = pos_raw[a:b]

        # 1. smooth positions (Savitzky-Golay preserves sprint peaks)
        sx = savgol_filter(seg[:, 0], win, SAVGOL_POLY)
        sy = savgol_filter(seg[:, 1], win, SAVGOL_POLY)
        # 2. derivatives via finite differences on smoothed positions
        vx = np.gradient(sx, DT)
        vy = np.gradient(sy, DT)
        ax = np.gradient(vx, DT)
        ay = np.gradient(vy, DT)

        pos[a:b, 0], pos[a:b, 1] = sx, sy
        vel[a:b, 0], vel[a:b, 1] = vx, vy
        acc[a:b, 0], acc[a:b, 1] = ax, ay

        # 3. hard clipping -> mark which frames are usable as a state
        speed = np.hypot(vx, vy)
        accel = np.hypot(ax, ay)
        dv = np.empty(n)
        dv[0] = 0.0
        dv[1:] = np.hypot(np.diff(vx), np.diff(vy))
        ok = (speed <= V_CLIP) & (accel <= A_CLIP) & (dv <= DV_CLIP)
        state_valid[a:b] = ok

    return CleanTrack(player, segment, pos, vel, acc, state_valid)
