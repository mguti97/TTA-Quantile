"""Reachable-area validation metric (Renkin et al., ICCS 2022, paper 381).

Ports their correctness/precision score for player *motion models* to our
*time-to-arrival* models. A TTA surface tau(.) is the dual of a reachable-area
model: the region a player can reach within a horizon Δt is the sublevel set

    R(Δt) = { x : tau_pred(x) <= Δt }.

So the paper's two halves map directly onto a TTA model:

  * correctness / ``hit_ratio`` -- fraction of trails whose actually-reached
    point ``x_t`` satisfies ``tau_pred(x_t) <= Δt``. This is exactly the
    lower-quantile *coverage* of the model.
  * precision -- ``n_correct / Σ area_correct`` where ``area`` is the pitch area
    of the sublevel set ``{x : tau_pred(x) <= Δt}`` (incorrect trails excluded,
    as in the paper, so a narrow-but-wrong region can't inflate precision).
  * ``score`` = 0 if ``hit_ratio < hit_ratio_min`` else ``precision``. We report
    ``score`` and its inverse ``mean_area`` (m^2); lower mean_area = a tighter
    model, comparable to the paper's Fig. 3 (their best: 71.7 m^2 at Δt = 1 s).

We threshold the inference grid rather than tracing a star-shaped polygon (the
paper's approach), which lets the reachable set be non-convex or disconnected --
a learned surface may well be either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FPS
from ..inference.heatmap import pitch_grid
from ..pipeline.cleaning import clean_track
from ..pipeline.features import transform


@dataclass
class Trails:
    """Exact-horizon trails reconstructed from a match, for one horizon Δt.

    All arrays are aligned and length N. ``state_pos``/``vel``/``acc`` describe
    the player at t0 (pitch frame); ``target_pos`` is where the player actually
    was Δt seconds later; ``frame``/``player``/``segment`` identify the trail so
    a held-out split can be applied downstream.
    """

    horizon: float
    state_pos: np.ndarray  # (N, 2)
    vel: np.ndarray        # (N, 2)
    acc: np.ndarray        # (N, 2)
    target_pos: np.ndarray  # (N, 2)
    frame: np.ndarray      # (N,) global frame index of t0
    player: np.ndarray     # (N,)
    segment: np.ndarray    # (N,)

    def __len__(self):
        return len(self.frame)

    def subsample(self, n: int, seed: int) -> "Trails":
        if n >= len(self):
            return self
        idx = np.random.default_rng(seed).choice(len(self), size=n, replace=False)
        idx.sort()
        return Trails(
            self.horizon, self.state_pos[idx], self.vel[idx], self.acc[idx],
            self.target_pos[idx], self.frame[idx], self.player[idx], self.segment[idx],
        )

    def select(self, mask: np.ndarray) -> "Trails":
        return Trails(
            self.horizon, self.state_pos[mask], self.vel[mask], self.acc[mask],
            self.target_pos[mask], self.frame[mask], self.player[mask], self.segment[mask],
        )


def build_trails(match, horizon_s: float) -> Trails:
    """Reconstruct every exact-Δt trail in the match for horizon ``horizon_s``.

    Mirrors ``pipeline.sampling`` but keeps the raw t0 state (needed to draw the
    full reachable area) and uses a single fixed offset so ``Δt`` is exact. No
    SUVAT noise filter is applied -- per the paper, outliers are absorbed by
    ``hit_ratio_min`` rather than removed.
    """
    off = int(round(horizon_s * FPS))
    cols = {k: [] for k in ("state_pos", "vel", "acc", "target_pos",
                            "frame", "player", "segment")}

    for seg_id, lo, hi in match.segment_slices():
        for p in range(match.P):
            in_field = match.in_field[lo:hi, p]
            if not np.any(in_field == 1):
                continue
            tr = clean_track(match.positions[lo:hi, p, :], in_field, player=p, segment=seg_id)
            if not tr.state_valid.any():
                continue
            si = np.where(tr.state_valid)[0]
            ti = si + off
            keep = ti < tr.L
            si, ti = si[keep], ti[keep]
            if si.size == 0:
                continue
            good = ~np.isnan(tr.pos[ti, 0]) & ~np.isnan(tr.pos[si, 0])
            si, ti = si[good], ti[good]
            if si.size == 0:
                continue
            cols["state_pos"].append(tr.pos[si])
            cols["vel"].append(tr.vel[si])
            cols["acc"].append(tr.acc[si])
            cols["target_pos"].append(tr.pos[ti])
            cols["frame"].append(lo + si)
            cols["player"].append(np.full(si.size, p))
            cols["segment"].append(np.full(si.size, seg_id))

    if not cols["frame"]:
        empty2 = np.empty((0, 2))
        empty1 = np.empty(0, dtype=int)
        return Trails(horizon_s, empty2, empty2, empty2, empty2, empty1, empty1, empty1)

    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return Trails(horizon_s, cat["state_pos"], cat["vel"], cat["acc"],
                  cat["target_pos"], cat["frame"], cat["player"], cat["segment"])


def _extra_slice(extra, s, e, repeat=1):
    """Slice per-trail extra columns to rows [s:e), repeating each ``repeat`` x.

    ``extra`` is an optional {name: (N,) array} of per-trail quantities a model
    needs alongside the geometry -- e.g. the per-player capacity columns
    ``vmax_pp``/``amax_pp`` the per-player motion models consume. They ride
    along the same repeat/tile pattern the states do, so a model sees one value
    per predicted row.
    """
    if not extra:
        return {}
    if repeat == 1:
        return {k: np.asarray(v)[s:e] for k, v in extra.items()}
    return {k: np.repeat(np.asarray(v)[s:e], repeat) for k, v in extra.items()}


def tta_to_targets(pred_fn, trails: Trails, chunk: int = 200_000,
                   extra: dict | None = None) -> np.ndarray:
    """Predicted arrival time from each trail's t0 state to its reached point.

    ``pred_fn`` maps a ``features.transform`` dict to a (M,) array of times.
    ``extra`` adds per-trail columns to that dict (see ``_extra_slice``).
    """
    N = len(trails)
    out = np.empty(N)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        tf = transform(trails.state_pos[s:e], trails.vel[s:e],
                       trails.acc[s:e], trails.target_pos[s:e])
        tf.update(_extra_slice(extra, s, e))
        out[s:e] = pred_fn(tf)
    return out


def reachable_areas(pred_fn, trails: Trails, res: float,
                    chunk_states: int = 64, extra: dict | None = None) -> np.ndarray:
    """Pitch area (m^2) of the predicted reachable set per trail.

    For each t0 state, predicts the arrival time to every grid cell and counts
    the cells reachable within ``trails.horizon``, times the cell area.
    """
    cells, _, _ = pitch_grid(res)
    G = cells.shape[0]
    cell_area = res * res
    N = len(trails)
    areas = np.empty(N)
    for s in range(0, N, chunk_states):
        e = min(s + chunk_states, N)
        cs = e - s
        sp = np.repeat(trails.state_pos[s:e], G, axis=0)
        vv = np.repeat(trails.vel[s:e], G, axis=0)
        aa = np.repeat(trails.acc[s:e], G, axis=0)
        cc = np.tile(cells, (cs, 1))
        tf = transform(sp, vv, aa, cc)
        tf.update(_extra_slice(extra, s, e, repeat=G))
        tta = np.asarray(pred_fn(tf)).reshape(cs, G)
        areas[s:e] = (tta <= trails.horizon).sum(axis=1) * cell_area
    return areas


def grid_tta(pred_fn, trails: Trails, res: float,
             chunk_states: int = 64, dtype=np.float32, extra: dict | None = None):
    """Predicted arrival time from each trail's t0 state to *every* grid cell.

    Returns ``(tta, cell_area)`` with ``tta`` of shape (N, G). Computing the
    grid once lets a caller threshold it at any level to trace a model's full
    coverage-vs-area trade-off without re-running inference.
    """
    cells, _, _ = pitch_grid(res)
    G = cells.shape[0]
    N = len(trails)
    out = np.empty((N, G), dtype=dtype)
    for s in range(0, N, chunk_states):
        e = min(s + chunk_states, N)
        cs = e - s
        sp = np.repeat(trails.state_pos[s:e], G, axis=0)
        vv = np.repeat(trails.vel[s:e], G, axis=0)
        aa = np.repeat(trails.acc[s:e], G, axis=0)
        cc = np.tile(cells, (cs, 1))
        tf = transform(sp, vv, aa, cc)
        tf.update(_extra_slice(extra, s, e, repeat=G))
        out[s:e] = np.asarray(pred_fn(tf)).reshape(cs, G)
    return out, res * res


def matched_coverage_area(pred_to_target: np.ndarray, grid: np.ndarray,
                          cell_area: float, coverage: float) -> dict:
    """Mean reachable area when each model is tuned to a common coverage.

    The reachable threshold is set to the ``coverage`` quantile of the model's
    own arrival-time predictions, so every model covers the same fraction of
    actually-reached points; we then compare pure area (precision) at equal
    correctness. Lower area is better.
    """
    thr = float(np.quantile(pred_to_target, coverage))
    correct = pred_to_target <= thr
    areas = (grid <= thr).sum(axis=1) * cell_area
    return {
        "coverage": coverage,
        "threshold_s": thr,
        "mean_area": float(areas[correct].mean()) if correct.any() else float("nan"),
    }


def score(pred_to_target: np.ndarray, areas: np.ndarray, horizon: float,
          hit_ratio_min: float) -> dict:
    """Combine correctness and precision into the paper's validation score.

    Returns a dict with the raw components so callers can report ``hit_ratio``
    and ``mean_area`` (= score^-1) alongside the gated ``score`` itself.
    """
    correct = pred_to_target <= horizon
    n = correct.size
    ncorrect = int(correct.sum())
    hit_ratio = ncorrect / n if n else float("nan")

    if ncorrect == 0:
        mean_area = float("nan")
        precision = 0.0
        r_eff = float("nan")
    else:
        sum_area = float(areas[correct].sum())
        mean_area = sum_area / ncorrect
        precision = ncorrect / sum_area  # == 1 / mean_area
        # Effective radius of an equal-area disk. Unlike mean_area (~t^2) this
        # scales ~linearly in Δt, so it stays comparable across horizons and
        # exposes whether models converge as Δt grows.
        r_eff = float(np.sqrt(mean_area / np.pi))

    gated = hit_ratio >= hit_ratio_min
    return {
        "horizon": horizon,
        "n": n,
        "n_correct": ncorrect,
        "hit_ratio": hit_ratio,
        "mean_area": mean_area,        # m^2, == score^-1 ; lower is better
        "r_eff": r_eff,                # m, == sqrt(mean_area/pi) ; cross-Δt comparable
        "precision": precision,        # 1/m^2
        "score": precision if gated else 0.0,
        "passes_gate": bool(gated),
    }
