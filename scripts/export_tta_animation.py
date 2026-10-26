"""Export data for the trajectory + arrival-time-grid animation (myfigures/tta_animation.html).

Picks every valid state in one held-out PFF game whose speed is within
``SPEED +- BAND`` m/s, follows the cleaned track for up to ``HORIZON_S`` seconds,
expresses the future in the player's heading frame (player at origin, velocity
along +x -- Sec. 3 of the paper), and dumps a subsample of those trajectories
together with the MLP's (default: area-loss smooth checkpoint) predicted arrival time on a heading-frame grid
for the canonical state (v = SPEED along +x, zero acceleration), plus the empirical
tau-percentile of observed arrival time per cell over ALL samples at that speed.

    python -m scripts.export_tta_animation [--speed 5] [--n 300] [--out myfigures/tta_animation_data.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.config import DATASET_NPZ, D_NORM, FPS, OUTPUTS_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from src.ffpc import load_ffpc_match
from src.model.mlp import TTAModel
from src.pipeline.cleaning import clean_track
from src.pipeline.features import transform

TEST_GAME = "3813"      # held-out (split==2) game of the canonical ffpc_match build
HORIZON_S = 8.0         # training window upper bound (config.HORIZON_MAX_S)
GRID_HALF = 40          # metres each side of the player, 1 m cells (config.GRID_RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    ap.add_argument("--band", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=0, help="trajectories to keep (0 = all candidates)")
    ap.add_argument("--stride", type=int, default=6, help="keep every Nth frame of each trail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="mlp_arealoss_lam0.01_smooth.pt",
                    help="checkpoint in OUTPUTS_DIR (6-feature base model); the canonical "
                         "{3,5}-horizon area loss has walls at t=3/5 s -- the *_smooth.pt "
                         "checkpoints give a continuous surface (visualisation only)")
    ap.add_argument("--tau", type=float, default=0.01, help="empirical percentile of arrival time per cell")
    ap.add_argument("--env-bins", type=int, default=36, help="angular sectors of the reach envelope")
    ap.add_argument("--env-pct", type=float, default=0.99, help="percentile of per-trail max radius per sector")
    ap.add_argument("--min-count", type=int, default=100, help="samples a cell needs for a percentile")
    ap.add_argument("--out", default="myfigures/tta_animation_data.json")
    args = ap.parse_args()

    match = load_ffpc_match(TEST_GAME)
    model = TTAModel()
    model.load_state_dict(torch.load(OUTPUTS_DIR / args.model, map_location=DEVICE))
    model.to(DEVICE).eval()
    samples = load_samples()
    out = dict(game=TEST_GAME, tau=args.tau, n_samples_total=int(samples["v"].size), fps=FPS, horizon_s=HORIZON_S, band=args.band,
               model=args.model, grid=dict(x0=-GRID_HALF, res=1.0, n=2 * GRID_HALF + 1),
               speeds={})
    for speed in args.speeds:
        out["speeds"][f"{speed:g}"] = export_speed(match, model, speed, args)
        emp, cnt, n_used = empirical_grid(samples, speed, args)
        out["speeds"][f"{speed:g}"].update(
            emp=[[None if np.isnan(v) else round(float(v), 3) for v in row] for row in emp],
            emp_n_samples=int(n_used), emp_min_count=args.min_count)
        print(f"  {speed:g} m/s: {n_used} samples -> percentile grid on {int((cnt >= args.min_count).sum())} cells")
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}")


def load_samples():
    """Every (state, target, arrival time) sample of the canonical build, heading frame."""
    z = np.load(DATASET_NPZ)
    X = z["X"]
    return dict(v=z["meta_v"], d=X[:, 0] * D_NORM, cos=X[:, 1], sin=X[:, 2], y=z["y"])


def empirical_grid(samples, speed, args):
    """Per-cell tau-percentile of the observed arrival time over ALL samples at this speed.

    Each sample's reached point is (d cos th, d sin th) in the heading frame; cells are
    the same 1 m grid as the model surface. Cells with fewer than min_count samples -> NaN.
    """
    m = np.abs(samples["v"] - speed) <= args.band
    d, c, s_, y = samples["d"][m], samples["cos"][m], samples["sin"][m], samples["y"][m]
    n = 2 * GRID_HALF + 1
    i = np.floor(d * c + 0.5).astype(np.int64) + GRID_HALF   # cell centre at integer metres
    j = np.floor(d * s_ + 0.5).astype(np.int64) + GRID_HALF
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    cell, y = i[ok] * n + j[ok], y[ok]
    order = np.lexsort((y, cell))
    cell, y = cell[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cell)) + 1]
    counts = np.diff(np.r_[starts, cell.size])
    grid = np.full(n * n, np.nan)
    cnt = np.zeros(n * n, dtype=np.int64)
    k = np.floor(args.tau * (counts - 1)).astype(np.int64)        # lower-interpolation percentile
    good = counts >= args.min_count
    grid[cell[starts[good]]] = y[starts[good] + k[good]]
    cnt[cell[starts]] = counts
    return grid.reshape(n, n), cnt.reshape(n, n), int(m.sum())


def export_speed(match, model, speed, args):
    off = int(round(HORIZON_S * FPS))
    cands = []  # (dx, dy, t) arrays in heading frame, plus speed, acc
    for seg_id, lo, hi in match.segment_slices():
        for p in range(match.P):
            infield = match.in_field[lo:hi, p]
            if not np.any(infield == 1):
                continue
            tr = clean_track(match.positions[lo:hi, p, :], infield, p, seg_id)
            if not tr.state_valid.any():
                continue
            spd = np.hypot(tr.vel[:, 0], tr.vel[:, 1])
            ok = tr.state_valid & (np.abs(spd - speed) <= args.band)
            last_end = -10**9
            for si in np.where(ok)[0]:
                if si < last_end:          # one trail per non-overlapping window
                    continue
                fut = tr.pos[si:si + off + 1]
                good = ~np.isnan(fut[:, 0])
                if good.sum() < int(2 * FPS):   # need at least 2 s of future
                    continue
                n = int(np.argmin(good)) if not good.all() else len(good)
                fut = fut[:n]
                phi = np.arctan2(tr.vel[si, 1], tr.vel[si, 0])
                c, s = np.cos(phi), np.sin(phi)
                dx, dy = fut[:, 0] - fut[0, 0], fut[:, 1] - fut[0, 1]
                rx, ry = dx * c + dy * s, -dx * s + dy * c
                t = np.arange(n) / FPS
                cands.append(dict(x=rx, y=ry, t=t, v=float(spd[si]),
                                  player=int(p), frame=int(lo + si)))
                last_end = si + n
    print(f"{len(cands)} candidate trails at {speed}+-{args.band} m/s in game {TEST_GAME}")

    # Angular reach envelope over ALL candidate trails (not just the drawn subsample):
    # per sector, the env_pct percentile of each trail's maximum radius inside that sector.
    nb = args.env_bins
    per_bin = [[] for _ in range(nb)]
    for c in cands:
        r = np.hypot(c["x"], c["y"])
        a = np.arctan2(c["y"], c["x"]); a[a < 0] += 2 * np.pi
        b = np.minimum(nb - 1, np.floor(a / (2 * np.pi) * nb).astype(int))
        best = np.full(nb, -1.0)
        np.maximum.at(best, b[r > 0], r[r > 0])
        for k in np.flatnonzero(best >= 0):
            per_bin[k].append(best[k])
    env_r = []
    for rs in per_bin:
        if not rs:
            env_r.append(None); continue
        rs = np.sort(rs)
        env_r.append(round(float(rs[min(len(rs) - 1, max(0, int(np.ceil(args.env_pct * len(rs))) - 1))]), 2))
    envelope = dict(bins=nb, pct=args.env_pct, n=len(cands), r=env_r)

    rng = np.random.default_rng(args.seed)
    keep = (np.arange(len(cands)) if args.n <= 0
            else rng.choice(len(cands), size=min(args.n, len(cands)), replace=False))
    trails = []   # x/y only, every `stride`th frame; sample k is at t = k * stride / fps
    for i in keep:
        c = cands[i]
        sl = slice(0, None, args.stride)
        trails.append(dict(x=np.round(c["x"][sl], 2).tolist(), y=np.round(c["y"][sl], 2).tolist()))

    # --- MLP arrival-time grid for the canonical state ---------------------
    xs = np.arange(-GRID_HALF, GRID_HALF + 1, 1.0)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    G = len(cells)
    tf = transform(np.zeros((G, 2)), np.tile([speed, 0.0], (G, 1)),
                   np.zeros((G, 2)), cells)
    with torch.no_grad():
        tta = model(torch.from_numpy(tf["X"]).to(DEVICE)).cpu().numpy().reshape(len(xs), len(xs))

    print(f"  {speed:g} m/s: {len(trails)} trails kept, grid tta {tta.min():.2f}-{tta.max():.2f} s")
    return dict(speed=speed, n_candidates=len(cands), trails=trails, trail_dt=args.stride / FPS,
                envelope=envelope,
                tta=np.round(tta, 3).tolist())   # tta[i][j] at x=xs[i], y=xs[j]


if __name__ == "__main__":
    main()
