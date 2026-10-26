"""Score TTA-surface smoothness of player-model checkpoints (debug/visualization).

For each checkpoint and speed, evaluates the model on a 1 m heading-frame grid
(canonical state: v along +x, zero accel, profile vmax = speed-matched default)
and reports the largest and mean |delta t| per 1 m radial step over 181
directions, excluding r < 2 m (the physical reversal step behind the player).
Walls (fixed {3,5} horizons) show up as 1-1.5 s jumps; creases pinned at the
4 m aux-grid radii as ~0.4 s jumps.

    python -m scripts.surface_smoothness outputs/ffpc_match/mlp_tau0.01.pt ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import V_NORM  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.features import transform  # noqa: E402

VMAX_DEFAULT = 9.0   # profile models: per-player max speed fed as 7th feature


def grid_tta(model, speed, half=40, vmax=VMAX_DEFAULT):
    xs = np.arange(-half, half + 1, 1.0)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    G = len(cells)
    X = transform(np.zeros((G, 2)), np.tile([speed, 0.0], (G, 1)),
                  np.zeros((G, 2)), cells)["X"]
    in_dim = model.net[0].in_features
    if in_dim == 7:
        X = np.concatenate([X, np.full((G, 1), vmax / V_NORM, np.float32)], axis=1)
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy().reshape(len(xs), len(xs))


def radial_jumps(t, half=40, r_min=2, r_max=40, n_dir=181):
    r = np.arange(r_min, r_max)
    worst, mean = 0.0, np.zeros(len(r) - 1)
    for a in np.linspace(-np.pi, np.pi, n_dir):
        xi = np.round(r * np.cos(a)).astype(int) + half
        yi = np.round(r * np.sin(a)).astype(int) + half
        dj = np.abs(np.diff(t[xi, yi]))
        worst = max(worst, dj.max())
        mean += dj
    return worst, mean / n_dir, r[:-1]


def load(path):
    sd = torch.load(path, map_location="cpu")
    m = TTAModel(in_dim=sd["net.0.weight"].shape[1])
    m.load_state_dict(sd)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    ap.add_argument("--profile", action="store_true", help="print the per-radius mean |dt|")
    args = ap.parse_args()
    print(f"{'checkpoint':44s} {'speed':>5s} {'max|dt|/m':>10s} {'mean|dt|/m':>11s}  (r>=2m)")
    for ck in args.ckpts:
        m = load(ck)
        for sp in args.speeds:
            worst, prof, r = radial_jumps(grid_tta(m, sp))
            print(f"{Path(ck).name:44s} {sp:5g} {worst:10.2f} {prof.mean():11.3f}")
            if args.profile:
                print("   " + " ".join(f"{rr}:{v:.2f}" for rr, v in zip(r, prof)))


if __name__ == "__main__":
    main()
