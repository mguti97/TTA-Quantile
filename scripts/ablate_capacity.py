"""Capacity-scaling ablation: does a bigger MLP buy accuracy?

Trains the baseline 6->64->64->1 MLP (4,673 params) and a "big" 6->256->256->
256->1 MLP (133,633 params, 3 hidden layers) at the same tau, both on the full
train split, evaluated on the same held-out test split. Reports overall
MAE/RMSE/pinball/coverage, stratified MAE (distance/speed/angle) and surface
roughness on one frame -- the full picture needed to tell "no gain" from
"no gain but smoother/rougher".

    python -m scripts.ablate_capacity                # tau = config.PINBALL_TAU
    python -m scripts.ablate_capacity --tau 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    AREA_LAM, DATASET_NPZ, GRID_RES, OUTPUTS_DIR, PINBALL_TAU,
)
from src.data_loader import load_match  # noqa: E402
from src.evaluation.metrics import DIST_BANDS, SPEED_BANDS, ANGLE_BANDS, coverage, mae, rmse  # noqa: E402
from src.inference.heatmap import pitch_grid  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.train import save_model, train  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.features import transform  # noqa: E402

BIG_HIDDEN = [256, 256, 256]
FRAME = 9300


def pinball(pred, y, tau):
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def roughness(Z):
    lap = np.abs(4 * Z[1:-1, 1:-1] - Z[:-2, 1:-1] - Z[2:, 1:-1] - Z[1:-1, :-2] - Z[1:-1, 2:])
    return float(lap.mean())


@torch.no_grad()
def predict(model, X, device, chunk=1_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        out[s:e] = model(torch.from_numpy(X[s:e]).to(device)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    ap.add_argument("--area-lam", type=float, default=AREA_LAM,
                    help=f"area-aware aux weight (default {AREA_LAM:g}; 0 disables)")
    args = ap.parse_args()
    tau = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    print(f"train {int(tr.sum()):,} | val {int(va.sum()):,} | test {int(te.sum()):,}"
          f"   device {device}   tau {tau:g}   area_lam {args.area_lam:g}")

    models = {}
    print("\n[train] baseline (6->64->64->1, 4,673 params)")
    m_base, best_base, _ = train(X[tr], y[tr], X[va], y[va], device=device, tau=tau,
                                 area_lam=args.area_lam)
    models["MLP baseline"] = m_base
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    save_model(m_base, OUTPUTS_DIR / f"mlp_capacity_baseline{suffix}.pt")

    print(f"\n[train] big (6->256->256->256->1, 133,633 params)")
    m_big, best_big, _ = train(X[tr], y[tr], X[va], y[va], device=device, tau=tau,
                               hidden=BIG_HIDDEN, area_lam=args.area_lam)
    models["MLP big"] = m_big
    save_model(m_big, OUTPUTS_DIR / f"mlp_capacity_big{suffix}.pt")

    y_te = y[te]
    preds = {name: predict(m, X[te], device) for name, m in models.items()}

    nominal = 1.0 - tau
    print(f"\n=== OVERALL (test {len(y_te):,}; target coverage {nominal:.3f}) ===")
    print(f"{'model':14s} {'MAE':>7s} {'RMSE':>7s} {'pinball':>9s} {'coverage':>9s}")
    for name, p in preds.items():
        print(f"{name:14s} {mae(p, y_te):7.3f} {rmse(p, y_te):7.3f} "
              f"{pinball(p, y_te, tau):9.5f} {coverage(p, y_te):9.4f}")

    dist = d["meta_d"][te]
    speed = d["meta_v"][te]
    theta = np.abs(d["meta_theta_deg"][te])
    for title, vals, bands in (("distance", dist, DIST_BANDS),
                               ("speed", speed, SPEED_BANDS),
                               ("angle", theta, ANGLE_BANDS)):
        print(f"\n=== stratified MAE by {title} ===")
        hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}" for lo, hi in bands)
        print(f"{'model':14s} {hdr}")
        for name, p in preds.items():
            cells = []
            for lo, hi in bands:
                m = (vals >= lo) & (vals < hi)
                cells.append(f"{mae(p[m], y_te[m]):6.3f}" if m.any() else "  -   ")
            print(f"{name:14s} " + "  ".join(cells))

    # ---- surface roughness on one frame --------------------------------------
    match = load_match()
    states = []
    for seg_id, lo, hi in match.segment_slices():
        if not (lo <= FRAME < hi):
            continue
        for p in range(match.P):
            if match.in_field[FRAME, p] != 1:
                continue
            trk = clean_track(match.positions[lo:hi, p, :], match.in_field[lo:hi, p], p, seg_id)
            li = FRAME - lo
            if li < trk.L and trk.state_valid[li] and not np.isnan(trk.pos[li, 0]):
                states.append((trk.pos[li], trk.vel[li], trk.acc[li]))
        break

    cells, nx, ny = pitch_grid(GRID_RES)
    G = cells.shape[0]
    rough = {name: [] for name in models}
    for pos, vel, acc in states:
        tf = transform(np.broadcast_to(pos, (G, 2)), np.broadcast_to(vel, (G, 2)),
                       np.broadcast_to(acc, (G, 2)), cells)
        for name, m in models.items():
            with torch.no_grad():
                s = m(torch.from_numpy(tf["X"]).to(device)).cpu().numpy().reshape(nx, ny)
            rough[name].append(roughness(s))

    print(f"\n=== surface roughness (mean |Laplacian|, frame {FRAME}, {len(states)} players) ===")
    for name in models:
        print(f"  {name:14s} {np.mean(rough[name]):.4f}")


if __name__ == "__main__":
    main()
