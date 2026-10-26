"""Ablation: does the passer's velocity help the ball-pass model?

The 4 pass features are [d, cosθ, sinθ, v_passer]. The passer's velocity enters
BOTH as the speed magnitude `v_passer` AND as the heading that defines the angle
θ (features.transform rotates the target into the passer's heading frame). So we
train a ladder of feature subsets, all with the identical recipe (Adam + τ=0.05
pinball + early stopping, fixed SEED), and compare on the held-out test split
against the analytic d/BALL_V_MAX baseline:

  distance-only [d]          -- genuinely velocity-free
  no-speed      [d,cosθ,sinθ] -- drops speed, keeps heading-relative geometry
  full          [d,cosθ,sinθ,v] -- the shipped model

    python -m scripts.ablate_passer_velocity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PASS_NPZ, PASS_OUTPUTS_DIR, PINBALL_TAU  # noqa: E402
from src.evaluation.metrics import DIST_BANDS, coverage, mae, rmse  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.train import save_model, train  # noqa: E402

VARIANTS = [
    ("distance-only [d]",          [0],          "mlp_pass_dist_only.pt"),
    ("no-speed [d,cosθ,sinθ]",      [0, 1, 2],    "mlp_pass_no_speed.pt"),
    ("full [d,cosθ,sinθ,v]",        [0, 1, 2, 3], "mlp_pass_full.pt"),
]


def pinball(pred, y):
    return pinball_loss(torch.from_numpy(np.asarray(pred, np.float32)),
                        torch.from_numpy(np.asarray(y, np.float32))).item()


def main():
    d = np.load(PASS_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    dist_te = d["meta_d"][te]
    t_ball = d["meta_t_ball"][te]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train {tr.sum():,} | val {va.sum():,} | test {te.sum():,} | device {device}")

    rows = []  # (name, pred_on_test)
    for name, cols, fname in VARIANTS:
        print(f"\n=== training {name}  (features {cols}) ===")
        Xc = X[:, cols]
        model, best_val, _ = train(Xc[tr], y[tr], Xc[va], y[va],
                                   device=device, in_dim=len(cols), verbose=False)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xc[te]).to(device)).cpu().numpy()
        save_model(model, PASS_OUTPUTS_DIR / fname)
        print(f"  best val pinball {best_val:.5f}  -> saved {fname}")
        rows.append((name, pred))

    rows.append(("baseline d/BALL_V_MAX", t_ball))

    nominal = 1.0 - PINBALL_TAU
    print(f"\n=== held-out test ({te.sum():,} passes) ===")
    print(f"{'model':28s}{'MAE':>8s}{'RMSE':>8s}{'pinball':>10s}{'coverage':>10s}")
    print("-" * 64)
    for name, p in rows:
        print(f"{name:28s}{mae(p, y[te]):>8.3f}{rmse(p, y[te]):>8.3f}"
              f"{pinball(p, y[te]):>10.5f}{coverage(p, y[te]):>10.3f}")
    print(f"(target coverage 1-tau = {nominal:.2f}; baseline is a lower bound)")

    print("\n=== MAE by distance band ===")
    hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}m" for lo, hi in DIST_BANDS)
    print(f"{'model':28s}{hdr}")
    for name, p in rows:
        cells = []
        for lo, hi in DIST_BANDS:
            mk = (dist_te >= lo) & (dist_te < hi)
            cells.append(f"{mae(p[mk], y[te][mk]):.3f}" if mk.any() else "  -  ")
        print(f"{name:28s}" + "    ".join(cells))


if __name__ == "__main__":
    main()
