"""Unified test-split comparison of the player-profile MLP vs the base model.

Evaluates on the SAME held-out test split every row shares:
  SUVAT, MLP base (6 feat), MLP+vmax.

    python -m scripts.compare_profile_feats                # tau = config.PINBALL_TAU
    python -m scripts.compare_profile_feats --tau 0.01     # loads *_tau0.01.pt variants
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATASET_NPZ, MODEL_PT, OUTPUTS_DIR, PINBALL_TAU, V_NORM,
)
from src.evaluation.metrics import coverage, mae, rmse  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_mlp(pt, Xfull, te):
    m = TTAModel(in_dim=Xfull.shape[1]).to(device)
    m.load_state_dict(torch.load(pt, map_location=device))
    m.eval()
    Xte = Xfull[te]
    out = np.empty(len(Xte), dtype=np.float32)
    chunk = 200_000
    for i in range(0, len(Xte), chunk):
        xb = torch.from_numpy(Xte[i:i + chunk]).to(device)
        out[i:i + chunk] = m(xb).cpu().numpy()
    return out


def base_path(tau):
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def profile_path(tau, *feats):
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    return OUTPUTS_DIR / f"mlp_profile_{'_'.join(feats)}{suffix}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    args = ap.parse_args()
    tau = args.tau

    d = np.load(DATASET_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    te = split == 2
    y_te = y[te]

    vmax = (d["meta_vmax"] / V_NORM).astype(np.float32)[:, None]
    X_vmax = np.concatenate([X, vmax], axis=1)

    specs = [
        ("MLP base (6 feat)", base_path(tau), X),
        ("MLP + vmax", profile_path(tau, "vmax"), X_vmax),
    ]
    rows = [("SUVAT (kinematic filter)", d["meta_t_suvat"][te])]
    for name, path, Xfull in specs:
        if not path.exists():
            print(f"(skipping {name}: {path} not found)")
            continue
        rows.append((name, predict_mlp(path, Xfull, te)))

    nominal = 1.0 - tau
    print(f"TEST samples: {len(y_te):,}   tau={tau:g}   (coverage target {nominal:.3f})\n")
    print(f"{'model':26s} {'MAE':>7} {'RMSE':>7} {'pinball':>9} {'cover':>7}")
    print("-" * 60)
    for name, pred in rows:
        pin = pinball_loss(
            torch.from_numpy(pred.astype(np.float32)),
            torch.from_numpy(y_te.astype(np.float32)),
            tau=tau,
        ).item()
        print(f"{name:26s} {mae(pred, y_te):7.3f} {rmse(pred, y_te):7.3f} "
              f"{pin:9.4f} {coverage(pred, y_te):7.3f}")


if __name__ == "__main__":
    main()
