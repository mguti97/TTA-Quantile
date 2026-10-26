"""Evaluate the ball-pass model vs the straight-line ball baseline.

    python -m scripts.evaluate_pass                # tau = config.PINBALL_TAU
    python -m scripts.evaluate_pass --tau 0.01     # loads outputs/ffpc_pass/mlp_tau0.01.pt

Reports overall MAE/RMSE/coverage and a stratified breakdown (by distance,
passer speed, pass angle) of the learned model against the d/BALL_V_MAX baseline
-- the ball-pipeline mirror of scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    PASS_MODEL_PT, PASS_N_FEATURES, PASS_NPZ, PASS_OUTPUTS_DIR, PINBALL_TAU,
)
from src.evaluation.metrics import (  # noqa: E402
    coverage,
    format_table,
    mae,
    rmse,
    stratified,
)
from src.model.mlp import TTAModel  # noqa: E402


def model_path(tau: float) -> Path:
    return PASS_MODEL_PT if tau == PINBALL_TAU else PASS_OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    args = ap.parse_args()
    tau = args.tau

    d = np.load(PASS_NPZ)
    te = d["split"] == 2
    X = d["X"][te]
    y = d["y"][te]
    t_ball = d["meta_t_ball"][te]
    dist = d["meta_d"][te]
    vpass = d["meta_v"][te]
    theta = d["meta_theta_deg"][te]
    print(f"test passes: {te.sum():,}   tau={tau:g}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTAModel(in_dim=PASS_N_FEATURES).to(device)
    model.load_state_dict(torch.load(model_path(tau), map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()

    print("\n=== overall (test) ===")
    print(f"{'':14s}{'MAE':>8s}{'RMSE':>8s}{'coverage':>10s}")
    for name, p in (("model", pred), ("ball baseline", t_ball)):
        print(f"{name:14s}{mae(p, y):>8.3f}{rmse(p, y):>8.3f}{coverage(p, y):>10.3f}")
    print(f"(target coverage 1-tau = {1 - tau:.2f}; baseline is a lower bound)")

    print("\n=== stratified MAE: model vs ball baseline ===")
    rows = stratified(pred, t_ball, y, dist, vpass, theta)
    print(format_table(rows))


if __name__ == "__main__":
    main()
