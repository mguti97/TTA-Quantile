"""Train the ball-pass minimum-arrival-time model.

Identical recipe to scripts/train.py (Adam + pinball + early stopping), but on
the 4-feature pass dataset and saving to PASS_MODEL_PT (or a tau-suffixed path).

    python -m scripts.train_pass                # tau = config.PINBALL_TAU
    python -m scripts.train_pass --tau 0.01     # -> outputs/ffpc_pass/mlp_tau0.01.pt
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
from src.model.loss import pinball_loss  # noqa: E402
from src.model.train import save_model, train  # noqa: E402


def load_split():
    d = np.load(PASS_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def model_path(tau: float) -> Path:
    return PASS_MODEL_PT if tau == PINBALL_TAU else PASS_OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    args = ap.parse_args()

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_split()
    print(f"train {X_tr.shape[0]:,} | val {X_va.shape[0]:,} | test {X_te.shape[0]:,} "
          f"| features {X_tr.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   tau: {args.tau:g}")
    model, best_val, _ = train(X_tr, y_tr, X_va, y_va, device=device,
                              in_dim=PASS_N_FEATURES, tau=args.tau)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
    test_pin = pinball_loss(torch.from_numpy(pred), torch.from_numpy(y_te), tau=args.tau).item()
    print(f"\nbest val pinball {best_val:.5f} | test pinball {test_pin:.5f}  (tau={args.tau:g})")

    out = model_path(args.tau)
    save_model(model, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
