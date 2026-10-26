"""Train the TTA model on the prebuilt dataset.

    python -m scripts.train                # tau = config.PINBALL_TAU -> outputs/<ds>/mlp.pt
    python -m scripts.train --tau 0.01     # other quantile           -> outputs/<ds>/mlp_tau0.01.pt
    python -m scripts.train --area-lam 0   # disable the area-aware auxiliary term

The area-aware auxiliary loss (config.AREA_LAM, canonical since 2026-07-03) is
ON by default; see src/model/arealoss.py for the trade-off it makes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AREA_LAM, DATASET_NPZ, MODEL_PT, OUTPUTS_DIR, PINBALL_TAU, SEED  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.train import save_model, train  # noqa: E402


def load_split():
    d = np.load(DATASET_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def model_path(tau: float, out_suffix: str = "") -> Path:
    base = MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"
    if not out_suffix:
        return base
    # variant runs never overwrite the canonical checkpoint
    return base.with_name(f"{base.stem}{out_suffix}{base.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    ap.add_argument("--area-lam", type=float, default=AREA_LAM,
                    help=f"area-aware aux weight (default {AREA_LAM:g}; 0 disables)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix appended to the checkpoint stem, e.g. --out-suffix "
                         "_nofilter -> outputs/<ds>/mlp_nofilter.pt (keeps variant "
                         "runs off the canonical mlp.pt)")
    ap.add_argument("--force-streaming", action="store_true",
                    help="keep training data CPU-resident and stream batches to the GPU, "
                         "bypassing the free-memory heuristic (for datasets too large to "
                         "safely fit resident, e.g. nba_match's 291M-row train split)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"seed for weight init / batch shuffling only -- the dataset "
                         f"build/split is unaffected (default {SEED})")
    args = ap.parse_args()

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_split()
    print(f"train {X_tr.shape[0]:,} | val {X_va.shape[0]:,} | test {X_te.shape[0]:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   tau: {args.tau:g}   area_lam: {args.area_lam:g}   seed: {args.seed}")
    model, best_val, _ = train(X_tr, y_tr, X_va, y_va, device=device, tau=args.tau,
                               area_lam=args.area_lam, force_streaming=args.force_streaming,
                               seed=args.seed)

    # report test pinball at the trained quantile
    model.eval()
    with torch.no_grad():
        pred = np.empty(len(X_te), dtype=np.float32)
        for i in range(0, len(X_te), 1_000_000):
            xb = torch.from_numpy(X_te[i:i + 1_000_000]).to(device)
            pred[i:i + 1_000_000] = model(xb).cpu().numpy()
    test_pin = pinball_loss(torch.from_numpy(pred), torch.from_numpy(y_te),
                            tau=args.tau).item()
    print(f"\nbest val pinball {best_val:.5f} | test pinball {test_pin:.5f}  (tau={args.tau:g})")

    out = model_path(args.tau, args.out_suffix)
    save_model(model, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
