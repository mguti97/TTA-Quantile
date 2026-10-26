"""Train the player-profile TTA MLP: base 6 features + per-player max speed.

Each sample carries its player's max speed (the 99th percentile of that
player's observed speeds, computed at build time and stored as the `vmax` meta
column). We append it -- normalised by V_NORM -- as a 7th input, so the MLP can
personalise its reachable envelope instead of assuming one global locomotion
capacity for everyone (Section 12 player profiles).

    python -m scripts.train_profile               # base + vmax (7 feat)

Train at a non-default quantile with --tau (saves to *_tau<t>.pt):
    python -m scripts.train_profile vmax --tau 0.01

Compares the profile model against SUVAT and the base 6-feature MLP on the same
held-out test split. Saves to outputs/<dataset>/mlp_profile_<feats>.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    AREA_LAM,
    DATASET_NPZ,
    MODEL_PT,
    N_FEATURES,
    OUTPUTS_DIR,
    PINBALL_TAU,
    SEED,
    V_NORM,
)
from src.evaluation.metrics import coverage, mae, rmse  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.train import save_model, train  # noqa: E402

# profile feature -> (meta key in the npz, normalisation constant)
PROFILE_FEATS = {
    "vmax": ("meta_vmax", V_NORM),
}


def _load_meta(d, key: str, n: int) -> np.ndarray:
    if key in d.files:
        return d[key]
    raise SystemExit(f"dataset has no {key} -- rebuild with `python -m scripts.build_dataset`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats", nargs="*", default=["vmax"],
                    help=f"profile feats to append: {list(PROFILE_FEATS)} (default vmax)")
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    ap.add_argument("--area-lam", type=float, default=AREA_LAM,
                    help=f"area-aware aux weight (default {AREA_LAM:g}; 0 disables)")
    ap.add_argument("--aux-horizons", type=str, default=None,
                    help="soft-area horizons: comma list ('3,5') or range 'start:stop:step' "
                         "('1:5:0.25'). Default: config.AREA_HORIZONS.")
    ap.add_argument("--aux-jitter", action="store_true",
                    help="re-draw the soft-area grid offset and horizons every step "
                         "(removes the creases/walls the fixed grid+horizons pin)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for the checkpoint name (e.g. 'smooth')")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"seed for weight init / batch shuffling only -- the dataset "
                         f"build/split is unaffected (default {SEED})")
    args = ap.parse_args()
    if args.aux_horizons is None:
        aux_horizons = None
    elif ":" in args.aux_horizons:
        lo, hi, step = (float(x) for x in args.aux_horizons.split(":"))
        aux_horizons = tuple(np.arange(lo, hi + 1e-9, step).round(6).tolist())
    else:
        aux_horizons = tuple(float(x) for x in args.aux_horizons.split(","))
    names = [a.lower() for a in args.feats] or ["vmax"]
    tau = args.tau
    bad = [n for n in names if n not in PROFILE_FEATS]
    if bad:
        raise SystemExit(f"unknown profile feat(s) {bad}; choose from {list(PROFILE_FEATS)}")

    d = np.load(DATASET_NPZ)
    n = d["y"].shape[0]
    extra_cols = []
    for n_ in names:
        key, norm = PROFILE_FEATS[n_]
        extra_cols.append((_load_meta(d, key, n) / norm).astype(np.float32)[:, None])

    X, y, split = d["X"], d["y"], d["split"]
    Xp = np.concatenate([X] + extra_cols, axis=1)
    tr, va, te = split == 0, split == 1, split == 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train {int(tr.sum()):,} | val {int(va.sum()):,} | test {int(te.sum()):,}")
    print(f"device: {device}   features: {Xp.shape[1]} (6 base + {'+'.join(names)})   "
          f"tau: {tau:g}   area_lam: {args.area_lam:g}   seed: {args.seed}"
          f"   aux_jitter: {args.aux_jitter}")

    model, best_val, _ = train(
        Xp[tr], y[tr], Xp[va], y[va], device=device, in_dim=Xp.shape[1], tau=tau,
        area_lam=args.area_lam, area_horizons=aux_horizons,
        area_jitter=args.aux_jitter, seed=args.seed
    )
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    if args.out_suffix:
        suffix += f"_{args.out_suffix}"
    out_pt = OUTPUTS_DIR / f"mlp_profile_{'_'.join(names)}{suffix}.pt"
    save_model(model, out_pt)
    print(f"\nbest val pinball {best_val:.5f}")
    print(f"saved -> {out_pt}")

    # ---- compare SUVAT / base 6-feat MLP / profile MLP on the test split -------
    y_te = y[te]

    @torch.no_grad()
    def predict(m, Xfull):
        m.eval()
        Xte = Xfull[te]
        out = np.empty(len(Xte), dtype=np.float32)
        chunk = 200_000  # bound peak GPU memory on the full-test forward pass
        for i in range(0, len(Xte), chunk):
            xb = torch.from_numpy(Xte[i:i + chunk]).to(device)
            out[i:i + chunk] = m(xb).cpu().numpy()
        return out

    base_pt = MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"
    rows = [("SUVAT", d["meta_t_suvat"][te])]
    if base_pt.exists():
        base = TTAModel(in_dim=N_FEATURES).to(device)
        base.load_state_dict(torch.load(base_pt, map_location=device))
        rows.append(("MLP base (6 feat)", predict(base, X)))
    else:
        print(f"(note: {base_pt} not found -- run `python -m scripts.train --tau {tau:g}` to compare)")
    rows.append((f"MLP profile (+{'+'.join(names)})", predict(model, Xp)))

    nominal = 1.0 - tau
    print(f"\nTEST samples: {len(y_te):,}   (coverage target {nominal:.3f})")
    print(f"{'model':24s} {'MAE':>7} {'RMSE':>7} {'pinball':>8} {'cover':>7}")
    for name, pred in rows:
        pin = pinball_loss(
            torch.from_numpy(pred.astype(np.float32)),
            torch.from_numpy(y_te.astype(np.float32)),
            tau=tau,
        ).item()
        print(f"{name:24s} {mae(pred, y_te):7.3f} {rmse(pred, y_te):7.3f} "
              f"{pin:8.4f} {coverage(pred, y_te):7.3f}")


if __name__ == "__main__":
    main()
