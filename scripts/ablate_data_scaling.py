"""Data-scaling ablation: does more training data buy accuracy?

Trains the baseline MLP on prefixes of k in {2,4,8,20,45} TRAIN matches (drawn,
in ascending match-index order, only from the TRAIN split -- never touching
val/test, so there is no leakage), evaluated every time on the SAME held-out
9-match TEST split. Prints the resulting learning-curve table.

    python -m scripts.ablate_data_scaling                # tau = config.PINBALL_TAU
    python -m scripts.ablate_data_scaling --tau 0.01
    python -m scripts.ablate_data_scaling --tau 0.01 --k 2 4 8 20 45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import _assign_keys  # noqa: E402
from src.config import AREA_LAM, DATASET_NPZ, OUTPUTS_DIR, PINBALL_TAU  # noqa: E402
from src.evaluation.metrics import coverage, mae, rmse  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.train import save_model, train  # noqa: E402

DEFAULT_KS = (2, 4, 8, 20, 45)


def pinball(pred, y, tau):
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


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
    ap.add_argument("--k", type=int, nargs="*", default=list(DEFAULT_KS),
                    help=f"train-match-count prefixes to sweep (default {DEFAULT_KS})")
    ap.add_argument("--area-lam", type=float, default=AREA_LAM,
                    help=f"area-aware aux weight (default {AREA_LAM:g}; 0 disables)")
    args = ap.parse_args()
    tau = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    X, y, split, match = d["X"], d["y"], d["split"], d["meta_match"]
    n_matches = len(d["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))
    train_matches = np.sort(np.where(per_match == 0)[0])
    n_train_pool = len(train_matches)

    va = split == 1
    te = split == 2
    y_te = y[te]
    print(f"train pool: {n_train_pool} matches | val {int(va.sum()):,} | test {int(te.sum()):,}"
          f"   device {device}   tau {tau:g}")

    ks = [k for k in args.k if k <= n_train_pool]
    if n_train_pool not in ks:
        ks.append(n_train_pool)

    results = []
    for k in ks:
        chosen = set(train_matches[:k].tolist())
        tr = np.isin(match, list(chosen)) & (split == 0)
        n_rows = int(tr.sum())
        label = f"{k}-match" if k < n_train_pool else f"full({k})"
        print(f"\n[train] {label}: {k} matches, {n_rows:,} rows", flush=True)
        model, best_val, _ = train(X[tr], y[tr], X[va], y[va], device=device, tau=tau,
                                   area_lam=args.area_lam)
        pred = predict(model, X[te], device)
        row = dict(label=label, k=k, rows=n_rows,
                   pinball=pinball(pred, y_te, tau), mae=mae(pred, y_te),
                   rmse=rmse(pred, y_te), cov=coverage(pred, y_te))
        results.append(row)
        print(f"[{label:>10s}]  {k:3d} matches, {n_rows:>12,} rows | "
              f"test pinball {row['pinball']:.5f}  MAE {row['mae']:.3f}  "
              f"RMSE {row['rmse']:.3f}  cov {row['cov']:.4f}")
        if k == n_train_pool:
            suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
            save_model(model, OUTPUTS_DIR / f"mlp_datascale_full{suffix}.pt")

    print(f"\n==== LEARNING CURVE (all evaluated on the SAME {int(te.sum()):,}-sample test set) ====")
    print(f"{'train':>10s} {'matches':>8s} {'rows':>13s} {'test pinball':>13s} "
          f"{'test MAE':>9s} {'test RMSE':>10s} {'cov':>8s}")
    for r in results:
        print(f"{r['label']:>10s} {r['k']:>8d} {r['rows']:>13,} {r['pinball']:>13.5f} "
              f"{r['mae']:>9.3f} {r['rmse']:>10.3f} {r['cov']:>8.4f}")
    if len(results) > 1:
        d_pin = results[-1]["pinball"] - results[0]["pinball"]
        d_mae = results[-1]["mae"] - results[0]["mae"]
        print(f"\nΔ(full − {results[0]['label']}):  pinball {d_pin:+.5f}   MAE {d_mae:+.3f}s")


if __name__ == "__main__":
    main()
