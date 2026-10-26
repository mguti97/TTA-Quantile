"""EXPERIMENTAL: does joint multi-quantile training stabilise the tau=0.01
extreme quantile, specifically the long-horizon / high-coverage reachable-area
degradation found in the single-quantile MLP (paper Table area, Delta t=5s,
90-99% coverage, where several physics baselines beat it)?

Trains a 6->64->64->K MLP jointly fitting QUANTILES (ascending, includes 0.01)
with a monotone (non-crossing) output via MultiQuantileTTAModel, early-stopping
on the tau=0.01 validation slice specifically (the metric we care about, same
selection criterion as the single-quantile baseline). Extracts the tau=0.01
slice and compares it against the canonical single-quantile mlp_tau0.01.pt on:

  (a) overall + stratified MAE/RMSE/pinball(0.01)/coverage
  (b) matched-coverage reachable area at 1/3/5s x {80,90,95,99}% -- the exact
      cells where the single-quantile MLP lost ground to physics.

Not part of the canonical pipeline: saves to outputs/<dataset>/mlp_multiquantile.pt,
does not touch mlp_tau0.01.pt or any figure/paper artifact.

    python -m scripts.experiment_multiquantile                              # default 7-quantile set
    python -m scripts.experiment_multiquantile --quantiles 0.01 0.05 0.10 0.25   # tail-concentrated
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import (  # noqa: E402
    DATASET_NPZ, GRID_RES, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, BATCH_SIZE,
    OUTPUTS_DIR, SEED,
)
from src.evaluation.metrics import (  # noqa: E402
    DIST_BANDS, SPEED_BANDS, ANGLE_BANDS, coverage, mae, rmse,
)
from src.evaluation.reachable_area import grid_tta, matched_coverage_area, tta_to_targets  # noqa: E402
from src.model.multiquantile import MultiQuantileTTAModel, multi_pinball_loss  # noqa: E402

DEFAULT_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
# module-level so the helpers below see them; set from --quantiles in main()
QUANTILES = list(DEFAULT_QUANTILES)
TARGET_IDX = 0  # index of tau=0.01 in QUANTILES (asserted in main)
TARGET_TAU = 0.01

HORIZONS = (1.0, 3.0, 5.0)
COVERAGES = (0.80, 0.90, 0.95, 0.99)
RA_TRAILS = 500
RA_RES = 1.0

# reference numbers from the canonical single-quantile mlp_tau0.01.pt, so this
# script is self-contained (no need to re-run the baseline to compare).
BASELINE_OVERALL = dict(mae=2.613, rmse=3.207, pinball=0.02761, coverage=0.9904)
BASELINE_AREA = {  # [horizon][coverage] -> m^2, from eval_coverage_sweep_tau0.01.log
    1.0: {0.80: 5.4, 0.90: 8.7, 0.95: 12.6, 0.99: 34.7},
    3.0: {0.80: 88.4, 0.90: 167.1, 0.95: 263.1, 0.99: 632.1},
    5.0: {0.80: 287.9, 0.90: 644.5, 0.95: 1244.6, 0.99: 2739.5},
}


def pinball(pred, y, tau):
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


@torch.no_grad()
def predict_slice(model, X, device, idx=TARGET_IDX, chunk=1_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        xb = torch.from_numpy(X[s:e]).to(device)
        out[s:e] = model(xb)[:, idx].cpu().numpy()
    return out


def train_multiquantile(X_tr, y_tr, X_va, y_va, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MultiQuantileTTAModel(QUANTILES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)
    n_tr = X_tr_t.size(0)
    eval_batch = BATCH_SIZE * 16

    best_val_target = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = multi_pinball_loss(model(X_tr_t[idx]), y_tr_t[idx], QUANTILES)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tot_target = 0.0
            n = X_va_t.size(0)
            for i in range(0, n, eval_batch):
                sl = slice(i, i + eval_batch)
                pred = model(X_va_t[sl])
                err = y_va_t[sl].unsqueeze(1) - pred[:, TARGET_IDX:TARGET_IDX + 1]
                l = torch.maximum(TARGET_TAU * err, (TARGET_TAU - 1.0) * err)
                tot_target += l.sum().item()
            val_target = tot_target / n

        if val_target < best_val_target - 1e-6:
            best_val_target = val_target
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"epoch {epoch:3d}  val_pinball[tau=0.01] {val_target:.5f}  "
              f"best {best_val_target:.5f}  patience {patience}", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_target


def main():
    global QUANTILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantiles", type=float, nargs="+", default=DEFAULT_QUANTILES,
                    help="ascending quantiles to fit jointly; must start with 0.01")
    args = ap.parse_args()
    QUANTILES = sorted(args.quantiles)
    assert QUANTILES[TARGET_IDX] == TARGET_TAU, "tau=0.01 must be the first quantile"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    print(f"train {int(tr.sum()):,} | val {int(va.sum()):,} | test {int(te.sum()):,}"
          f"   device {device}   quantiles {QUANTILES}")

    model, best_val = train_multiquantile(X[tr], y[tr], X[va], y[va], device)
    out_pt = OUTPUTS_DIR / f"mlp_multiquantile_{len(QUANTILES)}q.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_pt)
    print(f"\nbest val pinball[tau=0.01] {best_val:.5f}\nsaved -> {out_pt}")

    # ---- overall + stratified test metrics, tau=0.01 slice only -------------
    y_te = y[te]
    pred = predict_slice(model, X[te], device)
    dist = d["meta_d"][te]
    speed = d["meta_v"][te]
    theta = np.abs(d["meta_theta_deg"][te])

    print(f"\n=== OVERALL (test {len(y_te):,}; tau=0.01, target coverage 0.990) ===")
    print(f"{'model':22s} {'MAE':>7s} {'RMSE':>7s} {'pinball':>9s} {'coverage':>9s}")
    print(f"{'single-quantile (baseline)':22s} {BASELINE_OVERALL['mae']:7.3f} "
          f"{BASELINE_OVERALL['rmse']:7.3f} {BASELINE_OVERALL['pinball']:9.5f} "
          f"{BASELINE_OVERALL['coverage']:9.4f}")
    print(f"{'multi-quantile (tau=.01)':22s} {mae(pred, y_te):7.3f} {rmse(pred, y_te):7.3f} "
          f"{pinball(pred, y_te, TARGET_TAU):9.5f} {coverage(pred, y_te):9.4f}")

    for title, vals, bands in (("distance", dist, DIST_BANDS),
                               ("speed", speed, SPEED_BANDS),
                               ("angle", theta, ANGLE_BANDS)):
        print(f"\n=== stratified MAE by {title} (multi-quantile tau=.01 slice) ===")
        hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}" for lo, hi in bands)
        print(f"{hdr}")
        cells = []
        for lo, hi in bands:
            m = (vals >= lo) & (vals < hi)
            cells.append(f"{mae(pred[m], y_te[m]):6.3f}" if m.any() else "  -   ")
        print("  ".join(cells))

    # ---- matched-coverage reachable area, tau=0.01 slice ---------------------
    @torch.no_grad()
    def fn(tf):
        Xg = tf["X"]
        out = np.empty(len(Xg), dtype=np.float32)
        for s in range(0, len(Xg), 1_000_000):
            e = min(s + 1_000_000, len(Xg))
            xb = torch.from_numpy(Xg[s:e]).to(device)
            out[s:e] = model(xb)[:, TARGET_IDX].cpu().numpy()
        return out

    print(f"\n{'='*90}\nMatched-coverage reachable area (m^2): multi-quantile tau=.01 "
          "slice vs single-quantile baseline")
    print(f"{'Delta t':>8s} {'coverage':>9s} {'multi-quantile':>15s} "
          f"{'single-quantile':>16s} {'delta':>9s}")
    print("-" * 62)
    for h in HORIZONS:
        t0 = time.perf_counter()
        trails = load_split_trails(2, h, max_matches=1).subsample(RA_TRAILS, seed=SEED + 1)
        pred_tgt = tta_to_targets(fn, trails)
        grid, cell_area = grid_tta(fn, trails, res=RA_RES)
        for c in COVERAGES:
            area = matched_coverage_area(pred_tgt, grid, cell_area, c)["mean_area"]
            base = BASELINE_AREA[h][c]
            delta = area - base
            sign = "better" if delta < 0 else ("worse" if delta > 0 else "tie")
            print(f"{h:>7.0f}s {c*100:>8.0f}% {area:>15.1f} {base:>16.1f} "
                  f"{delta:>+8.1f} ({sign})")
        print(f"  [{h:.0f}s: {time.perf_counter()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
