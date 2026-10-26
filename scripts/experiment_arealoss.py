"""EXPERIMENTAL: pinball + differentiable AREA-AWARE auxiliary loss.

Hypothesis: the single-quantile tau=0.01 MLP loses matched-coverage reachable
area to physics at long horizons because pointwise pinball never sees the
integrated pitch area its surface produces -- per-direction slack in sparse
geometries inflates the sublevel set. Fix: add a differentiable relaxation of
the reachable-area fraction as an auxiliary training term.

For S states subsampled from each mini-batch, take their heading-frame
(v, a'x, a'y), evaluate the model on a fixed heading-frame Cartesian grid
(the surface is translation/rotation invariant, so heading-frame area equals
pitch area), and soft-count the cells inside the horizon-h reachable set:

    soft_area(h) = mean_cells sigmoid((h - tta_cell) / beta)

Minimising it can only push tta UP (out of the set) -- the gradient
d/d tta of sigmoid((h-tta)/beta) is strictly negative -- so the term is pure
area-shrinking pressure, opposed by the 99:1 pinball asymmetry wherever the
data actually contains fast arrivals. Total loss:

    L = pinball_{0.01}(batch) + lambda * mean_{h in HORIZONS_AUX} soft_area(h)

Early stopping on the same composite computed on a fixed validation slice
(pinball on the full val split + soft-area on a fixed val state subset).
Evaluation mirrors experiment_multiquantile.py: overall/stratified metrics and
matched-coverage reachable area vs the canonical single-quantile baseline.

Not part of the canonical pipeline: saves to outputs/<ds>/mlp_arealoss.pt.

    python -m scripts.experiment_arealoss
    python -m scripts.experiment_arealoss --lam 0.03
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
    BATCH_SIZE, D_NORM, DATASET_NPZ, EARLY_STOP_PATIENCE, GRID_RES, LR,
    MAX_EPOCHS, OUTPUTS_DIR, SEED,
)
from src.evaluation.metrics import (  # noqa: E402
    ANGLE_BANDS, DIST_BANDS, SPEED_BANDS, coverage, mae, rmse,
)
from src.evaluation.reachable_area import grid_tta, matched_coverage_area, tta_to_targets  # noqa: E402
from src.model.arealoss import stochastic_aux  # noqa: E402
from src.model.loss import pinball_loss  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402

TAU = 0.01
HORIZONS_AUX = (3.0, 5.0)   # horizons the soft-area term penalises
BETA = 0.25                 # sigmoid temperature (s)
AUX_STATES = 48             # states per step in the soft-area term
GRID_STEP = 4.0             # heading-frame grid resolution (m)
GRID_HALF = 48.0            # grid half-extent (m); covers the 5 s reachable set
VAL_AUX_STATES = 256        # fixed val states for the early-stopping composite

HORIZONS_EVAL = (1.0, 3.0, 5.0)
COVERAGES = (0.80, 0.90, 0.95, 0.99)
RA_TRAILS = 500
RA_RES = 1.0

# canonical single-quantile mlp_tau0.01.pt reference numbers (self-contained)
BASELINE_OVERALL = dict(mae=2.613, rmse=3.207, pinball=0.02761, coverage=0.9904)
BASELINE_AREA = {  # [horizon][coverage] -> m^2, from eval_coverage_sweep_tau0.01.log
    1.0: {0.80: 5.4, 0.90: 8.7, 0.95: 12.6, 0.99: 34.7},
    3.0: {0.80: 88.4, 0.90: 167.1, 0.95: 263.1, 0.99: 632.1},
    5.0: {0.80: 287.9, 0.90: 644.5, 0.95: 1244.6, 0.99: 2739.5},
}


def pinball_np(pred, y, tau=TAU):
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def heading_grid(device):
    """Fixed heading-frame Cartesian grid -> (d/D_NORM, cos, sin) columns, (G,3)."""
    ax = np.arange(-GRID_HALF, GRID_HALF + 1e-6, GRID_STEP)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    x = xx.ravel()
    y = yy.ravel()
    d = np.hypot(x, y)
    keep = d >= 2.0          # drop the degenerate near-origin cell(s)
    x, y, d = x[keep], y[keep], d[keep]
    cols = np.stack([d / D_NORM, x / d, y / d], axis=1).astype(np.float32)
    return torch.from_numpy(cols).to(device)   # (G, 3)


def soft_area(model, grid_cols, state_va, horizons=HORIZONS_AUX, beta=BETA,
              weights=None):
    """Weighted mean soft reachable-area fraction over states and horizons.

    grid_cols: (G,3) fixed geometry columns; state_va: (S,3) = (v, a'x, a'y)
    feature columns of the sampled states. Builds (S*G, 6) inputs.
    weights: per-horizon weights summing to 1; None -> uniform. The h-horizon
    set covers ~h^2 more grid cells, so under uniform weights the longest
    horizon's gradient dominates and shorter level sets deform as collateral;
    1/h^2 weights equalise the relative shrinking pressure.
    """
    S = state_va.size(0)
    G = grid_cols.size(0)
    Xg = torch.empty(S, G, 6, device=state_va.device)
    Xg[:, :, :3] = grid_cols.unsqueeze(0)
    Xg[:, :, 3:] = state_va.unsqueeze(1)
    tta = model(Xg.reshape(S * G, 6)).reshape(S, G)
    if weights is None:
        weights = [1.0 / len(horizons)] * len(horizons)
    total = 0.0
    for h, w in zip(horizons, weights):
        total = total + w * torch.sigmoid((h - tta) / beta).mean()
    return total


def train_arealoss(X_tr, y_tr, X_va, y_va, device, lam, horizons=HORIZONS_AUX,
                   weights=None, jitter=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    aux_rng = np.random.default_rng(SEED + 7)

    model = TTAModel(in_dim=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    grid_cols = heading_grid(device)
    wdesc = "uniform" if weights is None else \
        tuple(round(w, 4) for w in weights)
    print(f"heading-frame grid: {grid_cols.size(0)} cells @ {GRID_STEP:g}m, "
          f"aux horizons {tuple(round(h, 3) for h in horizons)}, beta {BETA:g}, "
          f"lambda {lam:g}, weights {wdesc}"
          + ("  [jitter: random grid offset + uniform horizons per step]" if jitter else ""))

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)
    n_tr = X_tr_t.size(0)
    eval_batch = BATCH_SIZE * 16

    # fixed validation states for the composite early-stopping criterion
    rng = np.random.default_rng(SEED + 3)
    va_idx = torch.from_numpy(
        rng.choice(X_va_t.size(0), size=VAL_AUX_STATES, replace=False)).to(device)
    va_states = X_va_t[va_idx, 3:6]

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb, tau=TAU)
            if lam > 0:
                if jitter:
                    g_cols, hs = stochastic_aux(aux_rng, device, horizons,
                                                step=GRID_STEP, half=GRID_HALF)
                else:
                    g_cols, hs = grid_cols, horizons
                loss = loss + lam * soft_area(model, g_cols, xb[:AUX_STATES, 3:6],
                                              horizons=hs, weights=weights)
            loss.backward()
            opt.step()

        # ---- composite validation: pinball on full val + soft-area on subset --
        model.eval()
        with torch.no_grad():
            tot = 0.0
            n = X_va_t.size(0)
            for i in range(0, n, eval_batch):
                sl = slice(i, i + eval_batch)
                tot += pinball_loss(model(X_va_t[sl]), y_va_t[sl], tau=TAU).item() \
                    * (min(i + eval_batch, n) - i)
            val_pin = tot / n
            val_area = soft_area(model, grid_cols, va_states, horizons=horizons,
                                 weights=weights).item() if lam > 0 else 0.0
        val = val_pin + lam * val_area

        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"epoch {epoch:3d}  val_pinball {val_pin:.5f}  val_softarea {val_area:.4f}  "
              f"composite {val:.5f}  best {best_val:.5f}  patience {patience}", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def predict(model, X, device, chunk=1_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        out[s:e] = model(torch.from_numpy(X[s:e]).to(device)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.01,
                    help="weight of the soft-area auxiliary term (default 0.01)")
    ap.add_argument("--aux-horizons", type=str, default=None,
                    help="soft-area horizons: comma list ('3,5') or range 'start:stop:step' "
                         "('1:5:0.25'). Default: the canonical (3, 5). A dense grid spreads "
                         "the area pressure over all level sets, avoiding the walls the "
                         "two-horizon recipe sculpts at t=3s and t=5s.")
    ap.add_argument("--aux-weights", type=str, default="uniform",
                    help="per-horizon weights for the soft-area term: 'uniform' "
                         "(default), 'invsq' (proportional to 1/h^2, equalising the "
                         "relative shrinking pressure across level sets), or a comma "
                         "list matching --aux-horizons")
    ap.add_argument("--aux-jitter", action="store_true",
                    help="re-draw the soft-area grid offset (uniform sub-cell) and "
                         "horizons (uniform in [min,max] of --aux-horizons) every step; "
                         "removes the creases the fixed 4 m grid pins at cell radii")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for the checkpoint name (e.g. 'smooth' -> "
                         "mlp_arealoss_lam<lam>_smooth.pt)")
    args = ap.parse_args()

    if args.aux_horizons is None:
        aux_horizons = HORIZONS_AUX
    elif ":" in args.aux_horizons:
        lo, hi, step = (float(x) for x in args.aux_horizons.split(":"))
        aux_horizons = tuple(np.arange(lo, hi + 1e-9, step).round(6).tolist())
    else:
        aux_horizons = tuple(float(x) for x in args.aux_horizons.split(","))

    if args.aux_weights == "uniform":
        aux_weights = None
    elif args.aux_weights == "invsq":
        raw = np.array([1.0 / h ** 2 for h in aux_horizons])
        aux_weights = tuple((raw / raw.sum()).tolist())
    else:
        raw = np.array([float(x) for x in args.aux_weights.split(",")])
        assert len(raw) == len(aux_horizons), \
            f"{len(raw)} weights for {len(aux_horizons)} horizons"
        aux_weights = tuple((raw / raw.sum()).tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    print(f"train {int(tr.sum()):,} | val {int(va.sum()):,} | test {int(te.sum()):,}"
          f"   device {device}   tau {TAU:g}")

    model, best_val = train_arealoss(X[tr], y[tr], X[va], y[va], device, args.lam,
                                     horizons=aux_horizons, weights=aux_weights,
                                     jitter=args.aux_jitter)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_pt = OUTPUTS_DIR / f"mlp_arealoss_lam{args.lam:g}{suffix}.pt"
    torch.save(model.state_dict(), out_pt)
    print(f"\nbest composite val {best_val:.5f}\nsaved -> {out_pt}")

    # ---- overall + stratified test metrics -----------------------------------
    y_te = y[te]
    pred = predict(model, X[te], device)
    dist = d["meta_d"][te]
    speed = d["meta_v"][te]
    theta = np.abs(d["meta_theta_deg"][te])

    print(f"\n=== OVERALL (test {len(y_te):,}; tau=0.01, target coverage 0.990) ===")
    print(f"{'model':26s} {'MAE':>7s} {'RMSE':>7s} {'pinball':>9s} {'coverage':>9s}")
    print(f"{'single-quantile (baseline)':26s} {BASELINE_OVERALL['mae']:7.3f} "
          f"{BASELINE_OVERALL['rmse']:7.3f} {BASELINE_OVERALL['pinball']:9.5f} "
          f"{BASELINE_OVERALL['coverage']:9.4f}")
    print(f"{'area-loss (lam=%g)' % args.lam:26s} {mae(pred, y_te):7.3f} {rmse(pred, y_te):7.3f} "
          f"{pinball_np(pred, y_te):9.5f} {coverage(pred, y_te):9.4f}")

    for title, vals, bands in (("distance", dist, DIST_BANDS),
                               ("speed", speed, SPEED_BANDS),
                               ("angle", theta, ANGLE_BANDS)):
        print(f"\n=== stratified MAE by {title} (area-loss model) ===")
        hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}" for lo, hi in bands)
        print(hdr)
        cells = []
        for lo, hi in bands:
            m = (vals >= lo) & (vals < hi)
            cells.append(f"{mae(pred[m], y_te[m]):6.3f}" if m.any() else "  -   ")
        print("  ".join(cells))

    # ---- matched-coverage reachable area --------------------------------------
    @torch.no_grad()
    def fn(tf):
        Xg = tf["X"]
        out = np.empty(len(Xg), dtype=np.float32)
        for s in range(0, len(Xg), 1_000_000):
            e = min(s + 1_000_000, len(Xg))
            out[s:e] = model(torch.from_numpy(Xg[s:e]).to(device)).cpu().numpy()
        return out

    print(f"\n{'='*90}\nMatched-coverage reachable area (m^2): area-loss model vs "
          "single-quantile baseline")
    print(f"{'Delta t':>8s} {'coverage':>9s} {'area-loss':>12s} "
          f"{'single-quantile':>16s} {'delta':>9s}")
    print("-" * 60)
    for h in HORIZONS_EVAL:
        t0 = time.perf_counter()
        trails = load_split_trails(2, h, max_matches=1).subsample(RA_TRAILS, seed=SEED + 1)
        pred_tgt = tta_to_targets(fn, trails)
        grid, cell_area = grid_tta(fn, trails, res=RA_RES)
        for c in COVERAGES:
            area = matched_coverage_area(pred_tgt, grid, cell_area, c)["mean_area"]
            base = BASELINE_AREA[h][c]
            delta = area - base
            sign = "better" if delta < 0 else ("worse" if delta > 0 else "tie")
            print(f"{h:>7.0f}s {c*100:>8.0f}% {area:>12.1f} {base:>16.1f} "
                  f"{delta:>+8.1f} ({sign})")
        print(f"  [{h:.0f}s: {time.perf_counter()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
