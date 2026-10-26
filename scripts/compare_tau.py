"""Compare MLPs trained at different pinball quantiles (τ = 0.05 vs τ = 0.01).

Each model is scored against ITS OWN quantile: pinball at the trained τ and
coverage against the 1 − τ target (marginal + feature-conditioned by distance
and by predicted-time band -- the trustworthy calibration view, spec §10.4).
The reachable-area comparison is τ-fair by construction: every model is tuned
to the SAME matched coverage and we compare the pitch area it needs, so a
tighter quantile only wins if it actually buys a smaller area at equal
correctness.

    python -m scripts.compare_tau                 # test metrics + matched-coverage area
    python -m scripts.compare_tau --skip-area     # test metrics only (fast)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import (  # noqa: E402
    DATASET_NPZ, GRID_RES, MODEL_PT, OUTPUTS_DIR, SEED, V_NORM,
)
from src.evaluation.metrics import (  # noqa: E402
    DIST_BANDS, HORIZON_BANDS, coverage, mae, rmse,
)
from src.evaluation.reachable_area import (  # noqa: E402
    grid_tta, matched_coverage_area, tta_to_targets,
)
from src.model.mlp import TTAModel  # noqa: E402

TAUS = (0.05, 0.01)
HORIZONS_S = (1.0, 5.0, 8.0)
COVERAGES = (0.90, 0.95, 0.99)
MAX_TRAILS = 2000
MAX_MATCHES = 1
TEST_SPLIT = 2


def pinball(pred, y, tau):
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def base_path(tau):
    return MODEL_PT if tau == 0.05 else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def vmax_path(tau):
    suffix = "" if tau == 0.05 else f"_tau{tau:g}"
    return OUTPUTS_DIR / f"mlp_profile_vmax{suffix}.pt"


def load_mlp(path, in_dim, device):
    m = TTAModel(in_dim=in_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def predict(m, X, device, chunk=1_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        out[s:e] = m(torch.from_numpy(X[s:e]).to(device)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-area", action="store_true",
                    help="skip the matched-coverage reachable-area section")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    te = d["split"] == TEST_SPLIT
    X, y = d["X"][te], d["y"][te].astype(np.float64)
    dist = d["meta_d"][te]
    vmax_norm = (d["meta_vmax"][te] / V_NORM).astype(np.float32)[:, None]
    X_vmax = np.concatenate([X, vmax_norm], axis=1)

    # (label, tau, path, in_dim, X for that model) -- skip variants not trained yet
    specs = []
    for tau in TAUS:
        specs.append((f"MLP tau={tau:g}", tau, base_path(tau), 6, X))
        specs.append((f"MLP+vmax tau={tau:g}", tau, vmax_path(tau), 7, X_vmax))
    missing = [(lbl, p) for lbl, _, p, _, _ in specs if not p.exists()]
    for lbl, p in missing:
        print(f"(skipping {lbl}: {p} not found)")
    specs = [s for s in specs if s[2].exists()]

    models, preds = {}, {}
    for lbl, tau, path, in_dim, Xm in specs:
        models[lbl] = (load_mlp(path, in_dim, device), tau)
        preds[lbl] = predict(models[lbl][0], Xm, device)

    # ---- headline test metrics, each model vs its OWN quantile target --------
    print(f"\nTEST samples: {len(y):,}   (dataset {DATASET_NPZ})")
    print(f"{'model':22s} {'tau':>5s} {'MAE':>7s} {'RMSE':>7s} {'pinball@tau':>12s} "
          f"{'cover':>7s} {'target':>7s} {'gap':>8s}")
    print("-" * 80)
    for lbl, tau, *_ in specs:
        p = preds[lbl]
        cov = coverage(p, y)
        print(f"{lbl:22s} {tau:>5g} {mae(p, y):>7.3f} {rmse(p, y):>7.3f} "
              f"{pinball(p, y, tau):>12.5f} {cov:>7.4f} {1-tau:>7.3f} {cov-(1-tau):>+8.4f}")

    # ---- feature-conditioned coverage (the trustworthy calibration view) -----
    for title, vals, bands, unit in (
        ("DISTANCE band", dist, DIST_BANDS, "m"),
        ("PRED-time band", None, HORIZON_BANDS, "s"),
    ):
        print(f"\nCoverage gap (cover − own target) by {title}:")
        hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}{unit}" for lo, hi in bands)
        print(f"{'model':22s} {hdr}")
        for lbl, tau, *_ in specs:
            p = preds[lbl]
            v = p if vals is None else vals
            cells = []
            for lo, hi in bands:
                m = (v >= lo) & (v < hi)
                cells.append(f"{coverage(p[m], y[m]) - (1-tau):+6.3f}" if m.any() else "   -  ")
            print(f"{lbl:22s} " + "   ".join(cells))
    print("(0 = perfectly calibrated in that band; negative = under-covers)")

    if args.skip_area:
        return

    # ---- matched-coverage reachable area (tau-fair; lower m² = tighter) ------
    # Base 6-feature MLPs only: the full-pitch grid pass rebuilds features per
    # cell, and the vmax variants would need per-trail vmax (a match reload);
    # the vmax models are compared on the test metrics above.
    print("\n" + "=" * 80)
    print("Matched-coverage reachable area (m², mean over held-out trails; "
          "each model tuned to the SAME coverage -> comparable across tau).")
    area_specs = [(lbl, tau) for lbl, tau, _, in_dim, _ in specs if in_dim == 6]
    labels = [lbl for lbl, _ in area_specs]
    for h in HORIZONS_S:
        trails = load_split_trails(TEST_SPLIT, h, max_matches=MAX_MATCHES)
        trails = trails.subsample(MAX_TRAILS, seed=SEED + 1)
        if len(trails) == 0:
            print(f"\nΔt={h:.0f}s: no held-out trails")
            continue

        pred_tgt, grids, cell_area = {}, {}, None
        for lbl, _ in area_specs:
            m = models[lbl][0]

            @torch.no_grad()
            def fn(tf, m=m):
                return m(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()

            pred_tgt[lbl] = tta_to_targets(fn, trails)
            grids[lbl], cell_area = grid_tta(fn, trails, res=GRID_RES)

        print(f"\nΔt={h:.0f}s  |  {len(trails):,} held-out trails")
        print(f"{'coverage':>8s} " + " ".join(f"{lbl:>18s}" for lbl in labels))
        print("-" * (10 + 19 * len(labels)))
        for q in COVERAGES:
            areas = {lbl: matched_coverage_area(pred_tgt[lbl], grids[lbl], cell_area, q)["mean_area"]
                     for lbl in labels}
            print(f"{q*100:>7.0f}% " + " ".join(f"{areas[lbl]:>18.1f}" for lbl in labels))


if __name__ == "__main__":
    main()
