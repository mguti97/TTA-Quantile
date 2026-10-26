"""Reachable-area validation metric (Renkin et al., ICCS 2022) for SUVAT/MLP.

For each horizon Δt we reconstruct exact-Δt trails on the held-out *test* split,
then score every model on:

  * hit_ratio  -- coverage: fraction of trails where tau_pred(x_t) <= Δt
  * mean_area  -- mean pitch area (m^2) of the reachable set {x: tau_pred(x)<=Δt}
                  over *correct* trails; this is score^-1, lower = tighter
  * score      -- 0 if hit_ratio < HIT_RATIO_MIN else 1/mean_area (paper eq. 4)

The Δt = 1 s row is directly comparable to the paper's Fig. 3 (best model
71.7 m^2).

    python -m scripts.eval_reachable_area                # tau = config.PINBALL_TAU
    python -m scripts.eval_reachable_area --tau 0.01     # loads mlp_tau0.01.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import GRID_RES, MODEL_PT, OUTPUTS_DIR, PINBALL_TAU, SEED  # noqa: E402
from src.evaluation.reachable_area import (  # noqa: E402
    reachable_areas,
    score,
    tta_to_targets,
)
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402

# ---- knobs -----------------------------------------------------------------
TEST_SPLIT = 2
# Sweep spans the 0.25-8 s training window. Beyond ~8 s a 7 m/s player can reach
# >55 m, so the reachable set is clipped by the pitch boundary and mean_area
# saturates toward the full-pitch area, losing its power to discriminate models --
# hence 8 s is the top of the sweep. Δt = 1-2 s stays the discriminating headline.
HORIZONS_S = [1.0, 2.0, 3.0, 5.0, 8.0]
HIT_RATIO_MIN = 0.90          # primary gate (paper used 0.99975)
HIT_RATIO_MINS = [0.80, 0.90, 0.95, 0.99]  # sensitivity sweep (fixed across Δt)
MAX_TRAILS_PER_HORIZON = 1500  # area needs a full-pitch grid pass per trail
MAX_MATCHES = 1                # held-out test matches to draw trails from (cached);
                               # raise toward all 9 for a larger sample, None = all


def model_path(tau: float) -> Path:
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def make_predictors(device, tau=PINBALL_TAU):
    """Return {name: pred_fn} where pred_fn maps a transform() dict -> times."""
    mlp = TTAModel().to(device)
    mlp.load_state_dict(torch.load(model_path(tau), map_location=device))
    mlp.eval()

    @torch.no_grad()
    def mlp_fn(tf):
        return mlp(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()

    return {
        "SUVAT": lambda tf: suvat_time(tf["v"], tf["cos_t"], tf["d"]),
        "MLP": lambda tf: mlp_fn(tf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = make_predictors(device, tau=args.tau)

    print(f"Reachable-area metric (Renkin et al. 2022)  |  tau={args.tau:g}  |  "
          f"hit_ratio_min={HIT_RATIO_MIN}")
    print(f"held-out TEST split  |  up to {MAX_TRAILS_PER_HORIZON:,} trails / horizon\n")

    # Cache raw per-trail predictions + areas so the score can be recomputed at
    # any hit_ratio_min without re-running inference (the gate is the only part
    # that depends on the threshold).
    raw = {name: [] for name in predictors}   # raw[name][i] = (pred_tgt, areas)
    results = {name: [] for name in predictors}
    horizons_run = []
    for h in HORIZONS_S:
        trails = load_split_trails(TEST_SPLIT, h, max_matches=MAX_MATCHES)
        trails = trails.subsample(MAX_TRAILS_PER_HORIZON, seed=SEED + int(round(h)))
        if len(trails) == 0:
            continue
        horizons_run.append(h)

        for name, pred_fn in predictors.items():
            pred_tgt = tta_to_targets(pred_fn, trails)
            areas = reachable_areas(pred_fn, trails, res=GRID_RES)
            raw[name].append((pred_tgt, areas))
            results[name].append(score(pred_tgt, areas, h, HIT_RATIO_MIN))

    if not horizons_run:
        print("No held-out trails at any horizon -- check the dataset/split.")
        return

    # ---- per-horizon table (primary threshold) ------------------------------
    print(f"{'Δt':>4s} {'model':6s} {'n':>7s} {'hit_ratio':>10s} "
          f"{'mean_area':>10s} {'r_eff':>7s} {'score(1/m²)':>12s}")
    print("-" * 64)
    for i, h in enumerate(horizons_run):
        for name in predictors:
            r = results[name][i]
            gate = "" if r["passes_gate"] else "  (gated→0)"
            print(f"{h:>4.0f} {name:6s} {r['n']:>7,d} {r['hit_ratio']:>10.4f} "
                  f"{r['mean_area']:>9.1f}m² {r['r_eff']:>6.2f}m {r['score']:>12.6f}{gate}")
        print()

    # ---- effective-radius curve: cross-Δt comparable, shows convergence -----
    print("=" * 64)
    print(f"Effective radius r_eff [m] vs Δt  (gate={HIT_RATIO_MIN:.0%}; "
          "lower=tighter, watch the model gap close as Δt grows):")
    print(f"  {'Δt':>4s} " + " ".join(f"{name:>8s}" for name in predictors))
    for i, h in enumerate(horizons_run):
        cells = " ".join(f"{results[name][i]['r_eff']:>8.2f}" for name in predictors)
        print(f"  {h:>4.0f} {cells}")

    # ---- threshold sensitivity: is the ranking stable across the gate? ------
    print("\n" + "=" * 64)
    print("Threshold sensitivity (mean_area [m²] at Δt = 1 s; '—' = gated→0):")
    print(f"  {'gate':>6s} " + " ".join(f"{name:>8s}" for name in predictors))
    for hrm in HIT_RATIO_MINS:
        cells = []
        for name in predictors:
            pred_tgt, areas = raw[name][0]
            r = score(pred_tgt, areas, horizons_run[0], hrm)
            cells.append(f"{r['mean_area']:>8.1f}" if r["passes_gate"] else f"{'—':>8s}")
        print(f"  {hrm:>6.0%} " + " ".join(cells))

    # ---- headline: Δt = 1 s, comparable to the paper's Fig. 3 ---------------
    print("\n" + "=" * 64)
    print("Δt = 1 s headline (paper Fig. 3 best model: 71.7 m² / r_eff 4.78 m):")
    for name in predictors:
        if results[name]:
            r = results[name][0]
            print(f"  {name:6s} mean reachable area = {r['mean_area']:6.1f} m²"
                  f"  (r_eff {r['r_eff']:.2f} m, coverage {r['hit_ratio']*100:5.2f}%)")


if __name__ == "__main__":
    main()
