"""Rank SUVAT / MLP on the reachable-area metric -- fairly.

A single hit_ratio_min gate is misleading because the models sit at different
operating points (SUVAT covers ~100% with a large area; the learned models
cover less with a tiny area). To compare them fairly we tune each model to a
*common coverage* (the matching quantile of its own arrival-time predictions)
and then compare the mean reachable area it needs. The model that covers the
same fraction of actually-reached points with the least pitch area wins.

Run:  python -m scripts.rank_reachable_area
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import GRID_RES, MODEL_PT, SEED  # noqa: E402
from src.evaluation.reachable_area import (  # noqa: E402
    grid_tta,
    matched_coverage_area,
    tta_to_targets,
)

TEST_SPLIT = 2
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402

HORIZONS_S = [1.0, 5.0, 8.0, 12.0]    # 1/5/8 in-range; 12s is BEYOND the 8s window
COVERAGES = [0.80, 0.85, 0.90, 0.95, 0.99]
MAX_TRAILS = 2000
MAX_MATCHES = 1                       # held-out test matches to draw trails from (cached)
MODELS = ("SUVAT", "MLP")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp = TTAModel().to(device)
    mlp.load_state_dict(torch.load(MODEL_PT, map_location=device))
    mlp.eval()

    @torch.no_grad()
    def mlp_fn(tf):
        return mlp(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()

    fns = {
        "SUVAT": lambda tf: suvat_time(tf["v"], tf["cos_t"], tf["d"]),
        "MLP": mlp_fn,
    }

    horizons = [float(a) for a in sys.argv[1:]] or HORIZONS_S
    print("Reachable-area ranking at MATCHED coverage -- mean area (m²), lower is better.")
    print(f"(pitch = {match_pitch_area():.0f} m²; areas near it = saturation to the whole pitch)\n")

    for h in horizons:
        trails = load_split_trails(TEST_SPLIT, h, max_matches=MAX_MATCHES)
        trails = trails.subsample(MAX_TRAILS, seed=SEED + 1)
        if len(trails) == 0:
            print(f"Δt={h:.0f}s: no held-out trails (run too short for this horizon)\n")
            continue

        pred_tgt, grids, cell_area = {}, {}, None
        for name in MODELS:
            pred_tgt[name] = tta_to_targets(fns[name], trails)
            grids[name], cell_area = grid_tta(fns[name], trails, res=GRID_RES)

        beyond = "  (BEYOND 8s training range)" if h > 8.0 else ""
        print(f"Δt={h:.0f}s  |  {len(trails):,} held-out trails{beyond}")
        print(f"{'coverage':>8s} " + " ".join(f"{m:>9s}" for m in MODELS)
              + f"  {'SUVAT/MLP':>10s}  winner")
        print("-" * 50)
        for q in COVERAGES:
            areas = {m: matched_coverage_area(pred_tgt[m], grids[m], cell_area, q)["mean_area"]
                     for m in MODELS}
            best = min(MODELS, key=lambda m: areas[m])
            r_suvat = areas["SUVAT"] / areas["MLP"] if areas["MLP"] else float("nan")
            print(f"{q*100:>7.0f}% " + " ".join(f"{areas[m]:>9.1f}" for m in MODELS)
                  + f"  {r_suvat:>9.2f}x  {best}")
        print()


def match_pitch_area():
    from src.config import PITCH_L, PITCH_W
    return PITCH_L * PITCH_W


if __name__ == "__main__":
    main()
