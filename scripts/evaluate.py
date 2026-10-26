"""Evaluate the trained model on the held-out test split (Section 10).

Reports overall MAE, the stratified breakdown vs the SUVAT baseline, and renders
a couple of qualitative per-player heatmaps.

    python -m scripts.evaluate                # tau = config.PINBALL_TAU
    python -m scripts.evaluate --tau 0.01     # loads mlp_tau0.01.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_NPZ, MODEL_PT, OUTPUTS_DIR, PINBALL_TAU  # noqa: E402
from src.data_loader import load_match  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    calibration,
    coverage,
    format_calibration,
    format_table,
    mae,
    rmse,
    stratified,
)
from src.evaluation.visualize import render_player_heatmap  # noqa: E402
from src.inference.heatmap import heatmaps_for_frame  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402


def model_path(tau: float) -> Path:
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    args = ap.parse_args()
    tau = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    te = d["split"] == 2
    X, y = d["X"][te], d["y"][te]
    meta = {k[5:]: d[k][te] for k in d.files if k.startswith("meta_")}

    model = TTAModel().to(device)
    model.load_state_dict(torch.load(model_path(tau), map_location=device))
    model.eval()
    pred = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 1_000_000):
            e = min(i + 1_000_000, len(X))
            pred[i:e] = model(torch.from_numpy(X[i:e]).to(device)).cpu().numpy()

    pred_suvat = meta["t_suvat"]
    print(f"TEST samples: {len(y):,}   tau={tau:g}")
    print(f"Overall MAE   model: {mae(pred, y):.3f}s   SUVAT: {mae(pred_suvat, y):.3f}s")
    print(f"Overall RMSE  model: {rmse(pred, y):.3f}s   SUVAT: {rmse(pred_suvat, y):.3f}s")
    nominal = 1.0 - tau
    print(f"Overall cover model: {coverage(pred, y):.4f}    SUVAT: {coverage(pred_suvat, y):.4f}"
          f"   (target {nominal:.3f})\n")
    rows = stratified(pred, pred_suvat, y, meta["d"], meta["v"], meta["theta_deg"])
    print(format_table(rows))

    # ---- quantile calibration: where does the tau=0.05 coverage drift? ------
    print("\nCalibration (empirical coverage by band):")
    cal = calibration(pred, y, meta["d"], meta["v"], meta["theta_deg"],
                      horizon=meta.get("horizon"))
    print(format_calibration(cal, nominal))

    # ---- qualitative: render heatmaps for one frame -------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    match = load_match()
    # frame with the most tracked players -- robust across datasets (NFL games are
    # far shorter than any fixed index, and occlusion varies frame to frame).
    frame = int(np.argmax(match.in_field.sum(axis=1)))
    states, labels = [], []
    for seg_id, lo, hi in match.segment_slices():
        if lo <= frame < hi:
            for p in range(match.P):
                if match.in_field[frame, p] != 1:
                    continue
                tr = clean_track(match.positions[lo:hi, p, :], match.in_field[lo:hi, p], p, seg_id)
                li = frame - lo
                if li < tr.L and tr.state_valid[li]:
                    states.append((tr.pos[li], tr.vel[li], tr.acc[li]))
                    labels.append(p)
            break

    if states:
        heat, suv, nx, ny = heatmaps_for_frame(model, states, device, return_suvat=True)
        # render the fastest-moving player as an illustrative case
        speeds = [np.hypot(*s[1]) for s in states]
        j = int(np.argmax(speeds))
        suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
        out = OUTPUTS_DIR / f"heatmap_frame{frame}_player{labels[j]}{suffix}.png"
        render_player_heatmap(
            heat[j], states[j][0], model_suvat=suv[j],
            title=f"{match.player_name[labels[j]]} | frame {frame} | "
                  f"v={speeds[j]:.1f} m/s  (white=model, red dashed=SUVAT)",
            path=out,
        )
        print(f"\nrendered {len(states)} players; sample -> {out}")


if __name__ == "__main__":
    main()
