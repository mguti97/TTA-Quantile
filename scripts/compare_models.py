"""Head-to-head: SUVAT vs MLP.

Reports test MAE + pinball on the held-out split, a distance-stratified MAE, and
the spatial roughness of each model's arrival surface on one frame (a proxy for
the visual "noise").  python -m scripts.compare_models
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import assign_splits  # noqa: E402
from src.config import (  # noqa: E402
    DATASET_NPZ,
    GRID_RES,
    MODEL_PT,
    PINBALL_TAU,
    PITCH_L,
    PITCH_W,
    SEED,
)
from src.data_loader import load_match  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    DIST_BANDS,
    HORIZON_BANDS,
    coverage,
    mae,
    rmse,
)
from src.evaluation.reachable_area import (  # noqa: E402
    build_trails,
    reachable_areas,
    score,
    tta_to_targets,
)
from src.inference.heatmap import pitch_grid  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.features import transform  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402


# reachable-area metric (Renkin et al. 2022): scored at the paper's Δt = 1 s,
# the horizon where the reachable set sits well inside the pitch.
RA_HORIZON_S = 1.0
RA_HIT_RATIO_MIN = 0.90
RA_MAX_TRAILS = 1500


def pinball(pred, y, tau=PINBALL_TAU):
    err = y - pred
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def roughness(Z):
    """Mean |discrete Laplacian| -- high-frequency energy of a surface."""
    lap = np.abs(4 * Z[1:-1, 1:-1] - Z[:-2, 1:-1] - Z[2:, 1:-1] - Z[1:-1, :-2] - Z[1:-1, 2:])
    return float(lap.mean())


def states_for_frame(match, frame):
    out = []
    for seg_id, lo, hi in match.segment_slices():
        if not (lo <= frame < hi):
            continue
        for p in range(match.P):
            if match.in_field[frame, p] != 1:
                continue
            tr = clean_track(match.positions[lo:hi, p, :], match.in_field[lo:hi, p], p, seg_id)
            li = frame - lo
            if li < tr.L and tr.state_valid[li] and not np.isnan(tr.pos[li, 0]):
                out.append((tr.pos[li], tr.vel[li], tr.acc[li]))
        break
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATASET_NPZ)
    te = d["split"] == 2
    X, y = d["X"][te], d["y"][te]
    dist = d["meta_d"][te]
    horizon = d["meta_horizon"][te]
    nominal = 1.0 - PINBALL_TAU

    mlp = TTAModel().to(device)
    mlp.load_state_dict(torch.load(MODEL_PT, map_location=device))
    mlp.eval()

    with torch.no_grad():
        p_mlp = mlp(torch.from_numpy(X).to(device)).cpu().numpy()
    p_suvat = d["meta_t_suvat"][te]

    preds = {"SUVAT": p_suvat, "MLP": p_mlp}

    print(f"TEST samples: {len(y):,}\n")
    print(f"{'model':6s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s}")
    print("-" * 46)
    for name, p in preds.items():
        print(f"{name:6s} {mae(p, y):>8.3f} {rmse(p, y):>8.3f} {pinball(p, y):>9.5f} "
              f"{coverage(p, y):>9.4f}")
    print(f"(target coverage 1-tau = {nominal:.3f})")

    print("\nMAE by distance band:")
    hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}m" for lo, hi in DIST_BANDS)
    print(f"{'model':6s} {hdr}")
    for name, p in preds.items():
        cells = []
        for lo, hi in DIST_BANDS:
            m = (dist >= lo) & (dist < hi)
            cells.append(f"{mae(p[m], y[m]):.3f}" if m.any() else "  -  ")
        print(f"{name:6s} " + "    ".join(cells))

    # ---- surface roughness on one frame -------------------------------------
    match = load_match()
    states = states_for_frame(match, 9300)
    cells, nx, ny = pitch_grid(GRID_RES)
    G = cells.shape[0]
    r = {k: [] for k in preds}
    for pos, vel, acc in states:
        tf = transform(np.broadcast_to(pos, (G, 2)), np.broadcast_to(vel, (G, 2)),
                       np.broadcast_to(acc, (G, 2)), cells)
        with torch.no_grad():
            sm = mlp(torch.from_numpy(tf["X"]).to(device)).cpu().numpy().reshape(nx, ny)
        ss = suvat_time(tf["v"], tf["cos_t"], tf["d"]).reshape(nx, ny)
        r["MLP"].append(roughness(sm))
        r["SUVAT"].append(roughness(ss))

    print(f"\nSurface roughness (mean |Laplacian|, frame 9300, {len(states)} players):")
    for name in preds:
        print(f"  {name:6s} {np.mean(r[name]):.4f}")

    # ---- inference time: one full-pitch frame = 22 players x 7140 cells ------
    import time

    N = 22 * 7140  # spec §9 batch size (157,080)
    rng = np.random.default_rng(0)
    Xb = X[rng.integers(0, len(X), N)]          # real feature distribution
    v_b, cos_b, d_b = Xb[:, 3] * 12.0, Xb[:, 1], Xb[:, 0] * 120.0
    Xb_t = torch.from_numpy(Xb)
    cuda = device.type == "cuda"

    def timed(fn, reps=10):
        fn()                                     # warm-up
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
            if cuda:
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3  # ms

    @torch.no_grad()
    def run_mlp():
        return mlp(Xb_t.to(device)).cpu().numpy()

    times = {
        "SUVAT": timed(lambda: suvat_time(v_b, cos_b, d_b)),
        "MLP": timed(run_mlp),
    }
    print(f"\nInference time, full frame ({N:,} rows = 22 players x 7140 cells):")
    for name in preds:
        dev = "GPU" if (name == "MLP" and cuda) else "CPU"
        print(f"  {name:6s} {times[name]:7.2f} ms/frame ({dev})")

    # ---- reachable-area metric (Renkin et al. 2022) at Δt = 1 s -------------
    @torch.no_grad()
    def mlp_grid(tf):
        return mlp(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()

    ra_fns = {
        "SUVAT": lambda tf: suvat_time(tf["v"], tf["cos_t"], tf["d"]),
        "MLP": mlp_grid,
    }
    trails = build_trails(match, RA_HORIZON_S)
    # single loaded match -> match id 0 for every trail (keeps the new
    # dict-based assign_splits signature happy; player/block modes still work)
    split = assign_splits({
        "frame": trails.frame,
        "segment": trails.segment,
        "player": trails.player,
        "match": np.zeros_like(trails.frame),
    })
    trails = trails.select(split == 2).subsample(RA_MAX_TRAILS, seed=SEED + 1)
    ra = {}
    for name, fn in ra_fns.items():
        pred_tgt = tta_to_targets(fn, trails)
        areas = reachable_areas(fn, trails, res=GRID_RES)
        ra[name] = score(pred_tgt, areas, RA_HORIZON_S, RA_HIT_RATIO_MIN)
    print(f"\nReachable-area metric (Δt={RA_HORIZON_S:.0f}s, {len(trails):,} test trails, "
          f"hit_ratio_min={RA_HIT_RATIO_MIN}; paper best 71.7 m²):")
    for name in preds:
        gate = "" if ra[name]["passes_gate"] else "  (below gate → score 0)"
        print(f"  {name:6s} coverage {ra[name]['hit_ratio']*100:5.2f}%   "
              f"area {ra[name]['mean_area']:6.1f} m²{gate}")

    # ---- quantile calibration: true-time vs predicted-time binning ----------
    # Binning coverage by the OUTCOME (true_t) ramps 0->1 even for a perfect
    # model (conditioning-on-outcome artifact); binning by PREDICTED time
    # (a function of the inputs) stays flat ~nominal when calibration is good.
    hdr = "  ".join(f"{lo}-{int(hi)}s" for lo, hi in HORIZON_BANDS)

    def cov_by(bin_vals_fn):
        for name, p in preds.items():
            bins = bin_vals_fn(p)
            cells = []
            for lo, hi in HORIZON_BANDS:
                m = (bins >= lo) & (bins < hi)
                cells.append(f"{coverage(p[m], y[m]):.3f}" if m.any() else "  -  ")
            print(f"{name:6s} " + "   ".join(cells))

    print(f"\nCoverage by TRUE-time band (binned by observed y; target {nominal:.3f}):")
    print(f"{'model':6s} {hdr}")
    cov_by(lambda p: horizon)
    print(f"\nCoverage by PRED-time band (binned by each model's prediction; target {nominal:.3f}):")
    print("  -> stays flat ~target if calibrated; confirms the true-time ramp is an artifact")
    print(f"{'model':6s} {hdr}")
    cov_by(lambda p: p)

    # ---- consolidated table -------------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'model':6s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s} "
          f"{'roughness':>10s} {'infer(ms)':>10s} {'area@1s':>9s}")
    print("-" * 84)
    for name, p in preds.items():
        print(f"{name:6s} {mae(p, y):>8.3f} {rmse(p, y):>8.3f} {pinball(p, y):>9.5f} "
              f"{coverage(p, y):>9.4f} {np.mean(r[name]):>10.4f} {times[name]:>10.2f} "
              f"{ra[name]['mean_area']:>7.1f}m²")


if __name__ == "__main__":
    main()
