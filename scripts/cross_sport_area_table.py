"""Cross-sport replica of Table 2 (tab:main): for ONE test dataset (selected via
the KOGNIA_DATASET env var, as usual), score the three sport-trained MLP
checkpoints (football/basketball/nfl) plus that sport's own fitted Two-segment
baseline, on MAE / RMSE / pinball (full held-out split) and matched-coverage
reachable area @99% at Delta t = 1s (500 test trails, same protocol as Table 2).

    KOGNIA_DATASET=ffpc_match python -m scripts.cross_sport_area_table
    KOGNIA_DATASET=nba_match  python -m scripts.cross_sport_area_table
    KOGNIA_DATASET=bdb_match  python -m scripts.cross_sport_area_table

Checkpoint per training sport is pinned explicitly below rather than resolved
via config.PINBALL_TAU, because outputs/{nba,bdb}_match/mlp.pt predate the
current PINBALL_TAU=0.01 default (an older tau=0.05 run) while mlp_tau0.01.pt
is the fresh tau=0.01 checkpoint there; outputs/ffpc_match/mlp.pt IS the fresh
tau=0.01 checkpoint (retrained after the 2026-07-04 dataset rebuild). Verified
by reproducing each sport's own in-domain numbers already in the paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import DATASET, DATASET_NPZ, N_FEATURES, ROOT, outputs_dir  # noqa: E402
from src.evaluation.metrics import coverage, mae, rmse  # noqa: E402
from src.evaluation.reachable_area import grid_tta, matched_coverage_area, tta_to_targets  # noqa: E402
from src.inference.heatmap import pitch_grid  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.motion_models import from_params  # noqa: E402
from src.pipeline.features import transform  # noqa: E402

TAU = 0.01
RA_HORIZON_S = 1.0
RA_TRAILS = 500
RA_RES = 1.0
RA_COV = 0.99
SEED = 0

TRAIN_CKPT = {
    "football (ffpc)":  ("ffpc_match", "mlp.pt"),
    "basketball (nba)": ("nba_match", "mlp_tau0.01.pt"),
    "nfl (bdb)":        ("bdb_match", "mlp_tau0.01.pt"),
}


def pinball(pred, y, tau=TAU):
    err = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpc-ckpt", type=str, default=TRAIN_CKPT["football (ffpc)"][1],
                    help="checkpoint name under outputs/ffpc_match/ for the football MLP")
    args = ap.parse_args()
    TRAIN_CKPT["football (ffpc)"] = ("ffpc_match", args.ffpc_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nominal = 1.0 - TAU

    d = np.load(DATASET_NPZ)
    te = d["split"] == 2
    X = d["X"][te]
    y = d["y"][te].astype(np.float64)
    dist = d["meta_d"][te].astype(np.float64)
    v = d["meta_v"][te].astype(np.float64)
    n = len(y)
    print(f"TEST dataset={DATASET}   TEST samples: {n:,}", flush=True)

    mlps = {}
    for label, (train_ds, ckpt_name) in TRAIN_CKPT.items():
        m = TTAModel(in_dim=N_FEATURES).to(device)
        m.load_state_dict(torch.load(outputs_dir(train_ds) / ckpt_name, map_location=device))
        m.eval()
        mlps[label] = m

    @torch.no_grad()
    def mlp_predict_X(m, Xa, chunk=1_000_000):
        out = np.empty(len(Xa), dtype=np.float32)
        for s in range(0, len(Xa), chunk):
            e = min(s + chunk, len(Xa))
            out[s:e] = m(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
        return out.astype(np.float64)

    # ---- own-sport fitted Two-segment baseline --------------------------------
    mm_path = outputs_dir(DATASET) / "motion_models.json"
    fitted = json.loads(mm_path.read_text())
    two_seg = from_params("TwoSegment", fitted["TwoSegment"]["params"])

    # ---- full held-out split: MAE / RMSE / pinball / coverage -----------------
    rows = {}
    for label, m in mlps.items():
        pred = mlp_predict_X(m, X)
        rows[label] = {"mae": mae(pred, y), "rmse": rmse(pred, y),
                        "pinball": pinball(pred, y), "coverage": coverage(pred, y)}

    ts_sub = min(n, 500_000)
    idx = np.random.default_rng(SEED).choice(n, size=ts_sub, replace=False)
    tf_sub = {"d": dist[idx], "v": v[idx],
              "cos_t": X[idx, 1].astype(np.float64), "sin_t": X[idx, 2].astype(np.float64)}
    y_sub = y[idx]
    pred_ts = np.asarray(two_seg.predict(tf_sub), dtype=np.float64)
    rows["Two-segment"] = {"mae": mae(pred_ts, y_sub), "rmse": rmse(pred_ts, y_sub),
                            "pinball": pinball(pred_ts, y_sub), "coverage": coverage(pred_ts, y_sub)}

    print(f"\n{'model':20s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s}"
          f"   (target coverage {nominal:.3f}; Two-segment on {ts_sub:,}-row subsample)")
    print("-" * 66)
    for label, r in rows.items():
        print(f"{label:20s} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['pinball']:>9.5f} {r['coverage']:>9.4f}")

    # ---- Delta t = 1s matched-coverage area @99% -------------------------------
    print(f"\nbuilding Delta t={RA_HORIZON_S:.0f}s TEST trails ...", flush=True)
    trails = load_split_trails(2, RA_HORIZON_S, max_matches=1).subsample(RA_TRAILS, seed=SEED + 1)
    print(f"  {len(trails)} trails", flush=True)

    cells, _, _ = pitch_grid(RA_RES)
    cell_area = RA_RES * RA_RES

    def mlp_grid_fn(m):
        @torch.no_grad()
        def fn(tf):
            return m(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()
        return fn

    area_fns = {label: mlp_grid_fn(m) for label, m in mlps.items()}
    area_fns["Two-segment"] = lambda tf: two_seg.predict(tf)

    print(f"\n{'model':20s} area@{int(RA_COV*100)}%cov(m2)")
    print("-" * 40)
    for label, fn in area_fns.items():
        t0 = time.perf_counter()
        pred_tgt = tta_to_targets(fn, trails)
        grid, ca = grid_tta(fn, trails, res=RA_RES, chunk_states=64)
        mc = matched_coverage_area(pred_tgt, grid, ca, RA_COV)
        print(f"{label:20s} {mc['mean_area']:8.1f}   ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
