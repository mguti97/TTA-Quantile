"""Cross-dataset evaluation: score a model trained on one sport against another
sport's held-out test set (or its own, for the in-domain reference).

The 6 input features and their normalisation constants (D_NORM, V_NORM, A_NORM)
are shared, sport-agnostic physics constants (src/config.py), so a checkpoint
trained on one dataset's samples.npz scores directly on another's -- no
retraining, no feature adaptation. This is the one piece the single-dataset
scripts (scripts/evaluate.py, scripts/compare_models.py) don't provide: they
hardcode model-checkpoint == test-dataset via the global KOGNIA_DATASET env
var, so this script builds every path explicitly instead of relying on it.

    python -m scripts.cross_eval --train ffpc_match --test bdb_match
    python -m scripts.cross_eval --train bdb_match  --test bdb_match --tau 0.05

Reports MAE/RMSE/pinball/coverage for the MLP, the TEST dataset's own fitted
motion-model baselines (outputs/<test>/motion_models.json -- refit per sport,
the legitimate physical comparison), and raw SUVAT for incidental context only
(SUVAT's role in this project is cleaning the sample noise floor at build time,
not the headline physical baseline -- see the motion models for that).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, PINBALL_TAU, ROOT, SEED, outputs_dir  # noqa: E402
from src.evaluation.metrics import coverage, mae, rmse  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.motion_models import from_params  # noqa: E402

# Some fitted motion models are expensive per row (e.g. WuSwartz grids 240
# acceleration directions per sample -> an (N, 240) float64 buffer; at a full
# multi-match test split of tens of millions of rows that's tens of GB and
# very slow). fit_motion_models.py/compare_models.py already size their own
# physical-model work to a few hundred/thousand trails for the same reason;
# mirror that here for the motion-model comparison only -- MLP and SUVAT are
# both cheap (GPU batch / closed-form) and still run on the FULL test split.
MOTION_MODEL_MAX_ROWS = 200_000


def pinball(pred, y, tau=PINBALL_TAU):
    err = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def model_ckpt(train_ds: str, tau: float) -> Path:
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    return outputs_dir(train_ds) / f"mlp{suffix}.pt"


def report(name: str, pred: np.ndarray, y: np.ndarray, tau: float) -> dict:
    return {
        "name": name,
        "mae": mae(pred, y),
        "rmse": rmse(pred, y),
        "pinball": pinball(pred, y, tau),
        "coverage": coverage(pred, y),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="dataset the MLP checkpoint was trained on")
    ap.add_argument("--test", required=True, help="dataset whose held-out test split to score against")
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile the checkpoint was trained at (default {PINBALL_TAU:g})")
    ap.add_argument("--no-motion-models", action="store_true",
                    help="skip the fitted-physical-model baselines (MLP + SUVAT only)")
    ap.add_argument("--motion-models-file", default="motion_models.json",
                    help="filename under outputs/<test>/ to load fitted motion models from "
                         "(default motion_models.json; ffpc_match's gate=0.95 fit -- matching "
                         "bdb/nba's default -- lives at motion_models_tau0.05_gate0.95.json)")
    args = ap.parse_args()
    tau = args.tau

    # honour the active suffix (caps tag etc.) so train and test come from the
    # same preprocessing generation
    from src.config import SAMPLES_SUFFIX
    npz_path = DATA_DIR / args.test / "processed" / f"samples{SAMPLES_SUFFIX}.npz"
    d = np.load(npz_path)
    te = d["split"] == 2
    X, y = d["X"][te], d["y"][te]
    dist = d["meta_d"][te]
    v = d["meta_v"][te]
    theta_deg = d["meta_theta_deg"][te]
    t_suvat = d["meta_t_suvat"][te]
    nominal = 1.0 - tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = model_ckpt(args.train, tau)
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} -- train it first (scripts.train, --train {args.train})")
    model = TTAModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    pred = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 1_000_000):
            e = min(i + 1_000_000, len(X))
            pred[i:e] = model(torch.from_numpy(X[i:e]).to(device)).cpu().numpy()

    rows = [report(f"MLP({args.train})", pred, y, tau), report("SUVAT", t_suvat, y, tau)]

    if not args.no_motion_models:
        mm_path = outputs_dir(args.test) / args.motion_models_file
        if mm_path.exists():
            n = len(y)
            if n > MOTION_MODEL_MAX_ROWS:
                idx = np.random.default_rng(SEED).choice(n, size=MOTION_MODEL_MAX_ROWS, replace=False)
                print(f"(motion models scored on a {MOTION_MODEL_MAX_ROWS:,}-row subsample "
                      f"of {n:,} test rows -- WuSwartz's per-row direction grid is too costly "
                      f"over the full split; MLP/SUVAT above use all {n:,})")
            else:
                idx = slice(None)
            dist_s, v_s, theta_s, y_s = dist[idx], v[idx], theta_deg[idx], y[idx]
            theta_rad = np.radians(theta_s)
            tf = {"d": dist_s.astype(np.float64), "v": v_s.astype(np.float64),
                  "cos_t": np.cos(theta_rad), "sin_t": np.sin(theta_rad)}
            fitted = json.loads(mm_path.read_text())
            for name, entry in fitted.items():
                if name == "_meta":
                    continue
                mm = from_params(name, entry["params"])
                rows.append(report(f"{name}({args.test}-fit)", mm.predict(tf), y_s, tau))
        else:
            print(f"(no {mm_path} -- run scripts.fit_motion_models --  physical baselines skipped)")

    print(f"train={args.train}  test={args.test}  tau={tau:g}  TEST samples: {len(y):,}\n")
    print(f"{'model':28s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name']:28s} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['pinball']:>9.5f} "
              f"{r['coverage']:>9.4f}")
    print(f"(target coverage 1-tau = {nominal:.3f})")


if __name__ == "__main__":
    main()
