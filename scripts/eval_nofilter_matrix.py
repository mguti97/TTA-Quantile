"""Sensitivity of the SUVAT label filter: canonical vs no-filter MLP, both
scored on both the canonical (filtered) and the no-filter test split.
Replaces the un-persisted inline script behind outputs/ffpc_match/nofilter_eval_matrix.log.
Run: python -m scripts.eval_nofilter_matrix
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASET_NPZ, OUTPUTS_DIR, PINBALL_TAU, PROCESSED_DIR, SAMPLES_SUFFIX  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
tau = PINBALL_TAU
print(f"device: {dev}  tau={tau}\n")

def load_model(p):
    m = TTAModel(in_dim=6).to(dev); m.load_state_dict(torch.load(p, map_location=dev)); m.eval(); return m

mlp = load_model(OUTPUTS_DIR / "mlp.pt")
mlp_nf = load_model(OUTPUTS_DIR / "mlp_nofilter.pt")
print(f"canonical MLP (filter ON): {OUTPUTS_DIR/'mlp.pt'}")
print(f"no-filter MLP (filter OFF): {OUTPUTS_DIR/'mlp_nofilter.pt'}\n")

# the nofilter npz is caps-independent (filter off -> caps never touch X/y)
NF_NPZ = PROCESSED_DIR / "samples_nofilter.npz"

def eval_on(npz_path, name):
    d = np.load(npz_path)
    te = d["split"] == 2
    X = torch.from_numpy(d["X"][te]).float()
    y = d["y"][te].astype(np.float64)
    v = d["meta_v"][te].astype(np.float64)          # raw m/s
    dd = d["meta_d"][te].astype(np.float64)         # raw m
    cos_t = np.cos(np.deg2rad(d["meta_theta_deg"][te].astype(np.float64)))
    tsuv = suvat_time(v, cos_t, dd)                 # current caps (10, 12)
    below = float((y < tsuv).mean())
    print(f"[{name}]  {npz_path.name}  test n={te.sum():,}  (below-SUVAT in this split: {below:.4%})")
    rows = []
    for label, model in (("SUVAT closed form", None), ("canonical MLP (filter ON)", mlp), ("no-filter MLP (filter OFF)", mlp_nf)):
        if model is None:
            pred = tsuv
        else:
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), 2_000_000):
                    preds.append(model(X[i:i+2_000_000].to(dev)).cpu().numpy().ravel())
            pred = np.concatenate(preds).astype(np.float64)
        err = y - pred
        rows.append((label, np.abs(err).mean(), np.sqrt((err**2).mean()),
                     np.maximum(tau*err, (tau-1)*err).mean(), (pred <= y).mean()))
    for label, mae, rmse, pb, cov in rows:
        print(f"  {label:<30} MAE {mae:.3f}  RMSE {rmse:.3f}  pinball {pb:.5f}  cover {cov:.4f}")
    print()

eval_on(DATASET_NPZ, "filtered (canonical, caps10_12 eval set)")
eval_on(NF_NPZ, "no-filter (all samples)")
