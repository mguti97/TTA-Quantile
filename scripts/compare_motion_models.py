"""Compare the fitted theoretical motion models against the baselines.

Head-to-head on the held-out TEST split: the fitted Renkin (a-d) and Wu & Swartz
(e) motion models vs the UNFIT SUVAT baseline and the MLP. Reports MAE / RMSE /
pinball / coverage, MAE stratified by distance / speed / angle, and the Renkin
reachable-area metric at Δt = 1 s.

    python -m scripts.compare_motion_models                 # default 500k subsample
    python -m scripts.compare_motion_models --n 1000000
    python -m scripts.compare_motion_models --tau 0.01      # score the tau=0.01 MLP

--tau loads the MLP trained at that quantile (mlp_tau<t>.pt for non-default) and
scores pinball/coverage against ITS 1-tau target; the motion models are the same
fitted-at-0.95-gate physics either way, so read their coverage against 0.95.
The matched-coverage area table is tau-fair by construction.

Run scripts.fit_motion_models first (writes outputs/<dataset>/motion_models.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_motion_models import (  # noqa: E402
    load_split_trails,
    trail_capacities,
)
from src.config import (  # noqa: E402
    DATASET,
    DATASET_NPZ,
    GRID_RES,
    MODEL_PT,
    OUTPUTS_DIR,
    PINBALL_TAU,
    SEED,
    V_NORM,
)
from src.evaluation.metrics import (  # noqa: E402
    ANGLE_BANDS,
    DIST_BANDS,
    SPEED_BANDS,
    coverage,
    mae,
    rmse,
)
from src.evaluation.reachable_area import (  # noqa: E402
    grid_tta,
    matched_coverage_area,
    score,
    tta_to_targets,
)
from src.model.mlp import TTAModel  # noqa: E402
from src.model.motion_models import (  # noqa: E402
    ALL_SPECS,
    from_params,
    is_perplayer,
)
from src.pipeline.suvat import suvat_time  # noqa: E402

RA_HORIZON_S = 1.0
RA_HIT_RATIO_MIN = 0.95
RA_TRAILS = 500
RA_RES = 1.0
WS_DIRS = 180          # acceleration directions for Wu & Swartz at compare time
WS_CHUNK = 20_000      # rows per Wu & Swartz call (bounds the N x n_dirs array)
AB_CHUNK = 200_000     # rows per Anzer & Bauer call (bounds the N x n_t tables)

# learned-baseline / unfit names that are NOT loaded from motion_models.json
BASELINES = ("SUVAT", "MLP")


def pinball(pred, y, tau):
    err = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def mlp_path(tau):
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def chunked_predict(model, tf, chunk, keys=("d", "v", "cos_t", "sin_t")):
    """Apply a motion model in row-chunks (bounds memory for the grid model).

    ``keys`` selects the per-row columns handed to the model -- the per-player
    variants additionally need "vmax_pp"/"amax_pp".
    """
    N = tf["d"].shape[0]
    out = np.empty(N)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sub = {k: tf[k][s:e] for k in keys}
        out[s:e] = model.predict(sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000,
                    help="common test subsample for the apples-to-apples table")
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"quantile of the MLP to load/score (default {PINBALL_TAU:g})")
    ap.add_argument("--mlp-path", type=str, default=None,
                    help="explicit MLP checkpoint to score (overrides the tau-derived path)")
    ap.add_argument("--profile-path", type=str, default=None,
                    help="profile-MLP checkpoint (7 inputs: the 6 features + the "
                         "per-player v_max/V_NORM). Scored alongside the base MLP "
                         "so the learned per-player model sits in the same table "
                         "as the per-player physics models.")
    ap.add_argument("--models-json", nargs="*", default=[],
                    help="extra fitted-motion-model json files (e.g. the per-player "
                         "P1/P2 fits) scored alongside motion_models.json")
    args = ap.parse_args()
    tau = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nominal = 1.0 - tau

    # ---- load fitted motion-model params -----------------------------------
    mm_paths = [OUTPUTS_DIR / "motion_models.json"] + [Path(p) for p in args.models_json]
    mm_names, motion_models, papers = [], {}, {}
    for mm_path in mm_paths:
        fitted = json.loads(mm_path.read_text())
        for name in [k for k in fitted if k in ALL_SPECS]:
            m = from_params(name, fitted[name]["params"])
            if name.startswith("WuSwartz"):
                m.n_dirs = WS_DIRS
            mm_names.append(name)
            motion_models[name] = m
            papers[name] = fitted[name].get("paper", ALL_SPECS[name].paper)

    # ---- load test split ----------------------------------------------------
    d = np.load(DATASET_NPZ)
    te = d["split"] == 2
    X = d["X"][te]
    y = d["y"][te].astype(np.float64)
    tf_full = {
        "d": d["meta_d"][te].astype(np.float64),
        "v": d["meta_v"][te].astype(np.float64),
        "cos_t": X[:, 1].astype(np.float64),
        "sin_t": X[:, 2].astype(np.float64),
        # per-player capacity columns, stored per sample by scripts.build_dataset
        # (train-split-only table under KOGNIA_VMAX_SCOPE=train_split) -- exactly
        # the information the profile MLP's 7th feature carries.
        "vmax_pp": d["meta_vmax"][te].astype(np.float64),
        "amax_pp": d["meta_amax"][te].astype(np.float64),
    }
    theta = d["meta_theta_deg"][te]
    t_suvat = d["meta_t_suvat"][te].astype(np.float64)

    print(f"dataset={DATASET}   TEST samples: {len(y):,}")
    print(f"fitted models: {', '.join(mm_names)}\n")

    # ---- learned baselines on the FULL test set (cheap) --------------------
    mlp_pt = Path(args.mlp_path) if args.mlp_path else mlp_path(tau)
    mlp = TTAModel().to(device)
    mlp.load_state_dict(torch.load(mlp_pt, map_location=device))
    mlp.eval()
    print(f"MLP: {mlp_pt.name} (tau={tau:g}; pinball/coverage scored at "
          f"target {nominal:.3f})")

    def mlp_predict(Xa):
        out = np.empty(len(Xa))
        with torch.no_grad():
            for s in range(0, len(Xa), 1_000_000):
                e = min(s + 1_000_000, len(Xa))
                out[s:e] = mlp(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
        return out

    full = {
        "SUVAT": t_suvat,
        "MLP": mlp_predict(X),
    }
    baselines = list(BASELINES)
    if args.profile_path:
        profile = TTAModel(in_dim=7).to(device)
        profile.load_state_dict(torch.load(args.profile_path, map_location=device))
        profile.eval()
        Xp = np.concatenate(
            [X, (tf_full["vmax_pp"] / V_NORM).astype(np.float32)[:, None]], axis=1)

        def profile_predict(Xa):
            out = np.empty(len(Xa))
            with torch.no_grad():
                for s in range(0, len(Xa), 1_000_000):
                    e = min(s + 1_000_000, len(Xa))
                    out[s:e] = profile(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
            return out

        full["MLP+vmax"] = profile_predict(Xp)
        baselines.append("MLP+vmax")
        print(f"profile MLP: {Path(args.profile_path).name}")
    print(f"{'model':10s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s}"
          "   (FULL test, baselines only)")
    print("-" * 60)
    for name in baselines:
        p = full[name]
        print(f"{name:10s} {mae(p, y):>8.3f} {rmse(p, y):>8.3f} {pinball(p, y, tau):>9.5f} "
              f"{coverage(p, y):>9.4f}")

    # ---- common subsample: every model on the same rows --------------------
    n = min(args.n, len(y))
    idx = np.random.default_rng(SEED).choice(len(y), size=n, replace=False)
    idx.sort()
    ys = y[idx]
    tf_s = {k: v[idx] for k, v in tf_full.items()}
    theta_s = theta[idx]
    Xs = X[idx]

    preds = {name: full[name][idx] for name in baselines}
    print(f"\nrunning motion models on {n:,}-row common subsample ...", flush=True)
    for name, m in motion_models.items():
        t0 = time.perf_counter()
        chunk = n
        if name.startswith("WuSwartz"):
            chunk = WS_CHUNK
        elif name.startswith("AnzerBauer"):   # circle + faithful hull variants
            chunk = AB_CHUNK
        keys = (("d", "v", "cos_t", "sin_t", "vmax_pp", "amax_pp")
                if is_perplayer(name) else ("d", "v", "cos_t", "sin_t"))
        preds[name] = chunked_predict(m, tf_s, chunk, keys=keys)
        print(f"  {name:22s} {time.perf_counter() - t0:6.1f}s", flush=True)

    order = baselines + mm_names

    # ---- headline table -----------------------------------------------------
    print(f"\n{'='*64}\nCOMMON TEST SUBSAMPLE ({n:,} rows)   target coverage = {nominal:.3f}")
    print(f"{'model':22s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'pinball':>9s} {'coverage':>9s}  src")
    print("-" * 76)
    for name in order:
        p = preds[name]
        src = "baseline" if name in baselines else papers[name]
        print(f"{name:22s} {mae(p, ys):>8.3f} {rmse(p, ys):>8.3f} {pinball(p, ys, tau):>9.5f} "
              f"{coverage(p, ys):>9.4f}  {src}")

    # ---- stratified MAE -----------------------------------------------------
    def band_table(title, vals, bands, unit):
        print(f"\nMAE by {title} band:")
        hdr = "  ".join(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}{unit}" for lo, hi in bands)
        print(f"{'model':22s} {hdr}")
        for name in order:
            p = preds[name]
            cells = []
            for lo, hi in bands:
                m = (vals >= lo) & (vals < hi)
                cells.append(f"{mae(p[m], ys[m]):5.2f}" if m.any() else "  -  ")
            print(f"{name:22s} " + "   ".join(cells))

    band_table("distance", tf_s["d"], DIST_BANDS, "m")
    band_table("speed", tf_s["v"], SPEED_BANDS, "m/s")
    band_table("angle", np.abs(theta_s), ANGLE_BANDS, "deg")

    # ---- reachable-area metric (Renkin) at Δt = 1 s -------------------------
    print(f"\nbuilding Δt={RA_HORIZON_S:.0f}s TEST trails for the reachable-area metric ...",
          flush=True)
    trails = load_split_trails(2, RA_HORIZON_S, max_matches=1).subsample(RA_TRAILS, seed=SEED + 1)
    caps = (trail_capacities(trails, 2, 1)
            if any(is_perplayer(nm) for nm in mm_names) else None)

    @torch.no_grad()
    def mlp_grid(tf):
        return mlp(torch.from_numpy(tf["X"]).to(device)).cpu().numpy()

    ra_fns = {
        "SUVAT": lambda tf: suvat_time(tf["v"], tf["cos_t"], tf["d"]),
        "MLP": mlp_grid,
    }
    if args.profile_path:
        # the profile MLP needs the per-player v_max as its 7th input, supplied
        # per row through the same capacity column the physics variants use
        caps_ra = caps if caps is not None else trail_capacities(trails, 2, 1)
        caps = caps_ra

        @torch.no_grad()
        def profile_grid(tf):
            Xg = np.concatenate(
                [tf["X"], (np.asarray(tf["vmax_pp"]) / V_NORM).astype(np.float32)[:, None]],
                axis=1)
            return profile(torch.from_numpy(Xg).to(device)).cpu().numpy()

        ra_fns["MLP+vmax"] = profile_grid
    for name, m in motion_models.items():
        ra_fns[name] = (lambda tf, mm=m: mm.predict(tf))

    # one grid pass per model serves both views: the fixed-Δt score (paper eq. 4)
    # and the matched-coverage area (each model tuned to a common coverage, so we
    # compare pure precision/area at equal correctness -- removes the operating-
    # point bias that makes raw fixed-Δt areas incomparable across models).
    MATCH_COV = (0.90, 0.95, 0.99)
    rows = {}
    for name in order:
        fn = ra_fns[name]
        cs = 8 if name.startswith("WuSwartz") else 64   # WuSwartz grids rows x n_dirs
        extra = caps if (is_perplayer(name) or name == "MLP+vmax") else None
        pred_tgt = tta_to_targets(fn, trails, extra=extra)
        grid, cell_area = grid_tta(fn, trails, res=RA_RES, chunk_states=cs, extra=extra)
        areas_fixed = (grid <= RA_HORIZON_S).sum(axis=1) * cell_area
        sc = score(pred_tgt, areas_fixed, RA_HORIZON_S, RA_HIT_RATIO_MIN)
        mc = {c: matched_coverage_area(pred_tgt, grid, cell_area, c)["mean_area"]
              for c in MATCH_COV}
        rows[name] = {"sc": sc, "mc": mc}

    print(f"\n(A) FIXED Δt={RA_HORIZON_S:.0f}s, {len(trails):,} test trails  "
          f"(paper eq. 4; hit_ratio_min={RA_HIT_RATIO_MIN}; paper's best 71.7 m²):")
    print(f"{'model':22s} {'coverage':>9s} {'mean_area':>10s} {'precision':>10s} {'gate':>5s}")
    print("-" * 62)
    for name in order:
        sc = rows[name]["sc"]
        gate = "ok" if sc["passes_gate"] else "LOW"
        a = sc["mean_area"]
        astr = f"{a:9.1f}m²" if np.isfinite(a) else f"{'nan':>10s}"
        print(f"{name:22s} {sc['hit_ratio']*100:>8.2f}% {astr} {sc['precision']:>10.5f} {gate:>5s}")

    print("\n(B) MATCHED-COVERAGE area (each model tuned to the SAME coverage; "
          "lower m² = more precise):")
    hdr = "  ".join(f"@{int(c*100)}%cov" for c in MATCH_COV)
    print(f"{'model':22s} {hdr}")
    print("-" * 48)
    for name in order:
        cells = [f"{rows[name]['mc'][c]:6.1f}m²" for c in MATCH_COV]
        print(f"{name:22s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
