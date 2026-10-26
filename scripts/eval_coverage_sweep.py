"""Matched-coverage reachable area for every model, swept over coverage and horizon.

Extends the Table 3 (tab:area) computation -- previously Δt=1s at 90%/95% matched
coverage only -- to the full grid {80,90,95,99}% coverage x {1,3,5}s horizon, for
all eight models (six fitted physics baselines, MLP, MLP+v_max profile).

    python -m scripts.eval_coverage_sweep                # tau = config.PINBALL_TAU
    python -m scripts.eval_coverage_sweep --tau 0.01     # loads *_tau0.01.pt variants

Requires motion_models.json (scripts.fit_motion_models), mlp.pt and
mlp_profile_vmax.pt already trained (scripts.run_h010.sh). Builds/reuses cached
trails at data/<dataset>/processed/trails_s2_h{1,3,5}_m1.npz.
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

from scripts.build_dataset import _assign_keys, load_vmax_table  # noqa: E402
from scripts.fit_motion_models import (  # noqa: E402
    load_split_trails,
    trail_capacities,
)
from src.config import (  # noqa: E402
    DATASET_NPZ,
    MODEL_PT,
    OUTPUTS_DIR,
    PINBALL_TAU,
    PLAYER_VMAX_PCT,
    SEED,
    VMAX_SCOPE,
    V_NORM,
)
from src.data_loader import iter_matches  # noqa: E402
from src.evaluation.reachable_area import matched_coverage_area, tta_to_targets  # noqa: E402
from src.inference.heatmap import pitch_grid  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.motion_models import (  # noqa: E402
    ALL_SPECS,
    from_params,
    is_perplayer,
)
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.features import transform  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402

HORIZONS = (1.0, 3.0, 5.0)
COVERAGES = (0.80, 0.90, 0.95, 0.99)
RA_TRAILS = 500
RA_RES = 1.0
WS_DIRS = 180


def _test_match_index() -> int:
    n_matches = len(np.load(DATASET_NPZ)["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))
    want = [i for i in range(n_matches) if per_match[i] == 2][:1]
    if not want:
        raise SystemExit("no match in TEST split")
    return want[0]


def _vmax_by_player(match) -> dict[int, float]:
    """Per-player v_max keyed by match-local index, as the built dataset has it.

    Under VMAX_SCOPE == "train_split" the profile comes from the saved global
    table (keyed by provider player id, population-median fallback) instead of
    this TEST match's own speeds -- see scripts.build_dataset.
    """
    if VMAX_SCOPE == "train_split":
        t = load_vmax_table()
        return {p: t["vmax"].get(int(g), t["vmax_fallback"])
                for p, g in enumerate(match.player_id)}
    speeds_by_player: dict[int, list[np.ndarray]] = {}
    for seg_id, lo, hi in match.segment_slices():
        for p in range(match.P):
            in_field = match.in_field[lo:hi, p]
            if not np.any(in_field == 1):
                continue
            track = clean_track(match.positions[lo:hi, p, :], in_field, player=p, segment=seg_id)
            if not track.state_valid.any():
                continue
            sv = track.state_valid
            speeds_by_player.setdefault(p, []).append(np.hypot(track.vel[sv, 0], track.vel[sv, 1]))
    return {p: float(np.percentile(np.concatenate(s), PLAYER_VMAX_PCT))
            for p, s in speeds_by_player.items()}


def model_path(tau: float) -> Path:
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def profile_path(tau: float) -> Path:
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    return OUTPUTS_DIR / f"mlp_profile_vmax{suffix}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trails", type=int, default=RA_TRAILS,
                    help=f"trails per horizon (default {RA_TRAILS}; the paper protocol)")
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    ap.add_argument("--mlp-path", type=str, default=None,
                    help="explicit base-MLP checkpoint (overrides the tau-derived path)")
    ap.add_argument("--profile-path", type=str, default=None,
                    help="explicit profile-MLP checkpoint (overrides the tau-derived path)")
    ap.add_argument("--models-json", nargs="*", default=[],
                    help="extra fitted-motion-model json files to include alongside "
                         "outputs/<dataset>/motion_models.json (e.g. the per-player "
                         "P1/P2 fits). Per-player models get the same leakage-free "
                         "capacity columns the profile MLP's v_max feature uses.")
    args = ap.parse_args()
    tau = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match_idx = _test_match_index()
    match = None
    for idx, m in enumerate(iter_matches()):
        if idx == match_idx:
            match = m
            break
        if idx > match_idx:
            break
    if match is None:
        raise SystemExit(f"could not load match index {match_idx}")
    print(f"TEST match[{match_idx}] {match.recording_id}", flush=True)
    vmax_by_player = _vmax_by_player(match)

    mm_paths = [OUTPUTS_DIR / "motion_models.json"] + [Path(p) for p in args.models_json]
    mm_names, motion_models = [], {}
    for mm_path in mm_paths:
        fitted = json.loads(mm_path.read_text())
        names = [k for k in fitted if k in ALL_SPECS]
        print(f"models from {mm_path.name}: {', '.join(names)}", flush=True)
        for name in names:
            m = from_params(name, fitted[name]["params"])
            if name.startswith("WuSwartz"):
                m.n_dirs = WS_DIRS
            mm_names.append(name)
            motion_models[name] = m

    mlp_pt = Path(args.mlp_path) if args.mlp_path else model_path(tau)
    profile_pt = Path(args.profile_path) if args.profile_path else profile_path(tau)
    print(f"MLP: {mlp_pt.name}   profile: {profile_pt.name}", flush=True)
    mlp = TTAModel().to(device)
    mlp.load_state_dict(torch.load(mlp_pt, map_location=device))
    mlp.eval()
    profile = TTAModel(in_dim=7).to(device)
    profile.load_state_dict(torch.load(profile_pt, map_location=device))
    profile.eval()

    @torch.no_grad()
    def mlp_predict(Xa):
        out = np.empty(len(Xa), dtype=np.float32)
        for s in range(0, len(Xa), 1_000_000):
            e = min(s + 1_000_000, len(Xa))
            out[s:e] = mlp(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
        return out

    @torch.no_grad()
    def profile_predict(Xa):
        out = np.empty(len(Xa), dtype=np.float32)
        for s in range(0, len(Xa), 1_000_000):
            e = min(s + 1_000_000, len(Xa))
            out[s:e] = profile(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
        return out

    order = ["MLP", "MLP+vmax"] + mm_names
    cells, _, _ = pitch_grid(RA_RES)
    G = cells.shape[0]
    cell_area = RA_RES * RA_RES

    results = {}  # results[horizon][model][coverage] = area

    for horizon in HORIZONS:
        t0 = time.perf_counter()
        trails = load_split_trails(2, horizon, max_matches=1).subsample(args.trails, seed=SEED + 1)
        N = len(trails)
        vmax_arr = np.array([vmax_by_player[int(p)] for p in trails.player], dtype=np.float64)
        vmax_norm = (vmax_arr / V_NORM).astype(np.float32)
        # per-player capacity columns for the per-player physics models (same
        # train-split-only table the profile MLP's v_max feature comes from)
        caps = (trail_capacities(trails, 2, 1)
                if any(is_perplayer(n) for n in mm_names) else None)
        print(f"\n=== horizon {horizon:.0f}s: {N} trails "
              f"({time.perf_counter()-t0:.1f}s to load) ===", flush=True)

        ra_fns = {
            "MLP": lambda tf: mlp_predict(tf["X"]),
            "MLP+vmax": None,  # handled specially below (needs per-row vmax)
        }
        for name, m in motion_models.items():
            ra_fns[name] = (lambda tf, mm=m: mm.predict(tf))

        results[horizon] = {}
        for name in order:
            t1 = time.perf_counter()
            chunk_states = 8 if name.startswith("WuSwartz") else 64

            if name == "MLP+vmax":
                tf = transform(trails.state_pos, trails.vel, trails.acc, trails.target_pos)
                Xp = np.concatenate([tf["X"], vmax_norm[:, None]], axis=1)
                pred_to_target = profile_predict(Xp)
                grid = np.empty((N, G), dtype=np.float32)
                for s in range(0, N, chunk_states):
                    e = min(s + chunk_states, N)
                    cs = e - s
                    sp = np.repeat(trails.state_pos[s:e], G, axis=0)
                    vv = np.repeat(trails.vel[s:e], G, axis=0)
                    aa = np.repeat(trails.acc[s:e], G, axis=0)
                    cc = np.tile(cells, (cs, 1))
                    vmn = np.repeat(vmax_norm[s:e], G)
                    tf_g = transform(sp, vv, aa, cc)
                    Xg = np.concatenate([tf_g["X"], vmn[:, None]], axis=1)
                    grid[s:e] = profile_predict(Xg).reshape(cs, G)
            else:
                fn = ra_fns[name]
                extra = caps if is_perplayer(name) else None
                pred_to_target = tta_to_targets(fn, trails, extra=extra)
                grid = np.empty((N, G), dtype=np.float32)
                for s in range(0, N, chunk_states):
                    e = min(s + chunk_states, N)
                    cs = e - s
                    sp = np.repeat(trails.state_pos[s:e], G, axis=0)
                    vv = np.repeat(trails.vel[s:e], G, axis=0)
                    aa = np.repeat(trails.acc[s:e], G, axis=0)
                    cc = np.tile(cells, (cs, 1))
                    tf_g = transform(sp, vv, aa, cc)
                    if extra is not None:
                        tf_g.update({k: np.repeat(v[s:e], G) for k, v in extra.items()})
                    grid[s:e] = np.asarray(fn(tf_g)).reshape(cs, G)

            results[horizon][name] = {
                c: matched_coverage_area(pred_to_target, grid, cell_area, c)["mean_area"]
                for c in COVERAGES
            }
            print(f"  {name:22s} {time.perf_counter()-t1:6.1f}s   "
                  + "  ".join(f"@{int(c*100)}%={results[horizon][name][c]:7.1f}m2"
                              for c in COVERAGES), flush=True)

    print(f"\n{'='*100}\nFULL SWEEP: matched-coverage area (m^2), model x horizon x coverage")
    hdr = "  ".join(f"{h:.0f}s@{int(c*100)}%" for h in HORIZONS for c in COVERAGES)
    print(f"{'model':22s} {hdr}")
    print("-" * 100)
    for name in order:
        cells_str = "  ".join(f"{results[h][name][c]:8.1f}" for h in HORIZONS for c in COVERAGES)
        print(f"{name:22s} {cells_str}")


if __name__ == "__main__":
    main()
