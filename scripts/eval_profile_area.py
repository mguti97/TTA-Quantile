"""Matched-coverage reachable-area for the player-profile MLP (base + v_max).

Table 3 (tab:area) reports SUVAT / MLP / six motion models but never the
profile MLP (MLP + v_max), because compare_motion_models.py hard-codes the base
6-feature model. This script adds that missing row, on the *same* cached
Δt=1s test trails (single match, 500-trail subsample, seed+1) so it's directly
comparable to the existing table.

    python -m scripts.eval_profile_area

Requires outputs/<dataset>/mlp_profile_vmax.pt (scripts.train_profile vmax) and
the cached trails at data/<dataset>/processed/trails_s2_h1_m1.npz (built by
scripts.compare_motion_models / scripts.fit_motion_models).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import _assign_keys, load_vmax_table  # noqa: E402
from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import (  # noqa: E402
    DATASET_NPZ,
    OUTPUTS_DIR,
    PLAYER_VMAX_PCT,
    SEED,
    VMAX_SCOPE,
    V_NORM,
)
from src.data_loader import iter_matches  # noqa: E402
from src.evaluation.reachable_area import matched_coverage_area  # noqa: E402
from src.inference.heatmap import pitch_grid  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.features import transform  # noqa: E402

RA_HORIZON_S = 1.0
RA_TRAILS = 500
RA_RES = 1.0
MATCH_COV = (0.90, 0.95)


def _test_match_index() -> int:
    n_matches = len(np.load(DATASET_NPZ)["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))
    want = [i for i in range(n_matches) if per_match[i] == 2][:1]
    if not want:
        raise SystemExit("no match in TEST split")
    return want[0]


def _vmax_by_player(match) -> dict[int, float]:
    """Per-player v_max keyed by match-local index, as the built dataset has it.

    "match" scope: reproduce build_dataset's per-player 99th-pct speed exactly.
    "train_split" scope: look the saved train-only table up by the player's
    global provider id (population-median fallback) -- no re-cleaning needed,
    and crucially no peeking at this TEST match's own frames.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-path", type=str, default=None,
                    help="explicit profile-MLP checkpoint "
                         "(default: outputs/<dataset>/mlp_profile_vmax.pt)")
    args = ap.parse_args()

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
    print(f"TEST match[{match_idx}] {match.recording_id}")

    vmax_by_player = _vmax_by_player(match)

    trails = load_split_trails(2, RA_HORIZON_S, max_matches=1).subsample(RA_TRAILS, seed=SEED + 1)
    print(f"{len(trails):,} trails (matches Table 3's cached SUVAT/MLP/physics rows)")

    vmax_arr = np.array([vmax_by_player[int(p)] for p in trails.player], dtype=np.float64)
    vmax_norm = (vmax_arr / V_NORM).astype(np.float32)

    model_path = Path(args.profile_path) if args.profile_path else OUTPUTS_DIR / "mlp_profile_vmax.pt"
    model = TTAModel(in_dim=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def predict(Xa: np.ndarray) -> np.ndarray:
        out = np.empty(len(Xa), dtype=np.float32)
        for s in range(0, len(Xa), 1_000_000):
            e = min(s + 1_000_000, len(Xa))
            out[s:e] = model(torch.from_numpy(Xa[s:e]).to(device)).cpu().numpy()
        return out

    # ---- pred_to_target: one row per trail --------------------------------
    tf = transform(trails.state_pos, trails.vel, trails.acc, trails.target_pos)
    Xp = np.concatenate([tf["X"], vmax_norm[:, None]], axis=1)
    pred_to_target = predict(Xp)

    # ---- grid_tta: (N, G) arrival time to every grid cell ------------------
    cells, _, _ = pitch_grid(RA_RES)
    G = cells.shape[0]
    N = len(trails)
    cell_area = RA_RES * RA_RES
    grid = np.empty((N, G), dtype=np.float32)
    chunk_states = 64
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
        grid[s:e] = predict(Xg).reshape(cs, G)

    mc = {c: matched_coverage_area(pred_to_target, grid, cell_area, c)["mean_area"]
          for c in MATCH_COV}

    print("\nMATCHED-COVERAGE area, MLP + v_max (profile), same trails as Table 3:")
    hdr = "  ".join(f"@{int(c*100)}%cov" for c in MATCH_COV)
    print(f"{'model':22s} {hdr}")
    cells_str = [f"{mc[c]:6.1f}m²" for c in MATCH_COV]
    print(f"{'MLP + v_max (profile)':22s} " + "   ".join(cells_str))


if __name__ == "__main__":
    main()
