"""Build the ball-pass dataset: FFPC tracking + events -> pass_samples.npz.

Ball-pipeline analog of scripts/build_dataset.py. For every PFF game it:
  1. reads the player Match, ball trajectory and successful pass events;
  2. cleans every player's track per half and assembles per-frame kinematics
     (so the passer's velocity at release is available);
  3. emits one sample per pass (ball release -> ball reception, label = flight
     time), applies the ball noise-floor / sanity filter, tags the match;
  4. holds out whole matches into train/val/test and writes a single .npz.

Run:  python -m scripts.build_pass_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    PASS_FPS,
    PASS_NPZ,
    PASS_PROCESSED_DIR,
    PASS_SPLIT_FRACS,
    SEED,
)
from src.ffpc_passes import iter_ffpc_pass_matches  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.passes import build_pass_features_and_filter, samples_from_passes  # noqa: E402


def _clean_match_kinematics(match):
    """Clean every player's track per segment -> global (T,P,2) pos/vel/acc + valid."""
    T, P = match.T, match.P
    posG = np.full((T, P, 2), np.nan, dtype=np.float32)
    velG = np.full((T, P, 2), np.nan, dtype=np.float32)
    accG = np.full((T, P, 2), np.nan, dtype=np.float32)
    validG = np.zeros((T, P), dtype=bool)
    for seg_id, lo, hi in match.segment_slices():
        for p in range(P):
            in_field = match.in_field[lo:hi, p]
            if not np.any(in_field == 1):
                continue
            tr = clean_track(match.positions[lo:hi, p, :], in_field, player=p, segment=seg_id)
            posG[lo:hi, p, :] = tr.pos
            velG[lo:hi, p, :] = tr.vel
            accG[lo:hi, p, :] = tr.acc
            validG[lo:hi, p] = tr.state_valid
    return posG, velG, accG, validG


def assign_match_splits(match_ids: np.ndarray) -> np.ndarray:
    """Hold out whole matches -> 0=train / 1=val / 2=test (fixed-seed shuffle)."""
    uniq = np.unique(match_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(PASS_SPLIT_FRACS[0] * n))
    n_va = int(round(PASS_SPLIT_FRACS[1] * n))
    split_of = {}
    for i, m in enumerate(uniq):
        split_of[int(m)] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return np.array([split_of[int(m)] for m in match_ids], dtype=np.int8)


def main() -> None:
    PASS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ball-pass dataset (FFPC, fps={PASS_FPS})")

    Xs, ys, metas = [], [], {}
    recording_ids: list[str] = []
    total_passes = total_raw = total_kept = 0

    for match_idx, pm in enumerate(iter_ffpc_pass_matches()):
        recording_ids.append(pm.recording_id)
        n_ball = int(np.isfinite(pm.ball[:, 0]).sum())
        print(f"[{match_idx}] {pm.recording_id}: T={pm.match.T} P={pm.match.P} "
              f"| {len(pm.passes)} successful passes | {n_ball} ball-visible frames")
        total_passes += len(pm.passes)

        posG, velG, accG, validG = _clean_match_kinematics(pm.match)
        raw = samples_from_passes(pm.passes, pm.ball, posG, velG, accG, validG)
        if raw["label"].size == 0:
            continue
        X, y, meta, (n_in, n_out) = build_pass_features_and_filter(raw)
        total_raw += n_in
        total_kept += n_out
        if y.size == 0:
            continue
        meta["match"] = np.full(y.size, match_idx, dtype=np.int32)
        Xs.append(X)
        ys.append(y)
        for k, v in meta.items():
            metas.setdefault(k, []).append(v)

    if not Xs:
        raise SystemExit("No pass samples produced -- check the FFPC raw files.")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = {k: np.concatenate(v) for k, v in metas.items()}
    split = assign_match_splits(meta["match"])

    print(f"\nMatches: {len(recording_ids)}  |  successful passes: {total_passes:,}")
    print(f"Pass samples with valid ball+passer: {total_raw:,}  |  "
          f"kept after filter: {total_kept:,} ({100*total_kept/max(total_raw,1):.1f}%)")
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        m = split == code
        print(f"  {name:5s}: {int(m.sum()):>8,}  | {len(np.unique(meta['match'][m]))} matches")

    np.savez_compressed(
        PASS_NPZ,
        X=X, y=y, split=split,
        recording_ids=np.array(recording_ids),
        **{f"meta_{k}": v for k, v in meta.items()},
    )
    print(f"\nSaved -> {PASS_NPZ}  (X={X.shape}, y={y.shape})")


if __name__ == "__main__":
    main()
