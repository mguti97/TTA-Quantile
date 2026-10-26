"""End-to-end preprocessing: raw tracking -> cleaned -> sampled -> transformed -> filtered.

Works for any dataset in the config registry: it iterates every Match the active
dataset yields (one per game), tags each sample with a
match id, and writes a single .npz holding X, y, per-sample meta, and a
train/val/test split (by match / player / time-block; see SPLIT_MODE).
Run:  python -m scripts.build_dataset   (set the dataset via config.DATASET)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATASET,
    DATASET_NPZ,
    DATASETS,
    FPS,
    PLAYER_AMAX_PCT,
    PLAYER_VMAX_PCT,
    PROCESSED_DIR,
    SEED,
    SPLIT_BLOCK_S,
    SPLIT_FRACS,
    SPLIT_MODE,
    SUVAT_FILTER,
    VMAX_SCOPE,
    VMAX_TABLE_NPZ,
)
from src.data_loader import iter_matches  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.sampling import build_features_and_filter, samples_from_track  # noqa: E402


def _assign_keys(key: np.ndarray) -> np.ndarray:
    """Map a per-sample group key -> 0=train/1=val/2=test by whole-group shuffle."""
    uniq = np.unique(key)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(SPLIT_FRACS[0] * n))
    n_va = int(round(SPLIT_FRACS[1] * n))
    split_of = {}
    for i, b in enumerate(uniq):
        split_of[int(b)] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return np.array([split_of[int(k)] for k in key], dtype=np.int8)


def assign_splits(meta: dict[str, np.ndarray]) -> np.ndarray:
    """Assign each sample to train/val/test per SPLIT_MODE.

    "match":  whole matches are held out -> the canonical split (Section 6.5);
              only meaningful for multi-match datasets (e.g. ffpc_match).
    "player": whole players are held out -> zero trajectory leakage, tests
              generalization to unseen individuals. Keyed per (match, player)
              so identities never collide across matches.
    "block":  contiguous SPLIT_BLOCK_S-second windows (per match & segment) are
              held out. Whole groups go to a single split (fixed-seed shuffle).
    """
    match = meta["match"].astype(np.int64)
    if SPLIT_MODE == "match":
        return _assign_keys(match)
    if SPLIT_MODE == "player":
        return _assign_keys(match * 100_000 + meta["player"].astype(np.int64))
    if SPLIT_MODE == "block":
        block = meta["frame"].astype(np.int64) // int(SPLIT_BLOCK_S * FPS)
        key = match * 1_000_000_000 + meta["segment"].astype(np.int64) * 10_000_000 + block
        return _assign_keys(key)
    raise ValueError(f"unknown SPLIT_MODE: {SPLIT_MODE!r}")


def n_matches() -> int:
    """How many matches iter_matches() will yield -- without loading any.

    The match-level split is a deterministic shuffle of the match INDICES
    (_assign_keys with the fixed SEED), so the count alone decides which matches
    are TRAIN, before a single sample exists. That is what lets the train_split
    v_max pass run ahead of the build loop.
    """
    loader = DATASETS[DATASET]["loader"]
    if loader == "pff":
        from src.ffpc import available_game_ids
    elif loader == "bdb":
        from src.bdb import available_game_ids
    elif loader == "nba":
        from src.nba import available_game_ids
    else:
        return 1                      # single-match dataset
    return len(available_game_ids())


def train_match_indices() -> np.ndarray:
    """Match indices landing in TRAIN, via the same _assign_keys as assign_splits."""
    return np.where(_assign_keys(np.arange(n_matches())) == 0)[0]


def build_vmax_table() -> dict[str, dict[int, float] | float]:
    """Global per-player v_max/a_max pooled over the TRAIN-split matches only.

    Keyed by the provider's GLOBAL player id (`Match.player_id`, persistent
    across matches) rather than the match-local index, so one table serves
    train/val/test alike: a test match's profile no longer peeks at that same
    match's future frames (the look-ahead leakage of VMAX_SCOPE == "match").

    Tracks are re-cleaned here (cleaning is cheap) and only the state-valid
    speed/accel magnitudes are kept, so at most one match is ever in memory.
    """
    assert SPLIT_MODE == "match", (
        "VMAX_SCOPE='train_split' pools over whole train matches, which only "
        f"matches the sample-level split when SPLIT_MODE == 'match' (got {SPLIT_MODE!r})"
    )
    if VMAX_TABLE_NPZ.exists():
        print(f"Reusing saved v_max table {VMAX_TABLE_NPZ} (delete it to rebuild)")
        return load_vmax_table()
    want = {int(i) for i in train_match_indices()}
    last = max(want)
    speeds: dict[int, list] = {}   # global player id -> [speed arrays]
    accels: dict[int, list] = {}   # global player id -> [accel-magnitude arrays]
    for match_idx, match in enumerate(iter_matches()):
        if match_idx in want:
            print(f"[vmax A/{len(want)}] match[{match_idx}] {match.recording_id}", flush=True)
            for seg_id, lo, hi in match.segment_slices():
                for p in range(match.P):
                    in_field = match.in_field[lo:hi, p]
                    if not np.any(in_field == 1):
                        continue
                    track = clean_track(match.positions[lo:hi, p, :], in_field,
                                        player=p, segment=seg_id)
                    if not track.state_valid.any():
                        continue
                    sv = track.state_valid
                    gid = int(match.player_id[p])
                    speeds.setdefault(gid, []).append(
                        np.hypot(track.vel[sv, 0], track.vel[sv, 1]).astype(np.float32))
                    accels.setdefault(gid, []).append(
                        np.hypot(track.acc[sv, 0], track.acc[sv, 1]).astype(np.float32))
        if match_idx >= last:
            break

    vmax = {g: float(np.percentile(np.concatenate(s), PLAYER_VMAX_PCT))
            for g, s in speeds.items()}
    amax = {g: float(np.percentile(np.concatenate(a), PLAYER_AMAX_PCT))
            for g, a in accels.items()}
    # Fallback for players who never appear in a TRAIN match (they exist only in
    # the val/test matches): the MEDIAN of the per-player table -- a population
    # prior that stays inside the observed range, unlike a mean, which the few
    # cameo players whose 99th percentile rests on a handful of frames would drag.
    fallbacks = (float(np.median(list(vmax.values()))), float(np.median(list(amax.values()))))
    print(f"v_max table: {len(vmax)} players from {len(want)} train matches "
          f"| fallback v_max={fallbacks[0]:.2f} m/s a_max={fallbacks[1]:.2f} m/s²")
    table = {"vmax": vmax, "amax": amax,
             "vmax_fallback": fallbacks[0], "amax_fallback": fallbacks[1]}
    _save_vmax_table(table)   # persist now: pass B is long and interruptible
    return table


def _save_vmax_table(table) -> None:
    ids = np.array(sorted(table["vmax"]), dtype=np.int64)
    np.savez_compressed(
        VMAX_TABLE_NPZ,
        player_id=ids,
        vmax=np.array([table["vmax"][int(g)] for g in ids], dtype=np.float32),
        amax=np.array([table["amax"][int(g)] for g in ids], dtype=np.float32),
        vmax_fallback=np.float32(table["vmax_fallback"]),
        amax_fallback=np.float32(table["amax_fallback"]),
    )
    print(f"Saved -> {VMAX_TABLE_NPZ}  ({len(ids)} players + median fallback)")


def load_vmax_table() -> dict[str, dict[int, float] | float]:
    """Read back the train_split v_max/a_max table (for the eval scripts)."""
    if not VMAX_TABLE_NPZ.exists():
        raise SystemExit(
            f"{VMAX_TABLE_NPZ} missing -- build it with "
            "`KOGNIA_VMAX_SCOPE=train_split python -m scripts.build_dataset`")
    z = np.load(VMAX_TABLE_NPZ)
    ids = [int(g) for g in z["player_id"]]
    return {"vmax": dict(zip(ids, (float(v) for v in z["vmax"]))),
            "amax": dict(zip(ids, (float(a) for a in z["amax"]))),
            "vmax_fallback": float(z["vmax_fallback"]),
            "amax_fallback": float(z["amax_fallback"])}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATASET}  (fps={FPS})  v_max scope: {VMAX_SCOPE}")

    # Pass A (train_split only): the leakage-free global v_max/a_max table.
    table = build_vmax_table() if VMAX_SCOPE == "train_split" else None
    n_fallback = 0  # (match, player) slots with no train-split observation

    Xs, ys, metas = [], [], {}
    recording_ids: list[str] = []
    total_raw = total_kept = total_below = 0
    n_tracks = 0

    for match_idx, match in enumerate(iter_matches()):
        recording_ids.append(match.recording_id)
        print(f"[{match_idx}] {match.recording_id}: T={match.T} frames, P={match.P} players")

        # Pass 1: clean every (segment, player) track once, group by player, and
        # gather each player's state-valid speeds & accel magnitudes to estimate
        # their max speed / max acceleration.
        tracks_by_player: dict[int, list] = {}      # p -> [(track, lo), ...]
        speeds_by_player: dict[int, list] = {}      # p -> [speed arrays]
        accels_by_player: dict[int, list] = {}      # p -> [accel-magnitude arrays]
        infield_by_player: dict[int, list] = {}     # p -> [global on-pitch row idx]
        for seg_id, lo, hi in match.segment_slices():
            for p in range(match.P):
                pos_raw = match.positions[lo:hi, p, :]
                in_field = match.in_field[lo:hi, p]
                if not np.any(in_field == 1):
                    continue
                # global row indices where this player is on the pitch -- used to
                # compute cumulative minutes-played per sample (fatigue feature).
                infield_by_player.setdefault(p, []).append(lo + np.where(in_field == 1)[0])
                track = clean_track(pos_raw, in_field, player=p, segment=seg_id)
                if not track.state_valid.any():
                    continue
                tracks_by_player.setdefault(p, []).append((track, lo))
                sv = track.state_valid
                speeds_by_player.setdefault(p, []).append(
                    np.hypot(track.vel[sv, 0], track.vel[sv, 1])
                )
                accels_by_player.setdefault(p, []).append(
                    np.hypot(track.acc[sv, 0], track.acc[sv, 1])
                )

        # Player-specific max speed / max accel = 99th pct of observed speeds /
        # accel magnitudes, pooled across both halves (personalised locomotion
        # capacity, Section 12) -- or, under VMAX_SCOPE == "train_split", read
        # off the global train-only table by the player's persistent provider id.
        if table is None:
            vmax_by_player = {
                p: float(np.percentile(np.concatenate(s), PLAYER_VMAX_PCT))
                for p, s in speeds_by_player.items()
            }
            amax_by_player = {
                p: float(np.percentile(np.concatenate(a), PLAYER_AMAX_PCT))
                for p, a in accels_by_player.items()
            }
        else:
            vmax_by_player, amax_by_player = {}, {}
            for p in speeds_by_player:
                gid = int(match.player_id[p])
                if gid not in table["vmax"]:
                    n_fallback += 1
                vmax_by_player[p] = table["vmax"].get(gid, table["vmax_fallback"])
                amax_by_player[p] = table["amax"].get(gid, table["amax_fallback"])

        # Sorted global on-pitch row indices per player -> minutes played at any
        # frame f is searchsorted(rows, f, "right") / FPS / 60.
        infield_sorted = {p: np.sort(np.concatenate(v)) for p, v in infield_by_player.items()}

        # Pass 2: sample each track, tagging samples with the player's max speed/accel.
        for p, tracklist in tracks_by_player.items():
            vmax = vmax_by_player[p]
            amax = amax_by_player[p]
            for track, lo in tracklist:
                n_tracks += 1
                raw = samples_from_track(track, global_frame0=lo)
                if raw["label"].size == 0:
                    continue
                X, y, meta, (n_in, n_out, n_below) = build_features_and_filter(
                    raw, player_vmax=vmax, player_amax=amax)
                total_raw += n_in
                total_kept += n_out
                total_below += n_below
                if y.size == 0:
                    continue
                # minutes played so far = cumulative on-pitch time at each sample's frame.
                meta["minutes"] = (
                    np.searchsorted(infield_sorted[p], meta["frame"], side="right")
                    / FPS / 60.0
                ).astype(np.float32)
                meta["match"] = np.full(y.size, match_idx, dtype=np.int32)
                Xs.append(X)
                ys.append(y)
                for k, v in meta.items():
                    metas.setdefault(k, []).append(v)

    if not Xs:
        raise SystemExit("No samples produced -- check the dataset/raw files.")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = {k: np.concatenate(v) for k, v in metas.items()}
    split = assign_splits(meta)

    n_matches = len(recording_ids)
    print(f"\nMatches: {n_matches}  |  tracks with usable states: {n_tracks}")
    print(f"Samples extracted: {total_raw:,}  |  kept after SUVAT filter: {total_kept:,} "
          f"({100*total_kept/max(total_raw,1):.1f}%)")
    # Always report what the SUVAT physics model flags as impossible, whether or
    # not it was actually applied -- in "off" mode this line is the sensitivity
    # experiment's evidence for how much data the filter touches.
    print(f"SUVAT filter: {SUVAT_FILTER}  |  samples below the SUVAT time: "
          f"{total_below:,} ({100*total_below/max(total_raw,1):.4f}% of extracted"
          + ("; REMOVED)" if SUVAT_FILTER == "on" else "; KEPT -- would have been removed)"))
    print(f"Split mode: {SPLIT_MODE}")
    if table is not None:
        # The pass-A train set was derived from the match COUNT; the sample-level
        # split from the match ids that actually produced samples. They must be
        # the same set, or pass A pooled the wrong matches.
        got = set(np.unique(meta["match"][split == 0]).astype(int).tolist())
        assert got == {int(i) for i in train_match_indices()}, (
            "train-split matches disagree between the upfront v_max pass and the "
            f"sample split ({sorted(got)} vs {train_match_indices().tolist()})")
        print(f"v_max scope: train_split  |  {n_fallback} (match,player) slots used "
              f"the population-median fallback")
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        m = split == code
        n_m = len(np.unique(meta["match"][m]))
        n_grp = len({(int(a), int(b)) for a, b in zip(meta["match"][m], meta["player"][m])})
        print(f"  {name:5s}: {int(m.sum()):>10,}  | {n_m} matches, {n_grp} (match,player) groups")

    np.savez_compressed(
        DATASET_NPZ,
        X=X, y=y, split=split,
        recording_ids=np.array(recording_ids),
        **{f"meta_{k}": v for k, v in meta.items()},
    )
    print(f"\nSaved -> {DATASET_NPZ}  (X={X.shape}, y={y.shape})")

    # (train_split mode: the v_max table was already saved right after pass A,
    # so it survives an interruption during the long pass B / save above.)


if __name__ == "__main__":
    main()
