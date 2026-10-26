"""Loader for the NFL Big Data Bowl 2019 (2017 season) tracking -> shared `Match`.

Ships as one tracking CSV per game under data/Big-Data-Bowl/Data/:

    tracking_gameId_<gameId>.csv   one row per (player, frame): time, x, y, s,
                                   dis, dir, event, nflId, displayName,
                                   jerseyNumber, team, frame.id, gameId, playId

plus game/play/player metadata CSVs we don't need for reachability. We map this
onto the pipeline's one contract, the `Match`:

    BDB game  ->  one Match       (one tracking CSV == one game == one Match)
    BDB play  ->  one segment     (a short, self-contained tracking run)

The player axis is the union of nfl_ids that appear in the game; the football
rows (team == "ball", nflId NA) are dropped -- the player pipeline models people.
A player is `in_field` only for the frames of the plays they take part in, NaN
elsewhere -- exactly the occlusion pattern the cleaning stage already handles, so
nothing downstream changes.

`frame.id` is 1-based per play at 10 fps; plays are laid out back-to-back so each
play is one contiguous segment (per-block time = frame index * DT). Coordinates
are yards; we convert to metres on read (`config.BDB_YARD_M`) so the physics
bounds, normalisation and SUVAT baseline apply unchanged. We read only raw x/y
and let the cleaning stage recompute its own smoothed kinematics, ignoring the
provided s/dis/dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config
from .data_loader import Match

# Only the columns the pipeline needs -- keep the CSV read lean (these files are big).
_TRACK_COLS = ["x", "y", "nflId", "displayName", "team", "frame.id", "playId"]


def _game_files() -> list[Path]:
    """Return the per-game tracking CSVs (optionally restricted to game ids), sorted."""
    wanted = None
    if config.BDB_GAME_IDS is not None:
        wanted = {str(g) for g in config.BDB_GAME_IDS}
    files = []
    for p in sorted(config.BDB_TRACKING_DIR.glob("tracking_gameId_*.csv")):
        gid = p.stem.split("_")[-1]                   # "tracking_gameId_2017090700" -> "2017090700"
        if wanted is not None and gid not in wanted:
            continue
        files.append(p)
    if not files:
        raise FileNotFoundError(f"no tracking_gameId_*.csv under {config.BDB_TRACKING_DIR}")
    return files


def available_game_ids() -> list[str]:
    """All BDB game ids available (or the BDB_GAME_IDS subset)."""
    return [p.stem.split("_")[-1] for p in _game_files()]


def _match_from_game(df: pd.DataFrame, game_id: str) -> Match:
    """Build one `Match` from a single game's tracking rows (coords in yards)."""
    # ---- drop the football, keep tracked people ----
    df = df[df["team"] != "ball"].copy()
    df = df[df["nflId"].notna()]
    df["nflId"] = df["nflId"].astype(np.int64)
    df["playId"] = df["playId"].astype(np.int64)
    df["frame.id"] = df["frame.id"].astype(np.int64)

    # ---- player axis: union of nfl_ids in the game ----
    ids = df["nflId"].to_numpy()
    player_id, inv = np.unique(ids, return_inverse=True)   # sorted ids, per-row index
    P = player_id.size

    # first-seen name / team per player (stable within a game). team is
    # home/away -> the two "sides"; the reachability model is side-agnostic, this
    # is carried only for provenance / eval colouring.
    first = df.drop_duplicates("nflId").set_index("nflId")
    player_name = [str(first.at[int(pid), "displayName"]) for pid in player_id]
    side_id = np.array(
        [0 if first.at[int(pid), "team"] == "home" else 1 for pid in player_id],
        dtype=np.int8,
    )

    # ---- per-play block geometry: plays laid out back-to-back ----
    # play -> number of frames = max 1-based frame.id observed in that play.
    n_frames = df.groupby("playId")["frame.id"].max()
    plays = sorted(n_frames.index.tolist())               # deterministic play order
    block_len = {pl: int(n_frames[pl]) for pl in plays}
    offset, seg_of_play = {}, {}
    cursor = 0
    for s, pl in enumerate(plays):
        offset[pl] = cursor
        seg_of_play[pl] = s
        cursor += block_len[pl]
    T = cursor

    positions = np.full((T, P, 2), np.nan, dtype=np.float32)
    in_field = np.zeros((T, P), dtype=np.int8)
    segment = np.empty(T, dtype=np.int64)
    time_in_match = np.empty(T, dtype=np.float64)
    for pl in plays:
        lo, hi = offset[pl], offset[pl] + block_len[pl]
        segment[lo:hi] = seg_of_play[pl]
        time_in_match[lo:hi] = np.arange(block_len[pl]) * config.DT

    # ---- scatter tracking rows into (frame, player) ----
    off_arr = df["playId"].map(offset).to_numpy()
    rows = off_arr + df["frame.id"].to_numpy() - 1        # frame.id is 1-based
    positions[rows, inv, 0] = df["x"].to_numpy(np.float32)
    positions[rows, inv, 1] = df["y"].to_numpy(np.float32)
    in_field[rows, inv] = 1

    positions *= config.BDB_YARD_M                        # yards -> metres

    return Match(
        positions=positions,
        in_field=in_field,
        segment=segment,
        frame=np.arange(T, dtype=np.int64),
        time_in_match=time_in_match,
        player_id=player_id.astype(np.int64),
        player_name=player_name,
        team_id=side_id.copy(),   # home/away doubles as the two "teams"
        side_id=side_id,
        recording_id=str(game_id),
    )


def iter_bdb_matches(game_ids: list[str] | None = None) -> Iterator[Match]:
    """Yield a `Match` per BDB game (one tracking CSV each)."""
    wanted = {str(g) for g in game_ids} if game_ids is not None else None
    for path in _game_files():
        gid = path.stem.split("_")[-1]
        if wanted is not None and gid not in wanted:
            continue
        df = pd.read_csv(path, usecols=_TRACK_COLS)
        yield _match_from_game(df, gid)


def load_bdb_match(game_id: str | None = None) -> Match:
    """Return one `Match` (the first game, or a specific `game_id`)."""
    return next(iter_bdb_matches(None if game_id is None else [game_id]))
