"""Loader for PFF FC tracking data -> the shared `Match` representation.

PFF ships one game per id across parallel folders under a competition dir:

    <competition>/Tracking Data/<id>.jsonl.bz2   one JSON object per frame
    <competition>/Metadata/<id>.json             [ { fps, teams, periods, ... } ]
    <competition>/Rosters/<id>.json              [ { player, shirtNumber, team } ]

We read the *raw* (unsmoothed) player coordinates and let the cleaning stage do
its own smoothing/derivatives, exactly as for the other providers. Coordinates
are already in metres with the origin at pitch centre (Tracking Spec v2.2 s7);
the feature transform is translation/rotation-invariant, so no re-framing is
needed -- only the frame rate differs, and that comes from config.FPS.

Two data-reality quirks handled here (Tracking Spec v2.2):
  * Frames are *duplicated* when a frame carries multiple possession events;
    we keep one row per (period, frameNum) so derivatives never see dt=0.
  * frameNum has gaps (non-main-camera segments, half-time). We index each
    period by frameNum relative to that period's first frame and leave NaN in
    the gaps, so velocities are never differentiated across a real-time jump.
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config
from .data_loader import Match

_VALID_PERIODS = (1, 2, 3, 4)


def _competition_for(game_id: str) -> Path:
    """Return the competition dir whose Tracking Data holds this game id."""
    for comp in config.ffpc_competition_dirs():
        if (comp / "Tracking Data" / f"{game_id}.jsonl.bz2").exists():
            return comp
    raise FileNotFoundError(f"no FFPC tracking file for game {game_id!r}")


def available_game_ids() -> list[str]:
    """All PFF game ids present on disk (sorted), or the FFPC_GAME_IDS subset."""
    if config.FFPC_GAME_IDS is not None:
        return [str(g) for g in config.FFPC_GAME_IDS]
    ids: set[str] = set()
    for comp in config.ffpc_competition_dirs():
        for f in (comp / "Tracking Data").glob("*.jsonl.bz2"):
            ids.add(f.name.split(".")[0])
    return sorted(ids, key=lambda s: (len(s), s))


def _build_player_axis(roster: list[dict], home_team_id: str):
    """Map (side, jerseyNum) -> player index; return parallel id/name/side arrays.

    side 0 == home, 1 == away. A shirt number is unique within a team for the
    whole match (substitutes wear different numbers), so (side, shirt) is a
    stable identity across the game.
    """
    key_to_idx: dict[tuple[int, str], int] = {}
    player_id, player_name, side_id = [], [], []
    for entry in roster:
        team_id = str(entry["team"]["id"])
        side = 0 if team_id == home_team_id else 1
        shirt = str(entry["shirtNumber"])
        key = (side, shirt)
        if key in key_to_idx:
            continue  # defensive: ignore duplicate roster rows
        key_to_idx[key] = len(player_id)
        player_id.append(int(entry["player"]["id"]))
        player_name.append(entry["player"].get("nickname") or str(entry["player"]["id"]))
        side_id.append(side)
    return (
        key_to_idx,
        np.asarray(player_id, dtype=np.int64),
        player_name,
        np.asarray(side_id, dtype=np.int8),
    )


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_ffpc_match(game_id: str) -> Match:
    """Parse one PFF game into a `Match` (raw positions, NaN where untracked)."""
    game_id = str(game_id)
    comp = _competition_for(game_id)
    meta = _read_json(comp / "Metadata" / f"{game_id}.json")[0]
    roster = _read_json(comp / "Rosters" / f"{game_id}.json")
    home_team_id = str(meta["homeTeam"]["id"])

    key_to_idx, player_id, player_name, side_id = _build_player_axis(roster, home_team_id)
    P = len(player_id)

    # ---- pass 1: stream frames, dedupe duplicates, collect per period --------
    # records[period] -> list of (frameNum, [(pidx, x, y), ...])
    records: dict[int, list[tuple[int, list[tuple[int, float, float]]]]] = {}
    seen_key: tuple[int, int] | None = None  # (period, frameNum) of last kept row
    n_unmatched = 0

    track_path = comp / "Tracking Data" / f"{game_id}.jsonl.bz2"
    with bz2.open(track_path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            fr = json.loads(line)
            period = fr.get("period")
            if period not in _VALID_PERIODS:
                continue
            frame_num = int(fr["frameNum"])
            if seen_key == (period, frame_num):
                continue  # duplicated frame (extra possession event)
            seen_key = (period, frame_num)

            pos: list[tuple[int, float, float]] = []
            for side, field in ((0, "homePlayers"), (1, "awayPlayers")):
                for pl in fr.get(field) or ():
                    x, y = pl.get("x"), pl.get("y")
                    if x is None or y is None:
                        continue
                    idx = key_to_idx.get((side, str(pl.get("jerseyNum"))))
                    if idx is None:
                        n_unmatched += 1
                        continue
                    pos.append((idx, float(x), float(y)))
            records.setdefault(period, []).append((frame_num, pos))

    if not records:
        raise ValueError(f"FFPC game {game_id}: no usable frames")

    # ---- pass 2: per period, index by frameNum-relative, concat in order -----
    pos_blocks, field_blocks, seg_blocks, time_blocks = [], [], [], []
    time_offset = 0.0
    for period in sorted(records):
        rows = records[period]
        fmin = min(r[0] for r in rows)
        fmax = max(r[0] for r in rows)
        Lp = fmax - fmin + 1

        positions = np.full((Lp, P, 2), np.nan, dtype=np.float32)
        in_field = np.zeros((Lp, P), dtype=np.int8)
        for frame_num, pos in rows:
            li = frame_num - fmin
            for idx, x, y in pos:
                positions[li, idx, 0] = x
                positions[li, idx, 1] = y
                in_field[li, idx] = 1

        pos_blocks.append(positions)
        field_blocks.append(in_field)
        seg_blocks.append(np.full(Lp, period, dtype=np.int64))
        time_blocks.append(time_offset + np.arange(Lp) * config.DT)
        time_offset += Lp * config.DT

    positions = np.concatenate(pos_blocks, axis=0)
    in_field = np.concatenate(field_blocks, axis=0)
    segment = np.concatenate(seg_blocks)
    time_in_match = np.concatenate(time_blocks)
    T = positions.shape[0]

    return Match(
        positions=positions,
        in_field=in_field,
        segment=segment,
        frame=np.arange(T, dtype=np.int64),
        time_in_match=time_in_match,
        player_id=player_id,
        player_name=player_name,
        team_id=side_id.copy(),   # home/away doubles as the two "teams"
        side_id=side_id,
        recording_id=game_id,
    )


def iter_ffpc_matches(game_ids: list[str] | None = None) -> Iterator[Match]:
    """Yield a `Match` per PFF game (defaults to all available / configured)."""
    for gid in (game_ids if game_ids is not None else available_game_ids()):
        yield load_ffpc_match(gid)
