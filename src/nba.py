"""Loader for NBA SportVU player tracking (2015-16 season) -> shared `Match`.

Ships as one 7z archive per game under
``data/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/``, each holding
a single big JSON. Inside, moments are grouped into "events" tied to play-by-play
triggers, and consecutive events heavily overlap (each replays several seconds of
context around its trigger) -- flattening naively would duplicate most frames.
We instead flatten every event's moments, dedupe by wall-clock timestamp (ms,
the one truly unique per-frame key) and sort ascending, which reconstructs one
continuous ~25 fps track per game:

    NBA game     -> one Match       (mirrors "one PFF game / one NFL game" == Match)
    NBA quarter  -> one segment     (mirrors NFL's per-play block / football's per-half)

Each moment is ``[quarter, wallclock_ms, game_clock_s, shot_clock_s, ?, rows]``
where rows is the ball (team_id -1, z = height in the air) followed by up to 10
players (team_id, player_id, x, y, z=0.0 always). The player axis is the union of
player ids appearing anywhere in the game (both rosters), since starters/bench
rotate through substitutions; a player is `in_field` only for the frames they're
actually one of the ~10 tracked bodies (exactly the occlusion pattern the
cleaning stage already handles).

Coordinates are NBA court feet; we convert to metres (`config.NBA_FOOT_M`) so the
shared physics bounds, normalisation, and SUVAT baseline apply unchanged.

Extraction is cached: the first read of a game unpacks its 7z into
``data/nba_match/raw/<gameid>.json`` (~100 MB each; ~66 GB for all 636 games) so
repeated builds don't re-decompress.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import py7zr

from . import config
from .data_loader import Match

_ARCHIVE_DIR = (
    config.DATA_DIR / "NBA-Player-Movements" / "data" / "2016.NBA.Raw.SportVU.Game.Logs"
)
_CACHE_DIR = config.dataset_dir("nba_match") / "raw"


def _archive_files() -> list[Path]:
    files = sorted(_ARCHIVE_DIR.glob("*.7z"))
    if not files:
        raise FileNotFoundError(f"no NBA .7z archives under {_ARCHIVE_DIR}")
    return files


def _game_id_of(path: Path) -> str | None:
    """The single JSON entry's stem, or None if the archive is empty/corrupt.

    A handful of archives in the vendored repo (linouk23/NBA-Player-Movements)
    are empty/truncated stubs (a few tens of bytes) rather than real 7z data --
    skip those rather than failing the whole build.
    """
    try:
        with py7zr.SevenZipFile(path, "r") as z:
            names = z.getnames()
    except py7zr.exceptions.Bad7zFile:
        names = []
    if len(names) != 1:
        print(f"  [nba] skipping corrupt/empty archive: {path.name}")
        return None
    return Path(names[0]).stem


def available_game_ids() -> list[str]:
    """All NBA game ids (or the config.NBA_GAME_IDS subset), cheap archive metadata only."""
    if config.NBA_GAME_IDS is not None:
        return [str(g) for g in config.NBA_GAME_IDS]
    return [gid for p in _archive_files() if (gid := _game_id_of(p)) is not None]


def _extract(path: Path, game_id: str) -> Path:
    out = _CACHE_DIR / f"{game_id}.json"
    if not out.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(path, "r") as z:
            z.extractall(path=_CACHE_DIR)
    return out


def _match_from_game(d: dict, game_id: str) -> Match:
    # home/visitor + rosters are per-event, not top-level, but fixed for the game.
    home, visitor = d["events"][0]["home"], d["events"][0]["visitor"]
    home_ids = {p["playerid"] for p in home["players"]}
    roster_name = {
        p["playerid"]: f"{p['firstname']} {p['lastname']}"
        for side in (home, visitor) for p in side["players"]
    }

    # ---- flatten every event's moments, dedupe by wall-clock ms, sort ----------
    by_ms: dict[int, tuple[int, list]] = {}
    for ev in d["events"]:
        for mo in ev["moments"]:
            quarter, ms, _game_clock, _shot_clock, _, rows = mo
            if ms not in by_ms:
                by_ms[ms] = (quarter, rows)
    ms_sorted = sorted(by_ms)
    T = len(ms_sorted)
    if T == 0:
        raise ValueError(f"no moments in game {game_id}")
    frame_q = np.array([by_ms[ms][0] for ms in ms_sorted], dtype=np.int64)

    # ---- flatten (frame, team, player, x, y) across all non-ball rows ---------
    flat_frame, flat_team, flat_pid, flat_x, flat_y = [], [], [], [], []
    for t, ms in enumerate(ms_sorted):
        for row in by_ms[ms][1]:
            if len(row) != 5:
                continue
            team_id_row, player_id_row, x, y, _z = row
            if team_id_row == -1:
                continue  # ball; not part of the player pipeline
            flat_frame.append(t)
            flat_team.append(team_id_row)
            flat_pid.append(player_id_row)
            flat_x.append(x)
            flat_y.append(y)

    flat_frame = np.asarray(flat_frame, dtype=np.int64)
    flat_pid = np.asarray(flat_pid, dtype=np.int64)
    flat_x = np.asarray(flat_x, dtype=np.float32)
    flat_y = np.asarray(flat_y, dtype=np.float32)

    player_id = np.unique(flat_pid)
    P = player_id.size
    pidx = np.searchsorted(player_id, flat_pid)

    player_name = [roster_name.get(int(pid), str(pid)) for pid in player_id]
    side_id = np.array([0 if int(pid) in home_ids else 1 for pid in player_id], dtype=np.int8)
    team_id = np.where(side_id == 0, home["teamid"], visitor["teamid"]).astype(np.int64)

    positions = np.full((T, P, 2), np.nan, dtype=np.float32)
    in_field = np.zeros((T, P), dtype=np.int8)
    positions[flat_frame, pidx, 0] = flat_x
    positions[flat_frame, pidx, 1] = flat_y
    in_field[flat_frame, pidx] = 1
    positions *= config.NBA_FOOT_M  # feet -> metres

    # per-segment (quarter) frame index * DT, mirroring bdb.py's per-block time
    time_in_match = np.empty(T, dtype=np.float64)
    change = np.where(np.diff(frame_q) != 0)[0] + 1
    bounds = [0, *change.tolist(), T]
    for a, b in zip(bounds[:-1], bounds[1:]):
        time_in_match[a:b] = np.arange(b - a) * config.DT

    return Match(
        positions=positions,
        in_field=in_field,
        segment=frame_q,
        frame=np.arange(T, dtype=np.int64),
        time_in_match=time_in_match,
        player_id=player_id.astype(np.int64),
        player_name=player_name,
        team_id=team_id,
        side_id=side_id,
        recording_id=str(game_id),
    )


def iter_nba_matches(game_ids: list[str] | None = None) -> Iterator[Match]:
    """Yield a `Match` per NBA game, extracting (and caching) each 7z on first use."""
    wanted = None
    if game_ids is not None:
        wanted = {str(g) for g in game_ids}
    elif config.NBA_GAME_IDS is not None:
        wanted = {str(g) for g in config.NBA_GAME_IDS}

    for path in _archive_files():
        gid = _game_id_of(path)
        if gid is None:
            continue
        if wanted is not None and gid not in wanted:
            continue
        json_path = _extract(path, gid)
        d = json.loads(json_path.read_text())
        try:
            yield _match_from_game(d, gid)
        except (ValueError, IndexError, KeyError) as e:
            # a handful of games ship with empty/malformed event data (incomplete
            # tracking capture in the source, not a parsing bug) -- skip them.
            print(f"  [nba] skipping game {gid}: {type(e).__name__}: {e}")


def load_nba_match(game_id: str | None = None) -> Match:
    """Return one `Match` (the first game, or a specific `game_id`)."""
    return next(iter_nba_matches(None if game_id is None else [game_id]))
