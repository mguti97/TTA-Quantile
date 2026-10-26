"""Ball-pass reader for PFF FC: tracking + event data -> a `PassMatch`.

This is the ball-pipeline analog of `load_ffpc_match` (src/ffpc.py). In one
stream of a game's tracking file it produces:

  * a player `Match` (raw positions, NaN where untracked) -- identical contract
    to the player pipeline, so cleaning/derivatives are reused unchanged;
  * the smoothed BALL trajectory aligned to the very same global frame index
    (`ball`, shape (T, 2), NaN where the ball is not VISIBLE);
  * the list of successful PASS events, each resolved to *global* release and
    reception frame indices plus the passer's player index.

Frame model (verified against the data, Tracking Spec v2.2):
  * Each frame carries `frameNum` (period-local, with gaps), `period`, a
    `ballsSmoothed` dict {visibility,x,y,z}, and the embedded `game_event`
    (with `start_frame`/`end_frame`) it belongs to.
  * A pass's release frame == its game_event.end_frame (ball leaves the passer);
    its reception frame == the *next* game_event's start_frame (receiver gains
    possession). For a completed pass the next on-ball event is the receiver, so
    this brackets the ball flight. Both choices were checked to give physically
    sensible ball speeds (median ~12 m/s, p90 ~19 m/s).

Pass outcome, passer identity and the chronological event order come from the
Event Data JSON (`<id>.json`); the frame boundaries come from tracking. We join
the two on `gameEventId`.
"""
from __future__ import annotations

import bz2
import json
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import config
from .data_loader import Match
from .ffpc import (
    _VALID_PERIODS,
    _build_player_axis,
    _competition_for,
    _read_json,
    available_game_ids,
)


@dataclass
class PassEvent:
    release_g: int     # global frame index of release (ball leaves passer)
    reception_g: int   # global frame index of reception (ball reaches receiver)
    passer_idx: int    # player index in the Match (-1 if passer not on roster axis)
    period: int


@dataclass
class PassMatch:
    match: Match
    ball: np.ndarray          # (T, 2) smoothed ball xy, NaN where not VISIBLE
    passes: list[PassEvent]
    recording_id: str


def _ball_xy(fr: dict):
    """Smoothed ball (x, y) if VISIBLE, else None."""
    bs = fr.get("ballsSmoothed")
    if isinstance(bs, dict) and bs.get("visibility") == "VISIBLE":
        x, y = bs.get("x"), bs.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def load_ffpc_pass_match(game_id: str) -> PassMatch:
    """Parse one PFF game into a `PassMatch` (player Match + ball + pass events)."""
    game_id = str(game_id)
    comp = _competition_for(game_id)
    meta = _read_json(comp / "Metadata" / f"{game_id}.json")[0]
    roster = _read_json(comp / "Rosters" / f"{game_id}.json")
    home_team_id = str(meta["homeTeam"]["id"])

    key_to_idx, player_id, player_name, side_id = _build_player_axis(roster, home_team_id)
    P = len(player_id)
    id_to_idx = {int(pid): i for i, pid in enumerate(player_id)}

    # ---- pass 1: stream frames; dedupe; collect players + ball + game-event frames
    # records[period] -> list of (frameNum, [(pidx,x,y),...], ball_xy_or_None)
    records: dict[int, list] = {}
    # ge_frame[game_event_id] -> (period, start_frame, end_frame), first occurrence
    ge_frame: dict[int, tuple[int, int, int]] = {}
    seen_key = None

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

            pos = []
            for side, field in ((0, "homePlayers"), (1, "awayPlayers")):
                for pl in fr.get(field) or ():
                    x, y = pl.get("x"), pl.get("y")
                    if x is None or y is None:
                        continue
                    idx = key_to_idx.get((side, str(pl.get("jerseyNum"))))
                    if idx is None:
                        continue
                    pos.append((idx, float(x), float(y)))
            records.setdefault(period, []).append((frame_num, pos, _ball_xy(fr)))

            geid = fr.get("game_event_id")
            ge = fr.get("game_event") or {}
            if geid is not None and int(geid) not in ge_frame and ge.get("start_frame") is not None:
                ge_frame[int(geid)] = (period, int(ge["start_frame"]), int(ge["end_frame"]))

    if not records:
        raise ValueError(f"FFPC game {game_id}: no usable frames")

    # ---- pass 2: per period, index frames relative to fmin; concat in order ------
    pos_blocks, field_blocks, ball_blocks, seg_blocks, time_blocks = [], [], [], [], []
    fmin_of: dict[int, int] = {}
    goffset_of: dict[int, int] = {}
    offset = 0
    time_offset = 0.0
    for period in sorted(records):
        rows = records[period]
        fmin = min(r[0] for r in rows)
        fmax = max(r[0] for r in rows)
        Lp = fmax - fmin + 1
        fmin_of[period] = fmin
        goffset_of[period] = offset

        positions = np.full((Lp, P, 2), np.nan, dtype=np.float32)
        in_field = np.zeros((Lp, P), dtype=np.int8)
        ball = np.full((Lp, 2), np.nan, dtype=np.float32)
        for frame_num, pos, bxy in rows:
            li = frame_num - fmin
            for idx, x, y in pos:
                positions[li, idx, 0] = x
                positions[li, idx, 1] = y
                in_field[li, idx] = 1
            if bxy is not None:
                ball[li] = bxy

        pos_blocks.append(positions)
        field_blocks.append(in_field)
        ball_blocks.append(ball)
        seg_blocks.append(np.full(Lp, period, dtype=np.int64))
        time_blocks.append(time_offset + np.arange(Lp) * config.PASS_DT)
        offset += Lp
        time_offset += Lp * config.PASS_DT

    positions = np.concatenate(pos_blocks, axis=0)
    in_field = np.concatenate(field_blocks, axis=0)
    ball = np.concatenate(ball_blocks, axis=0)
    segment = np.concatenate(seg_blocks)
    time_in_match = np.concatenate(time_blocks)
    T = positions.shape[0]

    def to_global(period: int, frame_num: int) -> int:
        """Map a period-local frameNum to the concatenated global index, or -1."""
        if period not in goffset_of:
            return -1
        li = frame_num - fmin_of[period]
        g = goffset_of[period] + li
        return g if 0 <= g < T else -1

    match = Match(
        positions=positions,
        in_field=in_field,
        segment=segment,
        frame=np.arange(T, dtype=np.int64),
        time_in_match=time_in_match,
        player_id=player_id,
        player_name=player_name,
        team_id=side_id.copy(),
        side_id=side_id,
        recording_id=game_id,
    )

    # ---- join the Event JSON: chronological game events + successful passes ------
    events = _read_json(comp / "Event Data" / f"{game_id}.json")

    # ordered, deduped game events that have known frames -> reception lookup
    ordered_ge: list[int] = []
    ge_pos: dict[int, int] = {}
    for e in events:
        geid = e.get("gameEventId")
        if geid is None:
            continue
        geid = int(geid)
        if geid in ge_pos or geid not in ge_frame:
            continue
        ge_pos[geid] = len(ordered_ge)
        ordered_ge.append(geid)

    passes: list[PassEvent] = []
    for e in events:
        pe = e.get("possessionEvents") or {}
        if pe.get("possessionEventType") != "PA":
            continue
        if pe.get("passOutcomeType") not in config.PASS_SUCCESS_OUTCOMES:
            continue
        geid = e.get("gameEventId")
        if geid is None or int(geid) not in ge_pos:
            continue
        geid = int(geid)
        k = ge_pos[geid]
        if k + 1 >= len(ordered_ge):
            continue
        period, _sf, ef = ge_frame[geid]
        nperiod, nsf, _nef = ge_frame[ordered_ge[k + 1]]
        if nperiod != period:
            continue  # reception in a different half -> not a real flight
        release_g = to_global(period, ef)
        reception_g = to_global(nperiod, nsf)
        if release_g < 0 or reception_g < 0:
            continue
        passer = pe.get("passerPlayerId")
        passer_idx = id_to_idx.get(int(passer), -1) if passer is not None else -1
        passes.append(PassEvent(release_g, reception_g, passer_idx, period))

    return PassMatch(match=match, ball=ball, passes=passes, recording_id=game_id)


def iter_ffpc_pass_matches(game_ids: list[str] | None = None) -> Iterator[PassMatch]:
    """Yield a `PassMatch` per PFF game (defaults to all available / configured)."""
    for gid in (game_ids if game_ids is not None else available_game_ids()):
        yield load_ffpc_pass_match(gid)
