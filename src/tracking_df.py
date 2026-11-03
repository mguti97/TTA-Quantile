"""Tracking DataFrame (forced to the FFPC format) -> `Match` -> TTA surfaces.

Lets you compute time-to-arrival surfaces from your own tracking data without
building a dataset. The input is a *long* pandas DataFrame, one row per
(frame, player), that must follow the PFF FC (``ffpc_match``) conventions the
shipped checkpoints were trained on:

    column      type        meaning
    ---------   ---------   ----------------------------------------------------
    period      int         1, 2 (3, 4 = extra time)
    frameNum    int         frame counter at 29.97 fps (gaps allowed -> NaN)
    team        str / int   "home" / "away" (or 0 / 1)
    jerseyNum   int / str   shirt number, unique within a team for the match
    x, y        float       metres, origin at the pitch centre, x along the
                            105 m length, y along the 68 m width
    player_id   (optional)  provider player id, carried into the output
    player_name (optional)  display name, carried into the output

`coerce_ffpc()` forces a DataFrame into that layout (column renames, team
labels, unit scale, corner->centre origin) and validates it; anything it cannot
fix raises a ValueError explaining what is wrong. `ffpc_jsonl_to_dataframe()`
flattens a raw PFF ``Tracking Data/<id>.jsonl.bz2`` file into the same layout.

The model only sees per-player kinematic state, so no ball, event or roster
data is needed. The frame rate is NOT resampled: frameNum must already be at
29.97 fps, because cleaning (Savitzky-Golay window, finite-difference DT) is
defined in FFPC frames.
"""
from __future__ import annotations

import bz2
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .data_loader import Match
from .inference.heatmap import heatmaps_for_frame, pitch_grid
from .pipeline.cleaning import clean_track

FFPC_COLUMNS = ("period", "frameNum", "team", "jerseyNum", "x", "y")
OPTIONAL_COLUMNS = ("player_id", "player_name")
FFPC_PITCH = (105.0, 68.0)
_VALID_PERIODS = (1, 2, 3, 4)
_TEAM_ALIASES = {"home": 0, "h": 0, "0": 0, "away": 1, "a": 1, "1": 1}
# Frames of context kept on each side of the requested frame. Cleaning is local
# (9-frame Savitzky-Golay + two np.gradient passes), so 2 s of context gives the
# same state as cleaning the whole period, at a fraction of the cost.
CONTEXT_S = 2.0


def _require_ffpc_config():
    if config.DATASET != "ffpc_match":
        raise RuntimeError(
            f"active dataset is {config.DATASET!r}; FFPC-format surfaces need the "
            "ffpc_match frame rate and pitch. Set KOGNIA_DATASET=ffpc_match before "
            "importing src (scripts/compute_surface.py does this for you)."
        )


# ---------------------------------------------------------------- coercion
def coerce_ffpc(
    df: pd.DataFrame,
    rename: dict[str, str] | None = None,
    scale: float = 1.0,
    origin: str = "center",
) -> pd.DataFrame:
    """Return a copy of `df` forced into the FFPC long format (see module doc).

    rename  -- {your_column: ffpc_column}, e.g. {"frame": "frameNum", "side": "team"}
    scale   -- multiply x, y by this to get metres (0.9144 for yards, 0.01 for cm)
    origin  -- "center" (FFPC) or "corner": coordinates run 0..105 / 0..68 from a
               pitch corner and are shifted to the centre.

    Drops rows with missing x/y and duplicated (period, frameNum, team, jerseyNum)
    rows (PFF repeats a frame when it carries several possession events).
    """
    out = df.rename(columns=rename or {}).copy()
    missing = [c for c in FFPC_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"tracking DataFrame is missing FFPC columns {missing}; have "
            f"{list(out.columns)}. Map yours with rename={{'yours': 'ffpc'}} "
            f"(CLI: --rename yours=ffpc)."
        )
    out = out[[c for c in (*FFPC_COLUMNS, *OPTIONAL_COLUMNS) if c in out.columns]]

    # team -> 0 (home) / 1 (away)
    team = out["team"].astype(str).str.strip().str.lower().map(_TEAM_ALIASES)
    if team.isna().any():
        bad = sorted(out.loc[team.isna(), "team"].astype(str).unique())[:5]
        raise ValueError(f"team must be home/away (or 0/1); got {bad}")
    out["team"] = team.astype(np.int8)

    out["period"] = pd.to_numeric(out["period"], errors="raise").astype(np.int64)
    bad_p = sorted(set(out["period"].unique()) - set(_VALID_PERIODS))
    if bad_p:
        raise ValueError(f"period must be in {_VALID_PERIODS}; got {bad_p}")
    out["frameNum"] = pd.to_numeric(out["frameNum"], errors="raise").astype(np.int64)
    # shirt numbers: compare as strings so 7, "7" and 7.0 are one player
    out["jerseyNum"] = out["jerseyNum"].map(
        lambda j: str(int(j)) if isinstance(j, (int, float, np.integer, np.floating))
        and float(j).is_integer() else str(j).strip()
    )

    for c in ("x", "y"):
        out[c] = pd.to_numeric(out[c], errors="coerce") * scale
    out = out.dropna(subset=["x", "y"])
    if origin == "corner":
        out["x"] -= FFPC_PITCH[0] / 2
        out["y"] -= FFPC_PITCH[1] / 2
    elif origin != "center":
        raise ValueError(f"origin must be 'center' or 'corner', got {origin!r}")

    _check_pitch_frame(out)
    out = out.drop_duplicates(["period", "frameNum", "team", "jerseyNum"], keep="first")
    return out.sort_values(["period", "frameNum", "team", "jerseyNum"]).reset_index(drop=True)


def _check_pitch_frame(df: pd.DataFrame):
    """Catch the usual wrong-format mistakes (corner origin, yards/cm, swapped axes)."""
    if df.empty:
        raise ValueError("no rows with finite x/y")
    L, W = FFPC_PITCH
    x_lo, x_hi = df["x"].quantile([0.001, 0.999])
    y_lo, y_hi = df["y"].quantile([0.001, 0.999])
    slack = 10.0  # players do step off the pitch
    hint = []
    if x_lo > -slack and x_hi > L / 2 + slack:
        hint.append("x looks corner-based (0..105): pass origin='corner' (--origin corner)")
    if max(abs(x_lo), abs(x_hi), abs(y_lo), abs(y_hi)) > 3 * L:
        hint.append("coordinates look like cm/mm or pixels: set scale (--scale) to get metres")
    if (y_hi - y_lo) > (x_hi - x_lo) + slack:
        hint.append("y spans more than x: FFPC has x along the pitch length; swap them")
    if hint or x_lo < -L / 2 - slack or x_hi > L / 2 + slack \
            or y_lo < -W / 2 - slack or y_hi > W / 2 + slack:
        raise ValueError(
            f"x in [{x_lo:.1f}, {x_hi:.1f}], y in [{y_lo:.1f}, {y_hi:.1f}] is not an "
            f"FFPC pitch frame (metres, centre origin, x in ±{L / 2:g}, y in ±{W / 2:g})."
            + ("".join(f"\n  - {h}" for h in hint))
        )


def ffpc_jsonl_to_dataframe(path: str | Path) -> pd.DataFrame:
    """Flatten a raw PFF ``Tracking Data/<id>.jsonl.bz2`` (or plain .jsonl) file."""
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            fr = json.loads(line)
            period = fr.get("period")
            if period not in _VALID_PERIODS:
                continue
            for team, field in (("home", "homePlayers"), ("away", "awayPlayers")):
                for pl in fr.get(field) or ():
                    if pl.get("x") is None or pl.get("y") is None:
                        continue
                    rows.append((period, int(fr["frameNum"]), team,
                                 pl.get("jerseyNum"), pl["x"], pl["y"]))
    return pd.DataFrame(rows, columns=list(FFPC_COLUMNS))


def read_tracking(path: str | Path) -> pd.DataFrame:
    """Read .csv / .parquet / .pkl / PFF .jsonl(.bz2) into a raw DataFrame."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.bz2")):
        return ffpc_jsonl_to_dataframe(path)
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    raise ValueError(f"unsupported tracking file {path} (csv, parquet, pkl, jsonl[.bz2])")


# ---------------------------------------------------------------- Match
def player_keys(df: pd.DataFrame) -> list[tuple[int, str]]:
    """(team, jerseyNum) per player, in `Match` player-axis order."""
    keys = {(int(t), str(j)) for t, j in df[["team", "jerseyNum"]].drop_duplicates().to_numpy()}
    # natural order: home 1, 2, ..., 10, then away (non-numeric shirts last)
    return sorted(keys, key=lambda k: (k[0], not k[1].isdigit(), int(k[1]) if k[1].isdigit() else 0, k[1]))


def match_from_dataframe(df: pd.DataFrame, recording_id: str = "dataframe") -> Match:
    """Build a `Match` from an already-coerced FFPC DataFrame (see `coerce_ffpc`).

    Mirrors `src.ffpc.load_ffpc_match`: one player axis keyed by (team, jersey),
    each period indexed by frameNum relative to its first frame, NaN in gaps.
    """
    key_to_idx = {k: i for i, k in enumerate(player_keys(df))}
    P = len(key_to_idx)
    first = df.drop_duplicates(["team", "jerseyNum"]).set_index(["team", "jerseyNum"])

    def _meta(col, default):
        if col not in first.columns:
            return [default(t, j) for (t, j) in key_to_idx]
        return [first.loc[(t, j), col] for (t, j) in key_to_idx]

    player_id = np.asarray(_meta("player_id", lambda t, j: -1), dtype=np.int64)
    player_name = [str(n) for n in _meta(
        "player_name", lambda t, j: f"{'home' if t == 0 else 'away'} #{j}")]
    side_id = np.asarray([t for (t, _) in key_to_idx], dtype=np.int8)
    pidx = np.fromiter((key_to_idx[k] for k in zip(df["team"], df["jerseyNum"])),
                       dtype=np.int64, count=len(df))

    pos_blocks, field_blocks, seg_blocks, frame_blocks, time_blocks = [], [], [], [], []
    time_offset = 0.0
    for period, g in df.assign(_p=pidx).groupby("period", sort=True):
        fmin, fmax = int(g["frameNum"].min()), int(g["frameNum"].max())
        Lp = fmax - fmin + 1
        li = g["frameNum"].to_numpy() - fmin
        pi = g["_p"].to_numpy()
        positions = np.full((Lp, P, 2), np.nan, dtype=np.float32)
        positions[li, pi, 0] = g["x"].to_numpy()
        positions[li, pi, 1] = g["y"].to_numpy()
        in_field = np.zeros((Lp, P), dtype=np.int8)
        in_field[li, pi] = 1
        pos_blocks.append(positions)
        field_blocks.append(in_field)
        seg_blocks.append(np.full(Lp, period, dtype=np.int64))
        frame_blocks.append(np.arange(fmin, fmax + 1, dtype=np.int64))  # = frameNum
        time_blocks.append(time_offset + np.arange(Lp) * config.DT)
        time_offset += Lp * config.DT

    return Match(
        positions=np.concatenate(pos_blocks),
        in_field=np.concatenate(field_blocks),
        segment=np.concatenate(seg_blocks),
        frame=np.concatenate(frame_blocks),   # provider frameNum, not 0..T-1
        time_in_match=np.concatenate(time_blocks),
        player_id=player_id,
        player_name=player_name,
        team_id=side_id.copy(),
        side_id=side_id,
        recording_id=recording_id,
    )


# ---------------------------------------------------------------- surfaces
@dataclass
class FrameSurfaces:
    period: int
    frame_num: int
    tta: np.ndarray          # (n, nx, ny) predicted minimum time to arrive (s)
    team_min: np.ndarray     # (2, nx, ny) min over each team's players (home, away)
    suvat: np.ndarray | None  # (n, nx, ny) SUVAT physics baseline, if requested
    team: np.ndarray         # (n,) 0 home / 1 away
    jersey: list[str]
    player_id: np.ndarray
    player_name: list[str]
    pos: np.ndarray          # (n, 2) cleaned position (m)
    vel: np.ndarray          # (n, 2) m/s
    acc: np.ndarray          # (n, 2) m/s^2
    xs: np.ndarray           # (nx,) cell-centre x (m)
    ys: np.ndarray           # (ny,) cell-centre y (m)
    skipped: list[str]       # players tracked at the frame but without a valid state


def select_frame(df: pd.DataFrame, frame_num: int, period: int | None = None) -> int:
    """Resolve the period for `frame_num` (frameNum can repeat across periods)."""
    periods = sorted(df.loc[df["frameNum"] == frame_num, "period"].unique())
    if period is not None:
        if period not in periods:
            raise ValueError(f"frameNum {frame_num} not tracked in period {period} "
                             f"(found in periods {periods or 'none'})")
        return int(period)
    if not periods:
        raise ValueError(f"frameNum {frame_num} not in the tracking data")
    if len(periods) > 1:
        raise ValueError(f"frameNum {frame_num} occurs in periods {periods}; pass period")
    return int(periods[0])


def load_model(path: str | Path, device):
    from .model.mlp import TTAModel

    sd = torch.load(path, map_location=device)
    in_dim = sd["net.0.weight"].shape[1]
    if in_dim != config.N_FEATURES:
        raise ValueError(
            f"{path} expects {in_dim} features; only the 6-feature base checkpoints "
            "(mlp.pt, mlp_arealoss_*.pt, mlp_seed*.pt, ...) work from tracking alone"
        )
    hidden = [sd[k].shape[0] for k in sorted(sd) if k.endswith(".weight")][:-1]
    model = TTAModel(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(sd)
    return model.eval()


def compute_surfaces(
    df: pd.DataFrame,
    frame_num: int,
    model,
    device="cpu",
    period: int | None = None,
    res: float = config.GRID_RES,
    with_suvat: bool = False,
) -> FrameSurfaces:
    """TTA surface of every player with a valid kinematic state at one frame.

    `df` must already be coerced (`coerce_ffpc`). The state (smoothed position,
    velocity, acceleration) is estimated exactly as at training time; players in
    a run too short to smooth, or whose state fails the physics clipping, are
    listed in `skipped` instead of getting a surface.
    """
    _require_ffpc_config()
    period = select_frame(df, frame_num, period)
    ctx = int(round(CONTEXT_S * config.FPS))
    win = df[(df["period"] == period) & df["frameNum"].between(frame_num - ctx, frame_num + ctx)]
    match = match_from_dataframe(win)
    t = int(np.searchsorted(match.frame, frame_num))

    keep, states, skipped = [], [], []
    for p in range(match.P):
        if match.in_field[t, p] != 1:
            continue
        tr = clean_track(match.positions[:, p], match.in_field[:, p], p, period)
        if tr.state_valid[t]:
            keep.append(p)
            states.append((tr.pos[t], tr.vel[t], tr.acc[t]))
        else:
            skipped.append(match.player_name[p])
    if not states:
        raise ValueError(f"no player has a valid state at period {period} frame {frame_num}")

    out = heatmaps_for_frame(model, states, device, res=res, return_suvat=with_suvat)
    heat, suv = (out[0], out[1]) if with_suvat else (out[0], None)
    cells, nx, ny = pitch_grid(res)
    xs, ys = cells[::ny, 0], cells[:ny, 1]

    team = match.side_id[keep]
    team_min = np.full((2, nx, ny), np.nan, dtype=np.float32)
    for s in (0, 1):
        if (team == s).any():
            team_min[s] = heat[team == s].min(axis=0)

    names = match.player_name
    keys = player_keys(win)
    return FrameSurfaces(
        period=period, frame_num=int(frame_num), tta=heat.astype(np.float32),
        team_min=team_min, suvat=None if suv is None else suv.astype(np.float32),
        team=team, jersey=[keys[p][1] for p in keep],
        player_id=match.player_id[keep], player_name=[names[p] for p in keep],
        pos=np.stack([s[0] for s in states]), vel=np.stack([s[1] for s in states]),
        acc=np.stack([s[2] for s in states]), xs=xs, ys=ys, skipped=skipped,
    )
