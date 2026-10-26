"""Fit the free parameters of the theoretical motion models.

Follows the optimisation routine of Renkin et al. (ICCS 2022, paper 381): each
model defines a reachable area R(Δt); we score a parameterisation by its
*validation score* on a sample of exact-Δt trails --

    score = precision = n_correct / Σ area_correct       (paper eq. 3)
    gated to 0 unless  hit_ratio >= hit_ratio_min          (paper eq. 4)

and we search the parameter space to maximise it (paper eq. 12). The paper used
Bayesian optimisation; we use SciPy's differential evolution (a global,
derivative-free optimiser that also handles the non-smooth gate) and loop over
the one discrete parameter (``keep_initial`` of model (d)) exactly as the paper
does. Fitting is done at Δt = 1 s on the TRAIN split only, so the held-out test
comparison stays leak-free. ``hit_ratio_min`` is set to 0.95 to match the
1 - τ = 0.95 coverage target of our quantile models (the paper used 0.99975 for
its raw, unfiltered data); override with --hit-ratio-min.

    python -m scripts.fit_motion_models
    python -m scripts.fit_motion_models --hit-ratio-min 0.99

Writes outputs/<dataset>/motion_models.json (params + achieved fit score).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import (  # noqa: E402
    _assign_keys,
    assign_splits,
    load_vmax_table,
)
from src.config import (  # noqa: E402
    DATASET,
    DATASET_NPZ,
    DATASETS,
    FPS,
    GRID_RES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    SEED,
    VMAX_SCOPE,
)
from src.data_loader import iter_matches, load_match  # noqa: E402
from src.evaluation.reachable_area import (  # noqa: E402
    Trails,
    build_trails,
    reachable_areas,
    score,
    tta_to_targets,
)
from src.model.motion_models import (  # noqa: E402
    ALL_SPECS,
    HULL_SPECS,
    MODEL_SPECS,
    PERPLAYER_SPECS,
    AnzerBauer,
    AnzerBauerHull,
    is_perplayer,
)
from src.pipeline.cleaning import clean_track  # noqa: E402
from src.pipeline.suvat import suvat_time  # noqa: E402

FIT_HORIZON_S = 1.0     # paper fits at Δt = 1 s
FIT_TRAILS = 600        # trails used for the area-based objective (grid pass each)
FIT_TRAILS_WS = 220     # Wu & Swartz grids over directions -> use fewer trails
FIT_RES = 2.0           # coarse grid (m) during the search; fine grid for report
WS_FIT_DIRS = 96        # acceleration directions while fitting Wu & Swartz
DE_MAXITER = 18
DE_POPSIZE = 10
INFEASIBLE = 1.0e4      # objective floor for parameterisations below the gate

# Anzer & Bauer empirical tables: their 0.5 km/h speed bins; time grid spans the
# 0-8 s evaluation window. Sparse bins (< ANZER_MIN_BIN trails) are filled by
# linear interpolation across neighbouring speed bins.
ANZER_BIN_MS = 0.5 / 3.6
ANZER_V_MAX = 11.0
ANZER_T_STEP = 0.2
ANZER_T_MAX = 8.0
ANZER_MIN_BIN = 300
ANZER_HULL_K = 64       # angles per hull in the radial (star-shaped) encoding


def _concat_trails(parts: list[Trails], horizon: float) -> Trails:
    parts = [p for p in parts if len(p)]
    if not parts:
        e2, e1 = np.empty((0, 2)), np.empty(0, dtype=int)
        return Trails(horizon, e2, e2, e2, e2, e1, e1, e1)
    return Trails(
        horizon,
        np.concatenate([p.state_pos for p in parts]),
        np.concatenate([p.vel for p in parts]),
        np.concatenate([p.acc for p in parts]),
        np.concatenate([p.target_pos for p in parts]),
        np.concatenate([p.frame for p in parts]),
        np.concatenate([p.player for p in parts]),
        np.concatenate([p.segment for p in parts]),
    )


def _trail_cache_path(split_code, horizon, max_matches):
    return PROCESSED_DIR / f"trails_s{split_code}_h{horizon:g}_m{max_matches}.npz"


def load_split_trails(split_code: int, horizon: float, max_matches: int = 2) -> Trails:
    """Δt-horizon trails restricted to a train/val/test split, leak-free.

    Single-match datasets: build from the one match and apply the
    player/block split. Multi-match datasets (ffpc): the per-match split is
    deterministic (``_assign_keys`` over the match indices, same seed as the
    dataset build), so we pick matches of the wanted split and lazily load them
    from ``iter_matches()`` (the k-th yielded match is match index k).

    Result is cached to disk (building ffpc trails re-loads several raw matches).
    """
    cache = _trail_cache_path(split_code, horizon, max_matches)
    if cache.exists():
        z = np.load(cache)
        return Trails(horizon, z["state_pos"], z["vel"], z["acc"],
                      z["target_pos"], z["frame"], z["player"], z["segment"])

    tr = _build_split_trails(split_code, horizon, max_matches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, state_pos=tr.state_pos, vel=tr.vel, acc=tr.acc,
                        target_pos=tr.target_pos, frame=tr.frame,
                        player=tr.player, segment=tr.segment)
    return tr


def _build_split_trails(split_code: int, horizon: float, max_matches: int) -> Trails:
    if DATASETS[DATASET]["loader"] not in ("pff", "nba"):  # single-match
        trails = build_trails(load_match(), horizon)
        split = assign_splits({
            "frame": trails.frame, "segment": trails.segment,
            "player": trails.player, "match": np.zeros_like(trails.frame),
        })
        return trails.select(split == split_code)

    n_matches = len(np.load(DATASET_NPZ)["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))     # 0/1/2 per match index
    want = [i for i in range(n_matches) if per_match[i] == split_code][:max_matches]
    if not want:
        raise SystemExit(f"no matches in split {split_code}")
    last = max(want)
    parts = []
    for idx, match in enumerate(iter_matches()):
        if idx in want:
            print(f"    + match[{idx}] {match.recording_id}", flush=True)
            parts.append(build_trails(match, horizon))
        if idx >= last:
            break
    return _concat_trails(parts, horizon)


# --------------------------------------------------------------------------- #
# per-player capacity columns for the trails (ACCV rebuttal)
#
# ``Trails.player`` is the MATCH-LOCAL player index (``range(match.P)`` inside
# ``build_trails``), not the provider's global player id -- and a multi-match
# split concatenates several matches, so the same local index means different
# people in different blocks. The capacity table is keyed by the GLOBAL PFF id,
# so we (1) recover which matches went into a split's trails (the same
# deterministic ``_assign_keys`` selection the builder used), (2) cut the
# concatenated trails back into per-match blocks, and (3) map local -> global
# through ``Match.player_id``.
#
# Block boundaries: ``build_trails`` emits segments in increasing order within a
# match, so the only places the segment id DROPS are the match joins. That is
# checked against the expected number of matches, and each block's local indices
# are checked against that match's roster size; anything unexpected raises.
# --------------------------------------------------------------------------- #
def split_match_indices(split_code: int, max_matches: int) -> list[int]:
    """The match indices whose trails make up ``load_split_trails(...)``."""
    n_matches = len(np.load(DATASET_NPZ)["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))
    return [i for i in range(n_matches) if per_match[i] == split_code][:max_matches]


def match_player_ids(match_idx: int) -> np.ndarray:
    """Global provider player ids for a match index, without loading tracking."""
    if DATASETS[DATASET]["loader"] == "pff":
        from src.ffpc import (_build_player_axis, _competition_for, _read_json,
                              available_game_ids)
        gid = available_game_ids()[match_idx]
        comp = _competition_for(gid)
        meta = _read_json(comp / "Metadata" / f"{gid}.json")[0]
        roster = _read_json(comp / "Rosters" / f"{gid}.json")
        _, pid, _, _ = _build_player_axis(roster, str(meta["homeTeam"]["id"]))
        return pid
    for idx, match in enumerate(iter_matches()):     # generic (slow) fallback
        if idx == match_idx:
            return np.asarray(match.player_id)
    raise SystemExit(f"match index {match_idx} not found")


def trail_global_player_ids(trails: Trails, split_code: int,
                            max_matches: int) -> np.ndarray:
    """(N,) global player id per trail (see the module comment above)."""
    want = split_match_indices(split_code, max_matches)
    seg = np.asarray(trails.segment)
    starts = [0, *(np.where(np.diff(seg) < 0)[0] + 1).tolist(), len(seg)]
    n_blocks = len(starts) - 1
    if n_blocks != len(want):
        raise SystemExit(
            f"trail->match mapping failed: found {n_blocks} match blocks in the "
            f"split-{split_code} trails but expected {len(want)} ({want}); the "
            "segment-drop boundary heuristic no longer holds -- rebuild the "
            "trails with per-match provenance before fitting per-player models")
    gid = np.empty(len(seg), dtype=np.int64)
    for b, midx in enumerate(want):
        lo, hi = starts[b], starts[b + 1]
        pid = match_player_ids(midx)
        loc = np.asarray(trails.player[lo:hi], dtype=np.int64)
        if loc.size and loc.max() >= len(pid):
            raise SystemExit(f"block {b} (match {midx}) has local player index "
                             f"{loc.max()} >= roster size {len(pid)}")
        gid[lo:hi] = pid[loc]
    return gid


def trail_capacities(trails: Trails, split_code: int, max_matches: int) -> dict:
    """{'vmax_pp','amax_pp'} per-trail capacity columns from the TRAIN-only table.

    Players absent from the table (they never appear in a train match) take the
    table's stored median fallbacks, exactly as scripts.build_dataset does for
    the profile MLP's v_max feature.
    """
    table = load_vmax_table()
    gid = trail_global_player_ids(trails, split_code, max_matches)
    vmax = np.array([table["vmax"].get(int(g), table["vmax_fallback"]) for g in gid])
    amax = np.array([table["amax"].get(int(g), table["amax_fallback"]) for g in gid])
    n_fb = int(sum(int(g) not in table["vmax"] for g in np.unique(gid)))
    print(f"    per-player capacity: {len(np.unique(gid))} players "
          f"({n_fb} on fallback), v_max {vmax.min():.2f}-{vmax.max():.2f} m/s, "
          f"a_max {amax.min():.2f}-{amax.max():.2f} m/s²", flush=True)
    return {"vmax_pp": vmax, "amax_pp": amax}


def subsample_idx(n_total: int, n: int, seed: int) -> np.ndarray:
    """The exact indices ``Trails.subsample(n, seed)`` keeps (to align extras)."""
    if n >= n_total:
        return np.arange(n_total)
    idx = np.random.default_rng(seed).choice(n_total, size=n, replace=False)
    idx.sort()
    return idx


def _make_model(name, cont_vals, discrete_vals):
    spec = ALL_SPECS[name]
    kwargs = {**spec.fixed, **dict(zip(spec.cont, cont_vals)), **discrete_vals}
    if name.startswith("WuSwartz"):
        kwargs["n_dirs"] = WS_FIT_DIRS
    return spec.cls(**kwargs)


def _objective(cont_vals, name, discrete_vals, trails, res, hrm, extra=None):
    """Negative validation score (to minimise). Cheap gate, area only if gated."""
    model = _make_model(name, cont_vals, discrete_vals)
    pred = tta_to_targets(model.predict, trails, extra=extra)
    hit_ratio = float(np.mean(pred <= FIT_HORIZON_S))
    if hit_ratio < hrm:
        # push toward feasibility without paying for a grid pass
        return INFEASIBLE + INFEASIBLE * (hrm - hit_ratio)
    areas = reachable_areas(model.predict, trails, res=res, extra=extra)
    correct = pred <= FIT_HORIZON_S
    mean_area = float(areas[correct].mean())  # == score^-1 ; minimise area
    return mean_area


def fit_one(name, trails_fit, hrm, rng, extra=None):
    spec = ALL_SPECS[name]
    bounds = list(spec.cont.values())
    discrete_combos = [dict()]
    if spec.discrete:
        keys = list(spec.discrete)
        from itertools import product
        discrete_combos = [dict(zip(keys, vals))
                           for vals in product(*spec.discrete.values())]

    best = None
    res_fit = FIT_RES
    for dvals in discrete_combos:
        de = differential_evolution(
            _objective, bounds,
            args=(name, dvals, trails_fit, res_fit, hrm, extra),
            maxiter=DE_MAXITER, popsize=DE_POPSIZE, tol=1e-3,
            seed=int(rng.integers(1 << 31)), polish=True, init="sobol",
        )
        cont = dict(zip(spec.cont, de.x))
        params = {**cont, **dvals}
        # final score on the SAME trails the objective used (hit_ratio is grid-
        # independent, so a config DE accepted as feasible passes the gate here),
        # at the fine grid for a comparable area.
        model = _make_model(name, de.x, dvals)
        pred = tta_to_targets(model.predict, trails_fit, extra=extra)
        areas = reachable_areas(model.predict, trails_fit, res=GRID_RES, extra=extra)
        sc = score(pred, areas, FIT_HORIZON_S, hrm)
        cand = {"params": params, "fit": sc}
        # prefer feasible (gated) configs; among those, smallest area
        key = (not sc["passes_gate"], sc["mean_area"] if sc["passes_gate"]
               else -sc["hit_ratio"])
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


def _train_match_tracks(max_matches: int) -> list:
    """Cleaned (pos, vel, state_valid) tracks from the train-split matches.

    One pass over the raw matches so the Anzer & Bauer tables can extract
    displacements at *every* time offset without re-loading per horizon.
    """
    if DATASETS[DATASET]["loader"] not in ("pff", "nba"):
        raise SystemExit("AnzerBauer fit needs a multi-match dataset (the "
                         "single-match player/block split would leak here)")
    n_matches = len(np.load(DATASET_NPZ)["recording_ids"])
    per_match = _assign_keys(np.arange(n_matches))
    want = [i for i in range(n_matches) if per_match[i] == 0][:max_matches]
    tracks = []
    for idx, match in enumerate(iter_matches()):
        if idx in want:
            print(f"    + match[{idx}] {match.recording_id}", flush=True)
            for seg_id, lo, hi in match.segment_slices():
                for p in range(match.P):
                    in_field = match.in_field[lo:hi, p]
                    if not np.any(in_field == 1):
                        continue
                    tr = clean_track(match.positions[lo:hi, p, :], in_field,
                                     player=p, segment=seg_id)
                    if tr.state_valid.any():
                        tracks.append((tr.pos, tr.vel, tr.state_valid))
        if idx >= max(want):
            break
    return tracks


def _anzer_displacements(tracks, t: float):
    """Heading-frame displacements after ``t`` seconds, over all train tracks.

    Returns (speed, rx, ry) for every valid player state whose position is
    still known ``t`` later: rx along the current velocity, ry perpendicular.
    Shared by the circle (``fit_anzer``) and hull (``fit_anzer_hull``) fits so
    both see exactly the same empirical point cloud.
    """
    off = int(round(t * FPS))
    vs, rxs, rys = [], [], []
    for pos, vel, valid in tracks:
        si = np.where(valid)[0]
        ti = si + off
        keep = ti < pos.shape[0]
        si, ti = si[keep], ti[keep]
        if si.size == 0:
            continue
        good = ~np.isnan(pos[ti, 0]) & ~np.isnan(pos[si, 0])
        si, ti = si[good], ti[good]
        if si.size == 0:
            continue
        dx = pos[ti, 0] - pos[si, 0]
        dy = pos[ti, 1] - pos[si, 1]
        vx, vy = vel[si, 0], vel[si, 1]
        phi = np.arctan2(vy, vx)
        cphi, sphi = np.cos(phi), np.sin(phi)
        vs.append(np.hypot(vx, vy))
        rxs.append(dx * cphi + dy * sphi)    # displacement, heading frame
        rys.append(-dx * sphi + dy * cphi)
    return np.concatenate(vs), np.concatenate(rxs), np.concatenate(rys)


def fit_anzer(hrm, max_matches: int = 2) -> dict:
    """Fit the Anzer & Bauer empirical circle tables (paper §3.2).

    Per (0.5 km/h speed bin, time offset): centre = mean displacement along the
    heading, radius = ``hrm``-quantile of the distance from that centre (the
    paper's convex hull is the quantile -> 1 limit; the gate quantile puts the
    model at the benchmark's operating point). A single global radius scale is
    then calibrated on train trails so the Δt = 1 s hit ratio meets the gate
    exactly -- the same objective the DE-fitted models optimise against.
    """
    tracks = _train_match_tracks(max_matches)
    t_steps = np.round(np.arange(ANZER_T_STEP, ANZER_T_MAX + 1e-9, ANZER_T_STEP), 10)
    edges = np.arange(0.0, ANZER_V_MAX + ANZER_BIN_MS / 2, ANZER_BIN_MS)
    v_centers = 0.5 * (edges[:-1] + edges[1:])
    nv, nt = len(v_centers), len(t_steps)
    C = np.full((nv, nt), np.nan)
    R = np.full((nv, nt), np.nan)

    for j, t in enumerate(t_steps):
        v, rx, ry = _anzer_displacements(tracks, t)

        bi = np.clip(np.digitize(v, edges) - 1, 0, nv - 1)
        order = np.argsort(bi, kind="stable")
        bounds = np.searchsorted(bi[order], np.arange(nv + 1))
        for b in range(nv):
            sl = order[bounds[b]:bounds[b + 1]]
            if sl.size < ANZER_MIN_BIN:
                continue
            cx = float(rx[sl].mean())
            C[b, j] = cx
            R[b, j] = float(np.quantile(np.hypot(rx[sl] - cx, ry[sl]), hrm))

    # fill sparse speed bins by interpolating across bins (constant beyond ends)
    for j in range(nt):
        pop = np.where(np.isfinite(R[:, j]))[0]
        C[:, j] = np.interp(v_centers, v_centers[pop], C[pop, j])
        R[:, j] = np.interp(v_centers, v_centers[pop], R[pop, j])
    R = np.maximum.accumulate(R, axis=1)   # radius can only grow with time

    # prepend the exact t=0 anchor (centre and radius both zero)
    t_grid = np.concatenate([[0.0], t_steps])
    C = np.concatenate([np.zeros((nv, 1)), C], axis=1)
    R = np.concatenate([np.zeros((nv, 1)), R], axis=1)

    def make(scale):
        return AnzerBauer(v_centers.tolist(), t_grid.tolist(), C.tolist(),
                          (R * scale).tolist(), radius_q=hrm)

    # calibrate one global radius scale so the Δt=1s train hit ratio meets the
    # gate (binning + interpolation would otherwise leave it a hair off). Uses
    # the SAME trail sample the fit score is reported on, exactly as the DE
    # models optimise their objective on it.
    trails_fit = load_split_trails(0, FIT_HORIZON_S, max_matches=max_matches) \
        .subsample(FIT_TRAILS, seed=SEED + 7)
    lo, hi = 0.8, 1.5
    if np.mean(tta_to_targets(make(hi).predict, trails_fit) <= FIT_HORIZON_S) < hrm:
        hi = 2.5   # generous fallback; never expected
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        hit = np.mean(tta_to_targets(make(mid).predict, trails_fit) <= FIT_HORIZON_S)
        lo, hi = (lo, mid) if hit >= hrm else (mid, hi)
    scale = hi
    model = make(scale)

    pred = tta_to_targets(model.predict, trails_fit)
    areas = reachable_areas(model.predict, trails_fit, res=GRID_RES)
    sc = score(pred, areas, FIT_HORIZON_S, hrm)
    params = {"v_centers": v_centers.tolist(), "t_grid": t_grid.tolist(),
              "center": C.tolist(), "radius": (R * scale).tolist(),
              "radius_q": hrm}
    desc = (f"table {nv}x{len(t_grid)} ({ANZER_BIN_MS*3.6:.1f} km/h bins x "
            f"{ANZER_T_STEP:g}s grid to {ANZER_T_MAX:g}s), radius_q={hrm}, "
            f"scale={scale:.4f}")
    return {"params": params, "fit": sc, "desc": desc}


def _hull_radial(rx, ry, K: int):
    """Convex hull of a 2-D point cloud, as (anchor, r(θ_k)) with K angles.

    The anchor is the cloud's mean (always interior to its own convex hull), so
    the hull is star-shaped about it and r(θ) is single-valued. With the hull's
    facet inequalities n·x + b <= 0, the ray anchor + s·u(θ) leaves the hull at
    s = min over facets with n·u > 0 of -(n·anchor + b) / (n·u) -- exact, no
    polygon walking. Returns (cx, cy, r) or None if the cloud is degenerate.
    """
    cx, cy = float(rx.mean()), float(ry.mean())
    pts = np.column_stack([rx, ry])
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    eq = hull.equations                       # (F, 3): n_x, n_y, b
    n = eq[:, :2]
    # distance from the anchor to each facet plane along its own normal (>0
    # for an interior anchor); floored so a numerically-on-the-hull anchor
    # cannot produce a negative radius.
    off = np.maximum(-(n @ np.array([cx, cy]) + eq[:, 2]), 1e-9)
    th = 2.0 * np.pi * np.arange(K) / K
    u = np.column_stack([np.cos(th), np.sin(th)])          # (K, 2)
    proj = u @ n.T                                          # (K, F)
    s = np.where(proj > 1e-12, off[None, :] / np.maximum(proj, 1e-12), np.inf)
    r = s.min(axis=1)
    if not np.all(np.isfinite(r)):
        return None
    return cx, cy, r


def _suvat_keep(v, rx, ry, t: float) -> np.ndarray:
    """SUVAT noise-floor screen, identical to pipeline.sampling's sample filter.

    A (source state, offset t, displacement) pair survives iff the displacement
    is non-degenerate (d > 1e-6, sampling's ``valid_d``) and physically
    reachable in the time it took: t >= t_suvat(v, cosθ, d), with the canonical
    caps from config (SUVAT_A_MAX / SUVAT_V_MAX) via the repo's own
    ``pipeline.suvat.suvat_time`` -- the same call sampling makes, so the hull's
    point cloud is filtered by exactly the rule that filtered the MLP's labels.
    """
    d = np.hypot(rx, ry)
    valid_d = d > 1e-6
    safe_d = np.where(valid_d, d, 1.0)
    return valid_d & (t >= suvat_time(v, rx / safe_d, d))


def fit_anzer_hull(hrm=0.99, max_matches: int = 2, suvat_screen: bool = False) -> dict:
    """Fit the FAITHFUL Anzer & Bauer envelope (paper §3.2): convex hulls.

    Same tracks, speed bins and time grid as ``fit_anzer``, but each
    (speed bin, time offset) cell keeps the convex hull of ALL its observed
    heading-frame displacements instead of a quantile circle -- no quantile, no
    circle, and (unlike our adaptation) no global radius scale: the hull IS the
    model. Hulls are stored radially, r(θ_k) about the cell's mean displacement,
    so interpolation across speed/time is a per-angle linear blend. One time
    offset is processed at a time and hulls are reduced immediately, so the raw
    point clouds never all live in memory at once.
    """
    tracks = _train_match_tracks(max_matches)
    t_steps = np.round(np.arange(ANZER_T_STEP, ANZER_T_MAX + 1e-9, ANZER_T_STEP), 10)
    edges = np.arange(0.0, ANZER_V_MAX + ANZER_BIN_MS / 2, ANZER_BIN_MS)
    v_centers = 0.5 * (edges[:-1] + edges[1:])
    nv, nt, K = len(v_centers), len(t_steps), ANZER_HULL_K
    CX = np.full((nv, nt), np.nan)
    CY = np.full((nv, nt), np.nan)
    R = np.full((nv, nt, K), np.nan)
    n_cells = n_degen = 0
    n_pairs_raw = n_pairs_kept = 0
    worst_cell = (1.0, None)     # (retention, (speed bin, t)) over hulled cells

    for j, t in enumerate(t_steps):
        v, rx, ry = _anzer_displacements(tracks, t)
        n_raw_t = v.size
        # pre-screen per-speed-bin counts, so per-cell retention is exact
        cnt_raw = np.bincount(np.clip(np.digitize(v, edges) - 1, 0, nv - 1),
                              minlength=nv)
        if suvat_screen:
            keep = _suvat_keep(v, rx, ry, t)
            v, rx, ry = v[keep], rx[keep], ry[keep]
        n_pairs_raw += n_raw_t
        n_pairs_kept += v.size
        bi = np.clip(np.digitize(v, edges) - 1, 0, nv - 1)
        order = np.argsort(bi, kind="stable")
        bounds = np.searchsorted(bi[order], np.arange(nv + 1))
        for b in range(nv):
            sl = order[bounds[b]:bounds[b + 1]]
            if sl.size < ANZER_MIN_BIN:
                continue
            got = _hull_radial(rx[sl], ry[sl], K)
            n_cells += 1
            if got is None:
                n_degen += 1
                continue
            CX[b, j], CY[b, j], R[b, j] = got
            if suvat_screen and cnt_raw[b]:
                ret = sl.size / cnt_raw[b]
                if ret < worst_cell[0]:
                    worst_cell = (ret, f"v≈{v_centers[b]:.2f} m/s, t={t:.1f}s")
        del v, rx, ry, bi, order
        print(f"    t={t:.1f}s  hulls done"
              + (f"  ({n_pairs_kept:,}/{n_pairs_raw:,} pairs kept)"
                 if suvat_screen else ""), flush=True)

    # fill sparse/degenerate speed bins by interpolating across bins, in the
    # radial representation (constant beyond the populated ends)
    for j in range(nt):
        pop = np.where(np.isfinite(CX[:, j]))[0]
        if pop.size == 0:
            raise SystemExit(f"no populated speed bin at t index {j}")
        CX[:, j] = np.interp(v_centers, v_centers[pop], CX[pop, j])
        CY[:, j] = np.interp(v_centers, v_centers[pop], CY[pop, j])
        for k in range(K):
            R[:, j, k] = np.interp(v_centers, v_centers[pop], R[pop, j, k])
    R = np.maximum.accumulate(R, axis=1)   # the envelope can only grow with time

    # prepend the exact t=0 anchor (anchor at the origin, zero extent)
    t_grid = np.concatenate([[0.0], t_steps])
    CX = np.concatenate([np.zeros((nv, 1)), CX], axis=1)
    CY = np.concatenate([np.zeros((nv, 1)), CY], axis=1)
    R = np.concatenate([np.zeros((nv, 1, K)), R], axis=1)

    model = AnzerBauerHull(v_centers.tolist(), t_grid.tolist(),
                           CX.tolist(), CY.tolist(), R.tolist())
    trails_fit = load_split_trails(0, FIT_HORIZON_S, max_matches=max_matches) \
        .subsample(FIT_TRAILS, seed=SEED + 7)
    pred = tta_to_targets(model.predict, trails_fit)
    areas = reachable_areas(model.predict, trails_fit, res=GRID_RES)
    sc = score(pred, areas, FIT_HORIZON_S, hrm)   # hrm reports the gate only:
    # the faithful model has nothing to calibrate against it
    params = {"v_centers": v_centers.tolist(), "t_grid": t_grid.tolist(),
              "center_x": CX.tolist(), "center_y": CY.tolist(),
              "radius": R.tolist(), "scale": 1.0}
    desc = (f"hull table {nv}x{len(t_grid)}x{K} ({ANZER_BIN_MS*3.6:.1f} km/h bins "
            f"x {ANZER_T_STEP:g}s grid to {ANZER_T_MAX:g}s), no scale "
            f"(faithful), degenerate cells {n_degen}/{n_cells}")
    out = {"params": params, "fit": sc, "desc": desc}
    if suvat_screen:
        ret = n_pairs_kept / max(n_pairs_raw, 1)
        out["suvat_screen"] = {
            "pairs_before": n_pairs_raw, "pairs_after": n_pairs_kept,
            "retention": ret, "removed_pct": 100.0 * (1.0 - ret),
            "worst_cell_retention": worst_cell[0], "worst_cell": worst_cell[1],
        }
        out["desc"] = desc + (
            f"; SUVAT-screened: kept {n_pairs_kept:,}/{n_pairs_raw:,} pairs "
            f"({100*ret:.2f}%, {100*(1-ret):.2f}% removed), worst hulled cell "
            f"{100*worst_cell[0]:.2f}% ({worst_cell[1]})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-ratio-min", type=float, default=0.95)
    ap.add_argument("--anzer-hull-only", action="store_true",
                    help="fit ONLY the faithful Anzer & Bauer convex-hull model "
                         "and write it to --models-json (default "
                         "outputs/<dataset>/motion_models_anzerhull.json); no "
                         "other model is fitted or touched")
    ap.add_argument("--suvat-screen", action="store_true",
                    help="with --anzer-hull-only: screen the displacement cloud "
                         "with the SUVAT noise floor (pipeline.sampling's own "
                         "filter) before hulling, so the hull and the MLP see "
                         "data filtered by the same physical-consistency rule. "
                         "Fits AnzerBauerHullSUVAT to a separate default file "
                         "(motion_models_anzerhull_suvat.json)")
    ap.add_argument("--models", nargs="*", default=list(MODEL_SPECS),
                    help="model names (see MODEL_SPECS / PERPLAYER_SPECS)")
    ap.add_argument("--models-json", type=str, default=None,
                    help="output json (default outputs/<dataset>/motion_models.json); "
                         "per-player variants MUST write their own file so the "
                         "canonical global-parameter fit is never overwritten")
    args = ap.parse_args()
    hrm = args.hit_ratio_min

    if args.anzer_hull_only:
        # standalone path: a fresh file holding only the hull model, so no
        # existing fitted-models json is read, merged into or overwritten.
        screened = args.suvat_screen
        name = "AnzerBauerHullSUVAT" if screened else "AnzerBauerHull"
        default_json = ("motion_models_anzerhull_suvat.json" if screened
                        else "motion_models_anzerhull.json")
        path = (Path(args.models_json) if args.models_json
                else OUTPUTS_DIR / default_json)
        print(f"dataset={DATASET}  Anzer & Bauer convex-hull fit only"
              + ("  (SUVAT-screened displacements)" if screened else " (faithful)"))
        t0 = time.perf_counter()
        res = fit_anzer_hull(hrm, suvat_screen=screened)
        sc = res["fit"]
        print(f"\n[{name}] {HULL_SPECS[name].paper}"
              f"   ({time.perf_counter() - t0:.0f}s)")
        print(f"  params: {res['desc']}")
        if screened:
            s = res["suvat_screen"]
            print(f"  SUVAT screen: kept {s['pairs_after']:,} of "
                  f"{s['pairs_before']:,} displacement pairs "
                  f"({100*s['retention']:.2f}%); removed {s['removed_pct']:.2f}%; "
                  f"worst hulled cell kept {100*s['worst_cell_retention']:.2f}% "
                  f"({s['worst_cell']})")
        print(f"  fit: hit_ratio={sc['hit_ratio']:.4f}  "
              f"mean_area={sc['mean_area']:.1f} m²  score={sc['score']:.5f}")
        out = {"_meta": {"dataset": DATASET, "horizon_s": FIT_HORIZON_S,
                         "hit_ratio_min": hrm, "fit_split": "train",
                         "vmax_scope": VMAX_SCOPE, "dataset_npz": str(DATASET_NPZ),
                         "note": ("SUVAT-screened hull; no scale calibration"
                                  if screened else
                                  "faithful hull replication; no scale calibration")},
               name: {"paper": HULL_SPECS[name].paper,
                      **{k: v for k, v in res.items() if k != "desc"},
                      "desc": res["desc"]}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, default=float))
        print(f"\nsaved -> {path}")
        return

    rng = np.random.default_rng(SEED)
    print(f"dataset={DATASET}  Δt={FIT_HORIZON_S}s  hit_ratio_min={hrm}")
    print("building Δt=1s TRAIN trails ...", flush=True)
    train = load_split_trails(0, FIT_HORIZON_S, max_matches=2)
    print(f"  {len(train):,} train trails", flush=True)

    # merge into any existing fit so a single new model can be added without
    # re-fitting the (slow) others.
    path = Path(args.models_json) if args.models_json else OUTPUTS_DIR / "motion_models.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["_meta"] = {"dataset": DATASET, "horizon_s": FIT_HORIZON_S,
                    "hit_ratio_min": hrm, "fit_split": "train",
                    "vmax_scope": VMAX_SCOPE, "dataset_npz": str(DATASET_NPZ)}
    # per-player capacity columns (only built if a per-player model is fitted)
    caps_full = None
    if any(is_perplayer(n) for n in args.models):
        print("  per-player capacity table lookup ...", flush=True)
        caps_full = trail_capacities(train, 0, 2)

    for name in args.models:
        spec = ALL_SPECS[name]
        t0 = time.perf_counter()
        if name == "AnzerBauer":   # direct empirical fit, no parameter search
            res = fit_anzer(hrm)
        else:
            n_fit = FIT_TRAILS_WS if name.startswith("WuSwartz") else FIT_TRAILS
            idx = subsample_idx(len(train), n_fit, SEED + 7)
            tf_fit = train.select(idx)
            extra = ({k: v[idx] for k, v in caps_full.items()}
                     if is_perplayer(name) else None)
            res = fit_one(name, tf_fit, hrm, rng, extra=extra)
            if is_perplayer(name):
                res["variant"] = name[-2:]
                res["per_player"] = sorted(
                    k for k in res["params"] if k in ("c_v", "c_a"))
        dt = time.perf_counter() - t0
        sc = res["fit"]
        pstr = res.pop("desc", None) or ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res["params"].items())
        gate = "" if sc["passes_gate"] else "  (BELOW GATE)"
        print(f"\n[{name}] {spec.paper}   ({dt:.0f}s)")
        print(f"  params: {pstr}")
        print(f"  fit: hit_ratio={sc['hit_ratio']:.4f}  mean_area={sc['mean_area']:.1f} m²"
              f"  score={sc['score']:.5f}{gate}")
        out[name] = {"paper": spec.paper, **res}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
