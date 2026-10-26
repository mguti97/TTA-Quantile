"""Generate the paper figures into paper/figures/ (run after training).

Each figure is guarded independently so one failure does not abort the rest.
Reuses the project's own inference / metric / motion-model utilities; nothing
here re-implements model maths.

    python -m scripts.make_figures                # tau = config.PINBALL_TAU
    python -m scripts.make_figures --tau 0.01     # loads mlp_tau0.01.pt
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import _assign_keys  # noqa: E402
from scripts.fit_motion_models import load_split_trails  # noqa: E402
from src.config import (  # noqa: E402
    A_NORM,
    D_NORM,
    DATASET_NPZ,
    HORIZON_MIN_S,
    MODEL_PT,
    N_FEATURES,
    OUTPUTS_DIR,
    PINBALL_TAU,
    SEED,
    V_NORM,
    VMAX_SCOPE,
)
from src.evaluation.metrics import (  # noqa: E402
    ANGLE_BANDS,
    DIST_BANDS,
    HORIZON_BANDS,
    calibration,
)
from src.evaluation.reachable_area import (  # noqa: E402
    grid_tta,
    matched_coverage_area,
)
from src.inference.heatmap import (  # noqa: E402
    features_for_state,
    heatmaps_for_frame,
    pitch_grid,
)
from src.evaluation.visualize import render_player_heatmap  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402
from src.model.motion_models import from_params  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUB = 1_500_000   # subsample for the band/calibration figures

# SUVAT is the tracking-signal cleaning filter, NOT an arrival-time baseline.
# The illustrative figures use the simplest fitted physical model -- constant
# speed (isotropic, d/v_max) -- as the naive-physics reference line.
REF_MODEL = "ConstSpeed"
REF_LABEL = "constant-speed physics"


def _ref_motion(mm):
    """The fitted constant-speed reference model (or None if unavailable)."""
    if mm is not None and REF_MODEL in mm:
        return from_params(REF_MODEL, mm[REF_MODEL]["params"])
    return None


def _ref_surface(ref, states, res=None):
    """Per-player arrival surface for the reference motion model, shape
    (n_players, nx, ny) -- mirrors heatmaps_for_frame's grid exactly."""
    from src.config import GRID_RES

    cells, nx, ny = pitch_grid(res or GRID_RES)
    out = []
    for (p, v, a) in states:
        tf = features_for_state(p, v, a, cells)
        out.append(np.asarray(ref.predict(tf), dtype=np.float64).reshape(nx, ny))
    return np.stack(out), nx, ny


def model_path(tau: float) -> Path:
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def profile_vmax_path(tau: float) -> Path:
    # Under KOGNIA_VMAX_SCOPE=train_split (the leak-free revision setting),
    # DATASET_NPZ is samples_vmaxtrain.npz -- identical X/y/split to
    # samples.npz, only meta_vmax differs (train-split v_max table) -- and the
    # matching profile checkpoint is mlp_profile_vmax_vmaxtrain.pt. The plain
    # mlp_profile_vmax.pt pools v_max over both halves (look-ahead leak).
    suffix = "" if tau == PINBALL_TAU else f"_tau{tau:g}"
    scope = "_vmaxtrain" if VMAX_SCOPE == "train_split" else ""
    return OUTPUTS_DIR / f"mlp_profile_vmax{suffix}{scope}.pt"


def _mlp(in_dim=N_FEATURES, path=None, tau=PINBALL_TAU):
    m = TTAModel(in_dim=in_dim).to(DEVICE)
    m.load_state_dict(torch.load(path or model_path(tau), map_location=DEVICE))
    m.eval()
    return m


def _mlp_predict(m, X, chunk=1_000_000):
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            e = min(s + chunk, len(X))
            out[s:e] = m(torch.from_numpy(X[s:e]).to(DEVICE)).cpu().numpy()
    return out


def _test_game_ids(d):
    rec = [str(x) for x in d["recording_ids"]]
    per_match = _assign_keys(np.arange(len(rec)))
    return [rec[i] for i in range(len(rec)) if per_match[i] == 2]


# --------------------------------------------------------------------------- F1
def fig_surface(d, mm, tau=PINBALL_TAU):
    """One player's MLP (white) vs constant-speed physics (red dashed) surface."""
    from src.ffpc import load_ffpc_match

    gid = _test_game_ids(d)[0]
    match = load_ffpc_match(gid)
    model = _mlp(tau=tau)
    ref = _ref_motion(mm)
    if ref is None:
        raise RuntimeError(f"{REF_MODEL} reference unavailable in motion_models.json")

    # pick a frame with many players and a fast mover
    best = None
    for seg_id, lo, hi in match.segment_slices():
        for frame in range(lo + 300, hi - 300, 150):
            on = np.where(match.in_field[frame] == 1)[0]
            if len(on) < 18:
                continue
            states, labels = [], []
            for p in on:
                tr = clean_track(match.positions[lo:hi, p, :],
                                 match.in_field[lo:hi, p], p, seg_id)
                li = frame - lo
                if li < tr.L and tr.state_valid[li] and not np.isnan(tr.pos[li, 0]):
                    states.append((tr.pos[li], tr.vel[li], tr.acc[li]))
                    labels.append(p)
            if len(states) < 14:
                continue
            speeds = [np.hypot(*s[1]) for s in states]
            if max(speeds) > 5.5:
                best = (frame, states, labels, speeds)
                break
        if best:
            break
    if not best:
        raise RuntimeError("no suitable frame found for surface figure")

    frame, states, labels, speeds = best
    heat, nx, ny = heatmaps_for_frame(model, states, DEVICE)
    ref_heat, _, _ = _ref_surface(ref, states)
    j = int(np.argmax(speeds))
    out = FIG_DIR / "surface_mlp_vs_suvat.pdf"
    render_player_heatmap(
        heat[j], states[j][0], model_suvat=ref_heat[j],
        title=f"Arrival-time surface (game {gid}, frame {frame}, "
              f"v={speeds[j]:.1f} m/s) — white=MLP, red dashed={REF_LABEL}",
        path=out,
    )
    return out


# --------------------------------------------------------------------------- F2
def fig_profile_floor(d, mm, tau=PINBALL_TAU):
    """1-D arrival time vs straight-ahead distance: physics ref, MLP, floor."""
    v = 5.0
    dd = np.linspace(0.0, 100.0, 400)
    X = np.zeros((len(dd), N_FEATURES), dtype=np.float32)
    X[:, 0] = dd / D_NORM
    X[:, 1] = 1.0          # cos theta (straight ahead)
    X[:, 2] = 0.0          # sin theta
    X[:, 3] = v / V_NORM
    # ax, ay = 0
    mlp = _mlp(tau=tau)
    ref = _ref_motion(mm)
    if ref is None:
        raise RuntimeError(f"{REF_MODEL} reference unavailable in motion_models.json")
    t_mlp = _mlp_predict(mlp, X)
    tf = {"d": dd, "v": np.full_like(dd, v),
          "cos_t": np.ones_like(dd), "sin_t": np.zeros_like(dd)}
    t_ref = np.asarray(ref.predict(tf), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dd, t_ref, label=REF_LABEL, color="#c0392b", lw=2)
    ax.plot(dd, t_mlp, label="MLP", color="#27ae60", lw=2)
    ax.axhline(HORIZON_MIN_S, color="k", ls=":", lw=1,
               label=f"horizon floor = {HORIZON_MIN_S:g} s")
    ax.set_xlabel("distance to target straight ahead (m)")
    ax.set_ylabel("predicted arrival time (s)")
    ax.set_title(f"Arrival-time profile (current speed {v:.0f} m/s, heading at target)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(alpha=0.3)
    out = FIG_DIR / "profile_1d_floor.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    # inset zoom near origin to show the floor
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    near = dd <= 8
    ax2.plot(dd[near], t_ref[near], label=REF_LABEL, color="#c0392b", lw=2)
    ax2.plot(dd[near], t_mlp[near], label="MLP", color="#27ae60", lw=2)
    ax2.axhline(HORIZON_MIN_S, color="k", ls=":", lw=1,
                label=f"floor = {HORIZON_MIN_S:g} s")
    ax2.set_xlabel("distance (m)")
    ax2.set_ylabel("arrival time (s)")
    ax2.set_title("Near-origin detail: learned floor at the horizon minimum")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.savefig(FIG_DIR / "profile_1d_floor_zoom.pdf", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    return out


# --------------------------------------------------------------------------- F3
def fig_stratified(d, mm, tau=PINBALL_TAU):
    """Grouped-bar MAE by distance band: best physics (two-segment), base MLP,
    and the player-profile MLP (+max-speed) -- the two learned models now
    reported together as the main-comparison result."""
    te = d["split"] == 2
    idx = np.random.default_rng(SEED).choice(int(te.sum()), size=min(SUB, int(te.sum())),
                                             replace=False)
    X = d["X"][te][idx]
    y = d["y"][te][idx].astype(np.float64)
    dist = d["meta_d"][te][idx].astype(np.float64)
    v = d["meta_v"][te][idx].astype(np.float64)

    mlp = _mlp(tau=tau)
    preds = {}
    if mm is not None and "TwoSegment" in mm:
        ts = from_params("TwoSegment", mm["TwoSegment"]["params"])
        tf = {"d": dist, "v": v, "cos_t": X[:, 1].astype(np.float64),
              "sin_t": X[:, 2].astype(np.float64)}
        preds["Two-segment"] = np.asarray(ts.predict(tf), dtype=np.float64)

    hull_json = OUTPUTS_DIR / "motion_models_anzerhull_suvat.json"
    if hull_json.exists():
        import json as _json
        hull_entry = _json.loads(hull_json.read_text())["AnzerBauerHullSUVAT"]
        hull = from_params("AnzerBauerHullSUVAT", hull_entry["params"])
        tf_h = {"d": dist, "v": v, "cos_t": X[:, 1].astype(np.float64),
                "sin_t": X[:, 2].astype(np.float64)}
        preds["Reachable hulls"] = np.asarray(hull.predict(tf_h), dtype=np.float64)

    preds["MLP"] = _mlp_predict(mlp, X).astype(np.float64)

    profile_pt = profile_vmax_path(tau)
    if profile_pt.exists():
        vmax = d["meta_vmax"][te][idx].astype(np.float32)
        Xp = np.concatenate([X, (vmax / V_NORM)[:, None].astype(np.float32)], axis=1)
        profile_mlp = _mlp(in_dim=N_FEATURES + 1, path=profile_pt, tau=tau)
        preds["MLP+vmax"] = _mlp_predict(profile_mlp, Xp).astype(np.float64)

    names = list(preds)
    bands = DIST_BANDS
    labels = [f"{lo}-{'∞' if np.isinf(hi) else int(hi)}m" for lo, hi in bands]
    maes = {n: [] for n in names}
    for lo, hi in bands:
        m = (dist >= lo) & (dist < hi)
        for n in names:
            maes[n].append(float(np.mean(np.abs(preds[n][m] - y[m]))) if m.any() else np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.8 / len(names)
    xpos = np.arange(len(bands))
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(x) for n, x in zip(names, np.linspace(0.15, 0.85, len(names)))}
    for i, n in enumerate(names):
        ax.bar(xpos + i * w, maes[n], w, label=n, color=colors.get(n))
    ax.set_xticks(xpos + w * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("distance band")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Arrival-time MAE by distance band (held-out test)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = FIG_DIR / "stratified_mae.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- F4
# Canonical matched-coverage areas from the full compare_motion_models run
# (500 test trails) -- plotted here so the figure matches Table `tab:area`
# exactly rather than re-estimating on a noisier subsample. Keyed by tau since
# the operating point (and hence the matched-coverage areas) moves with it.
RA_CANONICAL = {
    0.05: {                        # model: [@90% cov, @95% cov]  (m^2)
        "MLP": [7.7, 12.7],
        "Two-segment": [8.7, 13.4],
        "Const. accel.": [9.2, 21.5],
        "Constant speed": [41.6, 65.0],
    },
    # physics baselines refit at hit_ratio_min=0.99 (matching 1-tau=0.99);
    # MLP = the canonical area-loss model (config.AREA_LAM); values from
    # compare_motion_tau0.01_arealoss.log (table B).
    0.01: {                        # model: [@90% cov, @95% cov, @99% cov]  (m^2)
        "MLP": [9.0, 12.0, 33.2],
        "Two-segment": [10.5, 17.8, 34.1],
        "Const. accel.": [9.9, 18.6, 46.9],
        "Constant speed": [41.6, 65.0, 119.3],
    },
}


# --------------------------------------------------------------------------- F4b
# Full matched-coverage area sweep (all 8 models x 3 coverage levels x 3
# horizons), from outputs/ffpc_match/eval_coverage_sweep_fresh.log (rerun after
# retraining mlp.pt / mlp_profile_vmax.pt on the current ffpc_match build --
# the "FULL SWEEP" block). Keyed by tau since the operating point moves with
# it; only 0.01 populated (the canonical run) since that is the only sweep on
# record. Physics rows are pixel-identical to the prior sweep (same fitted
# motion_models.json, same cached held-out trails); only the two MLP rows moved.
AREA_SWEEP = {
    0.01: {
        # model: {horizon_s: [area @90%, @95%, @99%]}  (m^2)
        "MLP":                   {1: [8.9, 12.4, 31.6],   3: [170.3, 260.8, 661.6],  5: [518.2, 935.5, 1761.7]},
        # leak-free profile model: mlp_profile_vmax_vmaxtrain.pt (per-player
        # v_max from the train split only), same cached 500 test trails;
        # outputs/ffpc_match/vmaxtrain_pipeline.log, "EVAL coverage sweep"
        # FULL SWEEP block (2026-08-28 00:37). Superseded leaky row
        # (mlp_profile_vmax.pt, v_max pooled over both halves):
        #   {1: [9.0, 13.4, 30.9], 3: [165.7, 249.7, 603.1], 5: [505.3, 826.0, 1618.5]}
        "MLP+vmax":              {1: [8.8, 12.6, 31.7],   3: [167.2, 243.3, 637.1],  5: [523.6, 861.1, 1676.0]},
        # empirical Anzer-Bauer reachable hulls, SUVAT-screened input
        # (anzerhull_suvat_coverage_sweep.log, same cached trails)
        "Reachable hulls":       {1: [11.7, 17.9, 34.8],  3: [217.7, 358.0, 777.3],  5: [553.7, 1085.1, 2463.2]},
        "Constant speed":        {1: [41.6, 65.0, 119.3], 3: [338.8, 531.0, 1131.6], 5: [679.2, 1116.2, 2867.3]},
        "Constant accel.":       {1: [9.9, 18.6, 46.9],   3: [168.6, 270.9, 780.0],  5: [543.3, 903.7, 1931.7]},
        "Const. accel.+vmax":    {1: [9.8, 18.6, 48.0],   3: [159.9, 287.8, 984.7],  5: [547.6, 900.1, 2961.4]},
        "Two-segment":           {1: [10.5, 17.8, 34.1],  3: [233.6, 374.8, 763.6],  5: [595.7, 1075.7, 2509.0]},
        # per-player control (R1-M2): the two-segment model refit with the same
        # train-split per-player v_max table the profile MLP sees, one global
        # scale factor on top (perplayer_eval_coverage_sweep.log, "FULL SWEEP").
        # Same cached trails as every other row here.
        "Two-segment (profile)": {1: [12.6, 20.3, 40.4],  3: [214.8, 338.0, 746.8],  5: [518.5, 878.2, 2098.9]},
        "Free-direction accel.": {1: [11.5, 19.7, 43.6],  3: [224.5, 368.1, 837.0],  5: [579.0, 1125.9, 2464.8]},
        "Friction-damped":       {1: [12.9, 18.0, 35.0],  3: [203.3, 333.6, 734.4],  5: [543.3, 1024.3, 2261.1]},
    },
}

# Fixed categorical order: hue and marker/linestyle are assigned once per
# series and never re-cycled if the series subset changes. Learned models are
# solid + thick; physics baselines are dashed + thinner, each with a distinct
# marker shape so identity survives in grayscale print.
AREA_SWEEP_STYLE = {
    "MLP":                   dict(color="#2a78d6", marker="o", ls="-",  lw=2.4, ms=7),
    "MLP+vmax":              dict(color="#1baf7a", marker="s", ls="-",  lw=2.4, ms=7),
    "Reachable hulls":       dict(color="#6b6b6b", marker="h", ls="-.", lw=1.8, ms=7),
    "Constant speed":        dict(color="#eda100", marker="^", ls="--", lw=1.5, ms=6),
    "Constant accel.":       dict(color="#008300", marker="D", ls="--", lw=1.5, ms=6),
    "Const. accel.+vmax":    dict(color="#4a3aa7", marker="P", ls="--", lw=1.5, ms=6),
    "Two-segment":           dict(color="#e34948", marker="v", ls="--", lw=1.5, ms=6),
    # same hue as its global parent, dotted: reads as "that model, personalized"
    "Two-segment (profile)": dict(color="#8c1d1c", marker="v", ls=":",  lw=1.8, ms=6),
    "Free-direction accel.": dict(color="#e87ba4", marker="*", ls="--", lw=1.5, ms=8),
    "Friction-damped":       dict(color="#eb6834", marker="X", ls="--", lw=1.5, ms=6),
}


def fig_area_sweep(d=None, mm=None, tau=PINBALL_TAU):
    """1x3 grid: matched-coverage reachable area vs coverage, one panel per
    horizon (Delta t = 1/3/5 s), both MLP variants and all six physics
    baselines. Log y-axis per panel (each spans ~6-14x) so the tightly
    clustered learned models aren't squashed by the constant-speed outlier."""
    rows = AREA_SWEEP[tau]
    covs = (90, 95, 99)
    horizons = (1, 3, 5)
    names = list(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True)
    for ax, h in zip(axes, horizons):
        for n in names:
            st = AREA_SWEEP_STYLE[n]
            ax.plot(covs, rows[n][h], label=n, color=st["color"],
                     marker=st["marker"], ls=st["ls"], lw=st["lw"],
                     ms=st["ms"], mec="white", mew=0.6)
        ax.set_yscale("log")
        ax.set_xticks(covs)
        ax.set_xticklabels([f"{c}%" for c in covs])
        ax.set_xlim(88, 101)
        ax.set_xlabel("matched coverage")
        ax.set_title(f"$\\Delta t = {h}$ s")
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(axis="y", which="both", labelrotation=45)
    axes[0].set_ylabel("mean reachable area (m$^2$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    out = FIG_DIR / "area_sweep_grid.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_reachable_area(d, mm, tau=PINBALL_TAU):
    """Matched-coverage reachable area (canonical compare_motion_models values)."""
    rows = RA_CANONICAL[tau]
    covs = (0.90, 0.95, 0.99) if len(next(iter(rows.values()))) == 3 else (0.90, 0.95)
    names = list(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.8 / len(covs)
    xpos = np.arange(len(names))
    colors = {"MLP": "#27ae60", "Two-segment": "#8e44ad",
              "Const. accel.": "#e67e22", "Constant speed": "#c0392b"}
    alphas = np.linspace(0.45, 1.0, len(covs))
    for i, c in enumerate(covs):
        ax.bar(xpos + i * w, [rows[n][i] for n in names], w,
               label=f"@{int(c*100)}% coverage",
               color=[colors.get(n) for n in names], alpha=alphas[i])
    ax.set_xticks(xpos + w * (len(covs) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean reachable area (m²)  — lower = more precise")
    ax.set_title("Reachable area at matched coverage (Δt = 1 s, test trails)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = FIG_DIR / "reachable_area.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- F5
def fig_calibration(d, tau=PINBALL_TAU):
    """Coverage by feature band (flat ~target) vs by observed time (ramp artifact)."""
    te = d["split"] == 2
    n_te = int(te.sum())
    idx = np.random.default_rng(SEED + 2).choice(n_te, size=min(3_000_000, n_te),
                                                 replace=False)
    X = d["X"][te][idx]
    y = d["y"][te][idx]
    dist = d["meta_d"][te][idx]
    v = d["meta_v"][te][idx]
    theta = d["meta_theta_deg"][te][idx]
    horizon = d["meta_horizon"][te][idx] if "meta_horizon" in d.files else y

    mlp = _mlp(tau=tau)
    pred = _mlp_predict(mlp, X)
    rows = calibration(pred, y, dist, v, theta, horizon=horizon)
    nominal = 1.0 - tau

    def grab(dim):
        return [(lbl, cov) for (dm, lbl, n, cov) in rows if dm == dim and n > 0]

    dist_rows = grab("distance")
    true_rows = grab("true_t")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    a0 = axes[0]
    a0.plot([r[0] for r in dist_rows], [r[1] for r in dist_rows],
            "o-", color="#27ae60", label="MLP coverage")
    a0.axhline(nominal, color="k", ls="--", label=f"target {nominal:.2f}")
    a0.set_title(f"Coverage by DISTANCE band (feature-conditioned)\nflat ≈ {nominal:.2f} = well calibrated")
    a0.set_xlabel("distance band")
    a0.set_ylabel("empirical coverage  P(pred ≤ actual)")
    a0.set_ylim(min(0.75, nominal - 0.05), 1.02)
    a0.legend()
    a0.grid(alpha=0.3)

    a1 = axes[1]
    a1.plot([r[0] for r in true_rows], [r[1] for r in true_rows],
            "o-", color="#e67e22", label="MLP coverage")
    a1.axhline(nominal, color="k", ls="--", label=f"target {nominal:.2f}")
    a1.set_title("Coverage by OBSERVED-TIME band (outcome-conditioned)\nramp = conditioning artifact, not miscalibration")
    a1.set_xlabel("observed arrival-time band")
    a1.set_ylim(min(0.75, nominal - 0.05), 1.02)
    a1.legend()
    a1.grid(alpha=0.3)
    out = FIG_DIR / "calibration.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile (default {PINBALL_TAU:g})")
    ap.add_argument("--only", nargs="+", default=None,
                    help="render only these figures (e.g. area_sweep_grid stratified_mae)")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="output directory (default paper/figures)")
    args = ap.parse_args()
    tau = args.tau
    global FIG_DIR
    if args.fig_dir is not None:
        FIG_DIR = args.fig_dir
    if tau not in RA_CANONICAL:
        raise SystemExit(f"no RA_CANONICAL entry for tau={tau:g}; add one from "
                         "compare_motion_models.py --tau output before running")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device {DEVICE}  tau={tau:g}  ->  {FIG_DIR}")
    d = np.load(DATASET_NPZ)

    import json
    mm_path = OUTPUTS_DIR / "motion_models.json"
    mm = json.loads(mm_path.read_text()) if mm_path.exists() else None

    figs = [
        ("profile_1d_floor", lambda: fig_profile_floor(d, mm, tau=tau)),
        ("stratified_mae", lambda: fig_stratified(d, mm, tau=tau)),
        ("calibration", lambda: fig_calibration(d, tau=tau)),
        ("reachable_area", lambda: fig_reachable_area(d, mm, tau=tau)),
        ("area_sweep_grid", lambda: fig_area_sweep(d, mm, tau=tau)),
        ("surface_mlp_vs_suvat", lambda: fig_surface(d, mm, tau=tau)),
    ]
    if args.only:
        figs = [(n, f) for n, f in figs if n in args.only]
    print(f"VMAX_SCOPE={VMAX_SCOPE}  data={DATASET_NPZ.name}  "
          f"profile={profile_vmax_path(tau).name}")
    for name, fn in figs:
        try:
            out = fn()
            print(f"  [ok] {name:24s} -> {out}")
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
