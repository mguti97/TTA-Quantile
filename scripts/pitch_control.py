"""Proposed ball-aware pitch control on one held-out frame.

A location x is *controlled* by the team in possession when an attacker reaches it
before any defender AND the ball can be delivered there before any defender can:

        t_possession(x) < t_defense(x)   AND   t_ball(x) < t_defense(x)

where
  * t_possession(x) = min over attacking players of the learned player TTA
    surface (the fastest attacker to x);
  * t_defense(x)    = min over defending players of the same surface;
  * t_ball(x)       = learned ball arrival time from the current ball position,
    oriented by the ball-carrier's heading (the pass model).

We evaluate it at a real pass release in a TEST match (ball position and carrier
known) and render the controlled region. Reuses the player MLP, the pass MLP and
the existing grid/heatmap utilities.

    python -m scripts.pitch_control                # tau = config.PINBALL_TAU
    python -m scripts.pitch_control --tau 0.01     # loads mlp_tau0.01.pt (both models)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import _assign_keys  # noqa: E402
from src.config import (  # noqa: E402
    DATASET_NPZ,
    MODEL_PT,
    N_FEATURES,
    PASS_MODEL_PT,
    PASS_N_FEATURES,
    PASS_OUTPUTS_DIR,
    PINBALL_TAU,
    PITCH_L,
    PITCH_W,
)
from src.ffpc_passes import load_ffpc_pass_match  # noqa: E402
from src.inference.heatmap import (  # noqa: E402
    ball_arrival_heatmap,
    heatmaps_for_frame,
    pitch_grid,
)
from src.model.mlp import TTAModel  # noqa: E402
from src.pipeline.cleaning import clean_track  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def player_model_path(tau: float) -> Path:
    from src.config import OUTPUTS_DIR
    return MODEL_PT if tau == PINBALL_TAU else OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def pass_model_path(tau: float) -> Path:
    return PASS_MODEL_PT if tau == PINBALL_TAU else PASS_OUTPUTS_DIR / f"mlp_tau{tau:g}.pt"


def _load_mlp(in_dim, path):
    m = TTAModel(in_dim=in_dim).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def _test_game_ids():
    rec = [str(x) for x in np.load(DATASET_NPZ)["recording_ids"]]
    per_match = _assign_keys(np.arange(len(rec)))
    return [rec[i] for i in range(len(rec)) if per_match[i] == 2]


def _states_for_frame(match, frame):
    """Return [(player_idx, pos, vel, acc)] for on-field, state-valid players."""
    out = []
    for seg_id, lo, hi in match.segment_slices():
        if not (lo <= frame < hi):
            continue
        li = frame - lo
        for p in np.where(match.in_field[frame] == 1)[0]:
            tr = clean_track(match.positions[lo:hi, p, :], match.in_field[lo:hi, p], p, seg_id)
            if li < tr.L and tr.state_valid[li] and not np.isnan(tr.pos[li, 0]):
                out.append((int(p), tr.pos[li], tr.vel[li], tr.acc[li]))
        break
    return out


def _candidate_passes(pm):
    """Passes with a clear carrier, a full 22-player frame and mid-range length."""
    out = []
    for pe in pm.passes:
        if pe.passer_idx < 0:
            continue
        f = pe.release_g
        if not (0 <= f < pm.ball.shape[0]) or np.isnan(pm.ball[f, 0]):
            continue
        if np.isnan(pm.ball[pe.reception_g, 0]):
            continue
        d = float(np.hypot(*(pm.ball[pe.reception_g] - pm.ball[f])))
        if not (10.0 <= d <= 30.0):
            continue
        if int((pm.match.in_field[f] == 1).sum()) < 22:
            continue
        out.append((pe, d))
    return out


def _compute(pm, pe, mlp, pass_mlp):
    """All three surfaces and the controlled region for one pass release."""
    match = pm.match
    frame = pe.release_g
    poss_team = int(match.team_id[pe.passer_idx])
    states = _states_for_frame(match, frame)
    player_idx = [p for (p, _, _, _) in states]
    if poss_team not in [int(match.team_id[p]) for p in player_idx]:
        return None
    tuples = [(pos, vel, acc) for (_, pos, vel, acc) in states]
    heat, nx, ny = heatmaps_for_frame(mlp, tuples, DEVICE)   # (n_players, nx, ny)
    teams = np.array([int(match.team_id[p]) for p in player_idx])
    if not (teams == poss_team).any() or not (teams != poss_team).any():
        return None
    t_possession = heat[teams == poss_team].min(axis=0)
    t_defense = heat[teams != poss_team].min(axis=0)

    carrier_vel = next((v for (p, _, v, _) in states if p == pe.passer_idx),
                       np.zeros(2))
    ball_pos = pm.ball[frame]
    t_ball, _nx, _ny = ball_arrival_heatmap(pass_mlp, ball_pos, carrier_vel, DEVICE)

    controlled = (t_possession < t_defense) & (t_ball < t_defense)
    classic = t_possession < t_defense
    removed = classic & ~controlled          # spatially dominated but ball can't beat the defender
    return dict(frame=frame, poss_team=poss_team, states=states, nx=nx, ny=ny,
                t_ball=t_ball, controlled=controlled, classic=classic,
                ball_pos=ball_pos, frac=float(controlled.mean()),
                classic_frac=float(classic.mean()), removed_frac=float(removed.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PINBALL_TAU,
                    help=f"pinball quantile for both the player and pass MLP "
                         f"(default {PINBALL_TAU:g})")
    args = ap.parse_args()
    tau = args.tau

    gid = _test_game_ids()[0]
    print(f"loading TEST pass match {gid} ...  tau={tau:g}", flush=True)
    pm = load_ffpc_pass_match(gid)
    match = pm.match

    cands = _candidate_passes(pm)
    if not cands:
        raise SystemExit("no suitable pass found")
    mlp = _load_mlp(N_FEATURES, player_model_path(tau))
    pass_mlp = _load_mlp(PASS_N_FEATURES, pass_model_path(tau))

    # Evaluate the rule on a bounded set of candidates and keep the most
    # illustrative one: the pass where the ball term removes the most area from
    # the classic dominant region, so the ball-aware tightening is clearly visible.
    best = None
    for pe, d in cands[:40]:
        r = _compute(pm, pe, mlp, pass_mlp)
        if r is None:
            continue
        r["dist"] = d
        if best is None or r["removed_frac"] > best["removed_frac"]:
            best = r
    if best is None:
        raise SystemExit("no computable pass found")

    frame = best["frame"]
    poss_team = best["poss_team"]
    states = best["states"]
    nx, ny = best["nx"], best["ny"]
    t_ball = best["t_ball"]
    controlled = best["controlled"]
    classic = best["classic"]
    ball_pos = best["ball_pos"]
    frac = best["frac"]
    print(f"chosen pass: frame {frame}, possession team {poss_team}, "
          f"length {best['dist']:.1f} m, controlled {frac*100:.1f}% of pitch "
          f"(classic {best['classic_frac']*100:.1f}%, ball term removes "
          f"{best['removed_frac']*100:.1f} pts)")

    # ---- render --------------------------------------------------------------
    extent = [-PITCH_L / 2, PITCH_L / 2, -PITCH_W / 2, PITCH_W / 2]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor("#2b6e3f")

    # shaded controlled region (ball-aware) -- solid orange
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(controlled.T, 1.0, np.nan), origin="lower", extent=extent,
              cmap=ListedColormap(["#ff6600"]), alpha=0.95, aspect="equal",
              vmin=0, vmax=1, interpolation="nearest")
    # classic control boundary for contrast
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    ax.contour(XX, YY, classic.astype(float), levels=[0.5], colors="white",
               linewidths=1.0, linestyles="--", alpha=0.8)
    # ball-arrival isochrones
    ax.contour(XX, YY, t_ball, levels=[1, 2, 3], colors="cyan",
               linewidths=0.7, alpha=0.6)

    for (p, pos, vel, acc) in states:
        c = "#3498db" if int(match.team_id[p]) == poss_team else "#e74c3c"
        ax.plot(pos[0], pos[1], "o", color=c, ms=9, mec="black", zorder=5)
        ax.arrow(pos[0], pos[1], vel[0] * 0.4, vel[1] * 0.4, color=c,
                 width=0.15, head_width=0.7, length_includes_head=True, zorder=4)
    ax.plot(ball_pos[0], ball_pos[1], "o", color="white", ms=11, mec="black", zorder=6)

    ax.add_patch(plt.Rectangle((-PITCH_L / 2, -PITCH_W / 2), PITCH_L, PITCH_W,
                               fill=False, color="white", lw=1.2))
    ax.axvline(0, color="white", lw=0.8)
    ax.set_xlim(extent[0] - 2, extent[1] + 2)
    ax.set_ylim(extent[2] - 2, extent[3] + 2)
    ax.set_aspect("equal")
    ax.set_title(
        f"Ball-aware pitch control (game {gid}, frame {frame})\n"
        f"orange = {{x : t_poss < t_def and t_ball < t_def}} ({frac*100:.0f}% of pitch);  "
        f"white dashed = classic t_poss < t_def;  cyan = ball isochrones\n"
        f"blue = possession (team {poss_team}), red = defense, white = ball")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "pitch_control.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
