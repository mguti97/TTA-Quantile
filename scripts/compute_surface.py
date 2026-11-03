"""Compute time-to-arrival surfaces for one frame of a tracking DataFrame.

The tracking file is forced into the FFPC (PFF FC) long format -- one row per
(frame, player) with columns period, frameNum, team, jerseyNum, x, y in metres,
centre origin, 29.97 fps -- see src/tracking_df.py. Raw PFF
``Tracking Data/<id>.jsonl.bz2`` files are accepted as-is.

    python -m scripts.compute_surface --tracking my_match.parquet --frame 12345
    python -m scripts.compute_surface --tracking my.csv --frame 900 --period 2 \\
        --rename frame=frameNum side=team shirt=jerseyNum --origin corner --plot
    python -m scripts.compute_surface --tracking "Tracking Data/3813.jsonl.bz2" --frame 5000

Writes outputs/surfaces/<stem>_p<period>_f<frame>.npz with
    tta      (n, nx, ny)  per-player minimum time to arrive at each cell (s)
    team_min (2, nx, ny)  min over each team's players (0 home, 1 away)
    xs, ys                cell-centre coordinates (m); tta[i, ix, iy] is at (xs[ix], ys[iy])
    team, jersey, player_id, player_name, pos, vel, acc   per surface row
and, with --plot, a team-min PNG (plus one per player with --plot-players).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force the FFPC config (fps, cleaning windows, 105x68 centred pitch) before any
# src import: src.config resolves the active dataset at import time.
if os.environ.get("KOGNIA_DATASET", "ffpc_match") != "ffpc_match":
    print(f"note: overriding KOGNIA_DATASET={os.environ['KOGNIA_DATASET']} -> ffpc_match",
          file=sys.stderr)
os.environ["KOGNIA_DATASET"] = "ffpc_match"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import GRID_RES, MODEL_PT, ROOT  # noqa: E402
from src.tracking_df import coerce_ffpc, compute_surfaces, load_model, read_tracking  # noqa: E402


def _parse_rename(pairs):
    out = {}
    for p in pairs or ():
        if "=" not in p:
            raise SystemExit(f"--rename expects yours=ffpc, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def plot_team_min(s, path, horizons=(1, 2, 3)):
    import matplotlib.pyplot as plt

    from src.evaluation.visualize import _draw_pitch

    # where each team arrives first: (away - home) > 0 -> home is faster
    adv = s.team_min[1] - s.team_min[0]
    extent = [s.xs[0] - (s.xs[1] - s.xs[0]) / 2, s.xs[-1] + (s.xs[1] - s.xs[0]) / 2,
              s.ys[0] - (s.ys[1] - s.ys[0]) / 2, s.ys[-1] + (s.ys[1] - s.ys[0]) / 2]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#2b6e3f")
    lim = np.nanmax(np.abs(adv)) if np.isfinite(adv).any() else 1.0
    im = ax.imshow(adv.T, origin="lower", extent=extent, cmap="coolwarm_r",
                   vmin=-min(lim, 3), vmax=min(lim, 3), alpha=0.85)
    fig.colorbar(im, ax=ax, label="away TTA - home TTA (s)  [>0: home first]")
    XX, YY = np.meshgrid(s.xs, s.ys, indexing="ij")
    for side, col in ((0, "navy"), (1, "darkred")):
        if np.isfinite(s.team_min[side]).any():
            ax.contour(XX, YY, s.team_min[side], levels=list(horizons), colors=col,
                       linewidths=0.8, alpha=0.8)
    for i in range(len(s.team)):
        c = "navy" if s.team[i] == 0 else "darkred"
        ax.plot(*s.pos[i], "o", color=c, ms=8, mec="white")
        ax.arrow(*s.pos[i], *(s.vel[i] * 0.5), color=c, width=0.15, length_includes_head=True)
        ax.annotate(s.jersey[i], s.pos[i], color="white", fontsize=7, ha="center", va="center")
    _draw_pitch(ax)
    ax.set_title(f"period {s.period} frame {s.frame_num} | contours: team-min TTA "
                 f"{', '.join(f'{h:g}' for h in horizons)} s")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracking", required=True,
                    help="csv / parquet / pkl long DataFrame, or a PFF .jsonl(.bz2)")
    ap.add_argument("--frame", type=int, required=True, help="frameNum to evaluate")
    ap.add_argument("--period", type=int, default=None,
                    help="period (only needed if frameNum repeats across periods)")
    ap.add_argument("--model", default=str(MODEL_PT),
                    help="6-feature checkpoint (default outputs/ffpc_match/mlp.pt)")
    ap.add_argument("--rename", nargs="*", metavar="YOURS=FFPC",
                    help="column renames into the FFPC names, e.g. frame=frameNum side=team")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply x, y by this to get metres (0.9144 yards, 0.01 cm)")
    ap.add_argument("--origin", choices=("center", "corner"), default="center",
                    help="corner: x/y run 0..105 / 0..68 and are shifted to the centre")
    ap.add_argument("--res", type=float, default=GRID_RES, help="grid resolution (m)")
    ap.add_argument("--suvat", action="store_true", help="also store the SUVAT baseline")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "surfaces"))
    ap.add_argument("--plot", action="store_true", help="save a team-min PNG")
    ap.add_argument("--plot-players", action="store_true", help="save one PNG per player")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = coerce_ffpc(read_tracking(args.tracking), rename=_parse_rename(args.rename),
                     scale=args.scale, origin=args.origin)
    model = load_model(args.model, device)
    s = compute_surfaces(df, args.frame, model, device, period=args.period,
                         res=args.res, with_suvat=args.suvat)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.tracking).name.split(".")[0]
    base = out_dir / f"{stem}_p{s.period}_f{s.frame_num}"
    arrays = dict(tta=s.tta, team_min=s.team_min, xs=s.xs, ys=s.ys, team=s.team,
                  jersey=np.asarray(s.jersey), player_id=s.player_id,
                  player_name=np.asarray(s.player_name), pos=s.pos, vel=s.vel, acc=s.acc,
                  period=s.period, frame_num=s.frame_num, model=str(args.model))
    if s.suvat is not None:
        arrays["suvat"] = s.suvat
    np.savez_compressed(f"{base}.npz", **arrays)

    print(f"period {s.period} frame {s.frame_num}: {len(s.team)} surfaces "
          f"({(s.team == 0).sum()} home, {(s.team == 1).sum()} away), grid "
          f"{len(s.xs)}x{len(s.ys)} @ {args.res:g} m -> {base}.npz")
    if s.skipped:
        print(f"  no valid state (short run / failed clipping): {', '.join(s.skipped)}")
    if args.plot:
        plot_team_min(s, f"{base}_teams.png")
        print(f"  plot -> {base}_teams.png")
    if args.plot_players:
        from src.evaluation.visualize import render_player_heatmap

        for i in range(len(s.team)):
            side = "home" if s.team[i] == 0 else "away"
            path = f"{base}_{side}{s.jersey[i]}.png"
            render_player_heatmap(
                s.tta[i], s.pos[i], model_suvat=None if s.suvat is None else s.suvat[i],
                title=f"{s.player_name[i]} | p{s.period} f{s.frame_num} | "
                      f"v={np.hypot(*s.vel[i]):.1f} m/s", path=path)
        print(f"  {len(s.team)} player plots -> {base}_<side><jersey>.png")


if __name__ == "__main__":
    main()
