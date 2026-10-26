"""Heatmap rendering and model-vs-SUVAT contour comparison (Section 10, qualitative)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import PITCH_L, PITCH_W, PITCH_X0, PITCH_Y0  # noqa: E402


def _draw_pitch(ax):
    ax.add_patch(plt.Rectangle((PITCH_X0, PITCH_Y0), PITCH_L, PITCH_W,
                               fill=False, color="white", lw=1.2))
    ax.axvline(PITCH_X0 + PITCH_L / 2, color="white", lw=0.8)
    ax.set_xlim(PITCH_X0 - 2, PITCH_X0 + PITCH_L + 2)
    ax.set_ylim(PITCH_Y0 - 2, PITCH_Y0 + PITCH_W + 2)
    ax.set_aspect("equal")


def render_player_heatmap(heat, player_pos, model_suvat=None, title="", path=None):
    """Render one player's TTA heatmap; optionally overlay model vs SUVAT contours.

    `heat` is (nx, ny) in the pitch grid orientation (x along axis-0).
    """
    extent = [PITCH_X0, PITCH_X0 + PITCH_L, PITCH_Y0, PITCH_Y0 + PITCH_W]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#2b6e3f")
    im = ax.imshow(heat.T, origin="lower", extent=extent, cmap="viridis",
                   aspect="equal", alpha=0.9)
    fig.colorbar(im, ax=ax, label="time to arrive (s)")

    levels = [1, 2, 3, 4, 5]
    xs = np.linspace(extent[0], extent[1], heat.shape[0])
    ys = np.linspace(extent[2], extent[3], heat.shape[1])
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    ax.contour(XX, YY, heat, levels=levels, colors="white", linewidths=0.8, alpha=0.7)
    if model_suvat is not None:
        ax.contour(XX, YY, model_suvat, levels=levels, colors="red",
                   linewidths=0.8, linestyles="--", alpha=0.7)

    ax.plot(*player_pos, "o", color="yellow", ms=10, mec="black")
    _draw_pitch(ax)
    ax.set_title(title)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig


def plot_training_curve(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, marker=".")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val pinball loss")
    ax.set_title("Validation pinball loss")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
