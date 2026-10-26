"""Evaluation metrics: MAE + stratified breakdown vs the SUVAT baseline (Section 10)."""
from __future__ import annotations

import numpy as np

# Stratification bands (Section 10)
DIST_BANDS = [(0, 5), (5, 15), (15, 30), (30, np.inf)]
SPEED_BANDS = [(0, 2), (2, 5), (5, 8), (8, np.inf)]
ANGLE_BANDS = [(0, 45), (45, 90), (90, 135), (135, 180)]
HORIZON_BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8)]


def mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def rmse(pred, target):
    """Root-mean-square arrival-time error (seconds). Tail-sensitive sibling of
    MAE -- the metric Martens & Brefeld (2021) used for distance->time regression."""
    e = np.asarray(pred) - np.asarray(target)
    return float(np.sqrt(np.mean(e ** 2)))


def coverage(pred, target):
    """Empirical coverage of the lower (fast-arrival) quantile: P(actual >= pred).

    For a tau-quantile model of the *minimum* arrival time, ``pred`` is an
    optimistic time, so the player's real arrival ``target`` should satisfy
    ``pred <= target`` a fraction ``1 - tau`` of the time. This is the direct
    calibration check the quantile literature converges on; pinball penalises
    miscalibration but never reports it."""
    return float(np.mean(np.asarray(pred) <= np.asarray(target)))


def _band_label(bands, unit):
    out = []
    for lo, hi in bands:
        out.append(f"{lo}-{'inf' if np.isinf(hi) else int(hi)}{unit}")
    return out


def stratified(pred_model, pred_suvat, y, d, v, theta_deg):
    """Return a list of rows: (dimension, band, n, mae_model, mae_suvat)."""
    pred_model = np.asarray(pred_model)
    pred_suvat = np.asarray(pred_suvat)
    y = np.asarray(y)
    abs_theta = np.abs(theta_deg)

    rows = []
    for dim, vals, bands, unit in (
        ("distance", d, DIST_BANDS, "m"),
        ("speed", v, SPEED_BANDS, "m/s"),
        ("angle", abs_theta, ANGLE_BANDS, "deg"),
    ):
        for (lo, hi), lbl in zip(bands, _band_label(bands, unit)):
            m = (vals >= lo) & (vals < hi)
            n = int(m.sum())
            if n == 0:
                rows.append((dim, lbl, 0, float("nan"), float("nan")))
                continue
            rows.append((dim, lbl, n, mae(pred_model[m], y[m]), mae(pred_suvat[m], y[m])))
    return rows


def calibration(pred, y, d, v, theta_deg, horizon=None):
    """Per-band empirical quantile coverage P(actual >= pred).

    Same stratification as ``stratified`` (distance / speed / angle), plus two
    time-scale views that together expose the conditioning-on-outcome artifact:

      * ``true_t`` -- binned by the *observed* arrival time ``y``. Conditioning
        on the outcome makes coverage ramp 0->1 even for a perfect model, so a
        drift here is NOT evidence of miscalibration.
      * ``pred_t`` -- binned by the *predicted* time (a function of the inputs).
        This is a coarse feature-conditioning, so a well-calibrated model stays
        flat near 1 - tau here. Contrast with ``true_t``.

    Needs only ``pred`` and ``y`` -- no grid -- so it is cheap on the full test
    set. Returns rows (dim, band, n, coverage).
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    dims = [
        ("distance", np.asarray(d), DIST_BANDS, "m"),
        ("speed", np.asarray(v), SPEED_BANDS, "m/s"),
        ("angle", np.abs(theta_deg), ANGLE_BANDS, "deg"),
    ]
    if horizon is not None:
        dims.append(("true_t", np.asarray(horizon), HORIZON_BANDS, "s"))
    dims.append(("pred_t", pred, HORIZON_BANDS, "s"))

    rows = []
    for dim, vals, bands, unit in dims:
        for (lo, hi), lbl in zip(bands, _band_label(bands, unit)):
            m = (vals >= lo) & (vals < hi)
            n = int(m.sum())
            rows.append((dim, lbl, n, coverage(pred[m], y[m]) if n else float("nan")))
    return rows


def format_calibration(rows, nominal: float) -> str:
    """Render a calibration table; ``nominal`` is the target coverage (1 - tau)."""
    lines = [f"target coverage (1-tau) = {nominal:.3f}"]
    lines.append(f"{'dim':9s} {'band':12s} {'n':>10s} {'coverage':>9s} {'gap':>8s}")
    lines.append("-" * 52)
    for dim, lbl, n, cov in rows:
        gap = cov - nominal if not np.isnan(cov) else float("nan")
        lines.append(f"{dim:9s} {lbl:12s} {n:>10,d} {cov:>9.4f} {gap:>+8.4f}")
    return "\n".join(lines)


def format_table(rows) -> str:
    lines = [f"{'dim':9s} {'band':12s} {'n':>10s} {'MAE_model':>10s} {'MAE_suvat':>10s}  {'Δ':>7s}"]
    lines.append("-" * 64)
    for dim, lbl, n, mm, ms in rows:
        delta = ms - mm if not (np.isnan(mm) or np.isnan(ms)) else float("nan")
        lines.append(f"{dim:9s} {lbl:12s} {n:>10,d} {mm:>10.3f} {ms:>10.3f}  {delta:>+7.3f}")
    return "\n".join(lines)
