"""Theoretical player motion models (arrival-time form).

Implements the player motion models from two papers, expressed as
*time-to-arrival* predictors so they drop straight into our evaluation stack
(``features.transform`` dict -> (N,) arrival times, the same ``pred_fn``
contract used by ``evaluation.reachable_area`` and the compare scripts):

  Renkin, Bischofberger, Schikuta & Baca, "Validation and Optimisation of
  Player Motion Models in Football", ICCS 2022 (paper 381). Four models, each
  defining a reachable area R(Δt); we invert each to the minimum arrival time
  τ(target) = min{ t : target ∈ R(t) }:
    (a) constant speed                         params: v_max
    (b) constant acceleration                  params: a_max
    (c) constant acceleration + speed limit    params: a_max, v_max
    (d) two-segment constant speed             params: t_inert, keep_initial,
                                                       v_const, a_max, v_max

  Wu & Swartz, "A new metric for pitch control based on an intuitive motion
  model", Computational Statistics 2025 (s00180-024-01512-2). One player model:
    (e) constant non-directional acceleration of fixed magnitude ``a`` (its
        direction is chosen freely on the circle a_x²+a_y²=a²) until the speed
        cap ``s_max`` is reached, then constant velocity. Arrival time is the
        minimum over acceleration directions, found exactly as the paper does
        (eqs. 5-11): grid the acceleration angle, solve for the time the
        trajectory reaches the target.

  Anzer & Bauer, "Expected passes" (Data Min. Knowl. Discov. 2022), §3.2. The
  one *learned* (non-mechanistic) baseline: the reachable set at time t is a
  circle whose centre offset (along heading) and radius are empirical
  functions of the player's current speed, tabulated from training tracking
  data in fine speed bins (their 0.5 km/h) and inverted here to arrival times
  through the same reachability bisection as the mechanistic models. Where the
  paper takes convex hulls of everything observed, we set the radius to the
  ``radius_q`` quantile of the observed displacements so the model sits at the
  benchmark's correctness gate (the hull is the q -> 1 limit).

All math is done in the *heading frame* supplied by ``features.transform``:
the player sits at the origin with velocity ``(v, 0)`` and the target sits at
``(d·cos_t, d·sin_t)``. The transform dict therefore gives everything the
models need (``d``, ``v``, ``cos_t``, ``sin_t``); no positions are required.

Each model is a small dataclass carrying its fitted parameters with a
``predict(tf) -> (N,) seconds`` method. ``MODEL_SPECS`` describes the free
parameters (bounds + discrete choices) so the fitting script can optimise them
generically, and ``from_params`` rebuilds a model from a saved param dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pipeline.suvat import suvat_time

# Arrival times are capped here when a model genuinely cannot reach a target
# (e.g. model (c) sprinting at v_max away from a point behind it): the reachable
# set never contains it, so τ = ∞. We clip to a large finite value so the metric
# treats it as "miss" (pred > any horizon) without poisoning means with inf.
T_CAP = 60.0
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# vectorised minimum-arrival-time solver via reachability bracketing
# --------------------------------------------------------------------------- #
def _min_time_bisect(reach, t_hi, n_scan: int = 48, n_bisect: int = 44):
    """Smallest t in (0, t_hi] with ``reach(t)`` True, per sample (vectorised).

    ``reach`` maps a (N,) time array to a (N,) boolean array. ``t_hi`` is a
    per-sample upper bound at which the target is guaranteed reachable (or the
    cap). A coarse forward scan first brackets the *first* up-crossing (so the
    result is the genuine minimum even if reachability is not perfectly
    monotone in t), then bisection refines it.
    """
    t_hi = np.asarray(t_hi, dtype=np.float64)
    N = t_hi.shape[0]
    lo = np.zeros(N)
    hi = t_hi.copy()
    found = np.zeros(N, dtype=bool)
    prev = np.zeros(N)  # last scanned time still NOT reachable

    for f in np.linspace(0.0, 1.0, n_scan + 1)[1:]:
        t = f * t_hi
        r = reach(t)
        newly = r & ~found
        hi = np.where(newly, t, hi)
        lo = np.where(newly, prev, lo)
        found = found | r
        prev = np.where(found, prev, t)

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        r = reach(mid)
        hi = np.where(r, mid, hi)
        lo = np.where(r, lo, mid)
    return hi


def _targets(tf):
    """Pull (d, v, target_x, target_y) in the heading frame from a transform dict."""
    d = np.asarray(tf["d"], dtype=np.float64)
    v = np.asarray(tf["v"], dtype=np.float64)
    tx = d * np.asarray(tf["cos_t"], dtype=np.float64)
    ty = d * np.asarray(tf["sin_t"], dtype=np.float64)
    return d, v, tx, ty


def _ws_axis_time(v, a, smax, tx):
    """Wu & Swartz arrival time for a target on the heading axis (1-D motion).

    The player starts at speed ``v`` along +x; ``tx`` may be ahead (>=0) or
    behind (<0). Accelerate at magnitude ``a`` toward the target, capped at
    ``smax``; for a target behind, decelerate/reverse. Closed form per phase.
    """
    v = np.asarray(v, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)

    # forward target: accelerate +a until s_max, then cruise
    tstar = (smax - v) / a
    disc_f = v * v + 2.0 * a * np.maximum(tx, 0.0)
    t_f1 = (-v + np.sqrt(np.maximum(disc_f, 0.0))) / a       # reach during accel
    x_star = v * tstar + 0.5 * a * tstar * tstar
    t_f2 = tstar + (tx - x_star) / smax                       # reach during cruise
    t_fwd = np.where(t_f1 <= tstar, t_f1, t_f2)

    # backward target: decelerate then reverse at -a until -s_max, then cruise
    tdd = (v + smax) / a
    xdd = v * tdd - 0.5 * a * tdd * tdd
    disc_b = v * v - 2.0 * a * tx                             # tx<0 -> positive
    t_b1 = (v + np.sqrt(np.maximum(disc_b, 0.0))) / a
    t_b2 = tdd + (xdd - tx) / smax
    t_bwd = np.where(t_b1 <= tdd, t_b1, t_b2)

    out = np.where(tx >= 0, t_fwd, t_bwd)
    return np.where(np.isfinite(out) & (out > 0), out, T_CAP)


# --------------------------------------------------------------------------- #
# (a) constant speed  -- Renkin eq. (7)
# --------------------------------------------------------------------------- #
@dataclass
class ConstSpeed:
    v_max: float = 8.0
    name: str = "ConstSpeed"

    def predict(self, tf) -> np.ndarray:
        d, _, _, _ = _targets(tf)
        return d / np.maximum(self.v_max, _EPS)


# --------------------------------------------------------------------------- #
# (b) constant acceleration  -- Renkin eq. (8)
# reachable disk: centre x0 + v0 t, radius ½ a_max t²
# --------------------------------------------------------------------------- #
@dataclass
class ConstAccel:
    a_max: float = 9.0
    name: str = "ConstAccel"

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        a = np.maximum(self.a_max, _EPS)

        def reach(t):
            cx = v * t           # centre drifts along +x at speed v
            r = 0.5 * a * t * t  # radius
            return np.hypot(tx - cx, ty) <= r

        # dist_from_centre(t) <= d + v t, so ½a t_hi² = v t_hi + d => reachable
        t_hi = (v + np.sqrt(v * v + 2.0 * a * d)) / a
        return _min_time_bisect(reach, t_hi)


# --------------------------------------------------------------------------- #
# (c) constant acceleration with speed limit  -- Renkin eq. (9)
# two phases: accelerate at a_max until v_max (t_acc = v_max/a_max), then cruise.
# centre drifts at the clipped initial speed; radius = a_max·t_acc·(t-½t_acc).
# --------------------------------------------------------------------------- #
@dataclass
class ConstAccelVmax:
    a_max: float = 9.0
    v_max: float = 7.0
    name: str = "ConstAccelVmax"

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        a = np.maximum(self.a_max, _EPS)
        vmax = np.maximum(self.v_max, _EPS)
        v_clip = np.minimum(v, vmax)      # v0 clipped to v_max (Renkin §4.2c)
        t_acc = vmax / a                  # time to reach v_max from rest

        def radius(t):
            ta = np.minimum(t_acc, t)
            return a * ta * (t - 0.5 * ta)

        def reach(t):
            cx = v_clip * t
            return np.hypot(tx - cx, ty) <= radius(t)

        # radius grows ~v_max·t; centre drifts ~v_clip·t. If v_max > v_clip the
        # gap closes and the target is reachable in finite time; otherwise (the
        # player already sprints at v_max) some rear targets are unreachable.
        gap = vmax - v_clip
        t_lin = np.where(gap > _EPS, (d + 0.5 * vmax * t_acc) / np.maximum(gap, _EPS), T_CAP)
        t_hi = np.minimum(np.maximum(t_lin, t_acc + 0.1), T_CAP)
        return _min_time_bisect(reach, t_hi)


# --------------------------------------------------------------------------- #
# (d) two-segment constant speed  -- Renkin eq. (10)
# segment 1: drift along v0 for t_inert at speed v_inert;
# segment 2: straight to target at speed v_final. Closed form.
# --------------------------------------------------------------------------- #
@dataclass
class TwoSegment:
    t_inert: float = 0.22
    keep_initial: bool = True
    v_const: float = 7.0
    a_max: float = 9.0
    v_max: float = 7.0
    name: str = "TwoSegment"

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        v_inert = v if self.keep_initial else np.full_like(v, self.v_const)
        # v_final from a pretend-acceleration over the inert segment, capped.
        v_final = np.minimum(v_inert + self.a_max * self.t_inert, self.v_max)
        v_final = np.maximum(v_final, _EPS)
        # drift along heading +x; undefined direction when stationary -> no drift
        drift_x = np.where(v > _EPS, v_inert * self.t_inert, 0.0)
        dist_inert = np.hypot(tx - drift_x, ty)
        return self.t_inert + dist_inert / v_final


# --------------------------------------------------------------------------- #
# (e) Wu & Swartz: constant non-directional acceleration + speed cap (eqs 5-11)
# acceleration vector has fixed magnitude ``a`` and a freely chosen direction;
# the player accelerates until speed s_max then holds velocity. Arrival time =
# minimum over directions of the time the trajectory reaches the target.
# --------------------------------------------------------------------------- #
@dataclass
class WuSwartz:
    a: float = 5.0
    s_max: float = 9.2
    n_dirs: int = 240
    name: str = "WuSwartz"

    def predict(self, tf) -> np.ndarray:
        d, v, tx1, ty1 = _targets(tf)            # 1-D, length N
        N = d.shape[0]
        # ``a``/``s_max`` may be scalars (global fit) or per-sample (N,) arrays
        # (per-player fit). The direction-grid math broadcasts (N,1) x (1,K), so
        # array params enter it as COLUMNS; the 1-D helpers below take them flat.
        a_flat = np.maximum(np.asarray(self.a, dtype=np.float64), _EPS)
        smax_flat = np.maximum(np.asarray(self.s_max, dtype=np.float64), _EPS)
        a = a_flat[:, None] if a_flat.ndim else a_flat
        smax = smax_flat[:, None] if smax_flat.ndim else smax_flat
        v1 = np.minimum(v, smax_flat)                 # model assumes initial speed <= s_max

        # grid of acceleration directions on the circle a_x²+a_y²=a²
        psi = np.linspace(0.0, 2.0 * np.pi, self.n_dirs, endpoint=False)
        ax = a * np.cos(psi)[None, :]            # (1, K)
        ay = a * np.sin(psi)[None, :]
        vv = v1[:, None]                          # (N, 1)
        tx = tx1[:, None]
        ty = ty1[:, None]

        # t* : time the speed cap s_max is reached for this (ax,ay)  -- eq. (6)
        # a² t*² + 2 v ax t* + (v² - s_max²) = 0
        disc = np.maximum(a * a * smax * smax - (vv * ay) ** 2, 0.0)
        tstar = np.maximum((-vv * ax + np.sqrt(disc)) / (a * a), 0.0)

        # velocity & position at t* (end of acceleration phase)
        vy_star = ay * tstar
        y_star = 0.5 * ay * tstar * tstar
        x_star = vv * tstar + 0.5 * ax * tstar * tstar
        vx_star = vv + ax * tstar

        # --- solve y(t) = ty along each trajectory --------------------------
        # phase 1 (t<=t*): y = ½ ay t²   ;  phase 2 (t>t*): y = y* + (t-t*) vy*
        with np.errstate(invalid="ignore", divide="ignore"):
            t1 = np.sqrt(np.where(ay != 0, 2.0 * ty / ay, np.nan))   # phase-1 root
            t1_ok = np.isfinite(t1) & (t1 <= tstar + 1e-9)
            t2 = tstar + (ty - y_star) / np.where(vy_star != 0, vy_star, np.nan)
            t2_ok = np.isfinite(t2) & (t2 >= tstar - 1e-9)
        t_y = np.where(t1_ok, t1, np.where(t2_ok, t2, np.nan))

        # --- x position at t_y; the residual we drive to zero ---------------
        def x_at(t):
            x_pre = vv * t + 0.5 * ax * t * t
            x_post = x_star + (t - tstar) * vx_star
            return np.where(t <= tstar, x_pre, x_post)

        miss = np.where(np.isfinite(t_y), x_at(t_y) - tx, np.nan)  # (N, K)

        # wrap the grid (ψ is periodic) so a hit near the 0/2π seam is caught
        miss = np.concatenate([miss, miss[:, :1]], axis=1)
        t_yw = np.concatenate([t_y, t_y[:, :1]], axis=1)

        # arrival = smallest t_y over directions where ``miss`` changes sign
        arrival = np.full(N, T_CAP)
        s = np.sign(miss)
        for k in range(self.n_dirs):
            # NB: `a1` here is a MISS column, not an acceleration (the capacity
            # locals above are deliberately named a_flat/smax_flat to avoid it).
            a0, a1 = miss[:, k], miss[:, k + 1]
            cross = np.isfinite(a0) & np.isfinite(a1) & (s[:, k] * s[:, k + 1] < 0)
            if not cross.any():
                continue
            w = a0 / (a0 - a1)                    # linear interp of the crossing
            t_cross = t_yw[:, k] + w * (t_yw[:, k + 1] - t_yw[:, k])
            good = cross & np.isfinite(t_cross) & (t_cross > 0)
            arrival = np.where(good, np.minimum(arrival, t_cross), arrival)

        # On-axis targets (ty≈0) are degenerate for the grid (ay≈0 there); solve
        # the 1-D motion along the heading axis exactly and take the better time.
        on_axis = np.abs(ty1) < 0.05
        if on_axis.any():
            t_ax = _ws_axis_time(v1, a_flat, smax_flat, tx1)
            arrival = np.where(on_axis, np.minimum(arrival, t_ax), arrival)

        # Any target the angle grid still failed to bracket (a thin slice of
        # off-axis cases) falls back to the same two-phase kinematics solved
        # radially: accelerate at ``a`` toward the target, capped at ``s_max``
        # (== SUVAT with this model's parameters). Keeps arrivals finite.
        failed = arrival >= T_CAP - 1.0
        if failed.any():
            fb = suvat_time(v1, np.asarray(tf["cos_t"], dtype=np.float64), d,
                            a_max=a_flat, v_max=smax_flat)
            arrival = np.where(failed, fb, arrival)

        return np.minimum(arrival, T_CAP)


# --------------------------------------------------------------------------- #
# (f) Fujimura & Sugihara: constant acceleration damped by a resistive force
# proportional to speed, capping it at v_max (Brefeld et al. 2019, Mach. Learn.
# 108:127, eq. 2; from the s10994-018-5725-1 review). The reachable set at time
# t is a circle whose centre drifts (and saturates) with the initial velocity
# and whose radius grows under the friction-limited speed.
#
# Note: the Taki & Hasegawa (2000) model from the same review (eq. 1: centre
# v0·t, radius ½·a_max·t²) is mathematically identical to ``ConstAccel`` above
# (Renkin model b) -- so it is already implemented and not duplicated here.
# --------------------------------------------------------------------------- #
@dataclass
class FujimuraSugihara:
    alpha: float = 1.3      # resistive-force coefficient (Fujimura & Sugihara 2005)
    v_max: float = 8.0
    name: str = "FujimuraSugihara"

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        a = np.maximum(self.alpha, _EPS)
        vmax = np.maximum(self.v_max, _EPS)

        def reach(t):
            g = (1.0 - np.exp(-a * t)) / a   # (1-e^{-αt})/α : saturates at 1/α
            cx = v * g                        # centre drift (bounded by v/α)
            r = vmax * (t - g)                # radius (grows ~v_max·t for large t)
            return np.hypot(tx - cx, ty) <= r

        # radius >= v_max·(t - 1/α); centre drift <= v/α, so dist <= d + v/α.
        # t_hi makes radius >= d + v/α -> always reachable in finite time.
        t_hi = 1.0 / a + (d + v / a) / vmax + 0.1
        return _min_time_bisect(reach, t_hi)


# --------------------------------------------------------------------------- #
# (g) Anzer & Bauer: learned speed-binned circles (Expected passes, §3.2)
# The reachable set at time t for a player at speed v is a circle with centre
# c(v,t) ahead along the heading and radius r(v,t); both are *empirical* lookup
# tables fitted from training tracking (scripts.fit_motion_models.fit_anzer).
# Rows are speed bins, columns a uniform time grid starting at t=0 (c=r=0).
# Lookups are bilinear; past the last column both tables continue linearly, so
# arrival times stay well-defined at any horizon.
# --------------------------------------------------------------------------- #
@dataclass
class AnzerBauer:
    v_centers: list          # (nv,) speed-bin centres, m/s, increasing
    t_grid: list             # (nt,) uniform time grid, s; t_grid[0] == 0
    center: list             # (nv, nt) centre offset along heading, m
    radius: list             # (nv, nt) circle radius, m
    radius_q: float = 0.99
    name: str = "AnzerBauer"

    def __post_init__(self):
        self._vc = np.asarray(self.v_centers, dtype=np.float64)
        self._tg = np.asarray(self.t_grid, dtype=np.float64)
        self._C = np.asarray(self.center, dtype=np.float64)
        self._R = np.asarray(self.radius, dtype=np.float64)

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        vc, tg = self._vc, self._tg
        nv, nt = self._C.shape
        dt = tg[1] - tg[0]

        # blend the two speed rows bracketing each sample's speed (constant in
        # t, so done once); speeds outside the table clamp to the edge rows
        iv = np.clip(np.searchsorted(vc, v) - 1, 0, nv - 2)
        fv = np.clip((v - vc[iv]) / (vc[iv + 1] - vc[iv]), 0.0, 1.0)
        C = (1.0 - fv)[:, None] * self._C[iv] + fv[:, None] * self._C[iv + 1]
        R = (1.0 - fv)[:, None] * self._R[iv] + fv[:, None] * self._R[iv + 1]
        rows = np.arange(d.shape[0])

        def interp(tab, t):
            # uniform grid -> direct index; it==nt-2 with ft>1 extrapolates
            it = np.clip(((t - tg[0]) / dt).astype(np.int64), 0, nt - 2)
            ft = (t - tg[it]) / dt
            return tab[rows, it] * (1.0 - ft) + tab[rows, it + 1] * ft

        def reach(t):
            return np.hypot(tx - interp(C, t), ty) <= interp(R, t)

        # |target - centre| <= d + c(t), so r(t) - c(t) >= d guarantees reach;
        # past the grid the gap grows at the tables' end slope (floored so a
        # noisy flat tail can't stall the bound).
        gap_end = R[:, -1] - C[:, -1]
        slope = np.maximum((R[:, -1] - R[:, -2]) - (C[:, -1] - C[:, -2]), 0.2 * dt) / dt
        t_hi = np.where(gap_end >= d, tg[-1], tg[-1] + (d - gap_end) / slope + 1e-6)
        return _min_time_bisect(reach, np.minimum(t_hi, T_CAP))


# --------------------------------------------------------------------------- #
# (g') Anzer & Bauer, FAITHFUL: learned speed-binned CONVEX HULLS (§3.2)
#
# The paper's own construction: for every (speed bin, time offset) cell the
# reachable region is the convex hull of ALL observed heading-frame
# displacements of that cell -- the pure empirical envelope, with no quantile,
# no circle and no radius scaling. The hull IS the model; the global gate-scale
# calibration of ``AnzerBauer`` above is *our* adaptation, not the paper.
#
# Storage: each hull is kept as a RADIAL polygon r(θ_k), K evenly spaced angles
# around the cell's mean displacement (a convex hull is star-shaped about any
# interior point, so this is exact up to angular sampling). Interpolation is
# then bilinear in (speed, time) and linear in angle, mirroring the circle
# model's bilinear lookups. Past the last time column r continues linearly.
# --------------------------------------------------------------------------- #
@dataclass
class AnzerBauerHull:
    v_centers: list          # (nv,) speed-bin centres, m/s, increasing
    t_grid: list             # (nt,) uniform time grid, s; t_grid[0] == 0
    center_x: list           # (nv, nt) hull anchor, along heading, m
    center_y: list           # (nv, nt) hull anchor, perpendicular, m
    radius: list             # (nv, nt, K) hull boundary distance per angle, m
    scale: float = 1.0       # eval-time envelope multiplier (faithful model: 1)
    name: str = "AnzerBauerHull"

    def __post_init__(self):
        self._vc = np.asarray(self.v_centers, dtype=np.float64)
        self._tg = np.asarray(self.t_grid, dtype=np.float64)
        self._CX = np.asarray(self.center_x, dtype=np.float64)
        self._CY = np.asarray(self.center_y, dtype=np.float64)
        self._R = np.asarray(self.radius, dtype=np.float64) * float(self.scale)
        self._K = self._R.shape[2]
        self._dth = 2.0 * np.pi / self._K
        self._Rmin = self._R.min(axis=2)      # (nv, nt), for the t_hi bound

    def predict(self, tf) -> np.ndarray:
        d, v, tx, ty = _targets(tf)
        vc, tg = self._vc, self._tg
        nv, nt = self._CX.shape
        dt = tg[1] - tg[0]
        K, dth = self._K, self._dth

        # speed-row blend (constant in t); speeds outside the table clamp
        iv = np.clip(np.searchsorted(vc, v) - 1, 0, nv - 2)
        fv = np.clip((v - vc[iv]) / (vc[iv + 1] - vc[iv]), 0.0, 1.0)
        iv1 = iv + 1

        def blend_t(tab, t):
            """Bilinear (speed, time) lookup of an (nv, nt) table."""
            it = np.clip(((t - tg[0]) / dt).astype(np.int64), 0, nt - 2)
            ft = (t - tg[it]) / dt
            a = tab[iv, it] * (1.0 - ft) + tab[iv, it + 1] * ft
            b = tab[iv1, it] * (1.0 - ft) + tab[iv1, it + 1] * ft
            return (1.0 - fv) * a + fv * b

        def reach(t):
            cx = blend_t(self._CX, t)
            cy = blend_t(self._CY, t)
            ex, ey = tx - cx, ty - cy
            ang = np.mod(np.arctan2(ey, ex), 2.0 * np.pi)
            ak = ang / dth
            ik = ak.astype(np.int64) % K
            fk = ak - np.floor(ak)
            ik1 = (ik + 1) % K
            it = np.clip(((t - tg[0]) / dt).astype(np.int64), 0, nt - 2)
            ft = (t - tg[it]) / dt
            R = self._R

            def cell(i, j):     # angular interpolation at one (speed, time) node
                return R[i, j, ik] * (1.0 - fk) + R[i, j, ik1] * fk

            r0 = cell(iv, it) * (1.0 - ft) + cell(iv, it + 1) * ft
            r1 = cell(iv1, it) * (1.0 - ft) + cell(iv1, it + 1) * ft
            r = (1.0 - fv) * r0 + fv * r1
            return np.hypot(ex, ey) <= r

        # Guaranteed-reach bound: with the anchor at distance |c| from the
        # origin and the envelope at least r_min(t) in every direction, any
        # target with d <= r_min(t) - |c(t)| is inside. Blending the per-row
        # angular minima under-estimates the true min (min of a blend >= blend
        # of the minima), so the bound stays conservative. Past the grid both
        # continue at the tables' end slope, floored so a flat noisy tail can't
        # stall the bound (exactly the circle model's t_hi logic).
        def blend_v(col):
            return (1.0 - fv) * col[iv] + fv * col[iv1]

        rmin_end, rmin_prev = blend_v(self._Rmin[:, -1]), blend_v(self._Rmin[:, -2])
        c_end = np.hypot(blend_v(self._CX[:, -1]), blend_v(self._CY[:, -1]))
        c_prev = np.hypot(blend_v(self._CX[:, -2]), blend_v(self._CY[:, -2]))
        gap_end = rmin_end - c_end
        slope = np.maximum((rmin_end - rmin_prev) - (c_end - c_prev), 0.2 * dt) / dt
        t_hi = np.where(gap_end >= d, tg[-1], tg[-1] + (d - gap_end) / slope + 1e-6)
        return _min_time_bisect(reach, np.minimum(t_hi, T_CAP))


# --------------------------------------------------------------------------- #
# PER-PLAYER capacity variants (ACCV rebuttal)
#
# The reviewers asked for physics baselines that get the SAME per-player
# information as the "profile" MLP (which sees a per-player v_max feature). Each
# capacity parameter becomes
#
#       param_i = c * table_value_i
#
# with ONE global scale ``c`` fitted by the same DE protocol that fitted the
# global parameter it replaces; every effective/shape parameter (t_inert,
# keep_initial, v_const, alpha, ...) stays globally fitted exactly as before.
# ``table_value_i`` is the leakage-free per-player 99th-percentile speed /
# acceleration from the TRAIN-split-only capacity table (data/<ds>/processed/
# vmax_table_vmaxtrain.npz), passed per sample through the transform dict as
# ``tf["vmax_pp"]`` / ``tf["amax_pp"]`` (missing players carry the table's
# stored median fallback, injected by the caller).
#
# Two variants:
#   P1 -- per-player v_max only (c_v); acceleration params stay global.
#   P2 -- per-player v_max and a_max (c_v, c_a).
# A model whose P2 form has no acceleration parameter (ConstSpeed,
# FujimuraSugihara) is identical to its P1 form and is fitted once per variant.
#
# Each class simply builds its canonical counterpart with array-valued
# parameters, so the physics is bit-for-bit the same model.
# --------------------------------------------------------------------------- #
def _pp(tf, key: str) -> np.ndarray:
    """Per-sample capacity column from the transform dict (raises if absent)."""
    if key not in tf:
        raise KeyError(
            f"per-player motion model needs tf[{key!r}]; the caller must inject "
            "the per-trail capacity column (see scripts.fit_motion_models."
            "trail_capacities / reachable_area's `extra` argument)")
    return np.asarray(tf[key], dtype=np.float64)


@dataclass
class ConstSpeedPP:
    """Renkin (a) with v_max = c_v * table v_max_i."""
    c_v: float = 1.0
    name: str = "ConstSpeedPP"

    def predict(self, tf) -> np.ndarray:
        return ConstSpeed(v_max=self.c_v * _pp(tf, "vmax_pp")).predict(tf)


@dataclass
class ConstAccelPP:
    """Renkin (b) with a_max = c_a * table a_max_i (variant P2 only)."""
    c_a: float = 1.0
    name: str = "ConstAccelPP"

    def predict(self, tf) -> np.ndarray:
        return ConstAccel(a_max=self.c_a * _pp(tf, "amax_pp")).predict(tf)


@dataclass
class ConstAccelVmaxPP:
    """Renkin (c). v_max per player; a_max per player (c_a) or global (a_max)."""
    c_v: float = 1.0
    a_max: float | None = 9.0     # global accel (P1); None -> use c_a
    c_a: float | None = None      # per-player accel scale (P2)
    name: str = "ConstAccelVmaxPP"

    def predict(self, tf) -> np.ndarray:
        a = self.c_a * _pp(tf, "amax_pp") if self.c_a is not None else self.a_max
        return ConstAccelVmax(a_max=a, v_max=self.c_v * _pp(tf, "vmax_pp")).predict(tf)


@dataclass
class TwoSegmentPP:
    """Renkin (d). v_max per player; a_max per player (c_a) or global (a_max)."""
    t_inert: float = 0.22
    keep_initial: bool = True
    v_const: float = 7.0
    c_v: float = 1.0
    a_max: float | None = 9.0
    c_a: float | None = None
    name: str = "TwoSegmentPP"

    def predict(self, tf) -> np.ndarray:
        a = self.c_a * _pp(tf, "amax_pp") if self.c_a is not None else self.a_max
        return TwoSegment(t_inert=self.t_inert, keep_initial=self.keep_initial,
                          v_const=self.v_const, a_max=a,
                          v_max=self.c_v * _pp(tf, "vmax_pp")).predict(tf)


@dataclass
class WuSwartzPP:
    """Wu & Swartz (e). s_max per player; a per player (c_a) or global (a)."""
    c_v: float = 1.0
    a: float | None = 5.0
    c_a: float | None = None
    n_dirs: int = 240
    name: str = "WuSwartzPP"

    def predict(self, tf) -> np.ndarray:
        acc = self.c_a * _pp(tf, "amax_pp") if self.c_a is not None else self.a
        return WuSwartz(a=acc, s_max=self.c_v * _pp(tf, "vmax_pp"),
                        n_dirs=self.n_dirs).predict(tf)


@dataclass
class FujimuraSugiharaPP:
    """Fujimura & Sugihara with v_max = c_v * table v_max_i (alpha stays global:
    it is a time constant, not an acceleration, so no table value applies)."""
    alpha: float = 1.3
    c_v: float = 1.0
    name: str = "FujimuraSugiharaPP"

    def predict(self, tf) -> np.ndarray:
        return FujimuraSugihara(alpha=self.alpha,
                                v_max=self.c_v * _pp(tf, "vmax_pp")).predict(tf)



# --------------------------------------------------------------------------- #
# registry: free-parameter specs (for fitting) and reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    """One model's tunable parameters for the fitting routine."""
    cls: type
    paper: str
    cont: dict = field(default_factory=dict)        # name -> (low, high) bounds
    discrete: dict = field(default_factory=dict)    # name -> list of choices
    fixed: dict = field(default_factory=dict)       # name -> fixed value


MODEL_SPECS = {
    "ConstSpeed": ParamSpec(
        ConstSpeed, paper="Renkin (a)",
        cont={"v_max": (4.0, 13.0)},
    ),
    "ConstAccel": ParamSpec(
        ConstAccel, paper="Renkin (b)",
        cont={"a_max": (2.0, 30.0)},
    ),
    "ConstAccelVmax": ParamSpec(
        ConstAccelVmax, paper="Renkin (c)",
        cont={"a_max": (2.0, 30.0), "v_max": (4.0, 13.0)},
    ),
    "TwoSegment": ParamSpec(
        TwoSegment, paper="Renkin (d)",
        cont={"t_inert": (0.05, 1.0), "v_const": (4.0, 13.0),
              "a_max": (2.0, 30.0), "v_max": (4.0, 13.0)},
        discrete={"keep_initial": [True, False]},
    ),
    "WuSwartz": ParamSpec(
        WuSwartz, paper="Wu & Swartz (e)",
        cont={"a": (2.0, 12.0), "s_max": (5.0, 12.0)},
        fixed={"n_dirs": 240},
    ),
    "FujimuraSugihara": ParamSpec(
        FujimuraSugihara, paper="Fujimura & Sugihara",
        cont={"alpha": (0.3, 4.0), "v_max": (4.0, 13.0)},
    ),
    # learned tables, no free parameters to search: fitted directly from
    # training tracking by scripts.fit_motion_models.fit_anzer, not by DE
    "AnzerBauer": ParamSpec(
        AnzerBauer, paper="Anzer & Bauer (learned)",
    ),
}

# Faithful Anzer & Bauer convex-hull replication (scripts.fit_motion_models.
# fit_anzer_hull). Kept OUT of MODEL_SPECS -- like the per-player variants --
# so the canonical default fit/eval model list is unchanged; it is still
# reachable by name through ALL_SPECS.
HULL_SPECS = {
    "AnzerBauerHull": ParamSpec(
        AnzerBauerHull, paper="Anzer & Bauer (hull)",
    ),
    # same construction, but the displacement cloud is first screened by the
    # SUVAT noise floor -- the physical-consistency test the sample pipeline
    # already applies to the MLP's training labels (pipeline.sampling), so the
    # hull and the learned model see data filtered by the same rule.
    "AnzerBauerHullSUVAT": ParamSpec(
        AnzerBauerHull, paper="Anzer & Bauer (hull, SUVAT-screened)",
    ),
}


# Bounds for the capacity scales. c_v is tight (the table's 99th-pct speeds are
# already the right order); c_a is wide because 99th-pct accelerations from
# double-differenced tracking are noisy and sit near the A_CLIP ceiling.
C_V_BOUNDS = (0.7, 1.4)
C_A_BOUNDS = (0.3, 3.0)

# Per-player variants, kept OUT of MODEL_SPECS so the canonical fit/eval default
# model list is unchanged. Variant tag is the trailing "P1"/"P2".
PERPLAYER_SPECS = {
    # ---- P1: per-player v_max, global acceleration -------------------------
    "ConstSpeedP1": ParamSpec(
        ConstSpeedPP, paper="Renkin (a) + per-player v_max",
        cont={"c_v": C_V_BOUNDS},
    ),
    "ConstAccelVmaxP1": ParamSpec(
        ConstAccelVmaxPP, paper="Renkin (c) + per-player v_max",
        cont={"c_v": C_V_BOUNDS, "a_max": (2.0, 30.0)},
        fixed={"c_a": None},
    ),
    "TwoSegmentP1": ParamSpec(
        TwoSegmentPP, paper="Renkin (d) + per-player v_max",
        cont={"t_inert": (0.05, 1.0), "v_const": (4.0, 13.0),
              "a_max": (2.0, 30.0), "c_v": C_V_BOUNDS},
        discrete={"keep_initial": [True, False]},
        fixed={"c_a": None},
    ),
    "WuSwartzP1": ParamSpec(
        WuSwartzPP, paper="Wu & Swartz (e) + per-player s_max",
        cont={"a": (2.0, 12.0), "c_v": C_V_BOUNDS},
        fixed={"n_dirs": 240, "c_a": None},
    ),
    "FujimuraSugiharaP1": ParamSpec(
        FujimuraSugiharaPP, paper="Fujimura & Sugihara + per-player v_max",
        cont={"alpha": (0.3, 4.0), "c_v": C_V_BOUNDS},
    ),
    # ---- P2: per-player v_max AND a_max ------------------------------------
    "ConstSpeedP2": ParamSpec(     # no acceleration parameter -> same as P1
        ConstSpeedPP, paper="Renkin (a) + per-player v_max",
        cont={"c_v": C_V_BOUNDS},
    ),
    "ConstAccelP2": ParamSpec(
        ConstAccelPP, paper="Renkin (b) + per-player a_max",
        cont={"c_a": C_A_BOUNDS},
    ),
    "ConstAccelVmaxP2": ParamSpec(
        ConstAccelVmaxPP, paper="Renkin (c) + per-player v_max/a_max",
        cont={"c_v": C_V_BOUNDS, "c_a": C_A_BOUNDS},
        fixed={"a_max": None},
    ),
    "TwoSegmentP2": ParamSpec(
        TwoSegmentPP, paper="Renkin (d) + per-player v_max/a_max",
        cont={"t_inert": (0.05, 1.0), "v_const": (4.0, 13.0),
              "c_a": C_A_BOUNDS, "c_v": C_V_BOUNDS},
        discrete={"keep_initial": [True, False]},
        fixed={"a_max": None},
    ),
    "WuSwartzP2": ParamSpec(
        WuSwartzPP, paper="Wu & Swartz (e) + per-player s_max/a",
        cont={"c_a": C_A_BOUNDS, "c_v": C_V_BOUNDS},
        fixed={"n_dirs": 240, "a": None},
    ),
    "FujimuraSugiharaP2": ParamSpec(   # alpha is a time constant -> same as P1
        FujimuraSugiharaPP, paper="Fujimura & Sugihara + per-player v_max",
        cont={"alpha": (0.3, 4.0), "c_v": C_V_BOUNDS},
    ),
}

# every spec the fitting / eval scripts can look up by name
ALL_SPECS = {**MODEL_SPECS, **PERPLAYER_SPECS, **HULL_SPECS}


def is_perplayer(model_name: str) -> bool:
    """True if the named model consumes per-sample tf["vmax_pp"]/["amax_pp"]."""
    return model_name in PERPLAYER_SPECS


def variant_of(model_name: str) -> str | None:
    """"P1"/"P2" for a per-player model, else None."""
    return model_name[-2:] if model_name in PERPLAYER_SPECS else None


def from_params(model_name: str, params: dict):
    """Rebuild a motion model from a saved parameter dict."""
    spec = ALL_SPECS[model_name]
    kwargs = {**spec.fixed, **params}
    kwargs.pop("name", None)
    return spec.cls(**kwargs)
