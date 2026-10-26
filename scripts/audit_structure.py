"""Post-hoc structural audit of the learned TTA surface (ACCV rebuttal, 5MaP-M4).

The reviewer asks what structural guarantees are lost when an analytic motion
model is replaced by a learned tau(.). Three properties are checked empirically
on the TEST split, for real player kinematic states:

  1. RAY MONOTONICITY -- along a ray of fixed heading-frame angle, does
     tau_pred grow monotonically with target distance d?  (An analytic model
     has this by construction; an MLP does not.)
  2. SUBLEVEL-SET CONNECTIVITY -- is R(dt) = {x : tau_pred(x) <= dt} a single
     connected region, or does it fragment?
  3. STAR-SHAPEDNESS -- is R(dt) intersected with each ray a single interval
     anchored at the player?  This is the property that would let the reachable
     set be stored as a polygon (as Renkin et al. do).

The surface is translation/rotation invariant by construction (features.transform
puts the player at the origin with heading along +x), so the audit works entirely
in the HEADING frame: a state is fully described by (v, a'x, a'y) -- exactly
columns 3:6 of the built feature matrix -- and the geometry columns (d, cos, sin)
are swept over rays / a Cartesian grid.

Run:
    KOGNIA_DATASET=ffpc_match python -m scripts.audit_structure --states 3000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    AREA_GRID_HALF,
    DATASET,
    DATASET_NPZ,
    D_NORM,
    GRID_RES,
    MODEL_PT,
    OUTPUTS_DIR,
    SEED,
    VMAX_SCOPE,
    V_NORM,
)
from src.model.mlp import TTAModel  # noqa: E402

HORIZONS = (1.0, 2.0, 3.0, 4.0, 5.0)   # canonical eval horizons
N_ANGLES = 16
RAY_D_MIN, RAY_D_MAX, RAY_D_STEP = 0.5, 48.0, 0.25
VIOL_EPS = 1e-4        # seconds; below this a "drop" is float noise
VIOL_MATERIAL = 0.05   # seconds; a drop a practitioner would notice
SPECKLE_FRAC = 0.01    # secondary component < 1% of main == negligible speckle


# --------------------------------------------------------------------------
# model / data plumbing
# --------------------------------------------------------------------------
def pick_device() -> torch.device:
    """Least-loaded CUDA device (other jobs share these GPUs), else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    print(f"device: cuda:{best} ({best_free / 2**30:.1f} GiB free)", flush=True)
    return torch.device(f"cuda:{best}")


def load_model(path: Path, in_dim: int, device) -> TTAModel:
    m = TTAModel(in_dim=in_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def predict(model, X: torch.Tensor, chunk: int = 2_000_000) -> torch.Tensor:
    """tau for a (M, F) feature tensor, chunked to bound VRAM."""
    out = torch.empty(X.shape[0], dtype=torch.float32, device=X.device)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        out[s:e] = model(X[s:e])
    return out


def load_test_states(n_states: int, seed: int):
    """Sample n_states heading-frame kinematic states (v, a'x, a'y) from TEST.

    Reads the already-built canonical dataset; ``split`` was written by
    scripts.build_dataset.assign_splits (2 == test). Returns the normalised
    state columns (as the model consumes them) plus the raw v_max (m/s) each
    row's player carries, for the 7-feature profile model.
    """
    print(f"loading {DATASET_NPZ} ...", flush=True)
    t0 = time.perf_counter()
    z = np.load(DATASET_NPZ)
    split = z["split"]
    test_idx = np.where(split == 2)[0]
    print(f"  {len(split):,} samples, {len(test_idx):,} in TEST "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(test_idx, size=min(n_states, len(test_idx)), replace=False))
    X = z["X"]                       # (N, 6) float32
    state_cols = X[pick, 3:6].copy()  # v/V_NORM, a'x/A_NORM, a'y/A_NORM
    vmax = z["meta_vmax"][pick].astype(np.float32)
    del X, split
    print(f"  sampled {len(pick):,} TEST states "
          f"({time.perf_counter() - t0:.0f}s total)", flush=True)
    return state_cols, vmax


def build_inputs(geom: torch.Tensor, states: torch.Tensor,
                 vmax_norm: torch.Tensor | None) -> torch.Tensor:
    """(S, G, F) feature tensor from (G,3) geometry cols and (S,3) state cols."""
    S, G = states.shape[0], geom.shape[0]
    F = 6 if vmax_norm is None else 7
    X = torch.empty(S, G, F, device=states.device, dtype=torch.float32)
    X[:, :, :3] = geom.unsqueeze(0)
    X[:, :, 3:6] = states.unsqueeze(1)
    if vmax_norm is not None:
        X[:, :, 6] = vmax_norm.unsqueeze(1)
    return X.reshape(S * G, F)


# --------------------------------------------------------------------------
# 1 + 3: rays
# --------------------------------------------------------------------------
def ray_geometry(device):
    """(A*D, 3) geometry columns for A angles x D distances, plus the axes."""
    ds = np.arange(RAY_D_MIN, RAY_D_MAX + 1e-9, RAY_D_STEP)
    angs = np.arange(N_ANGLES) * (2 * np.pi / N_ANGLES)
    dd, aa = np.meshgrid(ds, angs, indexing="xy")     # (A, D)
    cols = np.stack([(dd / D_NORM).ravel(), np.cos(aa).ravel(), np.sin(aa).ravel()],
                    axis=1).astype(np.float32)
    return torch.from_numpy(cols).to(device), ds, angs


def ray_audit(model, states, vmax_norm, device, state_chunk=256):
    """Monotonicity (1) and star-shapedness (3) statistics over rays."""
    geom, ds, angs = ray_geometry(device)
    A, D = len(angs), len(ds)
    S = states.shape[0]

    n_rays = 0
    n_viol = 0                     # rays with any drop > VIOL_EPS
    n_viol_mat = 0                 # rays with a drop > VIOL_MATERIAL
    states_viol = 0
    max_drops = []                 # per-ray max drawdown (s), violating rays only
    viol_d = []                    # distances where a decreasing step occurs
    viol_ang = []                  # |angle| (deg from heading) of those steps
    viol_tau = []                  # tau level (s) at which the drop happens
    # star-shapedness: per horizon, count rays whose reachable mask is a prefix
    star_ok = {h: 0 for h in HORIZONS}
    star_states_ok = {h: 0 for h in HORIZONS}
    star_gap_len = {h: [] for h in HORIZONS}

    for s0 in range(0, S, state_chunk):
        s1 = min(s0 + state_chunk, S)
        st = states[s0:s1]
        vm = None if vmax_norm is None else vmax_norm[s0:s1]
        tau = predict(model, build_inputs(geom, st, vm)).reshape(-1, A, D)  # (cs,A,D)

        diff = tau[:, :, 1:] - tau[:, :, :-1]
        dec = diff < -VIOL_EPS                                    # (cs,A,D-1)
        ray_bad = dec.any(dim=2)                                  # (cs,A)
        # max drawdown along the ray = max(cummax(tau) - tau)
        drawdown = (torch.cummax(tau, dim=2).values - tau).amax(dim=2)  # (cs,A)

        n_rays += ray_bad.numel()
        n_viol += int(ray_bad.sum())
        n_viol_mat += int((drawdown > VIOL_MATERIAL).sum())
        states_viol += int(ray_bad.any(dim=1).sum())
        if ray_bad.any():
            max_drops.append(drawdown[ray_bad].cpu().numpy())
            ci, ai, di = torch.nonzero(dec, as_tuple=True)
            viol_d.append(ds[di.cpu().numpy()])
            viol_ang.append(np.degrees(angs[ai.cpu().numpy()]))
            viol_tau.append(tau[ci, ai, di].cpu().numpy())

        for h in HORIZONS:
            reach = tau <= h                                      # (cs,A,D)
            # prefix test: no True after the first False
            first_false = (~reach).float().argmax(dim=2)          # 0 if reach[0] False
            all_true = reach.all(dim=2)
            idx = torch.arange(D, device=tau.device).view(1, 1, D)
            after = reach & (idx >= first_false.unsqueeze(2))
            ok = all_true | (~after.any(dim=2))                   # (cs,A)
            star_ok[h] += int(ok.sum())
            star_states_ok[h] += int(ok.all(dim=1).sum())
            if (~ok).any():
                star_gap_len[h].append(after.sum(dim=2)[~ok].cpu().numpy())

    md = np.concatenate(max_drops) if max_drops else np.zeros(0)
    vd = np.concatenate(viol_d) if viol_d else np.zeros(0)
    va = np.concatenate(viol_ang) if viol_ang else np.zeros(0)
    va = np.abs(((va + 180.0) % 360.0) - 180.0)                   # -> [0,180] from heading
    return dict(n_rays=n_rays, n_viol=n_viol, n_viol_mat=n_viol_mat,
                states_viol=states_viol, n_states=S, max_drops=md,
                viol_d=vd, viol_ang=va, ds=ds,
                viol_tau=(np.concatenate(viol_tau) if viol_tau else np.zeros(0)),
                star_ok=star_ok, star_states_ok=star_states_ok,
                star_gap_len={h: (np.concatenate(v) if v else np.zeros(0))
                              for h, v in star_gap_len.items()},
                n_star_rays=S * A)


def report_rays(r, log):
    log(f"\n{'='*78}\n1. RAY MONOTONICITY  ({r['n_states']} states x {N_ANGLES} angles, "
        f"d = {RAY_D_MIN}..{RAY_D_MAX} m step {RAY_D_STEP} m)\n{'='*78}")
    log(f"  rays audited                      : {r['n_rays']:,}")
    log(f"  rays with ANY drop (> {VIOL_EPS:g} s)   : {r['n_viol']:,} "
        f"({100*r['n_viol']/r['n_rays']:.3f} %)")
    log(f"  rays with drop > {VIOL_MATERIAL:g} s        : {r['n_viol_mat']:,} "
        f"({100*r['n_viol_mat']/r['n_rays']:.3f} %)")
    log(f"  states with >=1 violating ray     : {r['states_viol']:,} "
        f"({100*r['states_viol']/r['n_states']:.2f} %)")
    md = r["max_drops"]
    if md.size:
        qs = np.percentile(md, [50, 90, 99, 100])
        log(f"  max drawdown per violating ray (s): median {qs[0]:.4f}  p90 {qs[1]:.4f}  "
            f"p99 {qs[2]:.4f}  max {qs[3]:.4f}")
        log(f"  drawdown > 0.01 s / 0.05 s / 0.1 s: "
            f"{100*(md>0.01).mean():.2f} % / {100*(md>0.05).mean():.2f} % / "
            f"{100*(md>0.1).mean():.2f} % of violating rays")
    else:
        log("  no violations at all -- surface is exactly ray-monotone on this sample")
        return
    vd, va = r["viol_d"], r["viol_ang"]
    log(f"  decreasing STEPS: {vd.size:,} of "
        f"{r['n_rays']*(len(r['ds'])-1):,} sampled steps "
        f"({100*vd.size/(r['n_rays']*(len(r['ds'])-1)):.4f} %)")
    edges = [0, 2, 5, 10, 20, 30, 40, 48.01]
    h, _ = np.histogram(vd, bins=edges)
    log("  where in d (m): " + "  ".join(
        f"[{edges[i]:g},{edges[i+1]:g}) {100*h[i]/max(h.sum(),1):.1f}%" for i in range(len(h))))
    aedges = [0, 30, 60, 90, 120, 150, 180.01]
    ha, _ = np.histogram(va, bins=aedges)
    log("  where in |angle from heading| (deg): " + "  ".join(
        f"[{aedges[i]:g},{aedges[i+1]:g}) {100*ha[i]/max(ha.sum(),1):.1f}%"
        for i in range(len(ha))))
    log(f"  ({100*(va>90).mean():.1f} % of violating steps are BEHIND the player, "
        f"|angle| > 90 deg)")
    vt = r["viol_tau"]
    if vt.size:
        log(f"  tau level at the drop (s)         : median {np.median(vt):.2f}  "
            f"p5 {np.percentile(vt,5):.2f}  min {vt.min():.2f}  max {vt.max():.2f}")
        log("  fraction of drops INSIDE a canonical horizon: " + "  ".join(
            f"tau<={h:.0f}s {100*(vt<=h).mean():.2f}%" for h in HORIZONS))


def report_star(r, log):
    log(f"\n{'='*78}\n3. STAR-SHAPEDNESS (reachable set n ray == single interval from the "
        f"player)\n{'='*78}")
    log(f"  {'dt':>4}  {'rays single-interval':>22}  {'states all-rays-ok':>20}  "
        f"{'median gap (cells)':>19}")
    for h in HORIZONS:
        ok, tot = r["star_ok"][h], r["n_star_rays"]
        sok = r["star_states_ok"][h]
        g = r["star_gap_len"][h]
        gs = f"{np.median(g):.0f}" if g.size else "-"
        log(f"  {h:4.0f}  {ok:,}/{tot:,} = {100*ok/tot:8.4f} %  "
            f"{100*sok/r['n_states']:18.3f} %  {gs:>19}")


# --------------------------------------------------------------------------
# 2: connectivity
# --------------------------------------------------------------------------
def grid_geometry(device, half: float, res: float):
    """Heading-frame Cartesian grid of cell CENTRES, (G,3) geometry columns."""
    ax = np.arange(-half + res / 2, half, res)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    x, y = xx.ravel(), yy.ravel()
    d = np.hypot(x, y)
    cols = np.stack([d / D_NORM, x / d, y / d], axis=1).astype(np.float32)
    return torch.from_numpy(cols).to(device), len(ax), (xx, yy)


def connectivity_audit(model, states, vmax_norm, device, half, res,
                       state_chunk=64, connectivity=1):
    geom, n, (xx, yy) = grid_geometry(device, half, res)
    S = states.shape[0]
    cell_area = res * res
    struct = ndimage.generate_binary_structure(2, connectivity)
    # cell index closest to the player (origin)
    oi = int(np.argmin(np.hypot(xx.ravel(), yy.ravel())))
    oix, oiy = oi // n, oi % n

    acc = {h: dict(n=0, empty=0, multi=0, multi_material=0, origin_not_largest=0,
                   sec_cells=[], sec_frac=[], ncomp=[], main_cells=[]) for h in HORIZONS}

    for s0 in range(0, S, state_chunk):
        s1 = min(s0 + state_chunk, S)
        st = states[s0:s1]
        vm = None if vmax_norm is None else vmax_norm[s0:s1]
        tau = predict(model, build_inputs(geom, st, vm)).reshape(-1, n, n).cpu().numpy()

        for h in HORIZONS:
            a = acc[h]
            for k in range(tau.shape[0]):
                mask = tau[k] <= h
                a["n"] += 1
                if not mask.any():
                    a["empty"] += 1
                    continue
                lab, ncomp = ndimage.label(mask, structure=struct)
                sizes = np.bincount(lab.ravel())[1:]
                order = np.argsort(sizes)[::-1]
                main = sizes[order[0]]
                a["ncomp"].append(ncomp)
                a["main_cells"].append(main)
                if ncomp > 1:
                    a["multi"] += 1
                    sec = sizes[order[1]]
                    a["sec_cells"].append(sec)
                    a["sec_frac"].append(sec / main)
                    if sec / main >= SPECKLE_FRAC:
                        a["multi_material"] += 1
                olab = lab[oix, oiy]
                if olab == 0:
                    # player's own cell unreachable -> attribute to nearest labelled cell
                    yidx, xidx = np.nonzero(lab)
                    j = np.argmin((yidx - oix) ** 2 + (xidx - oiy) ** 2)
                    olab = lab[yidx[j], xidx[j]]
                if sizes[olab - 1] != main:
                    a["origin_not_largest"] += 1
    for h in HORIZONS:
        a = acc[h]
        for k in ("sec_cells", "sec_frac", "ncomp", "main_cells"):
            a[k] = np.asarray(a[k], dtype=float)
        a["cell_area"] = cell_area
    return acc


def report_connectivity(acc, log, half, res, connectivity):
    log(f"\n{'='*78}\n2. SUBLEVEL-SET CONNECTIVITY  (heading-frame grid +/-{half:g} m @ "
        f"{res:g} m, {'4' if connectivity==1 else '8'}-connectivity)\n{'='*78}")
    log(f"  {'dt':>4} {'states':>7} {'empty':>6} {'>1 comp':>12} "
        f"{'>1 comp & sec>=1% main':>23} {'main area m2':>13} {'sec area m2 (p50/max)':>22} "
        f"{'origin!=largest':>16}")
    for h in HORIZONS:
        a = acc[h]
        n = a["n"]
        sec = a["sec_cells"] * a["cell_area"]
        secs = (f"{np.median(sec):.1f}/{sec.max():.1f}" if sec.size else "-")
        log(f"  {h:4.0f} {n:7d} {a['empty']:6d} "
            f"{a['multi']:5d} = {100*a['multi']/n:5.2f}% "
            f"{a['multi_material']:11d} = {100*a['multi_material']/n:6.3f}% "
            f"{np.mean(a['main_cells'])*a['cell_area']:13.1f} {secs:>22} "
            f"{100*a['origin_not_largest']/n:15.3f}%")
    for h in HORIZONS:
        a = acc[h]
        if a["sec_frac"].size:
            sf = a["sec_frac"]
            log(f"    dt={h:.0f}s  secondary/main size ratio over disconnected states: "
                f"median {np.median(sf):.5f}  p90 {np.percentile(sf,90):.5f}  max {sf.max():.5f}"
                f"   |  n_components median {np.median(a['ncomp']):.0f} "
                f"max {a['ncomp'].max():.0f}")


# --------------------------------------------------------------------------
def audit_model(name, model, states_t, vmax_norm_t, device, args, log):
    log(f"\n\n{'#'*78}\n# MODEL: {name}\n{'#'*78}")
    t0 = time.perf_counter()
    r = ray_audit(model, states_t, vmax_norm_t, device, state_chunk=args.ray_chunk)
    log(f"[ray sweep: {time.perf_counter()-t0:.1f}s]")
    report_rays(r, log)
    report_star(r, log)

    t0 = time.perf_counter()
    acc = connectivity_audit(model, states_t, vmax_norm_t, device, args.half, args.res,
                             state_chunk=args.grid_chunk, connectivity=args.connectivity)
    log(f"\n[grid sweep: {time.perf_counter()-t0:.1f}s]")
    report_connectivity(acc, log, args.half, args.res, args.connectivity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--half", type=float, default=AREA_GRID_HALF)
    ap.add_argument("--res", type=float, default=GRID_RES)
    ap.add_argument("--connectivity", type=int, default=1, choices=(1, 2),
                    help="1 = 4-connectivity (conservative), 2 = 8-connectivity")
    ap.add_argument("--ray-chunk", type=int, default=256)
    ap.add_argument("--grid-chunk", type=int, default=64)
    ap.add_argument("--mlp-path", type=str, default=str(MODEL_PT))
    ap.add_argument("--profile-path", type=str,
                    default=str(OUTPUTS_DIR / "mlp_profile_vmax_vmaxtrain.pt"))
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--log", type=str, default=str(OUTPUTS_DIR / "audit_structure.log"))
    args = ap.parse_args()

    logf = open(args.log, "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    log(f"structural audit -- dataset={DATASET}  vmax_scope={VMAX_SCOPE}  seed={args.seed}")
    log(f"horizons {HORIZONS}  grid +/-{args.half:g} m @ {args.res:g} m  "
        f"connectivity={'4' if args.connectivity == 1 else '8'}")

    device = pick_device()
    states, vmax = load_test_states(args.states, args.seed)
    log(f"states: {states.shape[0]}  v (m/s) median {np.median(states[:,0])*V_NORM:.2f} "
        f"p95 {np.percentile(states[:,0],95)*V_NORM:.2f}   "
        f"v_max (m/s) median {np.median(vmax):.2f}")

    states_t = torch.from_numpy(states).to(device)
    vmax_t = torch.from_numpy(vmax / V_NORM).to(device)

    base = load_model(Path(args.mlp_path), 6, device)
    audit_model(f"base MLP  ({Path(args.mlp_path).name})", base, states_t, None, device,
                args, log)

    if not args.no_profile and Path(args.profile_path).exists():
        prof = load_model(Path(args.profile_path), 7, device)
        audit_model(f"profile MLP+vmax  ({Path(args.profile_path).name})", prof,
                    states_t, vmax_t, device, args, log)
    elif not args.no_profile:
        log(f"\n(profile checkpoint {args.profile_path} not found -- skipped)")

    log(f"\nwrote {args.log}")
    logf.close()


if __name__ == "__main__":
    main()
