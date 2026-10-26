"""Bootstrap confidence intervals for the base-vs-profile-MLP comparison
(ACCV rebuttal deliverable B: no retraining, uses the EXISTING canonical
checkpoints).

Two models:
  base    outputs/<ds>/mlp.pt                       (6 feat, canonical samples.npz)
  profile outputs/<ds>/mlp_profile_vmax_vmaxtrain.pt (7 feat = 6 + vmax,
                                                       leakage-free samples_vmaxtrain.npz)

Three analyses, all on the TEST split:
  1. Per-model point metrics (MAE, pinball) on each model's own canonical dataset.
  2. Paired match-level (cluster) bootstrap: resample the 9 test matches with
     replacement, 10,000 iterations. Evaluates BOTH models on the SAME dataset
     (samples_vmaxtrain.npz -- the base model only consumes its own 6 feature
     columns, which are identical across the two npz files; verified below) so
     the base-vs-profile gap is a valid paired comparison. 95% CI via the
     percentile method for each model's MAE/pinball and for the gap.
  3. Sample-level bootstrap. A literal 10,000-iteration resample of the full
     21.4M-row test set is not memory/time feasible (~10^11-10^12 element
     operations). Two memory-safe substitutes, reported together:
       (a) analytic normal-approximation CI: mean +/- 1.96 * std/sqrt(n) --
           exact under CLT at n=21.4M for a bounded-variance per-sample stat.
       (b) an actual 10,000-iteration bootstrap over a fixed 2,000,000-row
           uniform subsample of the test set (resampling WITH replacement each
           iteration), to empirically confirm (a)'s SE.

    python -m scripts.bootstrap_ci

Logs to outputs/<dataset>/bootstrap_ci.log (tee this script's stdout).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, DATASET, N_FEATURES, OUTPUTS_DIR, PINBALL_TAU, SEED, V_NORM  # noqa: E402
from src.model.mlp import TTAModel  # noqa: E402

TAU = PINBALL_TAU  # 0.01, the tau both canonical checkpoints were trained at

CANON_NPZ = DATA_DIR / DATASET / "processed" / "samples.npz"
VMAXTRAIN_NPZ = DATA_DIR / DATASET / "processed" / "samples_vmaxtrain.npz"
BASE_PT = OUTPUTS_DIR / "mlp.pt"
PROFILE_PT = OUTPUTS_DIR / "mlp_profile_vmax_vmaxtrain.pt"

N_MATCH_BOOT = 10_000
N_SAMPLE_BOOT = 10_000
SAMPLE_BOOT_SUBSET = 2_000_000  # rows for the memory-safe sample-level bootstrap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pinball_np(pred: np.ndarray, y: np.ndarray, tau: float = TAU) -> np.ndarray:
    err = y - pred
    return np.maximum(tau * err, (tau - 1.0) * err)


@torch.no_grad()
def predict(model, X: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk]).to(device)
        out[i:i + chunk] = model(xb).cpu().numpy()
    return out


def load_model(path: Path, in_dim: int) -> TTAModel:
    m = TTAModel(in_dim=in_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m


def percentile_ci(vals: np.ndarray, alpha: float = 0.05):
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def match_cluster_bootstrap(matches: np.ndarray, per_match_n: dict, per_match_sum_abs: dict,
                             per_match_sum_pin: dict, n_iter: int, rng: np.random.Generator):
    """Percentile-method paired cluster bootstrap over unique match ids.

    Returns arrays of shape (n_iter,) for MAE and mean-pinball, built from
    precomputed per-match sums (so each iteration is O(#matches), not O(n))."""
    uniq = np.array(sorted(per_match_n.keys()))
    K = len(uniq)
    n_arr = np.array([per_match_n[m] for m in uniq], dtype=np.float64)
    abs_arr = np.array([per_match_sum_abs[m] for m in uniq], dtype=np.float64)
    pin_arr = np.array([per_match_sum_pin[m] for m in uniq], dtype=np.float64)

    idx = rng.integers(0, K, size=(n_iter, K))  # resample match slots with replacement
    n_boot = n_arr[idx].sum(axis=1)
    mae_boot = abs_arr[idx].sum(axis=1) / n_boot
    pin_boot = pin_arr[idx].sum(axis=1) / n_boot
    return mae_boot, pin_boot, idx  # return idx so the paired gap reuses the SAME resamples


def per_match_sums(match_ids: np.ndarray, abs_err: np.ndarray, pin: np.ndarray):
    n_d, abs_d, pin_d = {}, {}, {}
    for m in np.unique(match_ids):
        sel = match_ids == m
        n_d[int(m)] = int(sel.sum())
        abs_d[int(m)] = float(abs_err[sel].sum())
        pin_d[int(m)] = float(pin[sel].sum())
    return n_d, abs_d, pin_d


def main():
    t0 = time.time()
    print(f"dataset: {DATASET}   tau: {TAU:g}   device: {device}")
    print(f"base checkpoint:    {BASE_PT}")
    print(f"profile checkpoint: {PROFILE_PT}")
    for p in (BASE_PT, PROFILE_PT, CANON_NPZ, VMAXTRAIN_NPZ):
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")

    # ---------------------------------------------------------------- load data
    print("\nloading canonical samples.npz ...")
    dc = np.load(CANON_NPZ)
    te_c = dc["split"] == 2
    Xc_te, yc_te = dc["X"][te_c], dc["y"][te_c]
    match_c_te = dc["meta_match"][te_c]

    print("loading samples_vmaxtrain.npz ...")
    dv = np.load(VMAXTRAIN_NPZ)
    te_v = dv["split"] == 2
    Xv_te, yv_te = dv["X"][te_v], dv["y"][te_v]
    match_v_te = dv["meta_match"][te_v]
    vmax_v_te = dv["meta_vmax"][te_v]

    # sanity: same test matches, same base 6 feature columns semantics, same y
    # ordering isn't guaranteed row-for-row, but the (match, count) multiset and
    # column count must match for the paired comparison to be valid.
    assert Xc_te.shape[1] == N_FEATURES == Xv_te.shape[1], "base feature width mismatch"
    assert sorted(np.unique(match_c_te).tolist()) == sorted(np.unique(match_v_te).tolist()), \
        "canonical and vmaxtrain test splits cover different matches -- pairing invalid"
    assert len(Xc_te) == len(Xv_te), "canonical and vmaxtrain test splits differ in size"
    print(f"test samples: canonical {len(Xc_te):,} | vmaxtrain {len(Xv_te):,} "
          f"(same {len(np.unique(match_v_te))} matches: {sorted(np.unique(match_v_te).tolist())})")

    Xp_v_te = np.concatenate([Xv_te, (vmax_v_te / V_NORM).astype(np.float32)[:, None]], axis=1)

    # ---------------------------------------------------------------- models
    base_model = load_model(BASE_PT, in_dim=N_FEATURES)
    profile_model = load_model(PROFILE_PT, in_dim=N_FEATURES + 1)

    print("\npredicting ...")
    pred_base_canon = predict(base_model, Xc_te)          # base model, own canonical config
    pred_base_vmaxtrain = predict(base_model, Xv_te)      # base model, on vmaxtrain test set (for pairing)
    pred_profile_vmaxtrain = predict(profile_model, Xp_v_te)  # profile model, own canonical config

    abs_base_canon = np.abs(pred_base_canon - yc_te)
    pin_base_canon = pinball_np(pred_base_canon, yc_te)
    abs_base_vt = np.abs(pred_base_vmaxtrain - yv_te)
    pin_base_vt = pinball_np(pred_base_vmaxtrain, yv_te)
    abs_prof_vt = np.abs(pred_profile_vmaxtrain - yv_te)
    pin_prof_vt = pinball_np(pred_profile_vmaxtrain, yv_te)

    print(f"\n{'model / dataset':38s} {'MAE':>8s} {'pinball':>9s}")
    print(f"{'base MLP  (canonical samples.npz)':38s} {abs_base_canon.mean():8.4f} {pin_base_canon.mean():9.5f}")
    print(f"{'base MLP  (on vmaxtrain test set)':38s} {abs_base_vt.mean():8.4f} {pin_base_vt.mean():9.5f}")
    print(f"{'profile MLP (vmaxtrain, own cfg)':38s} {abs_prof_vt.mean():8.4f} {pin_prof_vt.mean():9.5f}")
    gap_point = abs_base_vt.mean() - abs_prof_vt.mean()
    print(f"\npoint base-vs-profile MAE gap (paired, on vmaxtrain test set): {gap_point:+.4f} s")

    # ============================================================ 1. match-level (cluster) bootstrap
    print(f"\n=== paired match-level bootstrap (resample 9 test matches, {N_MATCH_BOOT:,} iters) ===")
    rng = np.random.default_rng(SEED)

    n_c, abs_c, pin_c = per_match_sums(match_c_te, abs_base_canon, pin_base_canon)
    mae_boot_base_canon, pin_boot_base_canon, _ = match_cluster_bootstrap(
        match_c_te, n_c, abs_c, pin_c, N_MATCH_BOOT, np.random.default_rng(SEED))

    n_bv, abs_bv, pin_bv = per_match_sums(match_v_te, abs_base_vt, pin_base_vt)
    n_pv, abs_pv, pin_pv = per_match_sums(match_v_te, abs_prof_vt, pin_prof_vt)
    # SAME resample indices (same rng draw) for both models on vmaxtrain -> valid pairing
    rng_pair = np.random.default_rng(SEED + 1)
    uniq = np.array(sorted(n_bv.keys()))
    K = len(uniq)
    idx = rng_pair.integers(0, K, size=(N_MATCH_BOOT, K))

    def boot_from_idx(idx, n_d, abs_d, pin_d):
        n_arr = np.array([n_d[m] for m in uniq], dtype=np.float64)
        abs_arr = np.array([abs_d[m] for m in uniq], dtype=np.float64)
        pin_arr = np.array([pin_d[m] for m in uniq], dtype=np.float64)
        n_boot = n_arr[idx].sum(axis=1)
        return abs_arr[idx].sum(axis=1) / n_boot, pin_arr[idx].sum(axis=1) / n_boot

    mae_boot_base_vt, pin_boot_base_vt = boot_from_idx(idx, n_bv, abs_bv, pin_bv)
    mae_boot_prof_vt, pin_boot_prof_vt = boot_from_idx(idx, n_pv, abs_pv, pin_pv)
    gap_boot = mae_boot_base_vt - mae_boot_prof_vt  # paired: same resample of matches both sides

    def report(name, vals, point):
        lo, hi = percentile_ci(vals)
        print(f"{name:42s} point {point:8.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
              f"(boot mean {vals.mean():.4f}, std {vals.std():.5f})")

    print("\n-- MAE --")
    report("base MLP (canonical)", mae_boot_base_canon, abs_base_canon.mean())
    report("base MLP (on vmaxtrain test set)", mae_boot_base_vt, abs_base_vt.mean())
    report("profile MLP (vmaxtrain)", mae_boot_prof_vt, abs_prof_vt.mean())
    print("\n-- pinball --")
    report("base MLP (canonical)", pin_boot_base_canon, pin_base_canon.mean())
    report("base MLP (on vmaxtrain test set)", pin_boot_base_vt, pin_base_vt.mean())
    report("profile MLP (vmaxtrain)", pin_boot_prof_vt, pin_prof_vt.mean())
    print("\n-- base-vs-profile MAE gap (base - profile, paired, vmaxtrain test set) --")
    lo, hi = percentile_ci(gap_boot)
    excludes_zero = lo > 0 or hi < 0
    print(f"point gap {gap_point:+.4f} s   95% CI [{lo:+.4f}, {hi:+.4f}] s   "
          f"excludes 0: {excludes_zero}   (boot mean {gap_boot.mean():+.4f}, std {gap_boot.std():.5f})")

    # ============================================================ 2. sample-level bootstrap
    print(f"\n=== sample-level bootstrap / analytic CI (test n = {len(yv_te):,}) ===")
    print("Literal 10,000-iteration resampling of the full 21.4M-row test set is not "
          "memory/time feasible in this script (order 10^11 element ops per statistic). "
          "Reporting two memory-safe substitutes:")
    print("  (a) analytic normal-approximation CI: mean +/- 1.96 * std/sqrt(n) (CLT, exact "
          "at this n for a bounded-variance per-sample statistic).")
    print(f"  (b) an actual {N_SAMPLE_BOOT:,}-iteration bootstrap over a fixed uniform "
          f"{SAMPLE_BOOT_SUBSET:,}-row subsample of the test set, to empirically confirm (a).")

    def analytic_ci(x: np.ndarray):
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n)
        m = x.mean()
        return m, se, (m - 1.96 * se, m + 1.96 * se)

    for name, x in (
        ("base MLP MAE (canonical)", abs_base_canon),
        ("base MLP pinball (canonical)", pin_base_canon),
        ("base MLP MAE (vmaxtrain)", abs_base_vt),
        ("base MLP pinball (vmaxtrain)", pin_base_vt),
        ("profile MLP MAE (vmaxtrain)", abs_prof_vt),
        ("profile MLP pinball (vmaxtrain)", pin_prof_vt),
    ):
        m, se, (lo, hi) = analytic_ci(x)
        print(f"  {name:36s} mean {m:8.4f}  SE {se:.6f}  95% CI [{lo:.4f}, {hi:.4f}]")

    gap_sample = abs_base_vt - abs_prof_vt  # paired per-row gap on the SAME vmaxtrain rows
    m, se, (lo, hi) = analytic_ci(gap_sample)
    print(f"  {'base-vs-profile MAE gap (paired rows)':36s} mean {m:+8.4f}  SE {se:.6f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  excludes 0: {lo > 0 or hi < 0}")

    print(f"\n(b) chunked bootstrap on a fixed {SAMPLE_BOOT_SUBSET:,}-row uniform subsample:")
    sub_rng = np.random.default_rng(SEED + 2)
    n_test = len(yv_te)
    sub_idx = sub_rng.choice(n_test, size=SAMPLE_BOOT_SUBSET, replace=False)
    sub_gap = gap_sample[sub_idx]
    sub_mae_base = abs_base_vt[sub_idx]
    sub_mae_prof = abs_prof_vt[sub_idx]

    boot_rng = np.random.default_rng(SEED + 3)
    m_sub = SAMPLE_BOOT_SUBSET
    boot_gap = np.empty(N_SAMPLE_BOOT, dtype=np.float64)
    boot_mae_base = np.empty(N_SAMPLE_BOOT, dtype=np.float64)
    boot_mae_prof = np.empty(N_SAMPLE_BOOT, dtype=np.float64)
    t_boot0 = time.time()
    for i in range(N_SAMPLE_BOOT):
        bidx = boot_rng.integers(0, m_sub, size=m_sub)
        boot_gap[i] = sub_gap[bidx].mean()
        boot_mae_base[i] = sub_mae_base[bidx].mean()
        boot_mae_prof[i] = sub_mae_prof[bidx].mean()
    print(f"  ({N_SAMPLE_BOOT:,} iterations over {m_sub:,} rows took {time.time()-t_boot0:.1f}s)")

    for name, vals in (("base MLP MAE (subsample boot)", boot_mae_base),
                        ("profile MLP MAE (subsample boot)", boot_mae_prof),
                        ("base-vs-profile MAE gap (subsample boot)", boot_gap)):
        lo, hi = percentile_ci(vals)
        # rescale the subsample-bootstrap SE to the full test-set n for comparison with (a)
        se_sub = vals.std()
        se_full_equiv = se_sub * np.sqrt(SAMPLE_BOOT_SUBSET / n_test)
        print(f"  {name:42s} 95% CI @ subsample n [{lo:+.4f}, {hi:+.4f}]   "
              f"boot SE {se_sub:.6f}  (-> full-n-equivalent SE {se_full_equiv:.6f})")

    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
