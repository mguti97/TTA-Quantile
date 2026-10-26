#!/usr/bin/env bash
# Follow-ups after the Phase-4 comparison:
#  1. controlled audit re-run (old settings: vmax_scope=match, 5000 states)
#  2. AnzerBauer hull refit + its SUVAT-screened variant, then hull sweeps
#  3. base seed 46 (old band had it)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }

step "audit controlled (vmax_scope=match, 5000 states)"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.audit_structure --states 5000 > "$OUT/audit_structure_match5000.log" 2>&1 || echo "FAIL audit_ctrl"

step "base seed 46"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.train --seed 46 --out-suffix _seed46 > "$OUT/train_mlp_seed46.log" 2>&1 || echo "FAIL seed46" &
S46=$!

step "anzer hull fit (raw)"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.fit_motion_models --anzer-hull-only --hit-ratio-min 0.99 --models-json "$OUT/motion_models_anzerhull.json" > "$OUT/fit_anzerhull.log" 2>&1 || echo "FAIL hull"
step "anzer hull fit (suvat-screened)"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.fit_motion_models --anzer-hull-only --suvat-screen --hit-ratio-min 0.99 --models-json "$OUT/motion_models_anzerhull_suvat.json" > "$OUT/fit_anzerhull_suvat.log" 2>&1 || echo "FAIL hull_suvat"
step "hull compare + sweep"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.compare_motion_models --models-json "$OUT/motion_models_anzerhull.json" "$OUT/motion_models_anzerhull_suvat.json" > "$OUT/anzerhull_compare_motion.log" 2>&1 || echo "FAIL hull_compare"
CUDA_VISIBLE_DEVICES=2 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_coverage_sweep --models-json "$OUT/motion_models_anzerhull.json" "$OUT/motion_models_anzerhull_suvat.json" > "$OUT/anzerhull_coverage_sweep.log" 2>&1 || echo "FAIL hull_sweep"

wait $S46
step "re-aggregate seeds (with 46)"
CUDA_VISIBLE_DEVICES=3 KOGNIA_VMAX_SCOPE=train_split python -u rebuttal/aggregate_seeds.py > "$OUT/aggregate_seeds.log" 2>&1 || echo "FAIL aggregate"
echo "FOLLOWUPS_DONE $(date '+%F %T')"
