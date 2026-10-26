#!/usr/bin/env bash
# Phase 3 (ffpc): fit physics baselines + full evaluation under caps10_12.
# Sequential; step names/logs mirror the old-caps runs for easy diffing.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }

step "fit_motion_models"
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.fit_motion_models   > "$OUT/fit_motion.log" 2>&1 || echo "FAIL fit_motion"
step "evaluate"
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.evaluate            > "$OUT/evaluate.log" 2>&1 || echo "FAIL evaluate"
step "compare_motion_models"
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.compare_motion_models > "$OUT/compare_motion.log" 2>&1 || echo "FAIL compare_motion"
step "eval_coverage_sweep (train_split profile)"
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_coverage_sweep > "$OUT/eval_coverage_sweep.log" 2>&1 || echo "FAIL coverage_sweep"
step "eval_reachable_area"
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.eval_reachable_area > "$OUT/eval_reachable_area.log" 2>&1 || echo "FAIL reachable_area"
step "eval_profile_area"
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_profile_area > "$OUT/eval_profile_area.log" 2>&1 || echo "FAIL profile_area"
step "bootstrap_ci"
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.bootstrap_ci > "$OUT/bootstrap_ci.log" 2>&1 || echo "FAIL bootstrap_ci"
step "audit_structure"
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.audit_structure > "$OUT/audit_structure.log" 2>&1 || echo "FAIL audit_structure"
echo "PHASE3_FFPC_DONE $(date '+%F %T')"
