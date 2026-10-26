#!/usr/bin/env bash
# Cross-sport chain, take 2: nba train with --force-streaming (old-caps run
# streamed too; resident mode OOMs at 415M samples), then fits/evals + transfer.
set -uo pipefail
cd "$(dirname "$0")/.."
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }
step "nba train (streaming)"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=nba_match python -u -m scripts.train --force-streaming > outputs/nba_match_caps10_12/train_mlp.log 2>&1 || { echo "FAIL nba train"; exit 1; }
for DS in bdb_match nba_match; do
  OUT=outputs/${DS}_caps10_12
  step "$DS fit_motion_models"
  CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=$DS python -u -m scripts.fit_motion_models > "$OUT/fit_motion.log" 2>&1 || echo "FAIL $DS fit_motion"
  step "$DS compare_motion_models"
  CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=$DS python -u -m scripts.compare_motion_models > "$OUT/compare_motion.log" 2>&1 || echo "FAIL $DS compare"
done
step "cross_eval"
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.cross_eval > outputs/cross_eval_caps10_12.log 2>&1 || echo "FAIL cross_eval"
step "cross_sport_area_table"
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.cross_sport_area_table > outputs/cross_sport_area_caps10_12.log 2>&1 || echo "FAIL cross_area"
echo "XSPORT_DONE $(date '+%F %T')"
