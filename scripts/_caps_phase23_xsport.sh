#!/usr/bin/env bash
# Cross-sport chain: wait for nba build (file-based, no pgrep), train nba,
# then fit+eval bdb & nba, then the cross-sport transfer matrix.
set -uo pipefail
cd "$(dirname "$0")/.."
NBALOG=outputs/build_nba_caps10_12.log
until grep -q "Saved -> .*samples_caps10_12.npz" "$NBALOG" || grep -q "Traceback" "$NBALOG"; do sleep 120; done
grep -q "Traceback" "$NBALOG" && { echo "NBA BUILD FAILED"; exit 1; }

step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }
step "nba train"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=nba_match python -u -m scripts.train > outputs/nba_match_caps10_12/train_mlp.log 2>&1 || { echo "FAIL nba train"; exit 1; }
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
