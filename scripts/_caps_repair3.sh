#!/usr/bin/env bash
# Repair chain: refit ALL physics baselines at the canonical 0.99 gate
# (my first pass used the 0.95 default -- wrong), then redo dependent evals.
set -uo pipefail
cd "$(dirname "$0")/.."
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }
SIX="ConstSpeed ConstAccel ConstAccelVmax TwoSegment WuSwartz FujimuraSugihara"

step "ffpc refit @0.99"
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 > outputs/ffpc_match_caps10_12/fit_motion.log 2>&1 || echo "FAIL ffpc fit"
step "bdb refit @0.99 (six models; AnzerBauer needs multi-match)"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=bdb_match python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models $SIX > outputs/bdb_match_caps10_12/fit_motion.log 2>&1 || echo "FAIL bdb fit"
step "nba refit @0.99"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=nba_match python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 > outputs/nba_match_caps10_12/fit_motion.log 2>&1 || echo "FAIL nba fit"

step "ffpc compare_motion_models"
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.compare_motion_models > outputs/ffpc_match_caps10_12/compare_motion.log 2>&1 || echo "FAIL ffpc compare"
step "ffpc eval_coverage_sweep"
CUDA_VISIBLE_DEVICES=1 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_coverage_sweep > outputs/ffpc_match_caps10_12/eval_coverage_sweep.log 2>&1 || echo "FAIL sweep"
step "bdb compare"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=bdb_match python -u -m scripts.compare_motion_models > outputs/bdb_match_caps10_12/compare_motion.log 2>&1 || echo "FAIL bdb compare"
step "nba compare"
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=nba_match python -u -m scripts.compare_motion_models > outputs/nba_match_caps10_12/compare_motion.log 2>&1 || echo "FAIL nba compare"

step "cross_eval 3x3"
for TR in ffpc_match bdb_match nba_match; do for TE in ffpc_match bdb_match nba_match; do
  CUDA_VISIBLE_DEVICES=1 python -u -m scripts.cross_eval --train $TR --test $TE > "outputs/cross_eval_${TR%_match}_${TE%_match}_caps10_12.log" 2>&1 || echo "FAIL cross $TR->$TE"
done; done
step "cross_sport_area_table"
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.cross_sport_area_table > outputs/cross_sport_area_caps10_12.log 2>&1 || echo "FAIL cross_area"
echo "REPAIR3_DONE $(date '+%F %T')"
