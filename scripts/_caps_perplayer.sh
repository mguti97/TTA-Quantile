#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
export KOGNIA_VMAX_SCOPE=train_split   # resolves vmax_table_vmaxtrain_caps10_12.npz
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }
step "p1_rest"; CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models ConstSpeedP1 ConstAccelVmaxP1 TwoSegmentP1 FujimuraSugiharaP1 --models-json "$OUT/perplayer_fit_p1_rest.json" > "$OUT/perplayer_fit_p1_rest.log" 2>&1 || echo "FAIL p1_rest"
step "p2_rest"; CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models ConstSpeedP2 ConstAccelP2 ConstAccelVmaxP2 TwoSegmentP2 FujimuraSugiharaP2 --models-json "$OUT/perplayer_fit_p2_rest.json" > "$OUT/perplayer_fit_p2_rest.log" 2>&1 || echo "FAIL p2_rest"
step "p1_ws";   CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models WuSwartzP1 --models-json "$OUT/perplayer_fit_p1_ws.json" > "$OUT/perplayer_fit_p1_ws.log" 2>&1 || echo "FAIL p1_ws"
step "p2_ws";   CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models WuSwartzP2 --models-json "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_fit_p2_ws.log" 2>&1 || echo "FAIL p2_ws"
step "compare"; CUDA_VISIBLE_DEVICES=3 python -u -m scripts.compare_motion_models --models-json "$OUT/perplayer_fit_p1_rest.json" "$OUT/perplayer_fit_p2_rest.json" "$OUT/perplayer_fit_p1_ws.json" "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_compare_motion.log" 2>&1 || echo "FAIL pp compare"
step "sweep";   CUDA_VISIBLE_DEVICES=3 python -u -m scripts.eval_coverage_sweep --models-json "$OUT/perplayer_fit_p1_rest.json" "$OUT/perplayer_fit_p2_rest.json" "$OUT/perplayer_fit_p1_ws.json" "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_eval_coverage_sweep.log" 2>&1 || echo "FAIL pp sweep"
echo "PERPLAYER_DONE $(date '+%F %T')"
