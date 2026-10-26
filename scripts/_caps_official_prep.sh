#!/usr/bin/env bash
# Everything still needed before the caps10_12 numbers could go official.
# All outputs stay in caps-tagged paths; paper/ and rebuttal/ untouched.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }

step "cross_eval 3x3 on NEW split"
for TR in ffpc_match bdb_match nba_match; do for TE in ffpc_match bdb_match nba_match; do
  CUDA_VISIBLE_DEVICES=3 python -u -m scripts.cross_eval --train $TR --test $TE > "outputs/cross_eval_${TR%_match}_${TE%_match}_caps10_12.log" 2>&1 || echo "FAIL cross $TR->$TE"
done; done

step "cross_sport_area_table: bdb destination"
CUDA_VISIBLE_DEVICES=3 KOGNIA_DATASET=bdb_match python -u -m scripts.cross_sport_area_table > outputs/cross_sport_area_bdb_caps10_12.log 2>&1 || echo "FAIL area bdb"
step "cross_sport_area_table: nba destination"
CUDA_VISIBLE_DEVICES=3 KOGNIA_DATASET=nba_match python -u -m scripts.cross_sport_area_table > outputs/cross_sport_area_nba_caps10_12.log 2>&1 || echo "FAIL area nba"

step "no-filter area sweep"
CUDA_VISIBLE_DEVICES=3 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_coverage_sweep --mlp-path "$OUT/mlp_nofilter.pt" > "$OUT/nofilter_area_sweep.log" 2>&1 || echo "FAIL nofilter sweep"

step "per-player physics fits (rest groups)"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models ConstSpeedP1 ConstAccelVmaxP1 TwoSegmentP1 FujimuraSugiharaP1 --models-json "$OUT/perplayer_fit_p1_rest.json" > "$OUT/perplayer_fit_p1_rest.log" 2>&1 || echo "FAIL p1_rest"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models ConstSpeedP2 ConstAccelP2 ConstAccelVmaxP2 TwoSegmentP2 FujimuraSugiharaP2 --models-json "$OUT/perplayer_fit_p2_rest.json" > "$OUT/perplayer_fit_p2_rest.log" 2>&1 || echo "FAIL p2_rest"
step "per-player physics fits (WuSwartz, slow)"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models WuSwartzP1 --models-json "$OUT/perplayer_fit_p1_ws.json" > "$OUT/perplayer_fit_p1_ws.log" 2>&1 || echo "FAIL p1_ws"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.fit_motion_models --hit-ratio-min 0.99 --models WuSwartzP2 --models-json "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_fit_p2_ws.log" 2>&1 || echo "FAIL p2_ws"

step "per-player compare + coverage sweep"
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.compare_motion_models --models-json "$OUT/perplayer_fit_p1_rest.json" "$OUT/perplayer_fit_p2_rest.json" "$OUT/perplayer_fit_p1_ws.json" "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_compare_motion.log" 2>&1 || echo "FAIL pp compare"
CUDA_VISIBLE_DEVICES=3 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.eval_coverage_sweep --models-json "$OUT/perplayer_fit_p1_rest.json" "$OUT/perplayer_fit_p2_rest.json" "$OUT/perplayer_fit_p1_ws.json" "$OUT/perplayer_fit_p2_ws.json" > "$OUT/perplayer_eval_coverage_sweep.log" 2>&1 || echo "FAIL pp sweep"
echo "OFFICIAL_PREP_DONE $(date '+%F %T')"
