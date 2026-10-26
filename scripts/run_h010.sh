#!/usr/bin/env bash
# Chained driver for the 0.10 s horizon-window run (ffpc_match).
# Build -> train (MLP, vmax-profile) -> fit physics -> evaluate.
# Stops at the first failure (set -e). Master log: outputs/ffpc_match/run_h010.log
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Log dir follows config.OUTPUTS_DIR so a non-canonical SUVAT-caps run tees its
# logs next to its own checkpoints instead of overwriting the canonical ones.
OUT=$(python -c "from src.config import OUTPUTS_DIR; print(OUTPUTS_DIR)")
mkdir -p "$OUT"

banner () { echo; echo "==================== $1 :: $(date '+%H:%M:%S') ===================="; echo; }

banner "B build_dataset (0.10 window)"
python -m scripts.build_dataset 2>&1 | tee "$OUT/build_h010.log"

banner "C1 train base MLP"
python -m scripts.train 2>&1 | tee "$OUT/train_mlp_h010.log"

banner "C2 train vmax-profile MLP"
python -m scripts.train_profile vmax 2>&1 | tee "$OUT/train_profile_vmax_h010.log"

banner "D fit motion models"
python -m scripts.fit_motion_models 2>&1 | tee "$OUT/fit_motion_h010.log"

banner "E1 evaluate (calibration + heatmap)"
python -m scripts.evaluate 2>&1 | tee "$OUT/evaluate_h010.log"

banner "E2 compare_motion_models (headline + reachable-area)"
python -m scripts.compare_motion_models 2>&1 | tee "$OUT/compare_motion_h010.log"

banner "E3 eval_reachable_area (Δt sweep)"
python -m scripts.eval_reachable_area 2>&1 | tee "$OUT/eval_ra_h010.log"

banner "DONE"
echo "all phases complete"
