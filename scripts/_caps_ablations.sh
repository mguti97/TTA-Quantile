#!/usr/bin/env bash
# Ablations under caps10_12 (capacity + data-scaling), defaults as in the old
# tau0.01_arealoss runs. NOT official until approved.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
step () { echo; echo "==== $1 :: $(date '+%F %T') ===="; }
step "ablate_capacity"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.ablate_capacity > "$OUT/ablate_capacity.log" 2>&1 || echo "FAIL capacity"
step "ablate_data_scaling"
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.ablate_data_scaling > "$OUT/ablate_data_scaling.log" 2>&1 || echo "FAIL data_scaling"
echo "ABLATIONS_DONE $(date '+%F %T')"
