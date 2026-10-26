#!/usr/bin/env bash
# Phase 2 of the caps10_12 re-run: all trainings. Fan out over the 4 idle GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12

# ffpc base MLP (canonical seed 42 -> mlp.pt) + seed band, one GPU each
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.train                              > "$OUT/train_mlp.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.train --seed 43 --out-suffix _seed43 > "$OUT/train_mlp_seed43.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.train --seed 44 --out-suffix _seed44 > "$OUT/train_mlp_seed44.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u -m scripts.train --seed 45 --out-suffix _seed45 > "$OUT/train_mlp_seed45.log" 2>&1 &

# bdb train (small, CPU-light dataset; share GPU1 -- it fits easily)
CUDA_VISIBLE_DEVICES=1 KOGNIA_DATASET=bdb_match python -u -m scripts.train     > outputs/bdb_match_caps10_12/train_mlp.log 2>&1 &

# leakage-free profile MLP needs the vmaxtrain npz; start after base frees GPU0
wait $P0
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.train_profile vmax > "$OUT/train_profile_vmax.log" 2>&1 &

wait
echo "PHASE2_FFPC_BDB_DONE rc=$? $(date '+%F %T')"
