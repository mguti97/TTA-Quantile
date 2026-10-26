#!/usr/bin/env bash
# Profile seed band (43-46; canonical = seed 42) + the evals that needed it.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/ffpc_match_caps10_12
TP () { CUDA_VISIBLE_DEVICES=$1 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.train_profile vmax --seed $2 --out-suffix vmaxtrain_seed$2 > "$OUT/train_profile_seed$2.log" 2>&1; }
TP 0 43 & TP 2 44 & TP 3 45 & wait
TP 0 46
echo "==== bootstrap_ci retry :: $(date '+%F %T') ===="
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.bootstrap_ci > "$OUT/bootstrap_ci.log" 2>&1 || echo "FAIL bootstrap_ci"
echo "==== audit_structure retry :: $(date '+%F %T') ===="
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u -m scripts.audit_structure > "$OUT/audit_structure_full.log" 2>&1 || echo "FAIL audit"
echo "==== aggregate_seeds :: $(date '+%F %T') ===="
CUDA_VISIBLE_DEVICES=0 KOGNIA_VMAX_SCOPE=train_split python -u rebuttal/aggregate_seeds.py > "$OUT/aggregate_seeds.log" 2>&1 || echo "FAIL aggregate"
echo "PROFILE_SEEDS_DONE $(date '+%F %T')"
