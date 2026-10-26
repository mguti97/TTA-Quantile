#!/usr/bin/env bash
# Wait for the two ffpc caps10_12 builds to finish, then build nba alone.
# Sequenced deliberately: build_dataset holds all matches in RAM before saving,
# so ffpc x2 + nba concurrently risks OOM.
cd "$(dirname "$0")/.."
while pgrep -f "scripts.build_dataset" > /dev/null; do sleep 60; done
echo "=== ffpc builds done $(date '+%F %T'); starting nba ===" 
KOGNIA_DATASET=nba_match python -u -m scripts.build_dataset
echo "=== nba exit=$? $(date '+%F %T') ==="
