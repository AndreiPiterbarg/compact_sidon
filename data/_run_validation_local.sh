#!/bin/bash
set -u
cd /workspace/sidon-autocorrelation
export PYTHONPATH=/workspace/sidon-autocorrelation:/workspace/sidon-autocorrelation/tests
export PYTHONUNBUFFERED=1
echo "=== START $(date -u +%FT%TZ) ==="
echo '=== d=10 L3 z2_full ==='
python -u tests/lasserre_mosek_tuned.py --d 10 --order 3   --mode z2_full --n-bisect 8   --json data/mosek_d10_l3_z2full.json   || echo 'd=10 FAILED: '$?
echo "=== END $(date -u +%FT%TZ) ==="
touch /workspace/sidon-autocorrelation/data/mosek_validation_done
