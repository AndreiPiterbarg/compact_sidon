#!/bin/bash
set -u
cd /workspace/sidon-autocorrelation
export PYTHONPATH=/workspace/sidon-autocorrelation:/workspace/sidon-autocorrelation/tests
export PYTHONUNBUFFERED=1
mkdir -p /workspace/sidon-autocorrelation/data/mosek_d16_proof
echo "=== START $(date -u +%FT%TZ) ==="
echo "=== d=16 L3 z2_full + pre_elim at t="1.2802" ==="
python -u tests/lasserre_mosek_tuned.py   --d 16 --order 3 --mode z2_full --n-bisect 0   --t-lo 1.2802 --t-hi 1.2802   --pre-elim   --json /workspace/sidon-autocorrelation/data/mosek_d16_l3_z2full.json --proof-dir /workspace/sidon-autocorrelation/data/mosek_d16_proof   || echo 'd=16 FAILED:' $?
echo "=== END $(date -u +%FT%TZ) ==="
touch /workspace/sidon-autocorrelation/data/mosek_d16_done
