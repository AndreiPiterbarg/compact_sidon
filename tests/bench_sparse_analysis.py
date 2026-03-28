"""Analyze sparsity of children at d_child=32 to determine if sparse path helps."""
import sys, os, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import _compute_bin_ranges

M = 20
C_TARGET = 1.4

shard_dir = os.path.join(_project_dir, 'data', '_shards_L2')
shard_files = sorted([f for f in os.listdir(shard_dir)
                     if f.startswith('shard_') and f.endswith('.npy')
                     and '.m' not in f])
shard_path = os.path.join(shard_dir, shard_files[0])
l2_surv = np.load(shard_path, mmap_mode='r')[:2000]
l2_surv = np.array(l2_surv)

print(f"Loaded {len(l2_surv)} L2 survivors (d=16)")
print(f"Sample parent[0]: {l2_surv[0]}")
print(f"Sum: {l2_surv[0].sum()}")

# Analyze sparsity
zero_counts = []
for i in range(min(500, len(l2_surv))):
    parent = l2_surv[i]
    nz = np.count_nonzero(parent)
    zero_counts.append(16 - nz)

print(f"\nParent sparsity (d=16):")
print(f"  Zero bins per parent: mean={np.mean(zero_counts):.1f}, "
      f"median={np.median(zero_counts):.0f}, "
      f"min={np.min(zero_counts)}, max={np.max(zero_counts)}")

# Now analyze child sparsity
child_nz_counts = []
for i in range(min(200, len(l2_surv))):
    parent = l2_surv[i]
    result = _compute_bin_ranges(parent, M, C_TARGET, 32, 16)
    if result is None:
        continue
    lo_arr, hi_arr, total = result
    if total == 0:
        continue
    # Sample a few children
    child = np.empty(32, dtype=np.int32)
    for j in range(16):
        child[2*j] = lo_arr[j]
        child[2*j+1] = parent[j] - lo_arr[j]
    child_nz_counts.append(np.count_nonzero(child))
    # Also check mid-range child
    for j in range(16):
        mid = (lo_arr[j] + hi_arr[j]) // 2
        child[2*j] = mid
        child[2*j+1] = parent[j] - mid
    child_nz_counts.append(np.count_nonzero(child))

print(f"\nChild sparsity (d_child=32):")
print(f"  Nonzero bins per child: mean={np.mean(child_nz_counts):.1f}, "
      f"median={np.median(child_nz_counts):.0f}, "
      f"min={np.min(child_nz_counts)}, max={np.max(child_nz_counts)}")
print(f"  Sparsity ratio (zero/total): "
      f"{1 - np.mean(child_nz_counts)/32:.1%}")

# Distribution
from collections import Counter
counts = Counter(child_nz_counts)
print(f"\n  Distribution of nonzero count:")
for k in sorted(counts.keys()):
    print(f"    nnz={k}: {counts[k]} children ({counts[k]/len(child_nz_counts)*100:.1f}%)")
