#!/bin/bash
# Run this INSIDE the RunPod SSH session.
# It clones the repo, installs deps, builds the GPU kernel, and runs proofs.

set -e

echo "=== SETUP ==="
cd /workspace

# Clone repo (or pull if exists)
if [ -d compact_sidon ]; then
    cd compact_sidon && git pull && cd ..
else
    git clone https://github.com/AndreiPiterbarg/compact_sidon.git 2>/dev/null || {
        echo "Git clone failed. Trying to copy files..."
        echo "Please upload the repo manually."
        exit 1
    }
fi
cd compact_sidon

# Install Python deps
pip install numpy numba 2>/dev/null || pip3 install numpy numba 2>/dev/null

echo "=== BUILDING GPU KERNEL ==="
mkdir -p /tmp/sidon_gpu
cp gpu/cascade_kernel.cu gpu/cascade_host.cu gpu/cascade_kernel.h /tmp/sidon_gpu/
cd /tmp/sidon_gpu

# Detect GPU arch
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "GPU compute capability: sm_${GPU_ARCH}"

nvcc -arch=sm_${GPU_ARCH} -O3 -ftz=false -prec-div=true -prec-sqrt=true -fmad=false \
    -lineinfo cascade_kernel.cu cascade_host.cu -o cascade_prover.exe && echo "BUILD OK" || {
    echo "BUILD FAILED"
    exit 1
}

echo "=== RUNNING PROOFS ==="
cd /workspace/compact_sidon

# Run the CPU cascade proof script
python3 run_proof.py --m 35 --c_targets 1.28,1.30,1.33,1.35,1.37,1.40 --workers $(nproc)

echo "=== DONE ==="
