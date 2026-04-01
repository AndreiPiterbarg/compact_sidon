#!/bin/bash
# build.sh — Build the Sidon cascade GPU prover for NVIDIA H100 (sm_90).
#
# Usage:
#   ./build.sh              # release build (default)
#   ./build.sh debug        # debug build (enables printf tracing)
#   ./build.sh trace        # release with enumeration trace (no debug printf)
#
# Output: cascade_prover  (Linux ELF binary)
#
# This is a rigorous mathematical proof kernel.
# Compiler flags enforce EXACT arithmetic (no fast-math, no FMA, no FTZ).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure nvcc is on PATH (RunPod PyTorch images put it in /usr/local/cuda/bin)
if ! command -v nvcc &>/dev/null; then
    for cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$cuda_dir/nvcc" ]; then
            export PATH="$cuda_dir:$PATH"
            echo "Found nvcc at $cuda_dir"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit."
    exit 1
fi
echo "nvcc: $(nvcc --version | grep release)"

# Detect GPU architecture
ARCH="sm_90"  # H100 default
if command -v nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        ARCH="sm_${COMPUTE_CAP}"
        echo "Detected GPU compute capability: ${COMPUTE_CAP} -> ${ARCH}"
    fi
fi

# Base flags: strict arithmetic for correctness
NVCC_FLAGS=(
    -arch="$ARCH"
    -O3
    -ftz=false          # no flush-to-zero
    -prec-div=true      # precise division
    -prec-sqrt=true     # precise square root
    -fmad=false         # no fused multiply-add (prevents rounding surprises)
    -lineinfo           # debug symbols for profiling
    -Xcompiler -O3      # host compiler optimization
)

# Build mode
MODE="${1:-release}"
case "$MODE" in
    debug)
        NVCC_FLAGS+=(-DDEBUG)
        echo "Building in DEBUG mode (device printf enabled)"
        ;;
    trace)
        NVCC_FLAGS+=(-DTRACE)
        echo "Building in TRACE mode (enumeration completeness verification)"
        ;;
    release)
        # Traces off by default (opt-in via -DTRACE).  No printf overhead.
        echo "Building in RELEASE mode"
        ;;
    *)
        echo "Unknown mode: $MODE (use: release, debug, trace)"
        exit 1
        ;;
esac

OUTPUT="cascade_prover"

echo "  arch:   $ARCH"
echo "  output: $OUTPUT"
echo "  flags:  ${NVCC_FLAGS[*]}"
echo ""

nvcc "${NVCC_FLAGS[@]}" cascade_kernel.cu cascade_host.cu -o "$OUTPUT"

echo ""
echo "BUILD OK: $OUTPUT"
ls -lh "$OUTPUT"
