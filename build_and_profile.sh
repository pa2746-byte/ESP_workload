#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build_and_profile.sh
#   ./build_and_profile.sh <N> <dummy_iters> <fft_size> <batch> <fft_iters>
#
# Defaults are chosen to be practical on a T4.

N="${1:-16777216}"
DUMMY_ITERS="${2:-50}"
FFT_SIZE="${3:-4096}"
BATCH="${4:-256}"
FFT_ITERS="${5:-50}"

echo "Building CUDA targets..."
nvcc -O3 -lineinfo dummy_transfer_pipeline.cu -o dummy_pipeline -lnvToolsExt
nvcc -O3 -lineinfo fft_batched_transfer.cu -o fft_batched -lcufft -lnvToolsExt

echo "Running Nsight Systems profiles..."
nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_dummy_per_iter \
  ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_dummy_once \
  ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 0

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_fft_per_iter \
  ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_fft_once \
  ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 0

echo "Done."
echo "Generated reports:"
echo "  nsys_dummy_per_iter.qdrep / .nsys-rep"
echo "  nsys_dummy_once.qdrep     / .nsys-rep"
echo "  nsys_fft_per_iter.qdrep   / .nsys-rep"
echo "  nsys_fft_once.qdrep       / .nsys-rep"
