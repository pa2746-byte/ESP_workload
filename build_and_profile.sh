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
nvcc -O3 -std=c++17 -lineinfo dummy_transfer_pipeline.cu -o dummy_pipeline -lnvToolsExt
nvcc -O3 -std=c++17 -lineinfo fft_batched_transfer.cu -o fft_batched -lcufft -lnvToolsExt

echo "Running Nsight Systems profiles..."
nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_dummy_per_iter \
  ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_dummy_once \
  ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 0

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_fft_per_iter \
  ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none -o nsys_fft_once \
  ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 0

echo "Extracting nsys trace CSVs..."
nsys stats --report cuda_gpu_trace -o . --format csv nsys_dummy_per_iter.nsys-rep
nsys stats --report cuda_gpu_trace -o . --format csv nsys_fft_per_iter.nsys-rep

echo "Building flow JSONs from nsys traces..."
python3 nsys_trace_to_flow_json.py nsys_dummy_per_iter
python3 nsys_trace_to_flow_json.py nsys_fft_per_iter

echo "Running Nsight Compute (ncu) for SM/L2/DRAM metrics..."
# Use a small iter count — ncu is slow (replays each kernel multiple times)
NCU_ITERS=3

ncu --metrics l1tex__t_bytes,lts__t_bytes,dram__bytes \
    --csv ./dummy_pipeline "${N}" "${NCU_ITERS}" 1 \
    > ncu_dummy_per_iter.csv 2>/dev/null || \
  echo "  [warn] ncu dummy failed — may need sudo or kernel.perf_event_paranoid=0"

ncu --metrics l1tex__t_bytes,lts__t_bytes,dram__bytes \
    --csv ./fft_batched "${FFT_SIZE}" "${BATCH}" "${NCU_ITERS}" 1 \
    > ncu_fft_per_iter.csv 2>/dev/null || \
  echo "  [warn] ncu fft failed — may need sudo or kernel.perf_event_paranoid=0"

echo "Attaching ncu metrics to flow JSONs..."
python3 ncu_attach_metrics.py \
    --ncu  ncu_dummy_per_iter.csv \
    --flow nsys_dummy_per_iter_flow.json \
    --out  nsys_dummy_per_iter_ncu_flow.json

python3 ncu_attach_metrics.py \
    --ncu  ncu_fft_per_iter.csv \
    --flow nsys_fft_per_iter_flow.json \
    --out  nsys_fft_per_iter_ncu_flow.json

echo "Done."
echo "Generated reports:"
echo "  nsys_dummy_per_iter.nsys-rep"
echo "  nsys_fft_per_iter.nsys-rep"
echo "  nsys_dummy_per_iter_flow.json        (nsys only)"
echo "  nsys_dummy_per_iter_ncu_flow.json    (nsys + ncu SM metrics)"
echo "  nsys_fft_per_iter_flow.json          (nsys only)"
echo "  nsys_fft_per_iter_ncu_flow.json      (nsys + ncu SM metrics)"
