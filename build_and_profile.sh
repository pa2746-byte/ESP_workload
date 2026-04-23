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

DIAMOND_ITERS="${6:-50}"

echo "Building CUDA targets..."
nvcc -O3 -std=c++17 -lineinfo dummy_transfer_pipeline.cu -o dummy_pipeline -lnvToolsExt
nvcc -O3 -std=c++17 -lineinfo fft_batched_transfer.cu    -o fft_batched    -lcufft -lnvToolsExt
nvcc -O3 -std=c++17 -lineinfo diamond_pipeline.cu        -o diamond_pipeline -lnvToolsExt

echo "Running Nsight Systems profiles..."
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
  -o nsys_dummy_per_iter ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
  -o nsys_dummy_once ./dummy_pipeline "${N}" "${DUMMY_ITERS}" 0

nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
  -o nsys_fft_per_iter ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 1

nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
  -o nsys_fft_once ./fft_batched "${FFT_SIZE}" "${BATCH}" "${FFT_ITERS}" 0

nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
  -o nsys_diamond ./diamond_pipeline "${N}" "${DIAMOND_ITERS}"

echo "Extracting nsys trace CSVs..."
for PREFIX in nsys_dummy_per_iter nsys_fft_per_iter nsys_diamond; do
  nsys stats --report cuda_gpu_trace -o . --format csv "${PREFIX}.nsys-rep" 2>/dev/null || \
  nsys stats --report gpukernsum    -o . --format csv "${PREFIX}.nsys-rep" 2>/dev/null || \
  echo "  [warn] nsys stats failed for ${PREFIX} — trace CSV may be stale"
done

echo "Building flow JSONs from nsys traces..."
# Boost clock from: nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader
python3 nsys_trace_to_flow_json.py nsys_dummy_per_iter --gpu-clock-mhz 3105
python3 nsys_trace_to_flow_json.py nsys_fft_per_iter   --gpu-clock-mhz 3105
# Diamond uses stream_dag mode to detect cross-stream dependencies
python3 nsys_trace_to_flow_json.py nsys_diamond --aggregation stream_dag --gpu-clock-mhz 3105

echo "Running Nsight Compute (ncu) for SM/L2/DRAM metrics..."
# Use a small iter count — ncu is slow (replays each kernel multiple times)
NCU_ITERS=3
NCU_METRICS="l1tex__t_bytes.sum,lts__t_bytes.sum,dram__bytes.sum,sm__cycles_elapsed.avg"


ncu --metrics "${NCU_METRICS}" --csv \
    ./dummy_pipeline "${N}" "${NCU_ITERS}" 1 \
    > ncu_dummy_per_iter.csv 2>/dev/null || \
  echo "  [warn] ncu dummy failed"

ncu --metrics "${NCU_METRICS}" --csv \
    ./fft_batched "${FFT_SIZE}" "${BATCH}" "${NCU_ITERS}" 1 \
    > ncu_fft_per_iter.csv 2>/dev/null || \
  echo "  [warn] ncu fft failed"

echo "Attaching ncu metrics to flow JSONs..."
python3 ncu_attach_metrics.py \
    --ncu  ncu_dummy_per_iter.csv \
    --flow nsys_dummy_per_iter_flow.json \
    --out  nsys_dummy_per_iter_ncu_flow.json || \
  echo "  [skip] dummy ncu attach failed — run ncu manually first"

python3 ncu_attach_metrics.py \
    --ncu  ncu_fft_per_iter.csv \
    --flow nsys_fft_per_iter_flow.json \
    --out  nsys_fft_per_iter_ncu_flow.json || \
  echo "  [skip] fft ncu attach failed — run ncu manually first"

echo "Done."
echo "Generated reports:"
echo "  nsys_dummy_per_iter.nsys-rep"
echo "  nsys_fft_per_iter.nsys-rep"
echo "  nsys_diamond.nsys-rep"
echo "  nsys_dummy_per_iter_flow.json          (nsys only, linear)"
echo "  nsys_dummy_per_iter_ncu_flow.json      (nsys + ncu SM metrics)"
echo "  nsys_fft_per_iter_flow.json            (nsys only, linear)"
echo "  nsys_fft_per_iter_ncu_flow.json        (nsys + ncu SM metrics)"
echo "  nsys_diamond_stream_dag_flow.json      (diamond DAG topology)"
