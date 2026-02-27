# CUDA Workload + Nsight Dataflow Workflow Log

This document captures all work completed so far, including iteration steps, command history, issues encountered, and fixes.

## Iteration 1: Create CUDA workloads

### Goal
- Build two CUDA workloads for T4 profiling and dataflow graph generation:
  - Dummy transfer-heavy pipeline (`vector add -> scale -> copy kernel`)
  - Batched FFT pipeline (cuFFT)

### Files created
- `dummy_transfer_pipeline.cu`
- `fft_batched_transfer.cu`

### Key behavior designed
- Two transfer modes in both programs:
  - `mode=0` => transfer once (`H2D once + D2H once`)
  - `mode=1` => transfer every iteration (`H2D + D2H per iteration`)
- NVTX ranges added for clean Nsight timeline segmentation.
- Expected transfer bytes printed by each executable.

## Iteration 2: Add automation script

### File created
- `build_and_profile.sh`

### Purpose
- Compile both workloads.
- Run 4 Nsight Systems profiles:
  - dummy per-iter
  - dummy once
  - FFT per-iter
  - FFT once

### Commands
```bash
chmod +x build_and_profile.sh
./build_and_profile.sh
```

### Permission issue and fix (on VM)
- Error seen:
```bash
-bash: ./build_and_profile.sh: Permission denied
```
- Fix:
```bash
chmod +x build_and_profile.sh
./build_and_profile.sh
```
- Alternative:
```bash
bash build_and_profile.sh
```

## Iteration 3: Git repo setup + push troubleshooting

### Initial commands used
```bash
git init
git add .
git commit -m "Add models"
git branch -M main
git remote add origin https://github.com/pa2746-byte/ESP_workload.git
git push -u origin main
```

### Issue
- Push rejected because remote `main` already had commits.

### Attempted sync command
```bash
git fetch origin
git pull origin main --allow-unrelated-histories
```

### Issue
- Git required explicit reconcile strategy:
```bash
fatal: Need to specify how to reconcile divergent branches.
```

### Correct pull form
```bash
git pull --no-rebase --allow-unrelated-histories origin main
```

### Merge conflict encountered
- Conflict:
```bash
CONFLICT (add/add): Merge conflict in build_and_profile.sh
```

### Conflict resolution option used/recommended
- Keep local file:
```bash
git checkout --ours build_and_profile.sh
git add build_and_profile.sh
git commit -m "Resolve merge conflict in build_and_profile.sh"
git push -u origin main
```

## Iteration 4: Locate and inspect Nsight artifacts

### Reports confirmed in repo
- `nsys_dummy_per_iter.nsys-rep`
- `nsys_dummy_once.nsys-rep`
- `nsys_fft_per_iter.nsys-rep`
- `nsys_fft_once.nsys-rep`

## Iteration 5: Extract Nsight stats (version-specific adjustments)

### Initial command pattern (failed partially)
- Some report names were incompatible with installed Nsight version:
  - `gpu_memcpy_sum` not found
  - `gpu_kern_sum` not found
- `nsys export --sqlite` unsupported in this version.

### What worked
- Nsight version already generated `.sqlite` while running `nsys stats`.

### Discover valid report names
```bash
nsys stats --help-reports | grep -Ei "cuda|kern|mem|nvtx"
```

### Working stats command
```bash
for f in nsys_dummy_per_iter nsys_dummy_once nsys_fft_per_iter nsys_fft_once; do
  nsys stats \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,nvtx_sum,nvtx_pushpop_sum \
    -f csv -o "${f}_stats" "${f}.nsys-rep"
done
```

### Optional trace command (for flow reconstruction)
```bash
for f in nsys_dummy_per_iter nsys_dummy_once nsys_fft_per_iter nsys_fft_once; do
  nsys stats --report cuda_gpu_trace,nvtx_pushpop_trace -f csv -o "${f}_trace" "${f}.nsys-rep"
done
```

### Output verification
```bash
ls -lh *_stats_*.csv *_trace_*.csv *.sqlite
```

## Iteration 6: Build flow JSON from trace

### Target schema reference
- Example reference file used:
  - `n_4_e_4_seed_1_mu_100_sd_50_vseed_24.json`

### Conversion approach
- Parse `*_trace_cuda_gpu_trace.csv`
- Create nodes:
  - `id`, `vol`, `source_id`, `dest_id`
- Create edges:
  - simple temporal chain (`i-1 -> i`)
- Output:
  - `nsys_*_flow.json`

## Iteration 7: Graph rendering issue and resolution

### Error
- Python package installed but Graphviz binary missing:
```bash
ExecutableNotFound: failed to execute PosixPath('dot')
```

### Fix (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y graphviz
dot -V
```

### Conda alternative
```bash
conda install -y -c conda-forge graphviz python-graphviz
dot -V
```

## Iteration 8: Generate flow graphs without profiling

### New file added
- `generate_synthetic_flow.py`

### Purpose
- Generate analytical/synthetic flow JSON directly from workload parameters.
- No Nsight profiling required.
- Same node/edge schema as reference graph format.

### Commands used
```bash
python generate_synthetic_flow.py --workload dummy --mode per_iter --iters 50 --n 16777216
python generate_synthetic_flow.py --workload dummy --mode once --iters 50 --n 16777216
python generate_synthetic_flow.py --workload fft --mode per_iter --iters 50 --fft-size 4096 --batch 256
python generate_synthetic_flow.py --workload fft --mode once --iters 50 --fft-size 4096 --batch 256
```

### Outputs
- `synthetic_dummy_per_iter_flow.json`
- `synthetic_dummy_once_flow.json`
- `synthetic_fft_per_iter_flow.json`
- `synthetic_fft_once_flow.json`

## Iteration 9: Validate generated JSON files

### Files validated
- `synthetic_fft_per_iter_flow.json`
- `synthetic_fft_once_flow.json`
- `synthetic_dummy_per_iter_flow.json`
- `synthetic_dummy_once_flow.json`
- `nsys_fft_per_iter_flow.json`
- `nsys_fft_once_flow.json`
- `nsys_dummy_per_iter_flow.json`
- `nsys_dummy_once_flow.json`

### Validation checks run
- Top-level keys exist:
  - `directed`, `multigraph`, `graph`, `nodes`, `edges`
- Node keys exist:
  - `id`, `vol`, `source_id`, `dest_id`
- Edge keys exist:
  - `source`, `target`
- Edge references are in bounds.

### Result
- All flow JSON files are structurally valid.

## Quick command index

### Build + profile
```bash
./build_and_profile.sh
```

### Nsight report names discovery
```bash
nsys stats --help-reports | grep -Ei "cuda|kern|mem|nvtx"
```

### Stats export
```bash
for f in nsys_dummy_per_iter nsys_dummy_once nsys_fft_per_iter nsys_fft_once; do
  nsys stats \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,nvtx_sum,nvtx_pushpop_sum \
    -f csv -o "${f}_stats" "${f}.nsys-rep"
done
```

### Synthetic flow generation
```bash
python generate_synthetic_flow.py --workload dummy --mode per_iter --iters 50 --n 16777216
python generate_synthetic_flow.py --workload fft --mode per_iter --iters 50 --fft-size 4096 --batch 256
```

## Notes
- Nsight CLI behavior is version-dependent; always verify supported report names with `--help-reports`.
- `nsys stats` on this setup already produced `.sqlite`; explicit `nsys export --sqlite` was not needed/supported.
- `nsys_*_flow.json` reflects measured traces; `synthetic_*_flow.json` reflects analytical workload design.
