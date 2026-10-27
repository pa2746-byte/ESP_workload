This part of the project aims to automatically generate data-dependency graphs from program workloads using profiling and program-analysis techniques.

The immediate focus is on the graph-generation stage. 
Given a workload, the goal is to identify important computation and data-movement events, determine the dependencies between them, 
and export those relationships as a directed graph that can later be consumed by a hardware-performance simulator.

The project starts with small CUDA workloads whose dependency structure is easy to verify manually, including:

Vector addition — a simple input-to-compute-to-output flow\
Two-stage pipeline — a linear producer-consumer dependency\
Fork-join workload — independent branches that later synchronize\
Reduction — hierarchical many-to-one dependencies

These workloads provide controlled test cases for evaluating different graph-generation approaches, 
such as static code analysis, dynamic instrumentation, and profiling tools.

The longer-term goal is to scale the same methodology to more realistic workloads such as matrix multiplication, 
attention, and transformer models, while preserving enough information about dependencies and data movement to support later hardware-independent performance analysis.

## Two independent GPU streams followed by a join

`fork_join_streams.cu` adds a concurrency example without changing the original
four workloads. Two nonblocking worker streams compute `C = A + B` and
`D = A - B`. Each records a completion event. A third stream waits for both
events before computing `E = C * D`.

Input copies finish before the branches start. The host waits for the join
before downloading and checking every output element. Each kernel uses 256
threads per block and 1,048,593 elements, including a partially populated final
block. CUDA errors and numerical mismatches cause a nonzero exit status.

Build and run from the repository root on the NVIDIA machine:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -lineinfo workloads/fork_join_streams.cu -o build/fork_join_streams
./build/fork_join_streams
```

Expected output:

```text
E[0] = 8.000000 (expected 8.000000)
Validation: PASS (0 mismatches across 1048593 elements)
```

Independent streams allow overlap; actual overlap depends on GPU resources
and scheduling. Use Nsight Systems if you want to inspect the execution timeline.
See [NVIDIA's stream/event documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

Generate its transfer graph directly from source, without a GPU or Nsight:

```bash
python3 export_workload_transfers.py --workload fork_join_streams --render
```

Outputs: `results/logical_transfers/fork_join_streams_simulator.json`,
`fork_join_streams_metadata.json`, and `graphs/fork_join_streams_simulator.png`.
The default export still selects the original four examples. The stream example
produces 12 transfers and 13 dependencies, each transfer containing 4,194,372
bytes. Both branches remain independent; the join consumes both outputs.
The analyzer checks explicit stream/event synchronization and rejects missing
ordering of conflicting buffer accesses. No GPU execution is needed for this
analysis. Optional remote compilation and numerical validation remain separate.
