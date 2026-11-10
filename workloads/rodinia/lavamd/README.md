# LavaMD: configured original application flow

**Input:** Default one box with 100 particles.

**Output:** 10 transfer nodes, 10 dependency edges, 1 kernel launches.

The original main, neighbor-box initialization and CUDA wrapper are followed across their source files. One kernel accumulates forces; global struct reads and writes are modeled while shared-memory values remain opaque. No force values or numerical accuracy are validated. The LP64 record layout is assumed. Larger multi-box scenarios have not been validated.

The source comes from [gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/lavaMD), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. [provenance.json](provenance.json) records hashes; [../LICENSE](../LICENSE) and source notices retain licensing.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/lavamd/main.c \
  --analysis-config workloads/rodinia/lavamd/analysis.json \
  --out-dir results/rodinia_lavamd --render
```

Clang and Python are required; rendering uses matplotlib and NetworkX. No GPU is required. [Simulator JSON](../../../results/rodinia_lavamd/main_simulator.json), [metadata](../../../results/rodinia_lavamd/main_metadata.json), and [PNG](../../../results/rodinia_lavamd/graphs/main_simulator.png) are included.

Volumes represent unique logical byte footprints, not physical memory transactions. This optional analysis mode conservatively adds dependencies from prior CPU-visible downloads to subsequent transfers; these appear as `host_control_dependencies`. Integer bytes are propagated from input and source computations. Unknown kernel branches are accepted only when bounded exploration establishes identical footprints and barrier sequences. CPU numerical contracts, opaque host addresses and shared-memory abstraction do not establish numerical correctness.

`tests/test_five_source_workloads.py` contains the focused independent checks. These additions have **not been run on a GPU locally or validated in the simulator**. Use the [cycle-and-trace notebook](../../../notebooks/rodinia_five_cycles.ipynb) for matched GPU evidence; see [measurement scope](../CYCLES_CAPTURE.md).
