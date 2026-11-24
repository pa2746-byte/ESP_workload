# Particle Filter: configured original application flow

**Input:** 16 × 16 synthetic video, two frames and one particle.

**Output:** 14 transfer nodes, 17 dependency edges, 1 kernel launches.

The original application and frame loop are retained. Source-hash-checked contracts abstract video generation, random numeric helpers and likelihood arithmetic. CPU-only data-dependent addresses are opaque and do not establish CPU memory safety. Both paths of the GPU comparison are explored: for one particle they select the same index and have identical footprints. More particles can produce different footprints and are rejected. The initial uploads of xj/yj are preserved even though the kernel overwrites their contents.

The source comes from [gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/particlefilter), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. [provenance.json](provenance.json) records hashes; [../LICENSE](../LICENSE) and source notices retain licensing.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/particlefilter/ex_particle_CUDA_naive_seq.cu \
  --analysis-config workloads/rodinia/particlefilter/analysis.json \
  --out-dir results/rodinia_particlefilter --render
```

Clang and Python are required; rendering uses matplotlib and NetworkX. No GPU is required. [Simulator JSON](../../../results/rodinia_particlefilter/ex_particle_CUDA_naive_seq_simulator.json), [metadata](../../../results/rodinia_particlefilter/ex_particle_CUDA_naive_seq_metadata.json), and [PNG](../../../results/rodinia_particlefilter/graphs/ex_particle_CUDA_naive_seq_simulator.png) are included.

Volumes represent unique logical byte footprints, not physical memory transactions. This optional analysis mode conservatively adds dependencies from prior CPU-visible downloads to subsequent transfers; these appear as `host_control_dependencies`. Integer bytes are propagated from input and source computations. Unknown kernel branches are accepted only when bounded exploration establishes identical footprints and barrier sequences. CPU numerical contracts, opaque host addresses and shared-memory abstraction do not establish numerical correctness.

`tests/test_five_source_workloads.py` contains the focused independent checks. These additions have **not been run on a GPU locally or validated in the simulator**. Use the [cycle-and-trace notebook](../../../notebooks/rodinia_five_cycles.ipynb) for matched GPU evidence; see [measurement scope](../CYCLES_CAPTURE.md).
