# K-means: configured original application flow

**Input:** One float32 point [2.0], one feature and one cluster, binary input.

**Output:** 17 transfer nodes, 25 dependency edges, 3 kernel launches.

The original main, loader, cluster sweep and convergence loop are followed. A mapping kernel and two membership kernels execute: membership changes from -1 to 0, then converges. This fixture uses exactly representable values. Known floating state is not a general proof of GPU floating-point equivalence, FMA behavior or arbitrary convergence. Analysis models legacy texture references; the Colab build substitutes ordinary loads because modern CUDA removed that API. Logical accesses match, but the cache path and timing can differ. Device ordinal 1 in the source is remapped to Colab ordinal 0 only in staged build copies.

The source comes from [gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/kmeans), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. [provenance.json](provenance.json) records hashes; [../LICENSE](../LICENSE) and source notices retain licensing.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/kmeans/kmeans_cuda.cu \
  --analysis-config workloads/rodinia/kmeans/analysis.json \
  --out-dir results/rodinia_kmeans --render
```

Clang and Python are required; rendering uses matplotlib and NetworkX. No GPU is required. [Simulator JSON](../../../results/rodinia_kmeans/kmeans_cuda_simulator.json), [metadata](../../../results/rodinia_kmeans/kmeans_cuda_metadata.json), and [PNG](../../../results/rodinia_kmeans/graphs/kmeans_cuda_simulator.png) are included.

Volumes represent unique logical byte footprints, not physical memory transactions. This optional analysis mode conservatively adds dependencies from prior CPU-visible downloads to subsequent transfers; these appear as `host_control_dependencies`. Integer bytes are propagated from input and source computations. Unknown kernel branches are accepted only when bounded exploration establishes identical footprints and barrier sequences. CPU numerical contracts, opaque host addresses and shared-memory abstraction do not establish numerical correctness.

`tests/test_five_source_workloads.py` contains the focused independent checks. These additions have **not been run on a GPU locally or validated in the simulator**. Use the [cycle-and-trace notebook](../../../notebooks/rodinia_five_cycles.ipynb) for matched GPU evidence; see [measurement scope](../CYCLES_CAPTURE.md).
