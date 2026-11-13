# BFS: configured original application flow

**Input:** Four-node directed chain, source vertex 0.

**Output:** 58 transfer nodes, 209 dependency edges, 8 kernel launches.

The stop flag, frontier masks and costs are evaluated from the fixed input. The graph contains four traversal rounds and eight launches. Concurrent conflicting writes remain unsupported: this is not unrestricted support for arbitrary input graphs. The original application leaves the one-byte stop allocation live until process exit; metadata records it.

The source comes from [gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/bfs), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. [provenance.json](provenance.json) records hashes; [../LICENSE](../LICENSE) and source notices retain licensing.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/bfs/bfs.cu \
  --analysis-config workloads/rodinia/bfs/analysis.json \
  --out-dir results/rodinia_bfs --render
```

Clang and Python are required; rendering uses matplotlib and NetworkX. No GPU is required. [Simulator JSON](../../../results/rodinia_bfs/bfs_simulator.json), [metadata](../../../results/rodinia_bfs/bfs_metadata.json), and [PNG](../../../results/rodinia_bfs/graphs/bfs_simulator.png) are included.

Volumes represent unique logical byte footprints, not physical memory transactions. This optional analysis mode conservatively adds dependencies from prior CPU-visible downloads to subsequent transfers; these appear as `host_control_dependencies`. Integer bytes are propagated from input and source computations. Unknown kernel branches are accepted only when bounded exploration establishes identical footprints and barrier sequences. CPU numerical contracts, opaque host addresses and shared-memory abstraction do not establish numerical correctness.

`tests/test_five_source_workloads.py` contains the focused independent checks. These additions have **not been run on a GPU locally or validated in the simulator**. Use the [cycle-and-trace notebook](../../../notebooks/rodinia_five_cycles.ipynb) for matched GPU evidence; see [measurement scope](../CYCLES_CAPTURE.md).
