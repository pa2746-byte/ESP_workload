# B+tree: configured original application flow

**Input:** One key (0), one lookup, order 4, explicit query random draw 0, and newline on stdin.

**Output:** 12 transfer nodes, 10 dependency edges, 1 kernel launches.

The original main and CUDA lookup wrapper are retained. The leaf has height zero, so this is a leaf lookup, not a multi-level traversal. Hash-checked insert/height/serialization contracts describe CPU preparation for this exact input; known serialized byte ranges are independently checked by executing the original CPU code with GPU calls stubbed. No graph is supplied by these contracts: the kernel accesses and graph are still evaluated from source. Arbitrary trees and random queries are unsupported. A documented source correction changes commandPointer != newline to *commandPointer != newline; this preserves the intended parser behavior for the fixture. The capture build supplies the explicit rand() value rather than relying on platform-specific random sequences.

The source comes from [gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/b+tree), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. [provenance.json](provenance.json) records hashes; [../LICENSE](../LICENSE) and source notices retain licensing.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/btree/main.c \
  --analysis-config workloads/rodinia/btree/analysis.json \
  --out-dir results/rodinia_btree --render
```

Clang and Python are required; rendering uses matplotlib and NetworkX. No GPU is required. [Simulator JSON](../../../results/rodinia_btree/main_simulator.json), [metadata](../../../results/rodinia_btree/main_metadata.json), and [PNG](../../../results/rodinia_btree/graphs/main_simulator.png) are included.

Volumes represent unique logical byte footprints, not physical memory transactions. This optional analysis mode conservatively adds dependencies from prior CPU-visible downloads to subsequent transfers; these appear as `host_control_dependencies`. Integer bytes are propagated from input and source computations. Unknown kernel branches are accepted only when bounded exploration establishes identical footprints and barrier sequences. CPU numerical contracts, opaque host addresses and shared-memory abstraction do not establish numerical correctness.

`tests/test_five_source_workloads.py` contains the focused independent checks. These additions have **not been run on a GPU locally or validated in the simulator**. Use the [cycle-and-trace notebook](../../../notebooks/rodinia_five_cycles.ipynb) for matched GPU evidence; see [measurement scope](../CYCLES_CAPTURE.md).
