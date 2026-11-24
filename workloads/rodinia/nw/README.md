# Needleman–Wunsch: original application flow

This case starts at the unchanged `needle.cu` `main`, follows `runTest`, and
includes both wavefront launch loops and all three host/device copies. It uses
original `needle_kernel.cu` and `needle.h`, not a replacement kernel harness.
Source paths, revision, and byte hashes are recorded in `upstream.json`;
[the Rodinia license](../LICENSE) applies.

The default arguments are `32 10`: two sequences of length 32, penalty 10.
The allocated matrices include the boundary row and column, so each has
33 × 33 integers (4,356 bytes). There are three kernel launches: two
`needle_cuda_shared_1` launches with one and two blocks, followed by one
`needle_cuda_shared_2` launch with one block. Blocks contain 16 threads.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/nw/needle.cu \
  --analysis-config workloads/rodinia/nw/analysis.json \
  --out-dir results/rodinia_nw --render
```

Use `--parameter dimension=48` to analyze a different scenario. Dimensions
must be positive multiples of the upstream tile size (16) and fit the
analyzer's enumeration/region budgets. Penalty changes numerical results,
but not this application's buffer footprints.

## What the graph represents

The default has **12 nodes and 21 edges**:

- `t0`: upload the reference scores, 4,356 bytes.
- `t1`: upload the initialized dynamic-programming matrix, 4,356 bytes.
- `t2`, `t3`, `t4`: first wavefront reads 1,024 reference bytes and 132 matrix
  boundary bytes, then writes 1,024 matrix bytes.
- `t5`, `t6`, `t7`: second wavefront reads 2,048 reference bytes and 260 unique
  boundary bytes, then writes 2,048 matrix bytes. The two tiles share one
  corner element; it is counted once in the unique global footprint.
- `t8`, `t9`, `t10`: final wavefront reads 1,024 reference bytes and 132
  boundary bytes, then writes 1,024 matrix bytes.
- `t11`: download the complete 4,356-byte result matrix.

The graph's reference reads can appear at the same dependency level because
they do not depend on the earlier matrix updates. The original application
still launches kernels sequentially on the default stream. Operation ordering
is recorded separately in metadata; graph levels are not a CUDA timeline.

[Simulator JSON](../../../results/rodinia_nw/needle_simulator.json),
[metadata](../../../results/rodinia_nw/needle_metadata.json), and
[image](../../../results/rodinia_nw/graphs/needle_simulator.png).

## Analysis and validation scope

The analyzer evaluates host allocation sizes, scalar controls, initialization
loop bounds, launch geometry, global address expressions, and cleanup. Host
array contents and `rand()` values remain unknown. The original `maximum`
device helper is checked structurally for a body that uses only numeric local
variables, with no memory accesses or calls. Its unknown inputs yield an
unknown scalar result. Kernel shared-array contents also stay unknown.

This preserves the complete configured CUDA flow without pretending to execute
CPU/GPU numerical work. Unknown data still cannot select a global-memory
address or a launch/control path. Optional upstream `TRACEBACK` and `TIMING`
preprocessor paths are not enabled. This is not support for every build mode.

`tests/test_nw_source_model.py` independently computes tile diagonals, interiors,
left/top/corner boundaries, and historical byte-overlap dependencies for 16,
32, and 48. It also checks source provenance, lexical scope restoration,
invalid dimensions, out-of-bounds accesses, hidden calls in scalar helpers,
and unknown host data controlling launch counts.

No CUDA execution, numerical alignment validation, hardware traffic measurement,
or simulator performance validation has been performed for this addition.
