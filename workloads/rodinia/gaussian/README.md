# Rodinia Gaussian elimination: forward-substitution transfer graph

`gaussian.cu` is unchanged from `cuda/gaussian/gaussian.cu` in
[yuhc/gpu-rodinia, commit 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4](https://github.com/yuhc/gpu-rodinia/blob/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/gaussian/gaussian.cu).
The upstream license is preserved in [../LICENSE](../LICENSE).

This third Rodinia case adds **partial, strided buffer updates and in-place
kernel reads/writes across iterations**. The analyzer follows `ForwardSub`,
`Fan1`, and `Fan2` directly from Clang's AST. No workload-name dispatch or
prewritten graph is used. The source is parsed, not executed; no GPU is needed.

## Generate

From the repository root:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/gaussian/gaussian.cu \
  --analysis-config workloads/rodinia/gaussian/analysis.json \
  --out-dir results/rodinia_gaussian --render
```

Python and Clang are required; `--render` also needs matplotlib and networkx.
The default **3 x 3 matrix**, with two elimination iterations and four kernel
launches, produces **20 nodes and 39 edges**:

- [Simulator JSON](../../../results/rodinia_gaussian/gaussian_simulator.json)
- [Metadata](../../../results/rodinia_gaussian/gaussian_metadata.json)
- [DAG image](../../../results/rodinia_gaussian/graphs/gaussian_simulator.png)

Use `--parameter size=5 --out-dir results/rodinia_gaussian_5` for a separate
scenario. Existing results are overwritten only if you reuse their output path.

## Explicit analysis boundary

The entry point is **`ForwardSub`, not `main`**. The configuration supplies
`Size` and distinct host buffers `m`, `a`, and `b` with shapes `[Size,Size]`,
`[Size,Size]`, and `[Size]`. Their float contents remain unknown. This represents
successful input preparation before forward substitution. File parsing, matrix
initialization, CPU back-substitution, and the numerical solution are outside
this graph. It is a GPU-phase graph, not an end-to-end solver performance model.

`ForwardSub` itself is evaluated: allocations, three uploads, the loop of Fan1
and Fan2 launches, synchronization calls, three downloads, and frees all come
from source. `checkCUDAError` has a source-hash-checked success-path summary.
Timing calls produce unknown values; the configured mutable scalar
`totalKernelTime` may receive that unknown value. Timing cannot drive a branch
or address, and no timing measurement or compute latency is inferred.

## Regions and dependencies

For pivot `t` and matrix dimension `N`:

- Fan1 reads matrix `a` at column `t`, rows `t..N-1`, and writes multipliers
  in `m` at column `t`, rows `t+1..N-1`.
- Fan2 reads that multiplier column, the trailing `a[t:N,t:N]` square, and
  `b[t:N]`. It updates `a[t+1:N,t:N]` and `b[t+1:N]` in place.
- All original allocations are downloaded, including untouched cells from
  the uploads and updated regions from different iterations.

`vol` counts **unique bytes accessed by each transfer**, not its bounding span
or repeated memory instructions. For example, Fan1's first multiplier write
in the 3 x 3 case touches two separated floats: **8 bytes**, not the span
between them. Metadata records their half-open byte ranges `[12,16)` and
`[24,28)`; simulator nodes keep the usual four fields.

With `dependency_granularity: byte_regions`, reads wait for the latest writers
of overlapping bytes. Writes wait for overlapping writers and outstanding
readers. Each kernel output also waits for all its input transfers, a
conservative kernel-level policy. Read/read pairs and disjoint regions do not
add memory-hazard edges. Dependencies are not a reproduction of CUDA stream
scheduling. Transitive edges are retained.

In-place access is accepted only when no thread reads bytes written by another
thread and threads do not overlap their writes. Read-after-write within one
thread is currently rejected. Atomics, barriers/shared-memory communication,
aliased kernel arguments, data-dependent addresses, and arbitrary CUDA remain
unsupported. Existing NN/HotSpot3D configurations retain whole-buffer tracking.

The bounded evaluator limits aggregate launched threads (default 65,536) and
region-tracked live device allocations (default and maximum 1 MiB). These are
small correctness cases. Large benchmark inputs need symbolic range analysis.

## Independent source validation

```bash
python3 -B -m unittest discover -s tests -p test_gaussian_source_model.py -v
```

Tests derive expected cell footprints from the row/column formulas above and
expected edges by checking historical overlaps independently of the exporter.
They compare every node, volume, component, edge, and byte range for sizes
2, 3, 4, and 5; check the published graph and metadata; pin the upstream source;
and change a kernel guard to verify that source edits change the graph exactly
as expected. Invalid accesses, cross-thread races, missing initialization,
missing boundary parameters, and region-budget overflow must fail.

This validates logical footprints/dependencies for the tested scenarios. It
does not verify numerical accuracy, successful CUDA execution, simulator
compatibility, or hardware performance.
