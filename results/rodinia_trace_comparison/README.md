# Rodinia trace comparison

Open `report.html` for three side-by-side views per workload: observed GPU
operations, Clang operations grouped by operation_id, and original logical
read/write nodes. SVG and JSON files are also provided individually.

## Findings

- Nearest Neighbor: trace 3 nodes / 2 edges; original Clang 4 / 3;
  grouped Clang 3 / 2. Grouped topology matches exactly.
- HotSpot3D: trace 5 / 4; original Clang 9 / 11; grouped Clang 5 / 6.
  Every source edge is consistent with observed order. Different edge sets
  reflect buffer dependencies versus stream scheduling, not a failed match.
- Both match kernel names/counts, grid/block sizes and explicit copy direction/size.
- NN copies 4,104 bytes in and 2,052 out. HotSpot copies 49,152 bytes in twice
  and 49,152 out once. These are not measurements of kernel DRAM traffic.
- HotSpot's download executes after iteration 2, but source analysis identifies
  iteration 1 as its producer. Trace buffer identity is unavailable here.
- Kernel times: NN 2.336 µs; HotSpot 3.104 and 2.816 µs. Single profiled runs
  on Tesla T4, synthetic small inputs; not representative benchmark estimates.

The extractor validates manifest hashes and source fingerprints, correlates
GPU events to CUDA runtime calls, verifies successful API return values and
serial submissions on one host thread/context/stream, then adds adjacent
same-stream edges. It deliberately rejects more complex captures. No edge is
inferred solely from adjacency across arbitrary streams. Clang operation
matching is positional after checking type, kernel name/launch geometry or
copy direction/size; buffer labels come only from source metadata.

No independent validation of kernel footprint bytes, cache behavior, actual
pointer arguments, numerical correctness, or simulator performance was done.
Original Clang graphs are archived in this bundle's clang_reference directory;
this comparison verifies their source fingerprints but does not rerun Clang.
Device configuration is synthetic, though the predicted launch shapes match.

## Reproduce

From the repository root, with the original downloaded capture bundle:

```bash
python3 -B tools/compare_rodinia_capture.py /path/to/20261027T033003Z \
  --out results/rodinia_trace_comparison
```

Input SQLite hashes and exact metrics are in `comparison.json`. Raw reports,
binaries and input datasets are not republished in this results directory.
The original capture used source revision `4106565d567f39851e0371ab07a4efdeaf3fd3d2`.
