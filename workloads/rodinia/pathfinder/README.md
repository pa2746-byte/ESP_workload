# Rodinia Pathfinder

`pathfinder.cu` is unchanged from `cuda/pathfinder/pathfinder.cu` at
[yuhc/gpu-rodinia commit 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4](https://github.com/yuhc/gpu-rodinia/blob/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/pathfinder/pathfinder.cu).
Original bytes, including mixed line endings, are preserved. See
[the upstream license](../LICENSE).

Pathfinder uses dynamic programming to propagate path costs across a grid,
choosing among three neighboring costs in the previous row. It adds integer
buffers, row-window reads, host-copy offsets, and tiled repeated launches.
The analyzer follows the original `main`, `run`, `calc_path`, and
`dynproc_kernel`; there is no workload-specific graph template.

## Generate without a GPU

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/pathfinder/pathfinder.cu \
  --analysis-config workloads/rodinia/pathfinder/analysis.json \
  --out-dir results/rodinia_pathfinder --render
```

Python and Clang generate JSON; rendering also needs matplotlib and networkx.
The default is **5 rows x 16 columns, pyramid height 2**. The source makes two
launches of one 256-thread block, each advancing two rows. It produces
**9 nodes and 11 edges**:

- [DAG image](../../../results/rodinia_pathfinder/graphs/pathfinder_simulator.png)
- [Simulator JSON](../../../results/rodinia_pathfinder/pathfinder_simulator.json)
- [Metadata](../../../results/rodinia_pathfinder/pathfinder_metadata.json)

For another input, append `--parameter rows=6 --parameter cols=257
--parameter pyramid=2 --out-dir results/rodinia_pathfinder_257`, replacing the
original output directory. Separate directories preserve earlier graphs.

## Expected transfers

- `t0`: upload the initial cost row, **64 bytes**, into `gpuResult[0]`.
- `t1`: upload the remaining four grid rows, **256 bytes**, into `gpuWall`.
  This copy starts at `data + cols` in host memory.
- `t2`: first launch reads the cost row, **64 bytes**; `t3`: reads the first
  two wall rows, **128 bytes**; `t4`: writes the next cost row, **64 bytes**.
- `t5`: second launch reads that result, **64 bytes**; `t6`: reads the next
  two wall rows, **128 bytes**; `t7`: writes the final cost row, **64 bytes**.
- `t8`: download the final cost row, **64 bytes**.

The 11 edges are `0→2`, `0→7`, `1→3`, `1→6`, `2→4`, `2→7`,
`3→4`, `4→5`, `5→7`, `6→7`, and `7→8`.
The two wall reads have different byte ranges: `[0,128)` and `[128,256)`.
Changing the number of rows or pyramid height changes the launch count and
consumed row windows directly from the source.

## Assumptions and limits

The source-hash-checked `init` summary supplies configured rows/columns/pyramid
height and host buffer shapes. It assumes valid arguments and successful
allocation/initialization. The random dataset is **not generated** and costs
stay unknown. Its auxiliary `wall` pointer table is a cleanup-only opaque host
handle with zero transferable capacity; it is never treated as device data.
Device count and limits are configured assumptions. Other host operations,
including copies, pointer-array indexing, swaps, launches, and cleanup, are
analyzed from source.

Shared `prev` and `result` arrays use the explicit `opaque_values` model:
bounds are checked but their contents remain unknown. MIN expressions selecting
only scalar values can remain unknown because their result changes cost values,
not addresses or control flow. If such a value is used as an address or branch
condition, analysis fails. Conditional expressions cannot hide memory accesses
or other side effects in an unevaluated branch.

Shared transfers, initialization, races, and barrier correctness are not
simulated or validated. The global-footprint graph assumes valid internal
synchronization. It contains no internal GPU-transfer nodes, compute costs,
numerical solution, or physical DRAM measurements. Unique byte footprints are
deduplicated across overlapping tiles. Small-input thread/loop/region budgets
apply; no GPU or simulator execution occurs.

## Validation

```bash
python3 -B -m unittest discover -s tests -p test_tiled_rodinia_source_model.py -v
```

An independent row-region schedule and tile-interval calculation check all
nodes, volumes, component endpoints, edges, byte ranges, and launch dimensions
for widths 1, 16, 252, 257, and 511, including multiple blocks and partial final
launches. Tests check source hashes and committed graphs, change the download
pointer and verify its dependency, and reject stale input contracts, invalid
host-copy offsets, invalid shared/global accesses, unknown control/addresses,
and invalid geometry/budgets.
