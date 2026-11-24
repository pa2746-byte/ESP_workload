# LU decomposition: original application flow

This case follows the unchanged `lud.cu` `main` with arguments `-s 32`, including
argument parsing, matrix allocation, upload, `lud_cuda`, download, and cleanup.
The original `lud_cuda` launches the diagonal, perimeter, and interior kernels.
`analysis_unit.cu` only includes the original source files into one translation
unit so Clang can see the definitions; it does not provide a substitute `main`.
Original files are byte-for-byte copies of the paths/revision in `upstream.json`.
[The Rodinia license](../LICENSE) applies.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/lud/analysis_unit.cu \
  --analysis-config workloads/rodinia/lud/analysis.json \
  --out-dir results/rodinia_lud --render
```

Use `--parameter dimension=48` for another small scenario. The validated sizes
are 16, 32, and 48 with the upstream 16×16 tile. Other sizes remain subject to
bounds, launch, region, and enumeration checks. The `-s` input path is modeled;
file input (`-i`), optional CPU verification (`-v`), and `TIMING` are outside
this configured scenario.

## What the graph represents

The default has **four kernel launches, 10 nodes, and 19 edges**. All kernels
operate in place on `d_m`, a 32×32 float matrix (4,096 bytes):

- `t0`: upload the whole matrix, 4,096 bytes.
- `t1`, `t2`: first diagonal kernel reads a 1,024-byte tile and writes 960
  bytes. The first row of that tile is left unchanged.
- `t3`, `t4`: perimeter kernel reads 3,072 unique bytes and writes 1,984 bytes
  in the adjacent row and column tiles.
- `t5`, `t6`: interior kernel reads 3,072 unique bytes across perimeter and
  trailing tiles, then writes the 1,024-byte trailing tile.
- `t7`, `t8`: final diagonal kernel reads that 1,024-byte tile and writes
  960 bytes.
- `t9`: download the whole matrix, 4,096 bytes.

Dependencies account for overlapping reads and writes, including unchanged
regions whose most recent writer is still the initial upload. Nodes represent
logical unique footprints, not repeated instructions or measured DRAM bytes.

[Simulator JSON](../../../results/rodinia_lud/analysis_unit_simulator.json),
[metadata](../../../results/rodinia_lud/analysis_unit_metadata.json), and
[image](../../../results/rodinia_lud/graphs/analysis_unit_simulator.png).

## Analysis and validation scope

The complete configured CUDA launch/copy flow is analyzed from `main`.
Four source-hash-checked host summaries model `create_matrix`,
`stopwatch_start`, `stopwatch_stop`, and `get_interval_by_sec`. The matrix
summary requires the actual argument to match the configured size and provides
an appropriately sized host buffer with unknown contents. Timer summaries
provide no timing evidence. Their numerical values do not determine any GPU
operation in this scenario.

Shared-array contents remain unknown; the graph assumes valid synchronization.
Global footprints, bounds, overlapping-region hazards, and launch ordering are
checked. CPU matrix values, shared-memory arithmetic, and numerical LU results
are not executed or validated.

`tests/test_lud_source_model.py` independently builds diagonal, perimeter, and
trailing-square cell sets for 16, 32, and 48. It compares every byte range and
dependency, checks original source hashes, and rejects mismatched host-input
contracts, unsupported/malformed options, and an invalid tile dimension.
No GPU or simulator execution has been performed for this addition.
