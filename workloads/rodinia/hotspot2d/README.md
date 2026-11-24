# Rodinia HotSpot 2D

`hotspot.cu` is unchanged from `cuda/hotspot/hotspot.cu` at
[yuhc/gpu-rodinia commit 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4](https://github.com/yuhc/gpu-rodinia/blob/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/hotspot/hotspot.cu).
The original file bytes, including line endings, are preserved. See
[the upstream license](../LICENSE).

This is the tiled **2D** thermal simulation, separate from HotSpot3D. The
analyzer follows `main`, `run`, `compute_tran_temp`, and `calculate_temp` from
source. Allocations, uploads, tile geometry, repeated launches, buffer swaps,
and the final download come from the original program.

## Generate without a GPU

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/hotspot2d/hotspot.cu \
  --analysis-config workloads/rodinia/hotspot2d/analysis.json \
  --out-dir results/rodinia_hotspot2d --render
```

Python and Clang generate JSON; matplotlib and networkx are needed for rendering.
The default is a **16 x 16 grid, pyramid height 2, and 4 simulation steps**.
The source groups these into two kernel launches, each using four 16 x 16 thread
blocks. It produces **9 transfer nodes and 11 dependencies**. Every transfer
contains **1,024 bytes**, the unique footprint of 256 float cells.

- [DAG image](../../../results/rodinia_hotspot2d/graphs/hotspot_simulator.png)
- [Simulator JSON](../../../results/rodinia_hotspot2d/hotspot_simulator.json)
- [Metadata](../../../results/rodinia_hotspot2d/hotspot_metadata.json)

Use `--parameter size=17 --parameter pyramid=2 --parameter iterations=3
--out-dir results/rodinia_hotspot2d_17` to select another scenario (append those
options to the command above, replacing its output directory). Keep separate
output directories to retain earlier scenarios.

## Expected graph

The two uploads initialize temperature and power. Each launch reads both the
current temperature buffer and power, then writes the other temperature buffer.
The second launch reuses the initial temperature allocation; its write waits
for the earlier accesses to those bytes. The source returns the destination
buffer index, so the download consumes the **latest** temperature result.

The default node sequence is:

- `t0`, `t1`: upload temperature and power.
- `t2`, `t3`: first launch reads temperature and power; `t4`: writes temperature.
- `t5`, `t6`: second launch reads temperature and power; `t7`: writes temperature.
- `t8`: download the second launch's temperature result.

Edges are `0→2`, `0→7`, `1→3`, `1→6`, `2→4`, `2→7`, `3→4`,
`4→5`, `5→7`, `6→7`, and `7→8`. Transitive edges are retained.

## Assumptions and limits

`readinput` and `writeoutput` have source-hash-checked summaries. Input loading
is assumed successful; the named temperature/power files are not opened and
numerical contents stay unknown. Host allocations are assumed successful.
Configured device limits are assumptions, not measured hardware capabilities.

`shared_memory: opaque_values` explicitly enables global-footprint analysis
of the shared arrays. Their bounds are checked, but every shared value remains
unknown, even after assignment. Shared transfers, shared initialization/races,
and barrier correctness are **not simulated or validated**. This graph assumes
the kernel's internal synchronization is valid. It records only host/global
memory transfers and inter-kernel dependencies, with conservative kernel
input-to-output edges. It contains no internal GPU-transfer nodes.

This is sufficient here because global addresses and access guards depend on
configured dimensions, block/thread indices, and loop counters. A shared or
device value used to choose an address or branch remains unsupported. Graph
volumes count unique bytes per kernel, not repeated loads by overlapping blocks
or physical DRAM/cache traffic. No numerical accuracy or runtime is inferred.

Small inputs are bounded by the configured thread/loop budgets and 1 MiB live
device allocation limit for byte-region tracking. No GPU or simulator is run.

## Validation

```bash
python3 -B -m unittest discover -s tests -p test_tiled_rodinia_source_model.py -v
```

Independent grid-region and tile-interval formulas check every node, volume,
component, edge, byte range, and launch geometry for grids 1, 12, 16, 17, and 25,
with pyramid heights 1–3 and full/partial final iterations. The tests also check
source fingerprints, saved artifacts, source renaming, and rejection cases.
These expectations are test-only; the generator has no HotSpot graph template.
