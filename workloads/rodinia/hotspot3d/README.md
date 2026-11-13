# Rodinia HotSpot3D source analysis

This second Rodinia example models repeated updates of a three-dimensional
thermal grid. Both `3D.cu` and its included `opt1.cu` are unchanged copies from
[yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/hotspot3D),
commit `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`.
The upstream license is retained in [../LICENSE](../LICENSE).

From the repository root:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/hotspot3d/3D.cu \
  --analysis-config workloads/rodinia/hotspot3d/analysis.json \
  --out-dir results/rodinia_hotspot3d --render
```

This performs Clang source analysis, not GPU execution. Python and a CUDA-capable
Clang parser are required; rendering also needs matplotlib and networkx.

## Default scenario and outputs

The configuration supplies a 64 x 64 x 3 grid and two temperature-update
iterations. Every buffer contains 12,288 floats, or **49,152 bytes**. The source
chooses a 64 x 4 x 1 block and a 1 x 16 x 1 grid, totaling 4,096 threads per launch.
Each thread walks the layers in its column.

The graph has **9 nodes and 11 edges**:

- Two initial uploads: temperature and power.
- Three transfers per iteration: read power, read temperature, write temperature.
- One temperature download, with the source behavior described below.

Output JSON and metadata are in `results/rodinia_hotspot3d/` as
`3D_simulator.json` and `3D_metadata.json`. The image is
`results/rodinia_hotspot3d/graphs/3D_simulator.png`.

Use a different small scenario without editing CUDA:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/hotspot3d/3D.cu \
  --analysis-config workloads/rodinia/hotspot3d/analysis.json \
  --parameter layers=4 --parameter iterations=3 \
  --out-dir results/rodinia_hotspot3d_3_iterations --render
```

For this upstream launch geometry, square dimensions must be a positive multiple
of 64, and layers must be at least two. The original kernel has no general
bounds guard. Unsupported dimensions fail through launch/access/footprint
checks; the analyzer does not pad the grid or silently rewrite the kernel.

## Upstream download behavior

The host swaps `tIn_d` and `tOut_d` after **every** launch. Thus `tIn_d` points
to the newest temperature result, but the upstream download copies `tOut_d`.
The emitted graph preserves that behavior:

- With one iteration, the download reads the initial temperature buffer.
- With two iterations, it reads the first iteration's result.
- In general, it reads the previous iteration's buffer, not the latest result.

This is a finding from source analysis, not a numerical GPU validation. We have
not silently fixed the upstream source. A regression test changes the download
pointer in a temporary copy and verifies that the graph dependency changes to
the latest result. The vendored original remains unchanged.

## New analyzer capabilities and assumptions

The bounded AST evaluator now follows explicitly selected host helper functions,
resolves an explicitly listed included source file, evaluates configured command
line arguments, and handles bounded `for` loops, returns, scalar updates, and
pointer swaps. Each launch's stencil addresses and byte footprints come from
its kernel AST. Repeated neighbor reads count unique bytes per kernel, not
memory instructions or physical DRAM traffic. Buffer reuse retains dependencies
on earlier readers and writers. Only logical buffer dependencies become graph
edges; default-stream scheduling is recorded separately in metadata.

`hotspot_opt1` is **analyzed**, including its allocations, uploads, launch loop,
swaps, download, and frees. It has no hardcoded transfer template. Main's scalar
parameters and launch calculations are also evaluated from source.

Explicit hash-checked host summaries cover `readinput`, `writeoutput`,
`computeTempCPU`, `accuracy`, and `get_time`. File IO, CPU reference computation,
accuracy checking, and timing are outside this graph. Accuracy and time values
remain unknown; no success score or timing is fabricated. File names are
configuration placeholders; no dataset is loaded or written. CUDA cache-policy
selection is recognized but has no modeled cache/bandwidth effect.

Metadata records hashes for **both source units**, the resolved configuration,
launch dimensions, and the summaries used. Changing an analyzed kernel or host
launch loop triggers fresh analysis; changing a summarized helper or pinned
preprocessor directives requires reviewing its contract.

The configured thread budget (65,536 by default, absolute cap 262,144) applies
across **all launches** in one analysis. Each loop has a configured iteration
limit (1,024 by default, maximum 10,000), and aggregate loop iterations are
capped at one million. Large production cases need more scalable analysis.
This does not add support for shared-memory barriers, data-dependent accesses,
arbitrary helper calls, or every Rodinia benchmark. See the parent guide for
other restrictions. Validation was performed with Apple Clang 17, without a GPU
or simulator.

```bash
python3 -B -m unittest discover -s tests -p test_hotspot3d_source_model.py -v
```
