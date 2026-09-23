# ESP Workload Graphs

## Project goal

Generate workload graphs that describe **data transfers and their dependencies**.
These graphs are inputs to a simulator, which uses them to produce performance
metrics.

The intended workflow is:

**CUDA workload → transfer/dependency graph → simulator → performance metrics**

The graph describes the workload's data requirements. The simulator determines
how its modeled architecture executes those transfers. Whether the resulting
metrics include computation time depends on the simulator's compute model.

## Graph representation

The target format follows the existing JSON examples supplied as simulator inputs:

- Top-level fields: `directed`, `multigraph`, `graph`, `nodes`, and `edges`.
- Each node represents a transfer and contains `id`, `vol`, `src_comp`, and
  `dest_comp`.
- `vol` is the number of bytes transferred.
- `cpu` represents the host, `mem` accelerator-accessible memory, and `acc`
  accelerator compute. Component names may be mapped to numbered instances
  when required by a simulator architecture.
- Each edge contains `source` and `target` node IDs. It expresses a dependency:
  the target transfer waits for the source transfer to finish.

Logical dependencies must preserve independent branches. An observed ordering
on a CUDA stream is not, by itself, proof that one operation needs another's data.

## Why these four workloads?

These small, deterministic programs provide progressively different dependency
patterns while keeping buffer sizes and expected outputs easy to inspect.

### 1. Vector addition — the baseline

[`workloads/vector_add.cu`](workloads/vector_add.cu) computes `C = A + B`.
It exercises two input buffers, one compute stage, and one output buffer.
This is the simplest check of uploads, input reads, output writes, and download.
The expected printed value is `C[0] = 3`.

### 2. Pipeline — a sequential producer/consumer dependency

[`workloads/pipeline.cu`](workloads/pipeline.cu) computes `C = A + B`, then
`D = 2 × C`. The second stage requires the first stage's output. This tests
whether the graph represents intermediate data and a multi-stage dependency.
The expected printed value is `D[0] = 6`.

### 3. Fork-join — independent branches followed by a join

[`workloads/fork_join.cu`](workloads/fork_join.cu) computes `C = A + B` and
`D = A - B`, then `E = C × D`. Add and subtract share inputs but do not depend
on each other's outputs. Multiply requires both outputs.
This tests whether graph generation preserves independence and a join instead
of turning everything into a chain. The expected printed value is `E[0] = 8`.

The current CUDA code launches all three kernels on the default stream, so
they execute sequentially on the GPU. Their logical data graph still has two
independent branches. The graph does not promise simultaneous execution on a
single compute resource; the simulator must account for resource contention.

### 4. Reduction — shrinking data volumes across stages

[`workloads/reduction.cu`](workloads/reduction.cu) reduces 65,536 float32
elements into 256 partial sums and then one final sum. This tests dependencies
where transfer volumes change significantly between stages:
262,144 input bytes → 1,024 partial-sum bytes → 4 output bytes.
The expected result is `65536`, matching the program's printed expected value.

The transfer model describes the global buffers between the two kernels. It
does not expand the block/thread reduction tree or shared-memory accesses.

Together, these cover a baseline, a pipeline, branching/joining, and reduction.
They are controlled examples for developing graph generation, rather than a
comprehensive performance benchmark suite.

## Work completed through September 23, 2026

### Remote CUDA execution and trace conversion

All four programs were compiled and run on the remote NVIDIA machine. The
user-provided terminal output confirmed the expected printed results above.
Vector workloads print only the first output element; this is a sanity check,
not a full-array numerical validation.

Nsight Systems traces were converted into flow JSON and rendered as PNGs:

- Vector addition: 4 trace nodes, 3 edges.
- Pipeline: 5 trace nodes, 4 edges.
- Fork-join: 6 trace nodes, 5 edges.
- Reduction: 4 trace nodes, 3 edges.

The remote vector-add artifacts were saved under `results/20260923T142233Z/`.
The other three were saved under `results/20260923T215106Z/`. These paths refer
to the remote run; the artifacts have not been confirmed published to GitHub.

### Profiling compatibility fixes

- The remote Nsight Systems installation reported version `2022.4.2.50`.
- Automatic `.qdstrm` import failed. Explicit conversion with the matching
  Nsight Systems `QdstrmImporter` successfully produced readable reports.
- That installation uses the older `gputrace` report name. Its CSV output was
  renamed to the filename expected by the trace converter.
- [`nsys_trace_to_flow_json.py`](nsys_trace_to_flow_json.py) was fixed to accept
  missing sizes represented as `None`, `NaN`, `N/A`, or blank text. Unexpected
  malformed values still raise errors. The fix was pushed in commit `3f8e144`.
- Installing matplotlib and NetworkX in a remote virtual environment enabled
  PNG rendering. Layout warnings did not prevent image generation.

### Logical transfer graphs prepared locally

A separate exporter and four logical graphs have been developed and tested
locally. At the time of this README update, those additions have **not yet been
committed or pushed**; the trace converter already in this repository does not
automatically produce these simulator-format logical models.

The local models contain:

- Vector addition: 6 transfer nodes, 5 dependency edges.
- Pipeline: 8 transfer nodes, 7 dependency edges.
- Fork-join: 12 transfer nodes, 13 dependency edges.
- Reduction: 6 transfer nodes, 5 dependency edges.

They explicitly represent host uploads (`cpu → mem`), kernel input reads
(`mem → acc`), kernel output writes (`acc → mem`), and host downloads
(`mem → cpu`). Each read waits for its buffer producer; each kernel output
write waits for that kernel's input reads. Dependencies operate at whole-buffer
granularity. The four logical-model tests passed, including checks for reference
schema, reduction sizes, component mapping, and fork-join independence.

These models are manually specified from the current CUDA buffer accesses and
sizes. They are not a general CUDA parser or automatic dependency-recovery tool.
They need updating if the workloads change.

## Trace graphs versus simulator transfer graphs

The completed remote `--aggregation raw` exports represent **execution order**:
one node per trace event connected in a chain. They include kernel events and
use `source_id` / `dest_id`, rather than the reference simulator field names
`src_comp` / `dest_comp`. They are therefore not equivalent to the intended
simulator transfer graphs.

The logical models describe **buffer transfers and data dependencies**. Their
volumes count logical reads and writes. For example, both fork branches read
A and B, so both reads are represented even if GPU caching could reduce physical
DRAM traffic. A missing size on a trace kernel event does not mean that the
kernel accesses zero bytes.

The graph format alone does not encode compute latency, caching, bandwidth,
or contention policies. Those must come from the simulator's model or a future
explicit extension. Matching the reference schema is not proof of simulator
compatibility: access to the simulator is currently unavailable, so no simulator
execution or resulting performance metrics have been validated.

## Current focus and remaining work

1. Publish the local logical exporter, its tests, documentation, and selected
   small graph artifacts.
2. Inspect the rendered logical graphs and verify component mappings against
   the intended simulator architecture.
3. Run the graphs through the simulator when access becomes available and check
   its interpretation of transfer dependencies and resource constraints.
4. Extend beyond these four examples once the end-to-end input contract is
   validated.

Internal GPU profiling is deferred. A local Nsight Compute workflow has been
prepared, but no remote hardware-counter collection has been performed with it.
It would provide separate per-kernel DRAM measurements for comparison, not
replace the logical transfer volumes or reveal per-buffer traffic automatically.

## Existing tools

- [`nsys_trace_to_flow_json.py`](nsys_trace_to_flow_json.py): converts Nsight
  Systems CSV traces to flow JSON. `raw` and `semantic` produce chains;
  `stream_dag` infers scheduling relationships from streams and timing, not
  general data dependencies.
- [`flow_graph_networkx.py`](flow_graph_networkx.py): renders node-link JSON
  graphs using matplotlib and NetworkX.
- [`build_and_profile.sh`](build_and_profile.sh): older workflow for root-level
  dummy, FFT, and diamond workloads. It is not the four-workload workflow and
  includes a hardcoded clock assumption that should not be reused blindly.
- [`ncu_attach_metrics.py`](ncu_attach_metrics.py): existing Nsight Compute
  metric attachment utility; it has not been validated for the new logical
  transfer graphs.
