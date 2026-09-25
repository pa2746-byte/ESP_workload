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

Historical February–April 2026 results are collected in
[past graphs from nsight](<past graphs from nsight/>), including reports,
graph JSONs, rendered images, and related synthetic examples. Current Clang
outputs remain in [results/logical_transfers](results/logical_transfers/);
September Nsight run directories remain under `results/`.

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

## Generate simulator inputs on the remote server

The repository includes a generator for all four workloads. No files need to
be generated on the Mac or transferred from it. On the remote server:

```bash
cd ~/ESP_workload
git switch pa2746-byte-sampleworkloads
git pull --ff-only origin pa2746-byte-sampleworkloads
python3 export_workload_transfers.py
```

This writes four `*_simulator.json` files and separate explanatory metadata
under `results/logical_transfers/`. Only the simulator JSON files are intended
as simulator inputs. Generation requires Python 3 and `clang++`, but no GPU,
CUDA toolkit, or Nsight run. On Ubuntu, install Clang if needed with
`sudo apt install clang`. The analyzer was tested locally with Apple Clang 17;
remote generation has also succeeded after compatibility fixes described below.

To also generate PNGs using the previously configured virtual environment:

```bash
source .venv/bin/activate
python export_workload_transfers.py --render
```

Analyze another CUDA file directly:

```bash
python export_workload_transfers.py --source workloads/my_workload.cu --render
```

The generator reads current source on every run. Supported changes to sizes,
kernel arguments, copies, and global array accesses change the output without
editing Python models. Unsupported syntax or access patterns cause an error.

PNGs are written under `results/logical_transfers/graphs/`. Select a single
workload with `--workload fork_join`, or choose another destination with
`--out-dir`. Re-running replaces outputs with the same names.
See [transfer model documentation](workloads/TRANSFER_MODELS.md) for assumptions
and component naming options.

## How we use Clang

Clang reads the CUDA source and produces an **abstract syntax tree (AST)**:
a structured representation of declarations, expressions, and calls. Our
Python analyzer interprets that tree to generate transfer nodes and dependency
edges. Clang itself does not produce our simulator graph.

The implementation is split between
[`export_workload_transfers.py`](export_workload_transfers.py), the command-line
entry point, and [`cuda_source_model.py`](cuda_source_model.py), the analyzer.
It invokes `clang++` with `--cuda-host-only`, `-fsyntax-only`, and
`-Xclang -ast-dump=json`. This parses source without executing the workload or
collecting GPU measurements. `-nocudainc` / `-nocudalib` and a bundled
[analysis-only header](source_analysis/include/cuda_runtime.h) let it parse
the supported CUDA APIs without requiring the CUDA toolkit. That header is
not a CUDA runtime and must not be used for actual workload compilation.

The analyzer performs these steps:

1. Evaluate supported source expressions for sizes and launch dimensions.
2. Track device allocations and host/device copy calls.
3. Match each kernel launch to its definition and actual buffer arguments.
4. Identify supported global array reads/writes and their byte footprints.
5. Connect producers to consumers and add dependencies for buffer overwrites,
   while preserving independent reads and branches.
6. Write simulator JSON, explanatory metadata, and optional PNGs.

The original exporter contained manually specified models for the four
workloads. Those models have been replaced by source analysis. Supported
changes to sizes, accesses, or kernel arguments now change the output on the
next run without editing a workload-specific Python template. Test cases
modify source code to verify this behavior.

Clang AST output varies across versions. We fixed two differences encountered
on the remote server: the newer `__cudaPushCallConfiguration` launch helper,
and default arguments omitted from call-site AST nodes. The latter are resolved
from constructor parameter declarations rather than guessed. Both cases have
regression tests. The remotely generated graphs have been pulled back and
checked against fresh local analysis; their source hashes match the current
four CUDA files.

### Explicit streams and event joins

The analyzer also supports [fork_join_streams.cu](workloads/fork_join_streams.cu):
two independent nonblocking worker streams followed by a third stream that
waits for both completion events. Generate it without running CUDA on a GPU:

```bash
python3 export_workload_transfers.py --workload fork_join_streams --render
```

The generated JSON and metadata are in `results/logical_transfers/`; the PNG is
[graphs/fork_join_streams_simulator.png](results/logical_transfers/graphs/fork_join_streams_simulator.png).
It has 12 transfers and 13 dependencies, with 4,194,372 bytes per transfer.
The analyzer checks stream/event ordering and rejects missing synchronization
for conflicting buffer accesses. It also supports this example's primitive host
vectors and diagnostic-only CUDA error wrapper. Names and sizes come from the
source. The original four-example default remains unchanged.

Logical edges describe buffer dependencies; full operation ordering is recorded
separately in metadata. These outputs do not measure GPU overlap or execution
time. See [supported scope](workloads/TRANSFER_MODELS.md) for restrictions.

### Scope and portability

Clang is not limited to CUDA, but **our current analyzer is CUDA-specific**.
OpenCL, HIP, and SYCL inputs would need additional handling for their allocation,
copy, launch, and access conventions. The intended output remains the same
simulator JSON schema across supported front ends.

The present implementation supports a restricted source subset, not arbitrary
real-world applications. Runtime sizes, complex control flow, multiple-file
analysis, library calls, aliases, and irregular accesses need further work.
Unsupported constructs produce errors rather than guessed graphs. See
[supported patterns and limitations](workloads/TRANSFER_MODELS.md).

Nsight is optional for this source-to-graph workflow. It remains useful for
checking actual execution, copy sizes, and timing. Hardware traffic counters
are a separate validation path, not a prerequisite for logical graph generation.

## Future benchmarks to investigate

The next step is to analyze existing application sources, preserving their
computation rather than rewriting them into our own task framework. The
following are candidates, **not workloads already supported or validated**.
The order below is a proposed progression; exact implementation requirements
must be confirmed by inspecting each selected source version.

1. **Rodinia HotSpot — first target.** A processor thermal simulation with
   temperature and power inputs and repeated temperature updates. It extends
   our examples toward neighboring-cell accesses, boundary conditions, and
   iteration-to-iteration dependencies. Start with a small input for its
   existing CUDA implementation. Expect to extend multidimensional indexing,
   host launch loops, and runtime-size handling.
   [Application description](https://rodinia.cs.virginia.edu/hotspot.html).
2. **Rodinia SRAD — image processing.** A candidate for exploring a larger
   multi-stage application and intermediate-buffer dependencies. Inspect its
   CUDA implementation to determine the additional control-flow, reduction,
   and indexing support needed.
3. **Rodinia K-means — data mining.** A candidate for iterative computation
   whose access patterns and convergence behavior depend on input data. It
   can help establish where source-only analysis needs user-supplied runtime
   parameters or conservative dependency handling.
4. **Rodinia BFS — later stress test.** Graph traversal introduces irregular,
   data-dependent accesses. Use it to evaluate the limits of static analysis,
   not as an immediate promise of exact graph recovery.

Rodinia lists CUDA, OpenCL, and OpenMP implementations for these candidates.
These are separate implementations: availability across programming models
does not make our CUDA analyzer portable automatically or guarantee execution
on every accelerator. [Rodinia benchmark catalog](https://rodinia.cs.virginia.edu/).

For each benchmark, record the source version and input sizes, identify analyzer
gaps, extend and test generic analysis rules, and validate transfer sizes and
dependencies before treating its output as a simulator input. Do not silently
fall back to a hardcoded graph when source analysis fails.

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

### Logical transfer graph generation

A separate [exporter](export_workload_transfers.py), its
[tests](tests/test_workload_transfers.py), and documentation are included in this
repository. It directly generates the simulator-format logical graphs on the
machine where it runs. The Nsight trace converter continues to serve the
separate purpose of exporting observed execution.

The generated models contain:

- Vector addition: 6 transfer nodes, 5 dependency edges.
- Pipeline: 8 transfer nodes, 7 dependency edges.
- Fork-join: 12 transfer nodes, 13 dependency edges.
- Reduction: 6 transfer nodes, 5 dependency edges.

They explicitly represent host uploads (`cpu → mem`), kernel input reads
(`mem → acc`), kernel output writes (`acc → mem`), and host downloads
(`mem → cpu`). Each read waits for its buffer producer; each kernel output
write waits for that kernel's input reads. Dependencies operate at whole-buffer
granularity. Local tests cover schema, sizes, component mapping, fork-join
independence, command-line generation, changed source sizes and inputs,
changed dependencies, and rejection of unsupported patterns. Remote JSON and
PNG generation succeeded for all four workloads. The pulled JSONs match fresh
local source-analysis output and the recorded source hashes match current files.

The generator now uses Clang's syntax tree to derive these models from source,
replacing the initial hardcoded Python models. It supports a restricted CUDA
subset: straight-line host device operations, statically known sizes, 1-D
launches, and simple contiguous global array access patterns. It is not a
general analyzer for arbitrary CUDA programs. See the documented supported
subset and error conditions before adding a workload.

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

1. Begin source inspection of Rodinia HotSpot and identify the generic analyzer
   extensions needed for its existing CUDA implementation.
2. Verify component mappings against the intended simulator architecture;
   the four example graphs have already been generated and visually inspected.
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
