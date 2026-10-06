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

## Current status

Source-derived graph generation now covers **seven workload examples**: four
original CUDA samples, the additional stream fork-join sample, and two original
Rodinia applications. The analyzer extensions, tests, configurations, JSONs,
PNGs, and documentation are committed on `pa2746-byte-sampleworkloads`.

- **Original samples:** vector addition (6 nodes / 5 edges), pipeline (8 / 7),
  fork-join (12 / 13), and reduction (6 / 5).
- **Explicit stream fork-join:** two worker streams and an event-based join
  (12 / 13), with checks for missing synchronization.
- **Rodinia Nearest Neighbor:** unchanged upstream source, configured for 513
  records (4 / 3); adds struct-field and pointer-offset analysis.
- **Rodinia HotSpot3D:** unchanged upstream sources, configured for a 64 x 64 x 3
  grid and two iterations (9 / 11); adds stencil accesses, included source,
  bounded loops, repeated launches, and buffer swaps.

The latest implementation validation passed **33 relevant tests**. The original
four samples also have remote CUDA/Nsight run evidence. The two Rodinia examples
have been analyzed and tested locally with Apple Clang 17; they have **not** been
numerically validated on a GPU or run through the simulator. Their file loading,
CPU processing, and device parameters use explicit documented assumptions.
Clang itself was not modified; our Python analyzers that consume its AST were
extended. Unsupported cases still require additional analysis work.

Current output locations:

- [Original samples and stream example](results/logical_transfers/), with images
  in [graphs](results/logical_transfers/graphs/).
- [Nearest Neighbor](results/rodinia_nn/), including its
  [image](results/rodinia_nn/graphs/nn_cuda_simulator.png).
- [HotSpot3D](results/rodinia_hotspot3d/), including its
  [image](results/rodinia_hotspot3d/graphs/3D_simulator.png).

Use `*_simulator.json` as the intended simulator input. `*_metadata.json` records
source hashes, node descriptions, and analysis assumptions; Rodinia metadata
also includes resolved configuration and launch dimensions. PNGs visualize the
same graph. These files do not contain measured performance metrics.

## Graph representation

Historical February–April 2026 results are collected in
[past graphs from nsight](<past graphs from nsight/>), including reports,
graph JSONs, rendered images, and related synthetic examples. Current Clang
outputs are listed above; September Nsight run directories remain under `results/`.

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

The default generator command selects the original four workloads. The stream
example and Rodinia applications are selected explicitly below. No files need to
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

The command-line entry point is
[`export_workload_transfers.py`](export_workload_transfers.py). It uses two
analysis paths:

- [`cuda_source_model.py`](cuda_source_model.py): affine access analysis for the
  original samples and explicit stream/event example.
- [`cuda_configured_model.py`](cuda_configured_model.py): bounded AST evaluation
  for configured Rodinia inputs, selected by `--analysis-config`. It enumerates
  kernel thread addresses while keeping device data values unknown. Selected
  host helpers are analyzed; other helpers have explicit hash-checked contracts.

Both paths use Clang parsing through `cuda_source_model.py`, which invokes
`clang++` with `--cuda-host-only`, `-fsyntax-only`, and
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
real-world applications. Configured runtime sizes, selected included source
files, bounded loops, and direct pointer offsets are now supported in the
Rodinia analysis path. General dynamic control flow, arbitrary multi-file
programs, library calls, aliasing, shared-memory synchronization, and
input-data-dependent accesses still require more work. The configured path
limits total enumerated threads and loop iterations; it is intended for small
validation cases, not production-scale inputs. See
[supported patterns and limitations](workloads/TRANSFER_MODELS.md).

Nsight is optional for this source-to-graph workflow. It remains useful for
checking actual execution, copy sizes, and timing. Hardware traffic counters
are a separate validation path, not a prerequisite for logical graph generation.

## First Rodinia workload: Nearest Neighbor

The original Rodinia CUDA Nearest Neighbor source is now included unchanged,
with a bounded Clang analysis mode for small configured inputs. Run:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/nn/nn_cuda.cu \
  --analysis-config workloads/rodinia/nn/analysis.json \
  --out-dir results/rodinia_nn --render
```

For the configured 513 records, the graph has four transfers and three edges:
4,104-byte coordinate upload/read and 2,052-byte distance write/download.
The kernel's struct accesses, pointer offsets, launch geometry, and bounds guard
are analyzed from source. Input loading and CPU result selection use explicit
reviewed host contracts; device limits are configuration assumptions. No dataset
was loaded and no GPU or simulator execution was performed.

See [the Rodinia guide](workloads/rodinia/README.md) for provenance, parameter
changes, tests, output locations, and limits. This mode is an additional bounded
analysis path, not unrestricted support for the whole Rodinia suite.

## Second Rodinia workload: HotSpot3D

The unchanged Rodinia HotSpot3D source is also supported for small configured
inputs. It adds stencil accesses, bounded host/kernel loops, included CUDA
source, and alternating temperature buffers across iterations.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/hotspot3d/3D.cu \
  --analysis-config workloads/rodinia/hotspot3d/analysis.json \
  --out-dir results/rodinia_hotspot3d --render
```

The default 64 x 64 x 3 grid and two iterations produce nine transfers and eleven
dependencies, each transfer containing 49,152 bytes. The original implementation
downloads the **previous iteration's buffer** after swapping pointers; the graph
preserves this source behavior. See the [HotSpot3D guide](workloads/rodinia/hotspot3d/README.md)
for details, tests, configuration assumptions, and restrictions. No dataset, GPU,
or simulator execution is performed.

## Future benchmarks to investigate

After consolidating the two supported Rodinia cases, expand to other existing
application sources while preserving their computation. The following are
candidates, **not workloads already supported or validated**.
The order below is a proposed progression; exact implementation requirements
must be confirmed by inspecting each selected source version.

1. **Rodinia HotSpot (2D) — separate from the supported HotSpot3D variant.** A processor thermal simulation with
   temperature and power inputs and repeated temperature updates. It extends
   our examples toward neighboring-cell accesses, boundary conditions, and
   iteration-to-iteration dependencies. Start with a small input for its
   existing CUDA implementation. Multidimensional indexing and bounded host
   loops now have a foundation; the 2D variant still needs analysis of its
   shared-memory stencil, barriers, and boundary conditions.
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

## Work completed

### Initial remote CUDA execution and trace conversion (September 23, 2026)

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
to the remote run; the reports, CSVs, flow JSONs, and images are now tracked
in this repository.

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

The generator replaced its initial hardcoded Python models with Clang AST
analysis. The original affine path was extended for stream/event ordering;
the configured path now supports the two Rodinia applications described above.
These are restricted source-analysis paths, not a general CUDA verifier.

### Later analyzer and repository work

- Added stream/event checks, primitive host-vector handling, and validated CUDA
  error wrappers; published the independent-stream fork-join example.
- Added the unchanged Rodinia Nearest Neighbor source with provenance and license,
  explicit input/device configuration, struct layouts, and pointer-offset reads.
- Added unchanged HotSpot3D source units and analysis of its host helper,
  bounded host/kernel loops, neighboring-cell reads, and alternating buffers.
  Metadata records both source hashes. Tests preserve the actual previous-buffer
  download and verify that a test-only source correction changes the dependency.
- Added regression tests for changed sizes, names, launch geometry, dependencies,
  unsafe accesses, iteration budgets, and stale host-helper contracts.
- Matched the requested image style with dependency levels, colored transfer
  boxes, and byte volumes. Long dependency arrows now route around other boxes.
- Archived February–April outputs in `past graphs from nsight/` and kept current
  source-analysis outputs separate from historical trace graphs.

Reproduce the 33-test validation without a GPU:

```bash
python3 -B -m unittest discover -s tests -p '*source_model.py' -v
python3 -B -m unittest discover -s tests -p test_workload_transfers.py -v
```

The tests establish behavior for the supported examples and rejection cases.
They do not establish arbitrary CUDA correctness or simulator compatibility.

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

## Next steps, in priority order

1. **Reproduce the Rodinia exports on the remote Linux environment.** Pull the
   branch, run both documented commands and the test suite, and compare source
   hashes, configurations, transfer volumes, and edges with the committed
   artifacts. Record the Clang version. This validates compiler portability
   without requiring GPU execution.
2. **Resolve the HotSpot3D output choice before using it as a correctness
   benchmark.** Keep the original source as a reference. If the goal is the
   final temperature field, add a separately documented corrected variant that
   downloads the latest buffer, with tests distinguishing the two graphs.
   The current original-source graph intentionally describes the previous result.
3. **Tie input assumptions to actual datasets.** Add dataset manifests or small
   readers that establish record counts and grid dimensions, record input
   provenance, and check configuration consistency. Keep unmodeled CPU helpers
   explicit; gradually replace contracts where useful. No numerical accuracy
   claims should be made from the current assumed inputs.
4. **Make analysis scale beyond the small validation cases.** Replace repeated
   thread/address enumeration with symbolic range analysis for regular patterns.
   Preserve byte footprints and dependencies against the bounded evaluator as a
   reference, including boundary cells, buffer swaps, and overwrites. Extend
   region tracking and shared-memory/barrier handling before workloads need them.
5. **Validate the simulator contract when access is available.** Confirm component
   instance names, byte units, dependency semantics, compute costs, and resource
   contention. Decide explicitly whether logical dependencies alone are the
   desired input or whether CUDA stream scheduling also needs representation.
   Run a small known graph before interpreting any performance metrics. This
   step remains unvalidated while simulator access is unavailable.
6. **Expand benchmarks after those checks.** Inspect the 2D HotSpot variant or
   SRAD next, selecting a concrete source version and documenting its analysis
   gaps. Keep K-means/BFS for later data-dependent access work. HIP, OpenCL, or
   SYCL support will require additional front ends; CUDA support does not provide
   cross-accelerator portability by itself.

GPU execution and profiling are optional validation tracks, not prerequisites
for source graph generation. When useful, a GPU run can check numerical results
and Nsight Systems can check observed transfers and scheduling. Internal GPU
profiling remains deferred: local hardware-profiling drafts are not part of the
published workflow, and no remote hardware-counter collection has been performed
with them. Physical DRAM counters would be separate measurements, not replacements
for logical buffer volumes.

## Existing tools

- [`export_workload_transfers.py`](export_workload_transfers.py): source-to-graph
  CLI, with optional rendering and configured Rodinia analysis.
- [`cuda_source_model.py`](cuda_source_model.py) and
  [`cuda_configured_model.py`](cuda_configured_model.py): the two Clang AST
  analysis paths described above.
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
