# Simulator transfer models

Generate all four models from the repository root (Python 3 plus Clang):

```bash
python3 export_workload_transfers.py
```

This command runs on the remote server or any machine with Python 3 and
`clang++` with CUDA parsing support. It was tested on Apple Clang 17. No
Mac-generated files, CUDA execution, GPU access, or Nsight trace is required.
The bundled analysis-only CUDA header lets Clang parse without a CUDA toolkit.
Never use `source_analysis/include` when building actual CUDA binaries.
On Ubuntu, install Clang with `sudo apt install clang` if it is missing.
Select a compiler with `--clang /path/to/clang++` or the `CLANGXX` environment
variable. Unsupported compiler/source combinations produce parse errors.

Analyze a new file without adding a workload-specific Python model:

```bash
python3 export_workload_transfers.py --source workloads/my_workload.cu
```

For JSON and PNG generation together, use the remote virtual environment
where matplotlib and NetworkX are already installed:

```bash
source .venv/bin/activate
python export_workload_transfers.py --render
```

The optional renderer uses the same Python environment as the generator and
reports failures. JSON files already written remain available if rendering
fails. Re-running replaces files with the same names; choose another
`--out-dir` to keep a separate set.

Images use left-to-right dependency levels (`L0`, `L1`, ...), memory lanes,
and rounded boxes labeled with transfer ID, endpoints, and byte volume.
Orange denotes accelerator writes; purple denotes accelerator reads. Host
uploads and downloads use separate colors, identified in the legend. Units
are exact bytes or binary units (KiB, MiB, GiB), not rounded decimal MB.
Levels indicate dependencies, not elapsed time. Rendering does not add edges
or change simulator inputs. This layout requires a nonempty DAG.

Outputs are in `results/logical_transfers`. Use only `*_simulator.json` as
simulator inputs. The files match the reference shape: `directed`,
`multigraph`, empty `graph`, `nodes`, and `edges`. Nodes contain only `id`,
`vol` (integer bytes), `src_comp`, and `dest_comp`. Edges contain `source`
and `target` node IDs. Actual simulator compatibility is untested because
the simulator is unavailable.

`cpu` is the host, `mem` is accelerator-accessible memory, and `acc` is compute.
Configure names with `--cpu`, `--mem`, and `--acc`. Select one workload
with `--workload fork_join`.

Clang parses current source into an AST (abstract syntax tree). The analyzer
evaluates supported scalar expressions, tracks `cudaMalloc` sizes and
`cudaMemcpy` calls, resolves launches to kernel definitions, and identifies
global array reads/writes and their contiguous footprints. Editing source
sizes, arguments, accesses, or straight-line launch order updates the model
on the next run. Workload filenames and kernel names do not select templates.

Supported scope:

- A source file with `main` and visible, non-overloaded kernel definitions.
- Statically resolvable scalar parameters, arithmetic, simple integer macros,
  and `sizeof` of float, double, int, unsigned int, or char. Arithmetic is
  modeled mathematically; integer overflow/narrowing is outside the model.
- Straight-line `cudaMalloc`, host/device `cudaMemcpy`, and kernel launches
  in `main`. Host initialization loops cannot contain device operations or
  mutate parameters used to build the graph.
- Explicit nonblocking streams, timing-disabled events, record/wait snapshots,
  and device/stream synchronization. Conflicting accesses must be ordered by
  the same stream, an event wait, or host synchronization; missing ordering fails.
- Size-initialized primitive `std::vector` host buffers with `.data()` copies,
  and checked CUDA calls through a validated diagnostic-only error wrapper.
- Positive 1-D launch dimensions and direct, non-aliased buffer arguments.
- Zero-based contiguous accesses using the linear thread index, thread index,
  block index, or scalar index zero, with simple `<` or `==` bounds. This
  includes input bounds, one partial sum per block, and one final scalar.
- Whole-allocation writes and initialized prefix reads. Shared-memory-only
  loops are excluded from the global footprint, as in the reduction examples.

Unsupported examples include dynamic host control flow around launches,
runtime input sizes, pointer aliases/offsets, partial writes, strided or
data-dependent indexing, global accesses inside loops, kernel helper calls,
atomics, implicit blocking-stream synchronization, dynamic shared memory, multidimensional launches,
and device-to-device copies. They require additional analysis and currently
fail rather than producing a guessed graph. This is a deliberately restricted
analyzer, not a verifier of general CUDA correctness or a replacement for nvcc.
Sources should be trusted: Clang preprocessing can read included local files.

Uploads are `cpu -> mem`; kernel input reads are `mem -> acc`; kernel
output writes are `acc -> mem`; downloads are `mem -> cpu`. Each read
waits for its buffer producer; a kernel output write waits for all its
input reads. Overwrites also wait for earlier readers and writers of the same
allocation. Read/read pairs have no dependency. This represents whole-buffer
readiness, not element-level streaming. Compute duration is not modeled.

Fork-join's add and subtract read A and B independently. Their outputs C
and D feed multiply, whose output E waits for both inputs. No dependency
is added merely because kernels execute in one CUDA stream. Sharing the
`acc` endpoint does not imply multiple physical accelerators; resource
contention is left to the simulator.

Volumes count unique global buffer footprints per kernel, not memory
instructions or measured DRAM traffic. Repeated references to `A[i]` within
one kernel count the element once. Both
fork branches read A and B, so those reads count twice. Cache reuse,
transaction overhead, registers, and shared-memory traffic are excluded.
Reduction models global transfers for two kernels, not the internal tree.

Separate `*_metadata.json` files explain node IDs by stage and buffer,
record the source hash, and document assumptions. They are not simulator
inputs. The hash records the analyzed source; re-run after editing it.
All selected inputs are analyzed before output JSONs are written. If analysis
fails, old outputs are not deleted: do not mistake an old file for new output.
Existing Nsight exports remain separate
execution-order graphs.

Render in an environment with matplotlib and networkx installed:

```bash
for name in vector_add pipeline fork_join reduction; do
  MPLBACKEND=Agg python3 flow_graph_networkx.py \
    --input "results/logical_transfers/${name}_simulator.json" \
    --name "${name}_simulator" \
    --out-dir results/logical_transfers/graphs
done
```

Run validation:

```bash
python3 -B -m unittest discover -s tests -p 'test_*source*.py' -v
python3 -B -m unittest discover -s tests -p test_workload_transfers.py -v
```

## Explicit stream fork and join

```bash
python3 export_workload_transfers.py --workload fork_join_streams --render
```

This additional example is selected explicitly (`all` retains the original four).
No GPU or Nsight run is required. Stream/event handling is derived from API
calls, not stream names or workload-specific templates. Host-to-device uploads
must be synchronized before use on nonblocking streams; the join waits for
both branch events, and the host synchronizes before downloading its output.
Event waits capture the recorded work at the wait call, not future recordings.
This conservative subset requires users to synchronize before freeing buffers
or destroying streams; it does not infer every implicit CUDA synchronization.

Simulator edges remain logical buffer dependencies. They do not encode all
scheduling restrictions between unrelated buffers. Metadata records each
transfer's operation ID, stream, and ordered predecessor operations; these IDs
are distinct from transfer node IDs. This validates the supported source ordering
but does not measure overlap, runtime, or bandwidth. The graph alone cannot
reproduce an execution timeline. A read shared by both branches is represented
separately for each kernel, as in the original fork-join model.

## Configured Rodinia analysis

For the original Nearest Neighbor source, use `--source` together with
`--analysis-config workloads/rodinia/nn/analysis.json`. This selects an additional
bounded AST evaluator; the default affine analyzer and four-example batch are
unchanged. Use `--parameter records=1025` to override its named input count.
The configuration explicitly summarizes input helpers and supplies device limits.
The kernel and CUDA host operations are analyzed from source without execution.
See [Rodinia instructions and limitations](rodinia/README.md). Graph metadata
records the resolved configuration and launch dimensions; simulator JSON retains
the existing four-field node schema.

The configured mode also supports the unchanged Rodinia HotSpot3D implementation
using `workloads/rodinia/hotspot3d/analysis.json`. It evaluates selected host helper
bodies, included source units, bounded loops, and repeated default-stream
launches with pointer swaps. The total thread budget spans all launches.
See [HotSpot3D instructions](rodinia/hotspot3d/README.md), including the upstream
previous-buffer download behavior and explicit CPU/IO assumptions.


Gaussian adds an explicit parameterless host entry (`entry_function`), with
configured scalar globals and distinct host-buffer shapes (`entry_globals`).
This is a declared function boundary, not an analysis of its caller. The
optional `dependency_granularity: byte_regions` tracks last writers and pending
readers per byte, allowing partial/strided footprints and in-place accesses
with cross-thread race checks. Volumes are unique footprint sizes; metadata
`byte_ranges` are half-open intervals. Graph schema is unchanged. Region
tracking is limited by `max_tracked_bytes` (default/maximum 1 MiB of live device
allocations), in addition to the aggregate thread budget. Existing configurations
retain whole-buffer semantics. See [Gaussian scope and validation](rodinia/gaussian/README.md).


HotSpot 2D and Pathfinder explicitly select `shared_memory: opaque_values`.
Fixed numeric shared arrays are tracked for shape and bounds only; every read
returns unknown, including after writes and barriers. Shared transfers,
initialization, races, and barrier correctness are not modeled. Global
footprints assume valid kernel synchronization. Scalar-only MIN-like conditional
expressions may return unknown; unknown values still cannot select memory
addresses or control statements. Expressions hiding conditional memory accesses,
side effects, or pointer selections are rejected.

The configured evaluator now handles fixed host pointer arrays, allocations
through their elements, host copy offsets with bounds checks, bounded `break`,
selected numeric casts, and host `delete[]`. Source-hash-checked host summaries
may supply previously uninitialized globals with configured scalar values or
host-buffer shapes. A cleanup-only opaque host pointer has zero transferable
capacity. These assumptions are recorded in configuration; host summaries cannot
receive device pointers directly or through pointer arrays. No arbitrary host
or shared-memory execution is implied. See the
[HotSpot 2D](rodinia/hotspot2d/README.md) and
[Pathfinder](rodinia/pathfinder/README.md) scope guides.

## Additional configured-analysis features

NW and LUD opt into `lexical_scopes: true`, restoring block and loop-local
variables on exit. Bounded `while` loops and integer switches follow known
scalar conditions; unknown decisions still fail. The original LU driver's
separate short `getopt_long` options are modeled; long/combined/optional-value
forms are rejected rather than guessed.

`host_memory: "opaque_values"` permits numeric host initialization without
propagating array contents. Known host pointer accesses are bounds-checked.
Unknown indices into fixed host arrays retain only shape and unknown contents;
they cannot become device pointers or resolve launch controls. This abstraction
does not validate host numerical correctness. Simple default-constructed host
POD records remain opaque and cannot be used as modeled device buffers.

`scalar_device_functions` names helpers whose complete AST bodies must contain
only numeric local/parameter operations, conditionals, and returns. Calls,
pointer arguments, memory accesses, and global references are rejected. Unknown
arguments yield unknown scalar results; no memory effect is inferred or hidden.

Hash-pinned host summaries may declare `requires` scalar-argument checks and
`buffers` for numeric output-pointer allocations with explicit shapes. They
cannot receive device pointers. LU uses this for CPU matrix preparation, while
analyzing the original application's CUDA flow. These facilities do not add
GPU execution, timing, physical-traffic measurements, or simulator validation.


## Complete Backpropagation, SRAD v1 and CFD scenarios

`tests/test_additional_rodinia.py` independently constructs byte footprints and overlap dependencies for the original application paths. It covers Backpropagation at 16/32 inputs, SRAD at multiple grid sizes including a two-stage reduction, and CFD at one/two iterations. Negative cases reject stale host contracts, invalid dimensions, unknown index tables, missing/nonuniform block barriers, memory accesses hidden by unknown branches, modified mesh hashes and out-of-range neighbors.

The evaluator now reads selected C translation units, binds device helper references, tracks float3 fields and constant-memory arrays, and models a device-to-device copy as one transfer. Explicit integer initialization and hash-pinned input fixtures can resolve index tables through uploads. Other numerical device values remain unknown. CPU-mediated dependencies propagate conservatively through host buffers and declared CPU summaries; CPU time is omitted. Unknown scalar branches can merge only local scalar operations without memory accesses. Block barrier phase checks constrain global-memory conflicts; they do not prove shared-memory numerical correctness.

These graphs describe unique logical byte footprints and dependencies, not physical DRAM traffic, cache transactions, measured execution order or performance. The configured small scenarios have no new GPU-run or simulator validation.


## Five fixed-input application paths and optional value state

The BFS, LavaMD, Particle Filter, K-means and B+tree guides describe their exact supported scenarios. `cuda_integer_model.py` extends bounded AST evaluation with known integer bytes, optional floating bytes for the small K-means fixture, pre-launch snapshots plus thread-local writes, and identical-footprint path exploration. Unknown paths that differ in memory footprint or barriers are rejected, as are conflicting threads. Opaque host addresses do not prove CPU memory safety. CPU-visible downloads add conservative control dependencies to later transfers.

`tests/test_five_source_workloads.py` uses four focused test groups: independent schedules and saved-artifact comparisons for all five cases; execution of B+tree's original CPU preparation with GPU calls stubbed; selected unsupported-input checks; and notebook/capture/CSV preparation checks. The CPU preparation test compares only defined bytes, not uninitialized padding. K-means floating evaluation does not prove arbitrary floating-point convergence or GPU FMA equivalence. No new GPU run, cycle measurement or simulator result is claimed by these tests.
