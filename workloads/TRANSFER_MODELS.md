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
- Positive 1-D launch dimensions and direct, non-aliased buffer arguments.
- Zero-based contiguous accesses using the linear thread index, thread index,
  block index, or scalar index zero, with simple `<` or `==` bounds. This
  includes input bounds, one partial sum per block, and one final scalar.
- Whole-allocation writes and initialized prefix reads. Shared-memory-only
  loops are excluded from the global footprint, as in the reduction examples.

Unsupported examples include dynamic host control flow around launches,
runtime input sizes, pointer aliases/offsets, partial writes, strided or
data-dependent indexing, global accesses inside loops, kernel helper calls,
atomics, explicit streams, dynamic shared memory, multidimensional launches,
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
