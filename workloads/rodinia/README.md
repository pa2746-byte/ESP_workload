# Rodinia source-analysis experiments

Two examples are now available:

- **Nearest Neighbor**, described below: struct fields, pointer offsets, and
  configured input size.
- **[HotSpot3D](hotspot3d/README.md)**: stencil accesses, included CUDA source,
  bounded loops, repeated launches, and temperature-buffer swaps. Its upstream
  download selects the previous result; the linked guide explains how the graph
  preserves that behavior.

`nn/nn_cuda.cu` is an **unchanged upstream CUDA Nearest Neighbor implementation**
from [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia), commit
`9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`, path `cuda/nn/nn_cuda.cu`.
The source was compared byte-for-byte with that revision. Upstream licensing is
preserved in [LICENSE](LICENSE). No dataset or CUDA binary is bundled here.

## Generate the graph without a GPU

From the repository root, in a Python environment with Clang available:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/nn/nn_cuda.cu \
  --analysis-config workloads/rodinia/nn/analysis.json \
  --out-dir results/rodinia_nn \
  --render
```

Rendering additionally needs matplotlib and networkx. JSON generation needs
only Python and Clang. Source parsing and tests were validated with Apple
Clang 17; this is not a claim of a CUDA build, GPU execution, or a simulator run.
The analysis-only headers must never be used to build a CUDA executable.

Outputs:

- `results/rodinia_nn/nn_cuda_simulator.json`: transfer/dependency graph.
- `results/rodinia_nn/nn_cuda_metadata.json`: source hash, resolved configuration,
  assumptions, launch dimensions, and node descriptions.
- `results/rodinia_nn/graphs/nn_cuda_simulator.png`: rendered graph.

The supplied configuration models **513 records**. Each coordinate record has
two floats (8 bytes); each distance is one float (4 bytes). The derived graph is:

1. Upload coordinates: 4,104 bytes, `cpu -> mem`.
2. Kernel reads coordinates: 4,104 bytes, `mem -> acc`.
3. Kernel writes distances: 2,052 bytes, `acc -> mem`.
4. Download distances: 2,052 bytes, `mem -> cpu`.

There are four nodes and three edges. The configured device permits 256 threads
per block, so the original source chooses three blocks; its guard excludes the
255 extra threads from memory access. Both repeated latitude/longitude reads
are deduplicated into unique byte footprints, not counted as instructions.

Try another record count without editing the source or analyzer:

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/nn/nn_cuda.cu \
  --analysis-config workloads/rodinia/nn/analysis.json \
  --parameter records=1025 \
  --out-dir results/rodinia_nn_1025 \
  --render
```

## What is analyzed and what is assumed

The new bounded evaluator in `cuda_configured_model.py` walks Clang's AST for
host allocations, copies, scalar calculations, conditionals, and kernel launches.
It enumerates configured kernel threads and tracks struct-field reads and
pointer-offset writes as byte addresses. Buffer names, kernel names, launch
formulas, access footprints, and graph edges are derived from source. There is
no prewritten Nearest Neighbor graph template.

Device buffer contents remain unknown. A value-dependent branch or address
therefore causes an error. This analysis models memory accesses; it does not
calculate distances or validate the CPU's nearest-record selection.

Three host helper functions have explicit contracts in `nn/analysis.json`:

- `parseCommandline`: assume valid options, query `(30, 90)`, five requested
  results, quiet mode, and timing disabled. `writes` keys are zero-based argument
  positions; they specify values written through pointer arguments.
- `loadData`: assume a successful load of the configured record count into both
  host vectors. `vectors` keys identify vector argument positions. Dataset
  contents and file loading are **not executed or verified**.
- `findLowest`: CPU postprocessing contributes no GPU transfer nodes. Its CPU
  work and modifications to the host result buffers are outside this graph.

`body_sha256` checks the function's source signature and body. Preprocessor
lines are also pinned because macros can change helper behavior. If either
changes, analysis refuses the old contract. Review the helper before updating
its contract/hash; never refresh hashes automatically to bypass a failure.
These contracts are assumptions supplied for the experiment, not proofs that
arbitrary host helpers are side-effect-free. They receive no device pointers.

Device memory capacity, maximum grid dimensions, and threads per block are
explicit configuration values, not queried hardware. They determine the path
through Rodinia's original sizing and input-capacity checks. `records=513` is a
small configured scenario, not a claim that a real dataset was loaded.

## Limits and tests

This configured mode is separate from the original affine analyzer. It supports
this default-stream, single-file example and a restricted subset of scalar
expressions, plain primitive-field structs, and direct pointer arithmetic.
It rejects data-dependent control/indexing, unsupported calls/loops, pointer
aliases between kernel arguments, in-place kernels, overlapping writes by
threads, out-of-bounds accesses, non-contiguous footprints, and partial-buffer
writes. Integer overflow/narrowing, arbitrary C++ object lifetimes, general
scope resolution, and full host-program behavior are outside its model.

Thread enumeration across all launches defaults to a configured limit of 65,536 and has an absolute
cap of 262,144. This is a correctness experiment for small inputs; it is not yet
a scalable analysis of million-record runs. Increasing input size may require
symbolic range analysis rather than increasing the cap.

```bash
python3 -B -m unittest discover -s tests -p test_rodinia_source_model.py -v
```

Tests cover multiple input sizes, partial blocks, a two-dimensional grid,
changed device limits, renamed source identifiers, exact schema/volumes/edges,
and rejection of invalid accesses, races, data-dependent conditions, changed
host contracts, and invalid configurations. No GPU or simulator is used.
