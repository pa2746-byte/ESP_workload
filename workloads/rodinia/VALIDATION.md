# Source-to-graph validation

Scope: the committed Nearest Neighbor, HotSpot3D, and Gaussian configurations, using the
unchanged sources in this repository. This validation needs Python and Clang,
not CUDA execution, a GPU, or hardware profiling.

## Method

`tests/test_rodinia_graph_contract.py` holds reviewed expectations derived from
reading the CUDA source. Expected node volumes, component pairs, buffer roles,
and **every edge** are specified independently of the exporter. These test-only
expectations are not used to generate production graphs.

The checks compare both committed JSONs and fresh Clang analysis against those
expectations. They also compare saved metadata/configurations and source hashes.
Reviewed source fingerprints are pinned in the tests: changing a benchmark
requires reviewing its expected graph, rather than automatically accepting new
exporter output as correct. Independent coordinate calculations check the
configured kernel address coverage. Existing mutation tests check that source
changes affect the graph and unsafe cases are rejected.

Run the independent contracts:

```bash
python3 -B -m unittest discover -s tests -p test_rodinia_graph_contract.py -v
```

Run the full relevant suite (47 tests, including 9 Gaussian checks):

```bash
python3 -B -m unittest discover -s tests -p '*source_model.py' -v
python3 -B -m unittest discover -s tests -p test_workload_transfers.py -v
python3 -B -m unittest discover -s tests -p test_rodinia_graph_contract.py -v
```

## Nearest Neighbor: expected graph

Source: [nn_cuda.cu](nn/nn_cuda.cu), `LatLong`, `euclid`, and `main`.
Configuration: [analysis.json](nn/analysis.json), 513 records.

Each coordinate record has two floats (8 bytes); each distance is one float
(4 bytes). The launch has three blocks of 256 threads; `globalId < numRecords`
permits exactly indices 0 through 512 to access memory.

- `t0`: upload `d_locations`, 4,104 bytes, CPU to memory.
- `t1`: read `d_locations`, 4,104 bytes, memory to accelerator.
- `t2`: write `d_distances`, 2,052 bytes, accelerator to memory.
- `t3`: download `d_distances`, 2,052 bytes, memory to CPU.

The exact edge set is `0 -> 1`, `1 -> 2`, `2 -> 3`: uploaded input is consumed,
the kernel output depends on its input, and the download consumes that output.
Repeated references to each coordinate field count once in the unique footprint.
The committed graph and freshly generated graph match this expectation.

## HotSpot3D: expected graph

Sources: [3D.cu](hotspot3d/3D.cu) and [opt1.cu](hotspot3d/opt1.cu), especially
`hotspotOpt1` and `hotspot_opt1`. Configuration: 64 x 64 x 3 cells, two iterations.

Each float buffer contains 12,288 cells: **49,152 bytes**. The source launches
4,096 threads, each covering one x/y column and walking its three layers. The
bounded stencil's unique temperature reads cover the whole grid; power reads
and temperature writes also cover the whole grid. Neighbor reuse does not
multiply the logical footprint.

The expected roles are:

- `t0`: upload initial temperature into `tIn_d`.
- `t1`: upload power into `p_d`.
- `t2`, `t3`: first iteration reads power and initial temperature.
- `t4`: first iteration writes the allocation originally named `tOut_d`.
- `t5`, `t6`: second iteration reads power and the first result.
- `t7`: second iteration writes the allocation originally named `tIn_d`.
- `t8`: download the allocation originally named `tOut_d`.

Every node transfers 49,152 bytes. The exact 11 edges, grouped by reason, are:

- Input readiness: `0 -> 3`, `1 -> 2`, `1 -> 5`.
- First result depends on its two inputs: `2 -> 4`, `3 -> 4`.
- Second iteration consumes the first result: `4 -> 6`.
- Second result depends on its two inputs: `5 -> 7`, `6 -> 7`.
- Reusing the initial-temperature allocation waits for its previous writer and
  reader: `0 -> 7`, `3 -> 7`. The writer edge is transitively redundant but is
  intentionally emitted by the current whole-buffer dependency policy.
- Download consumes the first result: `4 -> 8`.

The committed graph and fresh analysis match all nine roles and all 11 edges.
The last edge follows the literal upstream code: pointer swaps occur after every
launch, but the download uses `tOut_d`, which now refers to the previous result.
The original source is preserved. A separate test changes only that download
pointer in a temporary copy and checks that the dependency moves to `t7`.

## Gaussian: independent region contracts

See [the Gaussian guide](gaussian/README.md) for the configured `ForwardSub`
boundary and row/column formulas. `tests/test_gaussian_source_model.py` builds
expected cell sets directly from those formulas, without using Clang or the
production dependency builder. For each pair of transfers it checks whether
overlapping cells have an intervening overwrite; it also adds the documented
kernel input-to-output dependencies. It compares all nodes, edges, and byte
ranges for sizes 2, 3, 4, and 5. The default has **20 nodes and 39 edges**.

The committed artifacts must match fresh analysis and that independent oracle.
A test-only change to Fan1's guard is checked against a separately adjusted
coordinate formula. Tests reject cross-thread read/write races, uninitialized
regions, out-of-bounds accesses, missing entry parameters, and oversized region
tracking. The original source is hash-pinned. No numerical solver run occurs.

## Interpretation and limits

These results validate the **configured logical buffer footprints and dependency
policy**. They do not prove arbitrary source-analysis correctness or numerical
correctness of these algorithms. A wrong arithmetic expression or wrong neighbor
selection can leave the same whole-buffer footprint and therefore the same graph.

The configuration assumes successful input loading and supplies sizes/device
limits. CPU work and IO are summarized; input files are not loaded by this audit.
Changing those assumptions requires another scenario and validation.

Simulator JSON does not encode all execution ordering. In HotSpot3D, the source
synchronizes before downloading, but there is no logical data dependency from
`t7` to `t8`: the download reads the previous buffer. Both nodes are graph sinks.
Treating completion of `t8` as completion of the entire workload would therefore
be incorrect. The simulator must define completion over all relevant work and,
if required, separately model CUDA scheduling constraints recorded in metadata.

Compute time, instruction counts, physical DRAM/cache traffic, and measured
performance are outside this validation. A GPU run is not required to perform
these logical-graph checks. It would be a separate numerical/timing validation.
