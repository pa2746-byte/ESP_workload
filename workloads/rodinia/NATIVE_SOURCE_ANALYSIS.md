# Experimental native source analysis

Three additional Rodinia applications now have complete configured source-execution graphs: **Myocyte (corrected source), Streamcluster (32 points), and MUMmerGPU (corrected pointer-storage build)**. Together with DWT2D, four of the eight additional applications have bounded source scenarios. Heartwall, Huffman, Hybrid Sort and Leukocyte remain pending.

These three cases use a new, explicitly labeled backend, `native_source_execution`. They do not establish support in the older Python AST interpreter, arbitrary-input support, CUDA numerical equivalence, or equivalence to the existing captured builds. No GPU measurements are attached.

## How this backend constructs a graph

Clang preprocesses the selected original translation units and produces ASTs. Source rewriting instruments device lvalues and transforms each launch into bounded CPU execution of its original kernel body. The original complete application drivers execute their input preparation, C++ objects, numerical calculations and host feedback. Kernel arguments evaluate once. Source accesses determine input/output footprints; GPU event counts and profiler values are never consulted.

For these three completed scenarios, the native runtime rejects shared memory, barriers, atomics and non-default streams, as well as uninitialized device reads, ordinary host-pointer kernel arguments, conflicting cross-thread global accesses, dynamic device allocation, unsupported device memory-library calls and concurrent host paths. ASan/UBSan additionally check the native run. Failure leaves `status.json` marked failed and does not publish a new graph as complete.

The simulator graph models **external kernel input reads and global output writes**. Reads of values written earlier by the same thread in that kernel are recorded in `launches[].global_access_footprints[].reads`, but are not invented as dependencies on pre-launch data. `external_reads` identifies input regions. Aggregate copies conservatively cover native object storage, including padding. This boundary model differs from an instruction-level memory-access graph and is labeled separately from the existing AST footprint model.

Dependency construction uses interval partitions, not per-byte Python tables. RAW/WAR/WAW edges follow overlapping regions, and kernel outputs conservatively depend on their external inputs. A CPU-visible download becomes a control dependency of subsequent source operations. Keeping the latest download as a frontier preserves older control dependencies transitively. Default-stream operation metadata retains its immediate predecessor; it does not add raw execution-order chains to the transfer DAG.

## Configured scenarios and source corrections

- **Myocyte:** arguments `1 1 0`, original data and complete driver. Two CAM helpers mistakenly wrote their communication result to `d_finavalu[com_offset]`, overlapping ECC output and leaving `d_com` unwritten. The original captured source is rejected by the cross-thread check. The corrected scenario writes through the declared `d_com` output. Literal percent signs in host timing diagnostics are also escaped to avoid invalid `printf` formats. All changes are hash-pinned in `myocyte/native_source_fixes.json` and applied only to an isolated copy. The corrected CPU source execution produces 39 launches, 312 transfer nodes and 883 edges. This is not a corrected-source GPU run.
- **Streamcluster:** arguments `2 4 2 32 32 16 none output.txt 1`, serial host execution and the original synthetic stream generator. Source bytes match the staged capture, but the input configuration is smaller than its 128-point, 8-dimensional GPU scenario. It produces 128 launches, 1,495 nodes and 4,256 edges on the recorded native environment. C++ classes, virtual stream calls and clustering decisions execute normally. The unused serial-path POSIX barrier object has a platform token; actual pthread barrier/thread execution is rejected.
- **MUMmerGPU:** original tiny reference/query inputs and `-l 1`, with the source's existing `TREETEX=0` pointer-storage option and `UNIX=1`. The captured null-node fix is retained. Two further host corrections prevent division by zero when an unused texture pitch is zero and add a defined return from `addMatchToBuffer`. They are recorded in `mummergpu/native_source_fixes.json`. The original host/suffix-tree drivers and both kernels complete: two launches, 85 nodes and 55 edges. The captured default texture build is not claimed supported by this backend.

All three runs retain source/input hashes, source corrections, compiler/host environment, backend hashes, sanitizer logs, per-launch global footprints and source execution events. `status.json` is the completion gate. Benchmark diagnostic timing printed in stdout is not a GPU measurement and must not be used for enrichment.

## Reproduction

First recover the hash-verified source/input layout using `tools/prepare_remaining_source_cases.py`. Then run, for each case:

```sh
python3 tools/analyze_native_source.py --config workloads/rodinia/myocyte/native_analysis.json --staged-dir /tmp/rodinia-matched-source --out-dir results/rodinia_myocyte_native
python3 tools/analyze_native_source.py --config workloads/rodinia/streamcluster/native_analysis.json --staged-dir /tmp/rodinia-matched-source --out-dir results/rodinia_streamcluster_native
python3 tools/analyze_native_source.py --config workloads/rodinia/mummergpu/native_analysis.json --staged-dir /tmp/rodinia-matched-source --out-dir results/rodinia_mummergpu_native
```

Builds and application writes occur in isolated temporary workspaces. The staged capture sources and repository benchmark sources are not patched in place.

## Limits and verification

Native arithmetic, library math, ABI layout, pitched allocation geometry and source-selected control flow are specific to the recorded CPU environment. GPU floating-point/FMA behavior, CUDA vector ABI, textures and hardware scheduling are not validated. The three completed scenarios use serial host execution and barrier-free kernels. The runtime also contains experimental opt-in checked block/shared-memory/barrier/atomic execution, bounded texture reads, original host-C translation units, generated AVI fixtures and serial stream projection for ongoing remaining-workload experiments. These extensions do not establish completion of Heartwall, Huffman, Hybrid Sort or Leukocyte. Serial stream projection does not validate concurrent CUDA streams. Default configurations retain rejection gates. These graphs cannot inherit measurements from different sources, arguments or compile options. Matching GPU reruns and execution/footprint checks are required before GPU enrichment.

`tests/test_native_source_model.py` checks source-dependent indexing, host feedback, argument side effects, aggregate copies, same-thread intermediate reads, initialized-region tracking, large interval footprints, partial hazards and rejection of races, bounds errors, host-pointer arguments and shared memory. `tests/test_native_rodinia_artifacts.py` checks completion, scenario provenance, corrections, graph integrity and independently specified Myocyte output extents. These checks do not prove application numerical correctness or simulator compatibility.

## Rendered graph artifacts

Run `python3 tools/render_source_graphs.py` after generating the source datasets. This renders DWT2D, Myocyte, Streamcluster and MUMmerGPU into each result directory's `graphs/` folder. The SVG and DOT retain every transfer node and dependency; the bounded PNG is an overview. `render_manifest.json` pins the input JSON hash and counts. These are logical source graphs, without GPU cycles or timings.
