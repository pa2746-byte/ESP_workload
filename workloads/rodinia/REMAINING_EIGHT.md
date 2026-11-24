# Eight additional Rodinia applications

The source packages originate at revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`; each directory retains its provenance. **Four of these eight now have complete bounded source scenarios: DWT2D in the Python AST evaluator, plus Myocyte, Streamcluster and MUMmerGPU in an experimental Clang-instrumented CPU source backend.** The three native scenarios use corrections or smaller/build-specific configurations and do not establish equivalence to the old GPU captures. Heartwall, Huffman, Hybrid Sort and Leukocyte remain pending. Parsing readiness is not graph support. See [native source scope and reproduction](NATIVE_SOURCE_ANALYSIS.md).

## Completed DWT2D source evaluation

`dwt2d/analysis.json` executes the original complete application driver with `-d 64x64 -c 3 -l 1 -5 input.rgb`. It produces 22 logical-transfer nodes and 23 dependency edges, with one component-conversion launch and three forward 5/3 wavelet launches. The graph includes initialization and three copies into the reused backup buffer. `OUTPUT` is undefined for this scenario, so the application does not download or write transform results.

Artifacts are in `results/rodinia_dwt2d/`. `tests/test_dwt2d_graph_contract.py` independently specifies volumes, buffer roles, launch geometry and dependencies from the driver and kernel code. This validates the configured footprint graph, not wavelet numerical correctness or arbitrary inputs.

The evaluator now executes selected source-backed C++ methods, static methods, template specializations, temporary functors, reference-returning helpers, trivial objects and single nonvirtual base subobjects. Shared numeric arrays inside objects retain opaque values and checked bounds; their communication, races and numerical correctness are not simulated. Virtual dispatch, nontrivial construction/destruction, object ABI copies, object values passed/returned by value and unsupported reference lifetimes remain errors.

Two reviewed filename-allocation corrections in `dwt2d/main.cu` postdate the GPU capture. `dwt2d/local_source_fixes.json` records both source hashes. No corrected-source GPU rerun is claimed. The matched captured headers also contain CUDA compatibility changes; the analysis configuration pins the staged source hashes rather than assuming the repository copy is byte-identical.

Reproduce from a fresh staging directory:

```sh
python3 tools/prepare_remaining_source_cases.py --archives /Users/priyalagarwal/Downloads --out-dir /tmp/rodinia-source-reproduction
python3 tools/analyze_captured_source_case.py --case dwt2d --staged-dir /tmp/rodinia-source-reproduction --out-dir results/rodinia_dwt2d
```

The second command checks the recorded source/input manifest, applies only the hash-verified filename fixes, and invokes the source exporter. It does not read profiler events, cycle values or captured launch counts to construct the graph. `source_scenario.json` records the source/capture distinction. GPU enrichment remains a separate step.

## Remaining work in the original AST evaluator / captured scenarios

- Heartwall: AVI input, constant-memory pointer-bearing records, record arrays and frame-dependent tracking state.
- Huffman: complete codebook preparation, shared histogram/prefix-scan values and variable-length output addresses.
- Hybrid Sort: shared histogram/bucket/scatter/merge state and scalable analysis of 4,194,304 floats; raising enumeration limits alone is insufficient.
- Leukocyte: original `CUDA/detect_main.c` driver, AVI/Meschach numerical preparation, tracking decisions and multiple GPU translation units.
- MUMmerGPU: original `src/mummergpu_main.cpp` driver, suffix-tree/query preparation, texture-backed aliases and input-dependent traversal; retain the successful null-node fix and `-l 1` arguments.
- Myocyte: the original captured source has a conflicting CAM output write and an unwritten communication buffer. It remains rejected. The corrected complete native scenario is available separately; the Python AST interpreter still lacks the full numerical/control model. Observed launch counts must not replace solver decisions.
- Streamcluster: original `streamcluster_cuda_cpu.cpp` driver, class/translation-unit identities, virtual input streams and host clustering decisions from GPU costs. The new nonvirtual object subset does not implement its complete object/control model.

Retain complete original application flows. CPU contracts must state source/input assumptions and cannot replace GPU-dependent decisions with guessed launch counts. Unresolved accesses or launches must fail; do not emit partial graphs as completed application graphs or disable bounds/race/unknown-value checks.

## Matched GPU evidence

All eight have successful captures. `captured_scenarios/` pins application arguments and source/input/archive hashes. The three successful bundles are `20261123T005703612561Z.zip` (DWT2D, Heartwall, Leukocyte, Myocyte, Streamcluster), `20261123T012815164507Z.zip` (Huffman, Hybrid Sort), and `20261123T020343856158Z.zip` (MUMmerGPU). Source staging verifies these bytes but does not infer graph dependencies from their traces.

Profiler timing/cycles belong in a separate enrichment layer after source evaluation succeeds. They annotate matched kernel executions rather than every logical read/write node. The captures do not measure per-buffer DRAM traffic or establish suite-wide numerical correctness.
