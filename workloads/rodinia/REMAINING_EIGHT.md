# Remaining eight application expansion

The eight upstream source packages are imported from revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. Each directory records source hashes in `provenance.json`. Importing sources does not establish analyzer support. The supported-case count remains 15 until configurations, source-derived graphs and independent validation pass.

## Implementation requirements from source inspection

- DWT2D: image loading, templated wavelet kernels, dynamic shared memory and shared value propagation.
- Heartwall: AVI frame input, arrays of records containing device pointers, constant-memory pointer records, and frame-dependent tracking.
- Huffman: codebook preparation, variable-length output indexing, prefix scans and dynamic shared memory.
- Hybrid Sort: histogram/bucket shared state, data-dependent scatter offsets and merge passes.
- Leukocyte: video input, Meschach numerical preparation and detection/tracking control flow across files.
- MUMmerGPU: suffix-tree/query preparation, texture-backed tree traversal and input-dependent addresses.
- Myocyte: adaptive solver control, numerical kernel values and multiple integration stages.
- Streamcluster: class-based input streams, clustering decisions and GPU-cost feedback into host control.

Retain original complete application drivers. Select nonempty small inputs that actually execute the relevant GPU path. CPU contracts must document source/input assumptions and cannot replace a GPU-dependent decision with a guessed launch count. If the evaluator cannot resolve an access or launch, report the gap rather than emit a purported complete graph. Existing race checks must remain enabled.

`analysis_readiness.json` records preliminary Clang parsing diagnostics where available. Parsing alone does not validate execution, graph dependencies, numerical correctness or GPU compatibility. GPU captures are now available for all eight matched scenarios (see below); source-derived graphs remain pending.

## Matched GPU inputs for source analysis

All eight now have successful GPU captures for the recorded scenarios. `captured_scenarios/` records the exact application arguments, source/input hashes and capture archive hashes. `tools/prepare_remaining_source_cases.py --archives <capture-zip-directory> --out-dir <fresh-directory>` verifies and restores those exact sources and inputs. It does not use traces to infer source dependencies, and does not emit DAGs.

The successful bundles are `20261123T005703612561Z.zip` (DWT2D, Heartwall, Leukocyte, Myocyte, Streamcluster), `20261123T012815164507Z.zip` (Huffman, Hybrid Sort), and `20261123T020343856158Z.zip` (MUMmerGPU). MUMmerGPU uses the recorded null-node fix and `-l 1`; source analysis must include that fix and argument.

Initial matched-source DWT2D evaluation exposed two generic evaluator gaps, now corrected: distinct nested headers called `common.h` need application-relative hash keys, and C `atoi("64x64")` must return 64. String parsing, bounded host string writes, argument-vector advancement and pinned host allocation now progress into the transform calls. These corrections do not constitute completed DWT2D or eight-workload DAG support. Full host flow, kernel footprints, data-dependent addresses, and scalable evaluation still require implementation and checking.


The next evaluator pass adds source-function overload resolution, non-type template argument identities/substitution, device-helper thread context, launch-bound parsing, and byte-size-preserving views of opaque shared arrays. Shared values remain unknown: this does not model shared-memory communication or physical traffic. The matched DWT2D evaluation still does not complete, and no finished DAG is claimed.

Bounded string evaluation detected two upstream DWT2D filename allocations missing their NUL terminator. `dwt2d/main.cu` now allocates the extra byte; `dwt2d/local_source_fixes.json` records the before/after hashes. The existing GPU capture predates this source correction. Input and kernel algorithm changes are not implied by these filename fixes, but corrected-source GPU execution has not been measured.

The current DWT2D `analysis_in_progress.json` is an experimental configuration, not a supported completed graph. Against the staged captured input and corrected filename allocations, it evaluates `c_CopySrcToComponents<int>` (16 blocks of 256 threads) and then reaches the wavelet kernel and stops at the unsupported class-template method `FDWT53<…>::run`. `evaluation_readiness.json` records the boundary. The configuration expects the original `../../data/dwt2d/input.rgb` relative layout restored by the staging tool.


## Analyzer review before further workload reruns

The evaluator is a bounded interpreter of selected Clang AST nodes, not a general CUDA interpreter. Passing Clang syntax checking does not establish that the evaluator can execute the application's host and device flow.

The reviewed fixes cover scalar const-reference temporaries (evaluated once with call-scoped storage), preserving thread/block context in device helpers, overloaded/non-type-template function identities, pinned-host allocation cleanup, known bounded host strings, and integer address operations. Small regression cases check these semantics before workload reruns.

Remaining architectural gaps should be addressed as grouped features:

- **Objects and methods:** DWT2D's wavelet implementation uses class templates and methods; Streamcluster uses classes and virtual input streams. Function-name dispatch alone is insufficient. Method identity, receiver storage and object lifetime need explicit support.
- **Shared/device state:** Shared values are currently opaque. This can express address-independent footprints but cannot resolve histograms, prefix scans, scatter addresses or adaptive decisions when they depend on those values. Huffman and Hybrid Sort require these capabilities; Myocyte also requires numerical/control-flow fidelity.
- **Pointer-bearing records and textures:** Heartwall, Leukocyte and MUMmerGPU use nested pointer/texture-backed structures. Their aliases and byte extents must survive copies and constant-memory records.
- **Scale:** The configured interpreter caps enumerated threads at 262144 and tracked device bytes at 1048576. Hybrid Sort's captured scenario has 4194304 floats. Raising limits alone does not provide scalable footprint analysis; symbolic ranges or validated repeated-loop summaries are needed.
- **Cross-file identity:** Overloads are disambiguated by Clang type and template arguments; full namespace/class/translation-unit identity still requires care. Ambiguous definitions continue to fail rather than silently selecting one.

No one of these limits should be bypassed by substituting captured launch counts or an empty kernel body. GPU metrics annotate independently generated source graphs; they are not the source of their logical dependencies.
