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

Initial matched-source DWT2D evaluation exposed two generic evaluator gaps, now corrected: distinct nested headers called `common.h` need application-relative hash keys, and C `atoi("64x64")` must return 64. The next reported blocker is the unsupported `strstr` call in dimension parsing. These corrections do not constitute completed DWT2D or eight-workload DAG support. Full host flow, kernel footprints, data-dependent addresses, and scalable evaluation still require implementation and checking.
