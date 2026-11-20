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

`analysis_readiness.json` records preliminary Clang parsing diagnostics where available. Parsing alone does not validate execution, graph dependencies, numerical correctness or GPU compatibility. No new GPU measurements or graphs have been produced for these eight yet.
