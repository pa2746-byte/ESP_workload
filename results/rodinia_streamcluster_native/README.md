# streamcluster: experimental source-execution graph

`source_simulator.json` is a logical external-kernel-input/output DAG derived from the original complete application driver under Clang-instrumented CPU execution. `source_metadata.json` records scope, source/input hashes, corrections, configuration, compiler environment and full global-access footprints. `source_execution_events.jsonl` is source-run evidence, not GPU profiler data. `status.json` must say `complete`.

This scenario uses corrected source, smaller inputs, or a different build option from the existing capture; GPU measurements are not attached. Native CPU arithmetic and ABI are not validated as CUDA equivalents. See [scope and reproduction](../../workloads/rodinia/NATIVE_SOURCE_ANALYSIS.md).

Graph renderings: [PNG preview](graphs/source_simulator.png), [zoomable SVG](graphs/source_simulator.svg), and [DOT source](graphs/source_simulator.dot). All transfers and dependencies are retained. GPU measurements are not attached.
