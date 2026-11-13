# Collect measured GPU cycles and transfer traces

[Open the notebook in Colab](https://colab.research.google.com/github/pa2746-byte/ESP_workload/blob/pa2746-byte-sampleworkloads/notebooks/rodinia_five_cycles.ipynb), choose a GPU runtime, and run its cells. It captures BFS, LavaMD, Particle Filter, K-means and B+tree using their documented source-analysis scenarios. No large sample sweep is performed: one Systems run and one separate Compute run per workload, with possible internal profiler replays.

`cycles.csv` reports `sm__cycles_elapsed.avg` and `sm__cycles_active.avg` in cycles for each kernel. These are hardware-counter values averaged across SM instances. Summing elapsed values across launches is a kernel-only total, **not end-to-end application cycles**: CPU work and host/device copies are excluded and do not share a single SM clock. Nsight Systems timestamps are retained separately for reconstructing event and transfer DAGs. No duration-times-frequency estimate is substituted for hardware counters.

Colab can deny counter access (`ERR_NVGPUCTRPERM`). The notebook keeps the error and any Systems evidence, skips repeated counter attempts after a permission denial, and reports missing cycles. A laptop profiler cannot recover counters that were not captured during execution. Use the same script on a Linux GPU machine with profiling permission when necessary. Profiler availability, compilation and hardware collection have not been verified on Colab during this local change.

The ZIP includes raw `.ncu-rep` reports, raw metric CSV, `.nsys-rep`/SQLite where available, stdout/error logs, exact commands, staged source hashes, repository revision, matched Clang graphs, and `status.json`. Expected launch counts are checked; missing or partial captures are not labeled complete. Numerical result correctness is not certified by matching launch counts.

Builds use staged copies and **never** use the analysis-only headers. Compatibility changes are logged: legacy synchronization API names; K-means device ordinal remapping, modern symbol-copy syntax and ordinary global loads replacing removed texture references; B+tree's explicit zero-valued random query input. K-means preserves the modeled logical reads but its physical cache behavior can differ from the legacy texture implementation. B+tree has the documented pointer-dereference correction in its vendored command parser. CPU contracts used by analysis are not substituted into GPU executions.

For a Linux GPU host:

```bash
python3 tools/capture_rodinia_cycles.py --out-dir /tmp/rodinia-cycles-new-run
```

Use a new output directory for each capture. `--workloads bfs lavamd` selects a subset; `--prepare-only` prepares sources and build commands without claiming GPU execution. The notebook attempts to locate/install tools from available APT repositories but never changes driver permissions.

After capture, compare copy sizes and launch counts first, then construct an observed event/dependency DAG and compare it to the logical graph. Keep measured kernel cycles as separate metadata. Simulator comparisons must account for its compute, bandwidth and contention model; matching the input schema alone does not establish performance agreement.

References: [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), [CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/), [counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
