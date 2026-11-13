# Hardware memory profiling

On the remote NVIDIA machine, from the repository root:

```bash
python3 profile_workload_memory.py
```

Requires `nvcc`, `ncu`, `nvidia-smi`, Git, and Python 3. No Python packages
are required. Each run uses a fresh `results/ncu_<timestamp>` directory.
The script compiles and profiles all four workloads and saves binaries,
reports, CSVs, application/profiler logs, per-kernel hardware JSON, and
run metadata (GPU details, tool versions, commit, dirty status, source
hashes, exact commands, cache policy, replay mode, and clock control).
If a command fails, inspect the named log; failed runs retain metadata.
Do not commit binaries or large reports by default.

Metrics are `dram__bytes_read.sum` and `dram__bytes_write.sum`, collected
per kernel invocation. Unsupported or unavailable counters fail explicitly;
they are never converted to zero. If the log reports `ERR_NVGPUCTRPERM`,
the remote administrator must enable access to NVIDIA performance counters.
The script does not change driver settings or invoke sudo.

`*_hardware.json` compares kernel-level logical input/output bytes with
measured DRAM bytes and links to transfer node IDs from the logical exporter.
Keep `run_metadata.json` with these files to preserve measurement conditions.
This is a sidecar, not a simulator input: `*_simulator.json` retains its exact
reference schema and logical volumes. Per-kernel counters cannot identify
individual buffer contributions, so no measured read total is arbitrarily
divided between A and B. CPU upload/download volumes are outside this
kernel counter comparison; those remain in the logical model and Nsight
Systems traces. No L1/L2 or computation-time model is inferred here.

Default: kernel replay, explicit `--cache-control all`, and `--clock-control
none`. Cache flushing gives a controlled measurement but does not reproduce
warm-cache reuse across kernels. To investigate application cache behavior:

```bash
python3 profile_workload_memory.py --replay-mode application --cache-control none
```

Application replay reruns the entire workload for collection passes. These
are still instrumented measurements, not guaranteed unperturbed execution.
Compare runs only with their cache/replay conditions in mind. The source's
default-stream ordering still applies when profiling, whereas the logical
graph intentionally preserves data independence.

Import an existing base-unit NCU CSV without a GPU:

```bash
python3 profile_workload_memory.py --workload vector_add --csv path/to/vector_add_ncu.csv
```

Imported CSV cache/replay/clock settings are marked unknown. The comparison
uses current explicit workload models; only use matching workload sources.
Both long metric rows and raw wide rows with a units row are supported.
Unknown units, malformed values, duplicate metrics, unexpected kernels,
and incomplete kernel coverage are rejected.

References:
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/

Local checks use synthetic CSV fixtures. Remote GPU collection and actual
simulator execution must be verified separately.
