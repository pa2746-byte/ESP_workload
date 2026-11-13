# Five Rodinia GPU captures — 10 November 2026

Successful Colab capture from repository commit `fb3f01b4e52fd1b724ae22bf89a7777037545003`, using the configured small BFS, LavaMD, Particle Filter, K-means and B+tree scenarios. All five report successful Nsight Systems trace collection and Nsight Compute cycle measurement; all 14 expected kernel launches were captured.

## Files

- [Complete original capture ZIP](20261110T152704530565Z.zip): unchanged user-provided archive, including raw `.nsys-rep`, SQLite, `.ncu-rep`, metric CSVs, logs, staged sources, build manifests, binaries and matched Clang graphs.
- [cycles.csv](cycles.csv): per-launch elapsed and active SM-average cycles.
- [status.json](status.json): capture status, launch counts and summed elapsed SM-average cycles.
- [repository.json](repository.json): source repository revision used for capture.
- [MEASUREMENT_SCOPE.txt](MEASUREMENT_SCOPE.txt): metric interpretation.
- [SHA256SUMS](SHA256SUMS): checksum of the original ZIP.

The extracted JSON/CSV files are exact copies from the archive. To inspect raw reports, extract the ZIP locally; no captured executable needs to be run to inspect the results.

## Recorded elapsed cycles

The sum of `sm__cycles_elapsed.avg` across the captured kernel launches is:

- BFS: 24,303 cycles across 8 launches.
- LavaMD: 249,301 cycles across 1 launch.
- Particle Filter: 3,761.2 cycles across 1 launch.
- K-means: 8,623.2 cycles across 3 launches.
- B+tree: 3,682.2 cycles across 1 launch.

These are hardware-counter measurements averaged across SM instances, not end-to-end application cycles. CPU work and host/device copies are excluded. Fractional values are expected from averaging. Nsight Systems and Nsight Compute captured separate executions. The profiler recorded uncontrolled caches and unmodified clocks: this single capture is preliminary evidence, not a stable performance benchmark. Numerical correctness and every DAG dependency edge have not been independently validated by these counts.

The earlier capture `20261110T151607829486Z` had valid Systems traces but no cycles because of an incompatible CLI option; it is not the successful measurement set published here.

See [capture instructions](../../../workloads/rodinia/CYCLES_CAPTURE.md) and the [Colab notebook](../../../notebooks/rodinia_five_cycles.ipynb).
