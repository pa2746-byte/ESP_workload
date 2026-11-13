# CFD: complete configured application flow

Source: [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/cfd), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. Licensing: [../LICENSE](../LICENSE). `provenance.json` records vendored file hashes. Analysis configuration and fixtures are local additions.

Scenario: **four-cell synthetic mesh and two iterations**. Expected graph: **102 nodes / 186 edges; seventeen kernel launches**.

Original initialization, step-factor, flux and three-stage time updates. The fixture exercises interior, wing and far-field neighbors; it is not a physically validated CFD benchmark. Constant-memory uploads and device-to-device copies are included. The sole source adaptation guards the iterations macro so analysis can override it; the original default remains 2000.

The original application control flow is analyzed for this explicit scenario. Host summaries are source-hash checked, not executed. This is bounded source analysis, not unrestricted application support or numerical execution. Unknown address/control decisions fail analysis. No GPU timing or simulator validation has been performed for this addition.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/cfd/euler3d.cu \
  --analysis-config workloads/rodinia/cfd/analysis.json \
  --out-dir results/rodinia_cfd --render
```

Requires Clang and Python; rendering also requires matplotlib and networkx. No GPU is needed. Outputs: [`euler3d_simulator.json`](../../../results/rodinia_cfd/euler3d_simulator.json), [`euler3d_metadata.json`](../../../results/rodinia_cfd/euler3d_metadata.json), and [PNG](../../../results/rodinia_cfd/graphs/euler3d_simulator.png). Volumes are logical unique byte footprints, not measured DRAM traffic. The metadata retains byte ranges and CPU-mediated dependencies.

Independent tests in `tests/test_additional_rodinia.py` calculate expected footprints and dependencies separately from the analyzer and check alternative small scenarios and rejection cases.
