# SRAD v1: complete configured application flow

Source: [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/srad/srad_v1), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. Licensing: [../LICENSE](../LICENSE). `provenance.json` records vendored file hashes. Analysis configuration and fixtures are local additions.

Scenario: **4 × 4 image and two iterations**. Expected graph: **66 nodes / 169 edges; ten kernel launches**.

Original extraction, preparation, reduction, coefficient, update and compression stages. Source-built integer neighbor tables are evaluated. Image values remain unknown under explicit image-loading contracts; no image quality or shared-memory numerical correctness is claimed. Scalar downloads feed the subsequent kernel dependencies.

The original application control flow is analyzed for this explicit scenario. Host summaries are source-hash checked, not executed. This is bounded source analysis, not unrestricted application support or numerical execution. Unknown address/control decisions fail analysis. No GPU timing or simulator validation has been performed for this addition.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/srad/main.cu \
  --analysis-config workloads/rodinia/srad/analysis.json \
  --out-dir results/rodinia_srad --render
```

Requires Clang and Python; rendering also requires matplotlib and networkx. No GPU is needed. Outputs: [`main_simulator.json`](../../../results/rodinia_srad/main_simulator.json), [`main_metadata.json`](../../../results/rodinia_srad/main_metadata.json), and [PNG](../../../results/rodinia_srad/graphs/main_simulator.png). Volumes are logical unique byte footprints, not measured DRAM traffic. The metadata retains byte ranges and CPU-mediated dependencies.

Independent tests in `tests/test_additional_rodinia.py` calculate expected footprints and dependencies separately from the analyzer and check alternative small scenarios and rejection cases.
