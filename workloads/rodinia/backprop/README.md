# Backpropagation: complete configured application flow

Source: [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia/tree/9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4/cuda/backprop), revision `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`. Licensing: [../LICENSE](../LICENSE). `provenance.json` records vendored file hashes. Analysis configuration and fixtures are local additions.

Scenario: **32 inputs, 16 hidden units and one output**. Expected graph: **18 nodes / 28 edges; two kernel launches**.

Original C driver and CUDA training flow, including CPU work between the two GPU phases. Hash-checked CPU summaries propagate the intermediate download into later uploads. Numerical training and CPU latency are not simulated.

The original application control flow is analyzed for this explicit scenario. Host summaries are source-hash checked, not executed. This is bounded source analysis, not unrestricted application support or numerical execution. Unknown address/control decisions fail analysis. No GPU timing or simulator validation has been performed for this addition.

```bash
python3 export_workload_transfers.py \
  --source workloads/rodinia/backprop/backprop_cuda.cu \
  --analysis-config workloads/rodinia/backprop/analysis.json \
  --out-dir results/rodinia_backprop --render
```

Requires Clang and Python; rendering also requires matplotlib and networkx. No GPU is needed. Outputs: [`backprop_cuda_simulator.json`](../../../results/rodinia_backprop/backprop_cuda_simulator.json), [`backprop_cuda_metadata.json`](../../../results/rodinia_backprop/backprop_cuda_metadata.json), and [PNG](../../../results/rodinia_backprop/graphs/backprop_cuda_simulator.png). Volumes are logical unique byte footprints, not measured DRAM traffic. The metadata retains byte ranges and CPU-mediated dependencies.

Independent tests in `tests/test_additional_rodinia.py` calculate expected footprints and dependencies separately from the analyzer and check alternative small scenarios and rejection cases.
