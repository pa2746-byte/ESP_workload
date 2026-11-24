FINAL SOURCE-DERIVED DAGs — FIVE RODINIA WORKLOADS

Each workload folder contains:
  DAG.png: graph image for viewing or presenting.
  simulator.json: logical transfer/dependency graph.
  metadata.json: source configuration, buffer accesses and assumptions.

These are the Clang-source-derived logical graphs for our configured small scenarios, not graphs reconstructed from GPU traces. Nodes aggregate logical buffer reads/writes and explicit copies; they do not represent measured physical DRAM transactions. Edges express modeled dependencies, including conservative host-control ordering.

Scenarios: BFS four-node chain; LavaMD one box/100 particles; Particle Filter 16x16 video/two frames/one particle; K-means one point/one feature/one cluster; B+tree one key/one lookup/order 4.

- BFS: 58 nodes, 209 edges.
- LavaMD: 10 nodes, 10 edges.
- Particle_Filter: 14 nodes, 17 edges.
- Kmeans: 17 nodes, 25 edges.
- Btree: 12 nodes, 10 edges.

A separate GPU capture successfully recorded expected kernel counts and measured kernel cycles for these five scenarios. It does not establish every dependency edge, numerical correctness, or simulator compatibility. Cycle measurements are not embedded in these DAGs.

Repository branch: pa2746-byte-sampleworkloads
https://github.com/pa2746-byte/ESP_workload
Analyzer changes and limits are documented in the repository README.
