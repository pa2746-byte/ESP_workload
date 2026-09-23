# Simulator transfer models

Generate all four models from the repository root (Python standard library only):

```bash
python3 export_workload_transfers.py
```

This command runs on the remote server or any machine with Python 3. No
Mac-generated files, CUDA execution, GPU access, or Nsight trace is required.
The generator applies explicit buffer-access models of the four bundled
workloads; it does not reconstruct arbitrary workloads from a trace.

For JSON and PNG generation together, use the remote virtual environment
where matplotlib and NetworkX are already installed:

```bash
source .venv/bin/activate
python export_workload_transfers.py --render
```

The optional renderer uses the same Python environment as the generator and
reports failures. JSON files already written remain available if rendering
fails. Re-running replaces files with the same names; choose another
`--out-dir` to keep a separate set.

Outputs are in `results/logical_transfers`. Use only `*_simulator.json` as
simulator inputs. The files match the reference shape: `directed`,
`multigraph`, empty `graph`, `nodes`, and `edges`. Nodes contain only `id`,
`vol` (integer bytes), `src_comp`, and `dest_comp`. Edges contain `source`
and `target` node IDs. Actual simulator compatibility is untested because
the simulator is unavailable.

`cpu` is the host, `mem` is accelerator-accessible memory, and `acc` is compute.
Configure names with `--cpu`, `--mem`, and `--acc`. Select one workload
with `--workload fork_join`.

These models are manually specified from the current CUDA sources, not
automatically inferred from Nsight or parsed from CUDA. Vector workloads
use 1,048,576 float32 elements (4,194,304 bytes per buffer). Reduction uses
65,536 float32 inputs, 256 float32 partial sums, and one float32 output.
Update the exporter when workload sizes or buffer accesses change.

Uploads are `cpu -> mem`; kernel input reads are `mem -> acc`; kernel
output writes are `acc -> mem`; downloads are `mem -> cpu`. Each read
waits for its buffer producer; a kernel output write waits for all its
input reads. This represents whole-buffer readiness, not element-level
streaming. Compute duration is not modeled.

Fork-join's add and subtract read A and B independently. Their outputs C
and D feed multiply, whose output E waits for both inputs. No dependency
is added merely because kernels execute in one CUDA stream. Sharing the
`acc` endpoint does not imply multiple physical accelerators; resource
contention is left to the simulator.

Volumes count logical buffer accesses, not measured DRAM traffic. Both
fork branches read A and B, so those reads count twice. Cache reuse,
transaction overhead, registers, and shared-memory traffic are excluded.
Reduction models global transfers for two kernels, not the internal tree.

Separate `*_metadata.json` files explain node IDs by stage and buffer,
record the source hash, and document assumptions. They are not simulator
inputs. The hash records provenance; it does not verify that the manual
model matches future source edits. Existing Nsight exports remain separate
execution-order graphs.

Render in an environment with matplotlib and networkx installed:

```bash
for name in vector_add pipeline fork_join reduction; do
  MPLBACKEND=Agg python3 flow_graph_networkx.py \
    --input "results/logical_transfers/${name}_simulator.json" \
    --name "${name}_simulator" \
    --out-dir results/logical_transfers/graphs
done
```

Run validation:

```bash
python3 -B -m unittest discover -s tests -v
```
