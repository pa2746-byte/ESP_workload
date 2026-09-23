#!/usr/bin/env python3
"""Export explicit buffer-access models for the four bundled CUDA workloads.

These are source-derived logical models, not trace inference or measured traffic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WORKLOADS = ("vector_add", "pipeline", "fork_join", "reduction")


def build_model(name, cpu="cpu", mem="mem", acc="acc"):
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload: {name}")
    if len({cpu, mem, acc}) != 3 or not all((cpu, mem, acc)):
        raise ValueError("Component names must be nonempty and distinct")
    nodes, edges, descriptions = [], [], []
    producers = {}

    def transfer(buffer, size, src, dst, stage, parents=()):
        node_id = len(nodes)
        nodes.append(dict(id=node_id, vol=size, src_comp=src, dest_comp=dst))
        descriptions.append(dict(id=node_id, buffer=buffer, stage=stage))
        edges.extend(dict(source=p, target=node_id) for p in parents)
        return node_id

    def upload(buffer, size):
        producers[buffer] = transfer(buffer, size, cpu, mem, "upload")

    def kernel(stage, inputs, output, size):
        reads = [transfer(buf, count, mem, acc, stage + ":read",
                          [producers[buf]]) for buf, count in inputs]
        producers[output] = transfer(output, size, acc, mem, stage + ":write", reads)

    if name == "reduction":
        # N=65536, 256 threads/block, 256 partial sums, sizeof(float)=4.
        upload("input", 65536 * 4)
        kernel("reduceStage1", [("input", 65536 * 4)], "partialSums", 256 * 4)
        kernel("reduceStage2", [("partialSums", 256 * 4)], "result", 4)
        output, size = "result", 4
    else:
        size = (1 << 20) * 4
        upload("A", size)
        upload("B", size)
        kernel("vectorAdd", [("A", size), ("B", size)], "C", size)
        output = "C"
        if name == "pipeline":
            kernel("scaleVector", [("C", size)], "D", size)
            output = "D"
        elif name == "fork_join":
            kernel("vectorSub", [("A", size), ("B", size)], "D", size)
            kernel("vectorMul", [("C", size), ("D", size)], "E", size)
            output = "E"
    transfer(output, size, mem, cpu, "download", [producers[output]])
    graph = dict(directed=True, multigraph=False, graph={}, nodes=nodes, edges=edges)
    return graph, descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all",) + WORKLOADS, default="all")
    parser.add_argument("--out-dir", type=Path, default=Path("results/logical_transfers"))
    parser.add_argument("--cpu", default="cpu")
    parser.add_argument("--mem", default="mem")
    parser.add_argument("--acc", default="acc")
    parser.add_argument("--render", action="store_true",
                        help="Also render PNGs using matplotlib and NetworkX")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in WORKLOADS if args.workload == "all" else (args.workload,):
        graph, descriptions = build_model(name, args.cpu, args.mem, args.acc)
        source = Path(__file__).resolve().parent / "workloads" / (name + ".cu")
        metadata = dict(
            workload=name,
            model="explicit source-derived buffer transfers",
            volume_unit="bytes",
            source=str(source.relative_to(source.parent.parent)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            nodes=descriptions,
            assumptions=[
                "Manually specified for the bundled workload sizes; no automatic CUDA parsing.",
                "Each kernel reads each input buffer once and writes its output once.",
                "Edges express whole-buffer readiness, not observed default-stream order.",
                "No compute time, cache effects, physical DRAM measurements, or internal shared-memory transfers.",
                "Source hash records provenance only; update the model if workload source changes.",
            ],
        )
        path = args.out_dir / (name + "_simulator.json")
        path.write_text(json.dumps(graph, indent=2) + "\n")
        (args.out_dir / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {path}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        if args.render:
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("flow_graph_networkx.py")),
                 "--input", str(path), "--name", name + "_simulator",
                 "--out-dir", str(args.out_dir / "graphs")],
                env={**os.environ, "MPLBACKEND": "Agg"}, check=True,
            )


if __name__ == "__main__":
    main()
