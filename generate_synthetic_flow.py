#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List


def add_node(nodes: List[Dict], vol: int, source_id: str, dest_id: str) -> int:
    node_id = len(nodes)
    nodes.append(
        {
            "vol": int(vol),
            "source_id": source_id,
            "dest_id": dest_id,
            "id": node_id,
        }
    )
    return node_id


def add_edge(edges: List[Dict], source: int, target: int) -> None:
    edges.append({"source": int(source), "target": int(target)})


def build_dummy_graph(n: int, iters: int, mode: str) -> Dict:
    # float32 vectors
    bytes_per_vec = n * 4
    nodes: List[Dict] = []
    edges: List[Dict] = []

    prev = None

    def chain(node_id: int) -> None:
        nonlocal prev
        if prev is not None:
            add_edge(edges, prev, node_id)
        prev = node_id

    if mode == "once":
        chain(add_node(nodes, bytes_per_vec, "cpu", "acc"))  # H2D a
        chain(add_node(nodes, bytes_per_vec, "cpu", "acc"))  # H2D b

        for _ in range(iters):
            chain(add_node(nodes, 0, "acc", "acc"))  # vecAdd
            chain(add_node(nodes, 0, "acc", "acc"))  # scale
            chain(add_node(nodes, 0, "acc", "acc"))  # copy kernel

        chain(add_node(nodes, bytes_per_vec, "acc", "cpu"))  # D2H out
    else:
        for _ in range(iters):
            chain(add_node(nodes, bytes_per_vec, "cpu", "acc"))  # H2D a
            chain(add_node(nodes, bytes_per_vec, "cpu", "acc"))  # H2D b
            chain(add_node(nodes, 0, "acc", "acc"))  # vecAdd
            chain(add_node(nodes, 0, "acc", "acc"))  # scale
            chain(add_node(nodes, 0, "acc", "acc"))  # copy kernel
            chain(add_node(nodes, bytes_per_vec, "acc", "cpu"))  # D2H out

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges,
    }


def build_fft_graph(fft_size: int, batch: int, iters: int, mode: str) -> Dict:
    # cufftComplex = 2 * float32
    bytes_per_batch = fft_size * batch * 8
    nodes: List[Dict] = []
    edges: List[Dict] = []

    prev = None

    def chain(node_id: int) -> None:
        nonlocal prev
        if prev is not None:
            add_edge(edges, prev, node_id)
        prev = node_id

    if mode == "once":
        chain(add_node(nodes, bytes_per_batch, "cpu", "acc"))  # H2D input
        for _ in range(iters):
            chain(add_node(nodes, 0, "acc", "acc"))  # FFT compute
        chain(add_node(nodes, bytes_per_batch, "acc", "cpu"))  # D2H output
    else:
        for _ in range(iters):
            chain(add_node(nodes, bytes_per_batch, "cpu", "acc"))  # H2D input
            chain(add_node(nodes, 0, "acc", "acc"))  # FFT compute
            chain(add_node(nodes, bytes_per_batch, "acc", "cpu"))  # D2H output

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataflow JSON without Nsight profiling."
    )
    parser.add_argument("--workload", choices=["dummy", "fft"], required=True)
    parser.add_argument("--mode", choices=["once", "per_iter"], default="per_iter")
    parser.add_argument("--iters", type=int, default=50)

    # Dummy params
    parser.add_argument("--n", type=int, default=1 << 24, help="Vector length for dummy")

    # FFT params
    parser.add_argument("--fft-size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)

    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: synthetic_<workload>_<mode>_flow.json)",
    )

    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    if args.workload == "dummy":
        if args.n <= 0:
            raise ValueError("--n must be > 0")
        graph = build_dummy_graph(args.n, args.iters, args.mode)
    else:
        if args.fft_size <= 0 or args.batch <= 0:
            raise ValueError("--fft-size and --batch must be > 0")
        graph = build_fft_graph(args.fft_size, args.batch, args.iters, args.mode)

    out_path = args.out or f"synthetic_{args.workload}_{args.mode}_flow.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)

    print(
        f"Wrote {out_path} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges."
    )


if __name__ == "__main__":
    main()
