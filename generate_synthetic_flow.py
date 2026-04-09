#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def direction_from_src_dst(source_id: str, dest_id: str) -> str:
    if source_id == "cpu" and dest_id == "acc":
        return "h2d"
    if source_id == "acc" and dest_id == "cpu":
        return "d2h"
    if source_id == "acc" and dest_id == "acc":
        return "compute"
    if source_id == "cpu" and dest_id == "cpu":
        return "h2h"
    return "unknown"


def add_node(nodes: List[Dict], vol: int, source_id: str, dest_id: str, **extra) -> int:
    node_id = len(nodes)
    vol_int = int(vol)
    node = {
        "vol": vol_int,
        "vol_bytes": vol_int,
        "vol_mb": round(vol_int / (1024 * 1024), 6) if vol_int > 0 else 0.0,
        "source_id": source_id,
        "dest_id": dest_id,
        "direction": direction_from_src_dst(source_id, dest_id),
        "id": node_id,
    }
    node.update(extra)
    nodes.append(node)
    return node_id


def add_edge(edges: List[Dict], source: int, target: int, vol_bytes: int = 0) -> None:
    vol_mb = round(vol_bytes / (1024 * 1024), 6) if vol_bytes > 0 else 0.0
    edges.append({
        "source": int(source),
        "target": int(target),
        "vol_bytes": vol_bytes,
        "vol_mb": vol_mb,
    })


def _direction_key(source_id: str, dest_id: str) -> str:
    return f"{source_id}->{dest_id}"


def summarize_transfer_totals(nodes: List[Dict]) -> Dict:
    def total_for(direction: str) -> int:
        return sum(
            int(n.get("vol_bytes", n.get("vol", 0)))
            for n in nodes
            if n.get("direction") == direction
        )

    h2d = total_for("h2d")
    d2h = total_for("d2h")
    d2d = total_for("d2d")
    h2h = total_for("h2h")
    return {
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "d2d_bytes": d2d,
        "h2h_bytes": h2h,
        "total_transfer_bytes": h2d + d2h + d2d + h2h,
    }


def summarize_edge_transfer_totals(edges: List[Dict]) -> Dict:
    total = sum(int(e.get("vol_bytes", 0)) for e in edges)
    return {"total_edge_vol_bytes": total}


def load_calibration(calibrate_from: str) -> Dict:
    path = Path(calibrate_from)
    if not path.exists():
        alt = Path(f"{calibrate_from}_flow.json")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"Could not find calibration source '{calibrate_from}' or '{alt}'."
            )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    dir_to_vols: Dict[str, List[int]] = {}
    dir_to_count: Dict[str, int] = {}
    subtype_count: Dict[str, int] = {}
    for n in nodes:
        src = n.get("source_id", "acc")
        dst = n.get("dest_id", "acc")
        key = _direction_key(src, dst)
        dir_to_count[key] = dir_to_count.get(key, 0) + 1
        v = int(n.get("vol_bytes", n.get("vol", 0)))
        if v > 0:
            dir_to_vols.setdefault(key, []).append(v)
        st = n.get("subtype")
        if st:
            subtype_count[st] = subtype_count.get(st, 0) + 1

    median_vol = {
        k: int(statistics.median(vs)) for k, vs in dir_to_vols.items() if vs
    }
    return {
        "source_file": str(path),
        "median_vol_by_direction": median_vol,
        "count_by_direction": dir_to_count,
        "subtype_count": subtype_count,
    }


def build_dummy_graph(n: int, iters: int, mode: str, calib: Optional[Dict] = None) -> Dict:
    # float32 vectors
    default_bytes = n * 4
    h2d_bytes = (
        calib["median_vol_by_direction"].get("cpu->acc", default_bytes)
        if calib
        else default_bytes
    )
    d2h_bytes = (
        calib["median_vol_by_direction"].get("acc->cpu", default_bytes)
        if calib
        else default_bytes
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    prev = None

    def chain(node_id: int, edge_vol: int = 0) -> None:
        nonlocal prev
        if prev is not None:
            add_edge(edges, prev, node_id, vol_bytes=edge_vol)
        prev = node_id

    if mode == "once":
        chain(
            add_node(nodes, h2d_bytes, "cpu", "acc",
                     op_type="memcpy", subtype="h2d_copy", op_name="H2D_a"),
        )
        chain(
            add_node(nodes, h2d_bytes, "cpu", "acc",
                     op_type="memcpy", subtype="h2d_copy", op_name="H2D_b"),
            edge_vol=h2d_bytes,
        )

        for i in range(iters):
            # First kernel after H2D (or after previous iteration's last kernel)
            prev_vol = h2d_bytes if i == 0 else h2d_bytes
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="vecAdd"),
                edge_vol=prev_vol,
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="scaleKernel"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="copyKernel"),
                edge_vol=h2d_bytes,
            )

        chain(
            add_node(nodes, d2h_bytes, "acc", "cpu",
                     op_type="memcpy", subtype="d2h_copy", op_name="D2H_out"),
            edge_vol=d2h_bytes,
        )
    else:
        for _ in range(iters):
            chain(
                add_node(nodes, h2d_bytes, "cpu", "acc",
                         op_type="memcpy", subtype="h2d_copy", op_name="H2D_a"),
                edge_vol=0,  # iteration boundary: ordering only
            )
            chain(
                add_node(nodes, h2d_bytes, "cpu", "acc",
                         op_type="memcpy", subtype="h2d_copy", op_name="H2D_b"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="vecAdd"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="scaleKernel"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="compute_kernel", op_name="copyKernel"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, d2h_bytes, "acc", "cpu",
                         op_type="memcpy", subtype="d2h_copy", op_name="D2H_out"),
                edge_vol=d2h_bytes,
            )

    transfer_totals = summarize_transfer_totals(nodes)
    edge_totals = summarize_edge_transfer_totals(edges)
    return {
        "directed": True,
        "multigraph": False,
        "graph": {
            "workload": "dummy",
            "mode": mode,
            "params": {"n": n, "iters": iters},
            "calibrated_from": calib.get("source_file") if calib else None,
            "transfer_totals": transfer_totals,
            "edge_transfer_totals": edge_totals,
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_fft_graph(
    fft_size: int, batch: int, iters: int, mode: str, calib: Optional[Dict] = None
) -> Dict:
    # cufftComplex = 2 * float32
    default_bytes = fft_size * batch * 8
    h2d_bytes = (
        calib["median_vol_by_direction"].get("cpu->acc", default_bytes)
        if calib
        else default_bytes
    )
    d2h_bytes = (
        calib["median_vol_by_direction"].get("acc->cpu", default_bytes)
        if calib
        else default_bytes
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    prev = None

    def chain(node_id: int, edge_vol: int = 0) -> None:
        nonlocal prev
        if prev is not None:
            add_edge(edges, prev, node_id, vol_bytes=edge_vol)
        prev = node_id

    if mode == "once":
        chain(
            add_node(nodes, h2d_bytes, "cpu", "acc",
                     op_type="memcpy", subtype="h2d_copy", op_name="H2D_fft_input"),
        )
        for _ in range(iters):
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="library_fft_kernel",
                         op_name="cufftExecC2C_forward"),
                edge_vol=h2d_bytes,
            )
        chain(
            add_node(nodes, d2h_bytes, "acc", "cpu",
                     op_type="memcpy", subtype="d2h_copy", op_name="D2H_fft_output"),
            edge_vol=d2h_bytes,
        )
    else:
        for _ in range(iters):
            chain(
                add_node(nodes, h2d_bytes, "cpu", "acc",
                         op_type="memcpy", subtype="h2d_copy", op_name="H2D_fft_input"),
                edge_vol=0,  # iteration boundary: ordering only
            )
            chain(
                add_node(nodes, 0, "acc", "acc",
                         op_type="kernel", subtype="library_fft_kernel",
                         op_name="cufftExecC2C_forward"),
                edge_vol=h2d_bytes,
            )
            chain(
                add_node(nodes, d2h_bytes, "acc", "cpu",
                         op_type="memcpy", subtype="d2h_copy", op_name="D2H_fft_output"),
                edge_vol=d2h_bytes,
            )

    transfer_totals = summarize_transfer_totals(nodes)
    edge_totals = summarize_edge_transfer_totals(edges)
    return {
        "directed": True,
        "multigraph": False,
        "graph": {
            "workload": "fft",
            "mode": mode,
            "params": {"fft_size": fft_size, "batch": batch, "iters": iters},
            "calibrated_from": calib.get("source_file") if calib else None,
            "transfer_totals": transfer_totals,
            "edge_transfer_totals": edge_totals,
        },
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
    parser.add_argument(
        "--calibrate-from",
        type=str,
        default="",
        help="Measured flow JSON path or prefix (e.g., nsys_dummy_per_iter).",
    )

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
    calib = load_calibration(args.calibrate_from) if args.calibrate_from else None

    if args.workload == "dummy":
        if args.n <= 0:
            raise ValueError("--n must be > 0")
        graph = build_dummy_graph(args.n, args.iters, args.mode, calib=calib)
    else:
        if args.fft_size <= 0 or args.batch <= 0:
            raise ValueError("--fft-size and --batch must be > 0")
        graph = build_fft_graph(args.fft_size, args.batch, args.iters, args.mode, calib=calib)

    out_path = args.out or f"synthetic_{args.workload}_{args.mode}_flow.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)

    print(
        f"Wrote {out_path} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges."
    )
    if calib:
        print(
            "Calibrated using "
            f"{calib['source_file']} with median volumes {calib['median_vol_by_direction']}"
        )


if __name__ == "__main__":
    main()
