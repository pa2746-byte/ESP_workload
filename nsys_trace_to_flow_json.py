# save as nsys_trace_to_flow_json.py
import csv
import json
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path

def pick_col(cols, candidates, required=True):
    lower = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(lower):
            if cand in c:
                return cols[i]
    if required:
        raise RuntimeError(f"Missing required column among {candidates}. Found: {cols}")
    return None

def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def _is_cpu_mem_kind(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in {"pinned", "pageable", "host", "cpu"}

def _is_acc_mem_kind(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in {"device", "acc", "gpu"}

def map_src_dst(op_name: str, src_mem_kind: str = "", dst_mem_kind: str = ""):
    # Prefer explicit memory-kind columns when available.
    if _is_cpu_mem_kind(src_mem_kind) and _is_acc_mem_kind(dst_mem_kind):
        return "cpu", "acc"
    if _is_acc_mem_kind(src_mem_kind) and _is_cpu_mem_kind(dst_mem_kind):
        return "acc", "cpu"
    if _is_acc_mem_kind(src_mem_kind) and _is_acc_mem_kind(dst_mem_kind):
        return "acc", "acc"
    if _is_cpu_mem_kind(src_mem_kind) and _is_cpu_mem_kind(dst_mem_kind):
        return "cpu", "cpu"

    # Fallback to parsing operation name (handle hyphens/underscores/spaces).
    raw = (op_name or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", raw)

    if "htod" in raw or "h2d" in raw or "host to device" in s:
        return "cpu", "acc"
    if "dtoh" in raw or "d2h" in raw or "device to host" in s:
        return "acc", "cpu"
    if "dtod" in raw or "d2d" in raw or "device to device" in s or "p2p" in raw:
        return "acc", "acc"
    if "memset" in raw:
        return "acc", "acc"

    # Kernels/default
    return "acc", "acc"

def parse_volume(row: dict, bytes_col: str):
    if not bytes_col:
        return 0
    raw = row.get(bytes_col, "")
    if raw is None or raw == "":
        return 0
    val = float(str(raw).replace(",", ""))
    col_lower = bytes_col.lower()
    if "(mb)" in col_lower:
        return int(val * 1024 * 1024)
    if "(kb)" in col_lower:
        return int(val * 1024)
    if "(gb)" in col_lower:
        return int(val * 1024 * 1024 * 1024)
    return int(val)

def classify_op(op_name: str, src: str, dst: str):
    raw = (op_name or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", raw)

    if "memset" in raw:
        return "memset", "memset"

    if "memcpy" in raw or "copy" in s:
        if src == "cpu" and dst == "acc":
            return "memcpy", "h2d_copy"
        if src == "acc" and dst == "cpu":
            return "memcpy", "d2h_copy"
        if src == "acc" and dst == "acc":
            return "memcpy", "d2d_copy"
        if src == "cpu" and dst == "cpu":
            return "memcpy", "h2h_copy"
        return "memcpy", "copy"

    if "fft" in s or "cufft" in s:
        return "kernel", "library_fft_kernel"

    return "kernel", "compute_kernel"

def direction_from_subtype(op_type: str, subtype: str):
    if op_type in {"kernel", "memset"}:
        return "compute"
    if subtype == "h2d_copy":
        return "h2d"
    if subtype == "d2h_copy":
        return "d2h"
    if subtype == "d2d_copy":
        return "d2d"
    if subtype == "h2h_copy":
        return "h2h"
    return "unknown"

def aggregate_semantic(raw_nodes):
    stage_to_endpoints = {
        "h2d": ("cpu", "acc"),
        "d2h": ("acc", "cpu"),
        "d2d": ("acc", "acc"),
        "h2h": ("cpu", "cpu"),
        "compute": ("acc", "acc"),
        "unknown": ("acc", "acc"),
    }

    agg_nodes = []
    for n in raw_nodes:
        stage = n.get("direction", "unknown")
        if stage not in stage_to_endpoints:
            stage = "unknown"
        if not agg_nodes or agg_nodes[-1]["direction"] != stage:
            src, dst = stage_to_endpoints[stage]
            agg_nodes.append(
                {
                    "id": len(agg_nodes),
                    "vol": 0,
                    "vol_bytes": 0,
                    "vol_mb": 0.0,
                    "source_id": src,
                    "dest_id": dst,
                    "direction": stage,
                    "op_type": "semantic_stage",
                    "subtype": f"{stage}_stage",
                    "op_name": stage.upper(),
                    "duration_ns": 0,
                    "duration_cycles": 0,
                    "cycles_at_end": 0,
                    "event_count": 0,
                    "op_names": [],
                }
            )

        cur = agg_nodes[-1]
        cur["vol"] += int(n.get("vol_bytes", n.get("vol", 0)))
        cur["vol_bytes"] += int(n.get("vol_bytes", n.get("vol", 0)))
        cur["duration_ns"] += int(n.get("duration_ns", 0))
        cur["duration_cycles"] += int(n.get("duration_cycles", 0))
        # cycles_at_end is the absolute interrupt point — keep the latest one.
        cur["cycles_at_end"] = max(cur["cycles_at_end"], int(n.get("cycles_at_end", 0)))
        cur["event_count"] += 1
        opn = n.get("op_name", "")
        if opn and len(cur["op_names"]) < 10 and opn not in cur["op_names"]:
            cur["op_names"].append(opn)

    for n in agg_nodes:
        n["vol_mb"] = round(n["vol_bytes"] / (1024 * 1024), 6) if n["vol_bytes"] > 0 else 0.0
        if n["duration_cycles"] == 0:
            del n["duration_cycles"]  # omit if no clock was provided
        if n["cycles_at_end"] == 0:
            del n["cycles_at_end"]

    agg_edges = [{"source": i - 1, "target": i} for i in range(1, len(agg_nodes))]
    return agg_nodes, agg_edges

def compute_transfer_totals(nodes):
    totals = {"h2d_bytes": 0, "d2h_bytes": 0, "d2d_bytes": 0, "h2h_bytes": 0}
    for n in nodes:
        direction = n.get("direction")
        vol = int(n.get("vol_bytes", n.get("vol", 0)))
        if direction == "h2d":
            totals["h2d_bytes"] += vol
        elif direction == "d2h":
            totals["d2h_bytes"] += vol
        elif direction == "d2d":
            totals["d2d_bytes"] += vol
        elif direction == "h2h":
            totals["h2h_bytes"] += vol
    totals["total_transfer_bytes"] = (
        totals["h2d_bytes"] + totals["d2h_bytes"] + totals["d2d_bytes"] + totals["h2h_bytes"]
    )
    return totals

def build_stream_dag_edges(nodes, gap_threshold_ns=100_000):
    """
    Build DAG edges for a multi-stream trace.

    Rule 1 — within each stream, consecutive ops are sequential.
    Rule 2 — cross-stream: add edge u→v when u is the last op that finished
              on a *different* stream before v started, and the gap is within
              gap_threshold_ns (i.e. v's stream was waiting on u via an event).

    After collecting all candidate edges a transitive reduction removes edges
    that are implied by a longer path, leaving only the true dependencies.
    """
    # Group node indices by stream (nodes are already sorted by start_ns).
    stream_ops = defaultdict(list)
    for idx, n in enumerate(nodes):
        stream_ops[n.get("stream_id", -1)].append(idx)

    edge_set = set()

    # Rule 1: within-stream sequential edges.
    for ops in stream_ops.values():
        for j in range(1, len(ops)):
            edge_set.add((ops[j - 1], ops[j]))

    # Rule 2: cross-stream dependency edges.
    for stream_i, ops_i in stream_ops.items():
        for idx_v in ops_i:
            start_v = nodes[idx_v].get("start_ns", 0)
            for stream_j, ops_j in stream_ops.items():
                if stream_j == stream_i:
                    continue
                # Walk backwards through stream_j to find the last op that
                # ended at or before start_v.
                for idx_u in reversed(ops_j):
                    end_u = (nodes[idx_u].get("start_ns", 0) +
                             nodes[idx_u].get("duration_ns", 0))
                    if end_u <= start_v:
                        if (start_v - end_u) <= gap_threshold_ns:
                            edge_set.add((idx_u, idx_v))
                        break  # only the most-recent op on this stream

    # Transitive reduction: drop edge (u, v) when v is already reachable
    # from u via another path.
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)

    def reachable_without(src, target, skip_edge):
        """DFS: is target reachable from src ignoring the direct skip_edge?"""
        visited = set()
        stack = [src]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in adj[node]:
                if (node, nb) == skip_edge:
                    continue
                if nb == target:
                    return True
                stack.append(nb)
        return False

    reduced = {
        (u, v) for u, v in edge_set
        if not reachable_without(u, v, skip_edge=(u, v))
    }

    return [{"source": u, "target": v} for u, v in sorted(reduced)]


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nsight CUDA trace CSV to flow JSON."
    )
    parser.add_argument("prefix", help="File prefix, e.g. nsys_dummy_per_iter")
    parser.add_argument(
        "--aggregation",
        choices=["raw", "semantic", "stream_dag"],
        default="raw",
        help=(
            "raw = one node per trace event, linear chain edges. "
            "semantic = collapsed stage nodes, linear chain edges. "
            "stream_dag = one node per trace event, DAG edges inferred from "
            "stream IDs and timing (use for multi-stream workloads)."
        ),
    )
    parser.add_argument(
        "--min-transfer-bytes",
        type=int,
        default=0,
        help="Drop tiny memcpy events below this threshold before graph build.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path (default: <prefix>_flow.json or *_semantic_flow.json).",
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=float,
        default=None,
        help=(
            "GPU SM clock in MHz used to convert duration_ns → duration_cycles. "
            "Pass the boost clock (e.g. 1590 for T4, 1410 for A100). "
            "If omitted, duration_cycles is not added to nodes."
        ),
    )
    args = parser.parse_args()

    prefix = args.prefix
    trace_csv = Path(f"{prefix}_trace_cuda_gpu_trace.csv")
    if args.out:
        out_json = Path(args.out)
    else:
        suffix = {
            "raw": "_flow.json",
            "semantic": "_semantic_flow.json",
            "stream_dag": "_stream_dag_flow.json",
        }.get(args.aggregation, "_flow.json")
        out_json = Path(f"{prefix}{suffix}")

    if not trace_csv.exists():
        raise FileNotFoundError(f"Missing {trace_csv}")

    with trace_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        name_col = pick_col(cols, ["name", "operation", "demangled"], required=False)
        start_col = pick_col(cols, ["start", "timestamp"], required=False)
        duration_col = pick_col(cols, ["duration"], required=False)
        stream_col = pick_col(cols, ["strm", "stream"], required=False)
        bytes_col = pick_col(cols, ["bytes", "size"], required=False)
        src_mem_kind_col = pick_col(cols, ["srcmemkd", "src mem", "source"], required=False)
        dst_mem_kind_col = pick_col(cols, ["dstmemkd", "dst mem", "dest"], required=False)
        grid_x_col = pick_col(cols, ["grdx"], required=False)
        grid_y_col = pick_col(cols, ["grdy"], required=False)
        grid_z_col = pick_col(cols, ["grdz"], required=False)
        block_x_col = pick_col(cols, ["blkx"], required=False)
        block_y_col = pick_col(cols, ["blky"], required=False)
        block_z_col = pick_col(cols, ["blkz"], required=False)

        rows = list(reader)

    # Sort by start time if available
    if start_col:
        rows.sort(key=lambda r: to_int(r.get(start_col), 0))

    nodes = []
    edges = []
    fallback_name_mapping_count = 0
    transfer_totals = {"h2d_bytes": 0, "d2h_bytes": 0, "d2d_bytes": 0, "h2h_bytes": 0}

    # Baseline timestamp: start of the first event (rows are sorted by start time).
    first_start_ns = to_int(rows[0].get(start_col), 0) if (rows and start_col) else 0

    for i, r in enumerate(rows):
        op = r.get(name_col, "") if name_col else ""
        src_mem = r.get(src_mem_kind_col, "") if src_mem_kind_col else ""
        dst_mem = r.get(dst_mem_kind_col, "") if dst_mem_kind_col else ""
        src, dst = map_src_dst(op, src_mem, dst_mem)
        if not src_mem_kind_col and not dst_mem_kind_col:
            fallback_name_mapping_count += 1
        vol = parse_volume(r, bytes_col) if bytes_col else 0
        op_type, subtype = classify_op(op, src, dst)
        direction = direction_from_subtype(op_type, subtype)
        start_ns = to_int(r.get(start_col), 0) if start_col else 0
        duration_ns = to_int(r.get(duration_col), 0) if duration_col else 0
        end_ns = start_ns + duration_ns
        stream_id = to_int(r.get(stream_col), -1) if stream_col else -1
        vol_mb = round(vol / (1024 * 1024), 6) if vol > 0 else 0.0

        if op_type == "memcpy" and vol < args.min_transfer_bytes:
            continue

        node = {
            "id": i,
            "vol": vol,
            "vol_bytes": vol,
            "vol_mb": vol_mb,
            "source_id": src,
            "dest_id": dst,
            "direction": direction,
            "op_type": op_type,
            "subtype": subtype,
            "op_name": op,
            "start_ns": start_ns,
            "duration_ns": duration_ns,
            "stream_id": stream_id,
        }

        if args.gpu_clock_mhz and duration_ns > 0:
            node["duration_cycles"] = int(duration_ns * args.gpu_clock_mhz / 1000.0)
            # Cycles elapsed from trace start to the END of this node — the
            # interrupt fires here, and this is how many cycles have run total.
            node["cycles_at_end"] = int((end_ns - first_start_ns) * args.gpu_clock_mhz / 1000.0)

        if op_type == "kernel":
            node["grid"] = {
                "x": to_int(r.get(grid_x_col), 0) if grid_x_col else 0,
                "y": to_int(r.get(grid_y_col), 0) if grid_y_col else 0,
                "z": to_int(r.get(grid_z_col), 0) if grid_z_col else 0,
            }
            node["block"] = {
                "x": to_int(r.get(block_x_col), 0) if block_x_col else 0,
                "y": to_int(r.get(block_y_col), 0) if block_y_col else 0,
                "z": to_int(r.get(block_z_col), 0) if block_z_col else 0,
            }

        nodes.append(node)

        if len(nodes) > 1:
            edges.append({"source": len(nodes) - 2, "target": len(nodes) - 1})

    if args.aggregation == "semantic":
        nodes, edges = aggregate_semantic(nodes)
    elif args.aggregation == "stream_dag":
        edges = build_stream_dag_edges(nodes)

    transfer_totals = compute_transfer_totals(nodes)

    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "aggregation": args.aggregation,
            "min_transfer_bytes": args.min_transfer_bytes,
            "transfer_totals": {
                **transfer_totals,
                "total_transfer_bytes": transfer_totals["total_transfer_bytes"],
            }
        },
        "nodes": nodes,
        "edges": edges
    }

    with out_json.open("w") as f:
        json.dump(graph, f, indent=4)

    print(f"Wrote {out_json} with {len(nodes)} nodes, {len(edges)} edges")
    if fallback_name_mapping_count:
        print(
            f"Note: {fallback_name_mapping_count} rows used name-based src/dst fallback "
            "(no explicit SrcMemKd/DstMemKd columns detected)."
        )

if __name__ == "__main__":
    main()
