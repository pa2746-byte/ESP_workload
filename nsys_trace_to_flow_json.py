# save as nsys_trace_to_flow_json.py
import csv
import json
import re
import sys
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python nsys_trace_to_flow_json.py <prefix>")
        sys.exit(1)

    prefix = sys.argv[1]
    trace_csv = Path(f"{prefix}_trace_cuda_gpu_trace.csv")
    out_json = Path(f"{prefix}_flow.json")

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

    for i, r in enumerate(rows):
        op = r.get(name_col, "") if name_col else ""
        src_mem = r.get(src_mem_kind_col, "") if src_mem_kind_col else ""
        dst_mem = r.get(dst_mem_kind_col, "") if dst_mem_kind_col else ""
        src, dst = map_src_dst(op, src_mem, dst_mem)
        if not src_mem_kind_col and not dst_mem_kind_col:
            fallback_name_mapping_count += 1
        vol = parse_volume(r, bytes_col) if bytes_col else 0
        op_type, subtype = classify_op(op, src, dst)
        duration_ns = to_int(r.get(duration_col), 0) if duration_col else 0
        stream_id = to_int(r.get(stream_col), -1) if stream_col else -1

        node = {
            "id": i,
            "vol": vol,
            "source_id": src,
            "dest_id": dst,
            "op_type": op_type,
            "subtype": subtype,
            "op_name": op,
            "duration_ns": duration_ns,
            "stream_id": stream_id,
        }

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

        if i > 0:
            edges.append({"source": i - 1, "target": i})

    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {},
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
