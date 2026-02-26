# save as nsys_trace_to_flow_json.py
import csv
import json
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

def map_src_dst(op_name: str):
    s = (op_name or "").lower()
    # memcpy direction patterns across Nsight versions
    if "htod" in s or "h2d" in s or "host to device" in s:
        return "cpu", "acc"
    if "dtoh" in s or "d2h" in s or "device to host" in s:
        return "acc", "cpu"
    if "dtod" in s or "d2d" in s or "device to device" in s or "p2p" in s:
        return "acc", "acc"
    if "memset" in s:
        return "acc", "acc"
    # kernels/default
    return "acc", "acc"

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
        bytes_col = pick_col(cols, ["bytes", "size"], required=False)

        rows = list(reader)

    # Sort by start time if available
    if start_col:
        rows.sort(key=lambda r: to_int(r.get(start_col), 0))

    nodes = []
    edges = []

    for i, r in enumerate(rows):
        op = r.get(name_col, "") if name_col else ""
        src, dst = map_src_dst(op)
        vol = to_int(r.get(bytes_col), 0) if bytes_col else 0

        nodes.append({
            "id": i,
            "vol": vol,
            "source_id": src,
            "dest_id": dst
        })

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

if __name__ == "__main__":
    main()
