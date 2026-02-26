# make_flow_graph.py
import pandas as pd
from graphviz import Digraph
import sys

prefix = sys.argv[1]  # e.g., nsys_dummy_per_iter
gpu = pd.read_csv(f"{prefix}_trace_cuda_gpu_trace.csv")

def pick_col(cands, required=True):
    cols = list(gpu.columns)
    low = {c.lower(): c for c in cols}
    for cand in cands:
        for c in cols:
            if cand in c.lower():
                return c
    if required:
        raise RuntimeError(f"Could not find any of {cands}. Available columns: {cols}")
    return None

name_col = pick_col(["name", "demangled", "operation"])
start_col = pick_col(["start", "timestamp"])
dur_col = pick_col(["duration", "time"])
stream_col = pick_col(["stream", "streamid", "stream id"], required=False)

# If stream column is missing, put all events on one lane
if stream_col is None:
    gpu["__stream_fallback__"] = 0
    stream_col = "__stream_fallback__"

gpu = gpu[[name_col, start_col, dur_col, stream_col]].copy()
gpu = gpu.sort_values(start_col).head(500)  # keep graph readable

g = Digraph("flow", format="png")
for i, r in gpu.iterrows():
    nid = f"n{i}"
    label = f"{r[name_col]}\\nstream={r[stream_col]}\\ndur={r[dur_col]}"
    g.node(nid, label)

for s, grp in gpu.groupby(stream_col, dropna=False):
    ids = [f"n{i}" for i in grp.index]
    for a, b in zip(ids, ids[1:]):
        g.edge(a, b, label=f"s{s}")

g.render(f"{prefix}_flow", cleanup=True)
print(f"Wrote {prefix}_flow.png")
