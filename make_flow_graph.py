# save as make_flow_graph.py
import pandas as pd
from graphviz import Digraph
import sys

prefix = sys.argv[1]  # e.g., nsys_dummy_per_iter
gpu = pd.read_csv(f"{prefix}_trace_cuda_gpu_trace.csv")

# Keep only relevant ops
name_col = [c for c in gpu.columns if "Name" in c or "name" in c][0]
start_col = [c for c in gpu.columns if "Start" in c or "start" in c][0]
dur_col = [c for c in gpu.columns if "Duration" in c or "duration" in c][0]
stream_col = [c for c in gpu.columns if "Stream" in c or "stream" in c][0]

gpu = gpu[[name_col, start_col, dur_col, stream_col]].copy()
gpu = gpu.sort_values(start_col).head(400)  # keep graph readable

g = Digraph("flow", format="png")
for i, r in gpu.iterrows():
    nid = f"n{i}"
    label = f"{r[name_col]}\\nstream={r[stream_col]}\\ndur={r[dur_col]}"
    g.node(nid, label)

# Connect temporal neighbors within each stream
for s, grp in gpu.groupby(stream_col):
    ids = [f"n{i}" for i in grp.index]
    for a, b in zip(ids, ids[1:]):
        g.edge(a, b, label=f"s{s}")

g.render(f"{prefix}_flow", cleanup=True)
print(f"Wrote {prefix}_flow.png")
