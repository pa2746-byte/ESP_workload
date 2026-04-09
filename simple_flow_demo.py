#!/usr/bin/env python3
"""
Minimal demo: build a flow JSON with vol_bytes, direction, and transfer_totals.
Uses the same helpers as generate_synthetic_flow.py — no GPU or Nsight required.

Run:
  python3 simple_flow_demo.py
  python3 simple_flow_demo.py --n 4096 --iters 3 --mode once
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on PYTHONPATH
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generate_synthetic_flow import build_dummy_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a small synthetic flow JSON with data values.")
    parser.add_argument("--n", type=int, default=1024, help="Vector length (float32 elements).")
    parser.add_argument("--iters", type=int, default=2, help="Iterations.")
    parser.add_argument(
        "--mode",
        choices=["once", "per_iter"],
        default="per_iter",
        help="Transfer mode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="demo_simple_flow.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    graph = build_dummy_graph(args.n, args.iters, args.mode, calib=None)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(graph, indent=4), encoding="utf-8")

    totals = graph["graph"].get("transfer_totals", {})
    print(f"Wrote {out_path.resolve()}")
    print(f"  nodes={len(graph['nodes'])}, edges={len(graph['edges'])}")
    print(f"  transfer_totals: {totals}")
    print("  First node sample:", graph["nodes"][0] if graph["nodes"] else "(empty)")


if __name__ == "__main__":
    main()
