#!/usr/bin/env python3
"""
Simple simulator: walk a node-link flow JSON, add an interrupt after each
node and record how many SM clock cycles have been used.

Behavior / assumptions:
- If the graph is a DAG we use topological sort to decide order; otherwise
  we fall back to the order of nodes in the JSON.
- Per-node cycles are taken from node['sm_metrics']['cycles_elapsed'] when
  present, else from node['sm_metrics']['cycles_active'], else 0.
- The script appends an `interrupt_events` list to the output JSON and also
  annotates each node with fields summarising the interrupt (boolean,
  per-node cycles, cumulative cycles at that point, and an event index).

Usage:
  python3 simulate_with_interrupts.py --flow some_flow.json --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx


def read_flow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def detect_edges_keyword(data: Dict[str, Any]) -> str:
    return "links" if "links" in data else "edges"


def build_graph_from_flow(data: Dict[str, Any]) -> nx.DiGraph:
    edges_keyword = detect_edges_keyword(data)
    return nx.node_link_graph(data, edges=edges_keyword)


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return 0


def cycles_from_node(node: Dict[str, Any]) -> int:
    """Per-node cycle cost: duration_cycles (wall-clock) or ncu sm_cycles_elapsed."""
    # duration_cycles covers both kernels and transfers (from nsys timestamps × clock).
    if "duration_cycles" in node:
        return _to_int(node["duration_cycles"])
    sm = node.get("sm_metrics", {}) or {}
    for key in ("sm_cycles_elapsed", "cycles_elapsed", "cycles_active"):
        if key in sm:
            return _to_int(sm[key])
    return 0


def cycles_at_end_from_node(node: Dict[str, Any]) -> int:
    """
    Absolute cycles elapsed from trace start to the END of this node.
    When present this is more accurate than summing per-node costs because
    it accounts for scheduling gaps between operations.
    """
    if "cycles_at_end" in node:
        return _to_int(node["cycles_at_end"])
    return -1  # sentinel: not available


def simulate_interrupts(flow: Dict[str, Any]) -> Dict[str, Any]:
    # Build graph to determine ordering when possible
    g = build_graph_from_flow(flow)

    # Map node id -> node dict from original flow
    nodes_list: List[Dict[str, Any]] = flow.get("nodes", [])
    id_key = "id"
    id_to_node = {n.get(id_key): n for n in nodes_list}

    # Decide processing order
    try:
        if nx.is_directed_acyclic_graph(g):
            order = list(nx.topological_sort(g))
        else:
            # fall back to nodes as listed in JSON
            order = [n.get(id_key) for n in nodes_list]
    except Exception:
        order = [n.get(id_key) for n in nodes_list]

    interrupt_events: List[Dict[str, Any]] = []
    cumulative = 0
    has_absolute_timestamps = any("cycles_at_end" in n for n in nodes_list)
    event_idx = 0

    for nid in order:
        node = id_to_node.get(nid)
        if node is None:
            continue

        per_node_cycles = cycles_from_node(node)
        absolute_cycles = cycles_at_end_from_node(node)

        if absolute_cycles >= 0:
            # Use the timestamp-derived absolute value: includes scheduling gaps,
            # more accurate than summing per-node durations.
            cumulative = absolute_cycles
        else:
            cumulative += per_node_cycles

        ev = {
            "event_index": event_idx,
            "type": "interrupt",
            "node_id": nid,
            "op_name": node.get("op_name") or node.get("name") or None,
            "cycles_this_node": per_node_cycles,
            "cumulative_cycles": cumulative,
        }
        interrupt_events.append(ev)

        node["interrupt_after"] = True
        node["interrupt_event_index"] = event_idx
        node["cycles_this_node"] = per_node_cycles
        node["cumulative_cycles_at_interrupt"] = cumulative

        event_idx += 1

    flow.setdefault("events", {})
    flow["events"]["interrupt_events"] = interrupt_events
    flow["events"]["total_interrupts"] = len(interrupt_events)
    flow["events"]["total_cycles"] = cumulative
    flow["events"]["cycle_source"] = (
        "absolute_timestamps" if has_absolute_timestamps else "sum_of_durations"
    )

    return flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert interrupts after each node and tally cycles.")
    parser.add_argument("--flow", required=True, help="Input flow JSON path")
    parser.add_argument("--out", default="", help="Output JSON path (default: <flow>_interrupts.json)")
    args = parser.parse_args()

    flow_path = Path(args.flow)
    if not flow_path.exists():
        sys.exit(f"Flow file not found: {flow_path}")

    flow = read_flow(flow_path)
    out_flow = simulate_interrupts(flow)

    out_path = Path(args.out) if args.out else flow_path.with_name(flow_path.stem + "_interrupts.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_flow, f, indent=4)

    events_meta = out_flow.get("events", {})
    interrupts = events_meta.get("interrupt_events", [])
    total_cycles = events_meta.get("total_cycles", 0)
    cycle_source = events_meta.get("cycle_source", "unknown")

    print(f"Wrote {out_path}  ({len(interrupts)} interrupts, source={cycle_source})")
    print()
    print(f"{'#':>4}  {'Operation':<52}  {'cycles_this_node':>18}  {'cumulative_cycles':>18}")
    print("-" * 96)
    for ev in interrupts:
        name = (ev.get("op_name") or "?")[:52]
        print(
            f"{ev['event_index']:>4}  {name:<52}  "
            f"{ev['cycles_this_node']:>18,}  "
            f"{ev['cumulative_cycles']:>18,}"
        )
    print("-" * 96)
    print(f"{'TOTAL':>58}  {'':>18}  {total_cycles:>18,}")


if __name__ == "__main__":
    main()
