#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


def print_to_terminal(g: nx.DiGraph) -> None:
    print("Nodes:", list(g.nodes(data=True)))
    print("Edges:", list(g.edges()))


TRANSFER_COLORS = {
    "write (acc -> mem)": "#df6940",
    "read (mem -> acc)": "#8b80df",
    "upload (cpu -> mem)": "#52788b",
    "download (mem -> cpu)": "#607466",
    "other transfer / compute": "#777777",
}


def endpoints(data):
    return (str(data.get("src_comp", data.get("source_id", "?"))),
            str(data.get("dest_comp", data.get("dest_id", "?"))))


def component_kind(name):
    return name.split("_", 1)[0]


def transfer_category(data):
    src, dst = map(component_kind, endpoints(data))
    return {("acc", "mem"): "write (acc -> mem)",
            ("mem", "acc"): "read (mem -> acc)",
            ("cpu", "mem"): "upload (cpu -> mem)",
            ("mem", "cpu"): "download (mem -> cpu)"}.get(
                (src, dst), "other transfer / compute")


def volume_label(value):
    if value is None:
        return "size unknown"
    size = int(value)
    # Exact binary units; do not label MiB as MB or round away small transfers.
    for unit, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= factor and size % factor == 0:
            return f"{size // factor:,} {unit}"
    return f"{size:,} B"


def transfer_layout(g):
    if not g or not nx.is_directed_acyclic_graph(g):
        raise ValueError("Transfer rendering requires a nonempty directed acyclic graph")
    layers = list(nx.topological_generations(g))
    lane_for = {}
    for node, data in g.nodes(data=True):
        memories = sorted({e for e in endpoints(data) if component_kind(e) == "mem"})
        lane_for[node] = " / ".join(memories) if memories else "other"
    lanes = sorted(set(lane_for.values()))
    pos, lane_centers = {}, {}
    cursor = 0.0
    for lane in lanes:
        groups = [[n for n in layer if lane_for[n] == lane] for layer in layers]
        capacity = max(map(len, groups))
        center = cursor - (capacity - 1) * 0.78
        lane_centers[lane] = center
        for level, group in enumerate(groups):
            def order(node):
                ys = [pos[p][1] for p in g.predecessors(node) if p in pos]
                return (-(sum(ys) / len(ys)) if ys else 0, str(node))
            group.sort(key=order)
            for row, node in enumerate(group):
                pos[node] = (level * 3.0, center + ((len(group) - 1) / 2 - row) * 1.56)
        cursor -= capacity * 1.56 + 1.2
    return layers, pos, lane_centers


def generate_figure(g: nx.DiGraph, filename: str, out_dir: str = "graph_img") -> str:
    os.makedirs(out_dir, exist_ok=True)
    layers, pos, lanes = transfer_layout(g)
    ymin = min(y for x, y in pos.values())
    ymax = max(y for x, y in pos.values())
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 2.1 + 1.8),
                                    max(4.5, (ymax - ymin) * 0.9 + 3)))
    ax.set_xlim(-3.3, (len(layers) - 1) * 3 + 1.4)
    ax.set_ylim(ymin - 1, ymax + 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    boxes = {}
    for node, (x, y) in pos.items():
        color = TRANSFER_COLORS[transfer_category(g.nodes[node])]
        box = FancyBboxPatch((x - 1.12, y - 0.51), 2.24, 1.02,
                             boxstyle="round,pad=0.035,rounding_size=0.09",
                             facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(box)
        boxes[node] = box
    for src, dst in g.edges():
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        # Route edges that skip generations around intervening boxes.
        curve = 0.16 if x2 - x1 > 3.1 else 0
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), patchA=boxes[src],
                                    patchB=boxes[dst], arrowstyle="-|>",
                                    mutation_scale=10, linewidth=0.9,
                                    color="#b7b7b7", shrinkA=3, shrinkB=3,
                                    connectionstyle=f"arc3,rad={curve}", zorder=1))
    for node, (x, y) in pos.items():
        data = g.nodes[node]
        src, dst = endpoints(data)
        ax.text(x, y + 0.25, f"t_{node}", color="white", weight="bold",
                fontsize=10, ha="center", va="center", zorder=4)
        ax.text(x, y - 0.02, f"{src} → {dst}", color="white",
                fontsize=9, ha="center", va="center", zorder=4)
        ax.text(x, y - 0.28, volume_label(data.get("vol", data.get("vol_bytes"))),
                color="white", fontsize=9, ha="center", va="center", zorder=4)
    for level in range(len(layers)):
        ax.text(level * 3, ymax + 0.87, f"L{level}", ha="center", va="center",
                fontsize=9, color="#888888")
    for lane, y in lanes.items():
        ax.text(-1.48, y, f"{lane}\ntransfers", ha="right", va="center",
                weight="bold", fontsize=11, color="#404040")
    categories = {transfer_category(data) for _, data in g.nodes(data=True)}
    handles = [Patch(color=color, label=label.replace("->", "→"))
               for label, color in TRANSFER_COLORS.items() if label in categories]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.02, "L = dependency level (not elapsed time) · Transfer sizes use bytes / binary units",
             ha="center", fontsize=8, color="#777777")
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.19, top=0.98)

    out_path = os.path.join(out_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def graph_1() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(1, source_id="cpu", dest_id="mem", vol=6400)
    g.add_node(2, source_id="mem", dest_id="acc_mac_1", vol=6400)
    g.add_node(3, source_id="acc_mac_1", dest_id="mem", vol=100)
    g.add_node(4, source_id="mem", dest_id="cpu", vol=100)
    g.add_node(5, source_id="mem", dest_id="cpu", vol=100)
    g.add_node(6, source_id="cpu", dest_id="mem", vol=100)
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (6, 5), (6, 2)])
    return g


def read_graph(path: str) -> nx.DiGraph:
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    edges_keyword = "links" if "links" in data else "edges"
    return nx.node_link_graph(data, edges=edges_keyword)


def save_graph(g: nx.DiGraph, graph_str: str, dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    filename = f"{graph_str}.json"
    path = os.path.join(dirpath, filename)
    data = nx.node_link_data(g, edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render flow graphs with NetworkX.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Existing node-link JSON to read (e.g., nsys_dummy_per_iter_flow.json).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate built-in demo graph_1 instead of reading --input.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="flow_graph",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="graph_img",
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--save-json-dir",
        type=str,
        default="",
        help="Optional directory to save graph back as node-link JSON.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print nodes/edges in terminal.",
    )
    args = parser.parse_args()

    if args.demo:
        g = graph_1()
    elif args.input:
        g = read_graph(args.input)
    else:
        raise SystemExit("Pass either --demo or --input <flow.json>.")

    if args.print:
        print_to_terminal(g)

    out_png = generate_figure(g, args.name, out_dir=args.out_dir)
    print(f"Saved figure: {out_png}")

    if args.save_json_dir:
        out_json = save_graph(g, args.name, args.save_json_dir)
        print(f"Saved graph JSON: {out_json}")


if __name__ == "__main__":
    main()
