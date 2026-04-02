#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx


def print_to_terminal(g: nx.DiGraph) -> None:
    print("Nodes:", list(g.nodes(data=True)))
    print("Edges:", list(g.edges()))


def generate_figure(g: nx.DiGraph, filename: str, out_dir: str = "graph_img") -> str:
    os.makedirs(out_dir, exist_ok=True)

    # For DAGs, use topological generations (clean layered flow layout).
    if nx.is_directed_acyclic_graph(g):
        for i, layer in enumerate(nx.topological_generations(g)):
            for node in layer:
                g.nodes[node]["subset"] = i
        pos = nx.multipartite_layout(g, subset_key="subset")
    else:
        # Fallback for cyclic graphs.
        pos = nx.spring_layout(g, seed=42)

    plt.figure(figsize=(14, 8))
    nx.draw(g, pos, with_labels=True, node_size=900, font_size=8, arrows=True)

    out_path = os.path.join(out_dir, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
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
