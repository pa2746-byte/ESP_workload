#!/usr/bin/env python3
"""Analyze CUDA sources and export logical buffer transfer graphs.

These are source-derived logical models, not trace inference or measured traffic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from cuda_source_model import analyze, UnsupportedSource
import subprocess
import sys


WORKLOADS = ("vector_add", "pipeline", "fork_join", "reduction")


def build_model(name, cpu="cpu", mem="mem", acc="acc"):
    return analyze(Path(__file__).parent / "workloads" / (name + ".cu"), cpu, mem, acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Analyze a CUDA source file instead of bundled workloads")
    parser.add_argument("--clang", help="Clang C++ executable (default: CLANGXX or clang++)")
    parser.add_argument("--analysis-config", type=Path,
                        help="Explicit host-input contracts and device parameters for bounded AST analysis (requires --source)")
    parser.add_argument("--parameter", action="append", default=[], metavar="NAME=INTEGER",
                        help="Override a named parameter in --analysis-config")
    parser.add_argument("--workload", choices=("all",) + WORKLOADS + ("fork_join_streams",), default="all",
                        help="all selects the original four examples; select fork_join_streams explicitly")
    parser.add_argument("--out-dir", type=Path, default=Path("results/logical_transfers"))
    parser.add_argument("--cpu", default="cpu")
    parser.add_argument("--mem", default="mem")
    parser.add_argument("--acc", default="acc")
    parser.add_argument("--render", action="store_true",
                        help="Also render PNGs using matplotlib and NetworkX")
    args = parser.parse_args()
    configuration = None
    if args.analysis_config:
        if not args.source: parser.error("--analysis-config requires --source")
        try:
            configuration = json.loads(args.analysis_config.read_text())
            for item in args.parameter:
                key, value = item.split("=", 1)
                if key not in configuration.get("parameters", {}):
                    parser.error(f"Unknown analysis parameter: {key}")
                configuration["parameters"][key] = int(value)
        except (OSError, ValueError) as exc:
            parser.error(f"Invalid analysis configuration: {exc}")
    elif args.parameter:
        parser.error("--parameter requires --analysis-config")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = [args.source] if args.source else [
        Path(__file__).resolve().parent / "workloads" / (name + ".cu")
        for name in (WORKLOADS if args.workload == "all" else (args.workload,))]
    # Analyze all inputs before writing outputs so unsupported sources don't leave a partial batch.
    models = []
    for source in sources:
        try:
            details = None
            if configuration is not None:
                from cuda_configured_model import analyze_configured
                graph, descriptions, details = analyze_configured(
                    source, configuration, args.cpu, args.mem, args.acc, args.clang)
            else:
                graph, descriptions = analyze(source, args.cpu, args.mem, args.acc, args.clang)
        except UnsupportedSource as exc:
            parser.error(f"{source}: {exc}")
        models.append((source, graph, descriptions, details))
    for source, graph, descriptions, details in models:
        name = source.stem
        metadata = dict(
            workload=name,
            model="Clang AST-derived logical global-buffer footprints",
            volume_unit="bytes",
            source=str(source.resolve()),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            nodes=descriptions,
            assumptions=[
                "Statically analyzed supported CUDA subset; see workloads/TRANSFER_MODELS.md.",
                "Unique contiguous buffer footprints per kernel; repeated element accesses are not instruction traffic.",
                "Edges express whole-buffer readiness, not observed default-stream order.",
                "Stream/event ordering is checked for conflicting buffer accesses; metadata records operation ordering, not timing.",
                "No compute time, cache effects, physical DRAM measurements, or internal shared-memory transfers.",
                "Re-run after source changes; unsupported constructs are errors, not guessed dependencies.",
            ],
        )
        if details is not None:
            metadata["configured_analysis"] = details
            metadata["assumptions"].extend([
                "Host input parsing/loading and CPU result selection use explicit, source-hash-checked summaries; they are not executed.",
                "Device properties are configured assumptions, not measurements; bounded kernel address evaluation requires no GPU.",
                "Device values remain unknown; data-dependent control/indexing and non-contiguous footprints are rejected.",
            ])
        path = args.out_dir / (name + "_simulator.json")
        path.write_text(json.dumps(graph, indent=2) + "\n")
        (args.out_dir / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {path}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        if args.render:
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("flow_graph_networkx.py")),
                 "--input", str(path), "--name", name + "_simulator",
                 "--out-dir", str(args.out_dir / "graphs")],
                env={**os.environ, "MPLBACKEND": "Agg"}, check=True,
            )


if __name__ == "__main__":
    main()
