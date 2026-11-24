#!/usr/bin/env python3
"""Collect per-kernel DRAM read/write counters and compare with logical accesses."""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

from export_workload_transfers import WORKLOADS, build_model


METRICS = {"dram__bytes_read.sum": "dram_read_bytes",
           "dram__bytes_write.sum": "dram_write_bytes"}


def parse_csv(text):
    """Accept NCU long metric rows and raw wide rows with a units row.

    Require base bytes; unavailable counters and duplicate measurements fail.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.strip() and next(csv.reader([line]))[0].lstrip("\ufeff") == "ID"), None)
    if start is None:
        raise ValueError("No NCU CSV header; inspect profiling logs")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:]).lstrip("\ufeff")))
    long_format = "Metric Name" in reader.fieldnames
    if not long_format and not set(METRICS) <= set(reader.fieldnames):
        raise ValueError("CSV is missing requested DRAM metrics")
    records, units = {}, {}

    def byte_value(value, unit):
        if unit.strip().lower() not in {"byte", "bytes"}:
            raise ValueError(f"Expected base byte units, got {unit!r}")
        try:
            number = Decimal(value.replace(",", "").strip())
        except InvalidOperation as exc:
            raise ValueError(f"Unavailable or invalid counter: {value!r}") from exc
        if not number.is_finite() or number < 0 or number != number.to_integral_value():
            raise ValueError(f"Invalid byte count: {value!r}")
        return int(number)

    for row in reader:
        if not (row.get("ID") or "").isdigit():
            if not long_format and all(row.get(m) for m in METRICS):
                units = {m: row[m] for m in METRICS}
            continue
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed NCU CSV row")
        if long_format and row["Metric Name"] not in METRICS:
            continue
        name = row.get("Kernel Name", "")
        if not name:
            raise ValueError("Missing kernel name")
        key = (row.get("Process ID", ""), row.get("Context", ""), row["ID"], name)
        record = records.setdefault(key, dict(kernel_name=name, kernel_id=row["ID"],
                                             process_id=key[0], context=key[1], metrics={}))
        selected = [row["Metric Name"]] if long_format else METRICS
        for metric in selected:
            friendly = METRICS[metric]
            if friendly in record["metrics"]:
                raise ValueError(f"Duplicate metric for launch {key}: {metric}")
            record["metrics"][friendly] = byte_value(
                row["Metric Value"] if long_format else row[metric],
                row["Metric Unit"] if long_format else units.get(metric, ""))
    if not records or any(set(r["metrics"]) != set(METRICS.values()) for r in records.values()):
        raise ValueError("No complete per-launch DRAM measurements")
    return list(records.values())


def compare(workload, records):
    graph, labels = build_model(workload)
    stages = {}
    for label in labels:
        if ":" not in label["stage"]:
            continue
        stage, access = label["stage"].split(":")
        item = stages.setdefault(stage, dict(stage=stage, logical_read_bytes=0,
                                            logical_write_bytes=0, transfer_node_ids=[]))
        item[f"logical_{access}_bytes"] += graph["nodes"][label["id"]]["vol"]
        item["transfer_node_ids"].append(label["id"])
    seen = set()
    for record in records:
        matches = [s for s in stages if re.search(r"\b" + re.escape(s) + r"\s*\(", record["kernel_name"])
                   or record["kernel_name"] == s]
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError(f"Unexpected or repeated kernel: {record['kernel_name']}")
        seen.add(matches[0])
        stages[matches[0]]["measurement"] = record
    if seen != set(stages):
        raise ValueError(f"Missing kernel measurements: {set(stages) - seen}")
    return list(stages.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all",) + WORKLOADS, default="all")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument("--replay-mode", choices=("kernel", "application"), default="kernel")
    parser.add_argument("--csv", type=Path, help="Convert an existing CSV; requires a single workload")
    args = parser.parse_args()
    if args.csv and args.workload == "all":
        parser.error("--csv requires --workload")
    root = Path(__file__).resolve().parent
    out = (args.out_dir or root / "results" / datetime.now(timezone.utc).strftime("ncu_%Y%m%dT%H%M%S%fZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="started", commands=[], cache_control=args.cache_control,
                    replay_mode=args.replay_mode, clock_control="none", metrics=list(METRICS),
                    measurement_scope="per-kernel device DRAM counters; excludes host copy accounting",
                    warnings=["Profiler replay and cache policy affect measurements.",
                              "Counters cannot attribute bytes to individual buffers.",
                              "Logical simulator volumes are not replaced by these counters."],
                    sources={})

    def run(command, filename):
        metadata["commands"].append(command)
        with (out / filename).open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command failed; inspect {out / filename}")
        return (out / filename).read_text()

    try:
        names = WORKLOADS if args.workload == "all" else (args.workload,)
        if args.csv:
            metadata.update(cache_control="unknown (imported CSV)", replay_mode="unknown (imported CSV)",
                            clock_control="unknown (imported CSV)")
        else:
            for command, filename in [(["ncu", "--version"], "ncu_version.txt"),
                                      (["nvcc", "--version"], "cuda_version.txt"),
                                      (["nvidia-smi", "-q"], "gpu.txt"),
                                      (["git", "rev-parse", "HEAD"], "commit.txt"),
                                      (["git", "status", "--short"], "git_status.txt")]:
                run(command, filename)
        for name in names:
            source = root / "workloads" / (name + ".cu")
            metadata["sources"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
            csv_path = out / (name + "_ncu.csv")
            if args.csv:
                csv_path.write_bytes(args.csv.read_bytes())
            else:
                binary = out / name
                run(["nvcc", "-O3", "-std=c++17", "-lineinfo", str(source), "-o", str(binary)], name + "_build.log")
                print(f"Profiling {name}; logs: {out}", flush=True)
                run(["ncu", "--metrics", ",".join(METRICS), "--cache-control", args.cache_control,
                     "--replay-mode", args.replay_mode, "--clock-control", "none",
                     "--export", str(out / name), str(binary)], name + "_profile.log")
                run(["ncu", "--import", str(out / (name + ".ncu-rep")), "--csv", "--page", "raw",
                     "--print-units", "base", "--print-metric-name", "name"], csv_path.name)
            result = dict(workload=name, kernels=compare(name, parse_csv(csv_path.read_text())))
            (out / (name + "_hardware.json")).write_text(json.dumps(result, indent=2) + "\n")
            print(f"Wrote {name}_hardware.json", flush=True)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
