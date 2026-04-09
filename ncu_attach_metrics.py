#!/usr/bin/env python3
"""
Parse an `ncu --csv` output and attach per-kernel L2/DRAM/L1 metrics
to matching kernel nodes in an existing flow JSON.

Usage:
  ncu --metrics l1tex__t_bytes,lts__t_bytes,dram__bytes \
      --csv ./dummy_pipeline 16777216 10 1 > ncu_dummy.csv

  python3 ncu_attach_metrics.py \
      --ncu   ncu_dummy.csv \
      --flow  nsys_dummy_per_iter_flow.json \
      --out   nsys_dummy_per_iter_ncu_flow.json
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Metrics we care about, in ncu's exact column name form.
# ncu --csv emits one row per (kernel, metric); we pivot these into a dict.
METRICS_OF_INTEREST = {
    "l1tex__t_bytes":                   "l1_bytes",
    "lts__t_bytes":                     "l2_bytes",
    "dram__bytes":                      "dram_bytes",
    "sm__shared_memory_load_throughput":"sm_shared_load_throughput_gbs",
    # Derived / alternative names nsight may emit depending on arch/version
    "l2_global_load_bytes":             "l2_bytes",
    "dram_read_bytes":                  "dram_bytes",
}

BANDWIDTH_METRICS = {
    "lts__t_sectors_srcunit_l1tex_op_read.sum":  "l2_read_sectors",
    "lts__t_sectors_srcunit_l1tex_op_write.sum": "l2_write_sectors",
}


def _strip(s: str) -> str:
    return s.strip().strip('"')


def _kernel_basename(name: str) -> str:
    """Return a short version of the kernel name for fuzzy matching."""
    # Drop template args and parameters: keep only the bare function name.
    s = re.sub(r"<[^>]*>", "", name)
    s = re.sub(r"\([^)]*\)", "", s)
    return s.strip().split("::")[-1].lower()


def parse_ncu_csv(path: Path) -> List[Dict]:
    """
    Parse ncu --csv output.

    ncu emits a long-format CSV:
      "ID","Process ID","Process Name","Host Name","Kernel Name",
      "Kernel Time","Context","Stream","Metric Name","Metric Unit","Metric Value"

    Returns a list of dicts, one per kernel invocation, with all metrics
    pivoted into a flat dict under 'metrics'.
    """
    kernels: Dict[tuple, Dict] = {}  # (id, kernel_name) -> aggregated record

    with path.open(newline="", encoding="utf-8-sig") as f:
        # ncu sometimes emits comment lines starting with "==" — skip them.
        lines = [l for l in f if not l.startswith("==")]

    if not lines:
        print(f"  [warn] {path} is empty — ncu likely failed. Skipping.")
        return []

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        print(f"  [warn] {path} has no header row — ncu likely failed. Skipping.")
        return []

    cols = [_strip(c) for c in reader.fieldnames]

    # Build a normalised-column reader
    def col(row: dict, *candidates: str) -> Optional[str]:
        for cand in candidates:
            for k, v in row.items():
                if _strip(k).lower() == cand.lower():
                    return _strip(v)
        return None

    for raw_row in reader:
        row = {_strip(k): _strip(v) for k, v in raw_row.items()}

        kid     = row.get("ID", "0")
        kname   = row.get("Kernel Name", row.get("Name", "unknown"))
        mname   = row.get("Metric Name", "")
        mval    = row.get("Metric Value", "")
        munit   = row.get("Metric Unit", "")
        stream  = row.get("Stream", "-1")
        ktime   = row.get("Kernel Time", "")

        key = (kid, kname)
        if key not in kernels:
            kernels[key] = {
                "kernel_id":   kid,
                "kernel_name": kname,
                "stream":      stream,
                "kernel_time": ktime,
                "metrics":     {},
            }

        # Normalise metric value (may have commas or spaces)
        try:
            val = float(mval.replace(",", "").replace(" ", ""))
        except ValueError:
            val = 0.0

        # Map to our friendly name if we recognise it
        friendly = METRICS_OF_INTEREST.get(mname) or BANDWIDTH_METRICS.get(mname)
        if friendly:
            # Accumulate in case ncu emits per-pass duplicates
            kernels[key]["metrics"][friendly] = (
                kernels[key]["metrics"].get(friendly, 0.0) + val
            )
        # Always store raw metric too
        kernels[key]["metrics"][mname] = (
            kernels[key]["metrics"].get(mname, 0.0) + val
        )

    return list(kernels.values())


def _match_kernel(ncu_name: str, flow_name: str) -> bool:
    """True if the ncu kernel name and the flow op_name refer to the same kernel."""
    n = ncu_name.lower()
    f = flow_name.lower()

    # Exact match
    if n == f or f in n or n in f:
        return True

    # Basename match (strip templates/args)
    nb = _kernel_basename(ncu_name)
    fb = _kernel_basename(flow_name)
    return nb and fb and (nb == fb or fb in nb or nb in fb)


def attach_metrics(flow: Dict, ncu_records: List[Dict]) -> Dict:
    """
    For each kernel node in the flow JSON, find all matching ncu records
    and attach their summed metrics.
    """
    # Group ncu records by kernel name for fast lookup
    by_name: Dict[str, List[Dict]] = defaultdict(list)
    for rec in ncu_records:
        by_name[rec["kernel_name"]].append(rec)

    matched_total = 0
    unmatched_nodes = []

    for node in flow.get("nodes", []):
        if node.get("op_type") != "kernel":
            continue

        op_name = node.get("op_name", "")

        # Find all ncu records that match this kernel name
        matched: List[Dict] = []
        for ncu_name, recs in by_name.items():
            if _match_kernel(ncu_name, op_name):
                matched.extend(recs)

        if not matched:
            unmatched_nodes.append(op_name)
            continue

        matched_total += 1

        # Sum metrics across all matched invocations
        combined: Dict[str, float] = defaultdict(float)
        for rec in matched:
            for k, v in rec["metrics"].items():
                combined[k] += v

        # Attach the friendly metrics we care about
        sm_metrics: Dict = {}
        for friendly in ["l1_bytes", "l2_bytes", "dram_bytes",
                         "l2_read_sectors", "l2_write_sectors",
                         "sm_shared_load_throughput_gbs"]:
            if friendly in combined:
                sm_metrics[friendly] = int(combined[friendly]) if "bytes" in friendly or "sectors" in friendly else combined[friendly]

        # Also attach L2 and DRAM in MB for readability
        if "l2_bytes" in sm_metrics:
            sm_metrics["l2_mb"] = round(sm_metrics["l2_bytes"] / (1024 * 1024), 3)
        if "dram_bytes" in sm_metrics:
            sm_metrics["dram_mb"] = round(sm_metrics["dram_bytes"] / (1024 * 1024), 3)
        if "l1_bytes" in sm_metrics:
            sm_metrics["l1_mb"] = round(sm_metrics["l1_bytes"] / (1024 * 1024), 3)

        node["sm_metrics"] = sm_metrics
        node["ncu_invocation_count"] = len(matched)

    return flow, matched_total, unmatched_nodes


def summarize_sm_metrics(nodes: List[Dict]) -> Dict:
    totals: Dict[str, float] = defaultdict(float)
    for node in nodes:
        for k, v in node.get("sm_metrics", {}).items():
            if isinstance(v, (int, float)):
                totals[k] += v
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach ncu SM/L2/DRAM metrics to kernel nodes in a flow JSON."
    )
    parser.add_argument("--ncu",  required=True, help="ncu --csv output file")
    parser.add_argument("--flow", required=True, help="Input flow JSON")
    parser.add_argument("--out",  default="",    help="Output JSON (default: <flow>_ncu.json)")
    args = parser.parse_args()

    ncu_path  = Path(args.ncu)
    flow_path = Path(args.flow)

    if not ncu_path.exists():
        sys.exit(f"ncu file not found: {ncu_path}")
    if not flow_path.exists():
        sys.exit(f"flow JSON not found: {flow_path}")

    print(f"Parsing ncu metrics from {ncu_path} ...")
    ncu_records = parse_ncu_csv(ncu_path)
    print(f"  Found {len(ncu_records)} kernel invocation records.")
    if not ncu_records:
        sys.exit("No ncu records found — check that ncu ran successfully.")

    with flow_path.open(encoding="utf-8") as f:
        flow = json.load(f)

    flow, matched, unmatched = attach_metrics(flow, ncu_records)

    # Add summary to graph metadata
    sm_totals = summarize_sm_metrics(flow.get("nodes", []))
    flow["graph"]["sm_metric_totals"] = {
        k: int(v) if "bytes" in k or "sectors" in k else round(v, 3)
        for k, v in sm_totals.items()
    }

    out_path = Path(args.out) if args.out else flow_path.with_name(
        flow_path.stem + "_ncu.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(flow, f, indent=4)

    print(f"Wrote {out_path}")
    print(f"  Kernel nodes matched: {matched}")
    if unmatched:
        print(f"  Unmatched kernel nodes (no ncu data): {unmatched}")
    if sm_totals:
        print("  SM metric totals:")
        for k, v in sm_totals.items():
            if "mb" in k:
                print(f"    {k}: {v:.1f} MB")
            elif "bytes" in k:
                print(f"    {k}: {int(v):,} bytes")


if __name__ == "__main__":
    main()
