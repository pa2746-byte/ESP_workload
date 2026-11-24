import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from export_workload_transfers import WORKLOADS, build_model


class TransferModelsTest(unittest.TestCase):
    def test_cli_from_another_directory(self):
        script = Path(__file__).resolve().parents[1] / "export_workload_transfers.py"
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, "-B", str(script), "--out-dir", "graphs"],
                           cwd=directory, check=True, capture_output=True, text=True)
            out = Path(directory) / "graphs"
            self.assertEqual(len(list(out.glob("*_simulator.json"))), 4)
            for name in WORKLOADS:
                self.assertEqual(json.loads((out / (name + "_simulator.json")).read_text()),
                                 build_model(name)[0])
                metadata = json.loads((out / (name + "_metadata.json")).read_text())
                self.assertEqual(len(metadata["source_sha256"]), 64)

    def test_reference_schema_and_dag(self):
        for name in WORKLOADS:
            graph, _ = build_model(name)
            self.assertEqual(set(graph), {"directed", "multigraph", "graph", "nodes", "edges"})
            for node in graph["nodes"]:
                self.assertEqual(set(node), {"id", "vol", "src_comp", "dest_comp"})
                self.assertIsInstance(node["vol"], int)
                self.assertGreater(node["vol"], 0)
            for edge in graph["edges"]:
                # IDs follow construction order, so this also proves acyclicity.
                self.assertLess(edge["source"], edge["target"])
                self.assertLess(edge["target"], len(graph["nodes"]))

    def test_fork_join_independence_and_join(self):
        graph, labels = build_model("fork_join")
        ancestors = {n["id"]: set() for n in graph["nodes"]}
        for n in graph["nodes"]:
            for e in graph["edges"]:
                if e["target"] == n["id"]:
                    ancestors[n["id"]].update(ancestors[e["source"]] | {e["source"]})
        writes = {n["stage"]: n["id"] for n in labels if n["stage"].endswith(":write")}
        add, sub, mul = [writes[s + ":write"] for s in ("vectorAdd", "vectorSub", "vectorMul")]
        self.assertNotIn(add, ancestors[sub])
        self.assertNotIn(sub, ancestors[add])
        self.assertTrue({add, sub} <= ancestors[mul])
        self.assertEqual(sum(n["vol"] for n in graph["nodes"]), 12 * (1 << 20) * 4)

    def test_reduction_sizes(self):
        graph, _ = build_model("reduction")
        self.assertEqual([n["vol"] for n in graph["nodes"]],
                         [262144, 262144, 1024, 1024, 4, 4])

    def test_component_mapping(self):
        graph, _ = build_model("vector_add", "cpu_0", "mem_0", "acc_0")
        endpoints = {n[k] for n in graph["nodes"] for k in ("src_comp", "dest_comp")}
        self.assertEqual(endpoints, {"cpu_0", "mem_0", "acc_0"})


if __name__ == "__main__":
    unittest.main()
