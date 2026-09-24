from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from cuda_source_model import analyze, UnsupportedSource, parse_source, walk, children, ref


ROOT = Path(__file__).resolve().parents[1]
VECTOR = (ROOT / "workloads/vector_add.cu").read_text()


class SourceAnalysisTest(unittest.TestCase):
    def test_modern_cuda_launch_helper(self):
        real_run = subprocess.run

        def modern_sdk(command, **kwargs):
            return real_run(command[:-1] + ["-Xclang", "-target-sdk-version=12.0", command[-1]], **kwargs)

        with patch("cuda_source_model.subprocess.run", side_effect=modern_sdk):
            source = ROOT / "workloads/vector_add.cu"
            ast = parse_source(source)
            calls = [n for n in walk(ast) if n["kind"] == "CUDAKernelCallExpr"]
            self.assertEqual(ref(children(children(calls[0])[1])[0]), "__cudaPushCallConfiguration")
            for name in ("vector_add", "pipeline", "fork_join", "reduction"):
                graph, _ = analyze(ROOT / "workloads" / (name + ".cu"))
                self.assertTrue(graph["nodes"])

    def analyze_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unrelated_filename.cu"
            source.write_text(text)
            return analyze(source)

    def test_changed_size_and_kernel_name(self):
        graph, labels = self.analyze_text(VECTOR.replace("1 << 20", "1 << 12")
                                         .replace("vectorAdd", "customCalculation"))
        self.assertEqual([n["vol"] for n in graph["nodes"]], [16384] * 6)
        self.assertIn("customCalculation:write", [n["stage"] for n in labels])

    def test_changed_kernel_input(self):
        graph, labels = self.analyze_text(VECTOR.replace("A[i] + B[i]", "A[i] * 2.0f"))
        reads = [n for n in labels if n["stage"].endswith(":read")]
        self.assertEqual([n["buffer"] for n in reads], ["d_A"])
        self.assertEqual(len(graph["nodes"]), 5)

    def test_dependency_changes_from_source(self):
        text = (ROOT / "workloads/fork_join.cu").read_text()
        text = text.replace("vectorSub<<<blocks, threads>>>(d_A, d_B, d_D, N)",
                            "vectorSub<<<blocks, threads>>>(d_C, d_B, d_D, N)")
        graph, labels = self.analyze_text(text)
        producer = next(n["id"] for n in labels if n["stage"] == "vectorAdd:write")
        consumer = next(n["id"] for n in labels if n["stage"] == "vectorSub:read" and n["buffer"] == "d_C")
        self.assertIn(dict(source=producer, target=consumer), graph["edges"])

    def test_reduction_sizes_come_from_source(self):
        text = (ROOT / "workloads/reduction.cu").read_text().replace("1 << 16", "1 << 15")
        graph, _ = self.analyze_text(text)
        self.assertEqual([n["vol"] for n in graph["nodes"]], [131072, 131072, 512, 512, 4, 4])

    def test_overwrite_waits_for_prior_reads(self):
        launch = "vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);"
        graph, labels = self.analyze_text(VECTOR.replace(
            launch, launch + "\n    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);"))
        read = next(n["id"] for n in labels if n["stage"] == "vectorAdd:read" and n["buffer"] == "d_A")
        uploads = [n["id"] for n in labels if n["stage"] == "upload" and n["buffer"] == "d_A"]
        self.assertIn(dict(source=read, target=uploads[-1]), graph["edges"])

    def test_duplicate_expression_counts_unique_footprint(self):
        graph, _ = self.analyze_text(VECTOR.replace("A[i] + B[i]", "A[i] + A[i] + B[i]"))
        self.assertEqual([n["vol"] for n in graph["nodes"]], [4194304] * 6)

    def test_larger_reduction_reflects_actual_read_extent(self):
        text = (ROOT / "workloads/reduction.cu").read_text().replace("1 << 16", "1 << 17")
        # The source only reduces 256 of the 512 partial sums, but that remains
        # a valid read footprint. Its input graph reflects that source behavior.
        graph, _ = self.analyze_text(text)
        self.assertEqual([n["vol"] for n in graph["nodes"]], [524288, 524288, 2048, 1024, 4, 4])

    def test_unsupported_sources_fail(self):
        variants = [
            VECTOR.replace("A[i] + B[i]", "A[i * 2] + B[i]"),
            VECTOR.replace("A[i] + B[i]", "A[(int)B[i]]"),
            VECTOR.replace("if (i < N)", "if (A[i] > 0)"),
            VECTOR.replace("C[i] =", "C[0] ="),
            VECTOR.replace("int blocks =", "float *alias = d_A;\n    int blocks ="),
            VECTOR.replace("vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);",
                           "if (N > 0) vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);"),
            VECTOR.replace("i++) {", "i++) { N;", 1).replace("const int N", "int N")
                  .replace("h_A[i] = 1.0f;", "N -= 1; h_A[i] = 1.0f;"),
            VECTOR.replace("cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);", ""),
            VECTOR.replace("C[i] = A[i] + B[i];", "C[i] = A[i] + B[i];\n    return;"),
        ]
        for text in variants:
            with self.subTest(source=text), self.assertRaises(UnsupportedSource):
                self.analyze_text(text)


if __name__ == "__main__":
    unittest.main()
