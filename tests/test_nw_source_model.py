"""Independent cell geometry for the unmodified Rodinia NW application."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from cuda_configured_model import analyze_configured, directives_hash
from cuda_source_model import UnsupportedSource

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'workloads/rodinia/nw/needle.cu'
CONFIG = json.loads(SOURCE.with_name('analysis.json').read_text())


def expected_events(n):
    """Model matrix tile diagonals, independently of CUDA launch/thread indices."""
    stride = n + 1
    full = set(range(stride * stride))
    events = [('referrence_cuda', 'write', full, None),
              ('matrix_cuda', 'write', full, None)]
    tiles = n // 16
    for diagonal in range(2 * tiles - 1):
        interior, boundary = set(), set()
        for row in range(tiles):
            col = diagonal - row
            if not 0 <= col < tiles:
                continue
            top, left = row * 16, col * 16
            interior |= {(top + r) * stride + left + c
                         for r in range(1, 17) for c in range(1, 17)}
            boundary |= {top * stride + left + c for c in range(17)}
            boundary |= {(top + r) * stride + left for r in range(1, 17)}
        events += [('referrence_cuda', 'read', interior, diagonal),
                   ('matrix_cuda', 'read', boundary, diagonal),
                   ('matrix_cuda', 'write', interior, diagonal)]
    events.append(('matrix_cuda', 'read', full, None))
    return events


def expected_edges(events):
    edges = set()
    for target, (buf, mode, cells, launch) in enumerate(events):
        for source, (prev_buf, prev_mode, prev_cells, prev_launch) in enumerate(events[:target]):
            if launch is not None and launch == prev_launch and mode == 'write' and prev_mode == 'read':
                edges.add((source, target))
            if buf != prev_buf or mode == prev_mode == 'read':
                continue
            remaining = cells & prev_cells
            for between_buf, between_mode, between_cells, _ in events[source + 1:target]:
                if between_buf == buf and between_mode == 'write':
                    remaining -= between_cells
            if remaining:
                edges.add((source, target))
    return edges


class NeedlemanWunschTests(unittest.TestCase):
    def test_published_artifacts_match_fresh_analysis(self):
        graph, nodes, details = analyze_configured(SOURCE, CONFIG)
        output = ROOT / 'results/rodinia_nw'
        self.assertEqual(json.loads((output / 'needle_simulator.json').read_text()), graph)
        metadata = json.loads((output / 'needle_metadata.json').read_text())
        self.assertEqual(metadata['nodes'], nodes)
        self.assertEqual(metadata['configured_analysis'], details)

    def test_complete_wavefront_geometry(self):
        for dimension in (16, 32, 48):
            with self.subTest(dimension=dimension):
                config = copy.deepcopy(CONFIG)
                config['parameters']['dimension'] = dimension
                graph, metadata, details = analyze_configured(SOURCE, config)
                events = expected_events(dimension)
                self.assertEqual(len(graph['nodes']), len(events))
                self.assertEqual(len(details['launches']), 2 * (dimension // 16) - 1)
                self.assertEqual(details['used_host_summaries'], [])
                for node, meta, (buffer, mode, cells, _) in zip(graph['nodes'], metadata, events):
                    self.assertEqual(meta['buffer'], buffer)
                    expected_bytes = {4 * cell + b for cell in cells for b in range(4)}
                    actual_bytes = {b for start, end in meta['byte_ranges'] for b in range(start, end)}
                    self.assertEqual(actual_bytes, expected_bytes)
                    self.assertEqual(node['vol'], len(expected_bytes))
                    self.assertEqual(node['dest_comp'] == 'mem', mode == 'write')
                self.assertEqual({(e['source'], e['target']) for e in graph['edges']}, expected_edges(events))

    def test_nested_and_loop_scopes_restore_outer_indices(self):
        code = """#include <cuda_runtime.h>
__global__ void copy(float *a, float *b) {
    int i = threadIdx.x;
    { i += 1; int i = 99; }
    i -= 1;
    for (int i = 0; i < 2; ++i) {}
    if (i == 0) { int i = 100; }
    b[i] = a[i];
}
int main() {
    float *h = (float*)malloc(16), *a, *b;
    cudaMalloc((void**)&a, 16); cudaMalloc((void**)&b, 16);
    cudaMemcpy(a, h, 16, cudaMemcpyHostToDevice);
    copy<<<1,4>>>(a,b);
    cudaMemcpy(h, b, 16, cudaMemcpyDeviceToHost);
    cudaFree(a); cudaFree(b); free(h); return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'scope.cu'
            source.write_text(code)
            config = copy.deepcopy(CONFIG)
            config.update(additional_sources=[], scalar_device_functions=[],
                          analyzed_functions=[], arguments=[],
                          preprocessor_sha256=directives_hash(source.read_bytes()))
            graph, _, _ = analyze_configured(source, config)
            self.assertEqual([n['vol'] for n in graph['nodes']], [16, 16, 16, 16])

    def test_upstream_sources_unchanged(self):
        expected = json.loads(SOURCE.with_name('upstream.json').read_text())
        for name, sha in expected['sha256'].items():
            self.assertEqual(hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest(), sha)

    def test_invalid_dimension_rejected(self):
        config = copy.deepcopy(CONFIG)
        config['parameters']['dimension'] = 17
        with self.assertRaisesRegex(UnsupportedSource, 'exit/error path'):
            analyze_configured(SOURCE, config)

    def test_penalty_does_not_change_footprints(self):
        other = copy.deepcopy(CONFIG)
        other['parameters']['penalty'] = 3
        self.assertEqual(analyze_configured(SOURCE, CONFIG)[0], analyze_configured(SOURCE, other)[0])

    def mutate(self, file, old, new):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for name in ('needle.cu', 'needle_kernel.cu', 'needle.h'):
            shutil.copyfile(SOURCE.with_name(name), directory / name)
        target = directory / file
        content = target.read_bytes()
        self.assertIn(old.encode(), content)
        target.write_bytes(content.replace(old.encode(), new.encode(), 1))
        return directory / 'needle.cu'

    def test_host_array_initializer_cannot_hide_calls(self):
        source = self.mutate('needle.cu', '{ 4, -1, -2', '{ maximum(4,4,4), -1, -2')
        with self.assertRaisesRegex(UnsupportedSource, 'initializer must be pure numeric constants'):
            analyze_configured(source, CONFIG)

    def test_scalar_helper_cannot_hide_calls(self):
        source = self.mutate('needle_kernel.cu', 'k = b;', 'k = maximum(a, b, c);')
        with self.assertRaisesRegex(UnsupportedSource, 'Scalar helper has unsupported effects'):
            analyze_configured(source, CONFIG)

    def test_host_data_cannot_determine_launch_count(self):
        source = self.mutate('needle.cu', 'int block_width = ( max_cols - 1 )/BLOCK_SIZE;',
                             'int block_width = input_itemsets[0];')
        with self.assertRaisesRegex(UnsupportedSource, 'unresolved condition'):
            analyze_configured(source, CONFIG)

    def test_host_pointer_cannot_replace_device_input(self):
        source = self.mutate('needle.cu', '>>>(referrence_cuda, matrix_cuda',
                             '>>>(referrence, matrix_cuda')
        with self.assertRaisesRegex(UnsupportedSource, 'ordinary host pointer'):
            analyze_configured(source, CONFIG)

    def test_out_of_bounds_kernel_is_rejected(self):
        source = self.mutate('needle_kernel.cu', 'matrix_cuda[index_nw]', 'matrix_cuda[cols * cols]')
        with self.assertRaisesRegex(UnsupportedSource, 'exceeds allocation'):
            analyze_configured(source, CONFIG)


if __name__ == '__main__':
    unittest.main()
