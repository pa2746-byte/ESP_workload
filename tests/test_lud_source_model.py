"""Independent LU block geometry, covering the complete original host driver."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from cuda_configured_model import analyze_configured, Evaluator, Pointer, Reference
from cuda_source_model import UnsupportedSource
from test_nw_source_model import expected_edges

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'workloads/rodinia/lud/analysis_unit.cu'
CONFIG = json.loads(SOURCE.with_name('analysis.json').read_text())


def expected_events(n):
    def rectangle(r0, r1, c0, c1):
        return {r * n + c for r in range(r0, r1) for c in range(c0, c1)}
    events = [('d_m', 'write', set(range(n*n)), None)]
    launch = 0
    for offset in range(0, n, 16):
        end = offset + 16
        diagonal = rectangle(offset, end, offset, end)
        events += [('d_m', 'read', diagonal, launch),
                   ('d_m', 'write', rectangle(offset+1, end, offset, end), launch)]
        launch += 1
        if end == n: continue
        row = rectangle(offset, end, end, n)
        column = rectangle(end, n, offset, end)
        events += [('d_m', 'read', diagonal | row | column, launch),
                   ('d_m', 'write', rectangle(offset+1, end, end, n) | column, launch)]
        launch += 1
        trailing = rectangle(end, n, end, n)
        events += [('d_m', 'read', row | column | trailing, launch),
                   ('d_m', 'write', trailing, launch)]
        launch += 1
    events.append(('d_m', 'read', set(range(n*n)), None))
    return events


class LUDecompositionTests(unittest.TestCase):
    def test_published_artifacts_match_fresh_analysis(self):
        graph, nodes, details = analyze_configured(SOURCE, CONFIG)
        output = ROOT / 'results/rodinia_lud'
        self.assertEqual(json.loads((output / 'analysis_unit_simulator.json').read_text()), graph)
        metadata = json.loads((output / 'analysis_unit_metadata.json').read_text())
        self.assertEqual(metadata['nodes'], nodes)
        self.assertEqual(metadata['configured_analysis'], details)

    def test_complete_driver_and_block_regions(self):
        for dimension in (16, 32, 48):
            with self.subTest(dimension=dimension):
                config = copy.deepcopy(CONFIG)
                config['parameters']['dimension'] = dimension
                graph, metadata, details = analyze_configured(SOURCE, config)
                events = expected_events(dimension)
                self.assertEqual(len(graph['nodes']), len(events))
                self.assertEqual(len(details['launches']), 3*(dimension//16)-2)
                for node, meta, (_, mode, cells, _) in zip(graph['nodes'], metadata, events):
                    expected_bytes = {4 * cell + b for cell in cells for b in range(4)}
                    actual_bytes = {b for start, end in meta['byte_ranges'] for b in range(start, end)}
                    self.assertEqual(actual_bytes, expected_bytes)
                    self.assertEqual(node['vol'], len(expected_bytes))
                    self.assertEqual(node['dest_comp'] == 'mem', mode == 'write')
                self.assertEqual({(e['source'], e['target']) for e in graph['edges']}, expected_edges(events))
                self.assertEqual(set(details['used_host_summaries']), {'create_matrix', 'stopwatch_start', 'stopwatch_stop', 'get_interval_by_sec'})

    def test_original_files_unchanged(self):
        manifest = json.loads(SOURCE.with_name('upstream.json').read_text())
        for name, sha in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest(), sha)

    def test_host_contract_cannot_mask_different_input_size(self):
        config = copy.deepcopy(CONFIG)
        config['arguments'] = ['-s', '48']
        with self.assertRaisesRegex(UnsupportedSource, 'argument does not match'):
            analyze_configured(SOURCE, config)

    def test_summary_cannot_hide_device_pointers_in_aggregates(self):
        pointer = Pointer('device_buffer', 0, 'float', True)
        nested = {'payload': [0, {'buffer': pointer}]}
        self.assertTrue(Evaluator.contains_device_pointer(nested))
        self.assertTrue(Evaluator.contains_device_pointer(Reference({'item': nested}, 'item')))
        self.assertFalse(Evaluator.contains_device_pointer({'host': [Pointer('h', 0, 'float')]}))

    def test_host_buffer_contract_checks_pointer_type(self):
        config = copy.deepcopy(CONFIG)
        config['host_summaries']['create_matrix']['buffers']['0']['base'] = 'int'
        with self.assertRaisesRegex(UnsupportedSource, 'output pointer'):
            analyze_configured(SOURCE, config)

    def test_bad_options_rejected(self):
        for arguments in (['-s'], ['-q'], ['--size', '32']):
            with self.subTest(arguments=arguments):
                config = copy.deepcopy(CONFIG)
                config['arguments'] = arguments
                with self.assertRaises(UnsupportedSource):
                    analyze_configured(SOURCE, config)

    def test_non_tile_dimension_rejected(self):
        config = copy.deepcopy(CONFIG)
        config['parameters']['dimension'] = 17
        with self.assertRaises(UnsupportedSource):
            analyze_configured(SOURCE, config)


if __name__ == '__main__':
    unittest.main()
