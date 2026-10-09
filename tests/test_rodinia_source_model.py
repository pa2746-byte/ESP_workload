"""Checks against the unchanged upstream Rodinia source; no GPU is executed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from cuda_source_model import UnsupportedSource, parse_source
from cuda_configured_model import analyze_configured

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'workloads/rodinia/nn/nn_cuda.cu'
CONFIG = ROOT / 'workloads/rodinia/nn/analysis.json'

class RodiniaSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ast = parse_source(SOURCE)
        cls.configuration = json.loads(CONFIG.read_text())

    def run_model(self, config=None, text=None):
        if text is None:
            # Parsing is integration-tested once; the AST is read-only during evaluation.
            with patch('cuda_configured_model.parse_source', return_value=self.ast):
                return analyze_configured(SOURCE, config or self.configuration)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.cu'
            path.write_text(text)
            return analyze_configured(path, config or self.configuration)

    def check_graph(self, graph, records):
        self.assertEqual([n['vol'] for n in graph['nodes']], [8*records,8*records,4*records,4*records])
        self.assertEqual([(n['src_comp'],n['dest_comp']) for n in graph['nodes']],
                         [('cpu','mem'),('mem','acc'),('acc','mem'),('mem','cpu')])
        self.assertEqual(graph['edges'], [{'source':0,'target':1},{'source':1,'target':2},{'source':2,'target':3}])
        self.assertTrue(all(set(n)=={'id','vol','src_comp','dest_comp'} for n in graph['nodes']))

    def test_input_sizes_and_partial_blocks(self):
        for count in (1, 256, 513, 1025):
            with self.subTest(records=count):
                config = copy.deepcopy(self.configuration)
                config['parameters']['records'] = count
                g, _, details = self.run_model(config)
                self.check_graph(g, count)
                self.assertEqual(details['launches'][0]['grid'], [(count+255)//256,1,1])
                self.assertEqual(details['launches'][0]['block'], [256,1,1])

    def test_two_dimensional_grid(self):
        config = copy.deepcopy(self.configuration)
        config['device']['max_grid_size'][0] = 2
        g, _, d = self.run_model(config)
        self.check_graph(g, 513)
        self.assertEqual(d['launches'][0]['grid'], [2,2,1])
        self.assertEqual(d['launches'][0]['enumerated_threads'], 1024)

    def test_device_thread_limit_changes_launch(self):
        config = copy.deepcopy(self.configuration)
        config['device']['max_threads_per_block'] = 128
        g, _, d = self.run_model(config)
        self.check_graph(g,513)
        self.assertEqual(d['launches'][0]['grid'], [5,1,1])

    def test_cli_parameter_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-B', str(ROOT / 'export_workload_transfers.py'),
                            '--source', str(SOURCE), '--analysis-config', str(CONFIG),
                            '--parameter', 'records=257', '--out-dir', directory],
                           check=True, capture_output=True, text=True, cwd=directory)
            self.check_graph(json.loads((Path(directory) / 'nn_cuda_simulator.json').read_text()),257)
            metadata = json.loads((Path(directory) / 'nn_cuda_metadata.json').read_text())
            self.assertEqual(metadata['configured_analysis']['configuration']['parameters']['records'],257)
            self.assertEqual(metadata['configured_analysis']['used_host_summaries'],
                             ['findLowest','loadData','parseCommandline'])

    def test_source_names_not_templates(self):
        text = SOURCE.read_text().replace('euclid','distanceKernel').replace('d_locations','inputCoordinates').replace('d_distances','outputDistances')
        g, m, d = self.run_model(text=text)
        self.check_graph(g,513)
        self.assertEqual(d['launches'][0]['kernel'],'distanceKernel')
        self.assertEqual(m[0]['buffer'],'inputCoordinates')

    def test_invalid_configurations_fail(self):
        for edit, message in [
            (lambda c: c['parameters'].update(records=0), 'positive'),
            (lambda c: c.update(max_enumerated_threads=512), 'max_enumerated_threads'),
            (lambda c: c['device'].update(free_memory_bytes=12), 'exit/error'),
            (lambda c: c['device']['max_grid_size'].__setitem__(0,0), 'positive'),
        ]:
            config=copy.deepcopy(self.configuration);edit(config)
            with self.subTest(message=message), self.assertRaisesRegex(UnsupportedSource,message): self.run_model(config)

    def test_unsafe_source_changes_fail(self):
        original = SOURCE.read_text()
        variants = [
            (original.replace('globalId < numRecords','globalId <= numRecords'), 'exceeds allocation'),
            (original.replace('d_distances+globalId','d_distances+0'), 'Multiple threads'),
            (original.replace('if (globalId < numRecords)', 'if (latLong->lat > 0)'), 'Data-dependent'),
            (original.replace('return recNum;', 'return recNum + 1;'), 'Host summary source changed'),
            (original.replace('#define REC_LENGTH 53','#define REC_LENGTH 54'), 'Preprocessor directives changed'),
        ]
        for text,message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(UnsupportedSource,message): self.run_model(text=text)

if __name__ == '__main__': unittest.main()
