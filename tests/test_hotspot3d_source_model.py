"""Original Rodinia HotSpot3D: loops, stencils, and ping-pong buffers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cuda_source_model import UnsupportedSource, parse_source
from cuda_configured_model import analyze_configured

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'workloads/rodinia/hotspot3d'
SOURCE = DIRECTORY / '3D.cu'

class Hotspot3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ast = parse_source(SOURCE)
        cls.config = json.loads((DIRECTORY / 'analysis.json').read_text())

    def model(self, parameters=None, source=None, kernel=None, config=None):
        c = copy.deepcopy(config or self.config)
        c['parameters'].update(parameters or {})
        if source is None and kernel is None:
            with patch('cuda_configured_model.parse_source', return_value=self.ast):
                return analyze_configured(SOURCE, c)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'3D.cu').write_text(source if source is not None else SOURCE.read_text())
            (p/'opt1.cu').write_text(kernel if kernel is not None else (DIRECTORY/'opt1.cu').read_text())
            return analyze_configured(p/'3D.cu', c)

    def test_default_graph_and_actual_download(self):
        graph, metadata, details = self.model()
        self.assertEqual((len(graph['nodes']),len(graph['edges'])),(9,11))
        self.assertEqual({n['vol'] for n in graph['nodes']},{64*64*3*4})
        self.assertEqual([l['grid'] for l in details['launches']],[[1,16,1]]*2)
        self.assertEqual([l['block'] for l in details['launches']],[[64,4,1]]*2)
        self.assertEqual(set(details['source_units']),{'3D.cu','opt1.cu'})
        writes=[n for n in metadata if n['stage'].endswith(':write')]
        download=metadata[-1]
        self.assertEqual(download['buffer'],writes[0]['buffer'])
        self.assertNotEqual(download['buffer'],writes[1]['buffer'])
        self.assertEqual({e['source'] for e in graph['edges'] if e['target']==download['id']},{writes[0]['id']})

    def test_iterations_layers_and_overwrite_dependencies(self):
        for iterations,layers in [(1,2),(3,4)]:
            with self.subTest(iterations=iterations,layers=layers):
                graph, metadata, details=self.model({'iterations':iterations,'layers':layers})
                self.assertEqual(len(graph['nodes']),3+3*iterations)
                self.assertEqual({n['vol'] for n in graph['nodes']},{64*64*layers*4})
                self.assertEqual(len(details['launches']),iterations)
                writes=[n for n in metadata if n['stage'].endswith(':write')]
                if iterations==1:
                    self.assertEqual(metadata[-1]['buffer'],metadata[0]['buffer'])
                else:
                    self.assertEqual(writes[0]['buffer'],writes[2]['buffer'])
                    edges={(e['source'],e['target']) for e in graph['edges']}
                    self.assertIn((writes[0]['id'],writes[2]['id']),edges)
                    prior_reads=[n for n in metadata if n['buffer']==writes[0]['buffer'] and n['stage'].endswith(':read') and writes[0]['id']<n['id']<writes[2]['id']]
                    self.assertTrue(prior_reads)
                    self.assertTrue(all((r['id'],writes[2]['id']) in edges for r in prior_reads))

    def test_invalid_dimensions_and_budgets(self):
        for params,message in [({'rows':65},'Non-contiguous'),({'rows':32},'positive'),({'layers':1},'exceeds allocation')]:
            with self.subTest(params=params),self.assertRaisesRegex(UnsupportedSource,message):self.model(params)
        c=copy.deepcopy(self.config);c['max_enumerated_threads']=4096
        with self.assertRaisesRegex(UnsupportedSource,'max_enumerated_threads'):self.model(config=c)
        c=copy.deepcopy(self.config);c['max_loop_iterations']=1
        with self.assertRaisesRegex(UnsupportedSource,'Loop iteration budget'):self.model(config=c)

    def test_included_kernel_changes_are_analyzed(self):
        kernel=(DIRECTORY/'opt1.cu').read_text().replace('hotspotOpt1','renamedStencil').replace('tOut[c] =','tOut[c+1] =')
        with self.assertRaisesRegex(UnsupportedSource,'exceeds allocation'):self.model(kernel=kernel)
        kernel=(DIRECTORY/'opt1.cu').read_text().replace('hotspotOpt1','renamedStencil')
        graph,_,details=self.model(kernel=kernel)
        self.assertEqual(len(graph['nodes']),9)
        self.assertEqual(details['launches'][0]['kernel'],'renamedStencil')

    def test_corrected_download_changes_graph_without_template(self):
        # Test-only mutation: the vendored upstream implementation is left unchanged.
        kernel=(DIRECTORY/'opt1.cu').read_text().replace('cudaMemcpy(tOut, tOut_d,','cudaMemcpy(tOut, tIn_d,')
        graph,meta,_=self.model(kernel=kernel)
        write=[n for n in meta if n['stage'].endswith(':write')][-1]
        self.assertEqual(meta[-1]['buffer'],write['buffer'])
        self.assertIn({'source':write['id'],'target':meta[-1]['id']},graph['edges'])

    def test_changed_included_summary_is_rejected(self):
        kernel=(DIRECTORY/'opt1.cu').read_text().replace('return (tv.tv_sec * 1000000)', 'return (tv.tv_sec * 1000)')
        with self.assertRaisesRegex(UnsupportedSource,'Host summary source changed'):self.model(kernel=kernel)

    def test_repeated_allocation_names_rejected(self):
        source=SOURCE.read_text()
        call=next(line for line in source.splitlines() if '    hotspot_opt1(' in line)
        with self.assertRaisesRegex(UnsupportedSource,'Allocation reuse'):
            self.model(source=source.replace(call,call+'\n'+call))

if __name__=='__main__':unittest.main()
