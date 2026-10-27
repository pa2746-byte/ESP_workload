"""Independent coordinate-space contracts for upstream Gaussian's GPU phase.

The oracle never reads the AST or calls the production dependency builder.
It derives cells from elimination mathematics and checks historical overlaps.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cuda_source_model import UnsupportedSource, parse_source
from cuda_configured_model import analyze_configured

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'workloads/rodinia/gaussian'
SOURCE = DIRECTORY / 'gaussian.cu'


def expected(size, fan1_gap=1):
    events = []
    def add(buffer, mode, cells, stage, launch=None):
        events.append((buffer, mode, set(cells), stage, launch))
    for b, count in [('m_cuda', size*size), ('a_cuda', size*size), ('b_cuda', size)]:
        add(b, 'write', range(count), 'upload')
    for t in range(size - 1):
        rows = range(t + 1, size)
        multiplier_rows = range(t + 1, size - fan1_gap + 1)
        # Each active Fan1 thread reads one entry below the pivot and the pivot.
        cells = {r*size+t for r in multiplier_rows}
        if cells:
            add('a_cuda', 'read', cells | {t*size+t}, 'Fan1:read', 2*t)
            add('m_cuda', 'write', cells, 'Fan1:write', 2*t)
        # Fan2 reads multipliers, the trailing matrix INCLUDING its pivot row,
        # and the trailing RHS. It updates only rows BELOW the pivot.
        add('m_cuda', 'read', [r*size+t for r in rows], 'Fan2:read', 2*t+1)
        add('a_cuda', 'read', [r*size+c for r in range(t,size) for c in range(t,size)], 'Fan2:read', 2*t+1)
        add('b_cuda', 'read', range(t,size), 'Fan2:read', 2*t+1)
        add('a_cuda', 'write', [r*size+c for r in rows for c in range(t,size)], 'Fan2:write', 2*t+1)
        add('b_cuda', 'write', rows, 'Fan2:write', 2*t+1)
    for b, count in [('m_cuda', size*size), ('a_cuda', size*size), ('b_cuda', size)]:
        add(b, 'read', range(count), 'download')
    edges = set()
    # A prior writer (or reader before a new write) is needed if at least one
    # shared cell has not been overwritten between the two events.
    for j, (b, mode, cells, stage, launch) in enumerate(events):
        for i, (pb, pm, pc, ps, pl) in enumerate(events[:j]):
            if launch is not None and launch == pl and mode == 'write' and pm == 'read':
                edges.add((i,j))
            if b != pb or (mode == pm == 'read'):
                continue
            overlap = cells & pc
            for ib, im, ic, _, _ in events[i+1:j]:
                if ib == b and im == 'write': overlap -= ic
            if overlap: edges.add((i,j))
    nodes = []
    for i, (_, mode, cells, stage, _) in enumerate(events):
        src,dst = ('cpu','mem') if stage=='upload' else ('mem','cpu') if stage=='download' else ('mem','acc') if mode=='read' else ('acc','mem')
        nodes.append(dict(id=i,vol=len(cells)*4,src_comp=src,dest_comp=dst))
    return dict(directed=True,multigraph=False,graph={},nodes=nodes,
                edges=[dict(source=s,target=t) for s,t in sorted(edges)]), events


class GaussianSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ast = parse_source(SOURCE)
        cls.config = json.loads((DIRECTORY/'analysis.json').read_text())

    def analyze(self, size=3):
        config = copy.deepcopy(self.config)
        config['parameters']['size'] = size
        with patch('cuda_configured_model.parse_source', return_value=self.ast):
            return analyze_configured(SOURCE,config)

    def mutate(self, old, new):
        source = SOURCE.read_text()
        self.assertIn(old,source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gaussian.cu'
            path.write_text(source.replace(old,new))
            return analyze_configured(path,self.config)

    def test_upstream_source_unchanged(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         '814fca1b71505b177062a44f87f35680d0aa3a86311ef5ff6bf4b6a4e462540b')

    def test_independent_graphs_and_footprints_for_multiple_sizes(self):
        for size in (2,3,4,5):
            with self.subTest(size=size):
                graph,metadata,details = self.analyze(size)
                oracle,events = expected(size)
                self.assertEqual(graph,oracle)
                self.assertEqual(len(graph['nodes']),6+7*(size-1))
                self.assertEqual([l['kernel'] for l in details['launches']],['Fan1','Fan2']*(size-1))
                for node, (buffer,mode,cells,stage,_) in zip(metadata,events):
                    self.assertEqual((node['buffer'],node['stage']),(buffer,stage))
                    footprint = {b for start,end in node['byte_ranges'] for b in range(start,end)}
                    self.assertEqual(footprint,{4*c+b for c in cells for b in range(4)})

    def test_published_graph_matches_independent_contract(self):
        graph,nodes,details = self.analyze()
        path=ROOT/'results/rodinia_gaussian'
        self.assertEqual(json.loads((path/'gaussian_simulator.json').read_text()),expected(3)[0])
        metadata=json.loads((path/'gaussian_metadata.json').read_text())
        self.assertEqual(metadata['nodes'],nodes)
        self.assertEqual(metadata['configured_analysis'],details)

    def test_changed_kernel_guard_changes_graph(self):
        graph,_,_=self.mutate('>= Size-1-t) return;\n\t*(m_cuda',
                             '>= Size-2-t) return;\n\t*(m_cuda')
        self.assertEqual(graph,expected(3,fan1_gap=2)[0])
        self.assertNotEqual(graph,expected(3)[0])

    def test_cross_thread_read_write_race_rejected(self):
        with self.assertRaisesRegex(UnsupportedSource,'Cross-thread'):
            self.mutate('a_cuda[Size*t+(yidx+t)]','a_cuda[Size*(t+1)+(yidx+t)]')

    def test_partial_uninitialized_download_rejected(self):
        with self.assertRaisesRegex(UnsupportedSource,'uninitialized device region'):
            self.mutate('cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );',';')

    def test_out_of_bounds_kernel_rejected(self):
        with self.assertRaisesRegex(UnsupportedSource,'exceeds allocation'):
            self.mutate('m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t',
                        'm_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+2)+t')

    def test_unknown_entry_parameter_rejected(self):
        config=copy.deepcopy(self.config)
        del config['entry_globals']['Size']
        with patch('cuda_configured_model.parse_source',return_value=self.ast):
            with self.assertRaises(UnsupportedSource): analyze_configured(SOURCE,config)

    def test_region_memory_budget_rejected_before_enumeration(self):
        config=copy.deepcopy(self.config)
        config['max_tracked_bytes']=16
        with patch('cuda_configured_model.parse_source',return_value=self.ast):
            with self.assertRaisesRegex(UnsupportedSource,'max_tracked_bytes'):
                analyze_configured(SOURCE,config)

if __name__=='__main__': unittest.main()
