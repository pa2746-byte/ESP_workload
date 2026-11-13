"""Independent, reviewed source-to-graph contracts for the published scenarios.

Expected graphs below do NOT call the exporter or its dependency-building code.
See workloads/rodinia/VALIDATION.md for the source derivation and exclusions.
Source hashes deliberately require a new review if the benchmark changes.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    'nn/nn_cuda.cu': 'cf4da3114f7f656be003f1aa2cdf424ae0a2179ec1c8bee4c6eb59172cccc390',
    'hotspot3d/3D.cu': '50c5074b5ce2d0603428bc37d3a15ed2b2cc4d247bfd639bafbc1a3ad61d0f11',
    'hotspot3d/opt1.cu': 'f8987fd50ab42b57303bc98c3bc4d91c81346573275be8967b20c89c66fe1257',
}


def expected_graph(nodes, edges):
    return dict(directed=True, multigraph=False, graph={},
                nodes=[dict(id=i, vol=v, src_comp=s, dest_comp=d)
                       for i, (v, s, d) in enumerate(nodes)],
                edges=[dict(source=s, target=t) for s, t in sorted(edges)])


# Independently derived from the input/output record types and euclid's guard.
NN = expected_graph([(4104,'cpu','mem'), (4104,'mem','acc'),
                     (2052,'acc','mem'), (2052,'mem','cpu')],
                    {(0,1),(1,2),(2,3)})
NN_ROLES = [('d_locations','upload'), ('d_locations','euclid:read'),
            ('d_distances','euclid:write'), ('d_distances','download')]
# From two stencil launches and the literal pointer swaps/download in opt1.cu.
HOTSPOT = expected_graph([(49152,s,d) for s,d in [
    ('cpu','mem'),('cpu','mem'),('mem','acc'),('mem','acc'),('acc','mem'),
    ('mem','acc'),('mem','acc'),('acc','mem'),('mem','cpu')]],
    {(0,3),(0,7),(1,2),(1,5),(2,4),(3,4),(3,7),(4,6),(4,8),(5,7),(6,7)})
HOTSPOT_ROLES = [('tIn_d','upload'),('p_d','upload'),
    ('p_d','hotspotOpt1:read'),('tIn_d','hotspotOpt1:read'),('tOut_d','hotspotOpt1:write'),
    ('p_d','hotspotOpt1:read'),('tOut_d','hotspotOpt1:read'),('tIn_d','hotspotOpt1:write'),
    ('tOut_d','download')]
CASES = [('nn','nn_cuda.cu','rodinia_nn/nn_cuda',NN,NN_ROLES),
         ('hotspot3d','3D.cu','rodinia_hotspot3d/3D',HOTSPOT,HOTSPOT_ROLES)]


class RodiniaGraphContractTests(unittest.TestCase):
    def test_reviewed_source_versions(self):
        for relative, expected in HASHES.items():
            with self.subTest(source=relative):
                self.assertEqual(hashlib.sha256((ROOT/'workloads/rodinia'/relative).read_bytes()).hexdigest(), expected,
                                 'Source changed: review the independent graph contract before updating this hash')

    def test_committed_graphs_match_independent_contracts(self):
        for _,_,output,expected,roles in CASES:
            with self.subTest(output=output):
                graph=json.loads((ROOT/'results'/f'{output}_simulator.json').read_text())
                metadata=json.loads((ROOT/'results'/f'{output}_metadata.json').read_text())
                self.assertEqual(graph,expected)  # every node, volume, endpoint, and edge
                self.assertEqual([(n['buffer'],n['stage']) for n in metadata['nodes']],roles)
                self.assertEqual([n['id'] for n in metadata['nodes']],list(range(len(roles))))

    def test_fresh_analysis_matches_independent_contracts(self):
        from cuda_configured_model import analyze_configured
        for folder,source,output,expected,roles in CASES:
            with self.subTest(source=source):
                directory=ROOT/'workloads/rodinia'/folder
                config=json.loads((directory/'analysis.json').read_text())
                metadata=json.loads((ROOT/'results'/f'{output}_metadata.json').read_text())
                graph,nodes,details=analyze_configured(directory/source,config)
                self.assertEqual(graph,expected)
                self.assertEqual(nodes,metadata['nodes'])
                self.assertEqual(config,metadata['configured_analysis']['configuration'])
                self.assertEqual(details['launches'],metadata['configured_analysis']['launches'])
                self.assertEqual(metadata['source_sha256'],HASHES[f'{folder}/{source}'])
                for name,digest in details['source_units'].items():
                    self.assertEqual(digest,HASHES[f'{folder}/{name}'])

    def test_nearest_neighbor_address_coverage(self):
        # Three blocks of 256; only indices < 513 access memory.
        active={block*256+thread for block in range(3) for thread in range(256)
                if block*256+thread<513}
        self.assertEqual(active,set(range(513)))
        self.assertEqual(len(active)*8,NN['nodes'][1]['vol'])
        self.assertEqual(len(active)*4,NN['nodes'][2]['vol'])

    def test_hotspot_stencil_address_coverage(self):
        # Independent coordinate-space formulation of the bounded 7-point stencil.
        reads,writes=set(),set()
        for z in range(3):
            for y in range(64):
                for x in range(64):
                    c=x+64*y+4096*z
                    writes.add(c)
                    reads.update([c, c if x==0 else c-1, c if x==63 else c+1,
                                  c if y==0 else c-64, c if y==63 else c+64,
                                  c if z==0 else c-4096, c if z==2 else c+4096])
        self.assertEqual(reads,set(range(12288)))
        self.assertEqual(writes,reads)
        self.assertEqual(len(reads)*4,HOTSPOT['nodes'][3]['vol'])
        self.assertEqual(len(writes)*4,HOTSPOT['nodes'][4]['vol'])

if __name__ == '__main__': unittest.main()
