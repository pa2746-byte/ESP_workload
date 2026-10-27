"""Independent global-footprint contracts for original HotSpot 2D / Pathfinder.

Expected graphs use mathematical grid/row regions and historical overlaps,
not AST traversal, emitted output, or the production dependency builder.
Shared-memory numerical correctness and barrier correctness are not claimed.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cuda_source_model import UnsupportedSource,parse_source
from cuda_configured_model import analyze_configured, body_hash, source_declarations

ROOT=Path(__file__).resolve().parents[1]
CASES={'hotspot2d':'hotspot','pathfinder':'pathfinder'}
HASHES={'hotspot2d':'0f679be008be285d26ebd7d71eba47b164914dd52873c26ee63beecda2f68cc0','pathfinder':'b1084864b2efc0b487f97463b055b809690aa948bf0308d1574d00788f8b3cff'}


def contract(case,params):
    """Source-derived region schedule; all indices here are float/int cells."""
    if case=='hotspot2d':
        extent=params['size']**2
        steps=params['iterations']
        result,power,kernel='MatrixTemp','MatrixPower','calculate_temp'
        fixed_cells=set(range(extent))
    else:
        extent=params['cols']
        steps=params['rows']-1
        result,power,kernel='gpuResult','gpuWall','dynproc_kernel'
        fixed_cells=set(range(steps*extent))
    pyramid=params['pyramid']
    events=[(result+'[0]','write',set(range(extent)),'upload',None),
            (power,'write',fixed_cells,'upload',None)]
    src=0
    for launch,start in enumerate(range(0,steps,pyramid)):
        count=min(pyramid,steps-start)
        # Temperature/power touch the full domain; Pathfinder consumes exactly
        # the next count wall rows, relative to the upload that excludes row 0.
        consumed=fixed_cells if case=='hotspot2d' else set(range(start*extent,(start+count)*extent))
        events += [(f'{result}[{src}]','read',set(range(extent)),kernel+':read',launch),
                   (power,'read',consumed,kernel+':read',launch),
                   (f'{result}[{1-src}]','write',set(range(extent)),kernel+':write',launch)]
        src=1-src
    events.append((f'{result}[{src}]','read',set(range(extent)),'download',None))
    edges=set()
    for j,(buf,mode,cells,stage,launch) in enumerate(events):
        for i,(pb,pm,pc,ps,pl) in enumerate(events[:j]):
            if launch is not None and launch==pl and mode=='write' and pm=='read':edges.add((i,j))
            if buf!=pb or mode==pm=='read':continue
            live=cells & pc
            for ib,im,ic,_,_ in events[i+1:j]:
                if ib==buf and im=='write':live-=ic
            if live:edges.add((i,j))
    nodes=[]
    for i,(_,mode,cells,stage,_) in enumerate(events):
        src,dst=('cpu','mem') if stage=='upload' else ('mem','cpu') if stage=='download' else ('mem','acc') if mode=='read' else ('acc','mem')
        nodes.append(dict(id=i,vol=4*len(cells),src_comp=src,dest_comp=dst))
    return dict(directed=True,multigraph=False,graph={},nodes=nodes,
                edges=[dict(source=s,target=t) for s,t in sorted(edges)]),events


def tile_coverage(length,tile,pyramid,iterations):
    """Independent interval geometry for loaded tiles and cropped output tiles."""
    blocks=(length+tile-2*pyramid-1)//(tile-2*pyramid)
    loaded,written=set(),set()
    for b in range(blocks):
        origin=b*(tile-2*iterations)-pyramid
        loaded.update(range(max(0,origin),min(length,origin+tile)))
        written.update(range(max(0,origin+iterations),min(length,origin+tile-iterations)))
    return loaded,written,blocks


class TiledRodiniaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources={k:ROOT/'workloads/rodinia'/k/(v+'.cu') for k,v in CASES.items()}
        cls.configs={k:json.loads(p.with_name('analysis.json').read_text()) for k,p in cls.sources.items()}
        cls.asts={k:parse_source(p) for k,p in cls.sources.items()}

    def analyze(self,case,params=None,config=None,text=None):
        config=copy.deepcopy(config or self.configs[case])
        if params:config['parameters'].update(params)
        if text is not None:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/self.sources[case].name;p.write_bytes(text.encode())
                return analyze_configured(p,config)
        with patch('cuda_configured_model.parse_source',return_value=self.asts[case]):
            return analyze_configured(self.sources[case],config)

    def check(self,case,params):
        graph,metadata,details=self.analyze(case,params)
        oracle,events=contract(case,params)
        self.assertEqual(graph,oracle)
        for m,(buffer,mode,cells,stage,_) in zip(metadata,events):
            self.assertEqual((m['buffer'],m['stage']),(buffer,stage))
            self.assertEqual({b for a,z in m['byte_ranges'] for b in range(a,z)},
                             {4*c+b for c in cells for b in range(4)})
        length=params['size'] if case=='hotspot2d' else params['cols']
        steps=params['iterations'] if case=='hotspot2d' else params['rows']-1
        tile=16 if case=='hotspot2d' else 256
        for launch,start in zip(details['launches'],range(0,steps,params['pyramid'])):
            loaded,written,blocks=tile_coverage(length,tile,params['pyramid'],min(params['pyramid'],steps-start))
            self.assertEqual(loaded,set(range(length)))
            self.assertEqual(written,set(range(length)))
            self.assertEqual(launch['grid'],[blocks,blocks,1] if case=='hotspot2d' else [blocks,1,1])
            self.assertEqual(launch['block'],[16,16,1] if case=='hotspot2d' else [256,1,1])
        return graph,metadata,details

    def test_upstream_sources_unchanged(self):
        for case,p in self.sources.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),HASHES[case])

    def test_hotspot_tile_boundaries_and_iterations(self):
        for size,pyramid,iterations in [(1,1,1),(12,2,4),(16,2,4),(17,2,3),(25,3,7)]:
            with self.subTest(size=size,pyramid=pyramid,iterations=iterations):
                self.check('hotspot2d',dict(size=size,pyramid=pyramid,iterations=iterations))

    def test_pathfinder_tile_boundaries_and_iterations(self):
        for cols,rows,pyramid in [(1,2,1),(16,5,2),(252,5,2),(257,6,2),(511,8,3)]:
            with self.subTest(cols=cols,rows=rows,pyramid=pyramid):
                self.check('pathfinder',dict(cols=cols,rows=rows,pyramid=pyramid))

    def test_published_graphs_and_metadata(self):
        for case,name in CASES.items():
            graph,nodes,details=self.check(case,self.configs[case]['parameters'])
            root=ROOT/'results'/('rodinia_'+case)
            self.assertEqual(json.loads((root/(name+'_simulator.json')).read_text()),graph)
            meta=json.loads((root/(name+'_metadata.json')).read_text())
            self.assertEqual(meta['nodes'],nodes)
            self.assertEqual(meta['configured_analysis'],details)
            self.assertTrue(any('barrier correctness' in s for s in meta['assumptions']))

    def test_renamed_kernel_is_not_a_graph_template(self):
        source=self.sources['hotspot2d'].read_bytes().decode().replace('calculate_temp','thermal_step')
        graph,nodes,_=self.analyze('hotspot2d',text=source)
        self.assertEqual(graph,contract('hotspot2d',self.configs['hotspot2d']['parameters'])[0])
        self.assertTrue(any(n['stage']=='thermal_step:read' for n in nodes))

    def test_changed_download_pointer_changes_dependency(self):
        source=self.sources['pathfinder'].read_bytes().decode().replace('gpuResult[final_ret]','gpuResult[1-final_ret]')
        graph,nodes,_=self.analyze('pathfinder',text=source)
        self.assertEqual(nodes[-1]['buffer'],'gpuResult[1]')
        self.assertEqual([e['source'] for e in graph['edges'] if e['target']==8],[4])

    def test_shared_values_require_explicit_opt_in(self):
        config=copy.deepcopy(self.configs['hotspot2d']);del config['shared_memory']
        with self.assertRaisesRegex(UnsupportedSource,'shared numeric arrays'):
            self.analyze('hotspot2d',config=config)

    def test_shared_bounds_fail(self):
        source=self.sources['hotspot2d'].read_bytes().decode().replace('temp_t[ty][tx];','temp_t[ty][tx+16];')
        with self.assertRaisesRegex(UnsupportedSource,'Shared array index out of bounds'):
            self.analyze('hotspot2d',text=source)

    def test_unknown_shared_control_and_address_fail(self):
        original=self.sources['pathfinder'].read_bytes().decode()
        for old,new in [('if (computed){','if (prev[tx]){'),('gpuResults[xidx]=','gpuResults[prev[tx]]=')]:
            with self.subTest(change=new),self.assertRaisesRegex(UnsupportedSource,'Data-dependent'):
                self.analyze('pathfinder',text=original.replace(old,new))
        # Even an explicitly stored constant must stay unknown: this mode does
        # not prove which shared write each thread observes.
        source=original.replace('prev[tx] = gpuSrc[xidx];','prev[tx] = 1;').replace('if (computed){','if (prev[tx]){')
        with self.assertRaisesRegex(UnsupportedSource,'Data-dependent'):
            self.analyze('pathfinder',text=source)

    def test_host_summary_cannot_hide_device_pointer_array(self):
        case='pathfinder'
        config=copy.deepcopy(self.configs[case])
        config['analyzed_functions'].remove('calc_path')
        declarations,_=source_declarations(self.asts[case],self.sources[case])
        fn=next(n for n,_ in declarations if n.get('name')=='calc_path' and n.get('kind')=='FunctionDecl')
        config['host_summaries']['calc_path']={'body_sha256':body_hash(fn,self.sources[case].read_bytes())}
        # With the direct pointer replaced by nullptr, only the pointer array
        # exposes device memory. The summary must still be rejected.
        for argument in ['gpuResult','&gpuResult[0]']:
            source=self.sources[case].read_bytes().decode().replace('calc_path(gpuWall, gpuResult',f'calc_path(nullptr, {argument}')
            with self.subTest(argument=argument),self.assertRaisesRegex(UnsupportedSource,'cannot receive device pointers'):
                self.analyze(case,config=config,text=source)

    def test_unknown_conditional_cannot_hide_memory_access(self):
        source=self.sources['pathfinder'].read_bytes().decode().replace('MIN(left, up)','MIN(left, gpuSrc[xidx])')
        with self.assertRaisesRegex(UnsupportedSource,'pure scalar expressions'):
            self.analyze('pathfinder',text=source)

    def test_global_bounds_fail(self):
        source=self.sources['pathfinder'].read_bytes().decode().replace('gpuWall[index]','gpuWall[index+rows*cols]')
        with self.assertRaisesRegex(UnsupportedSource,'exceeds allocation'):
            self.analyze('pathfinder',text=source)

    def test_pointer_array_bounds_fail(self):
        source=self.sources['hotspot2d'].read_bytes().decode().replace('cudaFree(MatrixTemp[1])','cudaFree(MatrixTemp[2])')
        with self.assertRaisesRegex(UnsupportedSource,'Pointer array index out of bounds'):
            self.analyze('hotspot2d',text=source)

    def test_stale_input_summary_fails(self):
        source=self.sources['pathfinder'].read_bytes().decode().replace('int seed = M_SEED','int seed = 42')
        with self.assertRaisesRegex(UnsupportedSource,'Host summary source changed'):
            self.analyze('pathfinder',text=source)

    def test_host_copy_offsets_are_checked(self):
        source=self.sources['pathfinder'].read_bytes().decode().replace('gpuWall, data+cols','gpuWall, data+cols+1')
        with self.assertRaisesRegex(UnsupportedSource,'Host copy exceeds'):
            self.analyze('pathfinder',text=source)

    def test_invalid_tiling_and_budgets_fail(self):
        for case,params in [('hotspot2d',{'pyramid':8}),('pathfinder',{'pyramid':128})]:
            with self.subTest(case=case),self.assertRaises(UnsupportedSource):self.analyze(case,params)
        config=copy.deepcopy(self.configs['hotspot2d']);config['max_enumerated_threads']=1
        with self.assertRaisesRegex(UnsupportedSource,'max_enumerated_threads'):self.analyze('hotspot2d',config=config)

if __name__=='__main__':unittest.main()
