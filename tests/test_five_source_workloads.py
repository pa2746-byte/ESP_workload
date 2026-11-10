"""Focused independent checks of the five fixed-input original application paths."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from cuda_configured_model import analyze_configured
from cuda_source_model import UnsupportedSource

ROOT=Path(__file__).resolve().parents[1]
CASES={'bfs':'bfs.cu','lavamd':'main.c','particlefilter':'ex_particle_CUDA_naive_seq.cu','kmeans':'kmeans_cuda.cu','btree':'main.c'}

def schedules(case):
    events=[]
    def event(buffer,stage,size,start=0):events.append((buffer,stage,set(range(start,start+size))))
    if case=='bfs':
        for b,s in [('d_graph_nodes',32),('d_graph_edges',12),('d_graph_mask',4),('d_updating_graph_mask',4),('d_graph_visited',4),('d_cost',16)]:event(b,'upload',s)
        for frontier in range(4):
            event('d_over','upload',1)
            event('d_graph_nodes','Kernel:read',8,8*frontier)
            if frontier<3:event('d_graph_edges','Kernel:read',4,4*frontier)
            event('d_graph_mask','Kernel:read',4)
            if frontier<3:
                event('d_graph_visited','Kernel:read',1,frontier+1);event('d_cost','Kernel:read',4,4*frontier)
            event('d_graph_mask','Kernel:write',1,frontier)
            if frontier<3:
                event('d_updating_graph_mask','Kernel:write',1,frontier+1);event('d_cost','Kernel:write',4,4*(frontier+1))
            event('d_updating_graph_mask','Kernel2:read',4)
            if frontier<3:
                for b in ['d_graph_mask','d_updating_graph_mask','d_graph_visited']:event(b,'Kernel2:write',1,frontier+1)
                event('d_over','Kernel2:write',1)
            event('d_over','download',1)
        event('d_cost','download',16)
    elif case=='lavamd':
        # LP64 box: four ints + long + int + padding + 26 neighbors of 24 bytes.
        for b,s in [('d_box_gpu',656),('d_rv_gpu',100*32),('d_qv_gpu',100*8),('d_fv_gpu',100*32)]:event(b,'upload',s)
        event('d_box_gpu','kernel_gpu_cuda:read',12,16)
        for b,s in [('d_rv_gpu',3200),('d_qv_gpu',800),('d_fv_gpu',3200)]:event(b,'kernel_gpu_cuda:read',s)
        event('d_fv_gpu','kernel_gpu_cuda:write',3200);event('d_fv_gpu','download',3200)
    elif case=='particlefilter':
        for b in ['arrayX_GPU','arrayY_GPU','xj_GPU','yj_GPU','CDF_GPU','u_GPU']:event(b,'upload',8)
        for b in ['arrayX_GPU','arrayY_GPU','CDF_GPU','u_GPU']:event(b,'kernel:read',8)
        for b in ['xj_GPU','yj_GPU']:event(b,'kernel:write',8)
        for b in ['yj_GPU','xj_GPU']:event(b,'download',8)
    elif case=='kmeans':
        event('feature_flipped_d','upload',4);event('feature_flipped_d','invert_mapping:read',4);event('feature_d','invert_mapping:write',4)
        # First pass changes membership -1 -> 0; second pass establishes convergence.
        for _ in range(2):
            for b in ['membership_d','clusters_d']:event(b,'upload',4)
            event('c_clusters','upload_constant',4)
            for b in ['c_clusters','feature_d']:event(b,'kmeansPoint:read',4)
            event('membership_d','kmeansPoint:write',4);event('membership_d','download',4)
    elif case=='btree':
        for b,s in [('recordsD',4),('knodesD',52),('currKnodeD',8),('offsetD',8),('keysD',4),('ansD',4)]:event(b,'upload',s)
        event('recordsD','findK:read',4)
        events.append(('knodesD','findK:read',set(range(8,12))|set(range(24,40))))
        event('currKnodeD','findK:read',8);event('keysD','findK:read',4)
        event('ansD','findK:write',4);event('ansD','download',4)
    return events


def expected_graph(events):
    writers={};readers={};nodes=[];edges=set();barriers=set();kernel_reads=[];kernel=None
    for i,(buffer,stage,offsets) in enumerate(events):
        writing=stage in {'upload','upload_constant'} or stage.endswith(':write')
        if ':' in stage:
            name=stage.split(':')[0]
            if name!=kernel:kernel_reads=[];kernel=name
        else:kernel_reads=[];kernel=None
        parents=set(barriers)
        for b in offsets:
            key=buffer,b
            if key in writers:parents.add(writers[key])
            if writing:parents.update(readers.get(key,()))
        if stage.endswith(':write'):parents.update(kernel_reads)
        if stage.endswith(':read'):kernel_reads.append(i)
        if writing:
            for b in offsets:writers[buffer,b]=i;readers[buffer,b]=set()
        else:
            for b in offsets:readers.setdefault((buffer,b),set()).add(i)
        src,dst=('cpu','mem') if stage.startswith('upload') else ('mem','cpu') if stage=='download' else ('acc','mem') if writing else ('mem','acc')
        nodes.append(dict(id=i,vol=len(offsets),src_comp=src,dest_comp=dst));edges.update((p,i) for p in parents)
        if stage=='download':barriers.add(i)
    return dict(directed=True,multigraph=False,graph={},nodes=nodes,edges=[dict(source=a,target=b) for a,b in sorted(edges)])


class FiveWorkloads(unittest.TestCase):
    def test_five_source_and_saved_graph_contracts(self):
        for case,source in CASES.items():
            with self.subTest(case=case):
                folder=ROOT/'workloads/rodinia'/case;c=json.loads((folder/'analysis.json').read_text())
                graph,metadata,details=analyze_configured(folder/source,c)
                events=schedules(case);self.assertEqual(graph,expected_graph(events))
                for m,(buffer,stage,offsets) in zip(metadata,events):
                    self.assertEqual((m['buffer'],m['stage']),(buffer,stage))
                    self.assertEqual({i for a,b in m['byte_ranges'] for i in range(a,b)},offsets)
                result=ROOT/'results'/('rodinia_'+case)/Path(source).stem
                self.assertEqual(json.loads(result.with_name(result.name+'_simulator.json').read_text()),graph)
                saved=json.loads(result.with_name(result.name+'_metadata.json').read_text())
                self.assertEqual(saved['nodes'],metadata);self.assertEqual(saved['configured_analysis'],details)
                self.assertEqual(len(details['launches']),{'bfs':8,'lavamd':1,'particlefilter':1,'kmeans':3,'btree':1}[case])

    def test_btree_preparation_against_original_cpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=ROOT/'workloads/rodinia/btree'
            subprocess.run(['clang','-std=gnu99','-Wno-error=implicit-function-declaration','-DRD_WG_SIZE=4','-I',str(source),str(source/'main.c'),str(source/'util/timer/timer.c'),str(ROOT/'tests/fixtures/btree_host_capture.c'),'-lm','-o',str(tmp/'capture')],check=True,capture_output=True)
            subprocess.run([str(tmp/'capture'),'file',str(source/'one_key.txt'),'command',str(source/'commands.txt')],input='\n',text=True,cwd=tmp,check=True,capture_output=True)
            actual=(tmp/'captured_records.bin').read_bytes()+(tmp/'captured_knodes.bin').read_bytes()
            spec=json.loads((source/'analysis.json').read_text())['host_summaries']['transform_to_cuda']['global_byte_buffer']
            expected=bytes.fromhex(spec['hex']);self.assertEqual(len(actual),len(expected))
            for a,b in spec['known_ranges']:self.assertEqual(actual[a:b],expected[a:b])

    def test_unknown_paths_and_changed_inputs_fail_closed(self):
        folder=ROOT/'workloads/rodinia/particlefilter';c=json.loads((folder/'analysis.json').read_text());c['arguments'][-1]='2'
        with self.assertRaises(UnsupportedSource):analyze_configured(folder/CASES['particlefilter'],c)
        folder=ROOT/'workloads/rodinia/btree';c=json.loads((folder/'analysis.json').read_text());c['source_sha256']='0'*64
        with self.assertRaisesRegex(UnsupportedSource,'source changed'):analyze_configured(folder/'main.c',c)
        folder=ROOT/'workloads/rodinia/bfs';c=json.loads((folder/'analysis.json').read_text());c['input_files']['chain4.txt']='0'*64
        with self.assertRaisesRegex(UnsupportedSource,'input file changed'):analyze_configured(folder/'bfs.cu',c)

    def test_capture_preparation_csv_and_notebook(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from capture_rodinia_cycles import prepare,parse_cycles
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for case in CASES:
                manifest=prepare(case,tmp,'75')
                self.assertTrue(manifest['build_commands'])
                for command in manifest['build_commands']:
                    self.assertNotIn('source_analysis', ' '.join(command))
            csvfile=tmp/'counter.csv'
            csvfile.write_text('==PROF== connected\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","kernel","sm__cycles_elapsed.avg","cycle","1,234"\n"0","kernel","gpu__time_duration.sum","nsecond","999"\n')
            rows=parse_cycles(csvfile,'test');self.assertEqual(len(rows),1);self.assertEqual(rows[0]['value'],1234)
            csvfile.write_text('ERR_NVGPUCTRPERM: access denied\n');self.assertEqual(parse_cycles(csvfile,'test'),[])
        notebook=json.loads((ROOT/'notebooks/rodinia_five_cycles.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')

if __name__=='__main__':unittest.main()
