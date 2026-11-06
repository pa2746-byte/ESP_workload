"""Independent region schedules for complete Backprop, SRAD-v1, and CFD paths.

These test-only formulas do not participate in graph generation. CPU dependency
contracts are checked explicitly, in addition to GPU region hazards.
"""
import copy
import functools
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from cuda_configured_model import analyze_configured
from cuda_source_model import UnsupportedSource

ROOT = Path(__file__).resolve().parents[1]
STEMS = {'backprop': 'backprop_cuda', 'srad': 'main', 'cfd': 'euler3d'}


def config(case):
    return json.loads((ROOT / 'workloads/rodinia' / case / 'analysis.json').read_text())


def analyze(case, configuration=None, source=None):
    source = source or ROOT / 'workloads/rodinia' / case / (STEMS[case] + '.cu')
    return analyze_configured(source, configuration or config(case))


@functools.lru_cache(None)
def default(case):
    return analyze(case)


def event(buffer, mode, cells, stage, launch=None, parents=(), source=None):
    cells = set(cells)
    accesses = [(buffer, mode, cells)]
    if source is not None: accesses.append((source, 'read', cells))
    return dict(buffer=buffer, mode=mode, cells=cells, stage=stage, launch=launch,
                parents=set(parents), accesses=accesses, source=source)


def graph_contract(events):
    """Historical intersections, with intervening overwrites removed."""
    edges = set()
    for target, current in enumerate(events):
        edges.update((p, target) for p in current['parents'])
        for source, previous in enumerate(events[:target]):
            if current['launch'] is not None and current['launch'] == previous['launch'] and current['mode'] == 'write' and previous['mode'] == 'read':
                edges.add((source, target))
            for buf, mode, cells in current['accesses']:
                for pb, pm, pc in previous['accesses']:
                    if pb != buf or mode == pm == 'read': continue
                    live = cells & pc
                    for between in events[source+1:target]:
                        for ib, im, ic in between['accesses']:
                            if ib == buf and im == 'write': live -= ic
                    if live: edges.add((source, target))
    nodes = []
    for i, e in enumerate(events):
        if e['stage'] in {'upload','upload_constant'}: endpoints = ('cpu','mem')
        elif e['stage'] == 'download': endpoints = ('mem','cpu')
        elif e['stage'] == 'device_copy': endpoints = ('mem','mem')
        else: endpoints = ('mem','acc') if e['mode'] == 'read' else ('acc','mem')
        nodes.append(dict(id=i, vol=4*len(e['cells']), src_comp=endpoints[0], dest_comp=endpoints[1]))
    return dict(directed=True, multigraph=False, graph={}, nodes=nodes,
                edges=[dict(source=s,target=t) for s,t in sorted(edges)])


def backprop_events(n):
    full = set(range((n+1)*17))
    core = {r*17+c for r in range(1,n+1) for c in range(1,17)}
    updated = core | set(range(1,17))
    e = [event('input_cuda','write',range(n+1),'upload'),
         event('input_hidden_cuda','write',full,'upload')]
    first = 'bpnn_layerforward_CUDA'
    for b,m,c in [('input_cuda','read',range(1,n+1)),('input_hidden_cuda','read',core),
                  ('input_hidden_cuda','write',core),('hidden_partial_sum','write',range(n))]:
        e.append(event(b,m,c,first+':'+m,0))
    e.append(event('hidden_partial_sum','read',range(n),'download'))
    e += [event('hidden_delta_cuda','write',range(17),'upload',parents=[6]),
          event('input_prev_weights_cuda','write',full,'upload'),
          event('input_hidden_cuda','write',full,'upload')]
    second = 'bpnn_adjust_weights_cuda'
    for b,m,c in [('input_cuda','read',range(1,n+1)),('input_hidden_cuda','read',updated),
                  ('hidden_delta_cuda','read',range(1,17)),('input_prev_weights_cuda','read',updated),
                  ('input_hidden_cuda','write',updated),('input_prev_weights_cuda','write',updated)]:
        e.append(event(b,m,c,second+':'+m,1))
    e += [event('input_cuda','read',range(n+1),'download'),
          event('input_hidden_cuda','read',full,'download')]
    return e


def srad_events(rows,cols,iterations):
    count = rows*cols
    all_cells = set(range(count))
    e = [event(b,'write',range(n),'upload') for b,n in [('d_iN',rows),('d_iS',rows),('d_jE',cols),('d_jW',cols),('d_I',count)]]
    launch = 0
    def kernel(name, reads, writes, parents=()):
        nonlocal launch
        for b,c in reads: e.append(event(b,'read',c,name+':read',launch))
        for b,c in writes: e.append(event(b,'write',c,name+':write',launch,parents))
        launch += 1
    kernel('extract',[('d_I',all_cells)],[('d_I',all_cells)])
    for _ in range(iterations):
        kernel('prepare',[('d_I',all_cells)],[('d_sums',all_cells),('d_sums2',all_cells)])
        remaining, stride = count, 1
        while True:
            blocks = (remaining+511)//512
            reads = set(range(0,remaining*stride,stride))
            writes = {b*512*stride for b in range(blocks)}
            kernel('reduce',[(b,reads) for b in ('d_sums','d_sums2')],[(b,writes) for b in ('d_sums','d_sums2')])
            if blocks == 1: break
            remaining, stride = blocks, stride*512
        parents = [len(e),len(e)+1]
        e += [event(b,'read',[0],'download') for b in ('d_sums','d_sums2')]
        kernel('srad',[('d_I',all_cells),('d_iN',range(rows)),('d_iS',range(rows)),('d_jE',range(cols)),('d_jW',range(cols))],
               [(b,all_cells) for b in ('d_dN','d_dS','d_dW','d_dE','d_c')], parents)
        kernel('srad2',[('d_I',all_cells),('d_iS',range(rows)),('d_jE',range(cols))]+[(b,all_cells) for b in ('d_dN','d_dS','d_dW','d_dE','d_c')], [('d_I',all_cells)])
    kernel('compress',[('d_I',all_cells)],[('d_I',all_cells)])
    e.append(event('d_I','read',all_cells,'download'))
    return e


def cfd_events(iterations):
    # Four-cell fixture: each cell has interior neighbors plus wing/far-field faces.
    constants = ['ff_variable','ff_flux_contribution_momentum_x','ff_flux_contribution_momentum_y','ff_flux_contribution_momentum_z','ff_flux_contribution_density_energy']
    areas, neighbors, normals, variables, old, flux, step = [f'alloc<{t}>:t:{i}' for i,t in enumerate(['float','int','float','float','float','float','float'],5)]
    e = [event(b,'write',range(5 if i==0 else 3),'upload_constant') for i,b in enumerate(constants)]
    e += [event(b,'write',range(n),'upload') for b,n in [(areas,4),(neighbors,16),(normals,48)]]
    launch = 0
    def kernel(name, reads, writes):
        nonlocal launch
        for b,c in reads: e.append(event(b,'read',c,name+':read',launch))
        for b,c in writes: e.append(event(b,'write',c,name+':write',launch))
        launch += 1
    for b in (variables,old,flux):
        kernel('cuda_initialize_variables',[(constants[0],range(5))],[(b,range(20))])
    e.append(event(step,'write',range(4),'memset'))
    for _ in range(iterations):
        e.append(event(old,'write',range(20),'device_copy',source=variables))
        kernel('cuda_compute_step_factor',[(areas,range(4)),(variables,range(20))],[(step,range(4))])
        for rk in range(3):
            kernel('cuda_compute_flux',[(constants[0],range(1,4))]+[(b,range(3)) for b in constants[1:]]+[(neighbors,range(16)),(normals,range(48)),(variables,range(20))],[(flux,range(20))])
            kernel('cuda_time_step',[(old,range(20)),(flux,range(20)),(step,range(4))],[(variables,range(20))])
    e.append(event(variables,'read',range(20),'download'))
    return e


class CompleteRodiniaTests(unittest.TestCase):
    def assert_contract(self, result, events):
        graph, metadata, _ = result
        self.assertEqual(graph, graph_contract(events))
        self.assertEqual(len(metadata),len(events))
        for meta,e in zip(metadata,events):
            self.assertEqual((meta['buffer'],meta['stage']), (e['buffer'],e['stage']))
            offsets = {x for a,b in meta['byte_ranges'] for x in range(a,b)}
            self.assertEqual(offsets,{4*c+b for c in e['cells'] for b in range(4)})
            if e['source']: self.assertEqual(meta['source_buffer'],e['source'])
            if e['parents']: self.assertEqual(meta['host_input_dependencies'],sorted(e['parents']))

    def test_backprop_complete_flow_and_cpu_dependency(self):
        self.assert_contract(default('backprop'),backprop_events(32))
        c=config('backprop');c['parameters']['inputs']=16
        self.assert_contract(analyze('backprop',c),backprop_events(16))
        self.assertEqual(len(default('backprop')[2]['launches']),2)

    def test_srad_complete_flow_and_reductions(self):
        self.assert_contract(default('srad'),srad_events(4,4,2))
        for rows,cols in [(4,8),(32,32)]:
            c=config('srad');c['parameters'].update(rows=rows,cols=cols,iterations=1)
            self.assert_contract(analyze('srad',c),srad_events(rows,cols,1))
        self.assertEqual(len(default('srad')[2]['launches']),10)

    def test_cfd_complete_flow_and_device_copies(self):
        self.assert_contract(default('cfd'),cfd_events(2))
        c=config('cfd');c['compile_defines']['iterations']=1
        self.assert_contract(analyze('cfd',c),cfd_events(1))
        self.assertEqual(len(default('cfd')[2]['launches']),17)

    def test_published_artifacts_match_analysis(self):
        for case, stem in STEMS.items():
            graph, nodes, details = default(case)
            folder = ROOT / 'results' / ('rodinia_' + case)
            self.assertEqual(json.loads((folder / (stem + '_simulator.json')).read_text()), graph)
            metadata = json.loads((folder / (stem + '_metadata.json')).read_text())
            self.assertEqual(metadata['nodes'], nodes)
            self.assertEqual(metadata['configured_analysis'], details)

    def test_integer_indices_are_derived_from_host_source(self):
        nodes=default('srad')[1]
        self.assertEqual(nodes[0]['known_integer_values'], [[0,0],[4,0],[8,1],[12,2]])
        self.assertEqual(nodes[1]['known_integer_values'], [[0,1],[4,2],[8,3],[12,3]])
        c=config('srad');c['known_integer_inputs']=False
        with self.assertRaisesRegex(UnsupportedSource,'Data-dependent array index'):
            analyze('srad',c)

    def copy_source(self,case):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/case
        shutil.copytree(ROOT/'workloads/rodinia'/case,path)
        return path

    def mutate(self,case,file,old,new):
        folder=self.copy_source(case);p=folder/file;raw=p.read_bytes()
        self.assertIn(old.encode(),raw)
        p.write_bytes(raw.replace(old.encode(),new.encode(),1))
        return folder/(STEMS[case]+'.cu')

    def test_changed_cpu_contract_is_rejected(self):
        source=self.mutate('backprop','backprop.c','l2[j] = squash(sum);','l2[j] = squash(sum + 1);')
        with self.assertRaisesRegex(UnsupportedSource,'Host summary source changed'):
            analyze('backprop',source=source)

    def test_srad_missing_barrier_is_rejected(self):
        folder=self.copy_source('srad')
        kernel=folder/'reduce_kernel.cu'
        kernel.write_bytes(kernel.read_bytes().replace(b'__syncthreads();',b'/* removed barrier */'))
        source=folder/'main.cu'
        with self.assertRaisesRegex(UnsupportedSource,'lacks a block barrier'):
            analyze('srad',source=source)

    def test_srad_nonuniform_barrier_is_rejected(self):
        source=self.mutate('srad','reduce_kernel.cu','__syncthreads();','if (tx == 0) __syncthreads();')
        with self.assertRaisesRegex(UnsupportedSource,'Nonuniform block barrier'):
            analyze('srad',source=source)

    def test_unknown_scalar_branch_cannot_hide_global_access(self):
        source=self.mutate('srad','srad_kernel.cu','d_c_loc = 0;','d_c_loc = d_I[0];')
        with self.assertRaisesRegex(UnsupportedSource,'Unknown branch must only update scalar locals'):
            analyze('srad',source=source)

    def test_cfd_input_hash_and_neighbor_bounds(self):
        folder=self.copy_source('cfd');source=folder/'euler3d.cu';mesh=folder/'mesh4.dat'
        mesh.write_text(mesh.read_text().replace('2 1 0 0','99 1 0 0',1))
        with self.assertRaisesRegex(UnsupportedSource,'input file changed'):
            analyze('cfd',source=source)
        c=config('cfd');c['input_files']['mesh4.dat']=hashlib.sha256(mesh.read_bytes()).hexdigest()
        with self.assertRaisesRegex(UnsupportedSource,'exceeds allocation'):
            analyze('cfd',c,source)

    def test_bad_backprop_geometry_is_rejected(self):
        c=config('backprop');c['parameters']['inputs']=17
        with self.assertRaises(UnsupportedSource): analyze('backprop',c)


if __name__=='__main__':unittest.main()
