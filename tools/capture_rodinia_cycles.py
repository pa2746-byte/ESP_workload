#!/usr/bin/env python3
"""Capture five matched Rodinia scenarios; hardware cycles are never inferred from time."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'bfs': ('bfs.cu', ['chain4.txt']),
    'lavamd': ('main.c', []),
    'particlefilter': ('ex_particle_CUDA_naive_seq.cu', ['-x','16','-y','16','-z','2','-np','1']),
    'kmeans': ('kmeans_cuda.cu', ['-b','-i','point1.bin','-m','1','-n','1']),
    'btree': ('main.c', ['file','one_key.txt','command','commands.txt']),
}
METRICS = ['sm__cycles_elapsed.avg', 'sm__cycles_active.avg']


def locate(name):
    if shutil.which(name): return shutil.which(name)
    patterns = ['/opt/nvidia/nsight-*/**/' + name,
                '/usr/local/cuda*/**/' + name]
    import glob
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern, recursive=True)
                  if Path(p).is_file()]
    return str(sorted(candidates)[-1]) if candidates else None


def replace_once(path, old, new, changes):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'Compatibility patch no longer matches {path.name}: {old[:50]}')
    path.write_text(text.replace(old, new))
    changes.append({'file': str(path.name), 'old': old, 'new': new})


def prepare(case, out, arch):
    source = out / 'sources' / case
    if source.exists(): raise ValueError('Use a new capture directory; sources already staged')
    shutil.copytree(ROOT / 'workloads/rodinia' / case, source)
    changes = []
    # Only staged build copies are changed; originals and patch records are retained.
    for path in source.rglob('*.cu'):
        text = path.read_text()
        if 'cudaThreadSynchronize' in text:
            path.write_text(text.replace('cudaThreadSynchronize', 'cudaDeviceSynchronize'))
            changes.append({'file':str(path.relative_to(source)), 'change':'Legacy synchronize API renamed'})
    if case == 'kmeans':
        wrapper = source / 'kmeans_cuda.cu'
        text = wrapper.read_text()
        if text.count('cudaSetDevice(1)') != 2: raise ValueError('Device remap patch changed')
        wrapper.write_text(text.replace('cudaSetDevice(1)', 'cudaSetDevice(0)'))
        changes.append({'file':wrapper.name,'change':'Remap logical device 1 to Colab device 0'})
        # CUDA 12+ removed texture references. Preserve the default kernel's logical reads
        # with ordinary loads; the hardware cache path can differ from the legacy version.
        text = wrapper.read_text()
        start = text.index('    cudaChannelFormatDesc chDesc0')
        end = text.index('/* copy clusters to constant memory */', start)
        wrapper.write_text(text[:start] + text[end:])
        replace_once(wrapper, 'cudaMemcpyToSymbol("c_clusters",', 'cudaMemcpyToSymbol(c_clusters,', changes)
        kernel = source / 'kmeans_cuda_kernel.cu'
        text = kernel.read_text()
        declarations = re.findall(r'^texture<float, 1, cudaReadModeElementType>.*?;\s*$', text, re.M)
        if len(declarations) != 3: raise ValueError('Texture declarations changed')
        text = re.sub(r'^texture<float, 1, cudaReadModeElementType>.*?;\s*$', '', text, flags=re.M)
        if text.count('tex1Dfetch(t_features,addr)') != 1: raise ValueError('Texture load changed')
        kernel.write_text(text.replace('tex1Dfetch(t_features,addr)', 'features[addr]'))
        changes.append({'file':kernel.name,'change':'Default CPU-reduction path only: legacy texture fetch replaced by global load; unused texture bindings removed'})
    if case == 'btree':
        (source/'configured_rand.c').write_text('/* Matches the documented random_values=[0] scenario. */\nint rand(void) { return 0; }\n')
        changes.append({'file':'configured_rand.c','change':'Supply the explicit zero-valued query random input'})
    # No analysis-only headers are ever included in GPU builds.
    (source/'runtime_compat.h').write_text('#ifdef __cplusplus\n#include <cuda_runtime.h>\n#include <cmath>\n#include <cstring>\n#include <cstdlib>\n#include <ctime>\n#else\n#include <string.h>\n#include <time.h>\nint cudaSetDevice(int);\n#endif\n')
    build = out/'build'/case
    build.mkdir(parents=True, exist_ok=True)
    defines = ['-DRD_WG_SIZE=4'] if case == 'btree' else []
    common = ['-O3','-include',str(source/'runtime_compat.h'), *defines]
    cuda = ['nvcc', *common, '-lineinfo','-std=c++14',f'-arch=sm_{arch}']
    commands = []
    if case in {'bfs','particlefilter'}:
        commands.append([*cuda,str(source/CASES[case][0]),'-o',str(build/case)])
    else:
        c_files = {
            'lavamd':['main.c','util/timer/timer.c','util/num/num.c'],
            'kmeans':['kmeans.c','cluster.c','kmeans_clustering.c','rmse.c'],
            'btree':['main.c','util/timer/timer.c','util/num/num.c','configured_rand.c'],
        }[case]
        cu_files = {
            'lavamd':['kernel/kernel_gpu_cuda_wrapper.cu','util/device/device.cu'],
            'kmeans':['kmeans_cuda.cu'],
            'btree':['kernel/kernel_gpu_cuda_wrapper.cu','kernel/kernel_gpu_cuda_wrapper_2.cu','util/cuda/cuda.cu'],
        }[case]
        objects=[]
        for index,file in enumerate(c_files):
            obj=build/f'host_{index}.o';objects.append(str(obj))
            commands.append(['gcc',*common,'-std=gnu99','-Wno-implicit-function-declaration','-fopenmp','-c',str(source/file),'-o',str(obj)])
        for index,file in enumerate(cu_files):
            obj=build/f'device_{index}.o';objects.append(str(obj))
            commands.append([*cuda,'-c',str(source/file),'-o',str(obj)])
        commands.append(['nvcc',*objects,'-Xcompiler','-fopenmp','-lm','-o',str(build/case)])
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}
    manifest={'workload':case,'entry':CASES[case][0],'arguments':CASES[case][1],
              'compatibility_changes':changes,'staged_sha256':hashes,'build_commands':commands,
              'executable':str(build/case),'cwd':str(source),'stdin':'\n' if case=='btree' else ''}
    (source/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    expected=out/'clang_graphs'/case
    shutil.copytree(ROOT/'results'/('rodinia_'+case), expected)
    return manifest


def parse_cycles(path, case):
    rows=list(csv.reader(path.read_text(errors='replace').splitlines()))
    header=next((i for i,r in enumerate(rows) if 'Metric Name' in r and 'Metric Value' in r),None)
    if header is None:return []
    names=rows[header];result=[]
    for row in rows[header+1:]:
        if len(row)!=len(names):continue
        d=dict(zip(names,row));metric=d.get('Metric Name')
        if metric not in METRICS:continue
        try:value=float(d['Metric Value'].replace(',',''))
        except ValueError:continue
        if not __import__('math').isfinite(value) or value<0:continue
        # Base-unit collection is requested; never accept a duration as cycles.
        if d.get('Metric Unit') not in {'cycle','cycles'}:continue
        result.append(dict(workload=case,launch_id=d.get('ID',''),kernel=d.get('Kernel Name',''),
                           metric=metric,unit='cycle',value=value))
    return result


def capture(out, cases, prepare_only=False):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    status={};all_cycles=[];counter_access_denied=False
    def run(command,label,cwd=None,stdin='',timeout=300):
        command=list(map(str,command))
        (logs/(label+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        try:
            p=subprocess.run(command,cwd=cwd,input=stdin,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
            code,text=p.returncode,p.stdout
        except (OSError,subprocess.TimeoutExpired) as e:code,text=-1,str(e)
        (logs/(label+'.log')).write_text(text)
        print(label, 'OK' if code==0 else f'FAILED ({code})', flush=True)
        if code:print(text[-1200:],flush=True)
        return code,text
    arch='75'
    if not prepare_only:
        code,_=run(['nvidia-smi'],'gpu')
        if code or not shutil.which('nvcc'):raise RuntimeError('Select a Colab GPU runtime with nvcc, then rerun in a new output directory.')
        run(['nvcc','--version'],'nvcc')
        code,text=run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],'compute_cap')
        arch=text.splitlines()[0].strip().replace('.','') if code==0 else ''
        if not arch.isdigit():raise RuntimeError('Cannot determine GPU compute capability')
    nsys,ncu=locate('nsys'),locate('ncu')
    if not prepare_only:
        for name,path in [('nsys',nsys),('ncu',ncu)]:
            if path:run([path,'--version'],name+'_version')
        run(['nvidia-smi','-q'],'gpu_details')
    for case in cases:
        info=status[case]={'cycles_measured':False,'trace_verified':False}
        try:
            manifest=prepare(case,out,arch)
            if prepare_only:info['prepared']=True;continue
            for i,cmd in enumerate(manifest['build_commands']):
                code,_=run(cmd,f'{case}_build_{i}')
                if code:raise RuntimeError('Build failed; see logs')
            stem=Path(CASES[case][0]).stem
            expected=json.loads((ROOT/'results'/('rodinia_'+case)/(stem+'_metadata.json')).read_text())
            expected_count=len(expected['configured_analysis']['launches']);info['expected_kernel_count']=expected_count
            exe=[manifest['executable'],*manifest['arguments']]
            cwd=manifest['cwd'];stdin=manifest['stdin']
            prefix=out/case
            if nsys:
                code,_=run([nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',prefix,*exe],case+'_nsys',cwd,stdin)
                info['nsys_returncode']=code
                report=prefix.with_suffix('.nsys-rep')
                if report.exists():
                    db=prefix.with_suffix('.sqlite')
                    run([nsys,'export','--type=sqlite','--force-overwrite=true','--output',db,report],case+'_export')
                    if db.exists():
                        with sqlite3.connect(db) as conn:
                            tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                            kernels=conn.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0] if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else 0
                        info['trace_kernel_count']=kernels;info['trace_verified']=code==0 and kernels==expected_count
            else:
                code,_=run(exe,case+'_plain',cwd,stdin);info['plain_returncode']=code
                info['trace_error']='Nsight Systems unavailable'
            if ncu and not counter_access_denied:
                csv_path=out/(case+'_ncu.csv')
                code,text=run([ncu,'--metrics',','.join(METRICS),'--clock-control','none','--cache-control','none',
                    '--replay-mode','kernel','--csv','--page','details','--print-metric-name','name','--print-units','base',
                    '--log-file',csv_path,'--export',out/(case+'_cycles'),'--force-overwrite',*exe],case+'_ncu',cwd,stdin)
                info['ncu_returncode']=code
                rows=parse_cycles(csv_path,case) if csv_path.exists() else []
                raw=text+(csv_path.read_text(errors='replace') if csv_path.exists() else '')
                if 'ERR_NVGPUCTRPERM' in raw:
                    counter_access_denied=True
                    info['cycle_error']='GPU hardware-counter access denied by host; no measured cycles available'
                elif code:info['cycle_error']='Nsight Compute failed; inspect logs'
                elapsed=[r for r in rows if r['metric']==METRICS[0]]
                info['cycles_measured']=code==0 and not counter_access_denied and len(elapsed)==expected_count
                if info['cycles_measured']:
                    all_cycles.extend(rows)
                    info['profiled_kernel_count']=len(elapsed)
                    info['sum_kernel_elapsed_sm_avg_cycles']=sum(r['value'] for r in elapsed)
                else:info.setdefault('cycle_error','No elapsed-cycle rows; metrics may be unsupported')
            else:info['cycle_error']='Hardware counters denied earlier in this session; skipped repeated attempts' if counter_access_denied else 'Nsight Compute unavailable'
        except Exception as e:info['error']=str(e)
        finally:(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (out/'cycles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['workload','launch_id','kernel','metric','unit','value']);w.writeheader();w.writerows(all_cycles)
    (out/'MEASUREMENT_SCOPE.txt').write_text('Cycles are measured per kernel by Nsight Compute, averaged across SM instances. The sum across launches is kernel-only, not end-to-end application cycles. It excludes CPU work and host/device copies. No duration-to-cycle conversion is performed. Nsight Systems timestamps are retained separately for transfer/order DAG comparison. Profiler replay and instrumentation can affect execution. See build manifests for source adaptations.\n')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--workloads',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    print(json.dumps(capture(args.out_dir,args.workloads,args.prepare_only),indent=2))
