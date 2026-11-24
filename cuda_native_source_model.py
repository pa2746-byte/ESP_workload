"""Clang-instrumented bounded CPU source execution for configured CUDA kernels.

Experimental backend. GPU captures never determine accesses or control flow.
It executes the complete original host program and instrumented device bodies
under documented CPU arithmetic; this is not CUDA numerical equivalence.
"""
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
from cuda_source_model import UnsupportedSource, children, walk

ROOT = Path(__file__).resolve().parent
PREFIX = '''template<class T> T& esp_source_access(T&,int);
template<class T> T esp_source_argument(T);
template<class T> T* esp_source_argument(T*);
template<class T,class I> T& esp_source_index(T*,I);
template<class T> T* esp_source_pointer(T*);
void esp_source_shared(void*,__SIZE_TYPE__,const char*);
void* esp_source_dynamic_shared(const char*);
void esp_source_barrier(const char*);
template<class G,class B,class F,class S> void esp_source_launch(G,B,__SIZE_TYPE__,const char*,F,S);
'''


def checked_run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise UnsupportedSource('Native source stage failed:\n' + result.stderr[-16000:])
    return result


def instrument(unit, build, include_dirs=(), defines=None, clang='clang++', source_root=None, shared_memory='reject'):
    unit, build = Path(unit).resolve(), Path(build).resolve()
    source_root=Path(source_root).resolve() if source_root else unit.parent
    if shared_memory not in {'reject','reject_accesses','checked_blocks'}:raise UnsupportedSource('Unknown native shared-memory mode')
    build.mkdir(parents=True, exist_ok=True)
    stub = ROOT/'source_analysis/include'
    host=unit.suffix in {'.cpp','.cc','.cxx'}
    language=['-x','c++'] if host else ['-x','cuda','--cuda-host-only','-nocudainc','-nocudalib']
    flags = [*language,'-std=c++17',
             '-Wno-reserved-user-defined-literal','-I'+str(stub),'-I'+str(ROOT/'source_analysis/native/include'),'-include',str(stub/'cuda_runtime.h')]
    flags += ['-include',str(ROOT/'source_analysis/native/parse_compat.hpp')]
    flags += ['-I'+str(Path(d).resolve()) for d in include_dirs]
    for key,value in (defines or {}).items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key) or type(value) is not int:
            raise UnsupportedSource('Native compile definitions must be named integer constants')
        flags.append(f'-D{key}={value}')
    source = checked_run([clang,*flags,'-E','-dD',str(unit)]).stdout.encode()
    math_names='exp|sqrt|sqrtf|pow|fabs|abs|fmod|log|log10|logb|scalbn|sin|cos|atan|tanh|isnan|isinf'
    if host:
        current_file=''
        for line in source.decode().splitlines():
            marker=re.match(r'^#\s+\d+\s+"([^"\n]+)"',line)
            if marker:current_file=marker[1]
            elif current_file and Path(current_file).is_relative_to(source_root) and not line.lstrip().startswith('#'):
                if re.search(r'__attribute__\s*\(\(\s*(?:global|device|constant)\s*\)\)',line):
                    raise UnsupportedSource('CUDA definitions must be in CUDA translation units, not host-only C++ units')
        host_text=source.decode()
        host_text=re.sub(r'^\s*__attribute__\s*\(\(\s*device\s*\)\)\s+[^;{}\n]*\b(?:'+math_names+r')\s*\([^;{}\n]*;','',host_text,flags=re.MULTILINE)
        host_text=re.sub(r'__attribute__\s*\(\(\s*(?:global|device|host|constant)\s*\)\)', '', host_text)
        host_text=re.sub(r'template\s*<class T>\s*T\s+(?:min|max)\(T,T\);','',host_text)
        source=host_text.encode()
    source=source.replace(b'struct texture { int filterMode; bool normalized; cudaChannelFormatDesc channelDesc; }',b'struct texture { int filterMode; bool normalized; cudaChannelFormatDesc channelDesc; int addressMode[3]; }')
    expanded = build/(unit.name+'.expanded.cu')
    expanded.write_bytes(source)
    ast = json.loads(checked_run([clang,*language,
                                '-std=c++17','-Wno-reserved-user-defined-literal','-fsyntax-only',
                                '-Xclang','-ast-dump=json',str(expanded)]).stdout)
    import bisect
    file_offsets=[];file_names=[]
    for marker in re.finditer(rb'^#\s+\d+\s+"([^"\n]+)"',source,re.MULTILINE):
        file_offsets.append(marker.end());file_names.append(marker[1].decode())
    def declaration_file(fn):
        offset=fn.get('range',{}).get('begin',{}).get('offset',-1)
        index=bisect.bisect_right(file_offsets,offset)-1
        return Path(file_names[index]) if index>=0 else None
    def user_function(fn):
        file=declaration_file(fn)
        return file is not None and file.is_relative_to(source_root)
    insertions, replacements, selected, indexes, pointers = {}, {}, [], {}, {}
    def extent(node):
        r=node.get('range',{});a=r.get('begin',{});b=r.get('end',{})
        if 'offset' not in a or 'offset' not in b:
            raise UnsupportedSource('Instrumented device expression needs an expanded direct source range')
        return a['offset'],b['offset']+b['tokLen']
    def text(node):
        a,b=extent(node);return source[a:b].decode()
    def wrap(node,mode):
        a,b=extent(node)
        key=(a,b)
        if key in insertions: insertions[key] |= mode
        else: insertions[key]=mode
    implicit_assignments={n.get('id') for n in walk(ast) if n['kind']=='CXXMethodDecl' and n.get('isImplicit') and n.get('name')=='operator='}
    seen=set()
    for fn in walk(ast):
        if fn.get('id') in seen or fn['kind'] not in {'FunctionDecl','CXXMethodDecl'}:continue
        seen.add(fn.get('id'))
        body=next((c for c in children(fn) if c['kind']=='CompoundStmt'),None)
        if body is None:continue
        if fn.get('isImplicit') or not user_function(fn):continue
        for n in walk(body):
            if n['kind']=='DeclRefExpr' and n.get('referencedDecl',{}).get('name','').startswith(('pthread_barrier_','pthread_create')):
                raise UnsupportedSource('Native source backend supports serial host execution only')
        device=any(c['kind'] in {'CUDADeviceAttr','CUDAGlobalAttr'} for c in children(fn))
        if device:
            selected.append(fn.get('name'))
            for n in walk(body):
                if n['kind'] in {'CXXNewExpr','CXXDeleteExpr'}:raise UnsupportedSource('Native backend rejects device allocation')
                if n['kind']=='CUDASharedAttr' and shared_memory=='reject':raise UnsupportedSource('Native source backend rejects shared memory')
                if n['kind']=='CUDAKernelCallExpr':raise UnsupportedSource('Native source backend rejects device launches')
                if shared_memory=='reject' and n['kind']=='DeclRefExpr' and n.get('referencedDecl',{}).get('name') in {'__syncthreads','atomicAdd','atomicAnd','atomicOr','atomicInc'}:
                    raise UnsupportedSource('Native source backend rejects barriers and atomics')
                cs=children(n)
                if n['kind']=='DeclStmt' and any(any(c['kind']=='CUDASharedAttr' for c in children(v)) for v in cs):
                    variables=[v for v in cs if any(c['kind']=='CUDASharedAttr' for c in children(v))]
                    registrations=[];dynamic=[]
                    for var in variables:
                        typ=var['type'].get('desugaredQualType',var['type']['qualType'])
                        base=typ.split('[')[0].replace('volatile ','').replace('const ','').strip()
                        if base not in {'int','unsigned int','float','double','char','unsigned char','uint'}:
                            raise UnsupportedSource('Checked shared storage requires primitive numeric elements')
                        identity=json.dumps(unit.name+':'+str(extent(var)[0])+':'+var['name'])
                        if var.get('storageClass')=='extern':
                            if '[]' not in typ:raise UnsupportedSource('Dynamic shared storage requires an unsized array')
                            dynamic.append(f'auto *{var["name"]}=({base}*)esp_source_dynamic_shared({identity});')
                        else:registrations.append(f'esp_source_shared((void*)&{var["name"]},sizeof({var["name"]}),{identity});')
                    if dynamic and len(variables)!=len(cs):raise UnsupportedSource('Mixed dynamic shared declarations are unsupported')
                    replacements[extent(n)]=( ''.join(dynamic) if dynamic else text(n)+''.join(registrations)).encode()
                if n['kind']=='CallExpr' and cs:
                    callee=cs[0]
                    while callee['kind'] in {'ImplicitCastExpr','ParenExpr'}:callee=children(callee)[0]
                    if callee.get('referencedDecl',{}).get('name')=='__syncthreads' and shared_memory!='reject':
                        replacements[extent(n)]=('esp_source_barrier('+json.dumps(unit.name+':'+str(extent(n)[0]))+')').encode()
                    if callee.get('referencedDecl',{}).get('name') in {'memcpy','memmove','memset','__builtin_memcpy','__builtin_memmove','__builtin_memset'}:
                        raise UnsupportedSource('Native backend rejects uninstrumented device memory library calls')
                if n['kind']=='CXXOperatorCallExpr' and len(cs)>=3:
                    callee=cs[0]
                    while callee['kind'] in {'ImplicitCastExpr','ParenExpr'}:callee=children(callee)[0]
                    if callee.get('referencedDecl',{}).get('id') in implicit_assignments:
                        wrap(cs[1],2)
                        if cs[2].get('valueCategory')=='lvalue':wrap(cs[2],1)
                if n['kind']=='CXXConstructExpr' and len(cs)==1 and cs[0].get('valueCategory')=='lvalue':wrap(cs[0],1)
                if n['kind']=='ArraySubscriptExpr': indexes[extent(n)]=(extent(cs[0]),extent(cs[1]))
                if n['kind']=='UnaryOperator' and n.get('opcode')=='*':
                    if extent(n)==extent(cs[0]):raise UnsupportedSource('Ambiguous pointer source range')
                    pointers[extent(n)]=extent(cs[0])
                if n['kind']=='ImplicitCastExpr' and n.get('castKind')=='LValueToRValue':wrap(cs[0],1)
                elif n['kind']=='BinaryOperator' and n.get('opcode')=='=':wrap(cs[0],2)
                elif n['kind']=='CompoundAssignOperator':wrap(cs[0],3)
                elif n['kind']=='UnaryOperator' and n.get('opcode') in {'++','--'}:wrap(cs[0],3)
        else:
            for call in walk(body):
                if call['kind']!='CUDAKernelCallExpr':continue
                cs=children(call);configuration=children(cs[1])[1:]
                dims=[]
                for expr in configuration:
                    dims.append('0' if expr['kind']=='CXXDefaultArgExpr' else text(expr))
                while len(dims)<4:dims.append('0')
                callee=text(cs[0]);args=[]
                for argument in cs[2:]:
                    value=text(argument)
                    if argument.get('castKind')=='NullToPointer':value='static_cast<'+argument['type']['qualType']+'>('+value+')'
                    args.append(value)
                captures=','.join(f'esp_arg_{i}=esp_source_argument(({a}))' for i,a in enumerate(args))
                invoke=callee+'('+','.join(f'esp_arg_{i}' for i in range(len(args)))+');'
                replacement=f'esp_source_launch(({dims[0]}),({dims[1]}),({dims[2]}),{json.dumps(callee)},[{captures}]() mutable {{{invoke}}},({dims[3]}))'
                replacements[extent(call)]=replacement.encode()
    keys=set(insertions)|set(replacements)|set(indexes)|set(pointers)
    def region(a,b,exclude=None):
        candidates=sorted((k for k in keys if k!=exclude and a<=k[0] and k[1]<=b),key=lambda k:(k[0],-k[1]))
        outer=[]
        for key in candidates:
            if outer and key[0]<outer[-1][1]:
                if key[1]>outer[-1][1]:raise UnsupportedSource('Crossing source instrumentation ranges')
                continue
            outer.append(key)
        parts=[];cursor=a
        for key in outer:
            parts.append(source[cursor:key[0]].decode());parts.append(render(key));cursor=key[1]
        parts.append(source[cursor:b].decode())
        return ''.join(parts)
    def render(key):
        if key in replacements:return replacements[key].decode()
        if key in indexes:
            base,index=indexes[key]
            code='([&]() -> decltype(auto) { auto __esp_base=('+region(*base)+'); auto __esp_index=('+region(*index)+'); return esp_source_index(__esp_base,__esp_index); }())'
        elif key in pointers:
            code='(*esp_source_pointer(('+region(*pointers[key])+')))'
        else:code=region(*key,exclude=key)
        if key in insertions:code='esp_source_access(('+code+'),'+str(insertions[key])+')'
        return code
    generated=region(0,len(source))
    # CUDA's placement-new overrides coexist with libc++ only while CUDA target
    # attributes are active. Native code uses libc++ placement new; device-side
    # allocations are rejected above, so these shim definitions are unnecessary.
    native_lines=[];current_file=''
    for line in generated.splitlines(keepends=True):
        marker=re.match(r'^#\s+\d+\s+"([^"\n]+)"',line)
        if marker:current_file=marker[1]
        if 'cuda_wrappers/new' in current_file and not marker:continue
        native_lines.append(line)
    generated=''.join(native_lines)
    # The parser shim declares CUDA math overloads. Native execution uses the
    # platform C/C++ math library instead; retaining shim overloads changes linkage.
    generated=re.sub(r'^\s*__attribute__\s*\(\(\s*device\s*\)\)\s+[^;{}\n]*\b(?:'+math_names+r')\s*\([^;{}\n]*;','',generated,flags=re.MULTILINE)
    generated=re.sub(r'__attribute__\s*\(\(\s*(?:global|device|host|constant)\s*\)\)', '', generated)
    generated=re.sub(r'__attribute__\s*\(\(\s*shared\s*\)\)', 'static', generated)
    generated=generated.replace('extern const uint3 threadIdx, blockIdx, blockDim, gridDim;',
                                'extern thread_local uint3 threadIdx, blockIdx, blockDim, gridDim;')
    if not host:
        for operation,comparison in [('min','<'),('max','>')]:
            definitions=''.join(f'inline {typ} {operation}({typ} a,{typ} b){{return a{comparison}b?a:b;}}' for typ in ('int','unsigned int','float','double'))
            generated=re.sub(r'template\s*<class T>\s*T\s+'+operation+r'\(T,T\);',definitions,generated)
    generated=generated.replace('struct texture { int filterMode; bool normalized; cudaChannelFormatDesc channelDesc; int addressMode[3]; }',
        'struct texture { int filterMode; bool normalized; cudaChannelFormatDesc channelDesc; const T* esp_data=nullptr; size_t esp_count=0; int addressMode[3]={0,0,0}; }')
    destination=build/(unit.name+'.native.cpp')
    destination.write_text(PREFIX+generated+'\n#undef int2\n#undef catch\n#undef min\n#undef max\n#include "'+str(ROOT/'source_analysis/native/runtime.hpp')+'"\n')
    return destination,dict(source_sha256=hashlib.sha256(unit.read_bytes()).hexdigest(),
                            expanded_sha256=hashlib.sha256(source).hexdigest(),device_functions=selected)


def graph_from_source_events(events):
    """Reuse region dependency construction for source-executed kernel I/O.

Reads satisfied by an earlier write in the same thread/kernel are retained in
launch metadata, not fabricated as inputs from a pre-launch buffer version.
"""
    from source_region_graph import SourceRegionGraph
    e=SourceRegionGraph();launches=[]
    def buffer(number):return f'device_{number}'
    for event in events:
        kind=event['event'];start=len(e.nodes)
        if kind=='allocate':e.allocate(buffer(event['buffer']),event['size']);continue
        if kind=='free':
            e.allocations.pop(buffer(event['buffer']));e.regions.pop(buffer(event['buffer']));continue
        if kind=='copy':
            size=event['size'];src,dst=event['source'],event['destination']
            if src<0:
                name=buffer(dst);region=[[event['destination_offset'],event['destination_offset']+size]]
                e.operation=e.order.schedule([(name,'write')],'default')
                e.transfer(name,region,'cpu','mem','upload',True)
            elif dst<0:
                name=buffer(src);region=[[event['source_offset'],event['source_offset']+size]]
                e.operation=e.order.schedule([(name,'read')],'default',blocking=True)
                node=e.transfer(name,region,'mem','cpu','download',False)
                e.control={node}
            else:
                source,target=buffer(src),buffer(dst)
                sr=e.normalize(source,[[event['source_offset'],event['source_offset']+size]])
                dr=[[event['destination_offset'],event['destination_offset']+size]]
                parents=e.dependencies(source,sr,False)
                e.operation=e.order.schedule([(source,'read'),(target,'write')],'default')
                node=e.transfer(target,dr,'mem','mem','device_copy',True,parents)
                e.update(source,sr,node,False)
        elif kind=='memset':
            name=buffer(event['buffer']);region=[[event['offset'],event['offset']+event['size']]]
            e.operation=e.order.schedule([(name,'write')],'default')
            e.transfer(name,region,'acc','mem','memset',True)
        elif kind=='kernel':
            accesses=[]
            for item in event['accesses']:
                if item['external_reads']:accesses.append((buffer(item['buffer']),'read'))
                if item['writes']:accesses.append((buffer(item['buffer']),'write'))
            e.operation=e.order.schedule(accesses,'default');parents=set()
            for item in event['accesses']:
                if item['external_reads']:
                    parents.add(e.transfer(buffer(item['buffer']),item['external_reads'],'mem','acc',event['name']+':read',False))
            for item in event['accesses']:
                if item['writes']:e.transfer(buffer(item['buffer']),item['writes'],'acc','mem',event['name']+':write',True,parents)
            launches.append(dict(kernel=event['name'],grid=event['grid'],block=event['block'],
                                 global_access_footprints=event['accesses'],node_ids=list(range(start,len(e.nodes)))))
        else:raise UnsupportedSource('Unknown source runtime event: '+kind)
    if not launches or not e.nodes:raise UnsupportedSource('Complete source execution produced no kernel footprint graph')
    return dict(directed=True,multigraph=False,graph={},nodes=e.nodes,
                edges=[dict(source=a,target=b) for a,b in sorted(e.edges)]),e.metadata,launches,dict(e.allocations)


def analyze_native_case(staged,config,build,out,clang='clang++'):
    """Copy and verify one scenario, compile its original drivers, then analyze."""
    import shutil
    staged,build,out=Path(staged).resolve(),Path(build).resolve(),Path(out).resolve()
    case=config['case'];record=ROOT/'workloads/rodinia/captured_scenarios'/f'{case}.json'
    scenario=json.loads(record.read_text());workspace=build/'source_workspace'
    if workspace.exists():raise UnsupportedSource('Native source analysis requires a fresh build directory')
    evidence={}
    for relative,expected in scenario['source_and_input_sha256'].items():
        source=(staged/relative).resolve()
        if not source.is_relative_to(staged):raise UnsupportedSource('Source path escapes staging directory')
        content=source.read_bytes();actual=hashlib.sha256(content).hexdigest()
        if actual!=expected:raise UnsupportedSource('Captured source/input changed: '+relative)
        evidence[relative]=actual
    for relative in evidence:
        destination=workspace/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(staged/relative,destination)
    corrections=[]
    if config.get('source_fixes'):
        correction_file=(ROOT/config['source_fixes']).resolve()
        if not correction_file.is_relative_to(ROOT):raise UnsupportedSource('Source-fix evidence must be in this repository')
        for fix in json.loads(correction_file.read_text()):
            path=(workspace/fix['path']).resolve()
            if not path.is_relative_to(workspace):raise UnsupportedSource('Source fix escapes workspace')
            before=path.read_bytes()
            if hashlib.sha256(before).hexdigest()!=fix['before_sha256']:raise UnsupportedSource('Source fix baseline changed')
            after=before.replace(fix['old'].encode(),fix['new'].encode())
            if hashlib.sha256(after).hexdigest()!=fix['after_sha256']:raise UnsupportedSource('Source fix result changed')
            path.write_bytes(after);corrections.append(fix)
    fixtures=[]
    for fixture in config.get('input_fixtures',[]):
        destination=(workspace/fixture['path']).resolve()
        if not destination.is_relative_to(workspace):raise UnsupportedSource('Fixture escapes source workspace')
        if destination.exists():raise UnsupportedSource('Fixture must not overwrite captured evidence')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if fixture['kind']=='literal':destination.write_text(fixture['text'])
        elif fixture['kind']=='byte_pattern':
            pattern=bytes.fromhex(fixture['pattern_hex']);repeats=fixture['repeats']
            if type(repeats) is not int or repeats<1 or len(pattern)*repeats>1048576:raise UnsupportedSource('Invalid bounded byte fixture')
            destination.write_bytes(pattern*repeats)
        elif fixture['kind']=='avi_first_frames':
            from native_input_fixtures import first_avi_frames
            original=(workspace/fixture['source']).resolve()
            if not original.is_relative_to(workspace):raise UnsupportedSource('AVI source escapes workspace')
            destination.write_bytes(first_avi_frames(original.read_bytes(),fixture['frames'],fixture.get('width')))
        else:raise UnsupportedSource('Unknown native input fixture')
        sha=hashlib.sha256(destination.read_bytes()).hexdigest()
        if fixture.get('sha256') and sha!=fixture['sha256']:raise UnsupportedSource('Generated fixture hash changed')
        fixtures.append(dict(path=fixture['path'],sha256=sha,definition=fixture));evidence[fixture['path']]=sha
    directory=workspace/scenario['working_directory'];generated=[];units={}
    includes=[(directory/p).resolve() for p in config.get('include_directories',[])]
    if any(not p.is_relative_to(directory) or not p.is_dir() for p in includes):
        raise UnsupportedSource('Native includes must exist inside the application directory')
    for name in config['translation_units']:
        path=(directory/name).resolve()
        if not path.is_relative_to(directory):raise UnsupportedSource('Translation unit escapes application directory')
        transformed,details=instrument(path,build/'compiled',includes,config.get('compile_defines'),clang,source_root=directory,shared_memory=config.get('shared_memory','reject'))
        generated.append(str(transformed));units[name]=details
    for name in config.get('host_c_translation_units',[]):
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or path.suffix!='.c':raise UnsupportedSource('Host C unit must be inside the application')
        object_file=build/(name.replace('/','_')+'.o')
        c_flags=['-x','c','-std=gnu89','-Wno-error=implicit-function-declaration','-Wno-error=implicit-int',
                 '-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(directory),'-I'+str(ROOT/'source_analysis/include')]
        c_flags+=['-I'+str(p) for p in includes]
        c_flags+=[f'-D{k}={v}' for k,v in config.get('compile_defines',{}).items()]
        checked_run([clang,*c_flags,'-c',str(path),'-o',str(object_file)])
        generated.append(str(object_file));units[name]=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),language='C',device_functions=[])
    executable=build/'source_application'
    flags=['-std=c++17','-O0','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    compilation=checked_run([clang,*flags,*generated,'-o',str(executable)])
    out.mkdir(parents=True,exist_ok=True)
    (out/'build.log').write_text(compilation.stderr)
    events=build/'source_events.jsonl'
    env={**os.environ,'ESP_SOURCE_EVENTS':str(events),
         'ASAN_OPTIONS':'detect_leaks=0:detect_stack_use_after_return=0',
         'UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1',
         'ESP_SOURCE_SHARED':config.get('shared_memory','reject'),
         'ESP_SOURCE_SERIAL_STREAMS':'1' if config.get('serial_stream_projection') else '0'}
    execution=subprocess.run([str(executable),*config['arguments']],cwd=directory,env=env,capture_output=True,text=True,timeout=config.get('timeout_seconds',120))
    (out/'stdout.log').write_text(execution.stdout);(out/'stderr.log').write_text(execution.stderr)
    if execution.returncode:raise UnsupportedSource('Complete native source application failed:\n'+execution.stderr[-12000:])
    for required in config.get('require_stdout',[]):
        if required not in execution.stdout:raise UnsupportedSource('Application validation marker missing: '+required)
    records=[json.loads(line) for line in events.read_text().splitlines()]
    graph,nodes,launches,live=graph_from_source_events(records)
    hashes={relative:hashlib.sha256((workspace/relative).read_bytes()).hexdigest() for relative in evidence}
    metadata=dict(model='Clang-instrumented CPU source execution: kernel external-input/output footprints',
                  backend='native_source_execution',experimental=True,
                  capture_provenance=dict(case=case,capture=scenario['capture'],archive_sha256=scenario['archive_sha256'],original_arguments=scenario['arguments']),
                  host_environment=dict(system=platform.system(),machine=platform.machine(),release=platform.release()),volume_unit='bytes',configuration=config,nodes=nodes,launches=launches,
                  source_and_input_sha256=hashes,source_corrections=corrections,input_fixtures=fixtures,source_units=units,
                  backend_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('cuda_native_source_model.py','source_region_graph.py','source_analysis/native/runtime.hpp','source_analysis/native/parse_compat.hpp','source_analysis/native/block_executor.hpp','native_input_fixtures.py')},
                  compiler=checked_run([clang,'--version']).stdout.splitlines()[0],compile_flags=flags,
                  allocations_live_at_process_exit=live,gpu_measurements_attached=False,
                  operation_order_representation='Immediate predecessor in serial default-stream source execution',
                  source_events_sha256=hashlib.sha256(events.read_bytes()).hexdigest(),
                  matches_captured_scenario=False,source_matches_capture=not corrections,
                  application_arguments_match_capture=config['arguments']==scenario['arguments'],
                  capture_build_equivalence_established=False,
                  assumptions=[
                      'Original complete host drivers and instrumented kernel bodies execute on the CPU; no GPU run or timing measurement occurs.',
                      'CPU math and source-selected branches are fixture-specific; GPU floating-point, FMA, texture, ABI and numerical equivalence are not established.',
                      'Aggregate copies conservatively cover their native object storage, including padding. Only external kernel input reads and global output writes form transfer nodes. All global read footprints, including reads after same-thread writes, remain in per-launch metadata.',
                      'Uninitialized device reads, unordered inter-thread memory conflicts and concurrent host execution are rejected.',
                      'Shared-memory execution mode: '+config.get('shared_memory','reject')+'. Checked block execution is experimental; it is not general CUDA scheduling equivalence.',
                      'Non-default streams are '+('projected onto serial CPU execution; concurrent-stream correctness is not established.' if config.get('serial_stream_projection') else 'rejected.'),
                      'ASan/UBSan check this native execution; this is not proof of general CUDA memory safety or shared-memory correctness.',
                      'Every later operation conservatively depends on prior CPU-visible downloads. Edges are logical region dependencies, not profiler event order.',
                      'GPU enrichment is separate and requires matching source/input/control-flow evidence; corrected-source or smaller scenarios cannot inherit old measurements.'
                  ])
    (out/'source_simulator.json').write_text(json.dumps(graph,indent=2)+'\n')
    (out/'source_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    shutil.copyfile(events,out/'source_execution_events.jsonl')
    (out/'status.json').write_text(json.dumps(dict(status='complete',backend='native_source_execution',experimental=True,launches=len(launches),nodes=len(graph['nodes']),edges=len(graph['edges']),gpu_measurements_attached=False),indent=2)+'\n')
    return graph,metadata
