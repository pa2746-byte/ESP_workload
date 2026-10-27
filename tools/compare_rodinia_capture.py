"""Compare this single-stream Rodinia capture with archived Clang graphs.
Rejects multistream/multiprocess captures: not a general synchronization parser.
"""
import argparse, json, sqlite3, hashlib, html
from pathlib import Path

def save(path, data): path.write_text(json.dumps(data, indent=2)+'\n')
def graph_svg(nodes, edges, title):
    width=510; height=100+len(nodes)*110
    pos={n['id']:i for i,n in enumerate(nodes)}
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(title)}</title><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#546a70"/></marker></defs>']
    for e in edges:
        a,b=pos[e['source']],pos[e['target']];y1=70+a*110+65;y2=70+b*110
        if b==a+1:d=f'M250 {y1} V{y2-5}'
        else:
            x=440+((a+b)%4)*14;d=f'M420 {y1-32} H{x} V{y2+30} H425'
        out.append(f'<path d="{d}" fill="none" stroke="#546a70" marker-end="url(#a)"/>')
    for n in nodes:
        y=70+pos[n['id']]*110
        out.append(f'<rect x="20" y="{y}" width="400" height="65" rx="7" fill="#eef4f2" stroke="#aebdb8"/><text x="32" y="{y+25}" font-family="Arial" font-size="15">{html.escape(str(n["id"])+": "+n["label"])}</text><text x="32" y="{y+48}" font-family="Arial" font-size="13">{html.escape(n.get("detail",""))}</text>')
    out.append('</svg>');return ''.join(out)

def analyze(bundle,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((bundle/'manifest.json').read_text())
    for f,h in manifest['sha256'].items():
        assert hashlib.sha256((bundle/f).read_bytes()).hexdigest()==h,f
    summary={'source_revision':manifest['source_revision'],'manifest_hashes_match':True,'workloads':{}}
    sections=[]
    for name in ['nn','hotspot3d']:
        reference=bundle/'clang_reference'/('rodinia_'+name)
        g=json.loads(next(reference.glob('*simulator.json')).read_text())
        m=json.loads(next(reference.glob('*metadata.json')).read_text())
        src=bundle/'sources'/name/Path(m['source']).name
        assert hashlib.sha256(src.read_bytes()).hexdigest()==m['source_sha256']
        for unit,h in m['configured_analysis'].get('source_units',{}).items():
            assert hashlib.sha256((src.parent/unit).read_bytes()).hexdigest()==h
        c=sqlite3.connect(f'file:{bundle/name}.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row
        strings=dict(c.execute('SELECT id,value FROM StringIds'))
        kinds=dict(c.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        runtime=[dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        assert all(r['returnValue']==0 for r in runtime),'CUDA API error'
        events=[]
        for table,typ in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy')]:
            for row in c.execute('SELECT * FROM '+table):
                r=dict(row);api=[a for a in runtime if a['correlationId']==r['correlationId']]
                assert len(api)==1
                r.update(type=typ,api_name=strings[api[0]['nameId']],api_start=api[0]['start'],api_end=api[0]['end'],host_thread=api[0]['globalTid'])
                r['label']=strings[r['shortName']] if typ=='kernel' else kinds[r['copyKind']]
                r['duration_us']=(r['end']-r['start'])/1000
                r['detail']=f"{r['duration_us']:.3f} µs"+(f" · {r['bytes']:,} B" if typ=='copy' else '')
                events.append(r)
        assert len({(e['globalPid'],e['deviceId'],e['contextId'],e['streamId'],e['host_thread']) for e in events})==1,'Only single stream/thread supported'
        events.sort(key=lambda e:e['api_start'])
        for i,e in enumerate(events):e['id']=i
        assert all(a['api_end']<=b['api_start'] and a['end']<=b['start'] for a,b in zip(events,events[1:]))
        edges=[{'source':a['id'],'target':b['id'],'kind':'same_stream_order','evidence':'same process/context/stream; correlated serial host submissions'} for a,b in zip(events,events[1:])]
        trace={'model':'observed GPU operations; stream-order DAG, not buffer-dependency DAG','nodes':events,'edges':edges}
        groups={};byid={n['id']:n for n in g['nodes']}
        for n in m['nodes']:groups.setdefault(n['operation_id'],[]).append(n)
        grouped=[]; mapping={}
        launches=iter(m['configured_analysis']['launches'])
        for op,ns in groups.items():
            kernel=':' in ns[0]['stage'];label=ns[0]['stage'].split(':')[0]
            gn={'id':op,'label':label,'detail':', '.join('t'+str(n['id']) for n in ns),'logical_nodes':[n['id'] for n in ns]}
            observed=events[len(grouped)]
            if kernel:
                launch=next(launches)
                assert observed['type']=='kernel' and observed['label']==launch['kernel']
                assert [observed['grid'+x] for x in 'XYZ']==launch['grid']
                assert [observed['block'+x] for x in 'XYZ']==launch['block']
            else:
                vol=byid[ns[0]['id']]['vol'];gn['label']+=' '+ns[0]['buffer'];gn['detail']+=f' · {vol:,} B'
                assert observed['type']=='copy' and observed['bytes']==vol
                # Decode direction via the trace enum, not a guessed ordinal.
                expected='Host-to-Device' if label=='upload' else 'Device-to-Host'
                assert expected.lower() in observed['label'].replace(' ','').lower(),observed['label']
            grouped.append(gn)
            for n in ns:mapping[n['id']]=op
        assert len(grouped)==len(events)
        pairs=sorted({(mapping[e['source']],mapping[e['target']]) for e in g['edges'] if mapping[e['source']]!=mapping[e['target']]})
        gedges=[{'source':a,'target':b,'kind':'source_buffer_dependency'} for a,b in pairs]
        assert all(a<b for a,b in pairs),'Observed stream order violates source dependency'
        raw=[dict(n,label=next(x['stage']+' '+x['buffer'] for x in m['nodes'] if x['id']==n['id']),detail=f"{n['vol']:,} logical B · {n['src_comp']} → {n['dest_comp']}") for n in g['nodes']]
        groupedgraph={'model':'Clang graph grouped by operation; no timing inference','nodes':grouped,'edges':gedges}
        save(out/(name+'_execution_dag.json'),trace);save(out/(name+'_clang_grouped_dag.json'),groupedgraph)
        save(out/(name+'_clang_original.json'),g)
        metrics={'trace_nodes':len(events),'trace_edges':len(edges),'clang_nodes':len(raw),'clang_edges':len(g['edges']),'grouped_clang_nodes':len(grouped),'grouped_clang_edges':len(gedges),'kernel_us':[e['duration_us'] for e in events if e['type']=='kernel'],'copy_bytes':sum(e['bytes'] for e in events if e['type']=='copy'),'gpu_active_us':sum(e['duration_us'] for e in events),'first_to_last_gpu_event_us':(events[-1]['end']-events[0]['start'])/1000,'checks':'source hashes, kernel names/counts, grid/block dimensions, copy directions/bytes and dependency-order consistency passed','sqlite_sha256':hashlib.sha256((bundle/(name+'.sqlite')).read_bytes()).hexdigest()}
        summary['workloads'][name]=metrics
        diagrams=[]
        for suffix,title,nodes,es in [('execution','Observed execution · stream order',events,edges),('clang_grouped','Clang · grouped buffer dependencies',grouped,gedges),('clang_original','Clang · original logical accesses',raw,g['edges'])]:
            svg=graph_svg(nodes,es,title);(out/(name+'_'+suffix+'.svg')).write_text(svg)
            diagrams.append(f'<div><h3>{title}</h3>{svg}</div>')
        rows=''.join(f'<tr><td>{e["id"]}</td><td>{html.escape(e["label"])}</td><td>{e.get("bytes","—")}</td><td>{e["duration_us"]:.3f}</td></tr>' for e in events)
        sections.append(f'<section><h2>{name}</h2><p>{metrics["checks"]}. Original Clang: {len(raw)} nodes / {len(g["edges"])} edges; observed operations: {len(events)} / {len(edges)}.</p><div class="diagrams">'+''.join(diagrams)+f'</div><details><summary>Measured GPU event durations</summary><table><tr><th>Operation</th><th>Event</th><th>Copy bytes</th><th>Duration (µs)</th></tr>{rows}</table><p>Active event sum: {metrics["gpu_active_us"]:.3f} µs. First GPU event start to last end: {metrics["first_to_last_gpu_event_us"]:.3f} µs. Neither is full application runtime.</p></details></section>')
        c.close()
    save(out/'comparison.json',summary)
    intro='''<h1>Rodinia: captured execution versus Clang dependencies</h1><p>Tesla T4 · capture 20261027T033003Z · 513 NN records; HotSpot3D 64×64×3, two iterations · deterministic synthetic inputs.</p><h2>Result: operation structure matches; dependency semantics differ</h2><p>Both traces match source-predicted kernel names, counts, launch dimensions, and explicit copy directions and sizes. All modeled dependencies are consistent with the observed order. This does not independently verify every kernel read/write footprint.</p><p><b>Nearest Neighbor:</b> the four Clang access nodes collapse to three operations: upload → euclid → download. This matches the three-event execution DAG.</p><p><b>HotSpot3D:</b> nine Clang access nodes collapse to five operations. The observed chain is upload temperature → upload power → iteration 1 → iteration 2 → download. Stream order adds constraints beyond the buffer DAG.</p><p><b>The important difference:</b> the source-derived download depends on iteration 1's output, even though it executes after iteration 2. The source swaps pointers and downloads the previous buffer. The trace lacks buffer addresses here, so it cannot independently confirm the downloaded buffer's identity. Do not replace the source dependency with an iteration-2 producer edge merely because of timing.</p><p>Clang read/write volumes are unique logical footprints. Trace copy bytes describe explicit copies. Kernel nodes deliberately have no physical-traffic volume; zero would be misleading. Stream-order edges do not carry buffer sizes.</p>'''
    tail='''<h2>Limits and simulator use</h2><ul><li>One profiled execution per workload; microsecond durations include profiling effects. These are observations, not stable performance estimates.</li><li>GPU event sums exclude host setup and gaps; event spans include gaps but still exclude application startup and CPU processing.</li><li>No cache, DRAM-traffic, internal synchronization, or numerical correctness validation. HotSpot's printed Accuracy alone is not proof of correctness.</li><li>Source/config hashes match the bundle. NN uses the recorded legacy API compatibility macro. Clang device configuration is synthetic, but actual launch dimensions match.</li><li>Use Clang for buffer relationships and the trace for operation timing and stream order. Preserve both edge types in a combined simulator model.</li><li>This extractor intentionally rejects multiple streams, contexts, processes or submitting threads; arbitrary traces require explicit event/synchronization analysis.</li></ul><p>SVG positions are schematic, not time-scaled. JSON IDs in grouped and observed graphs identify corresponding operations; original t IDs identify logical accesses. Source: archived SQLite and Clang JSON in the supplied bundle. See comparison.json for database hashes and exact measurements.</p>'''
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Rodinia DAG comparison</title><style>body{font:16px/1.55 system-ui;margin:35px auto;max-width:1450px;padding:0 25px;color:#203638}h1{font-size:28px}h2{margin-top:35px}section{border-top:1px solid #ccd5d1;margin-top:35px}.diagrams{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}svg{width:100%;height:auto}h3{font-size:17px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600}@media(max-width:850px){.diagrams{grid-template-columns:1fr}}</style>'+intro+''.join(sections)+tail+'</html>')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--out',type=Path,default=Path('results/rodinia_trace_comparison'));a=p.parse_args();analyze(a.bundle,a.out)
