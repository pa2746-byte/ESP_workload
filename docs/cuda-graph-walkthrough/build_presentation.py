"""Build a self-contained browser presentation from the guide and actual graph."""
from pathlib import Path
import html,json,re
ROOT=Path(__file__).resolve().parent

def box(x,y,w,label,detail='',accent=False):
    return f'<rect x="{x}" y="{y}" width="{w}" height="66" rx="8" fill="{"#e5f0ed" if accent else "#f1f2f3"}" stroke="#bac4c6"/><text x="{x+w/2}" y="{y+25}" text-anchor="middle" font-size="16" font-weight="600">{html.escape(label)}</text><text x="{x+w/2}" y="{y+48}" text-anchor="middle" font-size="13">{html.escape(detail)}</text>'
def arrow(x1,y1,x2,y2):return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#718485" stroke-width="1.5" marker-end="url(#arrow)"/>'
def svg(body,w=1000,h=210,label='Conceptual illustration'):
    return f'<figure><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#718485"/></marker></defs>{body}</svg><figcaption>{html.escape(label)}</figcaption></figure>'
def generated_dag():
    g=json.loads((ROOT/'generated/single_stream_simulator.json').read_text())
    positions={0:(15,135),1:(15,315),2:(215,15),3:(215,115),5:(215,335),6:(215,435),4:(420,65),7:(420,385),8:(625,65),9:(625,385),10:(830,225),11:(1035,225)}
    metadata=json.loads((ROOT/'generated/single_stream_metadata.json').read_text())
    names={}
    for node in metadata['nodes']:
        buffer=node['buffer'].removeprefix('d_')
        stage=node['stage']
        if ':' in stage:
            kernel,mode=stage.split(':')
            names[node['id']]=f'{dict(vectorAdd="Add",vectorSub="Sub",vectorMul="Mul").get(kernel,kernel)}: {mode} {buffer}'
        else:names[node['id']]=stage.title()+' '+buffer
    assert len(g['nodes'])==12 and len(g['edges'])==13, 'Review the teaching layout if the graph changes'
    body=''
    for e in g['edges']:
        x,y=positions[e['source']];xx,yy=positions[e['target']]
        body+=arrow(x+160,y+33,xx-4,yy+33)
    for n in g['nodes']:
        x,y=positions[n['id']]
        body+=f'<g class="transfer" tabindex="0"><title>t{n["id"]}: {names[n["id"]]}, {n["vol"]} bytes, {n["src_comp"]} to {n["dest_comp"]}</title>'+box(x,y,160,f't{n["id"]} · {names[n["id"]]}',f'{n["src_comp"]} → {n["dest_comp"]} · {n["vol"]} B',n['src_comp']=='acc')+'</g>'
    return svg(body,1210,520,'Generated DAG · 12 copy/access nodes / 13 prerequisites · 32 bytes per node · not a timeline')

def generated_chain():
    g=json.loads((ROOT/'generated/dependent_chain_simulator.json').read_text())
    meta=json.loads((ROOT/'generated/dependent_chain_metadata.json').read_text())
    positions={0:(10,20),1:(10,130),2:(210,20),3:(210,130),4:(410,75),5:(610,75),6:(810,75),7:(1010,75)}
    body=''
    for e in g['edges']:
        x,y=positions[e['source']];xx,yy=positions[e['target']]
        body+=arrow(x+170,y+33,xx-4,yy+33)
    for n,m in zip(g['nodes'],meta['nodes']):
        assert n['id']==m['id']
        mode=m['stage'].split(':')[-1]
        label=f"t{n['id']} · {mode} {m['buffer'].removeprefix('d_')}"
        detail=('logical ' + mode if ':' in m['stage'] else 'explicit copy')+f" · {n['vol']} B"
        x,y=positions[n['id']]
        body+=box(x,y,170,label,detail,n['id'] in (4,5))
    return svg(body,1200,220,'Generated dependency chain · 8 copy/access nodes / 7 prerequisites · highlighted: C producer → consumer')

def illustrations():
    thread=''
    for i in range(8):
        x=20+i*120
        thread+=box(x,65,108,f'i = {i}',f'block {i//4}, thread {i%4}',i>=4)
    thread+='<text x="20" y="30" font-size="18">Each thread computes C[i], D[i], or E[i] in its respective kernel.</text>'
    split=box(10,70,200,'Shared inputs','A: 32 B + B: 32 B')+box(320,20,220,'Add branch','reads all A and B')+box(320,120,220,'Subtract branch','reads all A and B')+arrow(210,103,315,53)+arrow(210,103,315,153)
    split+=box(620,20,170,'Partition worker 0','A[0:4] · 16 B',True)+box(620,120,170,'Partition worker 1','A[4:8] · 16 B',True)+'<text x="840" y="90" font-size="15">Disjoint regions</text><text x="840" y="115" font-size="15">Illustrative alternative</text>'
    join=''
    for x,a,b,c in [(10,'Elementwise join','C: 32 B + D: 32 B','E: 32 B'),(345,'Concatenation','chunk 0: 16 B + chunk 1: 16 B','combined: 32 B'),(680,'Reduction','8 floats: 32 B','sum: 4 B')]:
        join+=box(x,12,300,a,b)+arrow(x+150,80,x+150,112)+box(x,120,300,c,'Different operations; all may have fan-in',True)
    streams='<text x="10" y="28" font-size="17">Single stream: ordered kernels</text>'+box(10,50,230,'Add','must precede subtract')+box(320,50,230,'Subtract','must precede multiply')+box(630,50,230,'Multiply','after both')+arrow(245,83,315,83)+arrow(555,83,625,83)
    streams+='<text x="10" y="155" font-size="17">Worker streams: independent branches, then event waits</text>'+box(10,175,230,'Add stream','record addDone',True)+box(10,260,230,'Subtract stream','record subtractDone',True)+box(630,218,260,'Join stream','wait both → multiply')+arrow(245,208,625,251)+arrow(245,293,625,251)
    coal='<text x="10" y="25" font-size="17">First 8 lanes shown; the example uses 32-thread warps.</text>'
    for y,title,step in [(55,'Consecutive',1),(140,'Strided',32)]:
        coal+=f'<text x="10" y="{y+35}" font-size="16">{title}</text>'
        for lane in range(8):coal+=box(160+lane*103,y,96,f'lane {lane}',f'in[{lane*step}]',step==1)
    return {'CHAIN':generated_chain(),'THREADS':svg(thread,label='Conceptual thread mapping · teaching launch: 2 blocks × 4 threads'), 'DAG':generated_dag(),'SPLIT':svg(split,label='Shared-input fork versus true partition · conceptual comparison'),'JOIN':svg(join,label='Actual elementwise operation versus illustrative concatenation and reduction'),'STREAMS':svg(streams,h=345,label='Scheduling constraints only · lengths do not indicate durations or measured overlap'),'COALESCING':svg(coal,h=225,label='Illustrative address mapping · same aggregate footprint, different per-warp access patterns')}


def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    return s

def render(md,figures):
    out=[];code=None;listing=False
    for line in md.strip().splitlines():
        if line.startswith('```'):
            if code is None:code=[]
            else:out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');code=None
            continue
        if code is not None:code.append(line);continue
        if listing and not line.startswith('- '):out.append('</ul>');listing=False
        if not line or line.startswith('!['):continue
        marker=re.fullmatch(r'<!-- (\w+) -->',line)
        if marker:out.append(figures[marker[1]]);continue
        if line.startswith('> '):out.append('<aside class="notes">'+inline(line[2:])+'</aside>');continue
        if line.startswith('- '):
            if not listing:out.append('<ul>');listing=True
            out.append('<li>'+inline(line[2:])+'</li>')
        else:out.append('<p>'+inline(line)+'</p>')
    if listing:out.append('</ul>')
    return '\n'.join(out)

text=(ROOT/'README.md').read_text()
parts=re.split(r'^## (.+)$',text,flags=re.M)
figures=illustrations();slides=[];nav=[]
(ROOT/'figures').mkdir(exist_ok=True)
for name,figure in figures.items():
    standalone=re.search(r'<svg.*?</svg>',figure,re.S).group(0)
    (ROOT/'figures'/(name.lower()+'.svg')).write_text(standalone+'\n')
for n in range(1,len(parts),2):
    index=(n-1)//2;title=parts[n]
    slides.append(f'<section class="slide" id="s{index}" aria-label="{html.escape(title)}"><div class="eyebrow">CUDA → LOGICAL TRANSFERS · {index+1:02d}</div><h1>{html.escape(title)}</h1>'+render(parts[n+1],figures)+'</section>')
    nav.append(f'<option value="{index}">{html.escape(title)}</option>')
css='''
:root{font-family:Inter,Arial,sans-serif;color:#203336;background:#eef1f0;font-size:17px;line-height:1.5}*{box-sizing:border-box}body{margin:0}header{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:13px 26px;background:#203336;color:white;position:sticky;top:0;z-index:3}header strong{font-size:16px;margin-right:auto;flex:1 1 240px}select{max-width:min(480px,80vw)}button,select{font:inherit;font-size:14px;padding:8px 12px;border:1px solid #94aaa7;border-radius:5px;background:white;color:#203336;cursor:pointer}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #b36b29}main{max-width:1280px;margin:24px auto}.slide{background:#fff;padding:30px 42px;min-height:730px;margin-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#50756d;font-weight:700}h1{font-size:29px;line-height:1.2;margin:12px 0 22px;max-width:1000px}p{margin:11px 0}ul{padding-left:23px;margin:14px 0}li{margin:9px 0}strong{font-weight:700}a{color:#216c60;text-underline-offset:3px}pre{font-size:13px;line-height:1.38;padding:16px 20px;background:#f1f4f3;border-left:3px solid #527c73;overflow:auto}code{font-family:ui-monospace,Consolas,monospace;font-size:.88em}figure{margin:15px 0}svg{width:100%;max-height:250px;display:block;fill:#203336}#s3 svg{max-height:330px}#s8 svg{max-height:240px}figcaption{font-size:12px;color:#647476;margin-top:6px}.notes{display:none;margin-top:20px;padding:15px 19px;border-top:1px solid #bdcfca;color:#3a655d;font-size:15px}.show-notes .notes{display:block}.presenting .slide{display:none}.presenting .slide.active{display:block}.transfer:focus rect,.transfer:hover rect{stroke:#216c60;stroke-width:3}footer{padding:0 42px 20px;text-align:center;font-size:13px;color:#647476}@media(max-width:800px){header{flex-wrap:wrap;padding:10px}header strong{width:100%}.slide{padding:22px;min-height:0}h1{font-size:24px}select{max-width:70vw}}@media print{@page{size:A4 landscape;margin:12mm}body{background:white;font-size:12px;line-height:1.4}header,footer{display:none!important}main{margin:0;max-width:none}.slide,.presenting .slide{display:block!important;padding:0;min-height:0;margin:0;break-after:page}.slide:last-child{break-after:auto}h1{font-size:23px;margin:8px 0 12px}pre{font-size:10px;padding:10px}p{margin:8px 0}li{margin:5px 0}svg{max-height:170px}#s3 svg{max-height:235px}#s8 svg{max-height:200px}.notes{display:none!important}a{color:inherit}figure{break-inside:avoid}}
'''
script='''
const slides=[...document.querySelectorAll('.slide')];let current=0;
const picker=document.getElementById('picker');
function show(i){current=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle('active',j===current));picker.value=current;document.getElementById('count').textContent=(current+1)+' / '+slides.length;window.scrollTo(0,0);}
document.body.classList.add('presenting');show(0);
document.getElementById('previous').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);picker.onchange=()=>show(Number(picker.value));
document.getElementById('notes').onclick=()=>{let on=document.body.classList.toggle('show-notes');document.getElementById('notes').setAttribute('aria-pressed',on);};
document.getElementById('all').onclick=()=>{let on=document.body.classList.toggle('presenting');document.getElementById('all').textContent=on?'Read all':'Slide view';};
document.getElementById('print').onclick=()=>window.print();
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight'){show(current+1);e.preventDefault();}if(e.key==='ArrowLeft'){show(current-1);e.preventDefault();}});
'''
page='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>From CUDA to a dependency graph</title><style>'+css+'</style></head><body><header><strong>From CUDA to a dependency graph</strong><button id="previous" aria-label="Previous section">←</button><select id="picker" aria-label="Choose section">'+''.join(nav)+'</select><button id="next" aria-label="Next section">→</button><span id="count"></span><button id="notes" aria-pressed="false">Speaker notes</button><button id="all">Read all</button><button id="print">Print / PDF</button></header><main>'+''.join(slides)+'</main><footer>Source-derived teaching guide · October 2026 · No GPU execution or performance measurement</footer><script>'+script+'</script></body></html>'
(ROOT/'presentation.html').write_text(page)
print('Built presentation.html:',len(slides),'sections')
