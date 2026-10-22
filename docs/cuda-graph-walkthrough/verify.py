"""Verify the walkthrough's concrete graph claims against fresh source analysis."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from cuda_source_model import analyze
edges={(0,2),(0,5),(1,3),(1,6),(2,4),(3,4),(4,8),(5,7),(6,7),(7,9),(8,10),(9,10),(10,11)}
previous=None
for name in ('single_stream','explicit_streams'):
    source=HERE/'examples'/(name+'.cu')
    stored=json.loads((HERE/'generated'/(name+'_simulator.json')).read_text())
    meta=json.loads((HERE/'generated'/(name+'_metadata.json')).read_text())
    fresh,nodes=analyze(source)
    assert stored==fresh
    assert meta['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert nodes==meta['nodes']
    assert len(stored['nodes'])==12
    assert {n['vol'] for n in stored['nodes']}=={32}
    assert {(e['source'],e['target']) for e in stored['edges']}==edges
    copies=[n for n in stored['nodes'] if 'cpu' in (n['src_comp'],n['dest_comp'])]
    assert sum(n['vol'] for n in copies)==96
    assert sum(n['vol'] for n in stored['nodes'])==384
    if previous is not None:assert previous==stored
    previous=stored
print('PASS: fresh and saved graphs match; identical 12-node/13-edge DAGs; 32 B per node; 96 B host copies / 384 B logical total.')
