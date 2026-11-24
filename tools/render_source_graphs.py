#!/usr/bin/env python3
"""Render the four bounded source DAGs; SVG retains every node for zooming."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# Keep rendering dependency-free apart from the Graphviz CLI.
COLORS = {('acc', 'mem'): '#df6940', ('mem', 'acc'): '#8b80df',
          ('cpu', 'mem'): '#52788b', ('mem', 'cpu'): '#607466'}

def endpoints(node):
    return (str(node.get('src_comp', node.get('source_id', '?'))),
            str(node.get('dest_comp', node.get('dest_id', '?'))))

def color(node):
    return COLORS.get(tuple(x.split('_', 1)[0] for x in endpoints(node)), '#777777')

CASES = {
    'dwt2d': ('rodinia_dwt2d', 'main_simulator.json'),
    'myocyte': ('rodinia_myocyte_native', 'source_simulator.json'),
    'streamcluster': ('rodinia_streamcluster_native', 'source_simulator.json'),
    'mummergpu': ('rodinia_mummergpu_native', 'source_simulator.json'),
}

def render(case, directory, filename):
    source = ROOT / 'results' / directory / filename
    data = json.loads(source.read_text())
    nodes, edges = data['nodes'], data.get('edges', data.get('links', []))
    out = source.parent / 'graphs'
    out.mkdir(exist_ok=True)
    stem = source.stem
    title = f'{case.upper()} | {len(nodes):,} transfers, {len(edges):,} dependencies'
    title += '\nBounded source scenario | logical bytes | no GPU measurements'
    lines = ['digraph transfers {',
             'graph [rankdir=LR, bgcolor="white", pad=0.3, nodesep=0.12, ranksep=0.5, labelloc=t, fontname="Helvetica", fontsize=18, label='+json.dumps(title)+'];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fontcolor=white, penwidth=0];',
             'edge [color="#aaaaaa", arrowsize=0.5, penwidth=0.6];']
    for node in nodes:
        src, dst = endpoints(node)
        label = f"t_{node['id']}\n{src} → {dst}\n{str(node.get('vol', node.get('vol_bytes'))) + ' B'}"
        lines.append(f"n{node['id']} [label={json.dumps(label, ensure_ascii=False)}, fillcolor={json.dumps(color(node))}];")
    for edge in edges:
        lines.append(f"n{edge['source']} -> n{edge['target']};")
    lines.append('}')
    dot = out / f'{stem}.dot'
    dot.write_text('\n'.join(lines)+'\n')
    # Compute layout once. Bound optimization for the large Streamcluster DAG.
    positioned = out / f'{stem}.layout.dot'
    engine = 'dot'
    try:
        subprocess.run(['dot', '-Gnslimit=1', '-Gnslimit1=1', '-Gmclimit=0.05',
                        '-Tdot', str(dot), '-o', str(positioned)], check=True, timeout=45)
    except subprocess.TimeoutExpired:
        engine = 'sfdp'
        subprocess.run(['sfdp', '-Gstart=42', '-Goverlap=prism', '-Tdot',
                        str(dot), '-o', str(positioned)], check=True, timeout=45)
    # Full vector output is zoomable; bounded raster is an overview of that same DAG.
    subprocess.run(['neato', '-n2', '-Tsvg', str(positioned), '-o', str(out/f'{stem}.svg')], check=True)
    subprocess.run(['neato', '-n2', '-Tpng', '-Gsize=24,16', '-Gdpi=150', str(positioned), '-o', str(out/f'{stem}.png')], check=True)
    positioned.unlink()
    manifest = dict(case=case, layout_engine=engine, source_file=filename,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    nodes=len(nodes), edges=len(edges), gpu_measurements_attached=False,
                    note='All nodes and dependencies are retained. SVG supports zoom; PNG is an overview. Colors: orange writes, purple reads, blue uploads, green downloads.')
    (out/'render_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{case}: rendered {len(nodes)} nodes, {len(edges)} edges', flush=True)

if __name__ == '__main__':
    for case, (directory, filename) in CASES.items():
        render(case, directory, filename)
