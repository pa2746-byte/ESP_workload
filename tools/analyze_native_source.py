#!/usr/bin/env python3
"""Run the explicit Clang-instrumented CPU source backend, without GPU metrics."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cuda_native_source_model import analyze_native_case

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--staged-dir',type=Path,required=True)
parser.add_argument('--out-dir',type=Path,required=True)
args=parser.parse_args()
args.out_dir.mkdir(parents=True,exist_ok=True)
(args.out_dir/'status.json').write_text(json.dumps({'status':'running'})+'\n')
try:
    with tempfile.TemporaryDirectory(prefix='esp-native-source-') as build:
        graph,metadata=analyze_native_case(args.staged_dir,json.loads(args.config.read_text()),Path(build),args.out_dir)
except Exception as error:
    (args.out_dir/'status.json').write_text(json.dumps({'status':'failed','error':str(error)[-12000:]},indent=2)+'\n')
    raise
print(f"Completed source application: {len(metadata['launches'])} launches, {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
