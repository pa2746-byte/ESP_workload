#!/usr/bin/env python3
"""Check source parsing for the eight remaining applications; never emit guessed DAGs."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cuda_source_model import parse_source,UnsupportedSource
ROOT=Path(__file__).resolve().parents[1]
ENTRIES={'dwt2d':'main.cu','heartwall':'main.cu','huffman':'main_test_cu.cu','hybridsort':'main.cu','leukocyte':'CUDA/find_ellipse_kernel.cu','mummergpu':'src/mummergpu.cu','myocyte':'main.cu','streamcluster':'streamcluster_cuda.cu'}
def main():
    for case,entry in ENTRIES.items():
        folder=ROOT/'workloads/rodinia'/case
        includes=[p for p in (folder/'AVI',folder/'CUDA',folder/'meschach_lib',folder/'src') if p.is_dir()]
        try:
            parse_source(folder/entry,include_dirs=includes)
            result={'entry':entry,'parse':'passed','graph_support':'pending'}
        except UnsupportedSource as error:
            result={'entry':entry,'parse':'failed','diagnostic':str(error)[:5000],'graph_support':'pending'}
        (folder/'analysis_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
        print(case,result['parse'],flush=True)
if __name__=='__main__':main()
