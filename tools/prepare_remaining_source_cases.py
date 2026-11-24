#!/usr/bin/env python3
"""Stage source/input evidence for the eight captured applications, without inventing DAGs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

CASES = {
    'dwt2d': ('20261123T005703612561Z', 'main.cu'),
    'heartwall': ('20261123T005703612561Z', 'main.cu'),
    'leukocyte': ('20261123T005703612561Z', 'CUDA/detect_main.c'),
    'myocyte': ('20261123T005703612561Z', 'main.cu'),
    'streamcluster': ('20261123T005703612561Z', 'streamcluster_cuda_cpu.cpp'),
    'huffman': ('20261123T012815164507Z', 'main_test_cu.cu'),
    'hybridsort': ('20261123T012815164507Z', 'main.cu'),
    'mummergpu': ('20261123T020343856158Z', 'src/mummergpu_main.cpp'),
}
SOURCE_SUFFIXES = {'.c', '.cu', '.cpp', '.cc', '.cxx', '.h', '.hh', '.hpp', '.cuh'}


def prepare(archives, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory to avoid mixing scenarios')
    # Verify all selected bytes before creating any output.
    staged = {}
    for case, (capture, entry) in CASES.items():
        archive = Path(archives) / (capture + '.zip')
        with zipfile.ZipFile(archive) as z:
            status = json.loads(z.read('status.json'))[case]
            if not status.get('trace_collected') or not status.get('cycles_measured'):
                raise ValueError(f'{case}: capture is incomplete')
            hashes = json.loads(z.read('staged_sha256.json'))
            selected = {}
            for name in z.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archive member')
                source = name.startswith(f'staged/cuda/{case}/')
                data = name.startswith(f'staged/data/{case}/')
                bundled_input = source and '/data/' in name
                if not (data or bundled_input or (source and path.suffix in SOURCE_SUFFIXES)):
                    continue
                if name.endswith('/'):
                    continue
                content = z.read(name)
                relative = str(path.relative_to('staged'))
                digest = hashlib.sha256(content).hexdigest()
                if hashes.get(relative) != digest:
                    raise ValueError(f'Capture hash mismatch: {name}')
                selected[relative] = content
            if f'cuda/{case}/{entry}' not in selected:
                raise ValueError(f'Missing application entry for {case}')
            staged[case] = (selected, {
                'capture': capture,
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'entry': f'cuda/{case}/{entry}',
                'working_directory': f'cuda/{case}',
                'arguments': status['arguments'],
                'captured_kernel_count': status['kernel_count'],
                'analysis_status': 'pending',
                'source_and_input_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in selected.items()},
                'scope': 'Exact captured sources and inputs; no GPU trace is used to infer source dependencies.',
            })
    for case, (selected, manifest) in staged.items():
        for relative, content in selected.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        target = output / 'scenarios' / (case + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(case, len(selected), 'source/input files verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archives, args.out_dir)
