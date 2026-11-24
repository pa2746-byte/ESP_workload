#!/usr/bin/env python3
"""Evaluate a verified captured input from source; no profiler data is read.

First restore sources with prepare_remaining_source_cases.py. Reviewed local
source fixes are applied only when both their old and new hashes match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_remaining_source_cases import CASES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_case(case, staged):
    reference = ROOT/'workloads/rodinia'/case
    config_path = reference/'analysis.json'
    if not config_path.exists():
        raise ValueError(f'{case}: no completed source-analysis configuration; graph support remains pending')
    config = json.loads(config_path.read_text())
    manifest = json.loads((staged/'scenarios'/f'{case}.json').read_text())
    expected = json.loads((ROOT/'workloads/rodinia/captured_scenarios'/f'{case}.json').read_text())
    for key in ('capture', 'archive_sha256', 'entry', 'arguments', 'source_and_input_sha256'):
        if manifest[key] != expected[key]:
            raise ValueError(f'{case}: staged manifest differs from recorded scenario: {key}')
    if manifest['arguments'] != config['arguments']:
        raise ValueError('Source configuration arguments differ from staged scenario')
    source = staged/manifest['entry']
    corrections = []
    fixes = reference/'local_source_fixes.json'
    fix = json.loads(fixes.read_text()) if fixes.exists() else None
    for relative, sha in manifest['source_and_input_sha256'].items():
        path = (staged/relative).resolve()
        if not path.is_relative_to(staged.resolve()):
            raise ValueError('Scenario path escapes staged directory')
        actual = digest(path)
        if path == source.resolve() and fix is not None:
            if sha != fix['upstream_sha256'] or digest(reference/source.name) != fix['corrected_sha256']:
                raise ValueError('Reviewed source-fix evidence does not match scenario')
            if actual not in {sha, fix['corrected_sha256']}:
                raise ValueError('Staged entry has unreviewed modifications')
            corrections.append(dict(path=relative, **fix))
        elif actual != sha:
            raise ValueError(f'Staged source/input changed: {relative}')
    # Apply only after every selected source/input has been verified.
    if corrections and digest(source) != fix['corrected_sha256']:
        source.write_bytes((reference/source.name).read_bytes())
    return source, config_path, dict(
        scenario=case, capture=manifest['capture'],
        capture_archive_sha256=manifest['archive_sha256'],
        reviewed_source_corrections=corrections,
        gpu_measurements_attached=False,
        scope='Source-only evaluation. Capture identifies inputs/provenance; profiler events and counts do not determine the graph.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--staged-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    try:
        source, config, provenance = prepare_case(args.case, args.staged_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    command = [sys.executable, str(ROOT/'export_workload_transfers.py'),
               '--source', str(source), '--analysis-config', str(config), '--out-dir', str(args.out_dir)]
    if args.render: command.append('--render')
    subprocess.run(command, check=True)
    (args.out_dir/'source_scenario.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
