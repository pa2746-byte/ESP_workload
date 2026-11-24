"""Contracts for the three explicitly scoped native source scenarios."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class NativeRodiniaArtifacts(unittest.TestCase):
    def test_complete_artifacts_provenance_scope_and_dependency_integrity(self):
        cases={'myocyte':(39,312),'streamcluster':(128,1495),'mummergpu':(2,85)}
        for case,(count,nodes) in cases.items():
            with self.subTest(case=case):
                directory=ROOT/f'results/rodinia_{case}_native'
                graph=json.loads((directory/'source_simulator.json').read_text())
                metadata=json.loads((directory/'source_metadata.json').read_text())
                status=json.loads((directory/'status.json').read_text())
                config=json.loads((ROOT/f'workloads/rodinia/{case}/native_analysis.json').read_text())
                self.assertEqual(status['status'],'complete')
                self.assertEqual(metadata['configuration'],config)
                self.assertEqual(len(graph['nodes']),nodes)
                self.assertEqual(len(metadata['launches']),count)
                self.assertEqual([n['id'] for n in graph['nodes']],list(range(nodes)))
                self.assertTrue(all(e['source']<e['target']<nodes for e in graph['edges']))
                self.assertTrue(all(n['vol']>0 for n in graph['nodes']))
                for node,details in zip(graph['nodes'],metadata['nodes']):
                    self.assertEqual(node['vol'],sum(b-a for a,b in details['byte_ranges']))
                for name,digest in metadata['backend_sha256'].items():
                    self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
                self.assertFalse(metadata['gpu_measurements_attached'])
                self.assertFalse(metadata['matches_captured_scenario'])
                self.assertFalse(metadata['capture_build_equivalence_established'])
                self.assertEqual(metadata['source_events_sha256'],hashlib.sha256((directory/'source_execution_events.jsonl').read_bytes()).hexdigest())
                scenario=json.loads((ROOT/f'workloads/rodinia/captured_scenarios/{case}.json').read_text())
                corrections={f['path']:f for f in metadata['source_corrections']}
                for name,digest in metadata['source_and_input_sha256'].items():
                    self.assertEqual(digest,corrections[name]['after_sha256'] if name in corrections else scenario['source_and_input_sha256'][name])
                if case=='myocyte':
                    self.assertTrue(corrections)
                    self.assertTrue(metadata['application_arguments_match_capture'])
                    for launch in metadata['launches']:
                        footprints={a['buffer']:a for a in launch['global_access_footprints']}
                        self.assertEqual(footprints[1]['writes'],[[0,364]])
                        self.assertEqual(footprints[3]['writes'],[[0,12]])
                        self.assertEqual(footprints[1]['external_reads'],[])
                elif case=='streamcluster':
                    self.assertTrue(metadata['source_matches_capture'])
                    self.assertFalse(metadata['application_arguments_match_capture'])
                else:
                    self.assertEqual(config['compile_defines']['TREETEX'],0)
                    self.assertEqual([l['kernel'] for l in metadata['launches']],['mummergpuKernel','printKernel'])


if __name__=='__main__':unittest.main()
