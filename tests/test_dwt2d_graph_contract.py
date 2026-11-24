"""Independent source schedule for the captured 64x64 RGB forward 5/3 case.

main.cu initializes seven int planes; components.cu initializes/uploads RGB;
dwt.cu copies each component into a reused backup before its transform.
OUTPUT is undefined, so the complete driver performs no result downloads.
No profiler data or measured launch counts are used in this contract.
"""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Dwt2dGraphContract(unittest.TestCase):
    def test_full_source_schedule_and_provenance(self):
        result = ROOT/'results/rodinia_dwt2d'
        graph = json.loads((result/'main_simulator.json').read_text())
        metadata = json.loads((result/'main_metadata.json').read_text())
        pixels = 64*64
        plane, rgb = pixels*4, pixels*3
        convert = 'c_CopySrcToComponents<int>'
        wavelet = 'fdwt53Kernel<64,8>'
        roles = [(b,'memset',plane,'acc','mem') for b in
                 ('c_r_out','backup','c_g_out','c_b_out','c_r','c_g','c_b')]
        roles += [('d_src','memset',rgb,'acc','mem'), ('d_src','upload',rgb,'cpu','mem'),
                  ('d_src',convert+':read',rgb,'mem','acc')]
        roles += [(b,convert+':write',plane,'acc','mem') for b in ('c_r','c_g','c_b')]
        for component in ('r','g','b'):
            roles += [('backup','device_copy',plane,'mem','mem'),
                      ('c_'+component,wavelet+':read',plane,'mem','acc'),
                      ('c_'+component+'_out',wavelet+':write',plane,'acc','mem')]
        actual = [(d['buffer'],d['stage'],n['vol'],n['src_comp'],n['dest_comp'])
                  for n,d in zip(graph['nodes'],metadata['nodes'])]
        self.assertEqual(actual,roles)
        expected = {(i,j) for i,j in ((0,15),(1,13),(2,18),(3,21),(4,10),(5,11),(6,12),
                    (7,8),(8,9),(9,10),(9,11),(9,12),(10,13),(10,14),
                    (11,16),(11,17),(12,19),(12,20),(13,16),(14,15),(16,19),(17,18),(20,21))}
        self.assertEqual({(e['source'],e['target']) for e in graph['edges']},expected)
        details = metadata['configured_analysis']
        self.assertEqual([(l['kernel'],l['grid'],l['block']) for l in details['launches']],
                         [(convert,[16,1,1],[256,1,1])] + [(wavelet,[1,8,1],[64,1,1])]*3)
        self.assertEqual(details['used_host_summaries'],[])
        self.assertEqual(details['source_units'],details['configuration']['source_hashes'])
        self.assertIn('common.h',details['source_units'])
        self.assertIn('dwt_cuda/common.h',details['source_units'])
        provenance=json.loads((result/'source_scenario.json').read_text())
        self.assertFalse(provenance['gpu_measurements_attached'])
        self.assertTrue(provenance['reviewed_source_corrections'])


if __name__ == '__main__':
    unittest.main()
