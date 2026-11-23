"""Focused regression checks for complete-application input and source handling."""
import json
import tempfile
import unittest
from pathlib import Path
from cuda_configured_model import Evaluator, directives_hash
from cuda_source_model import UnsupportedSource

ROOT = Path(__file__).resolve().parents[1]

class RemainingSourceSupport(unittest.TestCase):
    def test_dimension_atoi_matches_c_prefix_semantics(self):
        e = Evaluator.__new__(Evaluator)
        e.eval = lambda n: n
        e.kernel = False
        e.summaries = {}
        e.analyzed_functions = set()
        e.opaque_host = True
        e.functions = {}
        for text, expected in [('64x64',64), (' -12tail',-12), ('x64',0), ('12\0more',12)]:
            self.assertEqual(e.call('atoi',[text]), expected)
        with self.assertRaisesRegex(UnsupportedSource,'32-bit'):
            e.call('atoi',[str(2**31)])

    def test_duplicate_header_names_have_independent_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in [('a','#define FIRST 1\n'),('b','#define SECOND 2\n')]:
                (root/name).mkdir()
                (root/name/'common.h').write_text(text)
            entry=root/'main.cu'
            entry.write_text('#include "a/common.h"\n#include "b/common.h"\nint main(){return 0;}\n')
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                preprocessor_sha256=directives_hash(entry.read_bytes()),
                additional_sources=['a/common.h','b/common.h'],
                additional_preprocessor_sha256={x:directives_hash((root/x).read_bytes()) for x in ['a/common.h','b/common.h']})
            Evaluator(entry,config,'cpu','mem','acc',None)
            (root/'b/common.h').write_text('#define SECOND 3\n')
            with self.assertRaisesRegex(UnsupportedSource,'b/common.h'):
                Evaluator(entry,config,'cpu','mem','acc',None)

if __name__=='__main__':unittest.main()
