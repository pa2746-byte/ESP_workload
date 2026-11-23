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

    def test_host_strings_preserve_terminators_and_reject_overflow(self):
        from cuda_integer_model import IntegerEvaluator
        from cuda_configured_model import Pointer
        e = IntegerEvaluator.__new__(IntegerEvaluator)
        e.kernel = False
        e.functions = {}
        e.eval = lambda value: value
        e.host_sizes = {'small': 8, 'enough': 9}
        e.known_host = {}
        e.host_allocation_count = 0
        with self.assertRaisesRegex(UnsupportedSource, 'terminating NUL'):
            e.call('strcpy', [Pointer('small',0,'char'), 'input.rgb'])
        e.host_sizes['enough'] = 10
        dst = Pointer('enough',0,'char')
        e.call('strcpy', [dst, 'input.rgb'])
        self.assertEqual(e.call('strlen',[dst]), 9)
        suffix = e.call('strstr',[dst,'.rgb'])
        self.assertEqual(suffix.offset, 5)
        self.assertEqual(e.host_string_bytes(suffix), b'.rgb')
        self.assertIsNone(e.call('strstr',[dst,'missing']))
        with self.assertRaisesRegex(UnsupportedSource, 'terminating NUL'):
            e.call('strcat',[dst,'x'])
        duplicate = e.call('strdup',[dst])
        self.assertEqual(e.host_string_bytes(duplicate), b'input.rgb')

    def test_overloads_and_integer_template_arguments_remain_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = Path(folder)/'main.cu'
            entry.write_text('int helper(int x){return x;} float helper(float x){return x;}\n'
                'template<int N> int count(){return N;}\n'
                '__device__ int lane(){return threadIdx.x;}\n'
                'int main(){return helper(count<2>())+helper(count<3>());}')
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                preprocessor_sha256=directives_hash(entry.read_bytes()),additional_sources=[],additional_preprocessor_sha256={})
            e=Evaluator(entry,config,'cpu','mem','acc',None)
            self.assertIn('count<2>',e.functions)
            self.assertIn('count<3>',e.functions)
            self.assertIn('helper [int (int)]',e.functions)
            self.assertIn('helper [float (float)]',e.functions)
            e.current_function = 'main'
            self.assertEqual(e.invoke_function('count<2>',[],[]),2)
            e.kernel = True
            e.env = {'threadIdx': {'x': 7, 'y': 0, 'z': 0}}
            e.current_function = 'main'
            self.assertEqual(e.invoke_function('lane',[],[]),7)
            self.assertEqual(e.env['threadIdx']['x'],7)

    def test_shared_cast_preserves_byte_extent_and_shift_bounds(self):
        from cuda_configured_model import SharedView
        e=Evaluator.__new__(Evaluator)
        e.types={'unsigned char':(1,1,{}),'float':(4,4,{}),'int':(4,4,{})}
        e.opaque_shared=True
        e.eval=lambda n: SharedView((768,), 'unsigned char')
        cast={'kind':'CStyleCastExpr','type':{'qualType':'float *'},'inner':[{'kind':'DeclRefExpr'}]}
        result=Evaluator.eval(e,cast)
        self.assertEqual((result.shape,result.base),((192,),'float'))
        e.eval=lambda n: n['value']
        shift={'kind':'BinaryOperator','opcode':'>>','type':{'qualType':'int'},'inner':[{'kind':'IntegerLiteral','value':768},{'kind':'IntegerLiteral','value':2}]}
        self.assertEqual(Evaluator.eval(e,shift),192)
        shift['inner'][1]['value']=32
        with self.assertRaisesRegex(UnsupportedSource,'shift count'):
            Evaluator.eval(e,shift)

    def test_scalar_temporary_reference_and_pinned_free(self):
        from cuda_configured_model import Reference, Pointer
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/'main.cu'
            entry.write_text('int take(const int &x){return x;} int main(){return take(3+2);}')
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=['take'],
                preprocessor_sha256=directives_hash(entry.read_bytes()),additional_sources=[],additional_preprocessor_sha256={})
            e=Evaluator(entry,config,'cpu','mem','acc',None)
            e.current_function='main'
            e.call_stack=[]
            self.assertEqual(e.invoke_function('main',[],[]),5)
            with self.assertRaisesRegex(UnsupportedSource,'const temporary'):
                e.store(Reference({'value':3},'value',read_only=True),4)
            e.eval=lambda n:n
            e.host_sizes['pinned']=16
            ptr=Pointer('pinned',0,'char')
            e.call('cudaFreeHost',[ptr])
            self.assertNotIn('pinned',e.host_sizes)
            with self.assertRaisesRegex(UnsupportedSource,'Unknown host allocation'):
                e.call('cudaFreeHost',[ptr])

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
