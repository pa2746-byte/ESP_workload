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

    def test_source_objects_bind_receivers_templates_and_field_references(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = Path(folder)/'main.cu'
            entry.write_text("""
                template<int N> struct Counter {
                    int value;
                    int add(int x) { value += x; return value + N; }
                    static int width() { return N; }
                };
                int main() {
                    Counter<2> a; Counter<3> b;
                    a.value = 5; b.value = 20;
                    return a.add(4) + b.add(1) + Counter<2>::width();
                }
            """)
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                source_objects=True, lexical_scopes=True,
                preprocessor_sha256=directives_hash(entry.read_bytes()),additional_sources=[],additional_preprocessor_sha256={})
            e=Evaluator(entry,config,'cpu','mem','acc',None)
            e.call_stack=[]
            self.assertEqual(e.invoke_function('main',[],[]),37)

    def test_object_methods_preserve_kernel_footprints_and_race_checks(self):
        from cuda_configured_model import analyze_configured
        program = """
            struct Base { int lane; __device__ void init(){ lane = threadIdx.x; } };
            struct Worker : Base {
                __device__ int &slot(int *p){ return p[lane]; }
                __device__ void run(int *src, int *dst){ init(); int &out = slot(dst); int &in = slot(src); out = in; }
            };
            __global__ void work(int *src, int *dst){ Worker worker; worker.run(src,dst); }
            int main(){
                int *a, *b; int host[4];
                cudaMalloc((void**)&a,16); cudaMalloc((void**)&b,16);
                cudaMemcpy(a,host,16,cudaMemcpyHostToDevice);
                work<<<1,4>>>(a,b);
                cudaMemcpy(host,b,16,cudaMemcpyDeviceToHost);
                cudaFree(a); cudaFree(b); return 0;
            }
        """
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/'main.cu'
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                source_objects=True, lexical_scopes=True,host_memory='opaque_values',
                dependency_granularity='byte_regions',
                additional_sources=[],additional_preprocessor_sha256={})
            entry.write_text(program)
            config['preprocessor_sha256']=directives_hash(entry.read_bytes())
            graph, _, details=analyze_configured(entry,config)
            self.assertEqual([n['vol'] for n in graph['nodes']],[16,16,16,16])
            self.assertEqual(graph['edges'],[{'source':0,'target':1},{'source':1,'target':2},{'source':2,'target':3}])
            self.assertEqual(details['launches'][0]['enumerated_threads'],4)
            entry.write_text(program.replace('return p[lane]', 'return p[0]'))
            with self.assertRaisesRegex(UnsupportedSource,'Multiple threads'):
                analyze_configured(entry,config)

    def test_source_object_temporary_functor_and_shared_array_references(self):
        program = """
            struct Step { __device__ void operator()(int &x) const { x += 1; } };
            template<int N> struct Tile {
                int values[N];
                __device__ int &operator[](int i){ return values[i]; }
                template<class F> __device__ void apply(const F &f){ f(values[threadIdx.x]); }
            };
            __global__ void work(int *dst){
                __shared__ Tile<4> tile;
                tile[threadIdx.x] = 0;
                tile.apply(Step());
                dst[threadIdx.x] = tile[threadIdx.x];
            }
            int main(){ int *a; cudaMalloc((void**)&a,16); work<<<1,4>>>(a); cudaFree(a); return 0; }
        """
        from cuda_configured_model import analyze_configured
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/'main.cu';entry.write_text(program)
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                source_objects=True, lexical_scopes=True,shared_memory='opaque_values',
                dependency_granularity='byte_regions',preprocessor_sha256=directives_hash(entry.read_bytes()),
                additional_sources=[],additional_preprocessor_sha256={})
            graph, _, _=analyze_configured(entry,config)
            self.assertEqual([n['vol'] for n in graph['nodes']],[16])
            self.assertEqual(graph['nodes'][0]['src_comp'],'acc')

    def test_source_objects_reject_nontrivial_lifetimes_and_dangling_references(self):
        variants = [
            ('struct A { A(){} }; int main(){ A a; return 0; }','nontrivial'),
            ('struct A { virtual int run(){return 0;} }; int main(){A a; return a.run();}', 'virtual'),
            ('struct A { int &run(){ int local; return local; } }; int main(){A a; a.run()=1; return 0;}', 'local storage'),
            ('struct A {}; void take(A a){} int main(){A a; take(a); return 0;}', 'construction|copy'),
            ('const int &take(const int &x){return x;} int main(){return take(3);}', 'local storage'),
        ]
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/'main.cu'
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=['take'],
                source_objects=True, lexical_scopes=True, additional_sources=[],additional_preprocessor_sha256={})
            for program, error in variants:
                with self.subTest(program=program):
                    entry.write_text(program)
                    config['preprocessor_sha256']=directives_hash(entry.read_bytes())
                    e=Evaluator(entry,config,'cpu','mem','acc',None)
                    e.call_stack=[]
                    with self.assertRaisesRegex(UnsupportedSource,error):
                        e.invoke_function('main',[],[])

    def test_namespace_objects_with_same_spelling_have_distinct_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/'main.cu'
            entry.write_text('namespace left { struct A { int get(){return 2;} }; } '
                             'namespace right { struct A { int get(){return 9;} }; } '
                             'int main(){ left::A a; right::A b; return a.get()+b.get(); }')
            config=json.loads((ROOT/'workloads/rodinia/nn/analysis.json').read_text())
            config.update(parameters={},arguments=[],host_summaries={},analyzed_functions=[],
                source_objects=True, lexical_scopes=True,additional_sources=[],additional_preprocessor_sha256={},
                preprocessor_sha256=directives_hash(entry.read_bytes()))
            e=Evaluator(entry,config,'cpu','mem','acc',None);e.call_stack=[]
            self.assertEqual(e.invoke_function('main',[],[]),11)

    def test_captured_source_runner_checks_all_inputs_before_applying_fixes(self):
        import hashlib
        from unittest.mock import patch
        import tools.analyze_captured_source_case as runner
        sha=lambda data: hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); staged=root/'staged'; reference=root/'workloads/rodinia/dwt2d'
            reference.mkdir(parents=True)
            (root/'workloads/rodinia/captured_scenarios').mkdir()
            (staged/'cuda/dwt2d').mkdir(parents=True);(staged/'scenarios').mkdir()
            source=staged/'cuda/dwt2d/main.cu';source.write_bytes(b'old')
            (reference/'main.cu').write_bytes(b'fixed')
            data=staged/'input';data.write_bytes(b'wrong')
            manifest=dict(capture='fixture',archive_sha256='fixture',entry='cuda/dwt2d/main.cu',arguments=[],
                source_and_input_sha256={'cuda/dwt2d/main.cu':sha(b'old'),'input':sha(b'input')})
            (staged/'scenarios/dwt2d.json').write_text(json.dumps(manifest))
            (root/'workloads/rodinia/captured_scenarios/dwt2d.json').write_text(json.dumps(manifest))
            (reference/'analysis.json').write_text(json.dumps({'arguments':[]}))
            (reference/'local_source_fixes.json').write_text(json.dumps({'upstream_sha256':sha(b'old'),'corrected_sha256':sha(b'fixed')}))
            with patch.object(runner,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'source/input changed'):
                    runner.prepare_case('dwt2d',staged)
                self.assertEqual(source.read_bytes(),b'old')
                data.write_bytes(b'input')
                _, _, evidence=runner.prepare_case('dwt2d',staged)
                self.assertEqual(source.read_bytes(),b'fixed')
                self.assertFalse(evidence['gpu_measurements_attached'])
                runner.prepare_case('dwt2d',staged)  # deterministic re-use

    def test_source_object_fields_do_not_hide_device_pointers_from_host_contracts(self):
        from cuda_object_model import SourceObject
        from cuda_configured_model import Pointer, Reference, Binding
        record=SourceObject('fixture',{})
        record.fields['self']=Binding(Reference({'record':record},'record'))
        record.fields['device']=Pointer('device',0,'int',True)
        e=Evaluator.__new__(Evaluator)
        e.eval=lambda value:value
        e.kernel=False
        e.summaries={'cpu_contract':{}}
        with self.assertRaisesRegex(UnsupportedSource,'cannot receive device pointers'):
            e.call('cpu_contract',[record])

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
