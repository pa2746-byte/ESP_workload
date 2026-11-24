"""Source execution semantics, footprint boundaries, and failure gates."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from cuda_native_source_model import instrument,checked_run,graph_from_source_events
from cuda_source_model import UnsupportedSource


class NativeSourceModelTests(unittest.TestCase):
    def execute(self,program,shared_memory="reject"):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'main.cu';source.write_text(program)
            transformed,_=instrument(source,root/'build',shared_memory=shared_memory)
            checked_run(['clang++','-std=c++17','-O0','-ffp-contract=off','-fsanitize=address,undefined',str(transformed),'-o',str(root/'run')])
            event_file=root/'events.jsonl'
            result=subprocess.run([str(root/'run')],capture_output=True,text=True,
                env={**os.environ,'ESP_SOURCE_EVENTS':str(event_file),'ASAN_OPTIONS':'detect_leaks=0:detect_stack_use_after_return=0','UBSAN_OPTIONS':'halt_on_error=1','ESP_SOURCE_SHARED':shared_memory})
            events=[json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
            return result,events

    def test_source_data_decides_addresses_and_host_feedback_decides_launches(self):
        result,events=self.execute('''
            __global__ void scatter(int *index,int *out){int i=threadIdx.x;out[index[i]]=index[i]+1;}
            __global__ void increment(int *out){out[threadIdx.x]+=1;}
            int main(){
                int values[2]={1,0},*index,*out;
                cudaMalloc(&index,8);cudaMalloc(&out,8);
                cudaMemcpy(index,values,8,cudaMemcpyHostToDevice);
                scatter<<<1,2>>>(index,out);
                cudaMemcpy(values,out,8,cudaMemcpyDeviceToHost);
                if(values[0]==1)increment<<<1,2>>>(out);
                cudaMemcpy(values,out,8,cudaMemcpyDeviceToHost);
                cudaFree(index);cudaFree(out);
                return values[0]==2 && values[1]==3 ? 0:1;
            }
        ''')
        self.assertEqual(result.returncode,0,result.stderr)
        graph,nodes,launches,_=graph_from_source_events(events)
        self.assertEqual([l['kernel'] for l in launches],['scatter','increment'])
        self.assertEqual([n['vol'] for n in graph['nodes']],[8]*7)
        self.assertEqual([n['stage'] for n in nodes],['upload','scatter:read','scatter:write','download','increment:read','increment:write','download'])
        self.assertIn({'source':3,'target':4},graph['edges'])

    def test_aggregate_copy_and_argument_side_effects_execute_once(self):
        result,events=self.execute('''
            struct Pair {int x,y;};
            __global__ void copy(Pair* in,Pair* out,int offset){int i=threadIdx.x;Pair value=in[i];value.x+=offset;out[i]=value;}
            int main(){Pair h[2]={{1,2},{3,4}},*in,*out;int count=0;
                cudaMalloc(&in,sizeof(h));cudaMalloc(&out,sizeof(h));
                cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice);
                copy<<<1,2>>>(in,out,count++);
                cudaMemcpy(h,out,sizeof(h),cudaMemcpyDeviceToHost);
                cudaFree(in);cudaFree(out);return count==1 && h[1].x==3 ? 0:1;}
        ''')
        self.assertEqual(result.returncode,0,result.stderr)
        graph,_,_,_=graph_from_source_events(events)
        self.assertEqual([n['vol'] for n in graph['nodes']],[16]*4)

    def test_same_thread_intermediate_reads_are_not_fabricated_external_inputs(self):
        result,events=self.execute("""
            __global__ void work(int *out){int i=threadIdx.x;out[i]=i;out[i]+=1;}
            int main(){int *out,host[2];cudaMalloc(&out,8);work<<<1,2>>>(out);
                cudaMemcpy(host,out,8,cudaMemcpyDeviceToHost);cudaFree(out);return host[1]==2?0:1;}
        """)
        self.assertEqual(result.returncode,0,result.stderr)
        graph,nodes,launches,_=graph_from_source_events(events)
        self.assertEqual([n['stage'] for n in nodes],['work:write','download'])
        access=launches[0]['global_access_footprints'][0]
        self.assertEqual(access['reads'],[[0,8]])
        self.assertEqual(access['external_reads'],[])

    def test_races_uninitialized_reads_and_host_arguments_fail(self):
        variants=[
            ('a[0]=threadIdx.x;','cudaMemset(a,0,16);','', 'Multiple threads write'),
            ('a[threadIdx.x]+=1;','','', 'uninitialized'),
            ('a[4]=1;','cudaMemset(a,0,16);','', 'exceeds|unmodeled'),
            ('a[threadIdx.x]=1;','','a=host;', 'not device storage'),
        ]
        for body,initialize,override,error in variants:
            with self.subTest(error=error):
                result,_=self.execute('__global__ void work(int *a){'+body+'} int main(){int *a;int host[4];cudaMalloc(&a,16);'+initialize+override+'work<<<1,4>>>(a);return 0;}')
                self.assertNotEqual(result.returncode,0)
                self.assertRegex(result.stderr,error)

    def test_interval_graph_handles_large_regions_and_partial_hazards(self):
        size=1<<30
        events=[{'event':'allocate','buffer':0,'size':size},
                {'event':'memset','buffer':0,'offset':0,'size':size},
                {'event':'kernel','name':'ends','grid':[1,1,1],'block':[1,1,1],
                 'accesses':[{'buffer':0,'external_reads':[[0,4],[size-4,size]],'reads':[[0,4],[size-4,size]],'writes':[[100,104]]}]},
                {'event':'copy','source':0,'destination':-1,'source_offset':size-4,'destination_offset':0,'size':4}]
        graph,_,_,_=graph_from_source_events(events)
        self.assertEqual([n['vol'] for n in graph['nodes']],[size,8,4,4])
        self.assertEqual({(e['source'],e['target']) for e in graph['edges']},{(0,1),(0,2),(1,2),(0,3)})
        from source_region_graph import SourceRegionGraph
        model=SourceRegionGraph();model.allocate('x',8)
        def transfer(ranges,writing):
            model.operation=model.order.schedule([('x','write' if writing else 'read')],'default')
            return model.transfer('x',ranges,'acc' if writing else 'mem','mem' if writing else 'acc','fixture',writing)
        transfer([[0,8]],True);transfer([[2,6]],False);transfer([[4,8]],True);transfer([[0,8]],False)
        self.assertEqual(model.edges,{(0,1),(0,2),(1,2),(0,3),(2,3)})
        fresh=SourceRegionGraph();fresh.allocate('x',8);fresh.operation=fresh.order.schedule([('x','read')],'default')
        with self.assertRaisesRegex(UnsupportedSource,'uninitialized'):
            fresh.transfer('x',[[0,4]],'mem','acc','fixture',False)

    def test_checked_shared_values_barriers_dynamic_storage_and_atomics(self):
        result,events=self.execute("""
            __global__ void work(int *src,int *dst){
                extern __shared__ int tile[];
                __shared__ int sum;
                int i=threadIdx.x;tile[i]=src[i];if(i==0)sum=0;
                __syncthreads();atomicAdd(&sum,tile[i]);__syncthreads();dst[i]=sum;
            }
            int main(){int input[4]={1,2,3,4},output[4],*src,*dst;
                cudaMalloc(&src,16);cudaMalloc(&dst,16);cudaMemcpy(src,input,16,cudaMemcpyHostToDevice);
                work<<<1,4,16>>>(src,dst);cudaMemcpy(output,dst,16,cudaMemcpyDeviceToHost);
                cudaFree(src);cudaFree(dst);return output[3]==10?0:1;}
        """,'checked_blocks')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([e['checked_block_barriers'] for e in events if e['event']=='kernel'],[2])
        graph,_,_,_=graph_from_source_events(events)
        self.assertEqual([n['vol'] for n in graph['nodes']],[16]*4)

    def test_checked_shared_errors_and_nonuniform_barriers_fail(self):
        variants=[
            ('__shared__ int a;dst[threadIdx.x]=a;', 'uninitialized'),
            ('__shared__ int a;a=threadIdx.x;__syncthreads();dst[threadIdx.x]=a;', 'Multiple threads'),
            ('if(threadIdx.x==0)__syncthreads();dst[threadIdx.x]=1;', 'Nonuniform|different block'),
        ]
        for body,error in variants:
            with self.subTest(error=error):
                result,_=self.execute('__global__ void work(int *dst){'+body+'} int main(){int *dst;cudaMalloc(&dst,16);work<<<1,4>>>(dst);return 0;}', 'checked_blocks')
                self.assertNotEqual(result.returncode,0)
                self.assertRegex(result.stderr,error)

    def test_unused_shared_declarations_do_not_imply_shared_value_support(self):
        result,events=self.execute('__global__ void work(int *dst){__shared__ int unused[4];dst[threadIdx.x]=1;} int main(){int *dst;cudaMalloc(&dst,16);work<<<1,4>>>(dst);cudaFree(dst);return 0;}', 'reject_accesses')
        self.assertEqual(result.returncode,0,result.stderr)
        graph,_,_,_=graph_from_source_events(events)
        self.assertEqual([n['vol'] for n in graph['nodes']],[16])

    def test_shared_state_is_explicitly_unsupported_in_native_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'main.cu'
            path.write_text('__global__ void work(int *a){__shared__ int tile[4];tile[threadIdx.x]=1;} int main(){return 0;}')
            with self.assertRaisesRegex(UnsupportedSource,'shared memory'):
                instrument(path,Path(directory)/'build')


if __name__=='__main__':unittest.main()
