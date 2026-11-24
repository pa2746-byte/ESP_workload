import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from '/Users/priyalagarwal/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/priyalagarwal/Desktop/ESP_workload';
const data=root+'/results/rodinia_trace_comparison';
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#203638',accent='#216C60';
function text(s,t,x,y,w,h,size=25,bold=false,color=ink,font='Arial') {let z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});z.text=t;z.text.style={typeface:font,fontSize:size,bold,color};return z;}
function slide(title,sub,notes=''){let s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,55,35,1170,85,40,true);if(sub)text(s,sub,55,120,1170,65,23,false,'#526865');text(s,String(p.slides.items.length),1190,680,40,25,16);s.speakerNotes.text=notes;return s;}
function box(s,label,x,y,w=240,h=70){const isCopy=/upload|download|host-to-device|device-to-host/i.test(label);let z=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:isCopy?'#FCE3B9':'#EEF4F2',line:{fill:isCopy?'#AE7223':'#9DB5AE',width:1}});z.text=label;z.text.style={typeface:'Arial',fontSize:22,color:ink};return z;}
function edge(s,a,b,side='bottom',to='top'){s.shapes.connect(a,b,{kind:'straight',fromSide:side,toSide:to,line:{fill:accent,width:2},tail:{type:'arrow',width:'med',length:'med'}});}
function draw(s,labels,edges,pos,w=230,h=66){const nodes=labels.map((t,i)=>box(s,t,...pos[i],w,h));for(let [a,b] of edges)edge(s,nodes[a],nodes[b],pos[b][0]>pos[a][0]?'right':'bottom',pos[b][0]>pos[a][0]?'left':'top');}
function code(s,t,x=55,y=195,w=800,h=390){text(s,t,x,y,w,h,22,false,ink,'Courier New');}
const read=async n=>JSON.parse(await fs.readFile(data+'/'+n,'utf8'));
const nn=await read('nn_execution_dag.json'),hs=await read('hotspot3d_execution_dag.json');
const gh=await read('hotspot3d_clang_original.json');
const notesBase='Capture: 20261027T033003Z, Tesla T4. Source revision 4106565d567f39851e0371ab07a4efdeaf3fd3d2. Sources: results/rodinia_trace_comparison/comparison.json and execution_dag JSON files. One profiled run per workload. Synthetic input data. Diagrams are schematic, not time-scaled.';
let s=slide('Rodinia code and GPU execution','Clang dependency graphs compared with captured CUDA operations',notesBase);
text(s,'Nearest Neighbor\nHotSpot3D',55,260,1100,170,52,true);
text(s,'Source analysis and a real Tesla T4 run\nOctober 2026',55,520,1100,90,28);
s=slide('What the graphs represent','We compare operations after grouping each kernel’s logical reads and writes.',notesBase);
text(s,'Clang graph',55,225,530,45,30,true);text(s,'Reads and writes of global buffers\nLogical byte footprints\nBuffer readiness and reuse dependencies',55,290,540,180,28);
text(s,'GPU execution graph',675,225,540,45,30,true);text(s,'Kernel launches and explicit copies\nMeasured event durations\nEdges from the same CUDA stream',675,290,540,180,28);
text(s,'A kernel read/write node does not prove a separate physical memory transfer.',55,555,1140,85,29,true,accent);
s=slide('Three levels of data movement','Each graph exposes a different level of detail.',notesBase);
text(s,'1  Explicit host–GPU copies',55,215,1150,42,30,true,accent);
text(s,'One API copy can move an entire array. Both graphs show these copies.',95,270,1100,65,27);
text(s,'2  Logical accesses inside a kernel',55,365,1150,42,30,true,accent);
text(s,'Clang combines thread accesses into buffer read/write footprints.\nNsight Systems places those accesses inside a single kernel event.',95,420,1100,90,27);
text(s,'3  Physical transactions within the GPU',55,550,1150,42,30,true,accent);
text(s,'Neither graph resolves individual cache or DRAM transactions.',95,605,1100,50,27);
s=slide('One kernel box contains hundreds of accesses','Nearest Neighbor processes 513 records in this capture.',notesBase+' Source: nn_cuda.cu euclid kernel and analysis.json. Each active thread reads one pair of coordinates and writes one float. These are logical element accesses, not physical transaction counts.');
text(s,'What the source implies',55,215,570,45,31,true);
text(s,'513 active threads\n\nEach reads 2 float coordinates\nEach writes 1 float distance\n\n768 threads launch, but the bounds\ncheck excludes 255 from these accesses.',55,280,590,300,27);
text(s,'What each graph shows',710,215,500,45,31,true);
text(s,'Clang: 2 kernel footprint nodes\n4,104 bytes read\n2,052 bytes written\n\nNsight Systems: 1 euclid event\n2.336 µs measured duration',710,280,500,300,27);
text(s,'Aggregation is a modeling choice. We can infer the element accesses from the source.',55,620,1150,65,27,true,accent);
s=slide('Nearest Neighbor: the CUDA code','513 records, each containing two float32 coordinates',notesBase+' Source: workloads/rodinia/nn/nn_cuda.cu. Excerpts below omit allocation and CPU result selection. Kernel expression copied from source, whitespace reformatted.');
code(s,`int globalId = blockDim.x *
    (gridDim.x * blockIdx.y + blockIdx.x)
    + threadIdx.x;
LatLong *latLong = d_locations + globalId;
if (globalId < numRecords) {
    float *dist = d_distances + globalId;
    *dist = (float)sqrt(
      (lat-latLong->lat)*(lat-latLong->lat)
      + (lng-latLong->lng)*(lng-latLong->lng));
}`,55,210,830,340);
text(s,'Input\n513 × 8 = 4,104 bytes\n\nOutput\n513 × 4 = 2,052 bytes\n\nLaunch\n3 blocks × 256 threads',920,210,310,330,25);
text(s,'Host uploads coordinates, launches euclid, then downloads distances.',55,590,1150,65,27,true,accent);
s=slide('Nearest Neighbor: Clang and GPU graphs','Clang aggregates thread accesses. Nsight Systems records the whole kernel.',notesBase+' Sources: nn_clang_original.json, nn_clang_grouped_dag.json, nn_execution_dag.json. Explicit copy byte counts agree.');
text(s,'Clang: 4 nodes, 3 edges',70,190,510,40,29,true);text(s,'GPU: 3 nodes, 2 edges',710,190,510,40,29,true);
draw(s,['t0  Upload coordinates\n4,104 B','t1  euclid reads coordinates\n4,104 logical B','t2  euclid writes distances\n2,052 logical B','t3  Download distances\n2,052 B'],[[0,1],[1,2],[2,3]],[[70,245],[70,345],[70,445],[70,545]],460,75);
draw(s,nn.nodes.map(n=>n.type==='kernel'?`euclid\n${n.duration_us.toFixed(3)} µs`:`${n.label}\n${n.bytes.toLocaleString()} B, ${n.duration_us.toFixed(3)} µs`),[[0,1],[1,2]],[[710,245],[710,395],[710,545]],460,75);
text(s,'t1 and t2 describe one kernel execution.',560,337,135,150,20,false,accent);
s=slide('HotSpot3D: launch loop and buffer reuse','64 × 64 × 3 cells, two iterations, 49,152 bytes per buffer',notesBase+' Source: workloads/rodinia/hotspot3d/opt1.cu, host launch loop. Exact executable statements with line wrapping. Allocations and initial uploads omitted.');
code(s,`for (int i = 0; i < numiter; ++i) {
  hotspotOpt1<<<grid_dim, block_dim>>>(
    p_d, tIn_d, tOut_d, stepDivCap,
    nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
  float *t = tIn_d;
  tIn_d = tOut_d;
  tOut_d = t;
}
cudaDeviceSynchronize();
cudaMemcpy(tOut, tOut_d, s,
           cudaMemcpyDeviceToHost);`,55,205,830,360);
text(s,'Each launch reads power\nand temperature, then\nwrites a temperature buffer.\n\nThe pointer swap changes\nwhich buffer each name\nrefers to.',920,210,305,300,25);
text(s,'After two swaps, tOut_d holds iteration 1’s output.',55,590,1150,65,29,true,accent);
s=slide('HotSpot3D: the original Clang graph','Each iteration gets its own read/write footprints. Individual thread accesses are aggregated.',notesBase+' Source: hotspot3d_clang_original.json. Labels A/B denote physical allocations, rather than changing pointer variable values. A initially holds temperature. B holds iteration 1 output. Kernel 2 overwrites A. Power is P. All nodes have 49,152 logical bytes.');
const labels=['t0 Upload A','t1 Upload P','t2 Read P (iter 1)','t3 Read A (iter 1)','t4 Write B (iter 1)','t5 Read P (iter 2)','t6 Read B (iter 2)','t7 Write A (iter 2)','t8 Download B'];
const pos=[[60,200],[60,340],[330,200],[330,340],[590,270],[590,440],[850,270],[850,440],[850,580]];
draw(s,labels,gh.edges.map(e=>[e.source,e.target]),pos,230,58);
text(s,'A: initial temperature allocation\nB: alternate temperature allocation\nP: power allocation\n\nRepeated stencil reads count unique bytes,\nnot every read instruction. 49,152 B per node.',55,500,560,170,22);
s=slide('HotSpot3D: captured GPU operations','Each kernel contains many memory accesses. This trace shows whole operations.',notesBase+' Source: hotspot3d_execution_dag.json. The trace gives copy directions, but naming uploads temperature and power uses source order.');
const hsLabels=['Upload temperature','Upload power','hotspotOpt1: iteration 1','hotspotOpt1: iteration 2','Download temperature'];
draw(s,hs.nodes.map((n,i)=>`${hsLabels[i]}\n${n.duration_us.toFixed(3)} µs${n.type==='copy'?', 49,152 B':''}`),[[0,1],[1,2],[2,3],[3,4]],[[60,200],[60,290],[60,380],[60,470],[60,560]],480,65);
text(s,'Measured structure matches Clang',650,220,560,70,31,true);
text(s,'2 kernel launches\n\nGrid: 1 × 16 × 1\nBlock: 64 × 4 × 1\n\n2 uploads and 1 download\n147,456 explicit copy bytes',650,325,560,280,28);
s=slide('Execution order and data dependency differ','The final HotSpot3D download illustrates why both graphs matter.',notesBase+' Source: hotspot3d_clang_grouped_dag.json edges include 2→4 and exclude 3→4. GPU stream chain includes 3→4. Full grouped source DAG also includes uploads and reuse dependencies. Diagram shows only the kernel/download portion. Trace virtualAddress fields are null, so buffer identity comes from source.');
text(s,'Clang: producer relationship',55,200,560,45,29,true);text(s,'GPU: stream order',690,200,535,45,29,true);
draw(s,['Iteration 1 writes B','Iteration 2 writes A','Download B'],[[0,1],[0,2]],[[55,275],[55,410],[350,545]],260,70);
draw(s,['Iteration 1','Iteration 2','Download'],[[0,1],[1,2]],[[790,275],[790,410],[790,545]],350,70);
text(s,'Source identifies B as the downloaded buffer. The trace alone cannot verify its identity.',55,650,1150,40,23,true,accent);
s=slide('What this comparison validates','Both workloads match the expected operation structure.',notesBase);
text(s,'Confirmed',55,220,520,45,32,true,accent);text(s,'Kernel names and launch counts\nGrid and block dimensions\nExplicit copy directions and sizes\nObserved order respects source edges',55,295,560,230,28);
text(s,'Still unverified',690,220,530,45,32,true);text(s,'Kernel logical memory footprints\nPhysical DRAM traffic and caching\nNumerical correctness\nSimulator performance predictions',690,295,540,230,28);
text(s,'Use Clang for buffer dependencies and the trace for measured execution order and timing.',55,575,1150,80,28,true);
for(const s of p.slides.items){if(s.shapes.items.some(z=>String(z.text).match(/Upload|Download|Host-to-Device|Device-to-Host/))){text(s,'Amber = explicit host–GPU copy',55,680,700,25,18,true,'#8A5519');}}
for(let i=0;i<p.slides.items.length;i++){await fs.writeFile(`${root}/.ppt-build/slide-${i+1}.png`,new Uint8Array(await (await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));}
const draft=root+'/.ppt-build/draft.pptx';await (await PresentationFile.exportPptx(p)).save(draft);
const skill='/Users/priyalagarwal/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
console.log(await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:root+'/outputs/Rodinia_Clang_vs_GPU_colored_transfers.pptx',pythonExecutable:'/Users/priyalagarwal/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Arial','Courier New']},verifyArtifactToolImport:true,receiptPath:root+'/.ppt-build/validation-colored.json'}));
