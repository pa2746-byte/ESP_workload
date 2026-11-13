"""Optional bounded integer-state evaluation for fixed-input CUDA applications.

Known integer bytes, and explicitly enabled floating bytes, are propagated rather than guessed. Threads read the pre-launch snapshot
plus their own writes; the configured evaluator still rejects inter-thread races.
This is source evaluation, not a GPU execution or a timing model.
"""
import copy
import struct
import math
import hashlib
import re
from pathlib import Path
from cuda_configured_model import (Evaluator, Pointer, Reference, HostStream, Unknown,
    UNKNOWN, SharedView, LoopBreak, ProgramExit, FunctionReturn, body_of)
from cuda_source_model import UnsupportedSource, children, walk, ref


class UnknownBranch(Exception):
    pass


class IntegerEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        self.known_host = {}
        self.known_device = {}
        self.pending_device = {}
        self.last_thread = None
        self.exploring = False
        self.pointer_slots = {}
        self.random_index=0
        self.host_barriers=set()
        super().__init__(*args, **kwargs)
        for relative,digest in self.config.get('source_hashes',{}).items():
            file=(self.source_directory/relative).resolve()
            if not file.is_relative_to(self.source_directory) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise UnsupportedSource('Configured preparation source changed')
        for name in self.config.get('mutable_file_globals', []):
            if self.global_types.get(name) not in {'FILE *', 'struct __sFILE *'}:
                raise UnsupportedSource('File global must be a FILE pointer')
            self.mutable_globals.add(name)
        for name in self.config.get('mutable_pointer_globals', []):
            if not self.global_types.get(name,'').endswith('*'):raise UnsupportedSource('Pointer global must be declared')
            self.mutable_globals.add(name)
        for name in self.mutable_globals:
            self.globals.setdefault(name, UNKNOWN)
        self.types.update({'bool': (1, 1, {}), 'long': (8, 8, {}),
                           'long long': (8, 8, {}), 'unsigned long': (8, 8, {})})

    def layout(self,typ):
        if typ.endswith('*'):return (8,8,{})
        return super().layout(typ)

    def eval(self,n):
        if (not self.kernel and n["kind"] == "CXXOperatorCallExpr"
                and n.get("type", {}).get("qualType") == "cudaChannelFormatDesc"
                and ref(children(n)[0]) == "operator="):
            cs = children(n)
            value = copy.deepcopy(self.eval(cs[2]))
            self.store(self.location(cs[1]), value)
            return value
        if n['kind']=='UnaryOperator' and n.get('opcode') in {'++','--'}:
            location=self.location(children(n)[0]);value=self.read(location)
            if isinstance(value,Pointer):
                result=self.offset(value,1 if n['opcode']=='++' else -1);self.store(location,result)
                return value if n.get('isPostfix') else result
            if type(value) is not int:raise UnsupportedSource('Increment requires a known integer or pointer')
            result=value+(1 if n['opcode']=='++' else -1);self.store(location,result)
            return value if n.get('isPostfix') else result
        if n['kind']=='StmtExpr' and not self.kernel:
            statements=children(children(n)[0])
            with self.local_scope(statements):
                for statement in statements[:-1]:self.statement(statement)
                return self.eval(statements[-1])

        if n['kind']=='ImplicitCastExpr' and n.get('castKind')=='ArrayToPointerDecay':
            if children(n)[0]['kind']=='StringLiteral':return super().eval(n)
            location=self.location(children(n)[0])
            value=self.read(location) if isinstance(location,Reference) else location
            if isinstance(value,Pointer) and '[' in value.base:
                return Pointer(value.buffer,value.offset,value.base.split('[')[0].strip(),value.device)
            return value
        if n['kind'] in {'CStyleCastExpr','CXXStaticCastExpr'} and self.config.get('symbolic_pointer_integers'):
            typ=self.clean(n['type']['qualType']);cs=children(n)
            if typ=='long':
                value=self.eval(cs[0])
                if isinstance(value,Pointer):return Pointer(value.buffer,value.offset,'char',value.device)
                if type(value) in (int,float):return int(value)
                if isinstance(value,Unknown):return value
                raise UnsupportedSource('Unsupported symbolic address cast')
        if n['kind']=='BinaryOperator' and n['opcode']=='-' and self.config.get('symbolic_pointer_integers'):
            a,b=[self.eval(c) for c in children(n)]
            if isinstance(a,Pointer) and isinstance(b,Pointer):
                if a.buffer!=b.buffer or a.base!='char' or b.base!='char':raise UnsupportedSource('Pointer difference requires the same byte allocation')
                return a.offset-b.offset
            if isinstance(a,Unknown) or isinstance(b,Unknown):return Unknown(set().union(*(v.dependencies for v in (a,b) if isinstance(v,Unknown))))
            if isinstance(a,Pointer) and type(b) is int:return self.offset(a,-b)
            if type(a) in (int,float) and type(b) in (int,float):return a-b
            raise UnsupportedSource('Unsupported symbolic pointer subtraction')

        if n['kind']=='CXXConstructExpr':
            typ=n['type']['qualType'];cs=children(n)
            if typ.startswith('texture<'):
                return self.eval(cs[0]) if cs else {}
            if typ=='cudaChannelFormatDesc':return {}
        value=super().eval(n)
        if self.config.get('known_floating_values') and n.get('type',{}).get('qualType')=='float' and type(value) in (int,float):
            value=struct.unpack('<f',struct.pack('<f',value))[0]
            if not math.isfinite(value):raise UnsupportedSource('Nonfinite float expression')
        return value

    def integer_type(self, base):
        return base in {'bool', 'char', 'int', 'unsigned int', 'long', 'long long', 'unsigned long'}

    def known_value(self, pointer, storage):
        floating = pointer.base in {'float','double'} and self.config.get('known_floating_values')
        if not self.integer_type(pointer.base) and not floating: return UNKNOWN
        size = self.layout(pointer.base)[0]
        data = [storage.get((pointer.buffer, pointer.offset+i)) for i in range(size)]
        if any(v is None for v in data): return UNKNOWN
        if floating:
            result=struct.unpack('<f' if pointer.base=='float' else '<d',bytes(data))[0]
            if not math.isfinite(result):raise UnsupportedSource('Nonfinite floating state')
            return result
        result = int.from_bytes(bytes(data), 'little', signed=pointer.base in {'char','int','long','long long'})
        return bool(result) if pointer.base == 'bool' else result

    def put_value(self, pointer, value, storage):
        size = self.layout(pointer.base)[0]
        for i in range(size): storage.pop((pointer.buffer,pointer.offset+i),None)
        if pointer.base in {'float','double'} and self.config.get('known_floating_values') and type(value) in (int,float):
            if not math.isfinite(value):raise UnsupportedSource('Nonfinite floating state')
            data=struct.pack('<f' if pointer.base=='float' else '<d',value)
            storage.update({(pointer.buffer,pointer.offset+i):v for i,v in enumerate(data)});return
        if not self.integer_type(pointer.base) or type(value) not in (int,bool): return
        try:
            data = int(value).to_bytes(size, 'little', signed=pointer.base in {'char','int','long','long long'})
        except OverflowError:
            raise UnsupportedSource('Integer state exceeds the source type range')
        storage.update({(pointer.buffer,pointer.offset+i):v for i,v in enumerate(data)})

    def thread_state(self):
        if not self.kernel: return
        key = tuple(self.env[k][a] for k in ('blockIdx','threadIdx') for a in ('x','y','z'))
        if key != self.last_thread:
            self.thread_values = dict(self.known_device)
            self.last_thread = key

    def location(self, n):
        if n['kind']=='UnaryOperator' and n['opcode']=='*' and not self.kernel:
            value=self.eval(children(n)[0])
            if isinstance(value,Reference):return value

        if n['kind']=='ArraySubscriptExpr' and not self.kernel and self.config.get('opaque_host_addresses'):
            cs=children(n);value=self.eval(cs[0]);index=self.eval(cs[1])
            if isinstance(value,Pointer) and not value.device and isinstance(index,Unknown):
                return Pointer(value.buffer,UNKNOWN,value.base)
        return super().location(n)

    def read(self, location):
        if isinstance(location,Pointer) and location.base.endswith('*') and not location.device:
            if location.offset<0 or location.offset+8>self.host_sizes[location.buffer]:raise UnsupportedSource('Pointer table read exceeds allocation')
            return self.pointer_slots.get((location.buffer,location.offset),UNKNOWN)

        if isinstance(location,Pointer) and not location.device and isinstance(location.offset,Unknown):
            return Unknown(self.host_dependencies.get(location.buffer,()))
        value = super().read(location)
        if isinstance(location, Pointer) and (self.integer_type(location.base) or self.config.get('known_floating_values') and location.base in {'float','double'}):
            self.thread_state()
            known = self.known_value(location, self.thread_values if location.device else self.known_host)
            if not isinstance(known, Unknown): return known
        return value

    def transfer(self,buffer,size,src,dst,stage,parents):
        result=super().transfer(buffer,size,src,dst,stage,set(parents)|self.host_barriers)
        if self.host_barriers:self.metadata[-1]['host_control_dependencies']=sorted(self.host_barriers)
        if stage=='download':self.host_barriers.add(result)
        return result

    def mark_host_output(self,value,dependencies):
        super().mark_host_output(value,dependencies)
        if isinstance(value,Pointer):
            for key in list(self.known_host):
                if key[0]==value.buffer and key[1]>=value.offset:del self.known_host[key]

    def store(self, location, value):
        if isinstance(location,Reference) and location.scalar_type=='float' and self.config.get('known_floating_values') and type(value) in (int,float):
            value=struct.unpack('<f',struct.pack('<f',value))[0]

        if isinstance(location,Reference) and location.owner is getattr(self,'globals',None) and isinstance(value,Pointer):
            if self.kernel or location.key not in self.config.get('mutable_pointer_globals',[]):raise UnsupportedSource('Unconfigured global pointer mutation')
            location.owner[location.key]=value;return
        if isinstance(location,Pointer) and location.base.endswith('*') and not location.device:
            if location.offset<0 or location.offset+8>self.host_sizes[location.buffer] or not isinstance(value,(Pointer,Unknown)) and value is not None:raise UnsupportedSource('Invalid pointer table write')
            self.pointer_slots[location.buffer,location.offset]=value;return

        if isinstance(location,Pointer) and not location.device and isinstance(location.offset,Unknown):
            if not isinstance(value,(int,float,bool,Unknown)):raise UnsupportedSource('Opaque host write must be numeric')
            for key in list(self.known_host):
                if key[0]==location.buffer:del self.known_host[key]
            if isinstance(value,Unknown):self.host_dependencies.setdefault(location.buffer,set()).update(value.dependencies)
            return

        if isinstance(location,SharedView) and not location.shape and isinstance(value,Pointer):
            if location.base!=value.base or not value.device or not self.layout(value.base)[2]:
                raise UnsupportedSource('Shared record assignment requires matching device record')
            for offset,base in self.layout(value.base)[2].values():
                self.read(Pointer(value.buffer,value.offset+offset,base,True))
            return

        if isinstance(location,Reference) and location.owner is getattr(self,'globals',None) and isinstance(value,HostStream):
            if self.kernel or location.key not in self.mutable_globals:
                raise UnsupportedSource('Unconfigured file-handle global')
            location.owner[location.key]=value
            return
        super().store(location,value)
        if isinstance(location,Pointer):
            self.thread_state()
            if location.device:
                self.put_value(location,value,self.thread_values)
                size=self.layout(location.base)[0]
                for i in range(size):
                    key=(location.buffer,location.offset+i)
                    self.pending_device[key]=self.thread_values.get(key)
            else: self.put_value(location,value,self.known_host)

    def launch(self,n):
        self.last_thread=None
        self.pending_device={}
        result=super().launch(n)
        for key,value in self.pending_device.items():
            if value is None:self.known_device.pop(key,None)
            else:self.known_device[key]=value
        self.launches[-1]['integer_state_evaluated']=True
        return result

    def truth(self, value):
        if isinstance(value,Unknown) and self.kernel and self.exploring:
            i=self.choice_index;self.choice_index+=1
            if i>=len(self.choices):raise UnknownBranch()
            return self.choices[i]

        if isinstance(value,HostStream): return True
        return super().truth(value)

    def statement(self,n):
        if self.kernel and not self.exploring and self.config.get('enumerate_unknown_paths') and n['kind']=='CompoundStmt':
            initial={k:copy.deepcopy(getattr(self,k)) for k in ('env','accesses','thread_reads','thread_writes','thread_access_events','thread_barriers','pending_device')}
            queue=[()];results=[];attempts=0
            self.exploring=True
            try:
                while queue:
                    choices=queue.pop();attempts+=1
                    if attempts>63:raise UnsupportedSource('Unknown-path exploration exceeds 63 states per thread')
                    for k,v in initial.items():setattr(self,k,copy.deepcopy(v))
                    self.last_thread=None;self.choices=choices;self.choice_index=0
                    try:
                        super().statement(n)
                    except UnknownBranch:
                        queue.extend((choices+(False,),choices+(True,)))
                        continue
                    except FunctionReturn:pass
                    results.append({k:copy.deepcopy(getattr(self,k)) for k in initial if k!='env'})
                for other in results[1:]:
                    if any(other[k]!=results[0][k] for k in ('accesses','thread_reads','thread_writes','thread_access_events','thread_barriers')):
                        raise UnsupportedSource('Unknown branch changes kernel memory footprint or barriers')
                merged=results[0]
                for key,value in list(merged['pending_device'].items()):
                    if any(r['pending_device'].get(key)!=value for r in results[1:]):merged['pending_device'][key]=None
                for k,v in merged.items():setattr(self,k,v)
            finally:self.exploring=False
            return
        if n['kind']=='IfStmt' and not self.kernel and self.config.get('opaque_host_addresses'):
            cs=children(n)
            if n.get('hasInit') or n.get('hasVar') or len(cs) not in (2, 3):
                raise UnsupportedSource('Unsupported conditional declaration')
            condition=self.eval(cs[0])
            if isinstance(condition,Unknown):
                allowed={'CompoundStmt','BinaryOperator','CompoundAssignOperator','ArraySubscriptExpr','DeclRefExpr','ImplicitCastExpr','CStyleCastExpr','IntegerLiteral','FloatingLiteral','ParenExpr','UnaryOperator','MemberExpr'}
                if any(x['kind'] not in allowed or x.get('opcode') in {'++','--'} or (x['kind']=='DeclRefExpr' and x['referencedDecl'].get('name') in self.mutable_globals) for branch in cs[1:] for x in walk(branch)):
                    raise UnsupportedSource('Opaque host branch must contain only numeric assignments')
                before={k:copy.deepcopy(getattr(self,k)) for k in ('env','known_host','host_dependencies')}
                states=[]
                for branch in (cs[1],cs[2] if len(cs)==3 else None):
                    for k,v in before.items():setattr(self,k,copy.deepcopy(v))
                    if branch is not None:super().statement(branch)
                    states.append({k:copy.deepcopy(getattr(self,k)) for k in before})
                for name in before['env']:
                    a,b=states[0]['env'][name],states[1]['env'][name]
                    self.env[name]=a if a==b else Unknown(self.host_provenance(a)|self.host_provenance(b))
                self.known_host={k:v for k,v in states[0]['known_host'].items() if states[1]['known_host'].get(k)==v}
                for buffer,deps in states[0]['host_dependencies'].items():self.host_dependencies.setdefault(buffer,set()).update(deps)
                return

            branch = cs[1] if self.truth(condition) else cs[2] if len(cs) == 3 else None
            if branch is not None:
                with self.local_scope([branch]):self.statement(branch)
            return

        if n['kind']=='DoStmt':
            body,condition=children(n)
            count=0
            while True:
                count+=1;self.total_loop_iterations+=1
                if count>self.loop_limit or self.total_loop_iterations>1000000:
                    raise UnsupportedSource('Loop iteration budget exceeded')
                try:
                    with self.local_scope([body]):self.statement(body)
                except LoopBreak:break
                if not self.truth(self.eval(condition)):break
            return
        return super().statement(n)

    def call(self,name,expressions,site=None):
        modeled={'rand','getchar','feof','fseek','ftell','rewind','fread','memset','open','read','close','getopt','calloc','memcpy','strcmp','sscanf','pow','fabs','ceil','time','srand','fopen','fscanf','fclose'}
        if name in modeled and name in self.functions and name not in self.summaries and name not in self.analyzed_functions:
            raise UnsupportedSource('Library model cannot replace a source-defined function')
        if name=='rand' and 'random_values' in self.config:
            if expressions or name in self.functions:raise UnsupportedSource('Invalid modeled random call')
            values=self.config['random_values']
            if self.random_index>=len(values):raise UnsupportedSource('Configured random input exhausted')
            value=values[self.random_index];self.random_index+=1
            if type(value) is not int or not 0<=value<=2147483647:raise UnsupportedSource('Invalid configured random input')
            return value
        if name in self.summaries and 'global_byte_buffer' in self.summaries[name]:
            args=[self.eval(n) for n in expressions];spec=self.summaries[name]
            for key,v in spec.get('requires',{}).items():
                if args[int(key)]!=v:raise UnsupportedSource('Host preparation arguments changed')
            if any(self.globals.get(key)!=value for key,value in spec.get('requires_globals',{}).items()):raise UnsupportedSource('Host preparation globals changed')
            digest=hashlib.sha256(self.source).hexdigest()
            if digest!=self.config.get('source_sha256'):raise UnsupportedSource('Host preparation source changed')
            data=bytes.fromhex(spec['global_byte_buffer']['hex']);key='host_contract_'+name
            if key in self.host_sizes or len(data)>self.region_limit:raise UnsupportedSource('Invalid host preparation allocation')
            self.host_sizes[key]=len(data)
            for start,end in spec['global_byte_buffer']['known_ranges']:
                if not 0<=start<=end<=len(data):raise UnsupportedSource('Invalid known byte interval')
                self.known_host.update({(key,i):data[i] for i in range(start,end)})
            for glob,alias in spec['global_byte_buffer']['aliases'].items():
                if glob not in self.config.get('mutable_pointer_globals',[]):raise UnsupportedSource('Unconfigured contract global')
                self.globals[glob]=Pointer(key,alias['offset'],alias['base'])
            self.used_summaries.add(name);return spec['return']
        if name in {'feof','fseek','ftell','rewind','fread','getchar','memset'} and not self.kernel:
            args=[self.eval(n) for n in expressions]
            if name=='getchar':
                if self.config.get('stdin_text')!='\n':raise UnsupportedSource('stdin must be configured')
                return 10
            if name=='memset':
                ptr,value,size=args
                if not isinstance(ptr,Pointer) or ptr.device or ptr.offset<0 or ptr.offset+size>self.host_sizes[ptr.buffer] or type(value) is not int:raise UnsupportedSource('Invalid host memset')
                self.known_host.update({(ptr.buffer,ptr.offset+i):value&255 for i in range(size)});return ptr
            stream=args[-1] if name=='fread' else args[0]
            if not isinstance(stream,HostStream):raise UnsupportedSource('Invalid file handle')
            if name=='feof':return bool(getattr(stream,'eof',False))
            if name=='fseek':
                if args[1:]!=[0,2]:raise UnsupportedSource('Only seek-to-end is supported')
                return 0
            if name=='ftell':return len(stream.raw)
            if name=='rewind':stream.position=0;stream.eof=False;return 0
            ptr,width,count,_=args;size=width*count
            if not isinstance(ptr,Pointer) or ptr.device or size>len(stream.raw) or ptr.offset+size>self.host_sizes[ptr.buffer]:raise UnsupportedSource('Invalid fread buffer')
            self.known_host.update({(ptr.buffer,ptr.offset+i):v for i,v in enumerate(stream.raw[:size])});return count

        if name=='cudaMemset':
            args=[self.eval(n) for n in expressions];result=super().call(name,expressions,site)
            ptr,value,size=args;self.known_device.update({(ptr.buffer,ptr.offset+i):value&255 for i in range(size)});return result
        if name=='__builtin_object_size':
            ptr=self.eval(expressions[0])
            return self.host_sizes[ptr.buffer]-ptr.offset if isinstance(ptr,Pointer) and not ptr.device else -1
        if name=='__builtin___memset_chk':
            self.eval(expressions[3]);return self.call('memset',expressions[:3],site)
        if name=='__builtin___memcpy_chk':
            self.eval(expressions[3])
            return self.call('memcpy',expressions[:3],site)

        if name in {'open','read','close','getopt','cudaCreateChannelDesc<float>','cudaCreateChannelDesc','cudaBindTexture','tex1Dfetch','cudaMemcpyToSymbol','calloc','memcpy'}:
            # These modeled library calls evaluate only side-effect-free arguments.
            if any(x['kind'] in {'CallExpr','CUDAKernelCallExpr','CompoundAssignOperator'} or x.get('opcode') in {'++','--','='} for n in expressions for x in walk(n)):
                raise UnsupportedSource('Modeled library arguments must have no side effects')
            args=[self.eval(n) for n in expressions]
            if name=='getopt':
                argc,argv,options=args;i=self.cli_state['optind']
                if i>=argc:return -1
                key=argv[i][1:]
                if len(key)!=1 or key not in options:raise UnsupportedSource('Unsupported command-line option')
                if options[options.index(key)+1:options.index(key)+2]==':':
                    if i+1>=argc:raise UnsupportedSource('Missing option argument')
                    self.cli_state['optarg']=argv[i+1];i+=1
                self.cli_state['optind']=i+1;return ord(key)
            if name=='open':
                filename=args[0];path=(self.source_directory/filename).resolve()
                if path.parent!=self.source_directory or args[1]!=0:raise UnsupportedSource('Only sibling read-only input is supported')
                data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                if self.config.get('input_files',{}).get(filename)!=digest:raise UnsupportedSource('Configured input file changed')
                self.input_file_hashes[filename]=digest;self.binary_input=[data,0];return 3
            if name=='close':return 0
            if name=='read':
                fd,dst,size=args
                if fd!=3:raise UnsupportedSource('Invalid input descriptor')
                data,pos=self.binary_input;chunk=data[pos:pos+size]
                if len(chunk)!=size:raise UnsupportedSource('Binary input exhausted')
                self.binary_input[1]+=size
                if isinstance(dst,Reference):
                    if dst.scalar_type!='int' or size!=4:raise UnsupportedSource('Only int header fields are supported')
                    self.store(dst,int.from_bytes(chunk,'little',signed=True))
                elif isinstance(dst,Pointer) and not dst.device and dst.offset+size<=self.host_sizes[dst.buffer]:
                    self.known_host.update({(dst.buffer,dst.offset+i):v for i,v in enumerate(chunk)})
                else:raise UnsupportedSource('Invalid binary destination')
                return size
            if name.startswith('cudaCreateChannelDesc'):return {}
            if name=='cudaBindTexture':
                _,target,ptr,desc,size=args
                if not isinstance(target,Reference) or not isinstance(ptr,Pointer) or not ptr.device or size>self.allocations[ptr.buffer]:raise UnsupportedSource('Invalid texture binding')
                self.read(target)['binding']=(ptr,size);return 0
            if name=='tex1Dfetch':
                texture,index=args
                if not self.kernel or not isinstance(texture,dict) or 'binding' not in texture or type(index) is not int:raise UnsupportedSource('Unresolved texture access')
                ptr,size=texture['binding']
                if not 0<=index*4<size:raise UnsupportedSource('Texture access exceeds binding')
                return self.read(Pointer(ptr.buffer,ptr.offset+index*4,'float',True))
            if name=='cudaMemcpyToSymbol' and isinstance(args[0],str):
                symbol,src,size,offset,direction=args
                if symbol not in self.constant_buffers or offset!=0 or direction!='cudaMemcpyHostToDevice' or size>self.allocations[symbol] or not isinstance(src,Pointer) or src.device or src.offset+size>self.host_sizes[src.buffer]:raise UnsupportedSource('Invalid legacy constant copy')
                self.operation=self.order.schedule([(symbol,'write')],'default')
                self.write_buffer(symbol,size,self.cpu,'upload_constant',self.host_provenance(src))
                for i in range(size):
                    value=self.known_host.get((src.buffer,src.offset+i))
                    if value is None:self.known_device.pop((symbol,i),None)
                    else:self.known_device[symbol,i]=value
                return 0
            if name in {'calloc','memcpy'}:
                result=super().call(name,expressions,site)
                if name=='calloc':
                    self.known_host.update({(result.buffer,i):0 for i in range(self.host_sizes[result.buffer])})
                else:
                    dst,src,size=args;data=[self.known_host.get((src.buffer,src.offset+i)) for i in range(size)]
                    for i,v in enumerate(data):
                        key=(dst.buffer,dst.offset+i)
                        if v is None:self.known_host.pop(key,None)
                        else:self.known_host[key]=v
                return result

        if not self.kernel and name in self.config.get('host_numeric_helpers',[]):
            fn=self.functions[name];params={p['name'] for p in children(fn) if p['kind']=='ParmVarDecl'}
            locals_=params|{n['name'] for n in walk(body_of(fn)) if n['kind']=='VarDecl'}
            if any(n['kind'] in {'CallExpr','ArraySubscriptExpr','MemberExpr','CUDAKernelCallExpr'} or (n['kind']=='DeclRefExpr' and n['referencedDecl'].get('name') not in locals_) or (n['kind']=='UnaryOperator' and n.get('opcode') in {'*','&'}) for n in walk(body_of(fn))):
                raise UnsupportedSource('Numeric helper must be memory-free and local-only')
            args=[self.eval(n) for n in expressions]
            if any(isinstance(a,Unknown) for a in args):return Unknown(set().union(*(a.dependencies for a in args if isinstance(a,Unknown))))
            return self.invoke_function(name,expressions,args)

        if name in {'strcmp','sscanf','pow','fabs','ceil'} and name not in self.functions and not self.kernel:
            args=[self.eval(n) for n in expressions]
            if name=='strcmp':
                if len(args)!=2 or not all(isinstance(a,str) for a in args):raise UnsupportedSource('strcmp requires known strings')
                return (args[0]>args[1])-(args[0]<args[1])
            if name=='sscanf':
                if len(args)!=3 or args[1] not in {'%d','%c'}:raise UnsupportedSource('Only single-field sscanf is supported')
                text=args[0]
                if isinstance(text,Pointer):
                    raw=[]
                    for i in range(text.offset,self.host_sizes[text.buffer]):
                        v=self.known_host.get((text.buffer,i))
                        if v is None:raise UnsupportedSource('Unknown input string')
                        if v==0:break
                        raw.append(v)
                    text=bytes(raw).decode()
                if not text:return -1
                if args[1]=='%c':self.store(args[2],ord(text[0]));return 1
                try:value=int(text.split()[0])
                except ValueError:raise UnsupportedSource('Invalid integer argument')
                self.store(args[2],value);return 1
            if any(isinstance(a,Unknown) for a in args):
                return Unknown(set().union(*(a.dependencies for a in args if isinstance(a,Unknown))))
            return math.ceil(args[0]) if name=='ceil' else abs(args[0]) if name=='fabs' else args[0]**args[1]
        if name in {'time', 'srand'} and name not in self.functions and not self.kernel:
            args=[self.eval(n) for n in expressions]
            if len(args)!=1:raise UnsupportedSource('Invalid clock/random call')
            return UNKNOWN if name=='time' else 0
        if name in {'fopen','fscanf','fclose'}:
            if self.kernel or name in self.functions:raise UnsupportedSource('Invalid file operation')
            args=[self.eval(n) for n in expressions]
            if name=='fopen':
                filename,mode=args
                if mode in {'w','w+','aw+'}:return HostStream([],True)
                if mode not in {'r','rb'} or filename not in self.config.get('input_files',{}):raise UnsupportedSource('Input needs a content hash')
                path=(self.source_directory/filename).resolve()
                if path.parent!=self.source_directory:raise UnsupportedSource('Input must be a sibling file')
                data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                if digest!=self.config['input_files'][filename]:raise UnsupportedSource('Configured input file changed')
                self.input_file_hashes[filename]=digest
                stream=HostStream(data.decode().split());stream.raw=data;return stream
            stream=args[0]
            if not isinstance(stream,HostStream):raise UnsupportedSource('Invalid file handle')
            if name=='fclose':return 0
            formats=re.findall(r'%[d]',args[1])
            if re.sub(r'%d','',args[1]).strip() or len(formats)!=len(args)-2 or stream.output:
                raise UnsupportedSource('Only integer fscanf fields are supported')
            if stream.position>=len(stream.tokens):stream.eof=True;return -1
            if stream.position+len(formats)>len(stream.tokens):raise UnsupportedSource('Partial formatted input is unsupported')
            for target in args[2:]:
                try:value=int(stream.tokens[stream.position])
                except ValueError:raise UnsupportedSource('Invalid integer input')
                self.store(target,value);stream.position+=1
            return len(formats)
        if name=='cudaMemcpy':
            # Arguments here are pure references/constants in supported call sites;
            # reject side effects instead of evaluating them twice.
            if any(x['kind'] in {'CallExpr','CUDAKernelCallExpr','CompoundAssignOperator'} or x.get('opcode') in {'++','--','='} for n in expressions for x in walk(n)):
                raise UnsupportedSource('Copy arguments must have no side effects')
            dst,src,size,direction=[self.eval(n) for n in expressions]
            if isinstance(src,Reference) and direction=='cudaMemcpyHostToDevice':
                if not isinstance(dst,Pointer) or dst.offset or self.layout(src.scalar_type)[0]!=size:raise UnsupportedSource('Scalar upload size mismatch')
                self.operation=self.order.schedule([(dst.buffer,'write')],'default')
                self.write_buffer(dst.buffer,size,self.cpu,'upload',self.host_provenance(src))
                self.put_value(Pointer(dst.buffer,0,src.scalar_type,True),self.read(src),self.known_device)
                return 0
            if isinstance(dst,Reference) and direction=='cudaMemcpyDeviceToHost':
                result=super().call(name,expressions,site)
                known=self.known_value(Pointer(src.buffer,src.offset,dst.scalar_type,True),self.known_device)
                if not isinstance(known,Unknown):self.store(dst,known)
                return result
            result=super().call(name,expressions,site)
            source=self.known_device if src.device else self.known_host
            target=self.known_device if dst.device else self.known_host
            snapshot=[source.get((src.buffer,src.offset+i)) for i in range(size)]
            for i,v in enumerate(snapshot):
                key=(dst.buffer,dst.offset+i)
                if v is None:target.pop(key,None)
                else:target[key]=v
            return result
        return super().call(name,expressions,site)

    def run(self):
        if not self.config.get('allow_process_exit_allocations'):return super().run()
        try:self.statement(body_of(self.functions[self.entry]))
        except ProgramExit:pass
        except FunctionReturn as r:
            if r.value not in (0,None):raise UnsupportedSource('Main returns failure')
        if not self.nodes:raise UnsupportedSource('No transfers generated')
        return (dict(directed=True,multigraph=False,graph={},nodes=self.nodes,
            edges=[dict(source=s,target=t) for s,t in sorted(self.edges)]),self.metadata,
            dict(configuration=self.config,launches=self.launches,
                source_units={p.name:hashlib.sha256(b).hexdigest() for p,b in self.source_files.items()},
                used_host_summaries=sorted(self.used_summaries),input_files=self.input_file_hashes,
                allocations_live_at_process_exit=dict(self.allocations)))
