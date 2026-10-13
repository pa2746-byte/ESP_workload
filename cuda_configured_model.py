"""Bounded Clang AST evaluation with explicit contracts for host-only input helpers.

No CUDA/C++ code is executed. Device values are unknown; data-dependent control
flow is rejected. Intended for small, reproducible source-analysis experiments.
"""
from dataclasses import dataclass
import hashlib
import itertools
import math
import re
from pathlib import Path

from cuda_source_model import (UnsupportedSource, children, walk, ref,
                               parse_source, constructor_arguments, StreamOrder)


class Unknown:
    pass

UNKNOWN = Unknown()


class FunctionReturn(Exception):
    def __init__(self, value):
        self.value = value


def source_declarations(ast, path, additional_sources=()):
    """Resolve Clang's elided top-level file locations for explicit source units."""
    files = {path.resolve(): path.read_bytes()}
    for relative in additional_sources:
        extra = (path.parent / relative).resolve()
        if extra.parent != path.parent.resolve():
            raise UnsupportedSource('Additional sources must be in the entry source directory')
        files[extra] = extra.read_bytes()
    current = None
    selected = []
    for node in children(ast):
        location = node.get('loc', {})
        if 'file' in location: current = Path(location['file']).resolve()
        if current in files: selected.append((node, files[current]))
    return selected, files

@dataclass
class Reference:
    owner: dict
    key: str

@dataclass(frozen=True)
class Pointer:
    buffer: str
    offset: int
    base: str
    device: bool = False

@dataclass
class Vector:
    base: str
    name: str
    count: int = 0


def body_of(fn):
    return next(c for c in children(fn) if c['kind'] == 'CompoundStmt')


def body_hash(fn, source):
    r = body_of(fn)['range']
    start, end = fn['range']['begin'], r['end']
    if 'offset' not in start or 'offset' not in end:
        raise UnsupportedSource('Host summary requires a directly written function body')
    return hashlib.sha256(source[start['offset']:end['offset'] + end['tokLen']]).hexdigest()


def directives_hash(source):
    """Pin preprocessing assumptions used by the reviewed host contracts."""
    logical = source.replace(b'\\\n', b'')
    directives = b'\n'.join(line.strip() for line in logical.splitlines() if line.lstrip().startswith(b'#'))
    return hashlib.sha256(directives).hexdigest()


class Evaluator:
    def __init__(self, path, config, cpu, mem, acc, clang):
        self.ast = parse_source(path, clang)
        self.source = path.read_bytes()
        declarations, self.source_files = source_declarations(self.ast, path, config.get('additional_sources', []))
        definitions = [n for n, _ in declarations if n['kind'] == 'FunctionDecl'
                       and any(c['kind'] == 'CompoundStmt' for c in children(n))]
        function_sources = {n['name']: contents for n, contents in declarations
                            if n['kind'] == 'FunctionDecl' and any(c['kind'] == 'CompoundStmt' for c in children(n))}
        self.functions = {n['name']: n for n in definitions}
        if len(self.functions) != len(definitions) or 'main' not in self.functions:
            raise UnsupportedSource('Configured analysis requires unique functions and main')
        self.config = config
        if config.get('version') != 1:
            raise UnsupportedSource('Unsupported analysis configuration version')
        self.parameters = config.get('parameters', {})
        self.device = config['device']
        for key in ('max_threads_per_block', 'free_memory_bytes', 'total_memory_bytes'):
            self.positive(self.device[key])
        if self.device['free_memory_bytes'] > self.device['total_memory_bytes']:
            raise UnsupportedSource('Free device memory exceeds total memory')
        if len(self.device['max_grid_size']) != 3:
            raise UnsupportedSource('Device grid limit must have three dimensions')
        for v in self.device['max_grid_size']: self.positive(v)
        self.limit = self.positive(config.get('max_enumerated_threads', 65536))
        if self.limit > 262144:
            raise UnsupportedSource('Configured analysis is limited to 262144 enumerated threads')
        self.summaries = config.get('host_summaries', {})
        if config.get('preprocessor_sha256') != directives_hash(self.source):
            raise UnsupportedSource('Preprocessor directives changed; review host contracts')
        for name, summary in self.summaries.items():
            if name not in self.functions or body_hash(self.functions[name], function_sources[name]) != summary['body_sha256']:
                raise UnsupportedSource(f'Host summary source changed: {name}; review its contract')
        for file, contents in self.source_files.items():
            if file != path.resolve() and config.get('additional_preprocessor_sha256', {}).get(file.name) != directives_hash(contents):
                raise UnsupportedSource('Included source directives changed; review host contracts')
        self.analyzed_functions = set(config.get('analyzed_functions', []))
        if self.analyzed_functions & self.summaries.keys():
            raise UnsupportedSource('A function cannot be both analyzed and summarized')
        self.loop_limit = self.positive(config.get('max_loop_iterations', 1024))
        if self.loop_limit > 10000: raise UnsupportedSource('Loop iteration limit exceeds 10000')
        self.total_loop_iterations = 0
        self.total_threads = 0
        for fn in self.functions.values():
            names = [n.get('name') for n in walk(body_of(fn)) if n['kind'] == 'VarDecl']
            if len(names) != len(set(names)):
                raise UnsupportedSource('Shadowed/redeclared variables require scope analysis')
        self.types = {'float': (4, 4, {}), 'double': (8, 8, {}), 'int': (4, 4, {}),
                      'unsigned int': (4, 4, {}), 'char': (1, 1, {})}
        # Only plain standard-layout records of primitive fields/arrays. No ABI guessing
        # for inheritance, packing, pointers, unions, or user-provided constructors.
        for node in children(self.ast):
            if node['kind'] != 'CXXRecordDecl' or not node.get('completeDefinition') or 'includedFrom' in node.get('loc', {}):
                continue
            data = node.get('definitionData', {})
            if not data.get('isPOD') or node.get('tagUsed') == 'union': continue
            fields, offset, alignment = {}, 0, 1
            members = [c for c in children(node) if not c.get('isImplicit')]
            if any(c['kind'].endswith('Attr') or c['kind'] not in {'FieldDecl', 'CXXRecordDecl'} for c in members): continue
            for field in members:
                if field['kind'] != 'FieldDecl': continue
                if field.get('isBitfield') or children(field): break
                typ = self.clean(field['type']['qualType'])
                try: size, align = self.layout(typ)[:2]
                except UnsupportedSource: break
                offset = (offset + align - 1) // align * align
                fields[field['name']] = (offset, typ)
                offset += size
                alignment = max(alignment, align)
            else:
                self.types[node['name']] = ((offset + alignment - 1) // alignment * alignment, alignment, fields)
        for node in children(self.ast):
            if node['kind'] == 'TypedefDecl':
                target = self.clean(node.get('type', {}).get('desugaredQualType', ''))
                if target in self.types: self.types[node['name']] = self.types[target]
        self.env = {'argc': UNKNOWN, 'argv': UNKNOWN}
        if 'arguments' in config:
            arguments = [str(self.configured(v)) if isinstance(v, dict) else v for v in config['arguments']]
            if not all(isinstance(v, str) for v in arguments): raise UnsupportedSource('Arguments must be strings or named parameters')
            self.env.update(argc=len(arguments)+1, argv=[str(path)] + arguments)
        self.current_function = 'main'
        self.call_stack = ['main']
        self.allocations, self.host_sizes = {}, {}
        self.allocated_names = set()
        self.host_allocation_count = 0
        self.nodes, self.edges, self.metadata = [], set(), []
        self.writers, self.readers, self.initialized = {}, {}, {}
        self.order = StreamOrder()
        self.operation = None
        self.kernel = False
        self.accesses = {}
        self.thread_writes = set()
        self.all_writes = set()
        self.cpu, self.mem, self.acc = cpu, mem, acc
        self.launches = []
        self.used_summaries = set()
        self.globals = {}
        for node, _ in declarations:
            if node['kind'] == 'VarDecl' and children(node):
                self.globals[node['name']] = self.eval(children(node)[-1])

    @staticmethod
    def clean(typ):
        return typ.replace('const ', '').replace('struct ', '').strip()

    @staticmethod
    def positive(value):
        if type(value) is not int or value <= 0:
            raise UnsupportedSource('Expected a positive integer configuration/size')
        return value

    def layout(self, typ):
        typ = self.clean(typ)
        if typ in self.types: return self.types[typ]
        match = re.fullmatch(r'(.+)\[(\d+)\]', typ)
        if match:
            size, align, _ = self.layout(match[1].strip())
            return size * int(match[2]), align, {}
        raise UnsupportedSource(f'Unsupported data layout: {typ}')

    def configured(self, value):
        if isinstance(value, dict) and set(value) == {'parameter'}:
            value = self.parameters[value['parameter']]
        if type(value) not in (int, float, bool):
            raise UnsupportedSource('Summary values must be numeric constants or parameters')
        return value

    def read(self, location):
        if isinstance(location, Reference): return location.owner.get(location.key, UNKNOWN)
        if isinstance(location, Pointer):
            if not location.device: return UNKNOWN
            self.access(location, 'read')
            return UNKNOWN
        raise UnsupportedSource('Unsupported lvalue read')

    def store(self, location, value):
        if isinstance(location, Reference):
            if location.owner is self.globals: raise UnsupportedSource('Global mutation is unsupported')
            location.owner[location.key] = value
        elif isinstance(location, Pointer) and location.device:
            self.access(location, 'write')
        else: raise UnsupportedSource('Unsupported memory write')

    def access(self, ptr, mode):
        if not self.kernel or ptr.buffer not in self.allocations:
            raise UnsupportedSource('Device memory access outside a supported kernel')
        size = self.layout(ptr.base)[0]
        if ptr.offset < 0 or ptr.offset + size > self.allocations[ptr.buffer]:
            raise UnsupportedSource('Kernel access exceeds allocation')
        keys = set(range(ptr.offset, ptr.offset + size))
        self.accesses.setdefault((ptr.buffer, mode), set()).update(keys)
        if mode == 'write':
            self.thread_writes.update((ptr.buffer, i) for i in keys)

    def location(self, n):
        k, cs = n['kind'], children(n)
        if k in {'ParenExpr', 'ImplicitCastExpr'}: return self.location(cs[0])
        if k == 'DeclRefExpr':
            name = n['referencedDecl']['name']
            if name in self.globals and name not in self.env: return Reference(self.globals, name)
            if name not in self.env: raise UnsupportedSource(f'Unmodeled global/reference: {name}')
            return Reference(self.env, name)
        if k == 'UnaryOperator' and n['opcode'] == '*':
            value = self.eval(cs[0])
            if not isinstance(value, Pointer): raise UnsupportedSource('Dereference of non-pointer')
            return value
        if k == 'MemberExpr':
            value = self.eval(cs[0])
            if isinstance(value, dict): return Reference(value, n['name'])
            if isinstance(value, Pointer):
                fields = self.layout(value.base)[2]
                if n['name'] not in fields: raise UnsupportedSource('Unknown record field')
                offset, typ = fields[n['name']]
                return Pointer(value.buffer, value.offset + offset, typ, value.device)
        if k == 'ArraySubscriptExpr':
            value, index = self.eval(cs[0]), self.eval(cs[1])
            if type(index) is not int: raise UnsupportedSource('Data-dependent array index')
            if isinstance(value, list):
                if not 0 <= index < len(value): raise UnsupportedSource('Host array index out of bounds')
                return Reference(dict(enumerate(value)), index)
            if isinstance(value, Pointer): return self.offset(value, index)
        if k == 'CXXOperatorCallExpr' and ref(cs[0]) == 'operator[]':
            vector, index = self.eval(cs[1]), self.eval(cs[2])
            if not isinstance(vector, Vector) or type(index) is not int or not 0 <= index < vector.count:
                raise UnsupportedSource('Invalid host vector index')
            return Pointer(vector.name, index * self.layout(vector.base)[0], vector.base)
        raise UnsupportedSource(f'Unsupported lvalue: {k}')

    def offset(self, pointer, index):
        if type(index) is not int: raise UnsupportedSource('Data-dependent pointer offset')
        return Pointer(pointer.buffer, pointer.offset + index * self.layout(pointer.base)[0], pointer.base, pointer.device)

    def truth(self, value):
        if type(value) not in (int, float, bool):
            raise UnsupportedSource('Data-dependent or unresolved condition')
        return bool(value)

    def eval(self, n):
        k, cs = n['kind'], children(n)
        if k in {'ImplicitCastExpr', 'ParenExpr', 'ExprWithCleanups', 'CXXBindTemporaryExpr', 'MaterializeTemporaryExpr'}:
            if len(cs) != 1: raise UnsupportedSource('Ambiguous wrapped expression')
            return self.eval(cs[0])
        if k == 'CXXDefaultArgExpr': return self.eval(cs[0]) if cs else None
        if k in {'CStyleCastExpr', 'CXXStaticCastExpr'}:
            value = self.eval(cs[0])
            if isinstance(value, Pointer):
                typ = self.clean(n['type']['qualType'])
                if not typ.endswith('*'): raise UnsupportedSource('Pointer to scalar cast')
                return Pointer(value.buffer, value.offset, typ[:-1].strip(), value.device)
            if isinstance(value, Reference): return value  # cudaMalloc's void** argument
            if isinstance(value, Unknown): return value
            typ = self.clean(n['type']['qualType'])
            if typ in {'float', 'double'}: return float(value)
            if typ in {'int', 'unsigned int', 'unsigned long', 'size_t'}:
                if typ != 'int' and value < 0: raise UnsupportedSource('Negative unsigned cast')
                return int(value)
            raise UnsupportedSource(f'Unsupported explicit cast: {typ}')
        if k == 'IntegerLiteral': return int(n['value'])
        if k == 'FloatingLiteral': return float(n['value'])
        if k == 'CXXBoolLiteralExpr': return n['value']
        if k == 'StringLiteral': return n['value']
        if k == 'CXXNullPtrLiteralExpr': return None
        if k == 'DeclRefExpr':
            name = n['referencedDecl']['name']
            if name in {'cudaMemcpyHostToDevice', 'cudaMemcpyDeviceToHost', 'cudaFuncCachePreferL1'}: return name
            if n['referencedDecl']['kind'] == 'FunctionDecl': return name
            if not self.kernel and name in {'stderr', 'stdout', '__stderrp', '__stdoutp'}: return UNKNOWN
            return self.read(self.location(n))
        if k in {'MemberExpr', 'ArraySubscriptExpr', 'CXXOperatorCallExpr'}:
            return self.read(self.location(n))
        if k == 'UnaryExprOrTypeTraitExpr' and n.get('name') == 'sizeof':
            return self.layout(n['argType']['qualType'])[0]
        if k == 'UnaryOperator':
            op = n['opcode']
            if op == '&': return self.location(cs[0])
            if op == '*': return self.read(self.location(n))
            value = self.eval(cs[0])
            if op == '!': return not self.truth(value)
            if op == '-': return UNKNOWN if isinstance(value, Unknown) else -value
            if op == '+': return value
            if op in {'++', '--'}:
                if type(value) is not int: raise UnsupportedSource('Increment requires a known integer')
                updated = value + (1 if op == '++' else -1)
                self.store(self.location(cs[0]), updated)
                return value if n.get('isPostfix') else updated
            raise UnsupportedSource(f'Unsupported unary operation: {op}')
        if k == 'ConditionalOperator': return self.eval(cs[1] if self.truth(self.eval(cs[0])) else cs[2])
        if k == 'CompoundAssignOperator':
            op = n['opcode']
            if op not in {'+=', '-='}: raise UnsupportedSource('Unsupported compound assignment')
            location = self.location(cs[0])
            a, b = self.read(location), self.eval(cs[1])
            value = UNKNOWN if isinstance(a, Unknown) or isinstance(b, Unknown) else a + (b if op == '+=' else -b)
            self.store(location, value)
            return value
        if k == 'BinaryOperator':
            op = n['opcode']
            if op == '=':
                value = self.eval(cs[1]); self.store(self.location(cs[0]), value); return value
            a = self.eval(cs[0])
            if op == '&&': return self.truth(a) and self.truth(self.eval(cs[1]))
            if op == '||': return self.truth(a) or self.truth(self.eval(cs[1]))
            b = self.eval(cs[1])
            if isinstance(a, Pointer) and op in {'+', '-'}: return self.offset(a, b if op == '+' else -b)
            if isinstance(a, Unknown) or isinstance(b, Unknown): return UNKNOWN
            if type(a) not in (int, float, bool) or type(b) not in (int, float, bool):
                raise UnsupportedSource('Unsupported binary operands')
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '/':
                if b == 0: raise UnsupportedSource('Division by zero')
                return (abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)) if type(a) is int and type(b) is int else a / b
            if op in {'<', '<=', '>', '>=', '==', '!='}:
                return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b, '==': a == b, '!=': a != b}[op]
            raise UnsupportedSource(f'Unsupported binary operation: {op}')
        if k == 'CXXConstructExpr':
            typ = n['type']['qualType']
            if typ == 'dim3':
                values = [self.eval(a) for a in constructor_arguments(n, self.ast)]
                if len(values) == 1 and isinstance(values[0], dict): return dict(values[0])
                if len(values) != 3: raise UnsupportedSource('Unsupported dim3 constructor')
                return dict(zip(('x', 'y', 'z'), values))
            if typ.startswith('std::vector<') and all(c['kind'] == 'CXXDefaultArgExpr' for c in cs):
                return None
            if typ == 'cudaDeviceProp' and not cs: return {}
            raise UnsupportedSource(f'Unsupported constructor: {typ}')
        if k == 'CUDAKernelCallExpr': return self.launch(n)
        if k == 'CallExpr': return self.call(ref(cs[0]), cs[1:])
        raise UnsupportedSource(f'Unsupported expression: {k}')

    def statement(self, n):
        k, cs = n['kind'], children(n)
        if k == 'CompoundStmt':
            for c in cs: self.statement(c)
        elif k == 'DeclStmt':
            for var in cs:
                name, typ = var['name'], self.clean(var['type']['qualType'])
                if self.kernel and '*' not in typ and '[' in typ:
                    raise UnsupportedSource('Kernel local arrays are unsupported in configured mode')
                vector = re.fullmatch(r'std::vector<(.+)>', typ)
                if vector:
                    if children(var): self.eval(children(var)[-1])
                    self.env[name] = Vector(vector[1], name)
                elif children(var): self.env[name] = self.eval(children(var)[-1])
                elif typ == 'cudaDeviceProp': self.env[name] = {}
                else: self.env[name] = UNKNOWN
        elif k == 'IfStmt':
            if n.get('hasInit') or n.get('hasVar') or len(cs) not in (2, 3):
                raise UnsupportedSource('Unsupported conditional declaration')
            if self.truth(self.eval(cs[0])): self.statement(cs[1])
            elif len(cs) == 3: self.statement(cs[2])
        elif k == 'NullStmt': pass
        elif k == 'ReturnStmt':
            raise FunctionReturn(self.eval(cs[0]) if cs else None)
        elif k == 'ForStmt':
            # Clang preserves an empty slot for a condition-variable declaration.
            raw = n.get('inner', [])
            if len(raw) != 5 or raw[1].get('kind') or not all(raw[i].get('kind') for i in (0,2,3,4)):
                raise UnsupportedSource('Unsupported for-loop structure')
            self.statement(raw[0])
            iterations = 0
            while self.truth(self.eval(raw[2])):
                iterations += 1
                self.total_loop_iterations += 1
                if iterations > self.loop_limit or self.total_loop_iterations > 1000000:
                    raise UnsupportedSource('Loop iteration budget exceeded')
                self.statement(raw[4])
                self.eval(raw[3])
        else: self.eval(n)

    def call(self, name, expressions):
        args = [self.eval(n) for n in expressions]
        if self.kernel:
            if name == 'sqrt' and len(args) == 1:
                return UNKNOWN if isinstance(args[0], Unknown) else math.sqrt(args[0])
            raise UnsupportedSource(f'Unsupported kernel call: {name}')
        if name in self.summaries:
            if any(isinstance(a, Pointer) and a.device for a in args):
                raise UnsupportedSource('Host summaries cannot receive device pointers')
            summary = self.summaries[name]
            self.used_summaries.add(name)
            for key, value in summary.get('writes', {}).items():
                self.store(args[int(key)], self.configured(value))
            for key, count in summary.get('vectors', {}).items():
                vector = args[int(key)]
                if not isinstance(vector, Vector): raise UnsupportedSource('Summary expects a host vector')
                vector.count = self.positive(self.configured(count))
                self.host_sizes[vector.name] = vector.count * self.layout(vector.base)[0]
            return UNKNOWN if summary.get('return_unknown') else self.configured(summary.get('return', 0))
        if name in self.analyzed_functions:
            if name in self.call_stack or len(self.call_stack) >= 16:
                raise UnsupportedSource('Recursive or excessively nested host calls are unsupported')
            fn = self.functions[name]
            params = [p for p in children(fn) if p['kind'] == 'ParmVarDecl']
            if len(args) != len(params): raise UnsupportedSource('Host argument count mismatch')
            saved, saved_fn = self.env, self.current_function
            self.env = {p['name']: a for p, a in zip(params, args)}
            self.current_function = name
            self.call_stack.append(name)
            try:
                self.statement(body_of(fn))
            except FunctionReturn as result:
                return result.value
            finally:
                self.call_stack.pop()
                self.env, self.current_function = saved, saved_fn
            return None
        if name == 'cudaGetDeviceProperties':
            if args[1] != self.device.get('ordinal', 0): raise UnsupportedSource('Device ordinal not configured')
            self.store(args[0], {'maxGridSize': self.device['max_grid_size'], 'maxThreadsPerBlock': self.device['max_threads_per_block']})
        elif name == 'cudaMemGetInfo':
            self.store(args[0], self.device['free_memory_bytes']); self.store(args[1], self.device['total_memory_bytes'])
        elif name in {'cudaThreadSynchronize', 'cudaDeviceSynchronize'}: self.order.synchronize()
        elif name == 'cudaMalloc':
            target, size = args
            if not isinstance(target, Reference): raise UnsupportedSource('Allocation needs a direct pointer')
            if target.key in self.allocated_names: raise UnsupportedSource('Allocation reuse is unsupported')
            self.allocated_names.add(target.key)
            self.allocations[target.key] = self.positive(size)
            # Type comes from the pointer declaration, not the void** cast at the API.
            declarations = [n for n in walk(body_of(self.functions[self.current_function])) if n['kind'] == 'VarDecl' and n.get('name') == target.key]
            if len(declarations) != 1: raise UnsupportedSource('Ambiguous allocation pointer declaration')
            base = self.clean(declarations[0]['type']['qualType']).rstrip('*').strip()
            self.layout(base)
            self.store(target, Pointer(target.key, 0, base, True))
        elif name in {'malloc', 'calloc'}:
            key = 'host_alloc_' + str(self.host_allocation_count)
            self.host_allocation_count += 1
            self.host_sizes[key] = self.positive(args[0] * args[1] if name == 'calloc' else args[0])
            return Pointer(key, 0, 'char')
        elif name == 'atoi':
            if len(args) != 1 or not isinstance(args[0], str) or not re.fullmatch(r'[+-]?\d+', args[0]):
                raise UnsupportedSource('atoi requires a configured integer string')
            return int(args[0])
        elif name == 'memcpy':
            dst, src, size = args
            self.positive(size)
            for ptr in (dst, src):
                if not isinstance(ptr, Pointer) or ptr.device or ptr.offset < 0 or ptr.offset + size > self.host_sizes.get(ptr.buffer, 0):
                    raise UnsupportedSource('Invalid host memcpy')
            return dst
        elif name == 'cudaFuncSetCacheConfig':
            if len(args) != 2 or args[0] not in self.functions or args[1] != 'cudaFuncCachePreferL1':
                raise UnsupportedSource('Unsupported cache configuration')
        elif name == 'cudaMemcpy':
            dst, src, size, direction = args
            self.positive(size)
            if not isinstance(dst, Pointer) or not isinstance(src, Pointer): raise UnsupportedSource('Unsupported copy pointer')
            device, host = (dst, src) if direction == 'cudaMemcpyHostToDevice' else (src, dst)
            if not device.device or host.device or device.offset or host.offset:
                raise UnsupportedSource('Only whole-buffer host/device copies are supported')
            if size > self.host_sizes.get(host.buffer, 0): raise UnsupportedSource('Host copy exceeds configured buffer')
            if direction == 'cudaMemcpyHostToDevice':
                self.operation = self.order.schedule([(device.buffer, 'write')], 'default')
                self.write_buffer(device.buffer, size, self.cpu, 'upload')
            elif direction == 'cudaMemcpyDeviceToHost':
                self.operation = self.order.schedule([(device.buffer, 'read')], 'default', blocking=True)
                self.read_buffer(device.buffer, size, self.cpu, 'download')
            else: raise UnsupportedSource('Unsupported copy direction')
        elif name == 'cudaFree':
            ptr = args[0]
            if not isinstance(ptr, Pointer) or not ptr.device or ptr.offset or ptr.buffer not in self.allocations:
                raise UnsupportedSource('Invalid cudaFree')
            self.order.synchronize()
            del self.allocations[ptr.buffer]
        elif name == 'free':
            ptr = args[0]
            if not isinstance(ptr, Pointer) or ptr.device or ptr.offset or ptr.buffer not in self.host_sizes:
                raise UnsupportedSource('Invalid host free')
            del self.host_sizes[ptr.buffer]
        elif name in {'printf', 'fprintf'}: pass
        elif name == 'exit': raise UnsupportedSource('Configured input takes an exit/error path')
        else: raise UnsupportedSource(f'Unsupported host call (requires a reviewed contract): {name}')
        return 0

    def launch(self, n):
        cs = children(n)
        name = ref(cs[0])
        fn = self.functions.get(name)
        if not fn or not any(c['kind'] == 'CUDAGlobalAttr' for c in children(fn)):
            raise UnsupportedSource('Missing kernel definition')
        config = children(cs[1])[1:]
        grid, block = self.eval(config[0]), self.eval(config[1])
        if len(config) != 4 or any(self.eval(c) not in (None, 0) for c in config[2:]):
            raise UnsupportedSource('Configured mode supports default-stream static-memory launches only')
        if not isinstance(grid, dict) or not isinstance(block, dict): raise UnsupportedSource('Invalid launch dimensions')
        dims = [self.positive(grid[k]) for k in ('x', 'y', 'z')] + [self.positive(block[k]) for k in ('x', 'y', 'z')]
        count = math.prod(dims)
        self.total_threads += count
        if self.total_threads > self.limit: raise UnsupportedSource('Launches exceed max_enumerated_threads; use a smaller input')
        if math.prod(dims[3:]) > self.device['max_threads_per_block'] or any(a > b for a, b in zip(dims[:3], self.device['max_grid_size'])):
            raise UnsupportedSource('Launch exceeds configured device limits')
        params = [p for p in children(fn) if p['kind'] == 'ParmVarDecl']
        args = [self.eval(c) for c in cs[2:]]
        if len(args) != len(params): raise UnsupportedSource('Kernel argument mismatch')
        if len({p.buffer for p in args if isinstance(p, Pointer)}) != sum(isinstance(p, Pointer) for p in args):
            raise UnsupportedSource('Aliased kernel arguments are unsupported')
        host = self.env
        self.kernel, self.accesses, self.all_writes = True, {}, set()
        try:
            for indices in itertools.product(*(range(d) for d in dims)):
                self.env = {p['name']: v for p, v in zip(params, args)}
                self.env.update(gridDim=grid, blockDim=block,
                                blockIdx=dict(zip(('x','y','z'), indices[:3])),
                                threadIdx=dict(zip(('x','y','z'), indices[3:])))
                self.thread_writes = set()
                try:
                    self.statement(body_of(fn))
                except FunctionReturn:
                    pass
                if self.all_writes & self.thread_writes: raise UnsupportedSource('Multiple threads write the same bytes')
                self.all_writes |= self.thread_writes
        finally:
            self.env, self.kernel = host, False
        reads = {b for b, mode in self.accesses if mode == 'read'}
        writes = {b for b, mode in self.accesses if mode == 'write'}
        if reads & writes: raise UnsupportedSource('In-place kernels require additional race/dependency analysis')
        self.operation = self.order.schedule(list(self.accesses), 'default')
        sizes = {}
        for key, offsets in self.accesses.items():
            if offsets != set(range(len(offsets))): raise UnsupportedSource('Non-contiguous or nonzero-offset footprint')
            sizes[key] = len(offsets)
        parents = [self.read_buffer(b, sizes[b, 'read'], self.acc, name + ':read') for b in self.allocations if b in reads]
        for b in self.allocations:
            if b in writes: self.write_buffer(b, sizes[b, 'write'], self.acc, name + ':write', parents)
        self.launches.append(dict(kernel=name, grid=list(dims[:3]), block=list(dims[3:]), enumerated_threads=count))
        return 0

    def transfer(self, buffer, size, src, dst, stage, parents):
        i = len(self.nodes)
        self.nodes.append(dict(id=i, vol=size, src_comp=src, dest_comp=dst))
        self.metadata.append(dict(id=i, buffer=buffer, stage=stage, operation_id=self.operation['id'],
                                  stream='default', ordered_after_operations=self.operation['ordered_after']))
        self.edges.update((p, i) for p in parents)
        return i

    def read_buffer(self, buffer, size, dst, stage):
        if self.initialized.get(buffer, 0) < size: raise UnsupportedSource('Read of uninitialized device buffer')
        i = self.transfer(buffer, size, self.mem, dst, stage, self.writers.get(buffer, set()))
        self.readers.setdefault(buffer, set()).add(i)
        return i

    def write_buffer(self, buffer, size, src, stage, parents=()):
        if size != self.allocations.get(buffer): raise UnsupportedSource('Writes must cover the whole allocation')
        i = self.transfer(buffer, size, src, self.mem, stage,
                          set(parents) | self.writers.get(buffer, set()) | self.readers.get(buffer, set()))
        self.writers[buffer], self.readers[buffer], self.initialized[buffer] = {i}, set(), size

    def run(self):
        try:
            self.statement(body_of(self.functions['main']))
        except FunctionReturn as result:
            if result.value not in (0, None): raise UnsupportedSource('Main returns failure for the configured input')
        if not self.nodes: raise UnsupportedSource('No transfers generated')
        if self.allocations: raise UnsupportedSource('Configured analysis requires all device buffers to be freed')
        return (dict(directed=True, multigraph=False, graph={}, nodes=self.nodes,
                     edges=[dict(source=s, target=t) for s, t in sorted(self.edges)]),
                self.metadata, dict(configuration=self.config, launches=self.launches,
                                    source_units={file.name: hashlib.sha256(data).hexdigest() for file, data in self.source_files.items()},
                                    used_host_summaries=sorted(self.used_summaries)))


def analyze_configured(path, config, cpu='cpu', mem='mem', acc='acc', clang=None):
    if len({cpu, mem, acc}) != 3 or not all((cpu, mem, acc)):
        raise UnsupportedSource('Component names must be nonempty and distinct')
    try:
        return Evaluator(path, config, cpu, mem, acc, clang).run()
    except (KeyError, TypeError, IndexError, StopIteration) as exc:
        raise UnsupportedSource(f'Invalid/unsupported configured source or configuration: {exc}') from exc
