"""Bounded Clang AST evaluation with explicit contracts for host-only input helpers.

No CUDA/C++ code is executed. Device values are unknown; data-dependent control
flow is rejected. Intended for small, reproducible source-analysis experiments.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import itertools
import math
import re
import json
import os
import subprocess
import ast as python_ast
from pathlib import Path

from cuda_source_model import (UnsupportedSource, children, walk, ref,
                               parse_source, constructor_arguments, StreamOrder)


class Unknown:
    def __init__(self, dependencies=()):
        self.dependencies = frozenset(dependencies)


def unknown_from(*values):
    return Unknown(set().union(*(v.dependencies for v in values if isinstance(v, Unknown))))

UNKNOWN = Unknown()


class FunctionReturn(Exception):
    def __init__(self, value):
        self.value = value


class LoopBreak(Exception):
    pass


class ProgramExit(Exception):
    pass


def parse_host_translation(path, clang=None, defines=None):
    include = Path(__file__).parent / 'source_analysis/include'
    command = [clang or os.environ.get('CLANGXX', 'clang++'), '-x', 'c', '-std=gnu89',
               '-Wno-error=implicit-function-declaration', '-Wno-error=implicit-int',
               '-I' + str(include), '-fsyntax-only', '-Xclang', '-ast-dump=json', str(path)]
    for key,value in (defines or {}).items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key) or type(value) is not int:raise UnsupportedSource('Invalid preprocessor definition')
        command.insert(1,f'-D{key}={value}')
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise UnsupportedSource('Clang could not parse host translation unit:\n' + result.stderr)
    ast = json.loads(result.stdout)
    # Clang IDs are process-local: namespace them before linking source units.
    prefix = path.name + ':'
    def qualify(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'id' and isinstance(child, str): value[key] = prefix + child
                else: qualify(child)
        elif isinstance(value, list):
            for child in value: qualify(child)
    qualify(ast)
    return ast



def source_declarations(ast, path, additional_sources=()):
    """Resolve Clang's elided top-level file locations for explicit source units."""
    files = {path.resolve(): path.read_bytes()}
    for relative in additional_sources:
        extra = (path.parent / relative).resolve()
        if not extra.is_relative_to(path.parent.resolve()):
            raise UnsupportedSource('Additional sources must be in the entry source directory')
        files[extra] = extra.read_bytes()
    current = None
    selected = []
    def linkage_nodes(nodes):
        for node in nodes:
            yield node
            if node['kind'] == 'LinkageSpecDecl': yield from linkage_nodes(children(node))
            if node['kind'] == 'FunctionTemplateDecl':
                for specialization in children(node):
                    if specialization['kind'] == 'FunctionDecl' and any(c['kind'] == 'TemplateArgument' for c in children(specialization)):
                        yield specialization
    for node in linkage_nodes(children(ast)):
        location = node.get('loc', {})
        if 'file' in location: current = Path(location['file']).resolve()
        if current in files: selected.append((node, files[current]))
    return selected, files

@dataclass
class Reference:
    owner: dict
    key: str | int
    label: str = None
    pointer_base: str = None
    scalar_type: str = None


@dataclass
class PointerArray:
    name: str
    base: str
    values: dict


@dataclass(frozen=True)
class HostView:
    """Opaque host array values; never usable as a device address or copy pointer."""
    shape: tuple


@dataclass(frozen=True)
class SharedView:
    """Only shape/bounds are modeled; contents ALWAYS remain unknown."""
    shape: tuple
    base: str = None

@dataclass(frozen=True)
class Pointer:
    buffer: str
    offset: int
    base: str
    device: bool = False

@dataclass
class HostStream:
    tokens: list
    output: bool = False
    position: int = 0


@dataclass
class Binding:
    location: object


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
        self.ast = parse_host_translation(path, clang, config.get('compile_defines')) if path.suffix == '.c' else parse_source(path, clang, config.get('compile_defines'))
        self.source_directory = path.parent.resolve()
        self.input_file_hashes = {}
        self.output_stream = HostStream([], True)
        self.source = path.read_bytes()
        declarations, self.source_files = source_declarations(self.ast, path, config.get('additional_sources', []))
        for relative in config.get('host_translation_units', []):
            host_path = (path.parent / relative).resolve()
            if not host_path.is_relative_to(path.parent.resolve()) or host_path.suffix != '.c':
                raise UnsupportedSource('Host translation units must be sibling C sources')
            host_ast = parse_host_translation(host_path, clang, config.get('compile_defines'))
            host_declarations, files = source_declarations(host_ast, host_path)
            declarations.extend(host_declarations)
            self.source_files.update(files)
            self.ast['inner'].extend(children(host_ast))
        for relative in config.get('cuda_translation_units', []):
            extra = (path.parent / relative).resolve()
            if not extra.is_relative_to(path.parent.resolve()) or extra.suffix != '.cu':
                raise UnsupportedSource('CUDA translation units must be within the source directory')
            extra_ast = parse_source(extra, clang, config.get('compile_defines'))
            selected, files = source_declarations(extra_ast, extra,
                [os.path.relpath(p, extra.parent) for p in self.source_files if p != extra and p.is_relative_to(extra.parent)])
            # Namespace process-local AST identifiers before linking units.
            for node in walk(extra_ast):
                if 'id' in node: node['id'] = relative + ':' + node['id']
                if 'referencedDecl' in node and 'id' in node['referencedDecl']:
                    node['referencedDecl']['id'] = relative + ':' + node['referencedDecl']['id']
            declarations.extend(selected)
            self.source_files.update(files)
            self.ast['inner'].extend(children(extra_ast))
        specializations = {}
        for node, _ in declarations:
            if node['kind'] != 'FunctionDecl': continue
            arguments = [c['type']['qualType'] for c in children(node) if c['kind'] == 'TemplateArgument' and 'type' in c]
            if arguments: specializations[node['id']] = node['name'] + '<' + ','.join(arguments) + '>'
        for node in walk(self.ast):
            if node.get('id') in specializations: node['name'] = specializations[node['id']]
            if node.get('referencedDecl', {}).get('id') in specializations:
                node['referencedDecl']['name'] = specializations[node['referencedDecl']['id']]
        definitions = [n for n, _ in declarations if n['kind'] == 'FunctionDecl'
                       and any(c['kind'] == 'CompoundStmt' for c in children(n))]
        function_sources = {n['name']: contents for n, contents in declarations
                            if n['kind'] == 'FunctionDecl' and any(c['kind'] == 'CompoundStmt' for c in children(n))}
        self.functions = {n['name']: n for n in definitions}
        if len(self.functions) != len(definitions) or 'main' not in self.functions:
            raise UnsupportedSource('Configured analysis requires unique functions and main')
        self.config = config
        self.opaque_shared = config.get('shared_memory', 'reject') == 'opaque_values'
        if config.get('shared_memory', 'reject') not in {'reject', 'opaque_values'}:
            raise UnsupportedSource('Unknown shared-memory model')
        self.entry = config.get('entry_function', 'main')
        if self.entry not in self.functions or (self.entry != 'main' and any(
                c['kind'] in {'ParmVarDecl', 'CUDAGlobalAttr'} for c in children(self.functions[self.entry]))):
            raise UnsupportedSource('Entry must be main or a parameterless host function')
        self.regions = config.get('dependency_granularity', 'whole_buffer') == 'byte_regions'
        if config.get('dependency_granularity', 'whole_buffer') not in {'whole_buffer', 'byte_regions'}:
            raise UnsupportedSource('Unknown dependency granularity')
        if config.get('version') != 1:
            raise UnsupportedSource('Unsupported analysis configuration version')
        self.parameters = config.get('parameters', {})
        self.region_limit = self.positive(config.get('max_tracked_bytes', 1048576))
        if self.region_limit > 1048576:
            raise UnsupportedSource('Region tracking is limited to 1048576 device bytes')
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
        self.lexical_scopes = config.get('lexical_scopes', False)
        if type(self.lexical_scopes) is not bool:
            raise UnsupportedSource('lexical_scopes must be a boolean')
        self.scope_frames = []
        self.opaque_host = config.get('host_memory', 'reject') == 'opaque_values'
        if config.get('host_memory', 'reject') not in {'reject', 'opaque_values'}:
            raise UnsupportedSource('Unknown host-memory model')
        for fn in self.functions.values():
            names = [n.get('name') for n in walk(body_of(fn)) if n['kind'] == 'VarDecl']
            if not self.lexical_scopes and len(names) != len(set(names)):
                raise UnsupportedSource('Shadowed/redeclared variables require scope analysis')
        self.types = {'float': (4, 4, {}), 'double': (8, 8, {}), 'int': (4, 4, {}),
                      'unsigned int': (4, 4, {}), 'char': (1, 1, {}), 'long': (8, 8, {}), 'long long': (8, 8, {}), 'bool': (1, 1, {}),
                      'float3': (12, 4, {axis: (4*i, 'float') for i, axis in enumerate('xyz')})}
        # Only plain standard-layout records of primitive fields/arrays. No ABI guessing
        # for inheritance, packing, pointers, unions, or user-provided constructors.
        for node in children(self.ast):
            if node['kind'] not in {'CXXRecordDecl', 'RecordDecl'} or not node.get('completeDefinition'):
                continue
            data = node.get('definitionData', {})
            if (node['kind'] == 'CXXRecordDecl' and not data.get('isPOD')) or node.get('tagUsed') == 'union': continue
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
                record_name = node.get('name')
                if not record_name:
                    record_name = next((d.get('name') for d in children(self.ast) if d['kind'] == 'TypedefDecl' and any(t.get('ownedTagDecl', {}).get('id') == node['id'] for t in walk(d))), None)
                if not record_name: continue
                self.types[record_name] = ((offset + alignment - 1) // alignment * alignment, alignment, fields)
        for node in children(self.ast):
            if node['kind'] == 'TypedefDecl':
                target = self.clean(node.get('type', {}).get('desugaredQualType', ''))
                if target in self.types: self.types[node['name']] = self.types[target]
        self.pod_records = {n.get('name') for n in walk(self.ast) if n.get('kind') == 'CXXRecordDecl' and n.get('definitionData', {}).get('isPOD')}
        self.env = {'argc': UNKNOWN, 'argv': UNKNOWN}
        if 'arguments' in config:
            arguments = [str(self.configured(v)) if isinstance(v, dict) else v for v in config['arguments']]
            if not all(isinstance(v, str) for v in arguments): raise UnsupportedSource('Arguments must be strings or named parameters')
            self.env.update(argc=len(arguments)+1, argv=[str(path)] + arguments)
        self.current_function = self.entry
        self.call_stack = [self.entry]
        self.allocations, self.host_sizes = {}, {}
        self.host_dependencies = {}
        self.track_host_dependencies = config.get('host_dependencies', False)
        self.track_integers = config.get('known_integer_inputs', False)
        self.host_integers, self.device_integers = {}, {}
        self.checked_barriers = config.get('checked_block_barriers', False)
        self.allocated_names = set()
        self.host_allocation_count = 0
        self.nodes, self.edges, self.metadata = [], set(), []
        self.writers, self.readers, self.initialized = {}, {}, {}
        self.region_writers, self.region_readers = {}, {}
        self.order = StreamOrder()
        self.operation = None
        self.kernel = False
        self.accesses = {}
        self.thread_writes = set()
        self.all_writes = set()
        self.cpu, self.mem, self.acc = cpu, mem, acc
        self.launches = []
        self.used_summaries = set()
        self.cli_state = {'optind': 1, 'optarg': None}
        self.enum_values = {}
        for enum in walk(self.ast):
            if enum['kind'] != 'EnumDecl': continue
            value = -1
            for member in children(enum):
                if member['kind'] != 'EnumConstantDecl': continue
                constants = [c for c in walk(member) if c.get('kind') == 'ConstantExpr' and 'value' in c]
                value = int(constants[0]['value']) if constants else value + 1
                self.enum_values[member['id']] = value
        self.globals = {}
        self.constant_buffers = set()
        for node, _ in declarations:
            if node['kind'] == 'VarDecl' and any(c['kind'] == 'CUDAConstantAttr' for c in children(node)):
                typ = self.clean(node['type']['qualType'])
                match = re.fullmatch(r'(.+)\[(\d+)\]', typ)
                if not match: raise UnsupportedSource('Only fixed arrays of constant device data are supported')
                name, base, count = node['name'], match[1].strip(), int(match[2])
                self.allocations[name] = self.layout(base)[0] * self.positive(count)
                if self.regions and sum(self.allocations.values()) > self.region_limit:
                    raise UnsupportedSource('Constant buffers exceed region budget')
                self.allocated_names.add(name)
                self.constant_buffers.add(name)
                self.globals[name] = Pointer(name, 0, base, True)
                continue
            if node['kind'] == 'VarDecl' and children(node):
                typ = self.clean(node['type']['qualType'])
                if self.opaque_host and re.fullmatch(r'(int|float|double)(?:\[\d+\])+', typ):
                    allowed = {'InitListExpr', 'IntegerLiteral', 'FloatingLiteral',
                               'UnaryOperator', 'ImplicitCastExpr', 'ConstantExpr',
                               'ParenExpr', 'ImplicitValueInitExpr', 'BinaryOperator'}
                    for part in walk(children(node)[-1]):
                        if (part['kind'] not in allowed
                                or (part['kind'] == 'UnaryOperator' and part['opcode'] not in {'+', '-'})
                                or (part['kind'] == 'BinaryOperator' and part['opcode'] not in {'+', '-', '*', '/', '%'})):
                            raise UnsupportedSource('Opaque host array initializer must be pure numeric constants')
                    self.globals[node['name']] = HostView(tuple(map(int, re.findall(r'\[(\d+)\]', typ))))
                else:
                    self.globals[node['name']] = self.eval(children(node)[-1])
        self.global_types = {n['name']: self.clean(n['type']['qualType'])
                        for n, _ in declarations if n['kind'] == 'VarDecl'}
        boundary = config.get('entry_globals', {})
        if boundary and self.entry == 'main':
            raise UnsupportedSource('Entry globals require an explicit function boundary')
        self.configure_globals(boundary)
        self.mutable_globals = set(config.get('mutable_scalar_globals', []))
        if any(self.global_types.get(n) not in {'int', 'unsigned int', 'float', 'double','long','bool'} for n in self.mutable_globals):
            raise UnsupportedSource('Mutable globals must be declared scalars')

    def configure_globals(self, boundary):
        for name, value in boundary.items():
            typ = self.global_types.get(name)
            if name in self.globals:
                raise UnsupportedSource('Global input contract cannot overwrite existing state')
            if isinstance(value, dict) and 'shape' in value:
                if not typ or not typ.endswith('*') or set(value) != {'shape'} or not value['shape']:
                    raise UnsupportedSource('Entry buffers require a declared primitive pointer and shape')
                base = typ[:-1].strip()
                size = self.layout(base)[0] * math.prod(self.positive(self.configured(d)) for d in value['shape'])
                self.host_sizes['entry:' + name] = size
                self.globals[name] = Pointer('entry:' + name, 0, base)
            elif value == {'opaque_host_pointer': True}:
                if not typ or not typ.endswith('*'):
                    raise UnsupportedSource('Opaque host input must be a pointer')
                # A cleanup-only handle: zero capacity forbids copying any bytes.
                self.host_sizes['entry:' + name] = 0
                self.globals[name] = Pointer('entry:' + name, 0, typ[:-1].strip())
            else:
                if typ != 'int': raise UnsupportedSource('Entry scalar globals must be int')
                self.globals[name] = self.positive(self.configured(value))

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
        if isinstance(value, dict) and set(value) == {'add'}:
            return sum(self.configured(v) for v in value['add'])
        if isinstance(value, dict) and set(value) == {'parameter'}:
            value = self.parameters[value['parameter']]
        if type(value) not in (int, float, bool):
            raise UnsupportedSource('Summary values must be numeric constants or parameters')
        return value

    def read(self, location):
        if isinstance(location, SharedView) and not location.shape and location.base in self.types and self.types[location.base][2]: return dict.fromkeys(self.types[location.base][2], UNKNOWN)
        if isinstance(location, (SharedView, HostView)): return location if location.shape else UNKNOWN
        if isinstance(location, Reference):
            value = location.owner.get(location.key, UNKNOWN)
            return self.read(value.location) if isinstance(value, Binding) else value
        if isinstance(location, Pointer):
            if self.layout(location.base)[2]: return location
            if not location.device:
                if self.kernel:
                    raise UnsupportedSource('Kernel cannot access an ordinary host pointer')
                if self.opaque_host and (location.offset < 0 or location.offset + self.layout(location.base)[0] > self.host_sizes.get(location.buffer, 0)):
                    raise UnsupportedSource('Host read exceeds allocation')
                if self.track_integers and location.base in {'int', 'unsigned int'} and (location.buffer, location.offset) in self.host_integers:
                    return self.host_integers[location.buffer, location.offset]
                return Unknown(self.host_dependencies.get(location.buffer, ())) if self.track_host_dependencies else UNKNOWN
            self.access(location, 'read')
            if self.track_integers and location.base in {'int', 'unsigned int'}:
                return self.device_integers.get((location.buffer, location.offset), UNKNOWN)
            return UNKNOWN
        raise UnsupportedSource('Unsupported lvalue read')

    def store(self, location, value):
        if isinstance(location, SharedView):
            if location.shape or not isinstance(value, (int, float, Unknown)):
                raise UnsupportedSource('Shared storage only supports scalar numeric values')
            # Do not propagate even known values: other threads may overwrite them.
        elif isinstance(location, Reference):
            if location.owner is self.globals and (self.kernel or location.key not in self.mutable_globals
                    or not isinstance(value, (int, float, Unknown))):
                raise UnsupportedSource('Global mutation is unsupported')
            location.owner[location.key] = value
        elif isinstance(location, Pointer) and location.device:
            self.access(location, 'write')
            self.invalidate_integers(self.device_integers, location.buffer, location.offset, self.layout(location.base)[0])
        elif isinstance(location, Pointer) and self.opaque_host and not self.kernel:
            if not isinstance(value, (int, float, Unknown)):
                raise UnsupportedSource('Opaque host storage accepts only numeric values')
            if location.offset < 0 or location.offset + self.layout(location.base)[0] > self.host_sizes.get(location.buffer, 0):
                raise UnsupportedSource('Host write exceeds allocation')
            if self.track_integers:
                self.invalidate_integers(self.host_integers, location.buffer, location.offset, self.layout(location.base)[0])
                if location.base in {'int', 'unsigned int'} and type(value) is int:
                    self.host_integers[location.buffer, location.offset] = value
            if self.track_host_dependencies and isinstance(value, Unknown):
                self.host_dependencies.setdefault(location.buffer, set()).update(value.dependencies)
            # Contents are intentionally unknown, including known writes. Never
            # infer control flow or device addresses from this abstraction.
        else: raise UnsupportedSource('Unsupported memory write')

    @staticmethod
    def invalidate_integers(storage, buffer, offset, size):
        for key in list(storage):
            if key[0] == buffer and key[1] < offset + size and key[1] + 4 > offset:
                del storage[key]

    def access(self, ptr, mode):
        if mode == 'write' and ptr.buffer in self.constant_buffers:
            raise UnsupportedSource('Kernel cannot write CUDA constant memory')
        if not self.kernel or ptr.buffer not in self.allocations:
            raise UnsupportedSource('Device memory access outside a supported kernel')
        size = self.layout(ptr.base)[0]
        if ptr.offset < 0 or ptr.offset + size > self.allocations[ptr.buffer]:
            raise UnsupportedSource('Kernel access exceeds allocation')
        keys = set(range(ptr.offset, ptr.offset + size))
        if self.checked_barriers:
            for offset in keys:
                self.thread_access_events.append(((ptr.buffer, offset), mode, self.current_block, len(self.thread_barriers)))
        self.accesses.setdefault((ptr.buffer, mode), set()).update(keys)
        if mode == 'write':
            self.thread_writes.update((ptr.buffer, i) for i in keys)
        elif self.regions:
            accesses = {(ptr.buffer, i) for i in keys}
            if accesses & self.thread_writes and not self.config.get('integer_state'):
                raise UnsupportedSource('Read after write within one thread is unsupported')
            self.thread_reads.update(accesses)

    def location(self, n):
        k, cs = n['kind'], children(n)
        if k in {'ParenExpr', 'ImplicitCastExpr'}: return self.location(cs[0])
        if k == 'DeclRefExpr':
            name = n['referencedDecl']['name']
            if self.opaque_host and name in self.cli_state and name not in self.env:
                return Reference(self.cli_state, name)
            if name in self.globals and name not in self.env:
                typ=self.clean(n['referencedDecl'].get('type',{}).get('qualType',''))
                return Reference(self.globals,name,pointer_base=typ[:-1].strip() if typ.endswith('*') else None,scalar_type=typ if typ in self.types else None)
            if name not in self.env: raise UnsupportedSource(f'Unmodeled global/reference: {name}')
            if isinstance(self.env[name], Binding): return self.env[name].location
            typ = self.clean(n['referencedDecl'].get('type', {}).get('qualType', ''))
            return Reference(self.env, name, pointer_base=typ[:-1].strip() if typ.endswith('*') else None, scalar_type=typ if typ in self.types else None)
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
            if isinstance(value, HostView) and not self.kernel:
                if not value.shape or (type(index) is int and not 0 <= index < value.shape[0]):
                    raise UnsupportedSource('Host array index out of bounds')
                if not isinstance(index, (int, Unknown)):
                    raise UnsupportedSource('Unsupported host array index')
                return HostView(value.shape[1:])
            if type(index) is not int: raise UnsupportedSource('Data-dependent array index')
            if isinstance(value, SharedView):
                if not value.shape or not 0 <= index < value.shape[0]:
                    raise UnsupportedSource('Shared array index out of bounds')
                return SharedView(value.shape[1:], value.base)
            if isinstance(value, PointerArray):
                if index not in value.values: raise UnsupportedSource('Pointer array index out of bounds')
                return Reference(value.values, index, f'{value.name}[{index}]', value.base)
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
        if value is None: return False
        if isinstance(value, dict) and not self.kernel and self.opaque_host: return True
        if isinstance(value, Pointer) and not value.device: return True  # modeled successful host allocation
        if type(value) not in (int, float, bool):
            raise UnsupportedSource('Data-dependent or unresolved condition')
        return bool(value)

    def eval(self, n):
        k, cs = n['kind'], children(n)
        if k in {'ImplicitCastExpr', 'ParenExpr', 'ConstantExpr', 'ExprWithCleanups', 'CXXBindTemporaryExpr', 'MaterializeTemporaryExpr'}:
            if len(cs) != 1: raise UnsupportedSource('Ambiguous wrapped expression')
            value = self.eval(cs[0])
            numeric_casts = {'IntegralToFloating', 'FloatingToIntegral', 'IntegralCast', 'FloatingCast'}
            if k == 'ImplicitCastExpr' and n.get('castKind') in numeric_casts and not isinstance(value, Unknown):
                return float(value) if self.clean(n['type']['qualType']) in {'float','double'} else int(value)
            return value
        if k == 'CXXDefaultArgExpr': return self.eval(cs[0]) if cs else None
        if k in {'CStyleCastExpr', 'CXXStaticCastExpr', 'CXXFunctionalCastExpr'}:
            value = self.eval(cs[0])
            if isinstance(value, Pointer):
                typ = self.clean(n['type']['qualType'])
                if not typ.endswith('*'): raise UnsupportedSource('Pointer to scalar cast')
                return Pointer(value.buffer, value.offset, typ[:-1].strip(), value.device)
            if isinstance(value, Reference): return value  # cudaMalloc's void** argument
            if isinstance(value, Unknown): return value
            typ = self.clean(n['type']['qualType'])
            if typ.endswith('*') and value == 0: return None
            if typ in {'float', 'double'}: return float(value)
            if typ in {'int', 'unsigned int', 'unsigned long', 'size_t'}:
                if typ != 'int' and value < 0: raise UnsupportedSource('Negative unsigned cast')
                return int(value)
            raise UnsupportedSource(f'Unsupported explicit cast: {typ}')
        if k in {'IntegerLiteral', 'CharacterLiteral'}: return int(n['value'])
        if k == 'InitListExpr' and not self.kernel and self.opaque_host:
            return [self.eval(c) for c in cs]
        if k == 'FloatingLiteral': return float(n['value'])
        if k == 'CXXBoolLiteralExpr': return n['value']
        if k == 'StringLiteral': return python_ast.literal_eval(n['value'])
        if k in {'CXXNullPtrLiteralExpr', 'GNUNullExpr'}: return None
        if k == 'DeclRefExpr':
            name = n['referencedDecl']['name']
            if name == 'cout' and not self.kernel: return self.output_stream
            if name == 'cudaSuccess': return 0
            if name in {'cudaMemcpyHostToDevice', 'cudaMemcpyDeviceToHost', 'cudaMemcpyDeviceToDevice', 'cudaFuncCachePreferL1'}: return name
            if n['referencedDecl'].get('id') in self.enum_values:
                return self.enum_values[n['referencedDecl']['id']]
            if n['referencedDecl']['kind'] == 'FunctionDecl': return name
            if not self.kernel and name in {'stderr', 'stdout', '__stderrp', '__stdoutp'}: return UNKNOWN
            return self.read(self.location(n))
        if k == 'CXXNewExpr' and not self.kernel and n.get('isArray') and len(cs) == 1:
            base = self.clean(n['type']['qualType']).rstrip('*').strip()
            return self.contract_value({'base': base, 'shape': [self.eval(cs[0])]})
        if k == 'CXXOperatorCallExpr' and ref(cs[0]) == 'operator=' and len(cs)==3:
            target=self.location(cs[1])
            if not isinstance(target,SharedView):raise UnsupportedSource('Only modeled shared-record copy assignment is supported')
            value=self.eval(cs[2]); self.store(target,value); return value
        if k == 'CXXOperatorCallExpr' and ref(cs[0]) == 'operator>>' and not self.kernel:
            stream = self.eval(cs[1])
            if not isinstance(stream, HostStream) or stream.output or len(cs) != 3:
                raise UnsupportedSource('Unsupported input operator')
            if stream.position >= len(stream.tokens): raise UnsupportedSource('Configured input file exhausted')
            typ = self.clean(cs[2]['type']['qualType'])
            if typ not in {'int','float','double'}: raise UnsupportedSource('Only numeric input extraction is supported')
            token = stream.tokens[stream.position]
            stream.position += 1
            self.store(self.location(cs[2]), int(token) if typ == 'int' else float(token))
            return stream
        if k == 'CXXOperatorCallExpr' and ref(cs[0]) == 'operator<<' and not self.kernel:
            stream = self.eval(cs[1])
            if not isinstance(stream, HostStream) or not stream.output: raise UnsupportedSource('Unsupported output operator')
            for argument in cs[2:]: self.eval(argument)
            return stream
        if k == 'CXXMemberCallExpr' and not self.kernel:
            member = cs[0]
            if member['kind'] != 'MemberExpr': raise UnsupportedSource('Unsupported member call')
            stream = self.eval(children(member)[0])
            if not isinstance(stream, HostStream): raise UnsupportedSource('Only configured stream member calls are supported')
            if member['name'] == 'operator>>' and not stream.output and len(cs) == 2:
                if stream.position >= len(stream.tokens): raise UnsupportedSource('Configured input file exhausted')
                typ = self.clean(cs[1]['type']['qualType'])
                if typ not in {'int','float','double'}: raise UnsupportedSource('Only numeric input extraction is supported')
                token = stream.tokens[stream.position]
                stream.position += 1
                self.store(self.location(cs[1]), int(token) if typ == 'int' else float(token))
                return stream
            if member['name'] == 'operator<<' and stream.output:
                for argument in cs[1:]: self.eval(argument)
                return stream
            raise UnsupportedSource('Unsupported stream operation')
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
            if op == '-': return value if isinstance(value, Unknown) else -value
            if op == '+': return value
            if op in {'++', '--'}:
                if type(value) is not int: raise UnsupportedSource('Increment requires a known integer')
                updated = value + (1 if op == '++' else -1)
                self.store(self.location(cs[0]), updated)
                return value if n.get('isPostfix') else updated
            raise UnsupportedSource(f'Unsupported unary operation: {op}')
        if k == 'ConditionalOperator':
            condition = self.eval(cs[0])
            if isinstance(condition, Unknown) and self.kernel and self.opaque_shared:
                # Permit MIN-like value selection only. No memory accesses, calls,
                # mutations or pointer selection can be hidden in either arm.
                allowed = {'DeclRefExpr','ImplicitCastExpr','ParenExpr','IntegerLiteral',
                           'FloatingLiteral','CXXBoolLiteralExpr','BinaryOperator','UnaryOperator'}
                for arm in cs[1:]:
                    for part in walk(arm):
                        binary = {'+','-','*','/','%','<','<=','>','>=','==','!='}
                        if (part['kind'] not in allowed
                                or (part['kind']=='BinaryOperator' and part['opcode'] not in binary)
                                or (part['kind']=='UnaryOperator' and part['opcode'] not in {'+','-'})):
                            raise UnsupportedSource('Unknown conditional arms must be pure scalar expressions')
                    if not isinstance(self.eval(arm), (int,float,Unknown)):
                        raise UnsupportedSource('Unknown conditional cannot select pointers')
                return UNKNOWN
            return self.eval(cs[1] if self.truth(condition) else cs[2])
        if k == 'CompoundAssignOperator':
            op = n['opcode']
            if op not in {'+=', '-=', '*=', '/='}: raise UnsupportedSource('Unsupported compound assignment')
            location = self.location(cs[0])
            a, b = self.read(location), self.eval(cs[1])
            if isinstance(a, Unknown) or isinstance(b, Unknown): value = unknown_from(a, b)
            elif op == '+=': value = a + b
            elif op == '-=': value = a - b
            elif op == '*=': value = a * b
            else:
                if b == 0: raise UnsupportedSource('Division by zero')
                value = (abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)) if type(a) is int and type(b) is int else a / b
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
            if isinstance(a, Unknown) or isinstance(b, Unknown): return unknown_from(a, b)
            if op in {'==', '!='} and (isinstance(a, (Pointer, str, HostStream)) or isinstance(b, (Pointer, str, HostStream))) and (a is None or b is None or a == 0 or b == 0):
                return op == '!='
            if type(a) not in (int, float, bool) or type(b) not in (int, float, bool):
                raise UnsupportedSource('Unsupported binary operands')
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '%':
                if type(a) is not int or type(b) is not int or b == 0:
                    raise UnsupportedSource('Remainder requires integers and a nonzero divisor')
                return a - (abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)) * b
            if op == '/':
                if b == 0: raise UnsupportedSource('Division by zero')
                return (abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)) if type(a) is int and type(b) is int else a / b
            if op in {'<', '<=', '>', '>=', '==', '!='}:
                return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b, '==': a == b, '!=': a != b}[op]
            raise UnsupportedSource(f'Unsupported binary operation: {op}')
        if k == 'CXXConstructExpr':
            typ = self.clean(n['type']['qualType'])
            if typ in self.types and self.types[typ][2] and len(cs)==1:
                import copy
                value=self.eval(cs[0])
                if isinstance(value,dict): return copy.deepcopy(value)
                raise UnsupportedSource('Record copy requires a modeled record value')
            if typ in self.types and self.types[typ][2] and not cs: return dict.fromkeys(self.types[typ][2], UNKNOWN)
            if not self.kernel and ('ifstream' in typ or 'basic_ifstream' in typ):
                name = self.eval(cs[0])
                if name not in self.config.get('input_files', {}): raise UnsupportedSource('Input file needs an explicit content hash')
                file = (self.source_directory / name).resolve()
                if file.parent != self.source_directory: raise UnsupportedSource('Input file must be a sibling of the source')
                content = file.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                if sha != self.config['input_files'][name]: raise UnsupportedSource('Configured input file changed')
                self.input_file_hashes[name] = sha
                return HostStream(content.decode().split())
            if not self.kernel and ('ofstream' in typ or 'basic_ofstream' in typ):
                if any(c['kind'] != 'CXXDefaultArgExpr' for c in cs[1:]):
                    raise UnsupportedSource('Only default output stream modes are supported')
                self.eval(cs[0])
                return HostStream([], True)
            if typ == 'dim3':
                values = [self.eval(a) for a in constructor_arguments(n, self.ast)]
                if len(values) == 1 and isinstance(values[0], dict): return dict(values[0])
                if len(values) != 3: raise UnsupportedSource('Unsupported dim3 constructor')
                return dict(zip(('x', 'y', 'z'), values))
            if typ.startswith('std::vector<') and all(c['kind'] == 'CXXDefaultArgExpr' for c in cs):
                return None
            if typ in {'cudaDeviceProp', 'timeval'} and not cs: return {}
            record = self.clean(n.get('type', {}).get('desugaredQualType', typ))
            if self.opaque_host and not self.kernel and not cs and record in self.pod_records:
                return dict.fromkeys(self.types[record][2], UNKNOWN) if record in self.types else UNKNOWN
            raise UnsupportedSource(f'Unsupported constructor: {typ}')
        if k == 'CUDAKernelCallExpr': return self.launch(n)
        if k == 'CXXDeleteExpr':
            if self.kernel or not n.get('isArray'): raise UnsupportedSource('Only host delete[] is supported')
            self.free_host(self.eval(cs[0]))
            return None
        if k == 'CallExpr': return self.call(ref(cs[0]), cs[1:], n.get('id'))
        raise UnsupportedSource(f'Unsupported expression: {k}')

    @contextmanager
    def local_scope(self, declarations):
        if not self.lexical_scopes:
            yield
            return
        frame, absent = {}, object()
        self.scope_frames.append((frame, absent))
        try:
            yield
        finally:
            self.scope_frames.pop()
            for name, value in frame.items():
                if value is absent: self.env.pop(name, None)
                else: self.env[name] = value

    def declare_local(self, name):
        if self.lexical_scopes:
            frame, absent = self.scope_frames[-1]
            # Capture at the declaration, not block entry: earlier statements
            # in this block may legitimately modify the outer binding.
            if name not in frame: frame[name] = self.env.get(name, absent)
            self.env[name] = UNKNOWN

    def statement(self, n):
        k, cs = n['kind'], children(n)
        if k == 'AttributedStmt':
            if len(cs) < 2 or any(c['kind'] != 'LoopHintAttr' for c in cs[:-1]):
                raise UnsupportedSource('Unsupported statement attribute')
            self.statement(cs[-1])
        elif k == 'CompoundStmt':
            with self.local_scope(cs):
                for c in cs: self.statement(c)
        elif k == 'DeclStmt':
            for var in cs:
                name, typ = var['name'], self.clean(var['type']['qualType'])
                self.declare_local(name)
                if var.get('storageClass')=='extern' and name in self.cli_state:
                    self.env[name]=Binding(Reference(self.cli_state,name)); continue
                if any(c['kind'] == 'CUDASharedAttr' for c in children(var)):
                    match = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)\s*((?:\[\d+\])+)', typ)
                    if not self.kernel or not self.opaque_shared or not match or match[1] not in self.types or any(c['kind'] != 'CUDASharedAttr' and not (c['kind']=='CXXConstructExpr' and not children(c) and bool(self.types[match[1]][2])) for c in children(var)):
                        raise UnsupportedSource('Only explicitly enabled fixed shared numeric arrays are supported')
                    self.env[name] = SharedView(tuple(int(d) for d in re.findall(r'\d+', match[2])), match[1])
                    continue
                local_array = re.fullmatch(r'(float|double|int)\s*\[(\d+)\]', typ)
                if local_array and not self.kernel and self.opaque_host:
                    if children(var): raise UnsupportedSource('Local numeric array initializers are unsupported')
                    self.env[name] = self.contract_value({'base': local_array[1], 'shape': [int(local_array[2])]})
                    continue
                pointer_array = re.fullmatch(r'(float|double|int)\s*\*\[(\d+)\]', typ)
                if pointer_array and not self.kernel:
                    if children(var): raise UnsupportedSource('Pointer array initializers are unsupported')
                    count = int(pointer_array[2])
                    if not 0 < count <= 1024: raise UnsupportedSource('Pointer array size exceeds limit')
                    self.env[name] = PointerArray(name,pointer_array[1],dict.fromkeys(range(count),UNKNOWN))
                    continue
                if self.kernel and '*' not in typ and '[' in typ:
                    raise UnsupportedSource('Kernel local arrays are unsupported in configured mode')
                vector = re.fullmatch(r'std::vector<(.+)>', typ)
                if vector:
                    if children(var): self.eval(children(var)[-1])
                    self.env[name] = Vector(vector[1], name)
                elif children(var): self.env[name] = self.eval(children(var)[-1])
                elif typ in {'cudaDeviceProp', 'timeval'}: self.env[name] = {}
                elif typ in self.types and self.types[typ][2]: self.env[name] = dict.fromkeys(self.types[typ][2], UNKNOWN)
                else: self.env[name] = UNKNOWN
        elif k == 'IfStmt':
            if n.get('hasInit') or n.get('hasVar') or len(cs) not in (2, 3):
                raise UnsupportedSource('Unsupported conditional declaration')
            condition = self.eval(cs[0])
            if isinstance(condition, Unknown) and self.kernel and self.config.get('opaque_scalar_branches') and not self.config.get('enumerate_unknown_paths'):
                self.merge_scalar_branches(condition, cs[1:])
                return
            branch = cs[1] if self.truth(condition) else cs[2] if len(cs) == 3 else None
            if branch is not None:
                with self.local_scope([branch]):
                    self.statement(branch)
        elif k == 'WhileStmt':
            if len(cs) != 2: raise UnsupportedSource('Unsupported while structure')
            iterations = 0
            while self.truth(self.eval(cs[0])):
                iterations += 1
                self.total_loop_iterations += 1
                if iterations > self.loop_limit or self.total_loop_iterations > 1000000:
                    raise UnsupportedSource('Loop iteration budget exceeded')
                try:
                    with self.local_scope([cs[1]]): self.statement(cs[1])
                except LoopBreak: break
        elif k == 'SwitchStmt':
            if len(cs) != 2 or cs[1]['kind'] != 'CompoundStmt':
                raise UnsupportedSource('Unsupported switch structure')
            value = self.eval(cs[0])
            if type(value) is not int: raise UnsupportedSource('Unresolved switch condition')
            sequence, labels, default = [], {}, None
            for part in children(cs[1]):
                while part['kind'] in {'CaseStmt', 'DefaultStmt'}:
                    parts = children(part)
                    if part['kind'] == 'CaseStmt':
                        if len(parts) != 2: raise UnsupportedSource('Unsupported case range')
                        label = self.eval(parts[0])
                        if type(label) is not int: raise UnsupportedSource('Unknown case value')
                        labels[label] = len(sequence)
                    else: default = len(sequence)
                    part = parts[-1]
                sequence.append(part)
            if any(part['kind'] == 'DeclStmt' for part in sequence):
                raise UnsupportedSource('Switch-local declarations require an explicit nested block')
            start = labels.get(value, default)
            if start is not None:
                with self.local_scope(sequence):
                    try:
                        for part in sequence[start:]: self.statement(part)
                    except LoopBreak: pass
        elif k == 'NullStmt': pass
        elif k == 'BreakStmt': raise LoopBreak()
        elif k == 'ReturnStmt':
            raise FunctionReturn(self.eval(cs[0]) if cs else None)
        elif k == 'ForStmt':
            # Clang preserves an empty slot for a condition-variable declaration.
            raw = n.get('inner', [])
            if len(raw) != 5 or raw[1].get('kind') or not all(raw[i].get('kind') for i in (0,2,3,4)):
                raise UnsupportedSource('Unsupported for-loop structure')
            with self.local_scope([raw[0]]):
                self.statement(raw[0])
                iterations = 0
                while self.truth(self.eval(raw[2])):
                    iterations += 1
                    self.total_loop_iterations += 1
                    if iterations > self.loop_limit or self.total_loop_iterations > 1000000:
                        raise UnsupportedSource('Loop iteration budget exceeded')
                    try:
                        with self.local_scope([raw[4]]):
                            self.statement(raw[4])
                    except LoopBreak: break
                    self.eval(raw[3])
        else: self.eval(n)

    def call(self, name, expressions, site=None):
        args = [self.eval(n) for n in expressions]
        if self.kernel and name in self.analyzed_functions:
            return self.invoke_function(name, expressions, args)
        if self.kernel:
            if name == '__syncthreads' and not args and self.opaque_shared:
                if self.checked_barriers: self.thread_barriers.append(site)
                # Footprint-only abstraction. Shared values stay unknown across
                # barriers; barrier correctness/communication is not simulated.
                return None
            if name in self.config.get('scalar_device_functions', []):
                return self.scalar_device_call(name, args)
            if name in {'__log2f', '__powf'}:
                if name == '__log2f' and len(args) == 1 and isinstance(args[0], (int,float)) and args[0] > 0 and args[0] <= 2**24 and float(args[0]).is_integer():
                    value = int(args[0])
                    if value & (value - 1) == 0: return float(value.bit_length() - 1)
                if name == '__powf' and len(args) == 2 and args[0] == 2 and isinstance(args[1], (int,float)) and float(args[1]).is_integer() and 0 <= args[1] <= 24: return float(2 ** int(args[1]))
                raise UnsupportedSource('Intrinsic result requires an exact supported power of two')
            if name in {'sqrt', 'sqrtf', 'exp', 'log'} and len(args) == 1:
                if name in self.functions: raise UnsupportedSource('Math model cannot replace a source-defined function')
                return args[0] if isinstance(args[0], Unknown) else getattr(math, 'sqrt' if name == 'sqrtf' else name)(args[0])
            raise UnsupportedSource(f'Unsupported kernel call: {name}')
        if name in self.summaries:
            if any(self.contains_device_pointer(a) for a in args):
                raise UnsupportedSource('Host summaries cannot receive device pointers')
            summary = self.summaries[name]
            self.used_summaries.add(name)
            for key, value in summary.get('requires', {}).items():
                if args[int(key)] != self.configured(value):
                    raise UnsupportedSource('Host summary argument does not match its contract')
            for key, spec in summary.get('buffers', {}).items():
                target = args[int(key)]
                if not isinstance(target, Reference) or set(spec) != {'base', 'shape'} or target.pointer_base != spec['base']:
                    raise UnsupportedSource('Host buffer contract requires an output pointer')
                base = spec['base']
                if base not in {'int', 'float', 'double'}:
                    raise UnsupportedSource('Host buffer contract requires a numeric element type')
                size = self.layout(base)[0] * math.prod(self.positive(self.configured(d)) for d in spec['shape'])
                buffer = 'host_contract_' + str(self.host_allocation_count)
                self.host_allocation_count += 1
                self.host_sizes[buffer] = self.positive(size)
                self.store(target, Pointer(buffer, 0, base))
            self.configure_globals(summary.get('globals', {}))
            for key, value in summary.get('writes', {}).items():
                self.store(args[int(key)], self.configured(value))
            for key, count in summary.get('vectors', {}).items():
                vector = args[int(key)]
                if not isinstance(vector, Vector): raise UnsupportedSource('Summary expects a host vector')
                vector.count = self.positive(self.configured(count))
                self.host_sizes[vector.name] = vector.count * self.layout(vector.base)[0]
            if 'reads' in summary or 'writes_unknown' in summary:
                dependencies = set().union(*(self.host_provenance(args[int(i)]) for i in summary.get('reads', [])))
                for i in summary.get('writes_unknown', []):
                    self.mark_host_output(args[int(i)], dependencies)
            if 'returns_record' in summary:
                return {name: self.contract_value(spec) for name, spec in summary['returns_record'].items()}
            return UNKNOWN if summary.get('return_unknown') else self.configured(summary.get('return', 0))
        if name in self.analyzed_functions:
            return self.invoke_function(name, expressions, args)
        if self.opaque_host and name in {'exp','sqrt','cos','sin'} and name not in self.functions and len(args) == 1:
            return args[0] if isinstance(args[0], Unknown) else getattr(math, name)(args[0])
        if name == 'min' and len(args) == 2 and all(type(v) is int for v in args): return min(args)
        if name == 'sdkGetAverageTimerValue': return UNKNOWN
        if name in {'sdkCreateTimer','sdkStartTimer','sdkStopTimer','getLastCudaError'}:
            if name in self.functions: raise UnsupportedSource('Diagnostic model cannot replace a source-defined function')
            if any(self.contains_device_pointer(a) for a in args): raise UnsupportedSource('Diagnostic helper received device pointer')
            return 0
        if self.opaque_host and name in {'getopt_long', 'rand', 'srand'} and name in self.functions:
            raise UnsupportedSource('Host library model cannot replace a source-defined function')
        if self.opaque_host and name == 'getopt_long':
            if len(args) != 5 or type(args[0]) is not int or not isinstance(args[1], list) or len(args[1]) != args[0] or not isinstance(args[2], str):
                raise UnsupportedSource('Unsupported getopt_long arguments')
            index = self.cli_state['optind']
            if index >= args[0] or not args[1][index].startswith('-'): return -1
            option = args[1][index]
            if len(option) != 2 or option[1] == '-':
                raise UnsupportedSource('Only separate short options are supported')
            options = args[2].strip('"').lstrip(':')
            letter = option[1]
            if letter not in options or letter == ':':
                raise UnsupportedSource('Unknown command-line option')
            position = options.index(letter)
            if options[position + 1:position + 3] == '::':
                raise UnsupportedSource('Optional short-option arguments are unsupported')
            takes_value = options[position + 1:position + 2] == ':'
            if takes_value and index + 1 >= args[0]: raise UnsupportedSource('Missing option value')
            self.cli_state['optarg'] = args[1][index + 1] if takes_value else None
            self.cli_state['optind'] += 2 if takes_value else 1
            return ord(letter)
        if self.opaque_host and name in {'rand', 'srand'}:
            if (name == 'rand' and args) or (name == 'srand' and (len(args) != 1 or type(args[0]) is not int)):
                raise UnsupportedSource('Unsupported random initialization call')
            return UNKNOWN if name == 'rand' else None
        if name == 'cudaGetLastError' and not args: return 0
        if name == 'cudaGetDevice' and len(args) == 1:
            self.store(args[0], self.device.get('ordinal', 0))
            return 0
        if name == 'gettimeofday':
            if len(args) != 2 or not isinstance(args[0], Reference) or args[1] not in (None, 0):
                raise UnsupportedSource('Unsupported gettimeofday arguments')
            self.store(args[0], {'tv_sec': UNKNOWN, 'tv_usec': UNKNOWN})
            return UNKNOWN
        if name == 'cudaGetDeviceCount':
            self.store(args[0], self.positive(self.device['device_count']))
        elif name == 'cudaSetDevice':
            if args != [self.device.get('ordinal', 0)]: raise UnsupportedSource('Unconfigured device selection')
        elif name == 'cudaGetDeviceProperties':
            if args[1] != self.device.get('ordinal', 0): raise UnsupportedSource('Device ordinal not configured')
            self.store(args[0], {'maxGridSize': self.device['max_grid_size'], 'maxThreadsPerBlock': self.device['max_threads_per_block']})
        elif name == 'cudaMemGetInfo':
            self.store(args[0], self.device['free_memory_bytes']); self.store(args[1], self.device['total_memory_bytes'])
        elif name in {'cudaThreadSynchronize', 'cudaDeviceSynchronize'}: self.order.synchronize()
        elif name == 'cudaMalloc':
            target, size = args
            if not isinstance(target, Reference): raise UnsupportedSource('Allocation needs a direct pointer')
            label = target.label or target.key
            if self.config.get('qualified_allocations'):
                label = f'{self.current_function}:{label}:{len(self.allocated_names)}'
            if label in self.allocated_names: raise UnsupportedSource('Allocation reuse is unsupported')
            self.allocated_names.add(label)
            self.allocations[label] = self.positive(size)
            if self.regions and sum(self.allocations.values()) > self.region_limit:
                raise UnsupportedSource('Allocations exceed max_tracked_bytes; use a smaller input')
            # Type comes from the pointer declaration, not the void** cast at the API.
            base = target.pointer_base
            if base is None:
                declarations = [n for n in walk(body_of(self.functions[self.current_function])) if n['kind'] == 'VarDecl' and n.get('name') == target.key]
                if len(declarations) != 1: raise UnsupportedSource('Ambiguous allocation pointer declaration')
                base = self.clean(declarations[0]['type']['qualType']).rstrip('*').strip()
            self.layout(base)
            self.store(target, Pointer(label, 0, base, True))
        elif name in {'malloc', 'calloc'}:
            key = 'host_alloc_' + str(self.host_allocation_count)
            self.host_allocation_count += 1
            self.host_sizes[key] = self.positive(args[0] * args[1] if name == 'calloc' else args[0])
            return Pointer(key, 0, 'char')
        elif name == 'atof':
            if len(args) != 1 or not isinstance(args[0], str): raise UnsupportedSource('atof requires a configured string')
            value = float(args[0])
            if not math.isfinite(value): raise UnsupportedSource('Nonfinite numeric argument')
            return value
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
            if src.buffer == dst.buffer and max(src.offset,dst.offset) < min(src.offset+size,dst.offset+size):
                raise UnsupportedSource('Overlapping host memcpy is unsupported')
            values = list(self.host_integers.items())
            self.invalidate_integers(self.host_integers,dst.buffer,dst.offset,size)
            for (buffer,offset),value in values:
                if buffer == src.buffer and src.offset <= offset and offset+4 <= src.offset+size:
                    self.host_integers[dst.buffer,dst.offset+offset-src.offset] = value
            if self.track_host_dependencies:
                self.host_dependencies.setdefault(dst.buffer,set()).update(self.host_dependencies.get(src.buffer,()))
            return dst
        elif name == 'cudaFuncSetCacheConfig':
            if len(args) != 2 or args[0] not in self.functions or args[1] != 'cudaFuncCachePreferL1':
                raise UnsupportedSource('Unsupported cache configuration')
        elif name == 'cudaMemcpyToSymbol':
            if len(args) != 5 or args[3] not in (None, 0) or args[4] not in (None, 'cudaMemcpyHostToDevice'):
                raise UnsupportedSource('Unsupported constant-memory copy')
            dst, src, size = args[:3]
            if not isinstance(dst, Pointer) or dst.buffer not in self.constant_buffers or size != self.allocations[dst.buffer]:
                raise UnsupportedSource('Constant copy must cover the declared symbol')
            if isinstance(src, Reference):
                if src.scalar_type not in self.types or self.layout(src.scalar_type)[0] != size:
                    raise UnsupportedSource('Constant copy source record size mismatch')
            elif not isinstance(src, Pointer) or src.device or src.offset + size > self.host_sizes.get(src.buffer, 0):
                raise UnsupportedSource('Constant copy source is not a valid host buffer')
            self.operation = self.order.schedule([(dst.buffer, 'write')], 'default')
            self.write_buffer(dst.buffer, size, self.cpu, 'upload_constant', self.host_provenance(src))
        elif name == 'cudaMemset':
            ptr, value, size = args
            if not isinstance(ptr, Pointer) or not ptr.device or ptr.offset or value != 0:
                raise UnsupportedSource('Only whole-buffer zero memset is supported')
            if size != self.allocations[ptr.buffer]: raise UnsupportedSource('Partial memset is unsupported')
            self.operation = self.order.schedule([(ptr.buffer, 'write')], 'default')
            self.invalidate_integers(self.device_integers,ptr.buffer,0,size)
            self.write_buffer(ptr.buffer, size, self.acc, 'memset')
        elif name == 'cudaMemcpy':
            dst, src, size, direction = args
            self.positive(size)
            if direction == 'cudaMemcpyDeviceToDevice':
                return self.copy_device(dst, src, size)
            if direction == 'cudaMemcpyDeviceToHost' and isinstance(dst, Reference) and isinstance(src, Pointer) and src.device:
                if dst.scalar_type not in self.types or self.layout(dst.scalar_type)[0] != size or src.offset:
                    raise UnsupportedSource('Scalar download size/type mismatch')
                self.operation = self.order.schedule([(src.buffer, 'read')], 'default', blocking=True)
                transfer = self.read_buffer(src.buffer, size, self.cpu, 'download')
                self.store(dst, Unknown({transfer}) if self.track_host_dependencies else UNKNOWN)
                return 0
            if not isinstance(dst, Pointer) or not isinstance(src, Pointer): raise UnsupportedSource('Unsupported copy pointer')
            device, host = (dst, src) if direction == 'cudaMemcpyHostToDevice' else (src, dst)
            if not device.device or host.device or device.offset:
                raise UnsupportedSource('Only zero-offset device copies are supported')
            if host.offset < 0 or host.offset + size > self.host_sizes.get(host.buffer, 0): raise UnsupportedSource('Host copy exceeds configured buffer')
            if direction == 'cudaMemcpyHostToDevice':
                if self.track_integers:
                    self.invalidate_integers(self.device_integers, device.buffer, 0, size)
                    for (buffer, offset), value in self.host_integers.items():
                        if buffer == host.buffer and host.offset <= offset and offset + 4 <= host.offset + size:
                            self.device_integers[device.buffer, offset - host.offset] = value
                self.operation = self.order.schedule([(device.buffer, 'write')], 'default')
                parents = self.host_dependencies.get(host.buffer, set()) if self.track_host_dependencies else ()
                self.write_buffer(device.buffer, size, self.cpu, 'upload', parents)
                if parents: self.metadata[-1]['host_input_dependencies'] = sorted(parents)
                if self.track_integers and any(buffer == device.buffer for buffer, _ in self.device_integers):
                    self.metadata[-1]['known_integer_values'] = [[offset, value] for (buffer, offset), value in sorted(self.device_integers.items()) if buffer == device.buffer]
            elif direction == 'cudaMemcpyDeviceToHost':
                self.operation = self.order.schedule([(device.buffer, 'read')], 'default', blocking=True)
                transfer = self.read_buffer(device.buffer, size, self.cpu, 'download')
                self.invalidate_integers(self.host_integers, host.buffer, host.offset, size)
                if self.track_host_dependencies:
                    self.host_dependencies.setdefault(host.buffer, set()).add(transfer)
            else: raise UnsupportedSource('Unsupported copy direction')
        elif name == 'cudaFree':
            ptr = args[0]
            if not isinstance(ptr, Pointer) or not ptr.device or ptr.offset or ptr.buffer not in self.allocations:
                raise UnsupportedSource('Invalid cudaFree')
            self.order.synchronize()
            del self.allocations[ptr.buffer]
        elif name == 'free':
            self.free_host(args[0])
        elif name in {'printf', 'fprintf'}: pass
        elif name == 'exit':
            if self.config.get('allow_success_exit') and args == [0]: raise ProgramExit()
            raise UnsupportedSource('Configured input takes an exit/error path')
        else: raise UnsupportedSource(f'Unsupported host call (requires a reviewed contract): {name}')
        return 0

    def invoke_function(self, name, expressions, args):
        if name in self.call_stack or len(self.call_stack) >= 16:
            raise UnsupportedSource('Recursive or excessively nested calls are unsupported')
        fn = self.functions[name]
        if self.kernel and not any(c['kind'] == 'CUDADeviceAttr' for c in children(fn)):
            raise UnsupportedSource('Kernel helper must be a device function')
        params = [p for p in children(fn) if p['kind'] == 'ParmVarDecl']
        if len(args) != len(params): raise UnsupportedSource('Function argument count mismatch')
        bindings = [Binding(self.location(e)) if p['type']['qualType'].endswith('&') else a
                    for p, e, a in zip(params, expressions, args)]
        saved, saved_fn = self.env, self.current_function
        self.env = {p['name']: a for p, a in zip(params, bindings)}
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

    def copy_device(self, dst, src, size):
        if not self.regions or not all(isinstance(p, Pointer) and p.device and not p.offset for p in (src, dst)) or src.buffer == dst.buffer:
            raise UnsupportedSource('Device copy requires distinct zero-offset buffers and byte regions')
        if size > self.allocations[src.buffer] or size > self.allocations[dst.buffer]:
            raise UnsupportedSource('Device copy exceeds allocation')
        writers = self.region_writers.get(src.buffer, {})
        if any(i not in writers for i in range(size)): raise UnsupportedSource('Device copy reads uninitialized bytes')
        self.operation = self.order.schedule([(src.buffer,'read'), (dst.buffer,'write')], 'default')
        self.write_buffer(dst.buffer, size, self.mem, 'device_copy', {writers[i] for i in range(size)})
        transfer = self.nodes[-1]['id']
        self.metadata[-1]['source_buffer'] = src.buffer
        for i in range(size): self.region_readers.setdefault(src.buffer, {}).setdefault(i,set()).add(transfer)
        self.invalidate_integers(self.device_integers, dst.buffer, 0, size)
        for (buffer, offset), value in list(self.device_integers.items()):
            if buffer == src.buffer and offset + 4 <= size: self.device_integers[dst.buffer,offset] = value
        return 0

    def merge_scalar_branches(self, condition, branches):
        allowed = {'CompoundStmt', 'IfStmt', 'BinaryOperator', 'UnaryOperator',
                   'ImplicitCastExpr', 'ParenExpr', 'DeclRefExpr', 'FloatingLiteral', 'IntegerLiteral'}
        for branch in branches:
            for node in walk(branch):
                if node['kind'] not in allowed:
                    raise UnsupportedSource('Unknown branch must only update scalar locals')
                if node['kind'] == 'DeclRefExpr' and not isinstance(self.env.get(node['referencedDecl']['name']), (int, float, Unknown)):
                    raise UnsupportedSource('Unknown branch references nonlocal or nonscalar state')
                if node['kind'] == 'UnaryOperator' and node['opcode'] not in {'+', '-'}:
                    raise UnsupportedSource('Unknown branch contains unsupported mutation')
                if node['kind'] == 'BinaryOperator' and node['opcode'] not in {'=', '+', '-', '*', '/', '<', '<=', '>', '>=', '==', '!='}:
                    raise UnsupportedSource('Unknown branch contains unsupported operator')
        saved = self.env
        outcomes = []
        try:
            for branch in list(branches) + ([None] if len(branches) == 1 else []):
                self.env = dict(saved)
                if branch is not None: self.statement(branch)
                outcomes.append(self.env)
        finally:
            self.env = saved
        for name in saved:
            a, b = outcomes[0][name], outcomes[1][name]
            if a is b or (type(a) in (int, float) and type(b) is type(a) and a == b): continue
            saved[name] = unknown_from(a, b, condition)

    def contract_value(self, spec):
        if not isinstance(spec, dict) or 'shape' not in spec: return self.configured(spec)
        if set(spec) != {'shape', 'base'} or spec['base'] not in {'float', 'double', 'int'}:
            raise UnsupportedSource('Invalid numeric host record field contract')
        shape = [self.positive(self.configured(v)) for v in spec['shape']]
        if math.prod(shape)*self.layout(spec['base'])[0] > self.region_limit:
            raise UnsupportedSource('Host record field exceeds bounded analysis size')
        if len(shape) == 2:
            return PointerArray('host_matrix', spec['base'],
                {i: self.contract_value({'base': spec['base'], 'shape': [shape[1]]}) for i in range(shape[0])})
        if len(shape) != 1: raise UnsupportedSource('Host record field requires one or two dimensions')
        name = 'host_contract_' + str(self.host_allocation_count)
        self.host_allocation_count += 1
        self.host_sizes[name] = shape[0] * self.layout(spec['base'])[0]
        return Pointer(name, 0, spec['base'])

    def host_provenance(self, value):
        if isinstance(value, Unknown): return set(value.dependencies)
        if isinstance(value, Pointer): return set(self.host_dependencies.get(value.buffer, ()))
        if isinstance(value, Reference): return self.host_provenance(self.read(value))
        if isinstance(value, PointerArray): return set().union(*(self.host_provenance(v) for v in value.values.values()))
        if isinstance(value, dict): return set().union(*(self.host_provenance(v) for v in value.values()))
        return set()

    def mark_host_output(self, value, dependencies):
        if isinstance(value, Pointer) and not value.device:
            self.invalidate_integers(self.host_integers,value.buffer,value.offset,self.host_sizes[value.buffer]-value.offset)
            self.host_dependencies.setdefault(value.buffer, set()).update(dependencies)
        elif isinstance(value, Reference): self.store(value, Unknown(dependencies))
        elif isinstance(value, PointerArray):
            for pointer in value.values.values(): self.mark_host_output(pointer, dependencies)
        else: raise UnsupportedSource('Host output contract requires numeric storage')

    def scalar_device_call(self, name, args):
        """Prove a helper has no memory effects before abstracting its result."""
        fn = self.functions.get(name)
        if not fn or not any(c['kind'] == 'CUDADeviceAttr' for c in children(fn)):
            raise UnsupportedSource('Scalar helper must be a defined device function')
        params = [c for c in children(fn) if c['kind'] == 'ParmVarDecl']
        locals_ = [c for c in walk(body_of(fn)) if c['kind'] == 'VarDecl']
        local_ids = {c['id'] for c in params + locals_}
        numeric = {'int', 'unsigned int', 'float', 'double'}
        allowed = {'CompoundStmt', 'IfStmt', 'ReturnStmt', 'DeclStmt', 'VarDecl',
                   'DeclRefExpr', 'ImplicitCastExpr', 'ParenExpr', 'IntegerLiteral',
                   'FloatingLiteral', 'BinaryOperator', 'UnaryOperator'}
        if len(params) != len(args) or any(self.clean(c['type']['qualType']) not in numeric for c in params + locals_):
            raise UnsupportedSource('Scalar helper requires numeric arguments/locals')
        for c in walk(body_of(fn)):
            if c['kind'] not in allowed:
                raise UnsupportedSource('Scalar helper has unsupported effects')
            if c['kind'] == 'DeclRefExpr' and c['referencedDecl']['id'] not in local_ids:
                raise UnsupportedSource('Scalar helper references nonlocal state')
            if c['kind'] == 'BinaryOperator' and c['opcode'] not in {'=', '+', '-', '*', '/', '%', '<', '<=', '>', '>=', '==', '!='}:
                raise UnsupportedSource('Unsupported scalar helper operation')
            if c['kind'] == 'UnaryOperator' and c['opcode'] not in {'+', '-'}:
                raise UnsupportedSource('Unsupported scalar helper unary operation')
        if any(not isinstance(a, (int, float, Unknown)) for a in args):
            raise UnsupportedSource('Scalar helper received a nonnumeric argument')
        if any(isinstance(a, Unknown) for a in args): return UNKNOWN
        saved = self.env
        self.env = {p['name']: a for p, a in zip(params, args)}
        try:
            self.statement(body_of(fn))
        except FunctionReturn as result:
            return result.value
        finally:
            self.env = saved
        raise UnsupportedSource('Scalar helper did not return a value')

    def free_host(self, ptr):
        if not isinstance(ptr, Pointer) or ptr.device or ptr.offset or ptr.buffer not in self.host_sizes:
            raise UnsupportedSource('Invalid host free')
        del self.host_sizes[ptr.buffer]

    @staticmethod
    def contains_device_pointer(value):
        if isinstance(value, Pointer): return value.device
        if isinstance(value, Reference):
            return Evaluator.contains_device_pointer(value.owner.get(value.key, UNKNOWN))
        if isinstance(value, PointerArray):
            return any(Evaluator.contains_device_pointer(v) for v in value.values.values())
        if isinstance(value, dict):
            return any(Evaluator.contains_device_pointer(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(Evaluator.contains_device_pointer(v) for v in value)
        return False

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
        all_reads = set()
        access_history, barrier_sequences = {}, {}
        scalar_dependencies = set().union(*(a.dependencies for a in args if isinstance(a, Unknown)))
        try:
            for indices in itertools.product(*(range(d) for d in dims)):
                self.env = {p['name']: v for p, v in zip(params, args)}
                self.env.update(gridDim=grid, blockDim=block,
                                blockIdx=dict(zip(('x','y','z'), indices[:3])),
                                threadIdx=dict(zip(('x','y','z'), indices[3:])))
                self.thread_writes = set()
                self.thread_reads = set()
                self.thread_access_events, self.thread_barriers = [], []
                self.current_block = tuple(indices[:3])
                try:
                    self.statement(body_of(fn))
                except FunctionReturn:
                    pass
                if self.all_writes & self.thread_writes: raise UnsupportedSource('Multiple threads write the same bytes')
                if self.checked_barriers:
                    sequence = tuple(self.thread_barriers)
                    if self.current_block in barrier_sequences and barrier_sequences[self.current_block] != sequence:
                        raise UnsupportedSource('Nonuniform block barrier participation')
                    barrier_sequences[self.current_block] = sequence
                    for key, mode, event_block, phase in self.thread_access_events:
                        for prev_mode, prev_block, prev_phase in access_history.get(key, ()):
                            if mode == prev_mode: continue
                            read_phase, write_phase = (phase, prev_phase) if mode == 'read' else (prev_phase, phase)
                            if event_block != prev_block or read_phase >= write_phase:
                                raise UnsupportedSource('Cross-thread read/write overlap lacks a block barrier')
                    for key, mode, event_block, phase in self.thread_access_events:
                        access_history.setdefault(key, set()).add((mode, event_block, phase))
                if self.regions and not self.checked_barriers and (self.thread_reads & self.all_writes or self.thread_writes & all_reads):
                    raise UnsupportedSource('Cross-thread read/write overlap requires synchronization analysis')
                self.all_writes |= self.thread_writes
                all_reads |= self.thread_reads
        finally:
            self.env, self.kernel = host, False
        reads = {b for b, mode in self.accesses if mode == 'read'}
        writes = {b for b, mode in self.accesses if mode == 'write'}
        if reads & writes and not self.regions: raise UnsupportedSource('In-place kernels require additional race/dependency analysis')
        self.operation = self.order.schedule(list(self.accesses), 'default')
        sizes = {}
        for key, offsets in self.accesses.items():
            if not self.regions and offsets != set(range(len(offsets))): raise UnsupportedSource('Non-contiguous or nonzero-offset footprint')
            sizes[key] = len(offsets)
        parents = [self.read_buffer(b, sizes[b, 'read'], self.acc, name + ':read',
                                   self.accesses[b, 'read'] if self.regions else None) for b in self.allocations if b in reads]
        parents += sorted(scalar_dependencies)
        for b in self.allocations:
            if b in writes:
                self.write_buffer(b, sizes[b, 'write'], self.acc, name + ':write', parents,
                                  self.accesses[b, 'write'] if self.regions else None)
                if scalar_dependencies: self.metadata[-1]['host_input_dependencies'] = sorted(scalar_dependencies)
        self.launches.append(dict(kernel=name, grid=list(dims[:3]), block=list(dims[3:]), enumerated_threads=count))
        return 0

    def transfer(self, buffer, size, src, dst, stage, parents):
        i = len(self.nodes)
        self.nodes.append(dict(id=i, vol=size, src_comp=src, dest_comp=dst))
        self.metadata.append(dict(id=i, buffer=buffer, stage=stage, operation_id=self.operation['id'],
                                  stream='default', ordered_after_operations=self.operation['ordered_after']))
        self.edges.update((p, i) for p in parents)
        return i

    def region_transfer(self, buffer, size, src, dst, stage, parents, offsets, writing):
        offsets = set(range(size)) if offsets is None else set(offsets)
        if len(offsets) != size or not offsets or min(offsets) < 0 or max(offsets) >= self.allocations.get(buffer, 0):
            raise UnsupportedSource('Transfer region exceeds allocation')
        writers = self.region_writers.setdefault(buffer, {})
        readers = self.region_readers.setdefault(buffer, {})
        if not writing and any(o not in writers for o in offsets):
            raise UnsupportedSource('Read of uninitialized device region')
        dependencies = set(parents) | {writers[o] for o in offsets if o in writers}
        if writing:
            for o in offsets: dependencies.update(readers.get(o, ()))
        i = self.transfer(buffer, size, src, dst, stage, dependencies)
        ranges = []
        for o in sorted(offsets):
            if ranges and ranges[-1][1] == o: ranges[-1][1] = o + 1
            else: ranges.append([o, o + 1])
        self.metadata[-1]['byte_ranges'] = ranges  # half-open intervals
        for o in offsets:
            if writing: writers[o], readers[o] = i, set()
            else: readers.setdefault(o, set()).add(i)
        return i

    def read_buffer(self, buffer, size, dst, stage, offsets=None):
        if self.regions:
            return self.region_transfer(buffer, size, self.mem, dst, stage, (), offsets, False)
        if self.initialized.get(buffer, 0) < size: raise UnsupportedSource('Read of uninitialized device buffer')
        i = self.transfer(buffer, size, self.mem, dst, stage, self.writers.get(buffer, set()))
        self.readers.setdefault(buffer, set()).add(i)
        return i

    def write_buffer(self, buffer, size, src, stage, parents=(), offsets=None):
        if self.regions:
            return self.region_transfer(buffer, size, src, self.mem, stage, parents, offsets, True)
        if size != self.allocations.get(buffer): raise UnsupportedSource('Writes must cover the whole allocation')
        i = self.transfer(buffer, size, src, self.mem, stage,
                          set(parents) | self.writers.get(buffer, set()) | self.readers.get(buffer, set()))
        self.writers[buffer], self.readers[buffer], self.initialized[buffer] = {i}, set(), size

    def run(self):
        try:
            self.statement(body_of(self.functions[self.entry]))
        except ProgramExit:
            pass
        except FunctionReturn as result:
            if result.value not in (0, None): raise UnsupportedSource('Main returns failure for the configured input')
        if not self.nodes: raise UnsupportedSource('No transfers generated')
        if set(self.allocations) - self.constant_buffers: raise UnsupportedSource('Configured analysis requires all device buffers to be freed')
        return (dict(directed=True, multigraph=False, graph={}, nodes=self.nodes,
                     edges=[dict(source=s, target=t) for s, t in sorted(self.edges)]),
                self.metadata, dict(configuration=self.config, launches=self.launches,
                                    source_units={file.name: hashlib.sha256(data).hexdigest() for file, data in self.source_files.items()},
                                    used_host_summaries=sorted(self.used_summaries),
                                    **({'input_files': self.input_file_hashes} if self.input_file_hashes else {})))


def analyze_configured(path, config, cpu='cpu', mem='mem', acc='acc', clang=None):
    if len({cpu, mem, acc}) != 3 or not all((cpu, mem, acc)):
        raise UnsupportedSource('Component names must be nonempty and distinct')
    try:
        if config.get('integer_state'):
            from cuda_integer_model import IntegerEvaluator
            return IntegerEvaluator(path, config, cpu, mem, acc, clang).run()
        return Evaluator(path, config, cpu, mem, acc, clang).run()
    except (KeyError, TypeError, IndexError, StopIteration) as exc:
        raise UnsupportedSource(f'Invalid/unsupported configured source or configuration: {exc}') from exc
