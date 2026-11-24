"""Source-backed, nonvirtual C++ objects for the bounded CUDA evaluator.

Objects have field identity, not a guessed ABI. They cannot be copied to/from
CUDA allocations. Shared numeric arrays keep the existing opaque-value model.
"""
import re
from dataclasses import dataclass
from cuda_source_model import UnsupportedSource, children


@dataclass
class SourceObject:
    record: str
    fields: dict
    shared: bool = False
    base: object = None


def canonical_type(text):
    text = re.sub(r'^(?:(?:const|volatile|class|struct)\s+)+', '', text.strip())
    text = re.sub(r'\btrue\b', '-1', text)
    text = re.sub(r'\bfalse\b', '0', text)
    return re.sub(r'\s+', '', text)


class ObjectModel:
    def __init__(self, declarations):
        self.records = {}
        self.methods = {}
        self.method_owners = {}
        self.record_names = {}
        self.names = {}
        self.seen = set()
        # Explicitly selected source declarations only; never model STL internals.
        for node, _ in declarations:
            if node['kind'] in {'NamespaceDecl', 'ClassTemplateDecl', 'ClassTemplateSpecializationDecl', 'CXXRecordDecl'}:
                self.collect(node, ())

    def collect(self, node, context):
        ident = node.get('id')
        if ident in self.seen:
            return
        self.seen.add(ident)
        kind = node['kind']
        if kind == 'NamespaceDecl':
            if not node.get('name'): raise UnsupportedSource('Anonymous namespace object identities are unsupported')
            context = (*context, node['name'])
        if kind in {'CXXRecordDecl', 'ClassTemplateSpecializationDecl'}:
            if not node.get('completeDefinition'):
                return
            name = node.get('name', '')
            arguments = []
            for arg in children(node):
                if arg['kind'] != 'TemplateArgument': continue
                if 'type' in arg: arguments.append(arg['type']['qualType'])
                elif 'value' in arg:
                    value = arg['value']
                    arguments.append(str(value).lower() if isinstance(value, bool) else str(value))
                else: return  # dependent/unsupported specialization
            if arguments: name += '<' + ','.join(arguments) + '>'
            owner = '::'.join((*context, name))
            self.records[ident] = node
            self.record_names[ident] = owner
            self.names.setdefault(canonical_type(owner), []).append(ident)
            context = (*context, name)
        for child in children(node):
            if child['kind'] == 'CXXMethodDecl' and any(c['kind'] == 'CompoundStmt' for c in children(child)):
                self.methods[child['id']] = child
                self.method_owners[child['id']] = ident
            elif child['kind'] in {'NamespaceDecl','ClassTemplateDecl','ClassTemplateSpecializationDecl','CXXRecordDecl','FunctionTemplateDecl'}:
                if child['kind'] == 'FunctionTemplateDecl':
                    for method in children(child):
                        if method['kind'] == 'CXXMethodDecl' and any(c['kind'] == 'CompoundStmt' for c in children(method)):
                            self.methods[method['id']] = method
                            self.method_owners[method['id']] = ident
                else: self.collect(child, context)

    def record_for(self, typ):
        key = canonical_type(typ)
        matches = set(self.names.get(key, ()))
        # Multiple TUs can contain equivalent instantiated definitions. Never
        # select an arbitrary receiver: calls bind to their own declaration ID.
        if len(matches) != 1:
            raise UnsupportedSource(f'Object type requires one source definition: {typ}')
        return next(iter(matches))

    def construct(self, typ, evaluator, shared=False, depth=0):
        from cuda_configured_model import UNKNOWN, SharedView
        if depth > 16: raise UnsupportedSource('Object nesting exceeds bounded limit')
        ident = self.record_for(typ)
        record = self.records[ident]
        bases = record.get('bases', [])
        if len(bases) > 1 or any(b.get('isVirtual') for b in bases) or record.get('tagUsed') == 'union' or record.get('definitionData', {}).get('isPolymorphic'):
            raise UnsupportedSource('Multiple/virtual inheritance, unions and virtual dispatch are unsupported')
        base = None
        if bases:
            spec = bases[0]['type']
            base = self.construct(spec.get('desugaredQualType', spec['qualType']), evaluator, shared, depth+1)
        data = record.get('definitionData', {})
        if not data.get('defaultCtor', {}).get('trivial') or not data.get('dtor', {}).get('trivial'):
            raise UnsupportedSource('Object requires nontrivial construction/destruction')
        fields = {}
        for field in children(record):
            if field['kind'] != 'FieldDecl': continue
            if field.get('isBitfield') or children(field):
                raise UnsupportedSource('Object bitfields and field initializers are unsupported')
            typ = field['type'].get('desugaredQualType', field['type']['qualType'])
            clean = evaluator.clean(typ)
            array = re.fullmatch(r'(.+?)\s*((?:\[\d+\])+)', clean)
            if array:
                element_type = array[1].strip()
                if not shared or element_type not in evaluator.types or evaluator.types[element_type][2]:
                    raise UnsupportedSource('Object arrays require shared numeric storage')
                fields[field['name']] = SharedView(tuple(int(n) for n in re.findall(r'\d+',array[2])), element_type)
            elif clean.endswith('*') or clean in evaluator.types:
                if shared: raise UnsupportedSource('Shared object scalar fields require a shared-state model')
                fields[field['name']] = UNKNOWN
            else:
                fields[field['name']] = self.construct(typ, evaluator, shared, depth+1)
        return SourceObject(ident, fields, shared, base)

    def base_view(self, value, typ):
        target = self.record_for(typ.rstrip(' *&'))
        while isinstance(value, SourceObject):
            if value.record == target: return value
            value = value.base
        raise UnsupportedSource('Base conversion does not match an object subobject')

    def invoke(self, evaluator, ident, expressions, receiver=None):
        if ident not in self.methods:
            raise UnsupportedSource('Method body is absent from selected source')
        method = self.methods[ident]
        if method.get('virtual') or method.get('pure'):
            raise UnsupportedSource('Virtual method dispatch is unsupported')
        static = method.get('storageClass') == 'static'
        if not static and (not isinstance(receiver, SourceObject) or receiver.record != self.method_owners[ident]):
            raise UnsupportedSource('Method receiver does not match its source record')
        arguments = []
        for argument in children(method):
            if argument['kind'] != 'TemplateArgument': continue
            if 'type' in argument: arguments.append(argument['type']['qualType'])
            elif 'value' in argument: arguments.append(str(argument['value']))
            else: raise UnsupportedSource('Unsupported method template argument')
        specialization = '<' + ','.join(arguments) + '>' if arguments else ''
        unit = ident.rsplit(':0x', 1)[0] if ':0x' in ident else 'entry'
        key = '@method:' + unit + ':' + self.record_names[self.method_owners[ident]] + '::' + method['name'] + specialization + ' [' + method['type']['qualType'] + ']'
        if key in evaluator.functions and evaluator.functions[key] is not method:
            raise UnsupportedSource('Ambiguous source method identity')
        evaluator.functions[key] = method
        args = [evaluator.eval(n) for n in expressions]
        # invoke_function owns argument binding, recursion limits and thread context.
        try:
            return evaluator.invoke_function(key, expressions, args, receiver=receiver)
        except UnsupportedSource as error:
            error.add_note('While evaluating ' + key)
            raise
