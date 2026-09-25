"""Clang-based, fail-closed CUDA analysis for simple 1-D buffer workloads.

Models unique contiguous global-memory footprints per kernel, not instruction
traffic. No GPU code is executed. See workloads/TRANSFER_MODELS.md for scope.
"""
import json
import os
import re
from pathlib import Path
import subprocess


class UnsupportedSource(ValueError):
    pass


def children(n):
    return [c for c in n.get("inner", []) if c.get("kind")]


def walk(n):
    yield n
    for c in children(n):
        yield from walk(c)


def unwrap(n):
    while n["kind"] in {"ImplicitCastExpr", "CStyleCastExpr", "ParenExpr", "CXXDefaultArgExpr"}:
        n = wrapped_expression(n)
    return n


def wrapped_expression(n):
    cs = children(n)
    if len(cs) != 1:
        raise UnsupportedSource(f"Missing or ambiguous expression in Clang {n['kind']} node")
    return cs[0]


def constructor_arguments(expr, ast):
    """Resolve defaults from declarations when Clang omits them at call sites."""
    args = children(expr)
    if not any(a["kind"] == "CXXDefaultArgExpr" and not children(a) for a in args):
        return args
    if expr.get("type", {}).get("qualType") != "dim3":
        raise UnsupportedSource("Default argument resolution is only supported for dim3")
    signature = expr.get("ctorType", {}).get("qualType")
    matches = [n for n in walk(ast) if n["kind"] == "CXXConstructorDecl"
               and n.get("name") == "dim3" and signature
               and n.get("type", {}).get("qualType") == signature]
    candidates = []
    for declaration in matches:
        params = [n for n in children(declaration) if n["kind"] == "ParmVarDecl"]
        if len(params) != len(args):
            continue
        resolved = []
        for arg, param in zip(args, params):
            if arg["kind"] == "CXXDefaultArgExpr" and not children(arg):
                if not param.get("init") or len(children(param)) != 1:
                    break
                resolved.append(children(param)[0])
            else:
                resolved.append(arg)
        if len(resolved) == len(args):
            candidates.append(resolved)
    if len(candidates) != 1:
        raise UnsupportedSource("Cannot uniquely resolve dim3 default arguments from constructor declaration")
    return candidates[0]


def ref(n):
    n = unwrap(n)
    if n["kind"] == "UnaryOperator" and n["opcode"] == "&":
        return ref(children(n)[0])
    if n["kind"] != "DeclRefExpr":
        raise UnsupportedSource("Expected a direct buffer/scalar reference; aliases and pointer offsets are unsupported")
    return n["referencedDecl"]["name"]


SIZES = {"float": 4, "double": 8, "int": 4, "unsigned int": 4, "char": 1}


def affine(n, env):
    """Return coefficients (blockIdx.x, threadIdx.x, constant)."""
    while n["kind"] in {"ImplicitCastExpr", "ParenExpr", "CXXDefaultArgExpr"}:
        n = wrapped_expression(n)
    if n["kind"] == "CStyleCastExpr":
        raise UnsupportedSource("Explicit scalar/index casts are unsupported")
    if n["kind"] == "CXXStaticCastExpr":
        target = n.get("type", {}).get("qualType")
        value = scalar(wrapped_expression(n), env)
        if target not in {"size_t", "unsigned long", "unsigned long long"} or not 0 <= value < 2**64:
            raise UnsupportedSource("Only nonnegative integer-to-size_t static casts are supported")
        return (0, 0, value)
    k, cs = n["kind"], children(n)
    if k in {"IntegerLiteral", "FloatingLiteral"}:
        value = float(n["value"]) if k == "FloatingLiteral" else int(n["value"])
        return (0, 0, value)
    if k == "DeclRefExpr":
        name = ref(n)
        if name not in env:
            raise UnsupportedSource(f"Unresolved scalar: {name}")
        return env[name]
    if k == "MemberExpr":
        key = ref(cs[0]) + "." + n["name"]
        if key not in env:
            raise UnsupportedSource(f"Unsupported index: {key}")
        return env[key]
    if k == "UnaryExprOrTypeTraitExpr" and n.get("name") == "sizeof":
        typ = n.get("argType", {}).get("qualType")
        if typ not in SIZES:
            raise UnsupportedSource(f"Unsupported sizeof type: {typ}")
        return (0, 0, SIZES[typ])
    if k == "BinaryOperator":
        a, b = (affine(c, env) for c in cs)
        op = n["opcode"]
        if op in {"+", "-"}:
            return tuple(x + (y if op == "+" else -y) for x, y in zip(a, b))
        if op == "*" and not any(b[:2]):
            return tuple(x * b[2] for x in a)
        if op == "*" and not any(a[:2]):
            return tuple(x * a[2] for x in b)
        if not any(a[:2] + b[:2]):
            x, y = a[2], b[2]
            if op == "/" and y:
                return (0, 0, int(x / y))
            if op == "<<" and isinstance(x, int) and isinstance(y, int) and 0 <= y < 64:
                return (0, 0, x << y)
    raise UnsupportedSource(f"Unsupported scalar/index expression: {k} {n.get('opcode', '')}")


def scalar(n, env):
    b, t, v = affine(n, env)
    if b or t or not isinstance(v, int):
        raise UnsupportedSource("Expected a statically known integer")
    return v


class StreamOrder:
    """Prove ordering of conflicting accesses; keep it separate from logical edges."""
    def __init__(self):
        self.streams = {"default": set()}
        self.events = {}
        self.completed = set()
        self.last_write = {}
        self.reads = {}
        self.operations = []

    def stream(self, name):
        if name not in self.streams:
            raise UnsupportedSource(f"Unknown or destroyed stream: {name}")
        return self.streams[name]

    def schedule(self, accesses, stream, blocking=False):
        ordered = self.stream(stream) | self.completed
        for buf, mode in accesses:
            prior = set()
            if buf in self.last_write: prior.add(self.last_write[buf])
            if mode == "write": prior |= self.reads.get(buf, set())
            if not prior <= ordered:
                raise UnsupportedSource(f"Missing stream/event synchronization for {mode} of {buf} on {stream}")
        op = len(self.operations)
        self.operations.append(dict(id=op, stream=stream, ordered_after=sorted(ordered)))
        for buf, mode in accesses:
            if mode == "read": self.reads.setdefault(buf, set()).add(op)
        for buf, mode in accesses:
            if mode == "write":
                self.last_write[buf] = op
                self.reads[buf] = set()
        self.streams[stream] = ordered | {op}
        if blocking: self.completed |= self.streams[stream]
        return self.operations[-1]

    def synchronize(self, stream=None):
        if stream is None:
            for history in self.streams.values(): self.completed |= history
        else:
            self.completed |= self.stream(stream)


def checked_call(stmt, funcs):
    """Unwrap only a proven error-check-only function, never arbitrary helpers."""
    if stmt["kind"] != "CallExpr": return stmt
    cs = children(stmt)
    name = ref(cs[0])
    if name.startswith("cuda") or name not in funcs: return stmt
    params = [p for p in children(funcs[name]) if p["kind"] == "ParmVarDecl"]
    if not params or params[0]["type"]["qualType"] != "cudaError_t": return stmt
    body = next(c for c in children(funcs[name]) if c["kind"] == "CompoundStmt")
    statements = children(body)
    if len(statements) != 1 or statements[0]["kind"] != "IfStmt":
        raise UnsupportedSource("CUDA error wrapper must contain only an error-status check")
    branch = children(statements[0])
    condition = unwrap(branch[0])
    if (len(branch) != 2 or condition["kind"] != "BinaryOperator" or
            condition.get("opcode") != "!=" or
            ref(children(condition)[0]) != params[0]["name"] or
            ref(children(condition)[1]) != "cudaSuccess"):
        raise UnsupportedSource("Unsupported CUDA error wrapper condition")
    for node in walk(branch[1]):
        if node["kind"] == "CallExpr":
            if ref(children(node)[0]) not in {"fprintf", "printf", "exit", "cudaGetErrorString"}:
                raise UnsupportedSource("CUDA wrapper contains non-diagnostic calls")
        elif node["kind"] in {"CompoundAssignOperator", "UnaryOperator", "CUDAKernelCallExpr", "CXXMemberCallExpr"} or (node["kind"] == "BinaryOperator" and node.get("opcode") == "="):
            raise UnsupportedSource("CUDA wrapper contains unsupported side effects")
    actual = unwrap(cs[1])
    if actual["kind"] != "CallExpr" or not ref(children(actual)[0]).startswith("cuda"):
        raise UnsupportedSource("CUDA error wrapper must wrap one CUDA API call")
    for extra in cs[2:]:
        if any(n["kind"].endswith("CallExpr") or n["kind"] in {"UnaryOperator", "BinaryOperator"} for n in walk(extra)):
            raise UnsupportedSource("Side effects in CUDA wrapper arguments")
    return actual


def parse_source(path, clang=None):
    include = Path(__file__).parent / "source_analysis" / "include"
    command = [clang or os.environ.get("CLANGXX", "clang++"), "-x", "cuda", "--cuda-host-only",
               "-nocudainc", "-nocudalib", "-I" + str(include), "-std=c++17", "-fsyntax-only",
               "-Xclang", "-ast-dump=json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise UnsupportedSource("clang++ is required; install Clang or set CLANGXX") from exc
    if result.returncode:
        raise UnsupportedSource("Clang could not parse source:\n" + result.stderr)
    return json.loads(result.stdout)


def kernel_accesses(fn, args, host_env, allocations, blocks, threads):
    params = [c for c in children(fn) if c["kind"] == "ParmVarDecl"]
    if len(params) != len(args):
        raise UnsupportedSource("Kernel argument count mismatch")
    env = {"blockIdx.x": (1, 0, 0), "threadIdx.x": (0, 1, 0),
           "blockDim.x": (0, 0, threads), "gridDim.x": (0, 0, blocks)}
    pointers, accesses, shared_arrays = {}, {}, set()
    for node in walk(fn):
        if node["kind"] == "CallExpr" and ref(children(node)[0]) != "__syncthreads":
            raise UnsupportedSource("Kernel helper calls/atomics are unsupported")
    for p, arg in zip(params, args):
        typ = p["type"]["qualType"]
        if "*" in typ:
            buf = ref(arg)
            base = typ.replace("const", "").replace("*", "").strip()
            if buf not in allocations or base not in SIZES or buf in pointers.values():
                raise UnsupportedSource("Unknown, aliased, or unsupported kernel buffer")
            pointers[p["name"]] = buf
            accesses[p["name"]] = {"size": SIZES[base]}
        else:
            env[p["name"]] = affine(arg, host_env)

    def has_global(n):
        return any(c["kind"] == "DeclRefExpr" and ref(c) in pointers for c in walk(n))

    def footprint(index, conditions):
        b, t, c = index
        if (b, t) not in {(threads, 1), (1, 0), (0, 1), (0, 0)} or c != 0:
            raise UnsupportedSource("Only contiguous zero-based 1-D accesses are supported")
        lo, hi = 0, b * (blocks - 1) + t * (threads - 1)
        for expr, op, bound in conditions:
            if expr == index:
                if op == "<": hi = min(hi, bound - 1)
                elif op == "==": lo, hi = max(lo, bound), min(hi, bound)
            elif expr == (0, 1, 0) and op == "==" and bound == 0 and index == (1, 0, 0):
                pass  # One writer per block.
            elif expr == (0, 1, 0) and op == "==" and bound == 0 and index == (0, 0, 0):
                pass  # One scalar writer.
            else:
                raise UnsupportedSource("Cannot prove global access extent under this condition")
        if lo != 0 or hi < 0:
            raise UnsupportedSource("Empty or nonzero-offset access is unsupported")
        return hi + 1

    def visit(n, local, conditions=(), mode="read"):
        n = unwrap(n)
        k, cs = n["kind"], children(n)
        if k == "CompoundStmt":
            for child in cs: visit(child, local, conditions)
        elif k == "DeclStmt":
            for var in cs:
                if not children(var): continue
                if "[" in var["type"]["qualType"]:
                    if not any(x["kind"] == "CUDASharedAttr" for x in children(var)):
                        raise UnsupportedSource("Only shared local arrays are supported")
                    shared_arrays.add(var["name"])
                    continue
                local[var["name"]] = affine(children(var)[-1], local)
        elif k == "IfStmt":
            cond = unwrap(cs[0])
            if cond["kind"] != "BinaryOperator" or cond.get("opcode") not in {"<", "=="}:
                raise UnsupportedSource("Unsupported kernel condition")
            lhs, rhs = children(cond)
            constraint = (affine(lhs, local), cond["opcode"], scalar(rhs, local))
            visit(cs[1], dict(local), conditions + (constraint,))
            if len(cs) > 2:
                if has_global(cs[2]):
                    raise UnsupportedSource("Global accesses in else branches are unsupported")
                visit(cs[2], dict(local), conditions)
        elif k == "ForStmt":
            if has_global(n):
                raise UnsupportedSource("Global accesses inside loops are unsupported")
            # Shared-memory-only reduction loops don't contribute global transfers.
            if any(c["kind"] == "CallExpr" and ref(children(c)[0]) != "__syncthreads" for c in walk(n)):
                raise UnsupportedSource("Unknown call inside kernel loop")
            for child in walk(n):
                if child["kind"] in {"BinaryOperator", "CompoundAssignOperator", "UnaryOperator"} and child.get("opcode") in {"=", "+=", "-=", "++", "--"}:
                    lhs = unwrap(children(child)[0])
                    if lhs["kind"] == "DeclRefExpr" and ref(lhs) in local:
                        raise UnsupportedSource("Loop mutates a scalar used in access analysis")
        elif k in {"BinaryOperator", "CompoundAssignOperator"}:
            op = n["opcode"]
            if op in {"=", "+=", "-=", "*=", "/="}:
                visit(cs[1], local, conditions)
                if op != "=": visit(cs[0], local, conditions)
                visit(cs[0], local, conditions, "write")
                if unwrap(cs[0])["kind"] == "DeclRefExpr":
                    raise UnsupportedSource("Kernel scalar reassignment is unsupported")
            else:
                for child in cs: visit(child, local, conditions)
        elif k == "ArraySubscriptExpr":
            name = ref(cs[0])
            if name in pointers:
                count = footprint(affine(cs[1], local), conditions)
                if mode == "write" and affine(cs[1], local) == (0, 0, 0):
                    if blocks != 1 or (threads != 1 and ((0, 1, 0), "==", 0) not in conditions):
                        raise UnsupportedSource("Scalar writes must have one provable writer")
                size = count * accesses[name]["size"]
                if size > allocations[pointers[name]]:
                    raise UnsupportedSource("Kernel access exceeds allocation")
                accesses[name][mode] = max(accesses[name].get(mode, 0), size)
            elif name not in shared_arrays or has_global(cs[1]):
                raise UnsupportedSource("Unknown array or data-dependent indexing")
        elif k == "CallExpr":
            if ref(cs[0]) != "__syncthreads":
                raise UnsupportedSource("Kernel helper calls/atomics are unsupported")
        elif k == "DeclRefExpr" and ref(n) in pointers:
            raise UnsupportedSource("Pointer use outside a direct array subscript")
        elif k == "DeclRefExpr" and ref(n) not in local:
            raise UnsupportedSource(f"Unknown kernel scalar/global: {ref(n)}")
        elif k not in {"IntegerLiteral", "FloatingLiteral", "DeclRefExpr"}:
            raise UnsupportedSource(f"Unsupported kernel operation: {k}")

    body = next(c for c in children(fn) if c["kind"] == "CompoundStmt")
    visit(body, dict(env))
    return [(pointers[p], access) for p, access in accesses.items() if "read" in access or "write" in access]


def analyze(path, cpu="cpu", mem="mem", acc="acc", clang=None):
    if len({cpu, mem, acc}) != 3 or not all((cpu, mem, acc)):
        raise UnsupportedSource("Component names must be nonempty and distinct")
    ast = parse_source(path, clang)
    definitions = [n for n in children(ast) if n["kind"] == "FunctionDecl"
                   and "includedFrom" not in n.get("loc", {})
                   and any(c["kind"] == "CompoundStmt" for c in children(n))]
    funcs = {n.get("name"): n for n in definitions}
    if len(funcs) != len(definitions):
        raise UnsupportedSource("Overloaded function definitions are unsupported")
    if "main" not in funcs:
        raise UnsupportedSource("Source must define main")
    env, allocations, writers, readers, initialized = {}, {}, {}, {}, {}
    nodes, edges, descriptions = [], set(), []
    order = StreamOrder()
    active_operation = None
    vectors = {}
    handles = {}

    def host_buffer(n, size):
        n = unwrap(n)
        if n["kind"] == "CXXMemberCallExpr":
            member = children(n)[0]
            if len(children(n)) != 1 or member.get("name") != "data":
                raise UnsupportedSource("Only vector.data() host pointers are supported")
            name = ref(children(member)[0])
            if name not in vectors or size > vectors[name]:
                raise UnsupportedSource("Unknown or undersized host vector")
            return name
        return ref(n)

    def flag(n, expected):
        if ref(n) != expected:
            raise UnsupportedSource(f"Only {expected} is supported here")

    def handle(n, kind):
        name = ref(n)
        if handles.get(name) != kind:
            raise UnsupportedSource(f"Expected a declared {kind} handle: {name}")
        return name

    def transfer(buf, size, src, dst, stage, parents=()):
        if size <= 0: raise UnsupportedSource("Transfer size must be positive")
        i = len(nodes)
        nodes.append(dict(id=i, vol=size, src_comp=src, dest_comp=dst))
        descriptions.append(dict(id=i, buffer=buf, stage=stage))
        if active_operation is not None:
            descriptions[-1].update(operation_id=active_operation["id"],
                                    stream=active_operation["stream"],
                                    ordered_after_operations=active_operation["ordered_after"])
        edges.update((p, i) for p in parents)
        return i

    def read(buf, size, dst, stage):
        if buf not in allocations or initialized.get(buf, 0) < size:
            raise UnsupportedSource(f"Read of uninitialized buffer region: {buf}")
        i = transfer(buf, size, mem, dst, stage, writers.get(buf, []))
        readers.setdefault(buf, set()).add(i)
        return i

    def write(buf, size, src, stage, parents=()):
        if buf not in allocations or size != allocations[buf]:
            raise UnsupportedSource("Writes must cover the full allocation; partial writes need region tracking")
        i = transfer(buf, size, src, mem, stage,
                     set(parents) | set(writers.get(buf, [])) | readers.get(buf, set()))
        writers[buf], readers[buf], initialized[buf] = [i], set(), size

    body = next(c for c in children(funcs["main"]) if c["kind"] == "CompoundStmt")
    for stmt in children(body):
        stmt = checked_call(stmt, funcs)
        k, cs = stmt["kind"], children(stmt)
        if k == "DeclStmt":
            for var in cs:
                typ = var["type"]["qualType"]
                if typ in {"cudaStream_t", "cudaEvent_t"}:
                    if children(var): raise UnsupportedSource("Initialized/aliased CUDA handles are unsupported")
                    handles[var["name"]] = typ
                    continue
                match = re.fullmatch(r"std::vector<(float|double|int|unsigned int|char)>", typ)
                if match:
                    init = unwrap(children(var)[0])
                    if init["kind"] != "CXXConstructExpr":
                        raise UnsupportedSource("Unsupported vector initialization")
                    args = children(init)
                    if not args or any(a["kind"] != "CXXDefaultArgExpr" for a in args[1:]):
                        raise UnsupportedSource("Only size-initialized host vectors are supported")
                    size = scalar(args[0], env) * SIZES[match[1]]
                    if size <= 0: raise UnsupportedSource("Invalid host vector size")
                    vectors[var["name"]] = size
                    continue
                if "*" in typ:
                    if children(var):
                        init = unwrap(children(var)[0])
                        if init["kind"] != "CallExpr" or ref(children(init)[0]) != "malloc":
                            raise UnsupportedSource("Pointer aliases/initializers are unsupported")
                    continue
                if children(var): env[var["name"]] = affine(children(var)[-1], env)
        elif k == "CUDAKernelCallExpr":
            name = ref(cs[0])
            if name not in funcs or not any(c["kind"] == "CUDAGlobalAttr" for c in children(funcs[name])):
                raise UnsupportedSource(f"Missing kernel definition: {name}")
            config = children(cs[1])[1:]
            dims = []
            for dim in config[:2]:
                expr = unwrap(dim)
                if expr["kind"] != "CXXConstructExpr": raise UnsupportedSource("Unsupported launch geometry")
                xyz = [scalar(c, env) for c in constructor_arguments(expr, ast)]
                if len(xyz) != 3 or xyz[1:] != [1, 1] or xyz[0] <= 0:
                    raise UnsupportedSource("Only positive 1-D launches are supported")
                dims.append(xyz[0])
            if len(config) != 4: raise UnsupportedSource("Unsupported launch configuration")
            if config[2]["kind"] != "CXXDefaultArgExpr" and scalar(config[2], env) != 0:
                raise UnsupportedSource("Dynamic shared memory is unsupported")
            stream = "default" if config[3]["kind"] == "CXXDefaultArgExpr" else handle(config[3], "cudaStream_t")
            accesses = kernel_accesses(funcs[name], cs[2:], env, allocations, *dims)
            active_operation = order.schedule([(buf, mode) for buf, access in accesses
                                                for mode in ("read", "write") if mode in access], stream)
            inputs = [read(buf, a["read"], acc, name + ":read") for buf, a in accesses if "read" in a]
            for buf, a in accesses:
                if "write" in a: write(buf, a["write"], acc, name + ":write", inputs)
        elif k == "CallExpr":
            name, args = ref(cs[0]), cs[1:]
            if name == "cudaMalloc":
                buf, size = ref(args[0]), scalar(args[1], env)
                if buf in allocations or size <= 0: raise UnsupportedSource("Allocation reuse or invalid size")
                allocations[buf] = size
            elif name == "cudaMemcpy":
                size, direction = scalar(args[2], env), ref(args[3])
                if direction == "cudaMemcpyHostToDevice":
                    dst, src = ref(args[0]), host_buffer(args[1], size)
                    if src in allocations: raise UnsupportedSource("Invalid host source")
                    active_operation = order.schedule([(dst, "write")], "default")
                    write(dst, size, cpu, "upload")
                elif direction == "cudaMemcpyDeviceToHost":
                    dst, src = host_buffer(args[0], size), ref(args[1])
                    if dst in allocations: raise UnsupportedSource("Invalid host destination")
                    active_operation = order.schedule([(src, "read")], "default", blocking=True)
                    read(src, size, cpu, "download")
                else: raise UnsupportedSource("Only host/device copies are supported")
            elif name == "cudaStreamCreateWithFlags":
                stream = handle(args[0], "cudaStream_t")
                flag(args[1], "cudaStreamNonBlocking")
                if stream in order.streams: raise UnsupportedSource("Stream handle reuse")
                order.streams[stream] = set()
            elif name == "cudaEventCreateWithFlags":
                event = handle(args[0], "cudaEvent_t")
                flag(args[1], "cudaEventDisableTiming")
                if event in order.events: raise UnsupportedSource("Event handle reuse")
                order.events[event] = None
            elif name == "cudaEventRecord":
                event, stream = handle(args[0], "cudaEvent_t"), handle(args[1], "cudaStream_t")
                if event not in order.events: raise UnsupportedSource("Record on uncreated event")
                order.events[event] = set(order.stream(stream)) | order.completed
            elif name == "cudaStreamWaitEvent":
                stream, event = handle(args[0], "cudaStream_t"), handle(args[1], "cudaEvent_t")
                order.stream(stream)
                if scalar(args[2], env) != 0: raise UnsupportedSource("Unsupported wait flags")
                if order.events.get(event) is None:
                    raise UnsupportedSource("Wait on unrecorded or destroyed event")
                # Snapshot at the wait call: later re-recording does not modify this wait.
                order.streams[stream] |= order.events[event]
            elif name == "cudaDeviceSynchronize":
                order.synchronize()
            elif name == "cudaStreamSynchronize":
                order.synchronize(handle(args[0], "cudaStream_t"))
            elif name == "cudaEventDestroy":
                event = handle(args[0], "cudaEvent_t")
                if event not in order.events: raise UnsupportedSource("Destroy of unknown event")
                del order.events[event]
            elif name == "cudaStreamDestroy":
                stream = handle(args[0], "cudaStream_t")
                if not order.stream(stream) <= order.completed:
                    raise UnsupportedSource("Synchronize streams before destroying them")
                del order.streams[stream]
            elif name == "cudaFree":
                buf = ref(args[0])
                if buf not in allocations:
                    raise UnsupportedSource(f"Free of unknown allocation: {buf}")
                pending = order.reads.get(buf, set()) | ({order.last_write[buf]} if buf in order.last_write else set())
                if not pending <= order.completed:
                    raise UnsupportedSource("Synchronize buffer users before cudaFree")
                allocations.pop(buf)
                initialized.pop(buf, None)
                writers.pop(buf, None)
                readers.pop(buf, None)
                order.reads.pop(buf, None)
                order.last_write.pop(buf, None)
            elif name not in {"printf", "fprintf", "free", "cudaGetLastError"}:
                raise UnsupportedSource(f"Unsupported host call: {name}")
        elif k == "ForStmt":
            # Permit host initialization only, never hidden launches/device calls.
            for n in walk(stmt):
                if n["kind"] == "CUDAKernelCallExpr" or (n["kind"] == "DeclRefExpr" and ref(n) in set(allocations) | set(handles)):
                    raise UnsupportedSource("Host loops involving device operations are unsupported")
                if n["kind"] == "CallExpr" and ref(children(n)[0]) not in {"printf", "fprintf"}:
                    raise UnsupportedSource("Unsupported call inside host loop")
                if n["kind"] == "CXXOperatorCallExpr":
                    if ref(children(n)[0]) != "operator[]" or ref(children(n)[1]) not in vectors:
                        raise UnsupportedSource("Only vector element access is supported in host loops")
                if n["kind"] == "CXXMemberCallExpr":
                    raise UnsupportedSource("Vector mutation/member calls inside host loops are unsupported")
            for node in walk(stmt):
                if node["kind"] in {"BinaryOperator", "CompoundAssignOperator", "UnaryOperator"} and node.get("opcode") in {"=", "+=", "-=", "*=", "/=", "++", "--"}:
                    lhs = unwrap(children(node)[0])
                    if lhs["kind"] == "DeclRefExpr" and ref(lhs) in env:
                        # Host validation counters may change; they cannot later be used as static parameters.
                        env.pop(ref(lhs))
        elif k == "ReturnStmt":
            break
        else:
            raise UnsupportedSource(f"Unsupported host statement: {k}")
    if not nodes: raise UnsupportedSource("No supported transfers found")
    return dict(directed=True, multigraph=False, graph={}, nodes=nodes,
                edges=[dict(source=s, target=t) for s, t in sorted(edges)]), descriptions
