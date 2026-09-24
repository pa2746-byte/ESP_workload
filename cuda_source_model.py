"""Clang-based, fail-closed CUDA analysis for simple 1-D buffer workloads.

Models unique contiguous global-memory footprints per kernel, not instruction
traffic. No GPU code is executed. See workloads/TRANSFER_MODELS.md for scope.
"""
import json
import os
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
                   and any(c["kind"] == "CompoundStmt" for c in children(n))]
    funcs = {n.get("name"): n for n in definitions}
    if len(funcs) != len(definitions):
        raise UnsupportedSource("Overloaded function definitions are unsupported")
    if "main" not in funcs:
        raise UnsupportedSource("Source must define main")
    env, allocations, writers, readers, initialized = {}, {}, {}, {}, {}
    nodes, edges, descriptions = [], set(), []

    def transfer(buf, size, src, dst, stage, parents=()):
        if size <= 0: raise UnsupportedSource("Transfer size must be positive")
        i = len(nodes)
        nodes.append(dict(id=i, vol=size, src_comp=src, dest_comp=dst))
        descriptions.append(dict(id=i, buffer=buf, stage=stage))
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
        k, cs = stmt["kind"], children(stmt)
        if k == "DeclStmt":
            for var in cs:
                typ = var["type"]["qualType"]
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
            if any(c["kind"] != "CXXDefaultArgExpr" for c in config[2:]):
                raise UnsupportedSource("Explicit streams/dynamic shared memory are unsupported")
            accesses = kernel_accesses(funcs[name], cs[2:], env, allocations, *dims)
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
                dst, src, size, direction = ref(args[0]), ref(args[1]), scalar(args[2], env), ref(args[3])
                if direction == "cudaMemcpyHostToDevice":
                    if src in allocations: raise UnsupportedSource("Invalid host source")
                    write(dst, size, cpu, "upload")
                elif direction == "cudaMemcpyDeviceToHost":
                    if dst in allocations: raise UnsupportedSource("Invalid host destination")
                    read(src, size, cpu, "download")
                else: raise UnsupportedSource("Only host/device copies are supported")
            elif name == "cudaFree":
                buf = ref(args[0])
                if buf not in allocations:
                    raise UnsupportedSource(f"Free of unknown allocation: {buf}")
                allocations.pop(buf)
                initialized.pop(buf, None)
                writers.pop(buf, None)
                readers.pop(buf, None)
            elif name not in {"printf", "free", "cudaDeviceSynchronize"}:
                raise UnsupportedSource(f"Unsupported host call: {name}")
        elif k == "ForStmt":
            # Permit host initialization only, never hidden launches/device calls.
            if any(n["kind"] in {"CallExpr", "CUDAKernelCallExpr"} or
                   (n["kind"] == "DeclRefExpr" and ref(n) in allocations) for n in walk(stmt)):
                raise UnsupportedSource("Host loops involving device operations are unsupported")
            for node in walk(stmt):
                if node["kind"] in {"BinaryOperator", "CompoundAssignOperator", "UnaryOperator"} and node.get("opcode") in {"=", "+=", "-=", "*=", "/=", "++", "--"}:
                    lhs = unwrap(children(node)[0])
                    if lhs["kind"] == "DeclRefExpr" and ref(lhs) in env:
                        raise UnsupportedSource("Host initialization loop mutates a graph parameter")
        elif k == "ReturnStmt":
            break
        else:
            raise UnsupportedSource(f"Unsupported host statement: {k}")
    if not nodes: raise UnsupportedSource("No supported transfers found")
    return dict(directed=True, multigraph=False, graph={}, nodes=nodes,
                edges=[dict(source=s, target=t) for s, t in sorted(edges)]), descriptions
