"""Static structure extraction via Python `ast` (M1 path for Mode 1 blocks).

NOTE: the eng-review-locked decision is to use bundled tree-sitter for the no-env
structure path (so blocks render without a Python interpreter). This ast-based version
is the faster M1 stand-in; swap to tree-sitter later. Tracked in BUILD_STATUS.md.
"""
import ast
from typing import Any, Dict, List

# tensor methods whose dim argument implies a minimum rank
_DIM_METHODS = {"sum", "mean", "prod", "amax", "amin", "max", "min", "argmax", "argmin",
                "norm", "std", "var", "logsumexp", "softmax", "log_softmax", "cumsum",
                "cumprod", "any", "all", "unbind", "chunk", "split", "tensor_split",
                "squeeze", "select", "narrow", "gather", "index_select", "flip"}


def shape_reqs(fn_node, skip_first: bool = False) -> List[Dict[str, Any]]:
    """Static per-PARAMETER rank constraints for one function, from how the param is used:
      • `B, L, D = x.shape` / `= x.size()`  -> rank EXACTLY 3 (a starred target -> minimum)
      • `x.permute(0, 2, 1, 3)`             -> rank exactly 4
      • `x.transpose(1, 2)` / `x.shape[2]` / `x.sum(dim=2)` -> rank ≥ 3
      • torch.einsum subscripts             -> exact (or ≥ with '...')
    Params with no such evidence are 'free' — rank-polymorphic AS FAR AS THIS FUNCTION
    pins it (a callee may still constrain it). Returns [{name, kind: exact|min|free,
    rank, via}] in parameter order. skip_first drops the bound arg of a METHOD whatever
    it's called (self, s, cls, …); self/cls are name-filtered regardless."""
    arg_names = [a.arg for a in fn_node.args.args]
    if skip_first and arg_names:
        arg_names = arg_names[1:]
    params = [a for a in arg_names if a not in ("self", "cls")]
    pset = set(params)
    info: Dict[str, Dict[str, Any]] = {p: {"kind": "free", "rank": None, "via": None} for p in params}

    def set_exact(p, r, via):
        if info[p]["kind"] != "exact":
            info[p].update(kind="exact", rank=r, via=via)

    def bump_min(p, r, via):
        d = info[p]
        if d["kind"] == "exact":
            return
        if d["kind"] == "free" or r > (d["rank"] or 0):
            info[p].update(kind="min", rank=r, via=via)

    def int_of(a):
        if isinstance(a, ast.Constant) and type(a.value) is int:
            return a.value
        if (isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub)
                and isinstance(a.operand, ast.Constant) and type(a.operand.value) is int):
            return -a.operand.value
        return None

    dim_min = lambda k: k + 1 if k >= 0 else -k  # rank needed to index dim k

    for n in ast.walk(fn_node):
        # B, L, D = x.shape   |   B, L, D = x.size()
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple):
            v, recv = n.value, None
            if isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name):
                recv = v.value.id
            elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "size"
                  and not v.args and isinstance(v.func.value, ast.Name)):
                recv = v.func.value.id
            if recv in pset:
                elts = n.targets[0].elts
                names = ", ".join("*" if isinstance(e, ast.Starred) else getattr(e, "id", "_") for e in elts)
                via = f"{names} = {recv}.shape"
                if any(isinstance(e, ast.Starred) for e in elts):
                    bump_min(recv, len(elts) - 1, via)
                else:
                    set_exact(recv, len(elts), via)
        # x.shape[i]
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "shape"
                and isinstance(n.value.value, ast.Name) and n.value.value.id in pset):
            i = int_of(n.slice)
            if i is not None:
                bump_min(n.value.value.id, dim_min(i), f"shape[{i}]")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            # torch.einsum('bld,dk->blk', x, w): subscript length pins each operand's rank
            if n.func.attr == "einsum" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                subs = n.args[0].value.split("->")[0].split(",")
                for i, a in enumerate(n.args[1:]):
                    if isinstance(a, ast.Name) and a.id in pset and i < len(subs):
                        sub = subs[i].strip()
                        via = f"einsum '{n.args[0].value}'"
                        if "..." in sub:
                            bump_min(a.id, len(sub.replace("...", "").replace(" ", "")), via)
                        else:
                            set_exact(a.id, len(sub.replace(" ", "")), via)
            # direct method calls on a param
            if isinstance(n.func.value, ast.Name) and n.func.value.id in pset:
                p, m = n.func.value.id, n.func.attr
                ints = [i for i in (int_of(a) for a in n.args) if i is not None]
                kw = {k.arg: int_of(k.value) for k in n.keywords if k.arg}
                if m == "size" and len(n.args) == 1 and ints:
                    bump_min(p, dim_min(ints[0]), f"size({ints[0]})")
                elif m == "permute" and len(ints) >= 2 and len(ints) == len(n.args):
                    set_exact(p, len(ints), f"permute({', '.join(map(str, ints))})")
                elif m == "transpose" and len(ints) == 2:
                    bump_min(p, max(dim_min(ints[0]), dim_min(ints[1])), f"transpose({ints[0]}, {ints[1]})")
                elif m in _DIM_METHODS:
                    k = kw.get("dim", ints[0] if ints else None)
                    if k is not None:
                        bump_min(p, dim_min(k), f"{m}(dim={k})")
    return [{"name": p, **info[p]} for p in params]


def structure_file(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src, path)
    except Exception as e:
        return {"path": path, "language": "python", "functions": [], "note": f"parse error: {e}"}

    functions: List[Dict[str, Any]] = []

    def add_fn(node, class_name):
        params = [
            {"name": a.arg, "type": _unparse(a.annotation)}
            for a in node.args.args
        ]
        intermediates = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        intermediates.append({"line": stmt.lineno, "name": tgt.id})
        functions.append(
            {
                "name": node.name,
                # The enclosing class, so the cockpit can disambiguate the N `forward`s in a
                # transformer file (MultiHeadSelfAttention.forward vs FeedForward.forward …).
                "className": class_name,
                "startLine": node.lineno,
                "endLine": getattr(node, "end_lineno", node.lineno),
                "params": params,
                "returns": _unparse(node.returns),
                "intermediates": intermediates,
                # static per-param rank constraints (exact / min / free) for the cockpit.
                # skip_first drops a method's receiver even when it isn't named self/cls (e.g. `s`).
                "shapeReqs": shape_reqs(node, skip_first=class_name is not None),
            }
        )

    # Top-level functions + one level of class methods (the common case). `name` stays the
    # bare method name so trace lookups by name keep working; `className` is display-only.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_fn(node, None)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_fn(sub, node.name)
    functions.sort(key=lambda f: f["startLine"])
    return {"path": path, "language": "python", "functions": functions}


def _unparse(node: Any):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None
