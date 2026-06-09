"""Trace core for the cockpit's Mode 1 (shape-annotated blocks).

Runs a callable under sys.settrace and captures every tensor local's shape per source
line in the target file. The `line` event fires BEFORE the line runs, so locals are
attributed to the PREVIOUS line (the one that just produced them). Duck-types tensors
so the helper need not import torch unless the user's code does.

Inputs are auto-synthesized (dtype/layer-aware), or pinned by the user with a magic
comment directly above the def/class:
    # fusion: input = torch.randint(0, 1000, (4, 32))   # tuple -> *args for multi-input
    # fusion: model = MyModel(dim=8)                     # construct with specific args
"""
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple


def _is_tensor(v: Any) -> bool:
    root = type(v).__module__.split(".")[0]
    return root in ("torch", "numpy") and hasattr(v, "shape") and hasattr(v, "dtype")


def trace_callable(fn: Callable[[], Any], target_file: str):
    """Return (records, error).
    records: { lineno: { varname: {"shape":[...], "dtype":str, "changed":bool} } }
    error:   the Exception if fn() raised (e.g. a shape-mismatch RuntimeError), else None.
    """
    records: Dict[int, Dict[str, Any]] = {}
    prev_line: Dict[int, Optional[int]] = {}
    last_shape: Dict[str, Tuple[int, ...]] = {}
    crash_line: list = [None]  # first (deepest) exception line == the real crash site

    def snapshot(frame, lineno: int) -> None:
        slot = records.setdefault(lineno, {})
        for name, val in list(frame.f_locals.items()):
            if not _is_tensor(val):
                continue
            try:
                shape = tuple(int(d) for d in val.shape)
            except Exception:
                continue
            changed = last_shape.get(name) != shape
            last_shape[name] = shape
            slot[name] = {
                "shape": list(shape),
                "dtype": str(getattr(val, "dtype", "")).replace("torch.", ""),
                "changed": changed,
            }

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != target_file:
            return None  # skip torch internals / other files
        fid = id(frame)
        if event == "call":
            prev_line[fid] = None
            return tracer
        if event in ("line", "return", "exception"):
            pl = prev_line.get(fid)
            if pl is not None:
                snapshot(frame, pl)
            if event == "exception" and crash_line[0] is None:
                crash_line[0] = frame.f_lineno  # raising line in the deepest traced frame
            if event == "line":
                prev_line[fid] = frame.f_lineno
            if event == "return" and _is_tensor(arg):
                # the returned value has no local name (e.g. `return self.fc2(h)`),
                # so attach its shape to the return line under a synthetic "return".
                try:
                    shape = tuple(int(d) for d in arg.shape)
                    changed = last_shape.get("return") != shape
                    last_shape["return"] = shape
                    records.setdefault(frame.f_lineno, {})["return"] = {
                        "shape": list(shape),
                        "dtype": str(getattr(arg, "dtype", "")).replace("torch.", ""),
                        "changed": changed,
                    }
                except Exception:
                    pass
        return tracer

    sys.settrace(tracer)
    try:
        fn()
        err: Optional[BaseException] = None
    except Exception as e:  # shape mismatch etc.
        err = e
    finally:
        sys.settrace(None)
    return records, err, crash_line[0]


def trace_file(path: str):
    """Exec a .py file under the tracer (opt-in "Trace this file").
    Returns (records, error_str_or_None, crash_line_or_None). Runs the module as
    __main__, so a file that runs a forward pass on execution gets real shapes."""
    import os
    import io
    import contextlib

    path = os.path.abspath(path)
    with open(path) as f:
        code = compile(f.read(), path, "exec")

    # The traced file's own print()/stderr must NOT pollute the JSON-RPC stdout
    # channel — capture and discard it while the module runs.
    sink = io.StringIO()

    def run():
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(code, {"__name__": "__main__", "__file__": path})

    records, err, crash_line = trace_callable(run, path)
    return records, (f"{type(err).__name__}: {err}" if err else None), crash_line


# --- input synthesis & invocation ------------------------------------------------

_DIRECTIVE_RE = re.compile(r"^\s*#\s*fusion:\s*(input|model)\s*=\s*(.+?)\s*$")


class _CannotInvoke(Exception):
    """We can't auto-call a target (needs args / directive failed). The message explains
    why; it is NOT a runtime/shape error (those propagate from the traced call instead)."""


def _parse_directives(src: str):
    """Find `# fusion: input = <expr>` / `# fusion: model = <expr>` lines (1-based)."""
    out = []
    for i, line in enumerate(src.splitlines(), start=1):
        m = _DIRECTIVE_RE.match(line)
        if m:
            out.append((i, m.group(1), m.group(2)))
    return out


def _pick_directive(directives, kind: str, def_line: int, window: int = 6):
    """The closest `kind` directive sitting just ABOVE def_line (within `window` lines)."""
    if not def_line:
        return None
    best = None
    for (ln, k, expr) in directives:
        if k == kind and def_line - window <= ln < def_line and (best is None or ln > best[0]):
            best = (ln, expr)
    return best[1] if best else None


def _eval_expr(expr: str, g: dict):
    """Eval a directive expression in the module's namespace + torch conveniences.
    (No more dangerous than exec'ing the user's own module, which we already do.)"""
    import torch

    ns = dict(g)
    ns.setdefault("torch", torch)
    for nm in ("randn", "randint", "zeros", "ones", "arange", "tensor", "full", "rand", "randn_like"):
        ns.setdefault(nm, getattr(torch, nm))
    return eval(expr, ns)


def _as_args(value):
    """A tuple directive means *multiple* positional args; anything else is one arg."""
    return tuple(value) if isinstance(value, tuple) else (value,)


def _required_pos(sig):
    return [
        p.name
        for p in sig.parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


def _synth_input(model):
    """Guess an input tensor from the model's first consuming layer — DTYPE-AWARE:
    Embedding -> long indices, Conv -> image, RNN/LSTM/GRU -> sequence, Linear -> vector."""
    import torch
    import torch.nn as nn

    try:
        mods = list(model.modules())
    except Exception:
        mods = []
    for m in mods:
        if isinstance(m, nn.Embedding):
            return torch.randint(0, m.num_embeddings, (2, 16)), f"randint(0, {m.num_embeddings}, (2, 16))"
        if isinstance(m, nn.Linear):
            return torch.randn(2, m.in_features), f"randn(2, {m.in_features})"
        if isinstance(m, nn.Conv2d):
            return torch.randn(2, m.in_channels, 32, 32), f"randn(2, {m.in_channels}, 32, 32)"
        if isinstance(m, nn.Conv1d):
            return torch.randn(2, m.in_channels, 32), f"randn(2, {m.in_channels}, 32)"
        if isinstance(m, nn.Conv3d):
            return torch.randn(2, m.in_channels, 16, 16, 16), f"randn(2, {m.in_channels}, 16, 16, 16)"
        if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            if getattr(m, "batch_first", False):
                return torch.randn(2, 16, m.input_size), f"randn(2, 16, {m.input_size})"
            return torch.randn(16, 2, m.input_size), f"randn(16, 2, {m.input_size})"
    return torch.randn(2, 8), "randn(2, 8) [guess]"


def _resolve_args(inst, method, method_line, directives, g):
    """(args_tuple, inside_parens_str). Honors `# fusion: input =`, else auto-synthesizes
    one tensor. Raises _CannotInvoke for multi-arg with no directive, or a bad directive."""
    import inspect

    expr = _pick_directive(directives, "input", method_line)
    if expr is not None:
        try:
            val = _eval_expr(expr, g)
        except Exception as e:
            raise _CannotInvoke(f"input directive failed: {expr!r}: {type(e).__name__}: {e}")
        return _as_args(val), expr
    name = getattr(method, "__name__", "fn")
    required = _required_pos(inspect.signature(method))
    if len(required) == 0:
        return (), ""
    if len(required) > 1:
        raise _CannotInvoke(f"{name}() needs {len(required)} args {required} — add  # fusion: input = (...)")
    if inst is None:
        raise _CannotInvoke(f"{name}() needs an arg — add  # fusion: input = ...")
    x, ishape = _synth_input(inst)
    return (x,), ishape


def _instantiate(g, cls, cls_name, class_line, directives):
    """(inst, construct_note). Honors `# fusion: model =`, else zero-arg cls().
    Raises _CannotInvoke if it needs ctor args (and no directive) or a directive fails."""
    import inspect

    expr = _pick_directive(directives, "model", class_line)
    if expr is not None:
        try:
            return _eval_expr(expr, g), expr
        except Exception as e:
            raise _CannotInvoke(f"model directive failed: {expr!r}: {type(e).__name__}: {e}")
    try:
        req = _required_pos(inspect.signature(cls))
    except (ValueError, TypeError):
        req = []
    if req:
        raise _CannotInvoke(f"{cls_name}() needs constructor args {req} — add  # fusion: model = {cls_name}(...)")
    return cls(), f"{cls_name}()"


def _seeded_call(callable_, model=None):
    """Wrap a call: deterministic seed, eval() the model, run under no_grad. Returns a
    thunk for trace_callable; seed/eval run just before the (traced) call. If torch
    isn't importable (a non-torch file), run the callable as-is."""
    try:
        import torch
    except Exception:
        return callable_

    def go():
        torch.manual_seed(0)
        if model is not None:
            try:
                model.eval()
            except Exception:
                pass
        with torch.no_grad():
            return callable_()

    return go


def _make_invocation(g, name, cls_name, method_line, class_line, directives):
    """Return (thunk, note) ready for trace_callable, or (None, reason) if we can't
    auto-call it. `note` is the exact call used (provenance), shown in the cockpit."""
    import inspect

    if cls_name is None:  # top-level function
        fn = g.get(name)
        if not callable(fn):
            return None, f"{name}: not found or not callable"
        try:
            args, ishape = _resolve_args(None, fn, method_line, directives, g)
        except _CannotInvoke as e:
            return None, str(e)
        return _seeded_call(lambda f=fn, a=args: f(*a)), f"{name}({ishape})"

    cls = g.get(cls_name)
    if not inspect.isclass(cls):
        return None, f"class {cls_name} not found"

    if name == "__init__":  # trace construction itself (build check) — never a synth tensor
        expr = _pick_directive(directives, "model", class_line)
        cnote = expr if expr else f"{cls_name}()"
        build = (lambda: _eval_expr(expr, g)) if expr else cls
        return _seeded_call(build), cnote

    try:
        inst, cnote = _instantiate(g, cls, cls_name, class_line, directives)
    except _CannotInvoke as e:
        return None, str(e)
    except Exception as e:
        return None, f"can't construct {cls_name}(): {type(e).__name__}: {e}"

    method = getattr(inst, name, None)
    if not callable(method):
        return None, f"{cls_name} has no method {name}"
    try:
        args, ishape = _resolve_args(inst, method, method_line, directives, g)
    except _CannotInvoke as e:
        return None, str(e)
    return _seeded_call(lambda m=method, a=args: m(*a), model=inst), f"{cnote}.{name}({ishape})"


def trace_function(path: str, name: str, line: int):
    """Call ONE function/method directly (no __main__, no debugger) and trace it.
    Auto-synthesizes inputs (or honors a `# fusion:` directive).
    Returns (records, error_or_None, crash_line, note)."""
    import ast
    import contextlib
    import io
    import os

    path = os.path.abspath(path)
    src = open(path).read()
    g = {"__name__": "fusion_traced", "__file__": path}  # NOT __main__ -> the file's main block won't run
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(src, path, "exec"), g)  # load defs (and any top-level code)
    except Exception as e:
        return {}, f"module load failed: {type(e).__name__}: {e}", None, ""

    directives = _parse_directives(src)
    cls_name, class_line, method_line = _locate(ast.parse(src, path), name, line)
    invoke, note = _make_invocation(g, name, cls_name, method_line, class_line, directives)
    if invoke is None:
        # not a crash — we just couldn't auto-call it; the note explains why
        return {}, None, None, note

    records, err, crash = trace_callable(invoke, path)
    return records, (f"{type(err).__name__}: {err}" if err else None), crash, note


def trace_module(path: str):
    """Comprehensively trace a file WITHOUT needing a __main__.

    For every model class and zero-arg function DEFINED in the file:
      • construct the model under the tracer  -> a BUILD-time shape check (__init__),
      • call its forward() with an auto-synthesized input (consistent batch=2, so the
        shapes agree with the per-function "trace this function"), honoring `# fusion:`.
    The file's own __main__ is deliberately NOT run. Returns a dict:
      {"records": {line: {var: {...}}},
       "problems": [{"line","message"}],
       "notes":    [{"label","line","note"}]}   # note = exact call used (provenance)
    """
    import ast
    import contextlib
    import inspect
    import io
    import os

    path = os.path.abspath(path)
    src = open(path).read()
    g = {"__name__": "fusion_traced", "__file__": path}  # __main__ block won't run
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(src, path, "exec"), g)
    except Exception as e:
        return {"records": {}, "problems": [{"line": 1, "message": f"module load failed: {type(e).__name__}: {e}"}], "notes": []}

    directives = _parse_directives(src)
    tree = ast.parse(src, path)
    records: Dict[int, Dict[str, Any]] = {}
    problems: list = []
    notes: list = []

    def merge(srcrec) -> None:
        for ln, vars_ in srcrec.items():
            records.setdefault(ln, {}).update(vars_)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = g.get(node.name)
            if not inspect.isclass(cls) or not _is_model(cls):
                continue
            expr = _pick_directive(directives, "model", node.lineno)
            if expr is None:
                try:
                    if _required_pos(inspect.signature(cls)):
                        notes.append({"label": node.name, "line": node.lineno,
                                      "note": f"{node.name}() needs constructor args — add  # fusion: model = {node.name}(...)"})
                        continue
                except (ValueError, TypeError):
                    pass
            # BUILD CHECK: construct under the tracer (catches shape errors in __init__).
            holder: list = []
            build = (lambda e=expr, h=holder: h.append(_eval_expr(e, g))) if expr else (lambda c=cls, h=holder: h.append(c()))
            recs, err, crash = trace_callable(_seeded_call(build), path)
            merge(recs)
            cnote = expr if expr else f"{node.name}()"
            if err is not None:
                problems.append({"line": crash or node.lineno, "message": f"build {node.name}(): {type(err).__name__}: {err}"})
                continue
            notes.append({"label": node.name, "line": node.lineno, "note": f"{cnote}  [build ok]"})
            inst = holder[0] if holder else None
            method = getattr(inst, "forward", None) if inst is not None else None
            if not callable(method):
                continue
            fwd_line = _method_line(node, "forward")
            try:
                args, ishape = _resolve_args(inst, method, fwd_line, directives, g)
            except _CannotInvoke as e:
                notes.append({"label": f"{node.name}.forward", "line": fwd_line, "note": str(e)})
                continue
            recs2, err2, crash2 = trace_callable(_seeded_call(lambda m=method, a=args: m(*a), model=inst), path)
            merge(recs2)
            if err2 is not None:
                problems.append({"line": crash2 or fwd_line, "message": f"{node.name}.forward: {type(err2).__name__}: {err2}"})
            else:
                notes.append({"label": f"{node.name}.forward", "line": fwd_line, "note": f"{cnote}.forward({ishape})"})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = g.get(node.name)
            if not callable(fn):
                continue
            try:
                args, ishape = _resolve_args(None, fn, node.lineno, directives, g)
            except _CannotInvoke:
                continue  # needs args & no directive -> skip silently at file level
            recs, err, crash = trace_callable(_seeded_call(lambda f=fn, a=args: f(*a)), path)
            merge(recs)
            if err is not None:
                problems.append({"line": crash or node.lineno, "message": f"{node.name}(): {type(err).__name__}: {err}"})
            else:
                notes.append({"label": node.name, "line": node.lineno, "note": f"{node.name}({ishape})"})

    return {"records": records, "problems": problems, "notes": notes}


def _is_model(cls) -> bool:
    """True if cls subclasses torch.nn.Module — checked via MRO names so we never import torch."""
    try:
        for base in cls.__mro__:
            if base.__name__ == "Module" and base.__module__.split(".")[0] == "torch":
                return True
    except Exception:
        pass
    return False


def _method_line(class_node, method_name: str) -> int:
    import ast

    for sub in class_node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
            return sub.lineno
    return class_node.lineno


def _locate(tree, name, line):
    """(cls_name, class_line, method_line) for function `name` (optionally matching a
    specific def line). cls_name/class_line are None for a top-level function."""
    import ast

    for node in tree.body:  # top-level function?
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name and line in (0, node.lineno):
            return None, None, node.lineno
    for node in ast.walk(tree):  # method in a class
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == name and line in (0, sub.lineno):
                    return node.name, node.lineno, sub.lineno
    for node in ast.walk(tree):  # fallback: any function with that name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return None, None, node.lineno
    return None, None, line or 0
